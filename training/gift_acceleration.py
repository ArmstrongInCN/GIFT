"""Optional GIFT training execution engines with unchanged scientific budgets."""
from __future__ import annotations

import torch

from gift.execution import StaticTensorCall
from experiments.formal._shared.gift_runtime import (
    correction_tensors, DEFAULT_P21_CORRECTION_POLICY, square_selected_mask,
)
from experiments.formal.train_gift_branches import _dual_rk4_step, _initialize_tracks, ROLLOUT_STEPS


class TrainingEngine:
    """One engine per phase; at most one graph per distinct batch shape.

    CPU uses eager tensor execution. CUDA capture failures are explicit errors,
    never silently counted as successful updates. All acceptance checks precede
    optimizer.step, including graph-accumulated correction failures.
    """

    def __init__(self, model, *, kind, frozen=None, scale=None, backend="cuda-graph"):
        if kind not in ("generator", "derivative", "short", "long"):
            raise ValueError("unknown GIFT training phase")
        if backend not in ("eager", "cuda-graph"):
            raise ValueError("unknown execution backend")
        self.model, self.kind, self.frozen, self.scale = model, kind, frozen, scale
        # Only trainable parameters enter the captured graph; frozen generator
        # weights stay out of the update and out of the replay.
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        # CUDA graphs replay a captured CUDA stream; on CPU the engine falls
        # back to eager execution, which the budget counts identically.
        self.backend = backend if self.parameters[0].device.type == "cuda" else "eager"
        self.graphs = {}
        self.masks = {}
        frozen_model = getattr(frozen, "model", None)
        self.frozen_parameters = tuple(frozen_model.parameters()) if frozen_model is not None else ()
        # Snapshot the frozen generator's tensor identity so a later in-place
        # change is caught before it corrupts a replayed graph.
        self.frozen_identity = tuple((p.data_ptr(), p._version, p.device, p.dtype) for p in self.frozen_parameters)
        model.enable_static_cache()
        self.branch_view = None
        if self.backend == "cuda-graph" and kind != "generator":
            from experiments.formal._shared.high_frequency import HighFrequencyBranch
            from training.gift_branch_kernels import BranchTrainingView
            if isinstance(model, HighFrequencyBranch):
                self.branch_view = BranchTrainingView(model)

    def backward(self, *batch):
        """Return detached loss values and a device-side rejection flag."""
        try:
            return self._backward(*batch)
        finally:
            if self.branch_view is not None:
                self.branch_view.end_segment()

    def _backward(self, *batch):
        # Limit layout sharing to the release protocol's verified full batches.
        # Smaller tails retain the reference graph: changing FFT/accumulation
        # layouts at batch size one can alter the last bits of raw gradients.
        # The captured high-frequency graph assumes a fixed full batch (16 for
        # the derivative phase, 5 for rollout); any other size uses the
        # reference eager path so layouts stay bit-stable.
        expected_batch = 16 if self.kind == "derivative" else 5
        view = self.branch_view if len(batch[0]) == expected_batch else None
        model = view if view is not None else self.model
        if self.kind in ("generator", "derivative"):
            if view is not None:
                view.begin_segment()
            state, target = batch[:2]
            if self.kind == "generator":
                loss = (model(state) - target).square().mean() / self.scale
            else:
                loss = ((model.forward_with_generator_output(state, batch[2]) - target) / self.scale).square().mean()
            loss.backward()
            return loss.detach().reshape(1), ~torch.isfinite(loss.detach())
        sequence = batch[0]
        key = (int(sequence.shape[-1]), sequence.device)
        mask = self.masks.get(key)
        if mask is None:
            mask = square_selected_mask(key[0], cutoff=21, device=key[1])
            self.masks[key] = mask
        low, high = _initialize_tracks(model, sequence[:, 0])
        failure = torch.zeros((), dtype=torch.bool, device=sequence.device)
        # Rollout horizons in RK4 steps (each step is one DT of physical time).
        # The short phase checks two mid-horizon frames; the long phase spreads
        # loss across six horizons and segments the backward at each boundary.
        targets = (5, 10) if self.kind == "short" else (10, 20, 25, 30, 40, 50)
        boundaries = (10,) if self.kind == "short" else (10, 20, 30, 40, 50)
        losses, segment = [], []
        for step in range(1, targets[-1] + 1):
            if view is not None and (step == 1 or step - 1 in boundaries):
                view.begin_segment()
            # Advance both the low-frequency truth and the high-frequency
            # residual one RK4 step against the frozen generator.
            low, high = _dual_rk4_step(self.frozen, model, low, high)
            failure = failure | ~torch.isfinite(low).all()
            corrected = correction_tensors(low.detach(), DEFAULT_P21_CORRECTION_POLICY, mask)
            low = corrected[0]
            failure = failure | corrected[2].any()
            high = model.project_q21(high)
            if step in targets:
                target = model.project_q21(sequence[:, ROLLOUT_STEPS.index(step)])
                value = ((high - target) / model.high_scale).square().mean()
                losses.append(value.detach())
                # The short reference uses stack(losses).mean(), whereas the
                # long reference divides each target before segmented sums.
                segment.append(value if self.kind == "short" else value / len(targets))
            if step in boundaries:
                loss = torch.stack(segment).mean() if self.kind == "short" else torch.stack(segment).sum()
                loss.backward()
                if view is not None:
                    view.end_segment()
                segment.clear()
                low, high = low.detach(), high.detach()
        values = torch.stack(losses)
        return values, failure | ~torch.isfinite(values).all()

    def gradients(self, *batch):
        """Compute raw gradients without clipping or updating parameters."""
        # Re-verify the frozen generator has not moved since capture; a changed
        # tensor identity would replay a graph against different weights.
        if tuple((p.data_ptr(), p._version, p.device, p.dtype) for p in self.frozen_parameters) != self.frozen_identity:
            raise RuntimeError("frozen generator changed; rebuild its adapter and training engine")
        if self.backend == "cuda-graph":
            key = tuple((x.shape, x.dtype, x.device) for x in batch)
            call = self.graphs.get(key)
            if call is None:
                # Include fixed affine tables in the storage guard as well.
                # Their values may be refitted in-place, but their addresses
                # must not be replaced beneath a captured generator forward.
                call = StaticTensorCall(self.backward, batch, parameters=tuple(self.model.parameters()), backward=True)
                self.graphs[key] = call
            values, failure = call(*batch)
        else:
            self.model.zero_grad(set_to_none=True)
            values, failure = self.backward(*batch)
        # Reject the whole batch before any update: a nonfinite state or a
        # failed local-correction gate must never advance the parameters.
        if bool(failure):
            raise FloatingPointError("GIFT batch rejected: nonfinite loss/state or local correction gate; no optimizer update")
        return values

    def step(self, optimizer, *batch):
        values = self.gradients(*batch)
        # Generator gradients use a looser 5.0 clip; branch/rollout phases share
        # the 1.0 clip that matches the derivative reference.
        torch.nn.utils.clip_grad_norm_(self.parameters, 5.0 if self.kind == "generator" else 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        if self.kind == "short":
            return float(values.mean())
        values = values.cpu().tolist()
        return sum(value / len(values) for value in values)

    def epoch(self, loader, optimizer, device):
        self.model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = tuple(value.to(device) for value in batch)
            value = self.step(optimizer, *batch)
            total += value * len(batch[0])
            count += len(batch[0])
        return total / count
