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
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        self.backend = backend if self.parameters[0].device.type == "cuda" else "eager"
        self.graphs = {}
        self.masks = {}
        frozen_model = getattr(frozen, "model", None)
        self.frozen_parameters = tuple(frozen_model.parameters()) if frozen_model is not None else ()
        self.frozen_identity = tuple((p.data_ptr(), p._version, p.device, p.dtype) for p in self.frozen_parameters)
        model.enable_static_cache()

    def backward(self, *batch):
        """Return detached loss values and a device-side rejection flag."""
        model = self.model
        if self.kind in ("generator", "derivative"):
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
        targets = (5, 10) if self.kind == "short" else (10, 20, 25, 30, 40, 50)
        boundaries = (10,) if self.kind == "short" else (10, 20, 30, 40, 50)
        losses, segment = [], []
        for step in range(1, targets[-1] + 1):
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
                segment.clear()
                low, high = low.detach(), high.detach()
        values = torch.stack(losses)
        return values, failure | ~torch.isfinite(values).all()

    def gradients(self, *batch):
        """Compute raw gradients without clipping or updating parameters."""
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
        if bool(failure):
            raise FloatingPointError("GIFT batch rejected: nonfinite loss/state or local correction gate; no optimizer update")
        return values

    def step(self, optimizer, *batch):
        values = self.gradients(*batch)
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
