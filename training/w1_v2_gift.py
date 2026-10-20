"""GIFT's native two-stage training on the common W1-v2 t=5..8 observations.

Only this controller changes: five-point derivatives and the existing eight
rollout targets are sampled inside the supplied observation slice. Generator
variable projection, branch operators, RK4, correction and native losses are
imported from the established implementation. Validation always predicts from
the first observed field and scores all supplied unknown future fields. The
sparse paths change observation spacing, while native RK4 remains at .02s.
On the .04s grid, two unavailable native supervision leads move to the nearest
observed times, with an exact midpoint choosing the earlier time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from gift import GIFTGenerator, FixedBandwidthGridGenerator, fit_affine_minimum_norm, load_generator
from experiments.formal._shared.gift_generator_training import (
    clone_state, configure_determinism, configure_variable_projection_parameters,
)
from experiments.formal._shared.high_frequency import (
    HighFrequencyBranch, HighFrequencyConfig, load_high_frequency_model,
)
from experiments.formal._shared.gift_runtime import (
    rollout_gift, DEFAULT_P21_CORRECTION_POLICY, DEFAULT_Q21_CORRECTION_POLICY,
)
from experiments.formal.train_gift_branches import (
    _compute_low_rhs, _project, _square_mask, _training_scales,
    _dual_rk4_step, _correct_low, _initialize_tracks,
)
from training.checkpoints import capture_rng, restore_rng, restore_optimizer, runtime_identity, state_digest, digest_file
from training.configuration import output_path, validate_resume, validate_training_state
from training.gift_prediction_control import phase_at, source_identity, synchronize
from training.gift_prediction_data import ObservationBank, observed_pairs, observed_sequences, ROLLOUT_OFFSETS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "gift.w1-fair151.window.v2"
INTEGRATION_DT = .02


def _rollout_steps(observation_dt):
    """Declare integration steps of the eight observed supervision targets.

    The .04s grid cannot observe native .1/.5s leads. The predeclared nearest
    observation rule, ties earlier, moves only these leads to .08/.48s. The
    short .2s and long1s horizons, .02s integration and ten-step TBPTT stay fixed.
    Original .02s/.1s grids supply every native lead exactly.
    """
    if observation_dt == .04:
        return np.asarray((0, 4, 10, 20, 24, 30, 40, 50), dtype=np.int64)
    if observation_dt in (.02, .1):
        return ROLLOUT_OFFSETS
    raise ValueError("GIFT W1 supports .02, .04 or .1 observation spacing")


def _observation_offsets(observation_dt):
    """Map declared physical supervision leads to exact observed indices, without interpolation.

    A .1s bank gathers indices0,1,2,4,5,6,8,10. A .04s bank gathers
    indices0,2,5,10,12,15,20,25 after the two explicitly declared lead changes.
    All targets remain true supplied observations rather than interpolated labels.
    """
    indices = _rollout_steps(observation_dt) * INTEGRATION_DT / observation_dt
    rounded = np.rint(indices).astype(np.int64)
    if not np.allclose(indices, rounded, atol=2e-12, rtol=0):
        raise ValueError("observation grid cannot exactly supply native GIFT rollout leads")
    return rounded


def _grid_pairs(raw, order, centres, observation_dt):
    """Five observed frames give [B,y,x] state/RHS in vorticity-per-second units.

    This is the native fourth-order central stencil with its denominator set
    by observation spacing, independent of the .02-second integrator.
    """
    if observation_dt == INTEGRATION_DT:
        return observed_pairs(raw, order, centres)
    rows, times = order, centres[order]
    state = np.ascontiguousarray(raw[rows, times], dtype=np.float32)
    target = (raw[rows, times-2] - 8.0*raw[rows, times-1]
              + 8.0*raw[rows, times+1] - raw[rows, times+2])/(12.0*observation_dt)
    return torch.from_numpy(state), torch.from_numpy(np.ascontiguousarray(target))


def _grid_sequences(raw, order, anchors, offsets):
    """Gather the eight declared supervision leads from real observation frames."""
    if np.array_equal(offsets, ROLLOUT_OFFSETS):
        return observed_sequences(raw, order, anchors)
    times = anchors[order, None] + offsets[None]
    return np.ascontiguousarray(raw[order[:, None], times], dtype=np.float32)


def _grid_spec(bank):
    """Describe the actual output grid and the unchanged native integration rule.

    Dense fixtures and historical banks need no additional binding fields.
    New sparse banks already bind their query times through the shared receipt.
    """
    points = (bank.training.shape[1] if hasattr(bank, "training") else
              bank.validation.shape[1] if hasattr(bank, "validation") else bank.binding.get("points", 151))
    times = np.asarray(bank.binding.get("query_times", np.arange(points)*INTEGRATION_DT), dtype=np.float64)
    if (times.shape != (points,) or points < 5 or times[0] != 0
            or not np.allclose(np.diff(times), times[1], atol=2e-12, rtol=0)
            or not np.isclose(times[-1], 3., atol=2e-12, rtol=0)):
        raise ValueError("GIFT requires a uniform local0..3 observation grid")
    dt = float(times[1])
    _observation_offsets(dt)
    if not np.isclose(dt / INTEGRATION_DT, round(dt / INTEGRATION_DT), atol=2e-12, rtol=0):
        raise ValueError("GIFT save spacing must be a complete number of native RK4 steps")
    return {"duration": 3., "points": points, "future_steps": points-1,
            "query_times": times.tolist(), "observation_dt": dt,
            "integration_dt": INTEGRATION_DT, "save_interval": dt, "step": INTEGRATION_DT,
            "rollout_native_requested_steps": ROLLOUT_OFFSETS.tolist(),
            "rollout_integration_steps": _rollout_steps(dt).tolist(),
            "rollout_observation_offsets": _observation_offsets(dt).tolist(),
            "rollout_supervision_seconds": (_rollout_steps(dt)*INTEGRATION_DT).tolist(),
            "rollout_target_policy": "nearest_observed_lead_ties_earlier" if dt == .04 else "exact_native_leads"}


@dataclass(frozen=True)
class GIFTTrainingConfig:
    """An explicit two-stage budget; epoch counts refer to visits of every trajectory.

    Development uses100 generator+100 branch epochs; formal runs use500+1000.
    LR multiplier scales the native phase reference rates, while one cosine
    cycle spans each entire stage. AdamW moments reset at native phase changes.
    """

    generator_phases: tuple[int, int] = (250, 250)
    branch_phases: tuple[int, int, int, int] = (800, 100, 50, 50)
    stage: str = "formal"
    lr_multiplier: float = 1.0
    order_seed: int = 2026100301
    sampling_seed: int = 2026100303
    effective_batch: int = 32
    generator_micro_batch: int = 16
    branch_micro_batch: int = 4
    validation_batch: int = 4
    validation_interval: int = 20
    minimum_pipeline_epochs: int = 500
    patience: int = 10
    relative_improvement: float = 0.001
    early_stopping: bool = True
    cutoff: int = 21
    rank: int = 8
    fixture_only: bool = False

    @classmethod
    def development(cls, lr_multiplier=1.0):
        """Return the declared200-epoch whole-pipeline development budget."""
        return cls(generator_phases=(50, 50), branch_phases=(80, 10, 5, 5),
                   stage="development", lr_multiplier=lr_multiplier, early_stopping=False)

    def validate(self):
        """Reject impossible budgets, accumulation and formal architecture choices."""
        if len(self.generator_phases) != 2 or len(self.branch_phases) != 4:
            raise ValueError("GIFT needs two generator and four branch phases")
        if self.stage not in ("development", "formal"):
            raise ValueError("GIFT stage must be development or formal")
        counts = (*self.generator_phases, *self.branch_phases, self.effective_batch,
                  self.generator_micro_batch, self.branch_micro_batch,
                  self.validation_batch, self.validation_interval,
                  self.minimum_pipeline_epochs, self.patience, self.cutoff, self.rank)
        if any(type(v) is not int or v < 1 for v in counts):
            raise ValueError("GIFT counts must be positive integers")
        if not math.isfinite(self.lr_multiplier) or self.lr_multiplier <= 0:
            raise ValueError("LR multiplier must be positive and finite")
        if not 0 <= self.relative_improvement < 1:
            raise ValueError("invalid relative-improvement criterion")
        if max(self.generator_micro_batch, self.branch_micro_batch) > self.effective_batch:
            raise ValueError("microbatch exceeds effective batch")
        if not self.fixture_only and (self.cutoff != 21 or self.rank != 8):
            raise ValueError("formal W1-v2 GIFT preserves cutoff21/rank8")


class PredictionObservationBank(ObservationBank):
    """Read the common slice and sample only its legal internal observation locations."""

    def __init__(self, path, config):
        """Require canonical IDs and exact local/absolute grids before loading fields."""
        path = Path(path).resolve(strict=True)
        with h5py.File(path, "r") as h:
            schema = h.attrs.get("schema")
        if schema != SCHEMA:
            # The shared sparse reader owns subset IDs, decoded-data receipts
            # and the observation-time contract. No external/test data are read.
            from training.w1_v2_data import WindowBank
            shared = WindowBank(path, fixture=config.fixture_only)
            self.training, self.validation = shared.training, shared.validation
            self.binding = dict(shared.binding)
            self.binding["training_ids"] = shared.training_ids.tolist()
            self.binding["validation_ids"] = shared.validation_ids.tolist()
            if set(self.binding["training_ids"]) & set(self.binding["validation_ids"]):
                raise ValueError("training/validation identity overlap")
            grid = _grid_spec(self)
            self.observation_dt = grid["observation_dt"]
            self.observation_offsets = _observation_offsets(self.observation_dt)
            self.integration_steps = _rollout_steps(self.observation_dt)
            self.config = config
            self.state_scale = float(np.sqrt(np.square(self.training, dtype=np.float64).mean()))
            if not math.isfinite(self.state_scale) or self.state_scale <= 0:
                raise ValueError("invalid training-only state scale")
            # Match the native GIFT scientific hash convention while retaining
            # the shared per-split hashes and exact seeded subset in binding.
            scientific = hashlib.sha256()
            for name in ("training", "validation"):
                scientific.update(name.encode("ascii"))
                scientific.update(np.asarray(self.binding[name+"_ids"], dtype="<i8").tobytes())
                scientific.update(np.asarray(grid["query_times"], dtype="<f8").tobytes())
                scientific.update(getattr(self, name).tobytes())
            self.binding.update(observations_sha256=scientific.hexdigest(), time_step=self.observation_dt,
                                absolute_window=[5., 8.], derivative_centres=[2, grid["points"]-3],
                                rollout_anchors=[0, grid["points"]-1-int(self.observation_offsets[-1])],
                                rollout_observation_offsets=self.observation_offsets.tolist(),
                                rollout_integration_steps=self.integration_steps.tolist(),
                                rollout_native_requested_steps=ROLLOUT_OFFSETS.tolist(),
                                rollout_supervision_seconds=grid["rollout_supervision_seconds"],
                                rollout_target_policy=grid["rollout_target_policy"],
                                integration_dt=INTEGRATION_DT, observation_dt=self.observation_dt,
                                read_extra_boundary_frames=False)
            return
        with h5py.File(path, "r") as h:
            if (h.attrs.get("trajectory_id_scheme") != "canonical"
                    or h.attrs.get("status") != "complete"):
                raise ValueError("W1-v2 slice schema/canonical ID declaration differs")
            if bool(h.attrs.get("fixture_only", False)) != config.fixture_only:
                raise ValueError("fixture declaration differs from this run")
            for name in ("training", "validation"):
                w = h[name + "/vorticity"]
                local = h[name + "/time"][:]
                absolute = h[name + "/absolute_time"][:]
                ids = h[name + "/trajectory_index"][:]
                if (w.shape[1] != 151 or local.shape != (151,) or absolute.shape != (151,)
                        or not np.allclose(local, np.arange(151)*.02, atol=2e-12, rtol=0)
                        or not np.allclose(absolute, 5 + np.arange(151)*.02, atol=2e-12, rtol=0)):
                    raise ValueError("W1-v2 observations must be exactly t5..8")
                if not config.fixture_only:
                    expected = np.arange(1000) if name == "training" else np.arange(1000, 1020)
                    if w.shape != (len(expected), 151, 64, 64) or not np.array_equal(ids, expected):
                        raise ValueError("formal W1-v2 population/field axes differ")
        super().__init__(path, fixture=config.fixture_only)
        self.config = config
        self.observation_dt = INTEGRATION_DT
        self.observation_offsets = ROLLOUT_OFFSETS
        self.integration_steps = ROLLOUT_OFFSETS
        self.binding.update(schema=SCHEMA, absolute_window=[5.0, 8.0],
                            derivative_centres=[2, 148], rollout_anchors=[0, 100],
                            read_extra_boundary_frames=False)

    def locations(self, epoch, *, validation=False, rollout=False):
        """Separate common trajectory order from method-specific internal stencil draws.

        epoch is the zero-based global whole-pipeline sampling key. Centres
        leave two observed frames on each side; anchors leave one physical
        second ahead. Both kinds of window stay inside the supplied t5..8 slice.
        """
        raw = self.validation if validation else self.training
        if type(epoch) is not int or epoch < 0:
            raise ValueError("global epoch must be a nonnegative integer")
        order = np.random.default_rng(np.random.SeedSequence(
            [self.config.order_seed, epoch])).permutation(len(raw))
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.config.sampling_seed, epoch, int(validation), int(rollout)]))
        locations = (rng.integers(0, raw.shape[1]-int(self.observation_offsets[-1]), len(raw))
                     if rollout else rng.integers(2, raw.shape[1]-2, len(raw)))
        return order, locations

    def pairs(self, seed, epoch, *, validation=False):
        """Reuse the native observed fourth-order stencil; model seed does not affect data order."""
        raw = self.validation if validation else self.training
        return _grid_pairs(raw, *self.locations(epoch, validation=validation), self.observation_dt)

    def sequences(self, seed, epoch, *, validation=False):
        """Gather declared physical rollout targets, with an independent internal-anchor stream."""
        raw = self.validation if validation else self.training
        return _grid_sequences(raw, *self.locations(epoch, validation=validation, rollout=True), self.observation_offsets)


def _atomic_torch(path, value):
    """Replace only a mutable snapshot inside this newly created run, without deleting other files."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path, value):
    """Atomically replace a current new-run record using strict JSON finite values."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sources():
    """Record reused source identities without making bytes a run-permission gate."""
    return {**source_identity(), "training/w1_v2_gift.py": digest_file(Path(__file__)),
            "training/w1_v2_data.py": digest_file(ROOT / "training" / "w1_v2_data.py")}


class _Journal:
    """Small rolling resume state; historical runs are never opened for writes."""

    def __init__(self, output, identity, resume):
        self.output = output_path(output, ROOT)
        self.identity = identity
        self.runtime = runtime_identity()
        self.payload = None
        self.rng = None
        self.validation_cursor = 0
        if resume:
            attempt = json.loads((self.output / "ATTEMPT.json").read_text(encoding="utf-8"))
            validate_resume(identity, attempt["identity"])
            boundary = torch.load(self.output / "latest.pt", map_location="cpu", weights_only=True)
            validate_resume(identity, boundary["identity"])
            validate_training_state(boundary["payload"])
            self.payload, self.rng = boundary["payload"], boundary["rng"]
            log = self.output / "validation.jsonl"
            if log.exists():
                for line in log.read_text(encoding="utf-8").splitlines():
                    self.validation_cursor = max(self.validation_cursor, json.loads(line)["epoch"])
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            with (self.output / "ATTEMPT.json").open("x", encoding="utf-8") as stream:
                json.dump({"identity": identity, "runtime": self.runtime,
                           "run_id": uuid.uuid4().hex}, stream, indent=2)

    def save(self, payload, *, epoch500=False):
        """Commit model/AdamW/cosine/RNG/selection and optional fixed-cost boundary."""
        value = {"schema": "gift.w1-v2.rolling-boundary.v1", "identity": self.identity,
                 "runtime": self.runtime, "rng": capture_rng(), "payload": payload}
        _atomic_torch(self.output / "latest.pt", value)
        if epoch500:
            _atomic_torch(self.output / "epoch500.pt", value)
        _atomic_json(self.output / "history.json", payload["history"])
        for row in payload["history"]:
            if row["epoch"] <= self.validation_cursor or row["validation"] is None:
                continue
            record = {"epoch": row["epoch"], "global_trajectory_epoch": row["global_trajectory_epoch"],
                      "phase": row["phase"], "eligible": row.get("selection_eligible", True),
                      "best_epoch_at_commit": None if payload["best"] is None else payload["best"]["epoch"],
                      "validation": row["validation"]}
            with (self.output / "validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.validation_cursor = row["epoch"]


def _optimizer(parameters, stage, config):
    """Construct native AdamW with one whole-stage cosine, retaining stage-specific decay."""
    rates = (.01, .003) if stage == "generator" else (.0015, .0002, .00008, .00008)
    rates = tuple(v * config.lr_multiplier for v in rates)
    total = sum(config.generator_phases if stage == "generator" else config.branch_phases)
    optimizer = torch.optim.AdamW(parameters, lr=rates[0], weight_decay=1e-8 if stage == "generator" else 1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=.01*rates[0])
    return optimizer, scheduler, rates


def _change_phase(optimizer, scheduler, base_lr):
    """Reset native phase moments and change LR amplitude without restarting cosine time.

    scheduler.last_epoch is the number of completed stage epochs. Rescaling its
    base and floor produces base_lr*(.01+.99*(1+cos(pi*epoch/Tmax))/2).
    The next native scheduler.step continues that same cycle.
    """
    optimizer.state.clear()
    factor = .01 + .99*(1 + math.cos(math.pi*scheduler.last_epoch/scheduler.T_max))/2
    for group in optimizer.param_groups:
        group["initial_lr"] = base_lr
        group["lr"] = base_lr * factor
    scheduler.base_lrs = [base_lr] * len(optimizer.param_groups)
    scheduler.eta_min = .01 * base_lr
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def _validation(low, branch, bank, device, config):
    """Native initial-only rollout and failure-preserving scoring on the bound output grid."""
    previous_rng = capture_rng()
    started = time.perf_counter()
    grid = _grid_spec(bank)
    points, future = grid["points"], grid["future_steps"]
    try:
        result = rollout_gift(low=low, branch=branch, initial_state=bank.validation[:, 0],
                              trajectory_ids=np.asarray(bank.binding["validation_ids"]),
                              duration=grid["duration"], save_interval=grid["save_interval"], step=grid["step"], device=device,
                              batch_size=config.validation_batch, correction=True, execution="eager")
        prediction = result.prediction
        if prediction.shape != bank.validation.shape:
            raise ValueError("GIFT validation shape differs from its common observation grid")
        relative = np.full((len(prediction), points), np.nan, dtype=np.float64)
        mse, failed = [], []
        for row, (pred, truth) in enumerate(zip(prediction, bank.validation)):
            if not np.isfinite(pred).all() or result.failure_step[row] >= 0:
                failed.append(bank.binding["validation_ids"][row])
                continue
            truth64 = truth.astype(np.float64)
            error = pred.astype(np.float64) - truth64
            denominator = np.linalg.norm(truth64.reshape(points, -1), axis=1)
            if np.any(denominator <= 0):
                raise ValueError("zero-norm validation observation cannot be scored relatively")
            relative[row] = np.linalg.norm(error.reshape(points, -1), axis=1) / denominator
            mse.append(float(np.square(error[1:]).mean()))
        score = None if failed else float(relative[:, 1:].mean())
        # A failed trajectory is retained as JSON null, never removed by nanmean.
        values = [[float(v) if math.isfinite(v) else None for v in row] for row in relative]
        return {"val_metric": score, "metric": f"future{future}_mean_per_frame_relative_l2",
                "trajectory_ids": bank.binding["validation_ids"], "relative_l2": values,
                "failed_trajectory_ids": failed, "failed_trajectory_count": len(failed),
                "t8_relative_l2": None if failed else float(relative[:, -1].mean()),
                f"future{future}_physical_mse": None if failed else float(np.mean(mse)),
                "seconds": time.perf_counter() - started}
    finally:
        restore_rng(previous_rng)


def _select(best, validation, epoch, model, *, eligible=True):
    """Keep a strictly lower finite validation value and earliest exact tie."""
    value = validation["val_metric"]
    if eligible and value is not None and (best is None or value < best["val_metric"]):
        return {"epoch": epoch, "val_metric": value, "model": clone_state(model),
                "validation": validation}
    return best


def _cost(history, stages, *, affine_fits=0, setup_seconds=0.):
    """Count actual updates/visits and timed work; phase epochs are not matched GPU work."""
    return {"trajectory_epochs": len(history), "phase_epoch_limits": list(stages),
            "actual_phase_epochs": [sum(row["phase"] == phase for row in history) for phase in range(len(stages))],
            "optimizer_updates": sum(row["optimizer_updates"] for row in history),
            "trajectories_visited": sum(row["trajectories_visited"] for row in history),
            "affine_fits": affine_fits,
            "committed_seconds": sum(row["seconds"] for row in history),
            "setup_seconds": setup_seconds,
            "total_training_seconds": setup_seconds+sum(row["seconds"] for row in history),
            "validation_seconds": sum(row["validation"]["seconds"] for row in history if row["validation"]),
            "timing_scope": "sampling, gradients, affine refits and scheduled native full-path validation; excludes setup/snapshot writes/paused time"}


def _generator_export(model, state, config, seed, bank, epoch, metric, cost):
    """Native load_generator-compatible parameters plus common W1-v2 selection metadata."""
    parameter_count = sum(v.numel() for v in model.parameters())
    return {"format_version": 3, "artifact_role": "trained_generator", "stage": config.stage,
            "model_configuration": model.configuration(), "model_state_dict": state,
            "model_state": state, "model_spec": {"method": "GIFT-generator", "family": "gift_generator", "history": 1,
                                                   **_grid_spec(bank)},
            "config": asdict(config), "training_configuration": asdict(config), "seed": seed,
            "epoch": epoch, "val_metric": metric, "training_data": bank.binding, "data_binding": bank.binding,
            "training_cost": {**cost, "parameter_count": parameter_count},
            "parameter_counts": {"generator": parameter_count, "convention": "native_parameter_tensor_numel"},
            "selection": f"best_future{_grid_spec(bank)['future_steps']}_validation_after_coarse_generator_phase",
            "fixture_only": config.fixture_only, "normalization": {"state_scale": bank.state_scale},
            "training_regime": "w1_v2_common_t5_t8", "condition": "noise_000"}


def _boundary(model, optimizer, scheduler, epoch, phase, history, best, *, completed=False, extra=None):
    """A complete same-run boundary, including the selected state rather than only its score."""
    return {"model": clone_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "phase": phase,
            "optimizer_updates": sum(row["optimizer_updates"] for row in history),
            "history": history, "best": best, "completed": completed, **(extra or {})}


def _validate_boundary(saved, counts):
    """Require phase and cosine time to agree with the saved completed epoch."""
    if saved is None:
        return
    epoch = saved["epoch"]
    if not 0 <= epoch <= sum(counts):
        raise ValueError("saved stage epoch exceeds its budget")
    if epoch and saved["phase"] != phase_at(epoch, counts)[0]:
        raise ValueError("saved phase disagrees with stage epoch")
    if saved["scheduler"]["last_epoch"] != epoch or saved["scheduler"]["T_max"] != sum(counts):
        raise ValueError("saved cosine time disagrees with completed epochs")


def run_generator(dataset, output, *, seed, config=GIFTTrainingConfig(), device="cuda",
                  resume=False, checkpoint_interval=20, stop_after_epoch=None):
    """Train the low generator, then export validation-selected and terminal states separately."""
    config.validate()
    setup_started = time.perf_counter()
    if checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be positive")
    total = sum(config.generator_phases)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= total:
        raise ValueError("stop epoch outside generator budget")
    device = torch.device(device)
    configure_determinism(seed, strict=True)
    bank = PredictionObservationBank(dataset, config)
    identity = {"role": "w1_v2_gift_generator", "configuration": asdict(config),
                "seed": seed, "data": bank.binding, "sources": _sources(), "device": str(device)}
    journal = _Journal(output, identity, resume)
    saved = journal.payload
    _validate_boundary(saved, config.generator_phases)
    if saved and saved["completed"] and (journal.output / "COMPLETE.json").exists():
        return json.loads((journal.output / "COMPLETE.json").read_text(encoding="utf-8"))
    model = GIFTGenerator(config.cutoff, config.rank, bank.state_scale, False).to(device)
    parameters = configure_variable_projection_parameters(model)
    states, targets = bank.pairs(seed, 0)
    target_scale = float(targets.square().mean().clamp_min(1e-20))
    if saved is None:
        fit_affine_minimum_norm(model, states, targets, device=device, subtract_quadratic=False)
    else:
        if saved["target_scale"] != target_scale:
            raise ValueError("generator training target scale differs")
        model.load_state_dict(saved["model"], strict=True)
    del states, targets
    optimizer, scheduler, rates = _optimizer(parameters, "generator", config)
    completed = 0 if saved is None else saved["epoch"]
    if saved:
        restore_optimizer(optimizer, saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(journal.rng)
    if stop_after_epoch is not None and stop_after_epoch <= completed:
        raise ValueError("stop epoch must follow saved progress")
    history = [] if saved is None else saved["history"]
    best = None if saved is None else saved["best"]
    previous_phase = 0 if saved is None else saved["phase"]
    setup_seconds = (0. if saved is None else saved.get("setup_seconds", 0.))+time.perf_counter()-setup_started
    if saved is None:
        journal.save(_boundary(model, optimizer, scheduler, 0, 0, history, best,
                               extra={"target_scale": target_scale, "setup_seconds": setup_seconds}))
    for epoch in range(completed + 1, total + 1):
        phase, phase_epoch = phase_at(epoch, config.generator_phases)
        if phase != previous_phase:
            _change_phase(optimizer, scheduler, rates[phase])
            previous_phase = phase
        synchronize(device)
        started = time.perf_counter()
        states, targets = bank.pairs(seed, epoch-1)
        model.train()
        weighted = 0.0
        updates = 0
        for first in range(0, len(states), config.effective_batch):
            stop = min(first + config.effective_batch, len(states))
            optimizer.zero_grad(set_to_none=True)
            for micro in range(first, stop, config.generator_micro_batch):
                last = min(micro + config.generator_micro_batch, stop)
                state, target = states[micro:last].to(device), targets[micro:last].to(device)
                loss = (model(state) - target).square().mean() / target_scale
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("generator loss nonfinite")
                (loss * ((last-micro)/(stop-first))).backward()
                weighted += float(loss.detach()) * (last-micro)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        fit_affine_minimum_norm(model, states, targets, device=device, subtract_quadratic=True)
        scheduler.step()
        boundary = phase_epoch == config.generator_phases[phase]
        validation = None
        if epoch % config.validation_interval == 0 or boundary or epoch == stop_after_epoch:
            model.eval()
            validation = _validation(FixedBandwidthGridGenerator(model, bank.training.shape[-1], reference_grid=bank.training.shape[-1]),
                                     None, bank, device, config)
            best = _select(best, validation, epoch, model, eligible=phase == 1)
        synchronize(device)
        row = {"epoch": epoch, "global_trajectory_epoch": epoch, "phase": phase,
               "phase_epoch": phase_epoch, "optimizer_updates": updates,
               "cumulative_optimizer_updates": sum(v["optimizer_updates"] for v in history)+updates,
               "trajectories_visited": len(states), "normalized_training_mse": weighted/len(states),
               "learning_rate_next": optimizer.param_groups[0]["lr"],
               "validation": validation, "selection_eligible": phase == 1,
               "seconds": time.perf_counter()-started}
        history.append(row)
        final = epoch == total
        payload = _boundary(model, optimizer, scheduler, epoch, phase, history, best,
                            completed=final, extra={"target_scale": target_scale, "setup_seconds": setup_seconds})
        if epoch % checkpoint_interval == 0 or boundary or final or epoch == stop_after_epoch or epoch == 500:
            journal.save(payload, epoch500=epoch == 500)
        if validation:
            print(json.dumps({"stage": "generator", "seed": seed, **row}, allow_nan=False), flush=True)
            if best and best["epoch"] == epoch:
                _atomic_torch(journal.output / "best.pt", _generator_export(model, best["model"], config, seed, bank,
                                                                           epoch, best["val_metric"], _cost(history, config.generator_phases, affine_fits=epoch+1, setup_seconds=setup_seconds)))
        if epoch == stop_after_epoch and not final:
            return {"status": "paused_at_committed_epoch", "epoch": epoch, "output": str(journal.output)}
    cost = _cost(history, config.generator_phases, affine_fits=total+1, setup_seconds=setup_seconds)
    terminal_metric = history[-1]["validation"]["val_metric"]
    _atomic_torch(journal.output / "terminal.pt", _generator_export(model, clone_state(model), config, seed, bank,
                                                                   total, terminal_metric, cost))
    if best:
        selected = _generator_export(model, best["model"], config, seed, bank, best["epoch"], best["val_metric"], cost)
        _atomic_torch(journal.output / "model.pt", selected)
        _atomic_torch(journal.output / "best.pt", selected)
    report = {"status": "complete" if best else "failed_no_finite_validation", "stage": "generator",
              "selected_epoch": None if best is None else best["epoch"],
              "val_metric": None if best is None else best["val_metric"], "training_cost": cost,
              "runtime": journal.runtime, "config": asdict(config), "output": str(journal.output)}
    _atomic_json(journal.output / "COMPLETE.json", report)
    return report


def _branch_epoch(model, frozen, bank, optimizer, config, seed, global_epoch, phase, device, scales):
    """Native derivative/short/long objectives accumulated to true trajectory batch32.

    Long rollout retains the native ten-step truncated graph boundaries. Each
    microbatch's derivative or target-mean loss is weighted by its real sample
    count; clipping and AdamW occur once per effective group, including its tail.
    """
    model.train()
    if phase == 0:
        states, targets = bank.pairs(seed, global_epoch)
        targets = torch.from_numpy(_project(targets.numpy(), ~_square_mask(21)))
        lows = torch.from_numpy(_compute_low_rhs(frozen, states.numpy(), device))
        values = (states, targets, lows)
        count = len(states)
    else:
        sequences = torch.from_numpy(bank.sequences(seed, global_epoch))
        count = len(sequences)
    # Sequence slot order and internal integration steps are separate: .04s
    # observations change unavailable .1/.5s targets to .08/.48s only.
    integration_steps = list(getattr(bank, "integration_steps", ROLLOUT_OFFSETS))
    total, updates = 0.0, 0
    for first in range(0, count, config.effective_batch):
        stop = min(first+config.effective_batch, count)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(first, stop, config.branch_micro_batch):
            last = min(micro+config.branch_micro_batch, stop)
            weight = (last-micro)/(stop-first)
            if phase == 0:
                state, target, low = (v[micro:last].to(device) for v in values)
                loss = ((model.forward_with_generator_output(state, low)-target)/scales[3]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("branch derivative loss nonfinite")
                (loss*weight).backward()
                value = float(loss.detach())
            else:
                sequence = sequences[micro:last].to(device)
                low, high = _initialize_tracks(model, sequence[:, 0])
                leads = tuple(integration_steps[1:3] if phase == 1 else integration_steps[2:])
                segment, value = [], 0.0
                for step in range(1, max(leads)+1):
                    low, high = _dual_rk4_step(frozen, model, low, high)
                    try:
                        low = _correct_low(low)
                    except RuntimeError as error:
                        # The native gate remains unchanged. Its exact failure
                        # is scientific/numerical, not a retryable infrastructure
                        # error; all unrelated RuntimeErrors retain their type.
                        if str(error) == "recursive local correction gate failed during training":
                            raise FloatingPointError(str(error)) from error
                        raise
                    high = model.project_q21(high)
                    if step in leads:
                        target = model.project_q21(sequence[:, integration_steps.index(step)])
                        loss = ((high-target)/model.high_scale).square().mean()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("branch rollout loss nonfinite")
                        segment.append(loss * (weight/len(leads)))
                        value += float(loss.detach())/len(leads)
                    if phase > 1 and step % 10 == 0:
                        if segment:
                            torch.stack(segment).sum().backward()
                            segment.clear()
                        low, high = low.detach(), high.detach()
                if segment:
                    torch.stack(segment).sum().backward()
            total += value*(last-micro)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise FloatingPointError("branch parameters nonfinite")
        updates += 1
    return total/count, updates


def _branch_export(model, state, generator_path, generator_digest, config, seed, bank, epoch, metric, cost):
    """Preserve the native branch loader schema and add the common pipeline specification."""
    generator_count = cost.get("generator", {}).get("parameter_count")
    branch_count = sum(v.numel() for v in model.parameters())
    return {"schema": "gift.high_frequency_branch.parameters.v2", "method": "GIFT", "stage": config.stage,
            "seed": seed, "training_seed": seed, "model_config": model.export_config(),
            "model_state": state, "state_scale": float(model.state_scale), "high_scale": float(model.high_scale),
            "low_rhs_scale": float(model.low_rhs_scale), "high_rhs_scale": float(model.high_rhs_scale),
            "phase": f"best_future{_grid_spec(bank)['future_steps']}_validation", "epoch": epoch,
            "selection_metric": metric, "val_metric": metric, "selection_data_resolution": 64,
            "fixed_upper_output_cutoff": None, "frozen_generator_sha256": None,
            "generator_weights_sha256": generator_digest,
            "support": {"low_generator": "P21", "learned_rhs": "complete Q21 through incoming-grid Nyquist support",
                        "common_explicit_source": "P22-P32", "coefficient_logits": "P12 before nonlinear activation",
                        "modes_above_32": "state-conditioned local differential action"},
            "equation_terms": [], "memory_or_hidden_state": False,
            "model_spec": {"method": "GIFT", "family": "gift", "history": 1,
                           "generator_checkpoint": str(Path(generator_path).resolve()),
                           "generator_weights_sha256": generator_digest, "grid": 64,
                           **_grid_spec(bank), "correction": True,
                           "branch_configuration": model.export_config(),
                           "p21_correction_policy": asdict(DEFAULT_P21_CORRECTION_POLICY),
                           "q21_correction_policy": asdict(DEFAULT_Q21_CORRECTION_POLICY)},
            "config": asdict(config), "training_data": bank.binding, "data_binding": bank.binding,
            "normalization": {"state_scale": float(model.state_scale), "high_scale": float(model.high_scale),
                              "low_rhs_scale": float(model.low_rhs_scale), "high_rhs_scale": float(model.high_rhs_scale)},
            "training_cost": cost, "data_profile": "w1_v2_common_t5_t8",
            "parameter_counts": {"generator": generator_count, "branch": branch_count,
                                 "total": None if generator_count is None else generator_count+branch_count,
                                 "convention": "native_parameter_tensor_numel"}}


def run_branch(dataset, low_model, output, *, seed, config=GIFTTrainingConfig(), device="cuda",
               resume=False, checkpoint_interval=20, stop_after_epoch=None):
    """Train a fresh branch against the selected frozen generator, with complete phase resume."""
    config.validate()
    setup_started = time.perf_counter()
    if config.cutoff != 21 or checkpoint_interval < 1:
        raise ValueError("branch requires cutoff21 and positive checkpoint interval")
    total, offset = sum(config.branch_phases), sum(config.generator_phases)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= total:
        raise ValueError("stop epoch outside branch budget")
    device = torch.device(device)
    configure_determinism(seed, strict=False)
    bank = PredictionObservationBank(dataset, config)
    generator_payload = torch.load(low_model, map_location="cpu", weights_only=True)
    if generator_payload.get("val_metric") is None or generator_payload.get("training_regime") != "w1_v2_common_t5_t8":
        raise ValueError("branch requires a finite-validation W1-v2 selected generator")
    if (generator_payload.get("training_data", {}).get("observations_sha256") != bank.binding["observations_sha256"]
            or generator_payload["model_configuration"]["cutoff"] != config.cutoff):
        raise ValueError("selected generator scientific inputs/support differ")
    generator_digest = state_digest(generator_payload["model_state_dict"])
    generator = load_generator(generator_payload, device=device).eval().requires_grad_(False)
    frozen = FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    states, targets = bank.pairs(seed, offset)
    qtargets = _project(targets.numpy(), ~_square_mask(21))
    low = _compute_low_rhs(frozen, states.numpy(), device)
    scales = _training_scales(states.numpy(), qtargets, low)
    del states, targets, qtargets, low
    model = HighFrequencyBranch(HighFrequencyConfig(), state_scale=scales[0], high_scale=scales[1],
                                low_rhs_scale=scales[2], high_rhs_scale=scales[3], frozen_generator=frozen).to(device)
    identity = {"role": "w1_v2_gift_branch", "configuration": asdict(config), "seed": seed,
                "data": bank.binding, "generator_weights_sha256": generator_digest,
                "scales": list(scales), "sources": _sources(), "device": str(device)}
    journal = _Journal(output, identity, resume)
    saved = journal.payload
    _validate_boundary(saved, config.branch_phases)
    if saved and saved["completed"] and (journal.output / "COMPLETE.json").exists():
        return json.loads((journal.output / "COMPLETE.json").read_text(encoding="utf-8"))
    if saved:
        model.load_state_dict(saved["model"], strict=True)
    optimizer, scheduler, rates = _optimizer(model.parameters(), "branch", config)
    completed = 0 if saved is None else saved["epoch"]
    if saved:
        restore_optimizer(optimizer, saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(journal.rng)
    if stop_after_epoch is not None and stop_after_epoch <= completed:
        raise ValueError("stop epoch must follow saved progress")
    history = [] if saved is None else saved["history"]
    best = None if saved is None else saved["best"]
    previous_phase = 0 if saved is None else saved["phase"]
    plateau_best = math.inf if saved is None else saved["plateau_best"]
    bad_validations = 0 if saved is None else saved["bad_validations"]
    stopped = False if saved is None else saved.get("early_stopped", False)
    setup_seconds = (0. if saved is None else saved.get("setup_seconds", 0.))+time.perf_counter()-setup_started
    if saved is None:
        journal.save(_boundary(model, optimizer, scheduler, 0, 0, history, best,
                               extra={"plateau_best": plateau_best, "bad_validations": bad_validations,
                                      "early_stopped": False, "setup_seconds": setup_seconds}))
    for epoch in range(completed+1, total+1):
        phase, phase_epoch = phase_at(epoch, config.branch_phases)
        if phase != previous_phase:
            _change_phase(optimizer, scheduler, rates[phase])
            previous_phase = phase
        synchronize(device)
        started = time.perf_counter()
        loss, updates = _branch_epoch(model, frozen, bank, optimizer, config, seed,
                                      offset+epoch-1, phase, device, scales)
        scheduler.step()
        boundary = phase_epoch == config.branch_phases[phase]
        validation = None
        if (offset+epoch) % config.validation_interval == 0 or boundary or epoch == stop_after_epoch:
            model.eval()
            validation = _validation(frozen, model, bank, device, config)
            best = _select(best, validation, epoch, model)
            # Plateau counting begins in the final curriculum phase; earlier
            # derivative validation cannot prevent planned rollout training.
            if phase == 3 and offset+epoch >= config.minimum_pipeline_epochs:
                value = validation["val_metric"]
                if value is not None and value < plateau_best*(1-config.relative_improvement):
                    plateau_best, bad_validations = value, 0
                else:
                    bad_validations += 1
                stopped = config.early_stopping and bad_validations >= config.patience
        synchronize(device)
        row = {"epoch": epoch, "global_trajectory_epoch": offset+epoch, "phase": phase,
               "phase_epoch": phase_epoch, "optimizer_updates": updates,
               "cumulative_optimizer_updates": sum(v["optimizer_updates"] for v in history)+updates,
               "trajectories_visited": len(bank.training), "normalized_training_mse": loss,
               "learning_rate_next": optimizer.param_groups[0]["lr"], "validation": validation,
               "seconds": time.perf_counter()-started}
        history.append(row)
        final = epoch == total or stopped
        payload = _boundary(model, optimizer, scheduler, epoch, phase, history, best, completed=final,
                            extra={"plateau_best": plateau_best, "bad_validations": bad_validations,
                                   "early_stopped": stopped, "setup_seconds": setup_seconds})
        if epoch % checkpoint_interval == 0 or boundary or final or epoch == stop_after_epoch or offset+epoch == 500:
            journal.save(payload, epoch500=offset+epoch == 500)
        if validation:
            print(json.dumps({"stage": "branch", "seed": seed, **row}, allow_nan=False), flush=True)
            if best and best["epoch"] == epoch:
                generator_cost = generator_payload["training_cost"]
                current_cost = _cost(history, config.branch_phases, setup_seconds=setup_seconds)
                pipeline_cost = {"generator": generator_cost, "branch": current_cost,
                                 "total_trajectory_epochs": generator_cost["trajectory_epochs"]+epoch,
                                 "total_optimizer_updates": generator_cost["optimizer_updates"]+current_cost["optimizer_updates"],
                                 "total_committed_seconds": generator_cost["committed_seconds"]+current_cost["committed_seconds"],
                                 "total_training_seconds": generator_cost.get("total_training_seconds", generator_cost["committed_seconds"])+current_cost["total_training_seconds"],
                                 "generator_weights_sha256": generator_digest}
                _atomic_torch(journal.output / "best.pt", _branch_export(
                    model, best["model"], low_model, generator_digest, config, seed, bank,
                    epoch, best["val_metric"], pipeline_cost))
        if epoch == stop_after_epoch and not final:
            return {"status": "paused_at_committed_epoch", "epoch": epoch, "output": str(journal.output)}
        if final:
            break
    stage_cost = _cost(history, config.branch_phases, setup_seconds=setup_seconds)
    generator_cost = generator_payload["training_cost"]
    cost = {"generator": generator_cost, "branch": stage_cost,
            "total_trajectory_epochs": generator_cost["trajectory_epochs"]+stage_cost["trajectory_epochs"],
            "total_optimizer_updates": generator_cost["optimizer_updates"]+stage_cost["optimizer_updates"],
            "total_committed_seconds": generator_cost["committed_seconds"]+stage_cost["committed_seconds"],
            "total_training_seconds": generator_cost.get("total_training_seconds", generator_cost["committed_seconds"])+stage_cost["total_training_seconds"],
            "generator_weights_sha256": generator_digest}
    terminal_metric = history[-1]["validation"]["val_metric"]
    _atomic_torch(journal.output / "terminal.pt", _branch_export(model, clone_state(model), low_model,
                  generator_digest, config, seed, bank, len(history), terminal_metric, cost))
    if best:
        selected = _branch_export(model, best["model"], low_model, generator_digest, config, seed,
                                  bank, best["epoch"], best["val_metric"], cost)
        _atomic_torch(journal.output / "model.pt", selected)
        _atomic_torch(journal.output / "best.pt", selected)
    report = {"status": "complete" if best else "failed_no_finite_validation", "stage": "branch",
              "selected_epoch": None if best is None else best["epoch"],
              "val_metric": None if best is None else best["val_metric"],
              "training_cost": cost, "early_stopped": stopped, "runtime": journal.runtime,
              "config": asdict(config), "output": str(journal.output)}
    _atomic_json(journal.output / "COMPLETE.json", report)
    return report


def load_gift_checkpoint(checkpoint, device="cpu"):
    """Strictly reconstruct the selected pair and verify its frozen-generator tensor binding."""
    checkpoint, device = Path(checkpoint).resolve(strict=True), torch.device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = payload["model_spec"]
    if spec.get("family") != "gift" or spec.get("history") != 1:
        raise ValueError("checkpoint is not a single-initial-field W1-v2 GIFT pair")
    _validate_checkpoint_grid(payload)
    from experiments.model_locations import generator_checkpoint
    generator = load_generator(generator_checkpoint(checkpoint, payload), device=device).eval().requires_grad_(False)
    if state_digest(generator.state_dict()) != spec["generator_weights_sha256"]:
        raise ValueError("paired generator tensor binding differs")
    low = FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    branch, native_payload = load_high_frequency_model(checkpoint, device, low)
    if native_payload["generator_weights_sha256"] != spec["generator_weights_sha256"]:
        raise ValueError("branch and common generator binding disagree")
    return low, branch, payload


def _validate_checkpoint_grid(payload):
    """Reject output-time or native-step mismatches before reconstructing a model.

    Historical dense exports declare duration/save_interval/step only. New
    exports additionally bind each output query time and observation spacing.
    """
    spec = payload["model_spec"]
    duration, interval, step = (float(spec[key]) for key in ("duration", "save_interval", "step"))
    if (not np.isclose(duration, 3., atol=2e-12, rtol=0)
            or not np.isclose(step, INTEGRATION_DT, atol=2e-12, rtol=0)
            or interval not in (.02, .04, .1)):
        raise ValueError("GIFT checkpoint output/native integration grid differs")
    times = np.arange(round(duration/interval)+1, dtype=np.float64)*interval
    expected_steps = _rollout_steps(interval).tolist()
    if interval == .04 and "rollout_integration_steps" not in spec:
        raise ValueError("GIFT .04 checkpoint must declare its changed supervision leads")
    for declaration in (spec, payload.get("data_binding", payload.get("training_data", {}))):
        if "query_times" in declaration:
            observed = np.asarray(declaration["query_times"], dtype=np.float64)
            if observed.shape != times.shape or not np.allclose(observed, times, atol=2e-12, rtol=0):
                raise ValueError("GIFT checkpoint query-time binding differs")
        if declaration.get("points", len(times)) != len(times) or declaration.get("future_steps", len(times)-1) != len(times)-1:
            raise ValueError("GIFT checkpoint frame count differs")
        if not np.isclose(declaration.get("observation_dt", interval), interval, atol=2e-12, rtol=0):
            raise ValueError("GIFT checkpoint observation spacing differs")
        if not np.isclose(declaration.get("integration_dt", step), step, atol=2e-12, rtol=0):
            raise ValueError("GIFT checkpoint integration spacing differs")
        if declaration.get("rollout_integration_steps", expected_steps) != expected_steps:
            raise ValueError("GIFT checkpoint supervision step binding differs")
        if declaration.get("rollout_observation_offsets", _observation_offsets(interval).tolist()) != _observation_offsets(interval).tolist():
            raise ValueError("GIFT checkpoint observation target binding differs")
        if declaration.get("rollout_native_requested_steps", ROLLOUT_OFFSETS.tolist()) != ROLLOUT_OFFSETS.tolist():
            raise ValueError("GIFT checkpoint native requested target binding differs")
        expected_policy = "nearest_observed_lead_ties_earlier" if interval == .04 else "exact_native_leads"
        if declaration.get("rollout_target_policy", expected_policy) != expected_policy:
            raise ValueError("GIFT checkpoint target mapping policy differs")
        actual_seconds = np.asarray(declaration.get("rollout_supervision_seconds", _rollout_steps(interval)*INTEGRATION_DT))
        if actual_seconds.shape != (8,) or not np.allclose(actual_seconds, _rollout_steps(interval)*INTEGRATION_DT, atol=2e-12, rtol=0):
            raise ValueError("GIFT checkpoint physical supervision lead binding differs")
    return times


def predict_gift_checkpoint(checkpoint, initial, trajectory_ids, *, device="cpu", batch_size=4):
    """Initial-only bound-grid prediction, preserving native trajectory-local failures."""
    low, branch, payload = load_gift_checkpoint(checkpoint, device)
    spec = payload["model_spec"]
    _validate_checkpoint_grid(payload)
    result = rollout_gift(low=low, branch=branch, initial_state=np.asarray(initial, dtype=np.float32),
                        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64), duration=spec["duration"],
                        save_interval=spec["save_interval"], step=spec["step"], device=torch.device(device),
                        batch_size=batch_size, correction=True, execution="eager")
    points = spec.get("points", round(spec["duration"]/spec["save_interval"])+1)
    if result.prediction.shape != (len(initial), points, 64, 64):
        raise ValueError("GIFT prediction shape differs from the bound model output grid")
    return result


def run_pipeline(dataset, output, *, seed, config=GIFTTrainingConfig(), device="cuda", resume=False):
    """Run/resume generator then branch and return the selected inference paths and both costs."""
    output = output_path(output, ROOT)
    generator_dir, branch_dir = output / "generator", output / "branch"
    generator = run_generator(dataset, generator_dir, seed=seed, config=config, device=device,
                              resume=resume and generator_dir.exists())
    paths = {"generator_model": str(generator_dir / "model.pt"),
             "branch_model": str(branch_dir / "model.pt"),
             "generator_terminal": str(generator_dir / "terminal.pt"),
             "branch_terminal": str(branch_dir / "terminal.pt")}
    if generator["status"] != "complete":
        return {"status": "failed_no_finite_generator_validation", "generator": generator, "paths": paths}
    branch = run_branch(dataset, generator_dir / "model.pt", branch_dir, seed=seed,
                        config=config, device=device, resume=resume and branch_dir.exists())
    result = {"status": branch["status"], "generator": generator, "branch": branch,
              "paths": paths, "training_cost": branch["training_cost"], "val_metric": branch["val_metric"]}
    _atomic_json(output / "PIPELINE.json", result)
    return result


def preflight(dataset, *, device="cuda", seed=7, rows=32, setup_rows=None, config=GIFTTrainingConfig()):
    """Measure real native work on a training-only subset; never read validation or select weights.

    rows controls the timed effective group. setup_rows controls training-only
    RMS, derivative scale, affine fits and branch scales; its default is rows,
    preserving the original32-row probe. A larger setup uses the common epoch0
    permutation's last rows as the measured tail, without redrawing IDs. The
    generator is still disposable (one timed update plus affine refit), not a
    validation-selected, fully trained generator. No models are exported.
    """
    config.validate()
    if type(rows) is not int or not 2 <= rows <= min(32, config.effective_batch):
        raise ValueError("preflight uses2..32 training trajectories")
    setup_rows = rows if setup_rows is None else setup_rows
    if type(setup_rows) is not int or setup_rows < rows:
        raise ValueError("preflight setup_rows must be an integer at least rows")
    setup_started = time.perf_counter()
    device = torch.device(device)
    configure_determinism(seed, strict=False)
    with h5py.File(dataset, "r") as h:
        from training.w1_v2_data import SUPPORTED_SCHEMAS
        if h.attrs.get("schema") not in SUPPORTED_SCHEMAS or h.attrs.get("status") != "complete":
            raise ValueError("preflight requires the completed common slice")
        population = h["training/vorticity"].shape[0]
        if setup_rows > population:
            raise ValueError("preflight setup_rows exceeds the training population")
        raw = np.asarray(h["training/vorticity"][:setup_rows], dtype=np.float32)
        ids = h["training/trajectory_index"][:setup_rows].astype(int).tolist()
        times = np.asarray(h["training/time"][:], dtype=np.float64)
        absolute = np.asarray(h["training/absolute_time"][:], dtype=np.float64)
        if (raw.shape != (setup_rows, len(times), 64, 64) or not np.isfinite(raw).all()
                or not np.allclose(absolute, 5+times, atol=2e-12, rtol=0)):
            raise ValueError("invalid training-only preflight observations")
    bank = SimpleNamespace(training=raw, binding={"query_times": times.tolist()})
    grid = _grid_spec(bank)
    offsets = _observation_offsets(grid["observation_dt"])
    bank.integration_steps = _rollout_steps(grid["observation_dt"])
    def locations(epoch, rollout):
        order = np.random.default_rng(np.random.SeedSequence([config.order_seed, epoch])).permutation(setup_rows)
        rng = np.random.default_rng(np.random.SeedSequence([config.sampling_seed, epoch, 0, int(rollout)]))
        positions = (rng.integers(0, len(times)-int(offsets[-1]), setup_rows)
                     if rollout else rng.integers(2, len(times)-2, setup_rows))
        return order, positions
    def setup_pairs(epoch):
        return _grid_pairs(raw, *locations(epoch, False), grid["observation_dt"])
    def measured_pairs(unused_seed, epoch):
        states, targets = setup_pairs(epoch)
        return states[-rows:], targets[-rows:]
    bank.pairs = measured_pairs
    bank.sequences = lambda unused_seed, epoch: _grid_sequences(raw, *locations(epoch, True), offsets)[-rows:]
    measured_order = locations(0, False)[0][-rows:]
    measurement_ids = [ids[int(row)] for row in measured_order]
    state_scale = float(np.sqrt(np.square(raw, dtype=np.float64).mean()))
    generator = GIFTGenerator(21, 8, state_scale, False).to(device)
    parameters = configure_variable_projection_parameters(generator)
    setup_states, setup_targets = setup_pairs(0)
    states, targets = setup_states[-rows:], setup_targets[-rows:]
    synchronize(device); started = time.perf_counter()
    fit_affine_minimum_norm(generator, setup_states, setup_targets, device=device, subtract_quadratic=False)
    synchronize(device); initial_affine_seconds = time.perf_counter()-started
    target_scale = float(setup_targets.square().mean().clamp_min(1e-20))
    optimizer, _, _ = _optimizer(parameters, "generator", config)
    synchronize(device); started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    weighted = 0.0
    for first in range(0, rows, config.generator_micro_batch):
        stop = min(first+config.generator_micro_batch, rows)
        state, target = states[first:stop].to(device), targets[first:stop].to(device)
        loss = (generator(state)-target).square().mean()/target_scale
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("preflight generator loss nonfinite")
        (loss*((stop-first)/rows)).backward()
        weighted += float(loss.detach())*(stop-first)/rows
    torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True); optimizer.step()
    synchronize(device); gradient_seconds = time.perf_counter()-started
    started = time.perf_counter()
    fit_affine_minimum_norm(generator, setup_states, setup_targets, device=device, subtract_quadratic=True)
    synchronize(device); refit_seconds = time.perf_counter()-started
    generator.eval().requires_grad_(False)
    frozen = FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    synchronize(device); scale_started = time.perf_counter()
    qtargets = _project(setup_targets.numpy(), ~_square_mask(21))
    low = _compute_low_rhs(frozen, setup_states.numpy(), device)
    scales = _training_scales(setup_states.numpy(), qtargets, low)
    synchronize(device); branch_scale_setup_seconds = time.perf_counter()-scale_started
    branch = HighFrequencyBranch(HighFrequencyConfig(), state_scale=scales[0], high_scale=scales[1],
                                low_rhs_scale=scales[2], high_rhs_scale=scales[3], frozen_generator=frozen).to(device)
    optimizer, scheduler, rates = _optimizer(branch.parameters(), "branch", config)
    synchronize(device)
    setup_seconds = time.perf_counter()-setup_started-gradient_seconds
    records = []
    for phase, name in enumerate(("derivative", "short_rollout", "long_rollout_1", "long_rollout_2")):
        if phase:
            _change_phase(optimizer, scheduler, rates[phase])
        synchronize(device); started = time.perf_counter()
        loss, updates = _branch_epoch(branch, frozen, bank, optimizer, config, seed, 0, phase, device, scales)
        synchronize(device)
        records.append({"phase": name, "rows": rows, "updates": updates,
                        "seconds": time.perf_counter()-started, "native_loss": loss})
    return {"schema": "gift.w1-v2.training-only-cost-preflight.v1", "status": "complete",
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "training_ids": measurement_ids if setup_rows > rows else ids,
            "setup_training_ids": ids, "measurement_training_ids": measurement_ids,
            "setup_rows": setup_rows, "measurement_rows": rows, "training_population": population,
            "measurement_selection": ("whole_setup_population" if setup_rows == rows else
                                      "common_epoch0_tail" if setup_rows % config.effective_batch == rows else
                                      "common_epoch0_last_requested_rows"),
            "sampling_epoch": 0, "validation_read": False, "test_read": False,
            "config": asdict(config), "seed": seed, "dataset": str(Path(dataset).resolve()),
            "time_grid": grid,
            "setup": {"rows": setup_rows, "full_training_population": setup_rows == population,
                      "seconds_excluding_timed_generator_gradient": setup_seconds,
                      "branch_scale_seconds": branch_scale_setup_seconds,
                      "state_scale": state_scale, "generator_target_scale": target_scale,
                      "branch_scales": list(scales),
                      "lifecycle": "initial_affine_on_setup_then_one_measured_generator_update_then_affine_refit_on_setup; branch_continues_one_measured_update_per_native_phase",
                      "validation_selected_generator": False},
            "generator": {"rows": rows, "gradient_updates": 1, "native_loss": weighted,
                          "gradient_seconds": gradient_seconds, "initial_affine_seconds": initial_affine_seconds,
                          "affine_refit_seconds": refit_seconds, "affine_rows": setup_rows,
                          "full_training_affine_time_measured": setup_rows == population,
                          "full1000_affine_time_measured": setup_rows == population == 1000},
            "branch": records, "sources": _sources(),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def main(argv=None):
    """Expose direct stage/pipeline calls; dry-run prints budgets without opening a GPU."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    action.add_argument("--resume", action="store_true")
    action.add_argument("--preflight", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("pipeline", "generator", "branch"), default="pipeline")
    parser.add_argument("--profile", choices=("development", "formal"), default="formal")
    parser.add_argument("--lr-multiplier", type=float, default=1.)
    parser.add_argument("--low-model", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--preflight-rows", type=int, default=32)
    parser.add_argument("--preflight-setup-rows", type=int)
    args = parser.parse_args(argv)
    config = GIFTTrainingConfig.development(args.lr_multiplier) if args.profile == "development" else GIFTTrainingConfig(lr_multiplier=args.lr_multiplier)
    if args.dry_run:
        result = {"config": asdict(config), "seed": args.seed, "writes": 0,
                  "generator_epochs": sum(config.generator_phases), "branch_epochs": sum(config.branch_phases)}
    elif args.preflight:
        result = preflight(args.dataset, device=args.device, seed=args.seed, config=config,
                           rows=args.preflight_rows, setup_rows=args.preflight_setup_rows)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    else:
        try:
            if args.stage == "pipeline":
                if args.stop_after_epoch is not None:
                    parser.error("use a single stage for explicit stop-after-epoch")
                result = run_pipeline(args.dataset, args.output, seed=args.seed, config=config,
                                      device=args.device, resume=args.resume)
            else:
                fn = run_generator if args.stage == "generator" else run_branch
                extra = {} if args.stage == "generator" else {"low_model": args.low_model}
                if args.stage == "branch" and args.low_model is None:
                    parser.error("branch needs --low-model")
                result = fn(args.dataset, output=args.output, seed=args.seed, config=config,
                            device=args.device, resume=args.resume, stop_after_epoch=args.stop_after_epoch, **extra)
            if result["status"] not in ("complete", "paused_at_committed_epoch"):
                raise FloatingPointError("GIFT pipeline has no finite qualifying validation: "+result["status"])
        except Exception as error:
            # This writes only the current requested new-run failure record;
            # the original exception remains nonzero for the orchestrator.
            output = output_path(args.output, ROOT)
            output.mkdir(parents=True, exist_ok=True)
            _atomic_json(output / "FAILURE.json", {
                "method": "GIFT", "seed": args.seed, "stage": args.profile,
                "status": "failed", "reason": repr(error),
                "failure_kind": "numerical" if isinstance(error, (FloatingPointError, ArithmeticError)) else "infrastructure",
            })
            raise
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
