"""Train the three formal GIFT high-frequency branches from project data.

The P21 generator is frozen. Every trainable branch receives the complete
N64 state, its Q21 component, and the same-state frozen generator output.
The selected observation cohort supplies 50 training and 20 validation
trajectories; test and higher-resolution fields are not read. The ordinary
four-vortex layout stores these training/validation rows as local IDs 0--69.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.formal._shared.common import (
    ProjectPaths,
    SEEDS,
    create_output,
    default_project_root,
    file_record,
    sha256_file,
    write_csv_new,
)
from experiments.formal._shared.gift_runtime import (
    DEFAULT_P21_CORRECTION_POLICY,
    apply_local_correction,
    load_low_generator,
)
from experiments.formal._shared.high_frequency import (
    HighFrequencyBranch,
    HighFrequencyConfig,
)
from training.gift_data import PROFILE_SOURCES, validate_input, validate_low_prerequisite
from training.gift_execution import (
    EXECUTION_CHOICES, EXECUTION_HELP, EXECUTION_SOURCES,
    configure_training_runtime, resolve_execution,
)


DT = 0.02
DERIVATIVE_EPOCHS = 18
DERIVATIVE_BATCH_SIZE = 16
SHORT_ROLLOUT_EPOCHS = 2
LONG_ROLLOUT_STAGES = 2
ROLLOUT_BATCH_SIZE = 5
VALIDATION_ROLLOUT_BATCH_SIZE = 10
DERIVATIVE_CENTRES = tuple(range(2, 499, 5))
ROLLOUT_STEPS = (0, 5, 10, 20, 25, 30, 40, 50)
ROLLOUT_ANCHORS = (0, 100, 200, 300, 400)


def parse_args() -> argparse.Namespace:
    """Select one branch seed, its explicitly trained low generator and the execution action."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--low-model", type=Path, help="explicit newly trained frozen P21 prerequisite for this independent branch run")
    parser.add_argument("--data-profile", choices=("released", "regenerated", "gaussian"), default="released")
    parser.add_argument("--resume", action="store_true", help="continue only this seed's own saved training run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution", choices=EXECUTION_CHOICES, default="auto", help=EXECUTION_HELP)
    return parser.parse_args()


def _expected_lite_ids(handle):
    """Return the declared 50-trajectory training and 20-trajectory validation populations."""
    from gift.prediction_cohorts import GAUSSIAN, observation_cohort, partitions
    if observation_cohort(handle) == GAUSSIAN:
        split = partitions(GAUSSIAN)
        return [('training', np.asarray(split['gift_lite_training'], dtype=np.int64)),
                ('validation', np.asarray(split['checkpoint_validation'], dtype=np.int64))]
    return [('training', np.arange(50, dtype=np.int64)),
            ('validation', np.arange(50, 70, dtype=np.int64))]


def _validate_training_source_metadata(paths: ProjectPaths) -> dict[str, Any]:
    """Check observation IDs, float32 N64 fields and 501 native dt=0.02 times without loading fields."""
    paths.require("standard_n64", "low_model")
    groups: dict[str, Any] = {}
    with h5py.File(paths.standard_n64, "r") as handle:
        for split, expected_ids in _expected_lite_ids(handle):
            ids = np.asarray(handle[f"{split}/trajectory_index"][:], dtype=np.int64)
            times = np.asarray(handle[f"{split}/time"][:], dtype=np.float64)
            field = handle[f"{split}/vorticity"]
            expected_shape = (len(expected_ids), 501, 64, 64)
            if not np.array_equal(ids, expected_ids):
                raise ValueError(f"{split} trajectory IDs differ")
            if not np.allclose(
                times, np.arange(501, dtype=np.float64) * DT, rtol=0.0, atol=2e-12
            ):
                raise ValueError(f"{split} time support differs")
            if field.shape != expected_shape or field.dtype != np.dtype("float32"):
                raise ValueError(f"{split} raw vorticity shape or dtype differs")
            groups[split] = {
                "trajectory_ids": [int(ids[0]), int(ids[-1])],
                "trajectory_count": len(ids),
                "field_shape": list(field.shape),
                "time_interval": [float(times[0]), float(times[-1])],
            }
    return groups


def _set_seed(seed: int) -> None:
    """Set all initialization RNGs and the supported training precision policy before constructing models."""
    configure_training_runtime(strict=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _square_mask(cutoff: int) -> np.ndarray:
    """Build a square N64 Fourier mask in native FFT ordering."""
    modes = np.fft.fftfreq(64, d=1.0 / 64.0)
    ky, kx = np.meshgrid(modes, modes, indexing="ij")
    return (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)


def _project(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Project [...,64,64] fields into a Fourier band and return real float32 arrays."""
    return np.fft.ifft2(
        np.fft.fft2(value, axes=(-2, -1)) * mask, axes=(-2, -1)
    ).real.astype(np.float32)


def _build_split(
    handle: h5py.File, split: str, expected_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build observed Q21 derivative targets and short rollout sequences for a declared split.

    The derivative uses every fifth centre of a fourth-order finite difference.
    Sequences use five absolute-time anchors and eight saved offsets, so training
    receives observed states rather than analytic PDE derivatives.
    """
    ids = np.asarray(handle[f"{split}/trajectory_index"][:], dtype=np.int64)
    times = np.asarray(handle[f"{split}/time"][:], dtype=np.float64)
    field = handle[f"{split}/vorticity"]
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{split} trajectory IDs differ")
    if not np.allclose(times, np.arange(501) * DT, atol=2e-12, rtol=0.0):
        raise ValueError(f"{split} time support differs")
    if field.shape != (len(ids), 501, 64, 64) or field.dtype != np.dtype("float32"):
        raise ValueError(f"{split} raw vorticity shape or dtype differs")

    centres = np.asarray(DERIVATIVE_CENTRES, dtype=np.int64)
    states = np.empty((len(ids) * len(centres), 64, 64), dtype=np.float32)
    q21_targets = np.empty_like(states)
    sequences = np.empty(
        (len(ids) * len(ROLLOUT_ANCHORS), len(ROLLOUT_STEPS), 64, 64),
        dtype=np.float32,
    )
    q21 = ~_square_mask(21)
    sample = 0
    sequence_row = 0
    rollout_offsets = np.asarray(ROLLOUT_STEPS, dtype=np.int64)
    for trajectory_row in range(len(ids)):
        raw = np.asarray(field[trajectory_row], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"{split} contains nonfinite vorticity")
        derivative = (
            raw[centres - 2]
            - 8.0 * raw[centres - 1]
            + 8.0 * raw[centres + 1]
            - raw[centres + 2]
        ) / (12.0 * DT)
        stop = sample + len(centres)
        states[sample:stop] = raw[centres]
        q21_targets[sample:stop] = _project(derivative, q21)
        sample = stop
        for anchor in ROLLOUT_ANCHORS:
            sequences[sequence_row] = raw[anchor + rollout_offsets]
            sequence_row += 1
    return states, q21_targets, sequences


def _load_raw_training_data(
    paths: ProjectPaths,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Load only the declared training/validation splits into derivative and sequence arrays."""
    paths.require("standard_n64")
    with h5py.File(paths.standard_n64, "r") as handle:
        expected = dict(_expected_lite_ids(handle))
        training = _build_split(handle, "training", expected['training'])
        validation = _build_split(handle, "validation", expected['validation'])
    return training, validation


@torch.no_grad()
def _compute_low_rhs(
    generator: Callable[[torch.Tensor], torch.Tensor],
    states: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Evaluate the frozen generator on [sample,64,64] states in non-gradient batches."""
    output = np.empty_like(states)
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        batch = torch.from_numpy(states[start:stop]).to(device)
        output[start:stop] = generator(batch).detach().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("frozen generator produced nonfinite values")
    return output


def _rms(values: np.ndarray) -> float:
    """Compute a positive finite training-field RMS with float64 accumulation."""
    values64 = np.asarray(values, dtype=np.float64)
    result = float(np.sqrt(np.mean(values64 * values64)))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("training scale is not finite and positive")
    return result


def _training_scales(
    states: np.ndarray, q21_rhs: np.ndarray, low_rhs: np.ndarray
) -> tuple[float, float, float, float]:
    """Measure full-state, Q21-state, P21-RHS and Q21-RHS scales from training samples only."""
    p21 = _square_mask(21)
    return (
        _rms(states),
        _rms(_project(states, ~p21)),
        _rms(_project(low_rhs, p21)),
        _rms(q21_rhs),
    )


def _checked(
    rhs: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor
) -> torch.Tensor:
    """Check that the frozen training RHS preserves a stage tensor's shape, dtype and device."""
    value = rhs(state)
    if (
        value.shape != state.shape
        or value.device != state.device
        or value.dtype != state.dtype
    ):
        raise ValueError("training RHS shape, device, or dtype differs")
    return value


def _dual_rk4_step(
    low_rhs: Callable[[torch.Tensor], torch.Tensor],
    branch: HighFrequencyBranch,
    low_state: torch.Tensor,
    high_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance low/high training tracks together with dt=0.02 and same-state generator conditioning.

    This helper follows the training stage definition. The inference rollout has
    its own explicit projection and failure bookkeeping around each stage.
    """
    def stage(
        stage_low: torch.Tensor, stage_high: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate both derivatives from the sum of the low and high stage states."""
        complete = stage_low + stage_high
        frozen = _checked(low_rhs, complete)
        learned = branch.forward_with_generator_output(complete, frozen)
        return frozen, learned

    low1, high1 = stage(low_state, high_state)
    low2, high2 = stage(
        low_state + 0.5 * DT * low1, high_state + 0.5 * DT * high1
    )
    low3, high3 = stage(
        low_state + 0.5 * DT * low2, high_state + 0.5 * DT * high2
    )
    low4, high4 = stage(low_state + DT * low3, high_state + DT * high3)
    return (
        low_state + (DT / 6.0) * (low1 + 2.0 * low2 + 2.0 * low3 + low4),
        high_state
        + (DT / 6.0) * (high1 + 2.0 * high2 + 2.0 * high3 + high4),
    )


def _correct_low(low_state: torch.Tensor) -> torch.Tensor:
    """Apply the gradient-free P21 training correction and reject a failed local gate."""
    with torch.no_grad():
        result = apply_local_correction(
            low_state.detach(), DEFAULT_P21_CORRECTION_POLICY
        )
    if bool(result.gate_failed.any()):
        raise RuntimeError("recursive local correction gate failed during training")
    return result.state


def _initialize_tracks(
    model: HighFrequencyBranch, anchor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an observed anchor into detached P21/Q21 starting states."""
    low, high = model.split_state(anchor)
    return low.detach(), high.detach()


@torch.no_grad()
def _validate_derivative(
    model: HighFrequencyBranch,
    loader: DataLoader,
    device: torch.device,
    high_rhs_scale: float,
) -> dict[str, float]:
    """Accumulate global Q21 derivative error and scale-normalized MSE over validation batches."""
    model.eval()
    error_squared = 0.0
    target_squared = 0.0
    normalized_squared = 0.0
    count = 0
    for state, target, frozen in loader:
        state = state.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        frozen = frozen.to(device, non_blocking=True)
        error = model.forward_with_generator_output(state, frozen) - target
        error_squared += float(torch.sum(error * error))
        target_squared += float(torch.sum(target * target))
        normalized_squared += float(torch.sum((error / high_rhs_scale) ** 2))
        count += error.numel()
    return {
        "normalized_mse": normalized_squared / count,
        "q21_global_relative_l2": math.sqrt(error_squared / target_squared),
    }


@torch.no_grad()
def _validate_rollout(
    model: HighFrequencyBranch,
    low_rhs: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    *,
    return_values: bool = False,
) -> dict[str, float | int | list[float]]:
    """Measure one-unit lead errors after 50 autonomous dt=0.02 training-route steps.

    Loader sequences provide their first observed anchor and final truth. Correction
    affects the low track; the high track is projected back to Q21. Optional returned
    values retain individual rollout errors for S3 seed comparisons.
    """
    model.eval()
    full_relative: list[float] = []
    q21_relative: list[float] = []
    finite_count = 0
    for (sequence,) in loader:
        sequence = sequence.to(device, non_blocking=True)
        low_state, high_state = _initialize_tracks(model, sequence[:, 0])
        for _ in range(50):
            low_state, high_state = _dual_rk4_step(
                low_rhs, model, low_state, high_state
            )
            low_state = _correct_low(low_state)
            high_state = model.project_q21(high_state)
        prediction = low_state + high_state
        truth = sequence[:, -1]
        truth_high = model.project_q21(truth)
        full_relative.extend(
            (
                (prediction - truth).flatten(1).norm(dim=1)
                / truth.flatten(1).norm(dim=1).clamp_min(1.0e-20)
            ).cpu().tolist()
        )
        q21_relative.extend(
            (
                (high_state - truth_high).flatten(1).norm(dim=1)
                / truth_high.flatten(1).norm(dim=1).clamp_min(1.0e-20)
            ).cpu().tolist()
        )
        finite_count += int(torch.isfinite(prediction).flatten(1).all(1).sum())
    result: dict[str, float | int | list[float]] = {
        "lead_1_full_relative_l2": float(np.mean(full_relative)),
        "lead_1_q21_relative_l2": float(np.mean(q21_relative)),
        "finite_count": finite_count,
        "population": len(full_relative),
    }
    if return_values:
        result["full_relative_values"] = full_relative
        result["q21_relative_values"] = q21_relative
    return result


def _short_epoch(
    model: HighFrequencyBranch,
    low_rhs: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Optimize Q21 rollout error at leads 0.1 and 0.2 through ten differentiable RK4 steps."""
    model.train()
    total = 0.0
    rows = 0
    for (sequence,) in loader:
        sequence = sequence.to(device, non_blocking=True)
        low_state, high_state = _initialize_tracks(model, sequence[:, 0])
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for step_index in range(1, 11):
            low_state, high_state = _dual_rk4_step(
                low_rhs, model, low_state, high_state
            )
            low_state = _correct_low(low_state)
            high_state = model.project_q21(high_state)
            if step_index in (5, 10):
                target_index = ROLLOUT_STEPS.index(step_index)
                target = model.project_q21(sequence[:, target_index])
                losses.append(
                    torch.mean(((high_state - target) / model.high_scale) ** 2)
                )
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(sequence)
        rows += len(sequence)
    return total / rows


def _long_epoch(
    model: HighFrequencyBranch,
    low_rhs: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Optimize six Q21 targets over one-unit lead using ten-step truncated backpropagation.

    Gradient contributions accumulate across segments before one optimizer update
    per batch; detached track boundaries limit the retained computation graph.
    """
    model.train()
    targets = (10, 20, 25, 30, 40, 50)
    total = 0.0
    rows = 0
    for (sequence,) in loader:
        sequence = sequence.to(device, non_blocking=True)
        low_state, high_state = _initialize_tracks(model, sequence[:, 0])
        optimizer.zero_grad(set_to_none=True)
        segment: list[torch.Tensor] = []
        batch_value = 0.0
        for step_index in range(1, 51):
            low_state, high_state = _dual_rk4_step(
                low_rhs, model, low_state, high_state
            )
            low_state = _correct_low(low_state)
            high_state = model.project_q21(high_state)
            if step_index in targets:
                target_index = ROLLOUT_STEPS.index(step_index)
                target = model.project_q21(sequence[:, target_index])
                value = torch.mean(((high_state - target) / model.high_scale) ** 2)
                segment.append(value / len(targets))
                batch_value += float(value.detach()) / len(targets)
            if step_index in (10, 20, 30, 40, 50):
                if segment:
                    torch.stack(segment).sum().backward()
                    segment.clear()
                low_state = low_state.detach()
                high_state = high_state.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += batch_value * len(sequence)
        rows += len(sequence)
    return total / rows


def _snapshot(model: HighFrequencyBranch) -> dict[str, torch.Tensor]:
    """Copy branch weights and scale buffers into independent CPU tensors."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _write_checkpoint(
    path: Path,
    model: HighFrequencyBranch,
    *,
    seed: int,
    phase: str,
    epoch: int,
    selection_metric: float,
    frozen_generator_sha256: str | None = None,
    generator_weights_sha256: str | None = None,
    training_data: dict[str, Any] | None = None,
    training_cost: dict[str, Any] | None = None,
) -> None:
    """Publish an inference export with branch weights, scales, spectral support and training provenance.

    Optimizer/sampling progress belongs to the resume journal, not this parameter
    file. The generator digest identifies the paired low model rather than source code.
    """
    payload = {
        "schema": "gift.high_frequency_branch.parameters.v2",
        "method": "GIFT",
        "seed": int(seed),
        "model_config": model.export_config(),
        "state_scale": float(model.state_scale.detach().cpu()),
        "high_scale": float(model.high_scale.detach().cpu()),
        "low_rhs_scale": float(model.low_rhs_scale.detach().cpu()),
        "high_rhs_scale": float(model.high_rhs_scale.detach().cpu()),
        "model_state": _snapshot(model),
        "phase": phase,
        "epoch": int(epoch),
        "selection_metric": float(selection_metric),
        "selection_data_resolution": 64,
        "training_seed": int(seed),
        "fixed_upper_output_cutoff": None,
        "frozen_generator_sha256": frozen_generator_sha256,
        "generator_weights_sha256": generator_weights_sha256,
        "training_data": training_data,
        "data_profile": training_data["profile"] if training_data is not None else "unverified_internal_call",
        "support": {
            "low_generator": "P21",
            "learned_rhs": "complete Q21 through incoming-grid Nyquist support",
            "common_explicit_source": "P22-P32",
            "coefficient_logits": "P12 before nonlinear activation",
            "modes_above_32": "state-conditioned local differential action",
        },
        "equation_terms": [],
        "memory_or_hidden_state": False,
    }
    if training_cost is not None:
        payload["training_cost"] = training_cost
    from training.checkpoints import write_model_export
    write_model_export(path, payload)


def _selection_report(phase, epoch, value):
    """Describe validation selection, earliest-tie rule and the reported qualification threshold."""
    return {
        "phase": phase,
        "epoch": epoch,
        "metric": "N64 validation lead=1 full-field relative L2",
        "value": value,
        "strictly_lower_earliest_tie_break": True,
        "qualification_threshold": 0.10,
        "qualification_passed": value <= 0.10,
    }


def _rollout_loader(
    sequences: np.ndarray, seed: int, *, shuffle: bool, batch_size: int
) -> DataLoader:
    """Wrap observed sequences, giving shuffled training passes their own seeded RNG."""
    return DataLoader(
        TensorDataset(torch.from_numpy(sequences)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(torch.Generator().manual_seed(seed + 1) if shuffle else None),
        pin_memory=torch.cuda.is_available(),
    )


def _training_identity(seed, paths, scales, device, input_binding=None, execution="eager"):
    """Describe numerical branch settings, selected inputs and execution provenance for a checkpoint run.

    Source/runtime records explain the run; resume compatibility depends on
    scientific settings and the loaded observations/generator weights.
    """
    root = getattr(paths, "root", PROJECT_ROOT)
    sources = {
        "experiments/formal/train_gift_branches.py",
        "experiments/formal/_shared/common.py",
        "experiments/formal/_shared/gift_runtime.py",
        "experiments/formal/_shared/high_frequency.py",
        "experiments/formal/_shared/gift_generator_training.py",
        "training/checkpoints.py", *PROFILE_SOURCES, *EXECUTION_SOURCES,
        *(f"src/gift/{name}" for name in
          ("__init__.py", "model.py", "identifiability.py", "identified.py", "paths.py")),
    }
    return {
        "method": "GIFT-high", "seed": seed,
        "execution": execution,
        "provenance": {"observations_file": str(paths.standard_n64),
                       "frozen_generator_file": str(paths.low_model)},
        "sources": {relative: sha256_file(root / relative) for relative in sorted(sources)},
        "input_provenance": input_binding,
        "epochs": [DERIVATIVE_EPOCHS, SHORT_ROLLOUT_EPOCHS, LONG_ROLLOUT_STAGES],
        "scales": list(scales),
        "runtime": {"device": str(device), "h5py": str(h5py.__version__),
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_threads": os.environ.get("MKL_NUM_THREADS")},
    }


def _preflight_resume(directory, seed, paths, device, input_binding, execution="auto"):
    """Check scientific resume inputs before preparing derivative/RHS arrays."""
    from training.configuration import validate_resume
    attempt = json.loads((directory / "ATTEMPT.json").read_text(encoding="utf-8"))
    scales = attempt["identity"].get("scales", [])
    if len(scales) != 4 or not all(math.isfinite(x) and x > 0 for x in scales):
        raise ValueError("Invalid saved branch scale identity")
    execution = resolve_execution(execution, device, resume=True, checkpoint_directory=directory)
    identity = _training_identity(seed, paths, scales, device, input_binding, execution)
    validate_resume(identity, attempt["identity"])


def _train_seed(
    seed: int,
    paths: ProjectPaths,
    training: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    scales: tuple[float, float, float, float],
    device: torch.device,
    *,
    checkpoint_directory: Path | None = None,
    resume: bool = False,
    boundary_hook: Callable[[str], None] | None = None,
    input_binding: dict[str, Any] | None = None,
    execution: str = "auto",
) -> tuple[HighFrequencyBranch, dict[str, Any], list[dict[str, Any]]]:
    """Train one seed through derivative, short-rollout and long-rollout phases.

    Inputs contain state, observed Q21 derivative, observed sequence and frozen
    low-RHS arrays for training/validation. Completed epoch boundaries preserve
    optimizer, scheduler, loader RNG and selection state. Resume continues the
    next committed index and a finished phase returns its selected weights.
    """
    _set_seed(seed)
    execution = resolve_execution(execution, device, resume=resume,
                                  checkpoint_directory=checkpoint_directory)
    # The fixed training protocol instantiates the frozen generator after setting the
    # seed. Its constructor consumes random numbers before branch
    # initialization even though its released parameters are then loaded.
    # Preserve that ordering so a seed denotes the same branch initialization.
    low_rhs = load_low_generator(paths, 64, device)
    train_state, train_target, train_sequence, train_frozen = training
    valid_state, valid_target, valid_sequence, valid_frozen = validation
    state_scale, high_scale, low_rhs_scale, high_rhs_scale = scales
    model = HighFrequencyBranch(
        HighFrequencyConfig(),
        state_scale=state_scale,
        high_scale=high_scale,
        low_rhs_scale=low_rhs_scale,
        high_rhs_scale=high_rhs_scale,
        frozen_generator=low_rhs,
    ).to(device)
    derivative_train = DataLoader(
        TensorDataset(
            torch.from_numpy(train_state),
            torch.from_numpy(train_target),
            torch.from_numpy(train_frozen),
        ),
        batch_size=DERIVATIVE_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    derivative_valid = DataLoader(
        TensorDataset(
            torch.from_numpy(valid_state),
            torch.from_numpy(valid_target),
            torch.from_numpy(valid_frozen),
        ),
        batch_size=DERIVATIVE_BATCH_SIZE,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    # Journalling must not alter the optimizer, batch order, or seed order above.
    # The loader's explicit generator is distinct from global Torch RNG state.
    store = None
    saved = None
    if checkpoint_directory is not None:
        from training.checkpoints import CheckpointStore, state_digest
        identity = _training_identity(seed, paths, scales, device, input_binding, execution)
        scientific = hashlib.sha256()
        for split, arrays in (("training", training), ("validation", validation)):
            for index, value in enumerate(arrays):
                value = np.ascontiguousarray(value)
                scientific.update(f"{split}:{index}:{value.shape}:{value.dtype}".encode("ascii"))
                scientific.update(memoryview(value).cast("B"))
        identity["selected_arrays_sha256"] = scientific.hexdigest()
        identity["generator_weights_sha256"] = state_digest(low_rhs.model.state_dict())
        store = CheckpointStore(checkpoint_directory, identity, resume=resume)
        saved = store.payload
    elif resume:
        raise ValueError("Resume needs this run's checkpoint directory")

    if saved is not None and saved["phase"] == "long" and saved["next_index"] == LONG_ROLLOUT_STAGES + 1:
        model.load_state_dict(saved["selected_state"], strict=True)
        model.eval()
        return model, _selection_report(saved["selected_phase"], saved["selected_epoch"],
                                        saved["selected_metric"]), saved["history"]

    def save_boundary(phase: str, next_index: int, loader: DataLoader,
                      extra: dict[str, Any]) -> None:
        """Commit the completed phase index, optimizer/loader state and current selection history."""
        if store is None:
            return
        boundary = f"{phase}_{next_index - 1}"
        store.save(boundary, {
            "phase": phase, "next_index": next_index,
            "model": _snapshot(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if phase == "derivative" else None,
            "loader_rng": loader.generator.get_state() if loader.generator is not None else None,
            "history": history, **extra,
        })
        if boundary_hook is not None:
            boundary_hook(boundary)

    history: list[dict[str, Any]] = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, DERIVATIVE_EPOCHS, eta_min=1e-4
    )
    derivative_best = math.inf
    derivative_state: dict[str, torch.Tensor] | None = None
    derivative_epoch = -1
    first_derivative = 1
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        history = saved["history"]
        if saved["phase"] == "derivative":
            first_derivative = saved["next_index"]
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            derivative_train.generator.set_state(saved["loader_rng"])
            derivative_best = saved["derivative_best"]
            derivative_state = saved["derivative_state"]
            derivative_epoch = saved["derivative_epoch"]
            store.restore_random_state()
        else:
            first_derivative = DERIVATIVE_EPOCHS + 1
    derivative_engine = None
    if execution == "cuda-graph":
        from training.gift_acceleration import TrainingEngine
        derivative_engine = TrainingEngine(model, kind="derivative", frozen=low_rhs,
                                           scale=high_rhs_scale, backend=execution)
    for epoch in range(first_derivative, DERIVATIVE_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        rows = 0
        for state, target, frozen in derivative_train:
            state = state.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            frozen = frozen.to(device, non_blocking=True)
            if derivative_engine is None:
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_with_generator_output(state, frozen)
                loss = torch.mean(((prediction - target) / high_rhs_scale) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_value = float(loss.detach())
            else:
                loss_value = derivative_engine.step(optimizer, state, target, frozen)
            loss_sum += loss_value * len(state)
            rows += len(state)
        metrics = _validate_derivative(
            model, derivative_valid, device, high_rhs_scale
        )
        record = {
            "phase": "derivative",
            "epoch": epoch,
            "optimizer_updates": len(derivative_train),
            "training_validation_seconds": time.perf_counter() - epoch_started,
            "training_normalized_mse": loss_sum / rows,
            **{f"validation_{key}": value for key, value in metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps({"seed": seed, **record}), flush=True)
        if metrics["normalized_mse"] < derivative_best:
            derivative_best = metrics["normalized_mse"]
            derivative_state = _snapshot(model)
            derivative_epoch = epoch
        scheduler.step()
        save_boundary("derivative", epoch + 1, derivative_train, {
            "derivative_best": derivative_best, "derivative_state": derivative_state,
            "derivative_epoch": derivative_epoch,
        })
    derivative_engine = None  # Release phase-specific graphs before rollout capture.
    if saved is None or saved["phase"] == "derivative":
        if derivative_state is None:
            raise RuntimeError("derivative phase produced no selected model")
        model.load_state_dict(derivative_state, strict=True)

    rollout_valid = _rollout_loader(
        valid_sequence,
        seed,
        shuffle=False,
        batch_size=VALIDATION_ROLLOUT_BATCH_SIZE,
    )
    if saved is None or saved["phase"] == "derivative":
        baseline = _validate_rollout(model, low_rhs, rollout_valid, device)
        selected_metric = float(baseline["lead_1_full_relative_l2"])
        selected_state = _snapshot(model)
        selected_phase = "derivative"
        selected_epoch = derivative_epoch
        history.append({"phase": "derivative_checkpoint", "epoch": 0, **baseline})
    else:
        selected_metric = saved["selected_metric"]
        selected_state = saved["selected_state"]
        selected_phase = saved["selected_phase"]
        selected_epoch = saved["selected_epoch"]

    def selected_values() -> dict[str, Any]:
        """Return the selected weight snapshot, phase, epoch and metric for boundary records."""
        return {"selected_metric": selected_metric, "selected_state": selected_state,
                "selected_phase": selected_phase, "selected_epoch": selected_epoch}

    rollout_train = _rollout_loader(
        train_sequence, seed, shuffle=True, batch_size=ROLLOUT_BATCH_SIZE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    first_short = 1
    if saved is not None and saved["phase"] == "short":
        first_short = saved["next_index"]
        optimizer.load_state_dict(saved["optimizer"])
        rollout_train.generator.set_state(saved["loader_rng"])
        store.restore_random_state()
    elif saved is not None and saved["phase"] == "long":
        first_short = SHORT_ROLLOUT_EPOCHS + 1
    short_engine = (TrainingEngine(model, kind="short", frozen=low_rhs, backend=execution)
                    if execution == "cuda-graph" else None)
    for epoch in range(first_short, SHORT_ROLLOUT_EPOCHS + 1):
        epoch_started = time.perf_counter()
        loss = (short_engine.epoch(rollout_train, optimizer, device) if short_engine is not None
                else _short_epoch(model, low_rhs, rollout_train, optimizer, device))
        metric = _validate_rollout(model, low_rhs, rollout_valid, device)
        record = {
            "phase": "short_free_rollout",
            "epoch": epoch,
            "optimizer_updates": len(rollout_train),
            "training_validation_seconds": time.perf_counter() - epoch_started,
            "training_normalized_q21_mse": loss,
            **metric,
        }
        history.append(record)
        print(json.dumps({"seed": seed, **record}), flush=True)
        if metric["lead_1_full_relative_l2"] < selected_metric:
            selected_metric = float(metric["lead_1_full_relative_l2"])
            selected_state = _snapshot(model)
            selected_phase = "short_free_rollout"
            selected_epoch = epoch
        save_boundary("short", epoch + 1, rollout_train, selected_values())
    model.load_state_dict(selected_state, strict=True)

    short_engine = None
    long_engine = (TrainingEngine(model, kind="long", frozen=low_rhs, backend=execution)
                   if execution == "cuda-graph" else None)
    first_long = saved["next_index"] if saved is not None and saved["phase"] == "long" else 1
    if saved is not None and saved["phase"] == "long":
        store.restore_random_state()
    for stage in range(first_long, LONG_ROLLOUT_STAGES + 1):
        epoch_started = time.perf_counter()
        rollout_train = _rollout_loader(
            train_sequence, seed, shuffle=True, batch_size=ROLLOUT_BATCH_SIZE
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=8e-5, weight_decay=1e-6
        )
        loss = (long_engine.epoch(rollout_train, optimizer, device) if long_engine is not None
                else _long_epoch(model, low_rhs, rollout_train, optimizer, device))
        metric = _validate_rollout(model, low_rhs, rollout_valid, device)
        record = {
            "phase": f"lead_1_truncated_rollout_stage_{stage}",
            "epoch": 1,
            "optimizer_updates": len(rollout_train),
            "training_validation_seconds": time.perf_counter() - epoch_started,
            "training_normalized_q21_mse": loss,
            **metric,
        }
        history.append(record)
        print(json.dumps({"seed": seed, **record}), flush=True)
        if metric["lead_1_full_relative_l2"] < selected_metric:
            selected_metric = float(metric["lead_1_full_relative_l2"])
            selected_state = _snapshot(model)
            selected_phase = f"lead_1_truncated_rollout_stage_{stage}"
            selected_epoch = 1
        model.load_state_dict(selected_state, strict=True)
        # The original protocol resets optimizer AND loader at each long stage.
        # Resume therefore continues at the next stage, not a partial gradient.
        save_boundary("long", stage + 1, rollout_train, selected_values())

    model.eval()
    selection = _selection_report(selected_phase, selected_epoch, selected_metric)
    return model, selection, history


def branch_training_cost(history):
    """Count completed loader passes, with one optimizer update per batch.

    Include all training, not only the selected epoch. Checkpointed history
    retains the counts and timings of previously committed epochs on resume.
    """
    rows = [row for row in history if row["phase"] != "derivative_checkpoint"]
    if not rows or any(type(row.get("optimizer_updates")) is not int or row["optimizer_updates"] < 1
                       or not math.isfinite(row.get("training_validation_seconds", math.nan))
                       or row["training_validation_seconds"] < 0 for row in rows):
        raise ValueError("incomplete branch training cost records")
    return {"optimizer_updates": sum(row["optimizer_updates"] for row in rows),
            "committed_training_validation_seconds": sum(row["training_validation_seconds"] for row in rows),
            "phases": [{key: row[key] for key in
                ("phase", "epoch", "optimizer_updates", "training_validation_seconds")} for row in rows],
            "timing_scope": "training passes and their validation, including data transfers",
            "timing_excludes": "shared generator pretraining, data preparation, model selection snapshots, checkpoint IO, paused and discarded work",
            "shared_generator_pretraining_included": False}


def main() -> None:
    """Inspect the branch plan or train one seed from explicit observation and low-generator inputs.

    Only N64 training and validation fields are opened; higher-resolution and test
    fields do not affect optimizer updates or checkpoint selection.
    """
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root)
    if args.data_profile == 'gaussian':
        from training.gift_data import input_file
        from gift.paths import data_root
        paths = replace(paths, standard_n64=input_file(data_root(paths.root), 'noise_000', args.data_profile))
    if paths.root != PROJECT_ROOT:
        raise ValueError("--project-root must identify the code being executed")
    if args.low_model is not None:
        low_model = args.low_model.expanduser().resolve(strict=True)
        if paths.root / "artifacts" in low_model.parents:
            raise ValueError("Fresh branch training requires an explicit trained prerequisite outside published artifacts")
        paths = replace(paths, low_model=low_model)
    elif args.run_training:
        raise ValueError("Train the P21 generator independently first and specify --low-model; no published checkpoint is loaded implicitly")
    source_groups = _validate_training_source_metadata(paths)
    from gift.paths import data_root
    input_binding = validate_input(data_root(paths.root), "noise_000", args.data_profile, full=args.run_training)
    if args.run_training:
        from experiments.formal._shared.gift_generator_training import GeneratorTrainingConfig
        payload = torch.load(paths.low_model, map_location="cpu", weights_only=True)
        validate_low_prerequisite(payload, input_binding, asdict(GeneratorTrainingConfig()))
        del payload
    seeds = (args.seed,) if args.seed is not None else SEEDS
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_complete",
                    "execution": args.execution,
                    "method": "GIFT high-frequency branch",
                    "seeds": list(seeds),
                    "raw_dataset": file_record(paths.standard_n64, paths.root),
                    "frozen_low_frequency_model": file_record(
                        paths.low_model, paths.root
                    ),
                    "groups": source_groups,
                    "input_provenance": input_binding,
                    "low_prerequisite_freshness_verified": False,
                    "derivative_centres": [
                        DERIVATIVE_CENTRES[0],
                        DERIVATIVE_CENTRES[-1],
                        5,
                    ],
                    "derivative_epochs": DERIVATIVE_EPOCHS,
                    "short_rollout_epochs": SHORT_ROLLOUT_EPOCHS,
                    "long_single_epoch_stages": LONG_ROLLOUT_STAGES,
                    "outputs_written": 0,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.seed is None:
        raise ValueError("Train one seed per independent run: specify --seed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Apply before preparing frozen-generator RHS tensors, not just before the
    # trainable branch. Kernel policy and precision must agree throughout a run.
    configure_training_runtime(strict=False)
    suffix = f"seed_{args.seed}" if args.seed is not None else "three_seeds"
    output = Path(
        args.output
        or paths.root / "reproduced_models" / f"gift_branches_{suffix}"
    ).resolve(strict=False)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    if args.resume and not (output / "checkpoints" / "LATEST.json").is_file():
        raise FileNotFoundError("Resume requires this run's saved boundary in --output/checkpoints")
    execution = resolve_execution(args.execution, device, resume=args.resume,
                                  checkpoint_directory=output/"checkpoints")

    started = time.perf_counter()
    if args.resume:
        # Set precision/RNG policy before preparing the frozen RHS arrays.
        # _train_seed repeats initialization in the established model order,
        # then restores the saved RNG state before the next training epoch.
        _set_seed(args.seed)
        _preflight_resume(output / "checkpoints", args.seed, paths, device, input_binding, execution)
    training_raw, validation_raw = _load_raw_training_data(paths)
    low_rhs = load_low_generator(paths, 64, device)
    train_frozen = _compute_low_rhs(low_rhs, training_raw[0], device)
    valid_frozen = _compute_low_rhs(low_rhs, validation_raw[0], device)
    scales = _training_scales(training_raw[0], training_raw[1], train_frozen)
    training = (*training_raw, train_frozen)
    validation = (*validation_raw, valid_frozen)
    if not args.resume:
        output = create_output(output, "GIFT_high_frequency_branch_training")
    from training.checkpoints import state_digest
    frozen_weights = state_digest(low_rhs.model.state_dict())
    reports: dict[str, Any] = {}
    all_history: list[dict[str, Any]] = []
    for seed in seeds:
        model, selection, history = _train_seed(
            seed, paths, training, validation, scales, device,
            checkpoint_directory=output / "checkpoints", resume=args.resume,
            input_binding=input_binding,
            execution=execution,
        )
        checkpoint = output / f"gift_seed_{seed}.pt"
        cost = branch_training_cost(history)
        _write_checkpoint(
            checkpoint,
            model,
            seed=seed,
            phase=selection["phase"],
            epoch=selection["epoch"],
            selection_metric=selection["value"],
            generator_weights_sha256=frozen_weights,
            training_data=dict(input_binding, execution=execution),
            training_cost=cost,
        )
        reports[str(seed)] = {
            "selection": selection,
            "model": file_record(checkpoint, output),
            "training_cost": cost,
        }
        for row in history:
            all_history.append(
                {
                    "seed": seed,
                    "phase": row["phase"],
                    "epoch": row["epoch"],
                    "optimizer_updates": row.get("optimizer_updates", 0),
                    "training_validation_seconds": row.get("training_validation_seconds", ""),
                    "training_metric": row.get(
                        "training_normalized_mse",
                        row.get("training_normalized_q21_mse", ""),
                    ),
                    "validation_derivative_normalized_mse": row.get(
                        "validation_normalized_mse", ""
                    ),
                    "validation_derivative_q21_global_relative_l2": row.get(
                        "validation_q21_global_relative_l2", ""
                    ),
                    "validation_lead_1_full_relative_l2": row.get(
                        "lead_1_full_relative_l2", ""
                    ),
                    "validation_lead_1_q21_relative_l2": row.get(
                        "lead_1_q21_relative_l2", ""
                    ),
                }
            )
    history_path = output / "training_history.csv"
    if not history_path.exists():
        write_csv_new(history_path, all_history)
    report = {
        "schema": "gift.formal.high-frequency-branch-training.v2",
        "execution": execution,
        "status": "complete",
        "experiment": "GIFT_high_frequency_branch_training",
        "seeds": list(seeds),
        "input": {
            "raw_training_and_validation_data": file_record(
                paths.standard_n64, paths.root
            ),
            "frozen_low_frequency_model": file_record(paths.low_model, paths.root),
            "training_data": input_binding,
        },
        "continuation": {
            "same_run_resume_used": args.resume,
            "published_high_branch_loaded": False,
            "frozen_low_generator_selected_explicitly": True,
            "saved_boundary": "end_of_each_derivative_epoch_short_epoch_or_long_stage",
        },
        "data_protocol": {
            "validated_source_groups": source_groups,
            "training_ids": source_groups['training']['trajectory_ids'],
            "validation_ids": source_groups['validation']['trajectory_ids'],
            "derivative": "fourth-order centred difference, dt=0.02",
            "derivative_centres": "2, 7, ..., 497",
            "training_derivative_samples": 50 * len(DERIVATIVE_CENTRES),
            "validation_derivative_samples": 20 * len(DERIVATIVE_CENTRES),
            "rollout_anchors_per_trajectory": list(ROLLOUT_ANCHORS),
            "rollout_target_steps": list(ROLLOUT_STEPS),
            "higher_resolution_data_opened": False,
        },
        "training_protocol": {
            "model_config": HighFrequencyConfig().__dict__,
            "inputs": [
                "complete state",
                "Q21 state",
                "same-state frozen P21 generator output",
            ],
            "target": "complete Q21 observed derivative",
            "derivative_phase": {
                "epochs": DERIVATIVE_EPOCHS,
                "batch_size": DERIVATIVE_BATCH_SIZE,
                "optimizer": "AdamW(lr=1.5e-3, weight_decay=1e-6)",
                "scheduler": "CosineAnnealingLR(18, eta_min=1e-4)",
            },
            "short_rollout_phase": {
                "epochs": SHORT_ROLLOUT_EPOCHS,
                "batch_size": ROLLOUT_BATCH_SIZE,
                "optimizer": "AdamW(lr=2e-4, weight_decay=1e-6)",
            },
            "long_rollout_phase": {
                "independent_single_epoch_stages": LONG_ROLLOUT_STAGES,
                "batch_size": ROLLOUT_BATCH_SIZE,
                "optimizer_reset_each_stage": True,
                "sample_order_reset_each_stage": True,
                "optimizer": "AdamW(lr=8e-5, weight_decay=1e-6)",
                "truncated_BPTT_segment_steps": 10,
            },
            "selection": (
                "strictly lower N64 validation lead=1 full-field relative L2; "
                "earliest wins ties"
            ),
            "known_equation_terms": [],
            "test_data_used": False,
        },
        "scales": {
            "complete_state_rms": scales[0],
            "q21_state_rms": scales[1],
            "frozen_low_rhs_rms": scales[2],
            "q21_rhs_rms": scales[3],
        },
        "per_seed": reports,
        "wall_seconds": time.perf_counter() - started,
        "cuda_bitwise_determinism_enforced": False,
    }
    from training.checkpoints import write_json_export
    write_json_export(output / "report.json", report)
    if not (output / "manifest.json").exists():
        publication = [output / f"gift_seed_{seed}.pt" for seed in seeds]
        publication += [history_path, output / "report.json"]
        write_json_export(output / "manifest.json", {
            "schema": "gift.experiment-manifest.v1",
            "status": report["status"], "experiment": report["experiment"],
            "files": [file_record(path, output) for path in sorted(publication)],
        })


if __name__ == "__main__":
    main()
