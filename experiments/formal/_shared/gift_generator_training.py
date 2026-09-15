"""Deterministic independent training for the interpretable GIFT generator.

This module is the maintained implementation of the original P21 training
protocol.  It reads only observed ``time``, ``trajectory_index`` and
``vorticity`` HDF5 datasets. Fresh runs never load model checkpoints. Explicit
resume accepts only this run's source/data/config-bound optimizer and model
state. Neither path reads cached derivatives, analytic right-hand sides,
forcing fields, physical coefficients or a PDE candidate dictionary.

The affine constant and linear Fourier tables are fitted by the current
rank-aware centered TSVD minimum-norm solver in :mod:`gift.identifiability`.
Only the low-rank quadratic factors are gradient trained.  Callers own output
directory creation; terminal files and saved boundaries are create-only. Only
the checkpoint store's small LATEST pointer is atomically updated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from gift import GIFTGenerator, fit_affine_minimum_norm, identifiability_grid
from training.checkpoints import CheckpointStore


@dataclass(frozen=True)
class GeneratorTrainingConfig:
    """Complete numerical protocol for one fresh generator."""

    training_trajectories: int = 50
    validation_trajectories: int = 20
    cutoff: int = 21
    rank: int = 8
    phase_steps: tuple[int, ...] = (6000, 6000)
    phase_learning_rates: tuple[float, ...] = (1.0e-2, 3.0e-3)
    batch_size: int = 16
    affine_refit_interval: int = 500
    validation_interval: int = 250
    rank_rcond: float | None = None
    seed: int = 2026072301
    strict_determinism: bool = True

    def validate(self) -> None:
        if len(self.phase_steps) != len(self.phase_learning_rates):
            raise ValueError("phase steps and learning rates must have equal length")
        if not self.phase_steps or min(self.phase_steps) < 1:
            raise ValueError("phase steps must be positive")
        if min(self.phase_learning_rates) <= 0.0:
            raise ValueError("phase learning rates must be positive")
        counts = (
            self.training_trajectories,
            self.validation_trajectories,
            self.cutoff,
            self.rank,
            self.batch_size,
            self.affine_refit_interval,
            self.validation_interval,
        )
        if min(counts) < 1:
            raise ValueError("training counts, cutoff and rank must be positive")
        if self.rank_rcond is not None and self.rank_rcond <= 0.0:
            raise ValueError("rank_rcond must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    display = str(resolved)
    if root is not None:
        try:
            display = resolved.relative_to(Path(root).resolve(strict=True)).as_posix()
        except ValueError:
            pass
    return {
        "path": display,
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def write_json_new(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("training history cannot be empty")
    fields = list(rows[0])
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def configure_determinism(seed: int, *, strict: bool) -> dict[str, Any]:
    """Configure all RNGs before model construction and return an audit record."""

    from training.gift_execution import configure_training_runtime
    configure_training_runtime(strict=strict)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {
        "seed": int(seed),
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_all_seeded": True,
        "torch_deterministic_algorithms": bool(strict),
        "torch_deterministic_warn_only": not bool(strict),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "data_loader_workers": 0,
    }


def fourth_order_derivative(
    state: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    if state.ndim != 3 or state.shape[0] < 5:
        raise ValueError("one trajectory must have shape [time,N,N] with time >= 5")
    derivative = (state[:-4] - 8.0 * state[1:-3] + 8.0 * state[3:-1] - state[4:]) / (
        12.0 * dt
    )
    return state[2:-2], derivative


def _load_observed_group(
    handle: h5py.File,
    group: str,
    trajectory_count: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, dict[str, Any]]:
    """Open only the three observation datasets allowed by the firewall."""

    time_values = np.asarray(handle[f"{group}/time"][:], dtype=np.float64)
    trajectory_ids = np.asarray(
        handle[f"{group}/trajectory_index"][:trajectory_count], dtype=np.int64
    )
    dataset = handle[f"{group}/vorticity"]
    if trajectory_count > dataset.shape[0]:
        raise ValueError(f"requested too many {group} trajectories")
    if len(trajectory_ids) != trajectory_count or len(np.unique(trajectory_ids)) != len(
        trajectory_ids
    ):
        raise ValueError(f"{group} trajectory identities are incomplete or duplicated")
    if len(time_values) < 5 or not np.isfinite(time_values).all():
        raise ValueError(f"{group} time support is invalid")
    increments = np.diff(time_values)
    dt = float(increments.mean())
    if dt <= 0.0 or not np.allclose(increments, dt, rtol=0.0, atol=2.0e-12):
        raise ValueError(f"{group} time support must be uniformly increasing")
    if dataset.ndim != 4 or dataset.shape[-1] != dataset.shape[-2]:
        raise ValueError(f"{group} vorticity must have shape [trajectory,time,N,N]")
    minimum_grid = 2
    if dataset.shape[-1] < minimum_grid:
        raise ValueError(f"{group} spatial grid is invalid")

    states: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    for trajectory in range(trajectory_count):
        raw = np.asarray(dataset[trajectory], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(
                f"{group} trajectory {trajectory} contains nonfinite values"
            )
        state, derivative = fourth_order_derivative(raw, dt)
        states.append(state)
        derivatives.append(derivative.astype(np.float32, copy=False))
    state_tensor = torch.from_numpy(np.concatenate(states, axis=0))
    derivative_tensor = torch.from_numpy(np.concatenate(derivatives, axis=0))
    metadata = {
        "trajectory_ids": trajectory_ids.tolist(),
        "raw_shape": [int(value) for value in dataset.shape],
        "selected_shape": [trajectory_count, *map(int, dataset.shape[1:])],
        "time_interval": [float(time_values[0]), float(time_values[-1])],
        "time_count": int(len(time_values)),
        "dt": dt,
        "derivative_centre_count": int(len(state_tensor)),
        "vorticity_source_dtype": str(dataset.dtype),
        "training_tensor_dtype": str(state_tensor.numpy().dtype),
    }
    return state_tensor, derivative_tensor, time_values, trajectory_ids, metadata


def inspect_observation_dataset(
    dataset: Path, config: GeneratorTrainingConfig
) -> dict[str, Any]:
    """Validate dataset structure without allocating complete training tensors."""

    config.validate()
    path = Path(dataset).resolve(strict=True)
    result: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for group, count in (
            ("training", config.training_trajectories),
            ("validation", config.validation_trajectories),
        ):
            required = [
                f"{group}/time",
                f"{group}/trajectory_index",
                f"{group}/vorticity",
            ]
            if any(name not in handle for name in required):
                raise KeyError(f"{group} observation datasets are incomplete")
            times = np.asarray(handle[f"{group}/time"][:], dtype=np.float64)
            ids = np.asarray(
                handle[f"{group}/trajectory_index"][:count], dtype=np.int64
            )
            field = handle[f"{group}/vorticity"]
            if len(ids) != count or len(np.unique(ids)) != count:
                raise ValueError(f"{group} trajectory identities differ")
            if field.shape[0] < count or field.shape[1] != len(times):
                raise ValueError(f"{group} vorticity shape differs from coordinates")
            if (
                field.shape[-1] != field.shape[-2]
                or field.shape[-1] < 2 * config.cutoff + 1
            ):
                raise ValueError(
                    f"{group} grid cannot represent cutoff {config.cutoff}"
                )
            increments = np.diff(times)
            if (
                len(times) < 5
                or not np.isfinite(times).all()
                or not np.all(increments > 0)
            ):
                raise ValueError(f"{group} time support is invalid")
            if not np.allclose(increments, increments[0], rtol=0.0, atol=2.0e-12):
                raise ValueError(f"{group} time support is not uniform")
            result[group] = {
                "trajectory_ids": ids.tolist(),
                "trajectory_count": count,
                "raw_shape": [int(value) for value in field.shape],
                "time_interval": [float(times[0]), float(times[-1])],
                "dt": float(increments[0]),
            }
    return result


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).flatten(1).norm(dim=1)
    denominator = target.flatten(1).norm(dim=1).clamp_min(1.0e-20)
    return numerator / denominator


def summarize(values: torch.Tensor) -> dict[str, float]:
    value = values.detach().double().cpu()
    return {
        "mean": float(value.mean()),
        "median": float(value.median()),
        "p95": float(torch.quantile(value, 0.95)),
        "maximum": float(value.max()),
    }


@torch.inference_mode()
def evaluate(
    model: GIFTGenerator,
    state: torch.Tensor,
    derivative: torch.Tensor,
    *,
    device: torch.device,
    indices: torch.Tensor | None = None,
) -> dict[str, float]:
    selected = torch.arange(len(state)) if indices is None else indices
    collected: list[torch.Tensor] = []
    for start in range(0, len(selected), 64):
        ids = selected[start : start + 64]
        prediction = model(state[ids].to(device))
        collected.append(relative_l2(prediction, derivative[ids].to(device)).cpu())
    return summarize(torch.cat(collected))


def clone_state(model: GIFTGenerator) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def configure_variable_projection_parameters(
    model: GIFTGenerator,
) -> list[torch.nn.Parameter]:
    model.linear_table.requires_grad_(False)
    model.constant_table.requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def rank_threshold_sensitivity(fit: Any, grid: Any) -> dict[str, Any]:
    active = grid.mask.detach().cpu()
    singular = fit.singular_values.detach().double().cpu()
    base_tolerance = fit.tolerance.detach().double().cpu()
    rows = []
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        rank = (singular > multiplier * base_tolerance[..., None]).sum(-1)
        rows.append(
            {
                "rcond_multiplier": multiplier,
                "rank_histogram": {
                    str(value): int(((rank == value) & active).sum())
                    for value in (0, 1, 2)
                },
            }
        )
    smaller = singular[..., 1]
    identifiable = active & (fit.numerical_rank.detach().cpu() == 2)
    margin = smaller[identifiable] / base_tolerance[identifiable]
    return {
        "rows": rows,
        "minimum_retained_singular_value_to_threshold_ratio": (
            float(margin.min()) if margin.numel() else None
        ),
    }


def _runtime_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def train_fresh_generator(
    *,
    dataset: Path,
    checkpoint: Path,
    report_path: Path,
    history_path: Path,
    condition: str,
    device_name: str,
    config: GeneratorTrainingConfig,
    project_root: Path,
    entry_script: Path,
    resume: bool = False,
    checkpoint_interval: int = 250,
    input_binding: dict[str, Any] | None = None,
    execution: str = "auto",
) -> dict[str, Any]:
    """Train from observations, optionally continuing this attempt's own state."""

    config.validate()
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if condition not in ("noise_000", "noise_001", "noise_010"):
        raise ValueError("unknown observation condition")
    data_path = Path(dataset).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=False)
    report_path = Path(report_path).resolve(strict=False)
    history_path = Path(history_path).resolve(strict=False)
    for destination in (checkpoint, report_path, history_path):
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite training output: {destination}"
            )
    if len({checkpoint.parent, report_path.parent, history_path.parent}) != 1:
        raise ValueError(
            "checkpoint, report and history must share one output directory"
        )

    device = torch.device(device_name)
    from training.gift_execution import EXECUTION_SOURCES, resolve_execution
    execution = resolve_execution(execution, device, resume=resume,
                                  checkpoint_directory=checkpoint.parent/"checkpoints")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training but CUDA is unavailable")
    deterministic = configure_determinism(config.seed, strict=config.strict_determinism)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset_sha256 = sha256_file(data_path)
    root = Path(project_root).resolve(strict=True)
    source_paths = [
        *(root / "src" / "gift" / name for name in
          ("__init__.py", "model.py", "identifiability.py", "identified.py")),
        root / "training" / "checkpoints.py",
        Path(__file__).resolve(strict=True),
        Path(entry_script).resolve(strict=True),
    ]
    if input_binding is not None:
        from training.gift_data import PROFILE_SOURCES
        source_paths.extend(root / relative for relative in PROFILE_SOURCES)
    source_paths = list(dict.fromkeys([*source_paths, *(root / name for name in EXECUTION_SOURCES)]))
    identity = {
        "role": "gift_low_generator", "condition": condition,
        "execution": execution,
        "configuration": asdict(config),
        "dataset_sha256": dataset_sha256, "dataset_bytes": data_path.stat().st_size,
        "input_provenance": input_binding,
        "sources": {path.relative_to(root).as_posix(): sha256_file(path)
                    for path in source_paths},
        "device": str(device), "h5py": str(h5py.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    store = CheckpointStore(checkpoint.parent / "checkpoints", identity, resume=resume)
    saved = store.payload
    resumed_boundary = store.boundary
    if saved is not None and saved.get("completed"):
        raise FileExistsError("this training attempt already completed")
    with h5py.File(data_path, "r") as handle:
        train_state, train_derivative, train_time, train_ids, train_meta = (
            _load_observed_group(handle, "training", config.training_trajectories)
        )
        valid_state, valid_derivative, valid_time, valid_ids, valid_meta = (
            _load_observed_group(handle, "validation", config.validation_trajectories)
        )
    if train_state.shape[-1] < 2 * config.cutoff + 1:
        raise ValueError("training grid cannot represent the requested cutoff")

    state_scale = float(train_state.square().mean().sqrt())
    if not math.isfinite(state_scale) or state_scale <= 0.0:
        raise ValueError("training state scale must be finite and positive")
    model = GIFTGenerator(
        cutoff=config.cutoff,
        rank=config.rank,
        state_scale=state_scale,
        conserve_spatial_mean=False,
    ).to(device)
    if saved is None:
        initial_fit = fit_affine_minimum_norm(
            model, train_state, train_derivative, device=device,
            rcond=config.rank_rcond, subtract_quadratic=False,
        )
        grid = identifiability_grid(model, int(train_state.shape[-1]), device=device)
        initial_identifiability = initial_fit.summary(grid)
    else:
        grid = identifiability_grid(model, int(train_state.shape[-1]), device=device)
        initial_identifiability = saved["initial_identifiability"]
        if saved["state_scale"] != state_scale:
            raise ValueError("resumed observation scale differs")
    nonlinear_parameters = configure_variable_projection_parameters(model)
    scale_ids = torch.linspace(
        0, len(train_derivative) - 1, min(256, len(train_derivative))
    ).long()
    target_scale = float(train_derivative[scale_ids].square().mean().clamp_min(1.0e-20))
    validation_sample_ids = torch.linspace(
        0, len(valid_state) - 1, min(128, len(valid_state))
    ).long()

    history: list[dict[str, Any]] = [] if saved is None else saved["history"]
    phase_records: list[dict[str, Any]] = [] if saved is None else saved["phase_records"]
    resume_phase = 1 if saved is None else saved["phase"]
    if not 1 <= resume_phase <= len(config.phase_steps):
        raise ValueError("saved phase is outside the configured budget")
    previous_elapsed = 0.0 if saved is None else saved["elapsed_seconds"]
    if saved is not None and saved["target_scale"] != target_scale:
        raise ValueError("resumed target scale differs")
    started = time.perf_counter()
    engine = None
    if execution == "cuda-graph":
        from training.gift_acceleration import TrainingEngine
        engine = TrainingEngine(model, kind="generator", scale=target_scale, backend=execution)
    for phase, (steps, learning_rate) in enumerate(
        zip(config.phase_steps, config.phase_learning_rates, strict=True), start=1
    ):
        if phase < resume_phase:
            continue
        random.seed(config.seed)
        np.random.seed(config.seed % (2**32 - 1))
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        optimizer = torch.optim.AdamW(
            nonlinear_parameters, lr=learning_rate, weight_decay=1.0e-8
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=learning_rate * 0.02
        )
        best_validation = math.inf
        best_step = 0
        best_state = clone_state(model)
        completed_step = 0
        if saved is not None and phase == resume_phase:
            completed_step = saved["step"]
            if not 0 <= completed_step <= steps or len(phase_records) != phase - 1:
                raise ValueError("saved step or preceding phases are inconsistent")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            best_validation, best_step = saved["best_validation"], saved["best_step"]
            best_state = saved["best_state"]
            store.restore_random_state()

        def save_boundary(step: int) -> None:
            store.save(f"phase_{phase}_step_{step}", {
                "model": clone_state(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "phase": phase, "step": step,
                "best_state": best_state, "best_validation": best_validation,
                "best_step": best_step, "history": history,
                "phase_records": phase_records,
                "initial_identifiability": initial_identifiability,
                "state_scale": state_scale, "target_scale": target_scale,
                "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                "sampling": "global_torch_cpu_randint_no_loader",
                "loader_generator_state": None, "completed": False,
            })

        if saved is None or phase != resume_phase:
            save_boundary(0)
        for step in range(completed_step + 1, steps + 1):
            ids = torch.randint(len(train_state), (config.batch_size,))
            batch_state = train_state[ids].to(device)
            batch_target = train_derivative[ids].to(device)
            if engine is None:
                prediction = model(batch_state)
                data_loss = (prediction - batch_target).square().mean() / target_scale
                optimizer.zero_grad(set_to_none=True)
                data_loss.backward()
                torch.nn.utils.clip_grad_norm_(nonlinear_parameters, 5.0)
                optimizer.step()
                data_loss_value = data_loss.detach()
            else:
                data_loss_value = engine.step(optimizer, batch_state, batch_target)
            scheduler.step()
            if step % config.affine_refit_interval == 0:
                fit_affine_minimum_norm(
                    model,
                    train_state,
                    train_derivative,
                    device=device,
                    rcond=config.rank_rcond,
                    subtract_quadratic=True,
                )
            if step == 1 or step % config.validation_interval == 0 or step == steps:
                validation = evaluate(
                    model,
                    valid_state,
                    valid_derivative,
                    device=device,
                    indices=validation_sample_ids,
                )["mean"]
                if not math.isfinite(validation):
                    raise FloatingPointError("validation metric became nonfinite")
                if validation < best_validation:
                    best_validation = validation
                    best_step = step
                    best_state = clone_state(model)
                row = {
                    "condition": condition,
                    "phase": phase,
                    "step": step,
                    "normalized_data_loss": float(data_loss_value),
                    "validation_relative_l2": validation,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                }
                history.append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
            if step % checkpoint_interval == 0 or step == steps:
                save_boundary(step)
        model.load_state_dict(best_state, strict=True)
        phase_records.append(
            {
                "phase": phase,
                "completed_steps": steps,
                "best_step": best_step,
                "best_validation_relative_l2": best_validation,
            }
        )

    final_fit = fit_affine_minimum_norm(
        model,
        train_state,
        train_derivative,
        device=device,
        rcond=config.rank_rcond,
        subtract_quadratic=True,
    )
    identifiability = final_fit.summary(grid)
    threshold_sensitivity = rank_threshold_sensitivity(final_fit, grid)
    training_metrics = evaluate(model, train_state, train_derivative, device=device)
    validation_metrics = evaluate(model, valid_state, valid_derivative, device=device)
    elapsed = previous_elapsed + time.perf_counter() - started
    training_cost = {
        "optimizer_updates": sum(item["completed_steps"] for item in phase_records),
        "phase_updates": [item["completed_steps"] for item in phase_records],
        "training_and_final_evaluation_seconds": elapsed,
        "timing_scope": "all training phases, affine refits, validation, final evaluation and intermediate checkpoint IO",
        "timing_excludes": "initial data preparation, initial affine fit, paused and discarded uncommitted work, terminal export",
        "budget_unit": "random-minibatch parameter updates, not epochs",
    }
    configuration = model.configuration()
    artifact = {
        "format_version": 3,
        "execution": execution,
        "condition": condition,
        "model_configuration": configuration,
        "model_state_dict": clone_state(model),
        "identifiability": identifiability,
        "rank_threshold_sensitivity": threshold_sensitivity,
        "identifiability_diagnostics": {
            "singular_values": final_fit.singular_values.detach().cpu(),
            "tolerance": final_fit.tolerance.detach().cpu(),
            "numerical_rank": final_fit.numerical_rank.detach().cpu(),
            "intercept_retained": final_fit.intercept_retained.detach().cpu(),
            "slope_retained": final_fit.slope_retained.detach().cpu(),
            "active_mask": grid.mask.detach().cpu(),
            "fourier_normalization": "torch.fft.fft2(state)/N^2",
            "centered_parameterization": "r=b+A*(x-mean(x))",
            "observation_dtype": "float32",
            "rcond": float(final_fit.rcond),
        },
        "training_data": {
            "file": data_path.name,
            "path": str(data_path),
            "sha256": dataset_sha256,
            "provenance": input_binding,
            "group": "training",
            "trajectory_ids": train_ids.tolist(),
            "raw_time_interval": [float(train_time[0]), float(train_time[-1])],
            "derivative_centre_count": int(len(train_state)),
        },
        "training_configuration": {
            **asdict(config),
            "phase_steps": list(config.phase_steps),
            "phase_learning_rates": list(config.phase_learning_rates),
            "raw_time_interval": [float(train_time[0]), float(train_time[-1])],
            "conserve_spatial_mean": False,
            "autoregressive_loss": False,
        },
        "artifact_role": "trained_generator",
        "training_cost": training_cost,
        "identification_method": "identifiability-aware-centered-TSVD-minimum-norm",
        "fresh_training": {
            "pretrained_model_loaded": False,
            "resume_checkpoint_loaded": resume,
            "optimizer_state_loaded": resume,
            "checkpoint_inputs": [resumed_boundary] if resume else [],
            "same_attempt_continuation": resume,
        },
    }
    with checkpoint.open("xb") as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())
    write_csv_new(history_path, history)

    report = {
        "schema": "gift.generator.independent-training.v1",
        "execution": execution,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "mode": "own_attempt_resume" if resume else "fresh_from_observations",
        "formal_protocol_defaults_used": config == GeneratorTrainingConfig(),
        "freshness_contract": {
            "pretrained_model_loaded": False,
            "resume_checkpoint_loaded": resume,
            "optimizer_state_loaded": resume,
            "checkpoint_inputs": [resumed_boundary] if resume else [],
            "same_attempt_continuation": resume,
            "cached_derivatives_loaded": False,
            "analytic_rhs_calls": 0,
            "known_pde_basis": False,
            "pde_dictionary": False,
        },
        "information_firewall": {
            "opened_hdf5_datasets": [
                "training/time",
                "training/trajectory_index",
                "training/vorticity",
                "validation/time",
                "validation/trajectory_index",
                "validation/vorticity",
            ],
            "forcing_or_physical_metadata_opened": False,
            "test_group_opened": False,
        },
        "data": {
            "source": file_record(data_path, root),
            "training": train_meta,
            "validation": valid_meta,
            "validation_trajectory_ids": valid_ids.tolist(),
            "derivative": "fourth-order centered finite difference",
            "derivative_centre_interval": [
                float(train_time[2]),
                float(train_time[-3]),
            ],
        },
        "model_configuration": configuration,
        "real_parameter_count": sum(p.numel() for p in model.parameters()),
        "optimization": {
            **asdict(config),
            "phase_steps": list(config.phase_steps),
            "phase_learning_rates": list(config.phase_learning_rates),
            "phase_records": phase_records,
            "state_scale": state_scale,
            "target_mean_square_scale": target_scale,
        },
        "determinism": deterministic,
        "identifiability_initial": initial_identifiability,
        "identifiability_final": identifiability,
        "rank_threshold_sensitivity": threshold_sensitivity,
        "metrics": {
            "training_derivative_relative_l2": training_metrics,
            "independent_validation_derivative_relative_l2": validation_metrics,
        },
        "runtime": {
            **_runtime_record(device),
            "training_and_final_evaluation_seconds": elapsed,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
        "sources": [file_record(path, root) for path in source_paths],
        "training_cost": training_cost,
        "outputs": {
            "checkpoint": file_record(checkpoint, checkpoint.parent),
            "history": file_record(history_path, history_path.parent),
        },
        "claims_boundary": {
            "minimum_norm_values_are_gauges_not_recovered_physical_coefficients": True,
            "bitwise_replay_requires_same_recorded_software_and_hardware": True,
            "scientific_reproduction_uses_frozen_protocol_and_validation_metrics": True,
        },
    }
    write_json_new(report_path, report)
    store.save("complete", {"completed": True, "model": clone_state(model),
                            "phase_records": phase_records, "history": history})
    return report
