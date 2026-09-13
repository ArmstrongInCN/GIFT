"""First-party experiment batching around externally supplied FNO models.

No spectral layers, network architecture or upstream training implementation is
distributed here. Frozen training statistics are lifted with GIFT's own periodic
even-grid solver utilities; no test statistics or interpolated time frames enter.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from adapters.models import load_checkpoint
from training.baseline_control import _fno3d_input, _normalizer_objects

from .common import ProjectPaths, activate_project

CONTEXT_STEPS = 46
FUTURE_STEPS = 150
M2_FUTURE_INDICES = np.asarray([24, 49, 74, 99, 124, 149], dtype=np.int64)
M3_FUTURE_INDICES = np.arange(4, 50, 5, dtype=np.int64)
FNO2D_SAFE_BATCH = {64: 10, 96: 5, 128: 2}
FNO3D_SAFE_BATCH = {64: 5, 96: 2, 128: 1}


def _terminal_contract(payload: dict, method: str) -> None:
    if payload.get("status") != "complete" or int(payload.get("terminal_epoch", -1)) != 500:
        raise ValueError(f"{method}: a completed 500-epoch terminal artifact is required")
    if payload.get("artifact_role") == "test_only" or payload.get("formal_configuration") is False:
        raise ValueError("A tiny fixture cannot be used as a formal experiment model")
    if payload.get("method") not in (method, method.lower().replace("-", "")):
        raise ValueError("Checkpoint method metadata differs")


def load_fno_models(paths: ProjectPaths, device: torch.device):
    paths.require("fno2d_model", "fno3d_model")
    models, payloads = [], []
    for name, checkpoint in (("fno2d", paths.fno2d_model), ("fno3d", paths.fno3d_model)):
        model, payload = load_checkpoint(name, checkpoint, str(device))
        _terminal_contract(payload, "FNO-2D" if name == "fno2d" else "FNO-3D")
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError("Nonfinite FNO state tensor")
        model.requires_grad_(False)
        models.append(model)
        payloads.append(payload)
    if payloads[0].get("normalization") != {"kind": "none"}:
        raise ValueError("FNO-2D must use unnormalized physical observations")
    return models[0], models[1], payloads[0], payloads[1]


def _normalizers(paths: ProjectPaths, payload: dict, grid: int, device: torch.device):
    activate_project(paths.root)
    from even_full_spectrum_ns import merge_odd_dft_to_even_native, split_even_native_dft

    source = payload["normalization"]
    if source.get("kind") != "paper_UnitGaussianNormalizer" or source.get("validation_or_test_used", False):
        raise ValueError("FNO-3D requires frozen training-only statistics")
    if grid not in (64, 96, 128):
        raise ValueError("Unsupported formal grid")
    lifted = {"eps": float(source["eps"])}
    for name in ("input_mean", "input_std", "output_mean", "output_std"):
        tensor = source[name].detach().cpu().to(torch.float32).contiguous()
        length = CONTEXT_STEPS if name.startswith("input") else FUTURE_STEPS
        if tuple(tensor.shape) != (64, 64, length):
            raise ValueError(f"Wrong shape for {name}")
        if grid != 64:
            coefficients = torch.fft.fft2(tensor.permute(2, 0, 1).contiguous(), dim=(-2, -1))
            padded = split_even_native_dft(coefficients, 3 * (grid // 2) + 1)
            native = merge_odd_dft_to_even_native(padded, grid)
            tensor = torch.fft.ifft2(native, dim=(-2, -1)).real.permute(1, 2, 0).contiguous().to(torch.float32)
        if not bool(torch.isfinite(tensor).all()) or (name.endswith("std") and not bool((tensor > 0).all())):
            raise ValueError(f"Invalid statistic {name}")
        lifted[name] = tensor
    first, second = _normalizer_objects(lifted, device)
    audit = {"source_grid": 64, "target_grid": grid,
             "spatial_method": "identity" if grid == 64 else "deterministic periodic Fourier interpolation",
             "temporal_interpolation": False, "test_statistics_used": False,
             "input_steps": CONTEXT_STEPS, "output_steps": FUTURE_STEPS}
    return first, second, audit


def _input_contract(context: np.ndarray, requested, batch_size: int, safe_batches: dict):
    if context.ndim != 4 or context.dtype != np.float32:
        raise ValueError("Expected float32 [trajectory,46,x,y] context")
    population, history, nx, ny = context.shape
    if history != 46 or nx != ny or nx not in safe_batches or population < 1:
        raise ValueError("Invalid context dimensions")
    if not 1 <= batch_size <= safe_batches[nx] or not np.isfinite(context).all():
        raise ValueError("Invalid batch size or nonfinite context")
    indices = np.asarray(requested, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or indices[0] < 0 or indices[-1] >= 150 or np.any(np.diff(indices) <= 0):
        raise ValueError("Report indices must be strictly increasing within 0..149")
    return population, nx, indices


@torch.inference_mode()
def _predict(model, context, requested, device, batch_size, *, normalizers=None):
    is3d = normalizers is not None
    population, grid, indices = _input_contract(context, requested, batch_size,
                                               FNO3D_SAFE_BATCH if is3d else FNO2D_SAFE_BATCH)
    output = np.empty((population, len(indices) + 1, grid, grid), dtype=np.float32)
    output[:, 0] = context[:, -1]
    finite_steps = np.zeros(150, dtype=np.int64)
    finite_values, calls = 0, 0
    for start in range(0, population, batch_size):
        batch = torch.as_tensor(np.ascontiguousarray(np.moveaxis(context[start:start + batch_size], 1, -1)), device=device)
        if is3d:
            input_norm, output_norm = normalizers
            raw = model(_fno3d_input(input_norm.encode(batch), FUTURE_STEPS))
            prediction = output_norm.decode(raw.reshape(len(batch), grid, grid, FUTURE_STEPS))
            calls += 1
        else:
            window = batch
            frames = []
            for _ in range(FUTURE_STEPS):
                next_frame = model(window)
                if tuple(next_frame.shape) != (len(batch), grid, grid, 1):
                    raise ValueError("FNO-2D returned a non-single-frame prediction")
                frames.append(next_frame)
                window = torch.cat((window[..., 1:], next_frame), dim=-1)
            prediction = torch.cat(frames, dim=-1)
            calls += FUTURE_STEPS
        full = np.moveaxis(prediction.cpu().numpy(), -1, 1)
        finite = np.isfinite(full)
        finite_steps += finite.reshape(len(batch), FUTURE_STEPS, -1).all(2).sum(0)
        finite_values += int(finite.sum())
        output[start:start + len(batch), 1:] = full[:, indices]
    audit = {"batch_size": batch_size, "batch_count": math.ceil(population / batch_size),
             "forward_calls": calls, "future_steps": FUTURE_STEPS,
             "finite_trajectory_count_by_future_step": finite_steps.tolist(),
             "finite_value_count": finite_values, "total_value_count": population * FUTURE_STEPS * grid * grid,
             "all_values_finite": bool(np.all(finite_steps == population))}
    audit.update({"non_recursive_calls_per_trajectory": 1, "output_steps_per_call": 150} if is3d else {"recursive_steps_per_trajectory": 150})
    return output, audit


def predict_fno2d(model, context_frames, report_future_indices, device, batch_size):
    return _predict(model, context_frames, report_future_indices, device, batch_size)


def predict_fno3d(paths, model, payload, context_frames, report_future_indices, device, batch_size):
    first, second, audit = _normalizers(paths, payload, context_frames.shape[-1], device)
    output, inference = _predict(model, context_frames, report_future_indices, device, batch_size,
                                 normalizers=(first, second))
    return output, inference, audit
