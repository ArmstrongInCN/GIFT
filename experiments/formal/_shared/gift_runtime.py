"""Project-local GIFT inference, dual-track RK4, and local correction runtime."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .common import ProjectPaths, activate_project, sha256_file
from .high_frequency import HighFrequencyBranch, load_high_frequency_model


@dataclass(frozen=True)
class LocalCorrectionPolicy:
    """Fixed local outlier policy for one spectral component."""

    score_threshold: float = 2.1
    amplitude_threshold: float = 22.0
    window_radius: int = 2
    selected_cutoff: int = 21
    maximum_flagged_fraction: float = 40.0 / 4096.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.score_threshold) or self.score_threshold <= 0.0:
            raise ValueError("score_threshold must be finite and positive")
        if (
            not math.isfinite(self.amplitude_threshold)
            or self.amplitude_threshold <= 0.0
        ):
            raise ValueError("amplitude_threshold must be finite and positive")
        if self.window_radius != 2:
            raise ValueError("the formal policy requires a periodic 5 by 5 window")
        if self.selected_cutoff != 21:
            raise ValueError("the formal policy requires selected cutoff 21")
        if not 0.0 < self.maximum_flagged_fraction < 1.0:
            raise ValueError("maximum_flagged_fraction must lie between zero and one")

    def maximum_flagged_points(self, grid: int) -> int:
        if grid < 2 * self.selected_cutoff + 1:
            raise ValueError("grid is too small to represent the selected band")
        return max(1, math.floor(self.maximum_flagged_fraction * grid * grid))


DEFAULT_P21_CORRECTION_POLICY = LocalCorrectionPolicy()
DEFAULT_Q21_CORRECTION_POLICY = LocalCorrectionPolicy(amplitude_threshold=12.0)
DEFAULT_CORRECTION_POLICY = DEFAULT_P21_CORRECTION_POLICY


@dataclass(frozen=True)
class CorrectionResult:
    state: torch.Tensor
    trigger_count_by_trajectory: torch.Tensor
    gate_failed_by_trajectory: torch.Tensor
    report: dict[str, Any]

    @property
    def trigger_count(self) -> torch.Tensor:
        return self.trigger_count_by_trajectory

    @property
    def gate_failed(self) -> torch.Tensor:
        return self.gate_failed_by_trajectory


def square_selected_mask(
    grid: int, *, cutoff: int, device: torch.device
) -> torch.Tensor:
    modes = torch.fft.fftfreq(grid, d=1.0 / grid, device=device)
    ky, kx = torch.meshgrid(modes, modes, indexing="ij")
    return (kx.abs() <= cutoff) & (ky.abs() <= cutoff)


def _periodic_neighbor_median(state: torch.Tensor, radius: int) -> torch.Tensor:
    neighbors = torch.stack(
        [
            torch.roll(state, shifts=(dy, dx), dims=(-2, -1))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if (dy, dx) != (0, 0)
        ],
        dim=0,
    )
    count = neighbors.shape[0]
    lower = torch.kthvalue(neighbors, count // 2, dim=0).values
    upper = torch.kthvalue(neighbors, count // 2 + 1, dim=0).values
    return 0.5 * (lower + upper)


@torch.inference_mode()
def apply_local_correction(
    state: torch.Tensor,
    policy: LocalCorrectionPolicy | None = None,
    *,
    selected_mask: torch.Tensor | None = None,
    selected_scope: str = "P21",
) -> CorrectionResult:
    """Apply one band-preserving post-step correction without modifying either RHS."""

    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch tensor")
    if state.ndim != 3 or state.shape[-1] != state.shape[-2]:
        raise ValueError("state must have shape [trajectory,N,N]")
    if state.is_complex() or not state.is_floating_point():
        raise TypeError("state must use a real floating dtype")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("post-step correction requires finite input trajectories")
    if selected_scope not in ("P21", "Q21"):
        raise ValueError("selected_scope must be P21 or Q21")
    if policy is None:
        policy = (
            DEFAULT_P21_CORRECTION_POLICY
            if selected_scope == "P21"
            else DEFAULT_Q21_CORRECTION_POLICY
        )
    grid = int(state.shape[-1])
    cap = policy.maximum_flagged_points(grid)
    if selected_mask is None:
        p21_mask = square_selected_mask(
            grid, cutoff=policy.selected_cutoff, device=state.device
        )
        selected_mask = p21_mask if selected_scope == "P21" else ~p21_mask
    if (
        selected_mask.shape != (grid, grid)
        or selected_mask.device != state.device
        or selected_mask.dtype != torch.bool
    ):
        raise ValueError("selected_mask must be a boolean N by N tensor on device")

    state_hat = torch.fft.fft2(state)
    selected = torch.fft.ifft2(state_hat * selected_mask).real
    centered = selected - selected.mean(dim=(-2, -1), keepdim=True)
    local_median = _periodic_neighbor_median(centered, policy.window_radius)
    residual = centered - local_median
    rms = centered.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    rms = rms.clamp_min(torch.finfo(state.dtype).tiny)
    limit = policy.score_threshold * rms
    trigger = (residual.abs() > limit) & (centered.abs() > policy.amplitude_threshold)
    counts = trigger.flatten(1).sum(1)
    gate_failed = counts > cap

    clipped = torch.clamp(residual, min=-limit, max=limit)
    raw_delta = torch.where(trigger, clipped - residual, torch.zeros_like(residual))
    raw_delta = torch.where(
        gate_failed[:, None, None], torch.zeros_like(raw_delta), raw_delta
    )
    delta_hat = torch.fft.fft2(raw_delta) * selected_mask
    delta_hat[..., 0, 0] = 0.0
    physical_delta = torch.fft.ifft2(delta_hat).real
    changed = (counts > 0) & ~gate_failed
    corrected = torch.where(changed[:, None, None], state + physical_delta, state)

    actual_delta = corrected - state
    actual_delta_hat = torch.fft.fft2(actual_delta)
    outside_selected = ~selected_mask
    outside_selected_max = (
        float(actual_delta_hat[..., outside_selected].abs().max().item())
        if bool(outside_selected.any())
        else 0.0
    )
    zero_mode_max = float(actual_delta_hat[..., 0, 0].abs().max().item())
    state_l2 = state.square().sum(dim=(-2, -1)).sqrt()
    delta_l2 = actual_delta.square().sum(dim=(-2, -1)).sqrt()
    denominator = state_l2.clamp_min(torch.finfo(state.dtype).tiny)
    return CorrectionResult(
        state=corrected,
        trigger_count_by_trajectory=counts,
        gate_failed_by_trajectory=gate_failed,
        report={
            "grid": grid,
            "selected_scope": selected_scope,
            "score_threshold": policy.score_threshold,
            "amplitude_threshold": policy.amplitude_threshold,
            "periodic_window": 2 * policy.window_radius + 1,
            "neighbor_count": (2 * policy.window_radius + 1) ** 2 - 1,
            "selected_cutoff": policy.selected_cutoff,
            "maximum_flagged_points": cap,
            "trigger_count": int(counts.sum().item()),
            "triggered_trajectory_count": int((counts > 0).sum().item()),
            "trigger_count_by_trajectory": counts.detach().cpu().tolist(),
            "gate_failed_trajectory_count": int(gate_failed.sum().item()),
            "gate_failed_by_trajectory": gate_failed.detach().cpu().tolist(),
            "maximum_local_score": float((residual.abs() / rms).max().item()),
            "maximum_centered_selected_amplitude": float(
                centered.abs().max().item()
            ),
            "maximum_raw_delta": float(raw_delta.abs().max().item()),
            "constructed_delta_zero_mode_absolute": float(
                delta_hat[..., 0, 0].abs().max().item()
            ),
            "constructed_delta_tail_coefficient_absolute": 0.0,
            "constructed_delta_outside_selected_mask_absolute": 0.0,
            "maximum_actual_zero_mode_absolute": zero_mode_max,
            "maximum_actual_tail_coefficient_absolute": outside_selected_max,
            "maximum_actual_outside_selected_mask_absolute": outside_selected_max,
            "delta_l2_by_trajectory": delta_l2.detach().cpu().tolist(),
            "delta_relative_l2_by_trajectory": (delta_l2 / denominator)
            .detach()
            .cpu()
            .tolist(),
        },
    )


@dataclass
class RolloutResult:
    prediction: np.ndarray
    finite_by_time: np.ndarray
    failure_step: np.ndarray
    failure_reason: tuple[str | None, ...]
    correction_trigger_count: np.ndarray
    correction_step_count: np.ndarray
    correction_triggered_trajectory_ids: list[int]
    correction: dict[str, Any]
    runtime: dict[str, Any]
    branch_audit: dict[str, Any]
    method_display_name: str = "GIFT"


def load_low_generator(
    paths: ProjectPaths, grid: int, device: torch.device
) -> Callable[[torch.Tensor], torch.Tensor]:
    paths.require("low_model")
    activate_project(paths.root)
    from gift import FixedBandwidthGridGenerator, load_generator

    model = load_generator(paths.low_model, device=device).eval()
    model.requires_grad_(False)
    return FixedBandwidthGridGenerator(model, grid, reference_grid=64)


def load_gift_models(
    paths: ProjectPaths, branch_path: Path, grid: int, device: torch.device
) -> tuple[Callable[[torch.Tensor], torch.Tensor], HighFrequencyBranch, dict[str, Any]]:
    low = load_low_generator(paths, grid, device)
    branch, payload = load_high_frequency_model(branch_path, device, low)
    expected_low_hash = str(payload["frozen_generator_sha256"]).upper()
    if expected_low_hash != sha256_file(paths.low_model):
        raise ValueError("high-frequency model was trained with a different P21 generator")
    return low, branch, payload


def _validate_rhs(
    rhs: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    value = rhs(state)
    if not isinstance(value, torch.Tensor) or value.shape != state.shape:
        raise ValueError(f"{label} RHS must return a tensor with the input shape")
    if value.device != state.device or value.dtype != state.dtype:
        raise ValueError(f"{label} RHS output device or dtype differs from its input")
    return value


def _project(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(value) * mask).real


def _exact_count(numerator: float, denominator: float, label: str) -> int:
    value = numerator / denominator
    rounded = int(round(value))
    if rounded < 1 or not math.isclose(
        value, rounded, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError(f"{label} must be a positive integer")
    return rounded


def _component_correction_report(
    *,
    enabled: bool,
    scope: str,
    policy: LocalCorrectionPolicy,
    grid: int,
    ids: np.ndarray,
    trigger_counts: np.ndarray,
    gate_failures: np.ndarray,
    audit_maxima: dict[str, float],
) -> dict[str, Any]:
    triggered = np.any(trigger_counts > 0, axis=0)
    gated = np.any(gate_failures, axis=0)
    return {
        "enabled": enabled,
        "selected_scope": scope,
        "score_threshold": policy.score_threshold,
        "amplitude_threshold": policy.amplitude_threshold,
        "periodic_window": 2 * policy.window_radius + 1,
        "neighbor_count": (2 * policy.window_radius + 1) ** 2 - 1,
        "selected_cutoff": policy.selected_cutoff,
        "maximum_flagged_fraction": policy.maximum_flagged_fraction,
        "maximum_flagged_points": policy.maximum_flagged_points(grid),
        "trigger_pixel_events": int(trigger_counts.sum()),
        "triggered_step_count": int(np.any(trigger_counts > 0, axis=1).sum()),
        "triggered_trajectory_count": int(triggered.sum()),
        "triggered_trajectory_ids": ids[triggered].astype(int).tolist(),
        "trigger_count_by_step_trajectory": trigger_counts.tolist(),
        "gate_failed_trajectory_count": int(gated.sum()),
        "gate_failed_trajectory_ids": ids[gated].astype(int).tolist(),
        "gate_failed_by_step_trajectory": gate_failures.tolist(),
        "correction_delta_zero_mean": True,
        "correction_delta_selected_component_only": True,
        **audit_maxima,
    }


def _correction_report(
    *,
    correction: bool,
    p21_policy: LocalCorrectionPolicy,
    q21_policy: LocalCorrectionPolicy,
    grid: int,
    ids: np.ndarray,
    p21_trigger_counts: np.ndarray,
    q21_trigger_counts: np.ndarray,
    p21_gate_failures: np.ndarray,
    q21_gate_failures: np.ndarray,
    p21_audit_maxima: dict[str, float],
    q21_audit_maxima: dict[str, float],
) -> dict[str, Any]:
    trigger_counts = p21_trigger_counts + q21_trigger_counts
    gate_failures = p21_gate_failures | q21_gate_failures
    triggered = np.any(trigger_counts > 0, axis=0)
    gated = np.any(gate_failures, axis=0)
    p21_report = _component_correction_report(
        enabled=correction,
        scope="P21",
        policy=p21_policy,
        grid=grid,
        ids=ids,
        trigger_counts=p21_trigger_counts,
        gate_failures=p21_gate_failures,
        audit_maxima=p21_audit_maxima,
    )
    q21_report = _component_correction_report(
        enabled=correction,
        scope="Q21",
        policy=q21_policy,
        grid=grid,
        ids=ids,
        trigger_counts=q21_trigger_counts,
        gate_failures=q21_gate_failures,
        audit_maxima=q21_audit_maxima,
    )
    audit_keys = p21_audit_maxima.keys() | q21_audit_maxima.keys()
    combined_audit = {
        key: max(p21_audit_maxima.get(key, 0.0), q21_audit_maxima.get(key, 0.0))
        for key in audit_keys
    }
    return {
        "enabled": correction,
        "applied_after_complete_rk4_step": correction,
        "independent_component_detection_and_gates": correction,
        "components": {"P21": p21_report, "Q21": q21_report},
        "trigger_pixel_events": int(trigger_counts.sum()),
        "triggered_step_count": int(np.any(trigger_counts > 0, axis=1).sum()),
        "triggered_trajectory_count": int(triggered.sum()),
        "triggered_trajectory_ids": ids[triggered].astype(int).tolist(),
        "trigger_count_by_step_trajectory": trigger_counts.tolist(),
        "gate_failed_trajectory_count": int(gated.sum()),
        "gate_failed_trajectory_ids": ids[gated].astype(int).tolist(),
        "gate_failed_by_step_trajectory": gate_failures.tolist(),
        "correction_delta_zero_mean": True,
        "correction_delta_component_preserving": True,
        **combined_audit,
    }


@torch.inference_mode()
def rollout_gift(
    *,
    low: Callable[[torch.Tensor], torch.Tensor],
    branch: HighFrequencyBranch | None,
    initial_state: np.ndarray,
    trajectory_ids: np.ndarray,
    duration: float,
    save_interval: float,
    device: torch.device,
    batch_size: int,
    step: float = 0.02,
    correction: bool = True,
    correction_policy: LocalCorrectionPolicy = DEFAULT_P21_CORRECTION_POLICY,
    q21_correction_policy: LocalCorrectionPolicy = DEFAULT_Q21_CORRECTION_POLICY,
) -> RolloutResult:
    """Run GIFT recursion with trajectory-local failure isolation.

    Every RK4 stage is represented by a P21 track and a complete Q21 track.
    Their sum is supplied once to the frozen generator; the resulting P21
    right-hand side is reused as the conditioning input of the high-frequency
    branch. The learned right-hand side advances all grid-representable Q21
    modes and has no model-fixed upper cutoff. After each complete RK4 step,
    recursive local correction is applied independently to the P21 and Q21
    tracks, with each correction projected back to its own component.
    """

    initial = np.asarray(initial_state)
    if initial.ndim != 3 or initial.shape[-1] != initial.shape[-2]:
        raise ValueError("initial_state must have shape [trajectory,N,N]")
    if initial.shape[-1] < 43:
        raise ValueError("initial_state grid is too small for selected cutoff 21")
    if not np.isfinite(initial).all():
        raise ValueError("initial_state must be finite")
    initial = np.asarray(initial, dtype=np.float32)
    ids = np.asarray(trajectory_ids, dtype=np.int64)
    if ids.shape != (len(initial),) or len(np.unique(ids)) != len(initial):
        raise ValueError("trajectory_ids must be unique and match initial_state")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested accelerator is unavailable")
    for name, value in (
        ("step", step),
        ("duration", duration),
        ("save_interval", save_interval),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    steps = _exact_count(duration, step, "duration/step")
    save_every = _exact_count(save_interval, step, "save_interval/step")
    saved_intervals = _exact_count(duration, save_interval, "duration/save_interval")
    saved_leads = np.arange(saved_intervals + 1, dtype=np.float64) * save_interval

    count, grid, _ = initial.shape
    prediction = np.full(
        (count, len(saved_leads), grid, grid), np.nan, dtype=np.float32
    )
    prediction[:, 0] = initial
    failure_step = np.full(count, -1, dtype=np.int64)
    failure_reason: list[str | None] = [None] * count
    p21_trigger_counts = np.zeros((steps, count), dtype=np.int64)
    q21_trigger_counts = np.zeros((steps, count), dtype=np.int64)
    p21_gate_failures = np.zeros((steps, count), dtype=bool)
    q21_gate_failures = np.zeros((steps, count), dtype=bool)
    low_calls = np.zeros(count, dtype=np.int64)
    high_calls = np.zeros(count, dtype=np.int64)
    audit_keys = (
        "maximum_local_score",
        "maximum_centered_selected_amplitude",
        "maximum_raw_delta",
        "constructed_delta_zero_mode_absolute",
        "constructed_delta_tail_coefficient_absolute",
        "constructed_delta_outside_selected_mask_absolute",
        "maximum_actual_zero_mode_absolute",
        "maximum_actual_tail_coefficient_absolute",
        "maximum_actual_outside_selected_mask_absolute",
    )
    p21_audit_maxima = {key: 0.0 for key in audit_keys}
    q21_audit_maxima = {key: 0.0 for key in audit_keys}

    modes = torch.fft.fftfreq(grid, d=1.0 / grid, device=resolved_device)
    ky, kx = torch.meshgrid(modes, modes, indexing="ij")
    low_mask = (kx.abs() <= 21) & (ky.abs() <= 21)
    q21_mask = ~low_mask
    p21_correction_mask = low_mask if correction else None
    q21_correction_mask = q21_mask if correction else None
    if branch is not None:
        config = getattr(branch, "config", None)
        if config is None or getattr(config, "cutoff", None) != 21:
            raise ValueError("enabled high-frequency branch must preserve P21")

    def low_project(value: torch.Tensor) -> torch.Tensor:
        return _project(value, low_mask)

    def q21_project(value: torch.Tensor) -> torch.Tensor:
        return _project(value, q21_mask)

    def compose(formal_value: torch.Tensor, high_value: torch.Tensor) -> torch.Tensor:
        return low_project(formal_value) + q21_project(high_value)

    maximum_low_track_q21 = 0.0
    maximum_high_track_p21 = 0.0
    maximum_raw_branch_rhs_p21 = 0.0
    maximum_applied_branch_rhs_p21 = 0.0

    def coupled_rhs(
        low_value: torch.Tensor, high_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal maximum_raw_branch_rhs_p21
        nonlocal maximum_applied_branch_rhs_p21
        complete = compose(low_value, high_value)
        frozen = low_project(_validate_rhs(low, complete, label="low"))
        if branch is None:
            return frozen, torch.zeros_like(high_value)
        raw = branch.forward_with_generator_output(complete, frozen)
        if (
            not isinstance(raw, torch.Tensor)
            or raw.shape != complete.shape
            or raw.device != complete.device
            or raw.dtype != complete.dtype
        ):
            raise ValueError("high RHS must return a tensor with the input shape")
        projected = q21_project(raw)
        maximum_raw_branch_rhs_p21 = max(
            maximum_raw_branch_rhs_p21, float(low_project(raw).abs().max())
        )
        maximum_applied_branch_rhs_p21 = max(
            maximum_applied_branch_rhs_p21,
            float(low_project(projected).abs().max()),
        )
        return frozen, projected

    started = time.perf_counter()
    for begin in range(0, count, batch_size):
        end = min(begin + batch_size, count)
        anchor = torch.as_tensor(
            initial[begin:end], dtype=torch.float32, device=resolved_device
        ).clone()
        formal_state = low_project(anchor)
        high_state = q21_project(anchor)
        active = torch.ones(end - begin, dtype=torch.bool, device=resolved_device)

        for step_index in range(steps):
            active_rows = torch.nonzero(active, as_tuple=False).flatten()
            if len(active_rows):
                formal_current = formal_state[active_rows]
                high_current = high_state[active_rows]
                low1, high1 = coupled_rhs(formal_current, high_current)

                formal2 = formal_current + 0.5 * step * low1
                high2_state = high_current + 0.5 * step * high1
                low2, high2 = coupled_rhs(formal2, high2_state)

                formal3 = formal_current + 0.5 * step * low2
                high3_state = high_current + 0.5 * step * high2
                low3, high3 = coupled_rhs(formal3, high3_state)

                formal4 = formal_current + step * low3
                high4_state = high_current + step * high3
                low4, high4 = coupled_rhs(formal4, high4_state)

                coefficient = step / 6.0
                formal_updated = low_project(
                    formal_current
                    + coefficient * (low1 + 2.0 * low2 + 2.0 * low3 + low4)
                )
                high_updated = q21_project(
                    high_current
                    + coefficient * (high1 + 2.0 * high2 + 2.0 * high3 + high4)
                )
                combined_updated = compose(formal_updated, high_updated)
                global_active = begin + active_rows.detach().cpu().numpy()
                low_calls[global_active] += 4
                if branch is not None:
                    high_calls[global_active] += 4

                finite = torch.isfinite(combined_updated).flatten(1).all(1)
                nonfinite_rows = active_rows[~finite]
                if len(nonfinite_rows):
                    for local in nonfinite_rows.detach().cpu().tolist():
                        global_index = begin + int(local)
                        failure_step[global_index] = step_index + 1
                        failure_reason[global_index] = "rk4_nonfinite"
                    active[nonfinite_rows] = False
                    formal_state[nonfinite_rows] = 0.0
                    high_state[nonfinite_rows] = 0.0

                finite_rows = active_rows[finite]
                finite_formal = formal_updated[finite]
                finite_high = high_updated[finite]
                if len(finite_rows) and correction:
                    corrected_p21 = apply_local_correction(
                        finite_formal,
                        correction_policy,
                        selected_mask=p21_correction_mask,
                        selected_scope="P21",
                    )
                    corrected_q21 = apply_local_correction(
                        finite_high,
                        q21_correction_policy,
                        selected_mask=q21_correction_mask,
                        selected_scope="Q21",
                    )
                    finite_global = begin + finite_rows.detach().cpu().numpy()
                    p21_counts = (
                        corrected_p21.trigger_count_by_trajectory.detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )
                    q21_counts = (
                        corrected_q21.trigger_count_by_trajectory.detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )
                    p21_gates = (
                        corrected_p21.gate_failed_by_trajectory.detach()
                        .cpu()
                        .numpy()
                    )
                    q21_gates = (
                        corrected_q21.gate_failed_by_trajectory.detach()
                        .cpu()
                        .numpy()
                    )
                    p21_trigger_counts[step_index, finite_global] = p21_counts
                    q21_trigger_counts[step_index, finite_global] = q21_counts
                    p21_gate_failures[step_index, finite_global] = p21_gates
                    q21_gate_failures[step_index, finite_global] = q21_gates
                    for key in p21_audit_maxima:
                        p21_audit_maxima[key] = max(
                            p21_audit_maxima[key],
                            float(corrected_p21.report[key]),
                        )
                        q21_audit_maxima[key] = max(
                            q21_audit_maxima[key],
                            float(corrected_q21.report[key]),
                        )

                    combined_gates = (
                        corrected_p21.gate_failed_by_trajectory
                        | corrected_q21.gate_failed_by_trajectory
                    )
                    gate_rows = finite_rows[combined_gates]
                    if len(gate_rows):
                        gate_local_indices = torch.nonzero(
                            combined_gates, as_tuple=False
                        ).flatten()
                        for correction_local in gate_local_indices.detach().cpu().tolist():
                            local = int(finite_rows[correction_local].item())
                            global_index = begin + local
                            failure_step[global_index] = step_index + 1
                            p21_failed = bool(
                                corrected_p21.gate_failed_by_trajectory[
                                    correction_local
                                ].item()
                            )
                            q21_failed = bool(
                                corrected_q21.gate_failed_by_trajectory[
                                    correction_local
                                ].item()
                            )
                            if p21_failed and q21_failed:
                                reason = "correction_gate_p21_q21"
                            elif p21_failed:
                                reason = "correction_gate_p21"
                            else:
                                reason = "correction_gate_q21"
                            failure_reason[global_index] = reason
                        active[gate_rows] = False
                        formal_state[gate_rows] = 0.0
                        high_state[gate_rows] = 0.0

                    accepted = ~combined_gates
                    accepted_rows = finite_rows[accepted]
                    accepted_complete = compose(
                        corrected_p21.state[accepted],
                        corrected_q21.state[accepted],
                    )
                    accepted_finite = (
                        torch.isfinite(accepted_complete).flatten(1).all(1)
                    )
                    bad_rows = accepted_rows[~accepted_finite]
                    if len(bad_rows):
                        for local in bad_rows.detach().cpu().tolist():
                            global_index = begin + int(local)
                            failure_step[global_index] = step_index + 1
                            failure_reason[global_index] = "correction_nonfinite"
                        active[bad_rows] = False
                        formal_state[bad_rows] = 0.0
                        high_state[bad_rows] = 0.0

                    good = accepted_finite
                    good_rows = accepted_rows[good]
                    if len(good_rows):
                        formal_state[good_rows] = low_project(
                            accepted_complete[good]
                        )
                        high_state[good_rows] = q21_project(
                            accepted_complete[good]
                        )
                elif len(finite_rows):
                    formal_state[finite_rows] = finite_formal
                    high_state[finite_rows] = finite_high

            completed_step = step_index + 1
            if completed_step % save_every == 0:
                save_index = completed_step // save_every
                combined = compose(formal_state, high_state)
                maximum_low_track_q21 = max(
                    maximum_low_track_q21,
                    float(q21_project(formal_state).abs().max()),
                )
                maximum_high_track_p21 = max(
                    maximum_high_track_p21,
                    float(low_project(high_state).abs().max()),
                )
                snapshot = combined.detach().cpu().numpy().copy()
                inactive = ~active.detach().cpu().numpy()
                snapshot[inactive] = np.nan
                prediction[begin:end, save_index] = snapshot

    prediction[:, 0] = initial
    finite_by_time = np.isfinite(prediction).all(axis=(-2, -1))
    trigger_counts = p21_trigger_counts + q21_trigger_counts
    total_triggers = trigger_counts.sum(axis=0)
    trigger_steps = (trigger_counts > 0).sum(axis=0)
    triggered_ids = ids[total_triggers > 0].astype(int).tolist()
    correction_report = _correction_report(
        correction=correction,
        p21_policy=correction_policy,
        q21_policy=q21_correction_policy,
        grid=grid,
        ids=ids,
        p21_trigger_counts=p21_trigger_counts,
        q21_trigger_counts=q21_trigger_counts,
        p21_gate_failures=p21_gate_failures,
        q21_gate_failures=q21_gate_failures,
        p21_audit_maxima=p21_audit_maxima,
        q21_audit_maxima=q21_audit_maxima,
    )
    runtime = {
        "seconds": time.perf_counter() - started,
        "device": str(resolved_device),
        "dtype": "float32",
        "grid": grid,
        "maximum_representable_absolute_mode": grid // 2,
        "model_fixed_upper_output_cutoff": None,
        "trajectory_count": count,
        "rk4_step": step,
        "total_steps": steps,
        "save_every_steps": save_every,
        "saved_lead_times": saved_leads.tolist(),
        "rhs_evaluations_per_active_step": 4,
        "low_rhs_evaluations_by_trajectory": low_calls.tolist(),
        "high_frequency_rhs_evaluations_by_trajectory": high_calls.tolist(),
        "teacher_forcing": False,
        "truth_restart": False,
        "high_frequency_branch_enabled": branch is not None,
        "dual_track": True,
        "branch_input_context": (
            "complete stage state and same-stage frozen generator output"
            if branch is not None
            else None
        ),
        "branch_model_context_matches_protocol": True,
        "learned_branch_support": (
            "Q21 through the incoming-grid Nyquist support"
            if branch is not None
            else None
        ),
        "high_state_policy": (
            (
                "recursive complete Q21 with post-step Q21 local correction"
                if correction
                else "recursive complete Q21"
            )
            if branch is not None
            else (
                "zero-RHS Q21 with post-step Q21 local correction"
                if correction
                else "anchor Q21 preserved"
            )
        ),
        "finite_count_by_time": finite_by_time.sum(axis=0).astype(int).tolist(),
    }
    branch_audit = (
        {
            "enabled": True,
            "autonomous_and_stateless": True,
            "same_stage_low_high_coupling": True,
            "single_frozen_generator_evaluation_reused_per_stage": True,
            "strict_low_track": True,
            "selected_cutoff": 21,
            "model_fixed_upper_output_cutoff": None,
            "effective_output_limit": "incoming-grid Nyquist support",
            "input_context": (
                "complete stage state and same-stage frozen generator output"
            ),
            "model_context_matches_protocol": True,
            "lead0_full_anchor_preserved": True,
            "maximum_lead0_anchor_absolute_delta": 0.0,
            "positive_lead_high_state_semantics": "recursive complete Q21",
            "learned_rhs_support": "complete Q21",
            "recursive_local_correction_components": (
                ["P21", "Q21"] if correction else []
            ),
            "maximum_raw_branch_rhs_in_P21": maximum_raw_branch_rhs_p21,
            "maximum_applied_branch_rhs_in_P21": (
                maximum_applied_branch_rhs_p21
            ),
            "maximum_Q21_leakage_from_low_track": maximum_low_track_q21,
            "maximum_P21_leakage_from_high_track": maximum_high_track_p21,
        }
        if branch is not None
        else {
            "enabled": False,
            "complete_Q21_anchor_preserved": not correction,
            "high_frequency_rhs": "identically zero",
            "Q21_state_evolution": (
                "post-step local correction only"
                if correction
                else "anchor preserved"
            ),
            "recursive_local_correction_components": (
                ["P21", "Q21"] if correction else []
            ),
            "same_complete_state_low_generator_protocol": True,
        }
    )
    return RolloutResult(
        prediction=prediction,
        finite_by_time=finite_by_time,
        failure_step=failure_step,
        failure_reason=tuple(failure_reason),
        correction_trigger_count=total_triggers,
        correction_step_count=trigger_steps,
        correction_triggered_trajectory_ids=triggered_ids,
        correction=correction_report,
        runtime=runtime,
        branch_audit=branch_audit,
        method_display_name=(
            "GIFT" if branch is not None else "GIFT（高频支路关闭）"
        ),
    )
