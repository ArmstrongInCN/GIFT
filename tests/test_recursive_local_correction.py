"""Unit tests for the recursive local correction.

The correction is a fixed, gradient-free rule applied after each complete RK4
step, once per spectral band. These tests pin the parts that must not drift:
the per-band policy constants, containment of the correction inside the band it
belongs to, the mask built when only a band name is given, the aggregate report,
the fact that exceeding the safety cap stops only the offending trajectory, and
the audit text recorded when the correction is switched off.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.formal._shared.gift_runtime import (
    DEFAULT_P21_CORRECTION_POLICY,
    DEFAULT_Q21_CORRECTION_POLICY,
    LocalCorrectionPolicy,
    apply_local_correction,
    rollout_gift,
    square_selected_mask,
)


def _zero_generator(state: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(state)


def _projected_impulse(grid: int, selected_mask: torch.Tensor) -> torch.Tensor:
    impulse = torch.zeros(grid, grid, dtype=torch.float64)
    impulse[0, 0] = 1000.0
    return torch.fft.ifft2(torch.fft.fft2(impulse) * selected_mask).real


def test_default_component_policies_are_fixed() -> None:
    assert DEFAULT_P21_CORRECTION_POLICY.score_threshold == 2.1
    assert DEFAULT_Q21_CORRECTION_POLICY.score_threshold == 2.1
    assert DEFAULT_P21_CORRECTION_POLICY.amplitude_threshold == 22.0
    assert DEFAULT_Q21_CORRECTION_POLICY.amplitude_threshold == 12.0
    assert DEFAULT_P21_CORRECTION_POLICY.window_radius == 2
    assert DEFAULT_Q21_CORRECTION_POLICY.window_radius == 2
    assert DEFAULT_P21_CORRECTION_POLICY.maximum_flagged_fraction == 40.0 / 4096.0
    assert DEFAULT_Q21_CORRECTION_POLICY.maximum_flagged_fraction == 40.0 / 4096.0
    assert [
        DEFAULT_P21_CORRECTION_POLICY.maximum_flagged_points(grid)
        for grid in (64, 96, 128)
    ] == [40, 90, 160]
    assert [
        DEFAULT_Q21_CORRECTION_POLICY.maximum_flagged_points(grid)
        for grid in (64, 96, 128)
    ] == [40, 90, 160]


@pytest.mark.parametrize("scope", ("P21", "Q21"))
def test_component_correction_delta_stays_in_selected_band(scope: str) -> None:
    grid = 64
    p21_mask = square_selected_mask(
        grid, cutoff=21, device=torch.device("cpu")
    )
    selected_mask = p21_mask if scope == "P21" else ~p21_mask
    state = _projected_impulse(grid, selected_mask)[None]
    policy = LocalCorrectionPolicy(
        amplitude_threshold=1.0,
        maximum_flagged_fraction=0.5,
    )

    result = apply_local_correction(
        state,
        policy,
        selected_mask=selected_mask,
        selected_scope=scope,
    )

    assert int(result.trigger_count_by_trajectory[0]) > 0
    assert not bool(result.gate_failed_by_trajectory[0])
    delta_hat = torch.fft.fft2(result.state - state)
    assert float(delta_hat[..., selected_mask].abs().max()) > 1.0
    assert float(delta_hat[..., ~selected_mask].abs().max()) < 1.0e-9
    assert float(delta_hat[..., 0, 0].abs().max()) < 1.0e-9
    assert result.report["selected_scope"] == scope


def test_q21_scope_builds_high_frequency_mask_when_omitted() -> None:
    grid = 64
    p21_mask = square_selected_mask(
        grid, cutoff=21, device=torch.device("cpu")
    )
    state = _projected_impulse(grid, ~p21_mask)[None]
    policy = LocalCorrectionPolicy(
        amplitude_threshold=1.0,
        maximum_flagged_fraction=0.5,
    )

    result = apply_local_correction(state, policy, selected_scope="Q21")

    assert int(result.trigger_count_by_trajectory[0]) > 0
    delta_hat = torch.fft.fft2(result.state - state)
    assert float(delta_hat[..., ~p21_mask].abs().max()) > 1.0
    assert float(delta_hat[..., p21_mask].abs().max()) < 1.0e-9

    default_result = apply_local_correction(
        torch.zeros_like(state), selected_scope="Q21"
    )
    assert default_result.report["amplitude_threshold"] == 12.0


def test_rollout_aggregate_is_sum_of_component_reports() -> None:
    grid = 64
    initial = np.zeros((1, grid, grid), dtype=np.float32)
    initial[0, 0, 0] = 1000.0

    result = rollout_gift(
        low=_zero_generator,
        branch=None,
        initial_state=initial,
        trajectory_ids=np.asarray([17], dtype=np.int64),
        duration=0.02,
        save_interval=0.02,
        device=torch.device("cpu"),
        batch_size=1,
    )

    correction = result.correction
    aggregate = np.asarray(
        correction["trigger_count_by_step_trajectory"], dtype=np.int64
    )
    p21 = np.asarray(
        correction["components"]["P21"]["trigger_count_by_step_trajectory"],
        dtype=np.int64,
    )
    q21 = np.asarray(
        correction["components"]["Q21"]["trigger_count_by_step_trajectory"],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(aggregate, p21 + q21)
    assert correction["trigger_pixel_events"] == int(aggregate.sum())
    np.testing.assert_array_equal(
        result.correction_trigger_count, aggregate.sum(axis=0)
    )
    assert int(p21.sum()) > 0
    assert int(q21.sum()) > 0


def test_component_gate_failure_stops_only_its_trajectory() -> None:
    grid = 64
    p21_mask = square_selected_mask(
        grid, cutoff=21, device=torch.device("cpu")
    )
    p21_impulse = _projected_impulse(grid, p21_mask).to(torch.float32)
    q21_impulse = _projected_impulse(grid, ~p21_mask).to(torch.float32)
    coordinate = np.arange(grid, dtype=np.float32) / grid
    stable = np.repeat(
        np.sin(2.0 * np.pi * 3.0 * coordinate)[None, :], grid, axis=0
    ).astype(np.float32)
    initial = np.stack(
        [p21_impulse.numpy(), q21_impulse.numpy(), stable], axis=0
    )
    p21_policy = LocalCorrectionPolicy(maximum_flagged_fraction=1.0 / 4096.0)
    q21_policy = LocalCorrectionPolicy(
        amplitude_threshold=12.0,
        maximum_flagged_fraction=1.0 / 4096.0,
    )

    result = rollout_gift(
        low=_zero_generator,
        branch=None,
        initial_state=initial,
        trajectory_ids=np.asarray([101, 202, 303], dtype=np.int64),
        duration=0.02,
        save_interval=0.02,
        device=torch.device("cpu"),
        batch_size=3,
        correction_policy=p21_policy,
        q21_correction_policy=q21_policy,
    )

    assert result.failure_step.tolist() == [1, 1, -1]
    assert result.failure_reason == (
        "correction_gate_p21",
        "correction_gate_q21",
        None,
    )
    assert result.finite_by_time[:, 1].tolist() == [False, False, True]
    p21_gates = result.correction["components"]["P21"][
        "gate_failed_by_step_trajectory"
    ]
    q21_gates = result.correction["components"]["Q21"][
        "gate_failed_by_step_trajectory"
    ]
    aggregate_gates = result.correction["gate_failed_by_step_trajectory"]
    assert p21_gates == [[True, False, False]]
    assert q21_gates == [[False, True, False]]
    assert aggregate_gates == [[True, True, False]]


def test_disabled_branch_audit_describes_q21_correction() -> None:
    grid = 64
    initial = np.zeros((1, grid, grid), dtype=np.float32)
    initial[0, 0, 0] = 1000.0

    result = rollout_gift(
        low=_zero_generator,
        branch=None,
        initial_state=initial,
        trajectory_ids=np.asarray([5], dtype=np.int64),
        duration=0.02,
        save_interval=0.02,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert result.branch_audit["complete_Q21_anchor_preserved"] is False
    assert result.branch_audit["Q21_state_evolution"] == (
        "post-step local correction only"
    )
    assert result.branch_audit["recursive_local_correction_components"] == [
        "P21",
        "Q21",
    ]
    assert result.runtime["high_state_policy"] == (
        "zero-RHS Q21 with post-step Q21 local correction"
    )
