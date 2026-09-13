"""Post-training equation-action and coefficient-coordinate calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .common import ProjectPaths, activate_project, summarize
from .gift_runtime import load_gift_models


def load_equation_states(paths: ProjectPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths.require("dense_n64")
    with h5py.File(paths.dense_n64, "r") as handle:
        ids = np.asarray(handle["test/trajectory_index"][:], dtype=np.int64)
        times = np.asarray(handle["test/time"][:], dtype=np.float64)
        selected = np.flatnonzero((times >= 5.0 - 1e-12) & (times <= 6.0 + 1e-12))
        states = np.asarray(handle["test/vorticity"][:, selected], dtype=np.float32)
    expected_ids = np.arange(1000, 1200, dtype=np.int64)
    expected_times = np.round(5.0 + np.arange(11) / 10.0, 12)
    if not np.array_equal(ids, expected_ids) or not np.array_equal(times[selected], expected_times):
        raise ValueError("M1 test population or time axis differs")
    if states.shape != (200, 11, 64, 64) or not np.isfinite(states).all():
        raise ValueError("M1 state tensor differs")
    return ids, expected_times, states


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> np.ndarray:
    top = numerator.flatten(1).double().norm(dim=1)
    bottom = denominator.flatten(1).double().norm(dim=1).clamp_min(1.0e-30)
    return (top / bottom).detach().cpu().numpy()


@torch.no_grad()
def evaluate_equation_model(
    paths: ProjectPaths,
    branch_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    include_actions: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute every reported M1 quantity from raw states and model parameters."""

    ids, times, states = load_equation_states(paths)
    low_adapter, branch, _ = load_gift_models(paths, branch_path, 64, device)
    # Coefficient coordinates require the underlying low generator components.
    low = low_adapter.model
    activate_project(paths.root)
    from even_full_spectrum_ns import EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver

    solver = EvenFullSpectrumNSSolver(
        EvenFullSpectrumConfiguration(grid=64, padding_grid=99), device=device
    )
    frequency = torch.fft.fftfreq(64, d=1.0 / 64.0, device=device)
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    p21 = (kx.abs() <= 21) & (ky.abs() <= 21)
    p31 = (kx.abs() <= 31) & (ky.abs() <= 31)
    p22_31 = p31 & ~p21
    nonzero = p21 & ((kx * kx + ky * ky) > 0)
    rows = {
        "low_vs_reference_P21": [],
        "branch_vs_reference_Q21": [],
        "branch_vs_reference_P22_31": [],
        "branch_vs_reference_outer_P31": [],
        "enabled_vs_reference_full": [],
    }
    actions: dict[str, list[np.ndarray]] = {
        "state": [],
        "P21_generator_action": [],
        "branch_Q21_action": [],
        "GIFT_full_action": [],
        "reference_P21_action": [],
        "reference_Q21_action": [],
        "reference_full_action": [],
    }
    forcing_xy = [0.0, 0.0]
    quadratic_xy = [0.0, 0.0]
    linear_gram = np.zeros((2, 2), dtype=np.float64)
    linear_rhs = np.zeros(2, dtype=np.float64)
    linear_y2 = 0.0
    branch_energy = 0.0
    branch_p21_energy = 0.0
    branch_outer_energy = 0.0
    flat = states.reshape(-1, 64, 64)
    for start in range(0, len(flat), batch_size):
        state = torch.from_numpy(flat[start : start + batch_size]).to(device)
        state_hat = torch.fft.fft2(state)
        selected_hat = state_hat * p21
        learned_hat = low.components_hat(state)
        learned_low = torch.fft.ifft2(sum(learned_hat.values())).real
        learned_branch = branch.forward_with_generator_output(state, learned_low)
        constant_hat = solver.forcing_hat.expand_as(state_hat)
        full_components = {
            "constant": constant_hat,
            "linear": solver.linear * state_hat,
            "quadratic": -solver.transport_hat(state_hat),
        }
        reference_hat = sum(full_components.values())
        reference_p21 = torch.fft.ifft2(reference_hat * p21).real
        reference_q21 = torch.fft.ifft2(reference_hat * (~p21)).real
        rows["low_vs_reference_P21"].append(_ratio(learned_low - reference_p21, reference_p21))
        rows["branch_vs_reference_Q21"].append(
            _ratio(learned_branch - reference_q21, reference_q21)
        )
        learned_branch_band = torch.fft.ifft2(torch.fft.fft2(learned_branch) * p22_31).real
        reference_branch_band = torch.fft.ifft2(reference_hat * p22_31).real
        rows["branch_vs_reference_P22_31"].append(
            _ratio(learned_branch_band - reference_branch_band, reference_branch_band)
        )
        learned_outer = torch.fft.ifft2(torch.fft.fft2(learned_branch) * (~p31)).real
        reference_outer = torch.fft.ifft2(reference_hat * (~p31)).real
        rows["branch_vs_reference_outer_P31"].append(
            _ratio(learned_outer - reference_outer, reference_outer)
        )
        reference_full = reference_p21 + reference_q21
        rows["enabled_vs_reference_full"].append(
            _ratio(learned_low + learned_branch - reference_full, reference_full)
        )
        if include_actions:
            for name, value in {
                "state": state,
                "P21_generator_action": learned_low,
                "branch_Q21_action": learned_branch,
                "GIFT_full_action": learned_low + learned_branch,
                "reference_P21_action": reference_p21,
                "reference_Q21_action": reference_q21,
                "reference_full_action": reference_full,
            }.items():
                actions[name].append(
                    value.detach().cpu().numpy().astype(np.float32, copy=False)
                )

        learned_constant = torch.fft.ifft2(learned_hat["constant"]).real.double()
        truth_constant = torch.fft.ifft2(constant_hat * p21).real.double()
        forcing_xy[0] += float((truth_constant * learned_constant).sum())
        forcing_xy[1] += float(truth_constant.square().sum())

        learned_quadratic = torch.fft.ifft2(learned_hat["quadratic"]).real.double()
        truth_quadratic = torch.fft.ifft2(full_components["quadratic"] * p21).real.double()
        quadratic_xy[0] += float((truth_quadratic * learned_quadratic).sum())
        quadratic_xy[1] += float(truth_quadratic.square().sum())

        selected_state = torch.fft.ifft2(state_hat * nonzero).real.double()
        laplacian = torch.fft.ifft2(-(kx * kx + ky * ky) * state_hat * nonzero).real.double()
        learned_linear = torch.fft.ifft2(learned_hat["linear"] * nonzero).real.double()
        x0 = selected_state.reshape(-1).cpu().numpy()
        x1 = laplacian.reshape(-1).cpu().numpy()
        y = learned_linear.reshape(-1).cpu().numpy()
        design = np.stack((x0, x1), axis=1)
        linear_gram += design.T @ design
        linear_rhs += design.T @ y
        linear_y2 += float(y @ y)

        branch_hat = torch.fft.fft2(learned_branch).to(torch.complex128)
        branch_energy += float(branch_hat.abs().square().sum())
        branch_p21_energy += float(branch_hat[..., p21].abs().square().sum())
        branch_outer_energy += float(branch_hat[..., ~p31].abs().square().sum())

    coefficients = np.linalg.solve(linear_gram, linear_rhs)
    linear_residual_squared = max(
        linear_y2 - 2.0 * float(coefficients @ linear_rhs) + float(coefficients @ linear_gram @ coefficients),
        0.0,
    )
    trajectory_count = len(ids)
    metric_arrays = {
        name: np.concatenate(parts).reshape(trajectory_count, 11)
        for name, parts in rows.items()
    }
    action_arrays = (
        {
            name: np.concatenate(parts).reshape(trajectory_count, 11, 64, 64)
            for name, parts in actions.items()
        }
        if include_actions
        else {}
    )
    sufficient_statistics = {
        "forcing_numerator": np.asarray(forcing_xy[0], dtype=np.float64),
        "forcing_denominator": np.asarray(forcing_xy[1], dtype=np.float64),
        "quadratic_numerator": np.asarray(quadratic_xy[0], dtype=np.float64),
        "quadratic_denominator": np.asarray(quadratic_xy[1], dtype=np.float64),
        "linear_gram": linear_gram,
        "linear_rhs": linear_rhs,
        "linear_y_squared": np.asarray(linear_y2, dtype=np.float64),
    }
    coordinates = {
        "forcing_scale_gamma": float(forcing_xy[0] / max(forcing_xy[1], 1e-30)),
        "quadratic_scale_beta": float(quadratic_xy[0] / max(quadratic_xy[1], 1e-30)),
        "linear_damping_alpha": float(-coefficients[0]),
        "viscosity_nu": float(coefficients[1]),
        "linear_fit_residual_relative_to_learned_action": float(
            np.sqrt(linear_residual_squared / max(linear_y2, 1e-30))
        ),
    }
    report = {
        "population": {
            "trajectory_ids": [1000, 1199],
            "trajectory_count": trajectory_count,
            "state_count": trajectory_count * 11,
        },
        "coordinates": coordinates,
        "coordinate_reference": {
            "forcing_scale_gamma": 1.0,
            "quadratic_scale_beta": 1.0,
            "linear_damping_alpha": 0.0,
            "viscosity_nu": 0.01,
        },
        "metrics": {name: summarize(value) for name, value in metric_arrays.items()},
        "support": {
            "branch_P21_l2_fraction": float(np.sqrt(branch_p21_energy / max(branch_energy, 1e-30))),
            "branch_outer_P31_l2_fraction": float(
                np.sqrt(branch_outer_energy / max(branch_energy, 1e-30))
            ),
            "learned_rhs_support": "complete Q21 through N64 Nyquist support",
            "model_fixed_upper_output_cutoff": None,
        },
        "known_equation_used_after_training_only": True,
        "sufficient_statistics": {
            name: value.tolist() for name, value in sufficient_statistics.items()
        },
    }
    return {
        "trajectory_ids": ids,
        "absolute_times": times,
        **action_arrays,
        **metric_arrays,
        **sufficient_statistics,
    }, report
