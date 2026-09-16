"""Canonical, independent prediction-test readers; M1 data stays unchanged.

Every model receives the same identity-selected test rows. Selection never
depends on finite predictions, error rankings, model family or training seed.
The raw observation containers remain read-only.
"""

from __future__ import annotations

import h5py
import numpy as np

from gift.data_splits import TEST_IDS, canonical_ids
from gift.prediction_cohorts import FOUR_VORTEX, GAUSSIAN, observation_cohort, partitions


def _observations(path, group, grid, *, cohort=FOUR_VORTEX):
    with h5py.File(path, "r") as handle:
        if observation_cohort(handle) != cohort:
            raise ValueError("Observation population differs from the declared experiment")
        if cohort == GAUSSIAN and group == 'N64/test':
            group = 'test'
        stored = handle[f"{group}/trajectory_index"]
        times = handle[f"{group}/time"]
        raw = handle[f"{group}/vorticity"]
        if stored.dtype != np.dtype("int64") or times.dtype != np.dtype("float64"):
            raise ValueError("prediction identity/time dtype differs")
        ids = np.asarray(stored[:], dtype=np.int64)
        if handle.attrs.get("trajectory_id_scheme") != "canonical":
            ids = np.asarray(canonical_ids(ids), dtype=np.int64)
        elif len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate canonical prediction identities")
        expected_ids = partitions(cohort)['test']
        rows = np.flatnonzero(np.isin(ids, expected_ids))
        selected = ids[rows]
        if not np.array_equal(selected, expected_ids):
            raise ValueError("independent test population is missing or reordered")
        if raw.dtype != np.dtype("float32") or raw.shape != (len(ids), len(times), grid, grid):
            raise ValueError("prediction observation field schema differs")
        time_values = np.asarray(times[:], dtype=np.float64)
        fields = np.asarray(raw[rows], dtype=np.float32)
    if (not np.isfinite(time_values).all() or np.any(np.diff(time_values) <= 0)
            or not np.isfinite(fields).all()):
        raise ValueError("invalid observation time support or nonfinite test truth")
    return selected, time_values, fields


def _positions(times, requested):
    requested = np.asarray(requested, dtype=np.float64)
    if requested.ndim != 1 or not len(requested) or np.any(np.diff(requested) <= 0):
        raise ValueError("requested times must be strictly increasing")
    result = []
    for value in requested:
        found = np.flatnonzero(np.isclose(times, value, rtol=0, atol=2e-12))
        if len(found) != 1:
            raise ValueError("requested time is not one uniquely saved observation")
        result.append(int(found[0]))
    return np.asarray(result, dtype=np.int64)


def load_n64_long(paths):
    ids, times, fields = _observations(paths.standard_n64, "test", 64,
                                      cohort=getattr(paths, 'prediction_cohort', FOUR_VORTEX))
    requested = np.arange(5.0, 8.0 + 0.25, 0.5)
    return ids, requested, fields[:, _positions(times, requested)]


def load_n64_short(paths):
    ids, times, fields = _observations(paths.dense_n64, "test", 64)
    expected = np.round(np.arange(41, 61) / 10, 12)
    if not np.allclose(times, expected, rtol=0, atol=2e-12):
        raise ValueError("N64 short observation times differ")
    return ids, times[:10], fields[:, :10], fields[:, 9:20]


def load_cross_resolution(paths):
    ids, _, context, truth = load_n64_short(paths)
    times = np.round(np.arange(41, 61) / 10, 12)
    output = {64: (ids, times, np.concatenate((context, truth[:, 1:]), axis=1))}
    for grid in (96, 128):
        ids, saved, fields = _observations(paths.cross_resolution, f"N{grid}/test", grid)
        if not np.allclose(saved, times, rtol=0, atol=2e-12):
            raise ValueError("cross-resolution observation times differ")
        output[grid] = ids, saved, fields
    return output


def load_fno_test_dt0p02(paths, grid, report_times):
    if grid not in (64, 96, 128):
        raise ValueError("unsupported prediction-test grid")
    cohort = getattr(paths, 'prediction_cohort', FOUR_VORTEX)
    ids, times, fields = _observations(paths.fno_test_dt0p02, f"N{grid}/test", grid, cohort=cohort)
    if cohort == GAUSSIAN:
        if grid != 64:
            raise ValueError('S4 defines N64 only, not a new cross-resolution experiment')
        selected = _positions(times, 4.1 + np.arange(196)*.02)
        times, fields = times[selected], fields[:,selected]
    frames = 196 if grid == 64 else 96
    if len(times) != frames or not np.allclose(times, 4.1 + np.arange(frames) * 0.02, rtol=0, atol=2e-12):
        raise ValueError("native prediction observation time support differs")
    positions = _positions(times, report_times)
    return ids, times[:46], fields[:, :46], times[positions], fields[:, positions]


def evaluate_prediction_rhs(paths, branch_path, *, device, batch_size, include_actions=False):
    """S3 full-RHS error on independent test states, without coefficient fitting."""
    import torch
    from .gift_runtime import load_gift_models
    from .equation import _ratio
    from even_full_spectrum_ns import EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver
    ids, _, _, states = load_n64_short(paths)
    low_adapter, branch, _ = load_gift_models(paths, branch_path, 64, device)
    solver = EvenFullSpectrumNSSolver(EvenFullSpectrumConfiguration(grid=64, padding_grid=99), device=device)
    frequency = torch.fft.fftfreq(64, d=1.0 / 64.0, device=device)
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    p21 = (kx.abs() <= 21) & (ky.abs() <= 21)
    rows = []
    flat = states.reshape(-1, 64, 64)
    with torch.no_grad():
        for first in range(0, len(flat), batch_size):
            state = torch.from_numpy(flat[first:first + batch_size]).to(device)
            state_hat = torch.fft.fft2(state)
            learned_low = torch.fft.ifft2(sum(low_adapter.model.components_hat(state).values())).real
            learned_high = branch.forward_with_generator_output(state, learned_low)
            reference_hat = (solver.forcing_hat.expand_as(state_hat) + solver.linear * state_hat
                             - solver.transport_hat(state_hat))
            reference = (torch.fft.ifft2(reference_hat * p21).real
                         + torch.fft.ifft2(reference_hat * (~p21)).real)
            rows.append(_ratio(learned_low + learned_high - reference, reference))
    values = np.concatenate(rows).reshape(len(ids), states.shape[1])
    return {"trajectory_ids": ids, "enabled_vs_reference_full": values}, {
        "population": {"trajectory_ids": [int(ids[0]), int(ids[-1])],
                       "trajectories": len(ids), "states": int(values.size)},
        "reference_used_only_for_evaluation": True,
    }
