"""Score W1-v2 checkpoints without using future observations as model inputs.

Raw predictions are committed before metric tables or figures. Failed trajectories
and missing/failed training seeds remain in the declared population. The principal
metric averages the declared future per-frame relative L2 errors, excluding t0.
The default151-point grid is retained; sparse76/31 use75/30 futures at .04/.1 seconds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

LOCAL_TIMES = np.arange(151, dtype=np.float64) * .02
ABSOLUTE_TIMES = 5. + LOCAL_TIMES
FORMAL_SEEDS = (11, 22, 33)
GRID_INTERVALS = {151: .02, 76: .04, 31: .1}
SPARSE_TRAINING_COUNTS = {76: 300, 31: 200}


def evaluation_grid(query_times=None, *, dt=None):
    """Validate/canonicalize local0..3 grids (151/.02,76/.04,31/.1).

    Equivalent dense-stride and observation-step labels can differ by an ulp. Boundary
    checks use absolute1e-12 tolerance, then return one canonical float64 grid.
    """
    if query_times is None:
        interval = .02 if dt is None else float(dt)
        matches = [count for count, spacing in GRID_INTERVALS.items()
                   if np.isclose(interval, spacing, rtol=0, atol=1e-12)]
        if not matches:
            raise ValueError("W1 evaluation dt must be .02, .04 or .1 seconds")
        count = matches[0]
        return np.arange(count, dtype=np.float64) * GRID_INTERVALS[count]
    values = np.asarray(query_times, dtype=np.float64)
    if values.ndim != 1 or len(values) not in GRID_INTERVALS or not np.isfinite(values).all():
        raise ValueError("W1 query_times must be a finite31-,76- or151-point local grid")
    expected = evaluation_grid(dt=GRID_INTERVALS[len(values)])
    if not np.allclose(values, expected, rtol=0, atol=1e-12):
        raise ValueError("W1 query grid must be uniform and span local0..3 seconds")
    if dt is not None and not np.isclose(float(dt), expected[1], rtol=0, atol=1e-12):
        raise ValueError("declared dt and query_times disagree")
    return expected


def sparse_training_contract(query_times):
    """The prespecified sparse profile's common subset/stride, without reading data."""
    grid = evaluation_grid(query_times)
    if len(grid) == 151:
        return None
    return dict(training_count=SPARSE_TRAINING_COUNTS[len(grid)], subset_seed=2026100304,
                time_stride=int(round(grid[1]/.02)))


def prespecified_keyframe_times(query_times):
    """Return real predeclared nodes; .04-grid uses5.48 instead of unsampled5.5."""
    grid = evaluation_grid(query_times)
    return np.array([5.48 if len(grid) == 76 else 5.5, 6., 7., 8.], dtype=np.float64)


def _numpy(value):
    """Detach tensor outputs without changing their numerical values."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _json_ready(value):
    """Use JSON null plus explicit status fields for nonfinite/missing statistics."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    """Write strict JSON to a new artifact; never overwrite historical evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_json_ready(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def file_sha256(path):
    """Record a file digest as provenance, without making source bytes a run gate."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _summary(values):
    """Retain a whole-population statistic only when every member is finite."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    complete = bool(finite.all()) and values.size > 0
    return {
        "population": int(values.size), "finite_count": int(finite.sum()),
        "complete_population": complete,
        "mean": float(values.mean()) if complete else None,
        "sample_sd": float(values.std(ddof=1)) if complete and values.size > 1 else None,
        "finite_subset_mean_diagnostic": float(values[finite].mean()) if finite.any() else None,
    }


def evaluate_predictions(prediction, truth, trajectory_ids, *, solver_failed=None, query_times=None):
    """Measure raw ``[B,T,64,64]`` on the explicitly declared31/76/151-point grid.

    Returns ``summary`` and numerical arrays under ``per_trajectory`` plus
    ``relative_l2_by_time`` and failure masks. A failed future makes its whole-
    future scores infinite, so ``selection_metric`` is infinite and the JSON
    population mean is null. Finite-subset diagnostics never replace that score.
    Nonfinite t0 alone affects the t0/all-point diagnostic, not future prediction.
    ``solver_failed`` records explicit solver exceptions even if an array is finite.
    """
    prediction, truth = _numpy(prediction), _numpy(truth)
    grid = evaluation_grid(query_times)
    ids = np.asarray(trajectory_ids)
    if (prediction.shape != truth.shape or prediction.ndim != 4
            or prediction.shape[1:] != (len(grid), 64, 64) or len(prediction) < 1):
        raise ValueError(f"prediction/truth must share nonempty [B,{len(grid)},64,64] shape")
    if (ids.ndim != 1 or len(ids) != len(truth) or ids.dtype.kind not in "iu"
            or len(set(map(int, ids))) != len(ids)):
        raise ValueError("trajectory IDs must be unique integers with one ID per field")
    if not np.isfinite(truth).all():
        raise ValueError("nonfinite truth cannot be used to score predictions")
    explicit = np.zeros(len(ids), dtype=bool) if solver_failed is None else np.asarray(solver_failed, dtype=bool)
    if explicit.shape != (len(ids),):
        raise ValueError("solver_failed must have one flag per trajectory")
    finite = np.isfinite(prediction).all(axis=(-2, -1))
    failed = ~finite[:, 1:].all(axis=1) | explicit
    first = np.full(len(ids), -1, dtype=np.int64)
    for row in range(len(ids)):
        bad = np.flatnonzero(~finite[row, 1:])
        if len(bad):
            first[row] = int(bad[0])
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        target = truth.astype(np.float64)
        squared_error = (prediction.astype(np.float64) - target) ** 2
        numerator = squared_error.sum(axis=(-2, -1))
        denominator = (target ** 2).sum(axis=(-2, -1))
        if np.any(denominator <= 0):
            raise ValueError("relative L2 requires nonzero truth energy at every scored frame")
        relative = np.sqrt(numerator / denominator)
        future_mean = relative[:, 1:].mean(axis=1)
        endpoint = relative[:, -1].copy()
        future_mse = squared_error[:, 1:].mean(axis=(1, 2, 3))
        future_path = np.sqrt(numerator[:, 1:].sum(axis=1) / denominator[:, 1:].sum(axis=1))
        all_points_mse = squared_error.mean(axis=(1, 2, 3))
    for values in (future_mean, endpoint, future_mse, future_path):
        values[failed] = np.inf
    per_trajectory = {
        "trajectory_id": ids.astype(np.int64), "future_mean_relative_l2": future_mean,
        "t8_relative_l2": endpoint, "future_mse": future_mse,
        "future_path_relative_l2": future_path, "t0_relative_l2": relative[:, 0],
        "all_points_mse_diagnostic": all_points_mse,
    }
    if len(grid) == 151:
        per_trajectory["all151_mse_diagnostic"] = all_points_mse
    summaries = {name: _summary(values) for name, values in per_trajectory.items() if name != "trajectory_id"}
    summary = {
        "status": "complete" if not failed.any() else "failed_trajectories",
        "population": len(ids), "failed_count": int(failed.sum()),
        "failure_rate": float(failed.mean()), "failed_ids": ids[failed].tolist(),
        "primary_metric_name": "future_mean_per_frame_relative_l2",
        "primary_metric": summaries["future_mean_relative_l2"]["mean"],
        "selection_metric": float(future_mean.mean()), "metrics": summaries,
        "mean_relative_l2_by_time": [_summary(relative[:, index])["mean"] for index in range(len(grid))],
        "t0_in_primary_score": False, "time_points": len(grid), "future_steps": len(grid)-1,
        "query_times": grid.tolist(), "absolute_times": (5+grid).tolist(), "dt": float(grid[1]),
    }
    return {
        "summary": summary, "per_trajectory": per_trajectory,
        "relative_l2_by_time": relative, "finite_by_time": finite,
        "failed": failed, "solver_failed": explicit,
        "first_nonfinite_future_index": first,
    }


@dataclass
class ObservationWindow:
    """Exact row/time selection; initial fields and future truth have separate reads."""
    path: Path
    group: str
    rows: np.ndarray
    columns: np.ndarray
    trajectory_ids: np.ndarray
    cohort: str
    query_times: np.ndarray | None = None

    @property
    def local_times(self):
        """The explicit canonical prediction nodes in physical seconds."""
        return evaluation_grid(self.query_times)

    @property
    def absolute_times(self):
        """Absolute5..8 nodes corresponding to this window's prediction times."""
        return 5 + self.local_times

    def initial(self, first, last):
        """Read only the observed t5 field before calling a predictor."""
        with h5py.File(self.path, "r") as handle:
            return np.stack([handle[self.group + "/vorticity"][int(row), int(self.columns[0])]
                             for row in self.rows[first:last]]).astype(np.float32)

    def truth(self):
        """Read evaluation targets after raw autonomous predictions are committed."""
        with h5py.File(self.path, "r") as handle:
            return np.stack([handle[self.group + "/vorticity"][int(row), self.columns]
                             for row in self.rows]).astype(np.float32)


def observation_window(path, group, cohort, *, query_times=None):
    """Select the entire declared cohort, never choose rows using model errors."""
    path = Path(path).expanduser().resolve(strict=True)
    grid = evaluation_grid(query_times)
    absolute_times = 5+grid
    expected = {"old_test": np.arange(1040, 1220), "confirmation": np.arange(100000, 100180),
                "validation": np.arange(1000, 1020)}[cohort]
    with h5py.File(path, "r") as handle:
        node = handle[group]
        fields = node["vorticity"]
        ids = np.asarray(node["trajectory_index"][:])
        if handle.attrs.get("trajectory_id_scheme") != "canonical":
            if cohort == "confirmation":
                raise ValueError("confirmation requires canonical independent IDs")
            from gift.data_splits import canonical_ids
            ids = np.asarray(canonical_ids(ids), dtype=np.int64)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(map(int, ids))) != len(ids):
            raise ValueError("observation IDs must be unique integers")
        if (fields.ndim != 4 or fields.shape[0] != len(ids) or fields.shape[2:] != (64, 64)
                or fields.dtype != np.dtype("float32")):
            raise ValueError("observations must be float32 [trajectory,time,64,64]")
        times = np.asarray(node["absolute_time"][:] if "absolute_time" in node else node["time"][:])
        if times.shape != (fields.shape[1],) or np.any(np.diff(times) <= 0):
            raise ValueError("observation time axis must be strictly increasing")
        columns = np.searchsorted(times, absolute_times)
        # Float32 labels can lie one ulp to either side; use the nearest real node.
        columns = np.asarray([min((max(0, int(index) - 1), min(len(times) - 1, int(index))),
                                  key=lambda candidate: abs(times[candidate] - target))
                              for index, target in zip(columns, absolute_times)], dtype=np.int64)
        if (len(set(map(int, columns))) != len(grid)
                or not np.allclose(times[columns], absolute_times, rtol=0, atol=1e-6)):
            raise ValueError(f"all {len(grid)} actual t5..8 nodes are required; interpolation is not allowed")
        by_id = {int(value): row for row, value in enumerate(ids)}
        if not set(map(int, expected)) <= set(by_id):
            raise ValueError("the complete declared trajectory population is absent")
        if cohort == "confirmation" and (len(ids) != 180 or not bool(handle.attrs.get("complete", False))):
            raise ValueError("confirmation must be a complete, independently generated 180-row artifact")
        rows = np.asarray([by_id[int(value)] for value in expected], dtype=np.int64)
    return ObservationWindow(path, group, rows, columns, expected.astype(np.int64), cohort, grid)


def aggregate_runs(runs, expected_seeds=FORMAL_SEEDS, *, query_times=None):
    """Summarize every declared seed, including failures and absent records."""
    result = {}
    for method in sorted({run["method"] for run in runs}):
        selected = [run for run in runs if run["method"] == method]
        by_seed = {int(run["seed"]): run for run in selected}
        if len(by_seed) != len(selected):
            raise ValueError("duplicate method/seed records must not be averaged")
        unexpected = set(by_seed) - set(expected_seeds)
        if unexpected:
            raise ValueError(f"unexpected seeds for {method}: {sorted(unexpected)}")
        missing = [int(seed) for seed in expected_seeds if seed not in by_seed]
        invalid = [int(seed) for seed, run in by_seed.items()
                   if run.get("status") != "complete" or run.get("summary", {}).get("primary_metric") is None]
        complete = not missing and not invalid
        grids = [evaluation_grid(run["summary"].get("query_times")) for run in selected
                 if "mean_relative_l2_by_time" in run.get("summary", {})]
        grid = grids[0] if grids else evaluation_grid(query_times)
        if query_times is not None and not np.array_equal(grid, evaluation_grid(query_times)):
            raise ValueError("seed report grid differs from the declared evaluation grid")
        if any(len(value) != len(grid) or not np.allclose(value, grid, rtol=0, atol=1e-12) for value in grids):
            raise ValueError("different prediction grids cannot be pooled across training seeds")
        metrics = {}
        for name in ("future_mean_relative_l2", "t8_relative_l2", "future_mse", "future_path_relative_l2",
                     "t0_relative_l2", "all_points_mse_diagnostic"):
            means = []
            if complete:
                for seed in expected_seeds:
                    source = by_seed[seed]["summary"]["metrics"]
                    metric = source.get(name, source.get("all151_mse_diagnostic", {})
                                        if name == "all_points_mse_diagnostic" else {})
                    means.append(metric.get("mean"))
            values = np.asarray(means, dtype=np.float64)
            metric_complete = complete and bool(np.isfinite(values).all())
            metrics[name] = {"mean": float(np.mean(values)) if metric_complete else None,
                             "training_seed_sample_sd": float(np.std(values, ddof=1)) if metric_complete and len(values) > 1 else None}
        if len(grid) == 151:
            metrics["all151_mse_diagnostic"] = metrics["all_points_mse_diagnostic"].copy()
        curves = np.asarray([by_seed[seed]["summary"]["mean_relative_l2_by_time"]
                             for seed in expected_seeds], dtype=np.float64) if complete else None
        # An invalid t0 diagnostic must not invalidate a finite future curve.
        mean_curve, sd_curve = None, None
        if complete:
            valid_time = np.isfinite(curves).all(axis=0)
            mean_curve = np.full(len(grid), np.nan)
            mean_curve[valid_time] = curves[:, valid_time].mean(axis=0)
            if len(expected_seeds) > 1:
                sd_curve = np.full(len(grid), np.nan)
                sd_curve[valid_time] = curves[:, valid_time].std(axis=0, ddof=1)
        result[method] = {
            "status": "complete" if complete else "incomplete_or_failed_seeds",
            "time_points": len(grid), "future_steps": len(grid)-1, "query_times": grid.tolist(),
            "expected_training_seeds": list(expected_seeds), "recorded_training_seeds": sorted(by_seed),
            "missing_training_seeds": missing, "invalid_seed_runs": invalid,
            "failed_training_seeds": [int(seed) for seed, run in by_seed.items()
                                      if run.get("status") == "training_failed"],
            "prediction_failure_rates_by_seed": {int(seed): run.get("summary", {}).get("failure_rate")
                                                  for seed, run in by_seed.items()},
            "failed_trajectory_counts_by_seed": {int(seed): run.get("summary", {}).get("failed_count")
                                                  for seed, run in by_seed.items()},
            "metrics": metrics, "seed_records": selected,
            "mean_relative_l2_by_time": mean_curve.tolist() if complete else None,
            "training_seed_sd_by_time": sd_curve.tolist() if sd_curve is not None else None,
        }
    return result


def paired_statistics(records, *, expected_seeds=FORMAL_SEEDS, bootstrap_replicates=2000, seed=2026100304):
    """Pair complete trajectory IDs; intervals condition on the trained seed models.

    Resample trajectories, retaining all trained models. Never intersect away
    failures or treat trajectory/dropout repeats as new training seeds.
    """
    groups = {}
    for record in records:
        by_seed = groups.setdefault(record["method"], {})
        if int(record["seed"]) in by_seed:
            raise ValueError("duplicate method/seed records must not be paired")
        by_seed[int(record["seed"])] = record
    output = []
    rng = np.random.default_rng(seed)
    for left, right in itertools.combinations(sorted(groups), 2):
        entry = {"left": left, "right": right, "delta_definition": "left minus right",
                 "bootstrap_scope": "trajectory sampling conditional on these trained models"}
        if any(set(groups[name]) != set(expected_seeds) for name in (left, right)):
            output.append(dict(entry, status="missing_seed_records"))
            continue
        aligned, ids = {}, None
        unavailable = False
        for name in (left, right):
            rows = []
            for model_seed in expected_seeds:
                record = groups[name][model_seed]
                if record.get("status") != "complete" or "per_trajectory" not in record:
                    unavailable = True
                    break
                current_grid = evaluation_grid(record["summary"].get("query_times"))
                if "paired_grid" not in entry:
                    entry["paired_grid"] = current_grid.tolist()
                elif not np.array_equal(current_grid, np.asarray(entry["paired_grid"])):
                    raise ValueError("different physical prediction grids cannot be paired")
                per = record["per_trajectory"]
                current_ids = np.asarray(per["trajectory_id"], dtype=np.int64)
                if len(set(map(int, current_ids))) != len(current_ids):
                    raise ValueError("duplicate paired trajectory IDs")
                if ids is None:
                    ids = np.sort(current_ids)
                if set(map(int, current_ids)) != set(map(int, ids)):
                    raise ValueError("paired cohorts differ; do not silently intersect populations")
                lookup = {int(value): index for index, value in enumerate(current_ids)}
                values = np.asarray(per["future_mean_relative_l2"], dtype=np.float64)
                rows.append(values[[lookup[int(value)] for value in ids]])
            if unavailable:
                break
            aligned[name] = np.mean(rows, axis=0)
        if unavailable or any(not np.isfinite(values).all() for values in aligned.values()):
            output.append(dict(entry, status="failed_predictions_or_seeds"))
            continue
        delta = aligned[left] - aligned[right]
        interval = None
        if bootstrap_replicates:
            means = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean()
                                for _ in range(bootstrap_replicates)])
            interval = np.quantile(means, [.025, .975]).tolist()
        output.append(dict(entry, status="complete", trajectory_ids=ids.tolist(),
                           delta_values=delta.tolist(), mean_delta=float(delta.mean()),
                           bootstrap_95_percent_interval=interval,
                           bootstrap_replicates=bootstrap_replicates, bootstrap_seed=seed))
    return output


def validate_checkpoint_grid(payload, query_times=None):
    """Reject mismatched model/data grids, time units and sparse training identities."""
    spec = payload["model_spec"]
    grid = evaluation_grid(spec.get("query_times"))
    requested = grid if query_times is None else evaluation_grid(query_times)
    if len(grid) != len(requested) or not np.allclose(grid, requested, rtol=0, atol=1e-12):
        raise ValueError("checkpoint query grid differs from the requested evaluation grid")
    binding = payload.get("data_binding", payload.get("training_data", {}))
    contract = sparse_training_contract(grid)
    if contract and not all(key in binding for key in (
            "points", "future_steps", "query_times", "training_count", "training", "validation",
            "subset_seed", "time_stride", "source_frame_indices", "manifest_sha256")):
        raise ValueError("sparse checkpoint requires explicit complete training/data grid binding")
    for record in (spec, binding):
        for key in ("grid", "grid_size"):
            if key in record and record[key] != 64:
                raise ValueError("checkpoint spatial grid must be64")
        for key, expected in (("points", len(grid)), ("future_steps", len(grid)-1)):
            if key in record and record[key] != expected:
                raise ValueError("checkpoint " + key + " disagrees with query_times")
        if "query_times" in record:
            candidate = evaluation_grid(record["query_times"])
            if len(candidate) != len(grid) or not np.allclose(candidate, grid, rtol=0, atol=1e-12):
                raise ValueError("model specification and training data query grids differ")
        for key in ("observation_dt", "observation_dt_seconds", "save_interval"):
            if key in record and not np.isclose(record[key], grid[1], rtol=0, atol=1e-12):
                raise ValueError("checkpoint " + key + " disagrees with physical query spacing")
    if contract:
        count, stride = contract["training_count"], contract["time_stride"]
        training = np.asarray(binding["training"]["ids"])
        validation = np.asarray(binding["validation"]["ids"])
        if (binding["training_count"] != count or training.shape != (count,) or training.dtype.kind not in "iu"
                or len(set(map(int, training))) != count or np.any((training < 0) | (training >= 1000))
                or not np.array_equal(validation, np.arange(1000, 1020))):
            raise ValueError(f"sparse{len(grid)} requires common{count} train-only IDs and the unchanged20 validation IDs")
        if (binding["subset_seed"] != contract["subset_seed"] or binding["time_stride"] != stride
                or not np.array_equal(training, np.random.default_rng(contract["subset_seed"]).permutation(1000)[:count])
                or not np.array_equal(np.asarray(binding["source_frame_indices"]), np.arange(250, 401, stride))):
            raise ValueError("sparse subset/seed/source indices differ from the locked profile")
        for split in ("training", "validation"):
            digest = binding[split].get("observations_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sparse profile requires train/validation observation SHA256 bindings")
        manifest = binding["manifest_sha256"]
        if not isinstance(manifest, str) or len(manifest) != 64 or any(c not in "0123456789abcdef" for c in manifest):
            raise ValueError("sparse profile requires its scientific selection manifest SHA256")
    if spec.get("family") == "gift" or str(spec["method"]).lower() == "gift":
        if contract and not all(key in spec for key in ("save_interval", "step", "integration_dt")):
            raise ValueError("sparse GIFT must declare its observation saving and internal integration.02")
        if (not np.isclose(spec.get("duration", 3.), 3., rtol=0, atol=1e-12)
                or not np.isclose(spec.get("step", .02), .02, rtol=0, atol=1e-12)
                or not np.isclose(spec.get("integration_dt", .02), .02, rtol=0, atol=1e-12)):
            raise ValueError("GIFT must retain3-second horizon and .02 internal integration")
    if spec["method"] == "fno3d":
        normalization = payload.get("normalization", {})
        for prefix, shape in (("input", (64, 64, 1)), ("output", (64, 64, len(grid)-1))):
            for suffix in ("mean", "std"):
                if np.shape(normalization.get(prefix + "_" + suffix)) != shape:
                    raise ValueError("FNO3D normalization shape differs from its declared query grid")
    return grid, binding


def common_data_identity(payload):
    """Compare shared observations/IDs, ignoring source paths and GIFT-only sampler details."""
    grid, binding = validate_checkpoint_grid(payload)
    if len(grid) == 151:
        return None
    return {"query_times": grid.tolist(), "training_count": binding["training_count"],
            "training": {key: binding["training"][key] for key in ("ids", "observations_sha256")},
            "validation": {key: binding["validation"][key] for key in ("ids", "observations_sha256")},
            "subset_seed": binding["subset_seed"], "time_stride": binding["time_stride"],
            "source_frame_indices": binding["source_frame_indices"], "manifest_sha256": binding["manifest_sha256"]}


def checkpoint_metadata(path, query_times=None):
    """Check frozen single-field weights before opening any evaluation targets."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_spec", "config", "epoch", "val_metric", "seed", "stage"):
        if key not in payload:
            raise ValueError("W1-v2 checkpoint missing " + key)
    spec = payload["model_spec"]
    if spec.get("history") != 1:
        raise ValueError("new main evaluation accepts only single-field models")
    if payload["stage"] not in ("development", "formal"):
        raise ValueError("checkpoint stage must be development or formal")
    metric = np.asarray(_numpy(payload["val_metric"]))
    if metric.shape or metric.dtype.kind not in "fiu" or not bool(np.isfinite(metric)):
        raise ValueError("selected checkpoint must have a finite validation score")
    validate_checkpoint_grid(payload, query_times)
    return payload


def _load_checkpoint(path, device):
    """Load the new checkpoint schema and reuse the training model constructor."""
    payload = checkpoint_metadata(path)
    spec = payload["model_spec"]
    if spec.get("family") == "gift" or str(spec["method"]).lower() == "gift":
        from training.w1_v2_gift import load_gift_checkpoint
        low, branch, payload = load_gift_checkpoint(path, device=device)
        return (low, branch), payload
    from training.w1_v2_train import build_network
    from experiments.model_locations import runtime_model_spec
    if "model_state" not in payload:
        raise ValueError("new checkpoint requires model_state")
    model = build_network(runtime_model_spec(spec), device="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload


def training_cost(checkpoint, payload):
    """Distinguish selected epoch from the total training run's actual cost."""
    directory = Path(checkpoint).parent
    cost = dict(payload.get("training_cost", {}))
    status_path, attempt_path = directory / "STATUS.json", directory / "ATTEMPT.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        cost["run_terminal_status"] = status.get("status")
        for key in ("epoch", "updates", "training_seconds", "validation_seconds", "peak_memory_bytes"):
            if key in status:
                cost["actual_run_" + key] = status[key]
        cost["status_sidecar"] = str(status_path)
        cost["status_sidecar_sha256"] = file_sha256(status_path)
    if attempt_path.exists():
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        if "parameter_count" in attempt:
            cost["parameter_count"] = attempt["parameter_count"]
    cost["selected_epoch"] = payload["epoch"]
    return cost


def predict_checkpoint(checkpoint, window, raw_path, *, device="cpu", batch_size=4, previous_raw=None):
    """Persist autonomous predictions and failures; targets are not passed to models."""
    import torch
    from training.w1_v2_train import forward_model
    from experiments.model_locations import runtime_model_spec
    setup_started = time.perf_counter()
    model, payload = _load_checkpoint(checkpoint, device)
    grid, _ = validate_checkpoint_grid(payload, window.local_times)
    points = len(grid)
    setup_seconds = time.perf_counter() - setup_started
    spec = payload["model_spec"]
    runtime_spec = runtime_model_spec(spec)
    gift = spec.get("family") == "gift" or str(spec["method"]).lower() == "gift"
    failures = np.zeros(len(window.trajectory_ids), dtype=bool)
    reasons = [""] * len(failures)
    completed, previous_cost = 0, {}
    if previous_raw is not None:
        with h5py.File(previous_raw, "r") as previous:
            completed = int(previous.attrs.get("completed_trajectories", 0))
            previous_cost = json.loads(previous.attrs.get("inference_cost_json", "{}"))
            if completed and not previous_cost:
                raise ValueError("committed partial predictions require saved inference cost for resume")
            failures[:completed] = previous["solver_failed"][:completed]
            reasons[:completed] = previous["failure_reason"].asstr()[:completed].tolist()
    if not 0 <= completed < len(failures):
        raise ValueError("partial raw progress must lie within the declared population")
    started = time.perf_counter()
    segment_inference_seconds = 0.
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with h5py.File(raw_path, "x") as raw:
        raw.attrs.update(status="incomplete", checkpoint=str(checkpoint),
                         inference_input="omega(t5) only", cohort=window.cohort,
                         model_spec_json=json.dumps(_json_ready(spec), allow_nan=False))
        raw.create_dataset("trajectory_index", data=window.trajectory_ids)
        raw.create_dataset("absolute_time", data=5+grid)
        raw.create_dataset("time", data=grid)
        raw.attrs.update(time_points=points, future_steps=points-1, prediction_dt=float(grid[1]))
        prediction = raw.create_dataset("prediction", shape=(len(failures), points, 64, 64),
                                        dtype="f4", chunks=(1, 1, 64, 64), compression="lzf", fillvalue=np.nan)
        stored_failures = raw.create_dataset("solver_failed", data=failures)
        stored_reasons = raw.create_dataset("failure_reason", data=np.asarray(reasons, dtype=h5py.string_dtype("utf-8")))
        if completed:
            with h5py.File(previous_raw, "r") as previous:
                for first in range(0, completed, batch_size):
                    prediction[first:min(first + batch_size, completed)] = previous["prediction"][first:min(first + batch_size, completed)]
        raw.attrs["completed_trajectories"] = completed

        def commit_cost(last):
            """Commit actual segment costs together with the numerical progress."""
            peak = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None
            previous_peak = previous_cost.get("peak_cuda_allocated_bytes")
            if peak is not None and previous_peak is not None:
                peak = max(peak, previous_peak)
            cost = {"inference_seconds": previous_cost.get("inference_seconds", 0.) + segment_inference_seconds,
                    "prediction_and_io_seconds": previous_cost.get("prediction_and_io_seconds", 0.) + time.perf_counter() - started,
                    "model_setup_seconds": previous_cost.get("model_setup_seconds", 0.) + setup_seconds,
                    "batch_size": batch_size, "device": str(device), "training_cost": training_cost(checkpoint, payload),
                    "parameter_count": payload.get("parameter_counts", {}).get("total") if gift else
                                       sum(parameter.numel() for parameter in model.parameters()),
                    "parameter_counts": payload.get("parameter_counts") if gift else None,
                    "peak_cuda_allocated_bytes": peak, "inference_measurement_resumed": previous_raw is not None,
                    "inference_segments": previous_cost.get("inference_segments", []) + [
                        {"first_trajectory": completed, "committed_until": last, "device": str(device),
                         "batch_size": batch_size, "inference_seconds": segment_inference_seconds}],
                    "cost_scope": "actual inference work; interrupted native full-path work can include uncommitted outputs"}
            cost["seconds_per_trajectory"] = cost["inference_seconds"] / len(failures)
            raw.attrs["inference_cost_json"] = json.dumps(_json_ready(cost), allow_nan=False)
            stored_failures[:] = failures
            stored_reasons[:] = reasons
            raw.flush()
            raw.attrs["completed_trajectories"] = last
            raw.flush()
            return cost

        cost = commit_cost(completed)
        if gift:
            # Native rollout owns batching and per-trajectory failure isolation.
            # Load the pair once, rather than charging repeated checkpoint loads.
            from experiments.formal._shared.gift_runtime import rollout_gift
            initial = window.initial(completed, len(failures))
            if not np.isfinite(initial).all():
                raise ValueError("nonfinite initial observations")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            result = rollout_gift(low=model[0], branch=model[1], initial_state=initial,
                                  trajectory_ids=window.trajectory_ids[completed:], duration=3., save_interval=float(grid[1]),
                                  step=.02, device=torch.device(device), batch_size=batch_size,
                                  correction=True, execution="eager")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            segment_inference_seconds = time.perf_counter() - before
            gift_prediction = _numpy(result.prediction)
            if gift_prediction.shape != (len(failures) - completed, points, 64, 64):
                raise ValueError("GIFT predictor returned the wrong physical field layout")
            failures[completed:] = np.asarray(result.failure_step) >= 0
            reasons[completed:] = list(result.failure_reason)
            cost = commit_cost(completed)
        for first in range(completed, len(failures), batch_size):
            last = min(len(failures), first + batch_size)
            if gift:
                prediction[first:last] = gift_prediction[first - completed:last - completed]
                cost = commit_cost(last)
                continue
            initial = window.initial(first, last)
            if not np.isfinite(initial).all():
                raise ValueError("nonfinite initial observations")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            try:
                with torch.inference_mode():
                    value = _numpy(forward_model(model, torch.from_numpy(initial).to(device), runtime_spec,
                                                 payload.get("normalization"), device, training=False))
                if value.shape != (last - first, points, 64, 64):
                    raise ValueError("predictor returned the wrong physical field layout")
            except (FloatingPointError, ArithmeticError) as error:
                value = np.full((last - first, points, 64, 64), np.nan, dtype=np.float32)
                failures[first:last] = True
                reasons[first:last] = [type(error).__name__ + ": " + str(error)] * (last - first)
            # Structural/runtime/OOM errors propagate: never relabel a programming
            # or resource problem as a successfully completed scientific run.
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            segment_inference_seconds += time.perf_counter() - before
            prediction[first:last] = value
            cost = commit_cost(last)
            print(json.dumps({"status": "predictions_committed", "method": spec["method"],
                              "seed": payload["seed"], "completed": last, "population": len(failures)}), flush=True)
        raw.attrs["status"] = "complete"
    return payload, cost


def score_raw_predictions(raw_path, truth, metric_path):
    """Reopen committed raw numbers and persist the complete metric arrays."""
    with h5py.File(raw_path, "r") as raw:
        if raw.attrs.get("status") != "complete":
            raise ValueError("raw numerical predictions are not complete")
        grid = evaluation_grid(raw["time"][:])
        if not np.allclose(raw["absolute_time"][:], 5+grid, rtol=0, atol=1e-12):
            raise ValueError("saved local/absolute prediction times disagree")
        result = evaluate_predictions(raw["prediction"][:], truth, raw["trajectory_index"][:],
                                      solver_failed=raw["solver_failed"][:], query_times=grid)
    with h5py.File(metric_path, "x") as saved:
        saved.attrs["status"] = "incomplete"
        saved.create_dataset("time", data=grid)
        saved.create_dataset("absolute_time", data=5+grid)
        for key in ("relative_l2_by_time", "finite_by_time", "failed", "solver_failed", "first_nonfinite_future_index"):
            saved.create_dataset(key, data=result[key])
        per = saved.create_group("per_trajectory")
        for key, value in result["per_trajectory"].items():
            per.create_dataset(key, data=value)
        saved.attrs["status"] = "complete"
    return result


def draw_figures(output):
    """Draw scientific curves only from the previously saved numeric report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    if figures_complete(output):
        return
    report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
    absolute_times = np.asarray(report.get("absolute_times", ABSOLUTE_TIMES), dtype=np.float64)
    figures = output / "figures"
    if figures.exists():
        # Preserve a partially drawn directory after interruption as evidence.
        index = 1
        while (output / f"figures_retry_{index:03d}").exists():
            index += 1
        figures = output / f"figures_retry_{index:03d}"
    figures.mkdir(exist_ok=False)
    fig, axis = plt.subplots(figsize=(7, 4))
    plotted = 0
    for name, group in report["methods"].items():
        if group["status"] != "complete":
            continue
        mean = np.asarray(group["mean_relative_l2_by_time"][1:])
        axis.plot(absolute_times[1:], mean, label=name)
        if group["training_seed_sd_by_time"] is not None:
            sd = np.asarray(group["training_seed_sd_by_time"][1:])
            axis.fill_between(absolute_times[1:], mean - sd, mean + sd, alpha=.15)
        plotted += 1
    axis.set(xlabel="Physical time", ylabel="Mean relative L2 error",
             title="W1-v2: single field at t=5, future prediction")
    if plotted:
        axis.legend()
    else:
        axis.text(.5, .5, "No complete declared seed population", transform=axis.transAxes, ha="center")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(figures / ("future_error." + extension), dpi=180)
    plt.close(fig)
    keyframe = draw_keyframes(output, figures, report)
    write_json(figures / "provenance.json", {"numeric_report_sha256": file_sha256(output / "numeric_report.json"),
                                              "shading": "sample SD across declared training seeds", "t0_excluded": True,
                                              "keyframes": keyframe})
    marker = output / "FIGURES_COMPLETE.json"
    if marker.exists():
        preserve_interrupted(marker)
    write_json(marker, {"numeric_report_sha256": file_sha256(output / "numeric_report.json"),
                        "files": [str(path.relative_to(output)) for path in sorted(figures.iterdir())]})


def figures_complete(output):
    """Recognize fully committed figures without overwriting partial drawings."""
    output = Path(output)
    marker = output / "FIGURES_COMPLETE.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    return (record["numeric_report_sha256"] == file_sha256(output / "numeric_report.json")
            and all((output / path).is_file() for path in record["files"]))


def draw_keyframes(output, figures, report):
    """Show seed11/ID1040 or100000 at predeclared actual nodes from saved numbers."""
    import matplotlib.pyplot as plt
    cohort = report["cohort"]
    fixed_id = {"old_test": 1040, "confirmation": 100000}.get(cohort)
    truth_path = output / "raw" / "truth.h5"
    if fixed_id is None or not truth_path.exists():
        return {"status": "unavailable", "reason": "only old-test/confirmation with saved truth has a prescribed figure"}
    absolute_times = np.asarray(report.get("absolute_times", ABSOLUTE_TIMES), dtype=np.float64)
    wanted_times = prespecified_keyframe_times(absolute_times-5)
    columns = np.asarray([int(np.argmin(abs(absolute_times-target))) for target in wanted_times])
    if not np.allclose(absolute_times[columns], wanted_times, rtol=0, atol=1e-12):
        raise ValueError("prespecified keyframe times must be actual prediction nodes")
    with h5py.File(truth_path, "r") as handle:
        ids = handle["trajectory_index"][:]
        row = np.flatnonzero(ids == fixed_id)
        if len(row) != 1:
            raise ValueError("prespecified keyframe trajectory is absent or repeated")
        truth = handle["vorticity"][int(row[0]), columns]
    selected, predictions = [], {}
    for method, group in report["methods"].items():
        records = [run for run in group["seed_records"] if int(run["seed"]) == 11]
        record = records[0] if records else None
        selected.append((method, record))
        if record is not None and record.get("raw_prediction_file"):
            path = output / record["raw_prediction_file"]
            with h5py.File(path, "r") as handle:
                row = np.flatnonzero(handle["trajectory_index"][:] == fixed_id)
                if len(row) != 1:
                    raise ValueError("prespecified keyframe ID differs between models")
                predictions[method] = handle["prediction"][int(row[0]), columns]
    field_limit = max(float(np.abs(truth).max()), 1e-12)
    error_limit = 1e-12
    for value in predictions.values():
        error = np.abs(value - truth)
        finite_error = error[np.isfinite(error)]
        if len(finite_error):
            error_limit = max(error_limit, float(finite_error.max()))
    fig, axes = plt.subplots(1 + 2*len(selected), 4, figsize=(11, 2*(1 + 2*len(selected))), squeeze=False)
    field_image, error_image = None, None
    for col, target in enumerate(absolute_times[columns]):
        field_image = axes[0, col].imshow(truth[col], origin="lower", cmap="RdBu_r",
                                           vmin=-field_limit, vmax=field_limit)
        axes[0, col].set_title(f"t={target:g}")
    axes[0, 0].set_ylabel("Truth")
    for method_index, (method, record) in enumerate(selected):
        prediction_row = 1 + 2*method_index
        axes[prediction_row, 0].set_ylabel(method + "\nprediction")
        axes[prediction_row + 1, 0].set_ylabel(method + "\nabsolute error")
        for col in range(4):
            if method not in predictions:
                for offset in (0, 1):
                    axes[prediction_row + offset, col].text(.5, .5, "Seed 11 unavailable/failed",
                                                             ha="center", va="center", transform=axes[prediction_row + offset, col].transAxes)
                continue
            value = predictions[method][col]
            field_image = axes[prediction_row, col].imshow(np.ma.masked_invalid(value), origin="lower",
                                                            cmap="RdBu_r", vmin=-field_limit, vmax=field_limit)
            error_image = axes[prediction_row + 1, col].imshow(np.ma.masked_invalid(np.abs(value - truth[col])),
                                                               origin="lower", cmap="magma", vmin=0, vmax=error_limit)
            if not np.isfinite(value).all():
                axes[prediction_row, col].text(.5, .5, "Nonfinite prediction", ha="center", va="center",
                                                transform=axes[prediction_row, col].transAxes)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"W1-v2: prespecified trajectory {fixed_id}, training seed 11")
    fig.subplots_adjust(left=.16, right=.85, top=.95, bottom=.03, hspace=.16, wspace=.08)
    fig.colorbar(field_image, cax=fig.add_axes([.88, .55, .015, .32]), label="Vorticity (truth-fixed limits)")
    if error_image is not None:
        fig.colorbar(error_image, cax=fig.add_axes([.88, .12, .015, .32]), label="Absolute error (shared limits)")
    for extension in ("png", "svg"):
        fig.savefig(figures / ("fixed_id_keyframes." + extension), dpi=160)
    plt.close(fig)
    return {"status": "complete", "trajectory_id": fixed_id, "training_seed": 11,
            "absolute_time": absolute_times[columns].tolist(), "selection": "fixed before inspecting errors",
            "source_time_indices": columns.tolist(),
            "keyframe_rule": "predeclared5.48 for .04-grid; otherwise5.5; then6,7,8; no interpolation",
            "field_limits": [-field_limit, field_limit], "error_limits": [0, error_limit],
            "truth_sha256": file_sha256(truth_path),
            "source_raw_files": [record.get("raw_prediction_file") for _, record in selected if record is not None]}


def preserve_interrupted(path):
    """Rename one incomplete file without deletion or overwriting prior evidence."""
    path = Path(path)
    index = 0
    while True:
        target = path.with_name(f"{path.stem}.interrupted_{index:03d}{path.suffix}")
        if not target.exists():
            path.rename(target)
            return target
        index += 1


def raw_checkpoint_status(path, window, checkpoint, payload):
    """Bind reusable raw outputs to this fixed checkpoint and full cohort."""
    with h5py.File(path, "r+") as raw:
        if (raw.attrs.get("checkpoint") != str(checkpoint)
                or json.loads(raw.attrs.get("model_spec_json", "{}")) != _json_ready(payload["model_spec"])
                or not np.array_equal(raw["trajectory_index"][:], window.trajectory_ids)
                or not np.array_equal(raw["absolute_time"][:], window.absolute_times)
                or raw["prediction"].shape != (len(window.trajectory_ids), len(window.local_times), 64, 64)):
            raise ValueError("raw prediction checkpoint, model or declared cohort differs")
        completed = int(raw.attrs.get("completed_trajectories", 0))
        if not 0 <= completed <= len(window.trajectory_ids):
            raise ValueError("raw prediction progress is inconsistent")
        cost = json.loads(raw.attrs.get("inference_cost_json", "{}"))
        if completed and not cost:
            raise ValueError("committed raw predictions lack saved inference cost")
        if completed == len(window.trajectory_ids) and raw.attrs.get("status") != "complete":
            # Recover the final marker if interruption followed its last batch.
            raw.attrs["status"] = "complete"
        return raw.attrs.get("status", "incomplete"), cost


def evaluate_checkpoints(checkpoints, data_file, group, cohort, output, *, device="cpu", batch_size=4,
                         expected_seeds=FORMAL_SEEDS, failed_runs=(), bootstrap_replicates=2000, plots=True,
                         resume=False, query_times=None):
    """Evaluate specified frozen models on one complete, identical population."""
    output = Path(output).expanduser().resolve()
    if output.exists() and not resume:
        raise FileExistsError("evaluation output must be new; old numerical results are preserved")
    if resume and not output.is_dir():
        raise FileNotFoundError("resume requires this evaluation attempt's existing output")
    if batch_size < 1 or len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("batch_size must be positive and expected seeds unique/nonempty")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative")
    grid = evaluation_grid(query_times)
    checkpoints = [Path(path).expanduser().resolve(strict=True) for path in checkpoints]
    frozen = [checkpoint_metadata(path, grid) for path in checkpoints]
    common_bindings = [common_data_identity(payload) for payload in frozen]
    if common_bindings and any(value != common_bindings[0] for value in common_bindings[1:]):
        raise ValueError("models were not trained on the same declared grid/subset/observations")
    failures = []
    for path in failed_runs:
        failed = json.loads(Path(path).read_text(encoding="utf-8"))
        if not all(key in failed for key in ("method", "seed", "reason")):
            raise ValueError("failed-run JSON requires method, seed and reason")
        failures.append(dict(failed, stage=failed.get("stage", "formal"), status="training_failed",
                             summary={"primary_metric": None}))
    identities = [(payload["model_spec"]["method"], int(payload["seed"])) for payload in frozen]
    identities += [(failed["method"], int(failed["seed"])) for failed in failures]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("declare at least one run, with no duplicate method/seed identities")
    if any(seed not in expected_seeds for _, seed in identities):
        raise ValueError("checkpoint/failed-run seed differs from the declared seed population")
    stages = {payload["stage"] for payload in frozen} | {failed["stage"] for failed in failures}
    if len(stages) != 1:
        raise ValueError("development and formal weights require separate evaluation reports")
    if stages == {"development"} and cohort != "validation":
        raise ValueError("development candidates must not use test/confirmation for selection")
    if cohort != "validation" and tuple(expected_seeds) != FORMAL_SEEDS:
        raise ValueError("test/confirmation require prespecified training seeds 11,22,33")
    window = observation_window(data_file, group, cohort, query_times=grid)
    bindings = [{"checkpoint": str(path), "sha256": file_sha256(path), "model_spec": payload["model_spec"],
                 "config": payload["config"], "seed": payload["seed"], "stage": payload["stage"],
                 "selected_epoch": payload["epoch"], "val_metric": payload["val_metric"]}
                for path, payload in zip(checkpoints, frozen)]
    plan = {"cohort": cohort, "interpretation": "posthoc_exploratory" if cohort == "old_test" else cohort,
               "data_file": str(window.path), "group": group, "trajectory_ids": window.trajectory_ids,
               "absolute_times": 5+grid, "query_times": grid, "time_points": len(grid), "future_steps": len(grid)-1,
               "dt": float(grid[1]), "common_training_binding": common_bindings[0] if common_bindings else None,
               "keyframe_absolute_times": prespecified_keyframe_times(grid),
               "checkpoints": [str(Path(path).resolve()) for path in checkpoints],
               "checkpoint_bindings": bindings, "declared_failed_runs": failures,
               "expected_training_seeds": list(expected_seeds), "bootstrap_replicates": bootstrap_replicates,
               "failure_rule": "full-population metric invalid; no finite-subset ranking"}
    if resume:
        saved_plan = json.loads((output / "evaluation_plan.json").read_text(encoding="utf-8"))
        # Earlier31/151 receipts used these same fixed cases before recording
        # their times explicitly. Preserve compatible scientific-run resume.
        saved_plan.setdefault("keyframe_absolute_times", _json_ready(prespecified_keyframe_times(grid)))
        if saved_plan != _json_ready(plan):
            raise ValueError("resume checkpoint weights/configurations or evaluation population differ")
        if (output / "numeric_report.json").exists():
            try:
                report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                preserve_interrupted(output / "numeric_report.json")
            else:
                if plots:
                    draw_figures(output)
                return report
    else:
        output.mkdir(parents=True)
        (output / "raw").mkdir()
        (output / "metrics").mkdir()
        write_json(output / "evaluation_plan.json", plan)
    records, truth = [], None
    for index, checkpoint in enumerate(checkpoints):
        raw_path = output / "raw" / f"run_{index:02d}.h5"
        metric_path = output / "metrics" / f"run_{index:02d}.h5"
        record_path = output / "metrics" / f"run_{index:02d}.json"
        payload = frozen[index]
        previous_raw = None
        if raw_path.exists():
            status, cost = raw_checkpoint_status(raw_path, window, checkpoint, payload)
            if status != "complete":
                previous_raw = preserve_interrupted(raw_path)
        if not raw_path.exists():
            payload, cost = predict_checkpoint(checkpoint, window, raw_path, device=device, batch_size=batch_size,
                                               previous_raw=previous_raw)
        if record_path.exists() and metric_path.exists():
            with h5py.File(metric_path, "r") as metrics:
                metric_complete = metrics.attrs.get("status") == "complete"
            if metric_complete:
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
                else:
                    if (record["checkpoint_sha256"] != bindings[index]["sha256"]
                            or record["raw_prediction_sha256"] != file_sha256(raw_path)):
                        raise ValueError("saved metric sources differ from the committed model/predictions")
                    records.append(record)
                    continue
        if record_path.exists():
            preserve_interrupted(record_path)
        if metric_path.exists():
            preserve_interrupted(metric_path)
        if truth is None:
            truth_path = output / "raw" / "truth.h5"
            if truth_path.exists():
                with h5py.File(truth_path, "r") as stored:
                    complete_truth = stored.attrs.get("status") == "complete"
                    if complete_truth:
                        if (not np.array_equal(stored["trajectory_index"][:], window.trajectory_ids)
                                or not np.array_equal(stored["absolute_time"][:], window.absolute_times)):
                            raise ValueError("saved truth cohort/time differs")
                        truth = stored["vorticity"][:]
                if not complete_truth:
                    preserve_interrupted(truth_path)
            if truth is None:
                truth = window.truth()
                with h5py.File(truth_path, "x") as stored:
                    stored.attrs["status"] = "incomplete"
                    stored.create_dataset("trajectory_index", data=window.trajectory_ids)
                    stored.create_dataset("absolute_time", data=window.absolute_times)
                    stored.create_dataset("time", data=grid)
                    stored.create_dataset("vorticity", data=truth, chunks=(1, 1, 64, 64), compression="lzf")
                    stored.attrs["status"] = "complete"
        measured = score_raw_predictions(raw_path, truth, metric_path)
        record = {"method": payload["model_spec"]["method"], "seed": int(payload["seed"]),
                  "stage": payload["stage"], "status": measured["summary"]["status"],
                  "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                  "model_spec": payload["model_spec"], "config": payload["config"],
                  "selected_epoch": payload["epoch"], "validation_selection_metric": payload["val_metric"],
                  "summary": measured["summary"], "per_trajectory": measured["per_trajectory"],
                  "cost": cost, "raw_prediction_sha256": file_sha256(raw_path),
                  "raw_prediction_file": str(raw_path.relative_to(output)),
                  "metric_file": str(metric_path.relative_to(output))}
        write_json(record_path, record)
        records.append(record)
    records.extend(failures)
    methods = aggregate_runs(records, expected_seeds, query_times=grid)
    pairs = paired_statistics(records, expected_seeds=expected_seeds, bootstrap_replicates=bootstrap_replicates)
    report = {"schema": "gift.w1-v2-evaluation.v1", "status": "numerical_complete", "cohort": cohort,
              "interpretation": "posthoc_exploratory" if cohort == "old_test" else cohort,
              "data_file": str(window.path), "group": group, "population": len(window.trajectory_ids),
              "trajectory_ids": window.trajectory_ids, "expected_training_seeds": list(expected_seeds),
              "absolute_times": 5+grid, "query_times": grid, "time_points": len(grid), "future_steps": len(grid)-1,
              "dt": float(grid[1]), "common_training_binding": common_bindings[0] if common_bindings else None,
              "keyframe_absolute_times": prespecified_keyframe_times(grid),
              "methods": methods, "paired_comparisons": pairs,
              "limitations": ["Trajectory intervals condition on these trained models, not a population of training seeds.",
                              "Finite-subset diagnostics do not replace full-population scores.",
                              "The old test cohort has already influenced protocol development." if cohort == "old_test" else
                              "Confirmation results must not feed further configuration selection."],
              "numeric_files": [str(path.relative_to(output)) for path in sorted((output / "metrics").iterdir())]}
    write_json(output / "numeric_report.json", report)
    if plots:
        draw_figures(output)
    return report


def main(argv=None):
    """Explicit checkpoint evaluation; no training or automatic checkpoint search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], type=Path)
    parser.add_argument("--failed-run", action="append", default=[], type=Path)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--group", default="N64/test")
    parser.add_argument("--cohort", choices=("old_test", "confirmation", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dt", type=float, choices=(.02, .04, .1), default=.02)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.checkpoint and not args.failed_run:
        parser.error("supply frozen checkpoints or explicit failed-run records")
    if args.batch_size < 1 or args.bootstrap_replicates < 0 or len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("batch size, bootstrap count and expected seeds are invalid")
    evaluate_checkpoints(args.checkpoint, args.data_file, args.group, args.cohort, args.output,
                         device=args.device, batch_size=args.batch_size, expected_seeds=tuple(args.expected_seeds),
                         failed_runs=args.failed_run, bootstrap_replicates=args.bootstrap_replicates,
                         plots=not args.skip_plots, resume=args.resume, query_times=evaluation_grid(dt=args.dt))


if __name__ == "__main__":
    main()
