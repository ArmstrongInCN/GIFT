"""Independently audit completed W1/W2 raw measurements and sole training slots.

This CPU-only reader never imports a model, calls a predictor/scorer, fits a
target statistic, starts training, or changes saved experimental evidence.
Immutable final receipts are created only after every required check passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

W1_METHODS = ("GIFT", "fno2d", "fno3d", "anie_ns", "uno", "unet", "GIFT-N50")
W2_METHODS = tuple(label for label in W1_METHODS if label != "anie_ns")
NEW_METHODS = ("unet", "uno", "GIFT-N50")
SEEDS = {"GIFT": 20260820, "GIFT-N50": 20260820, "fno2d": 0, "fno3d": 0,
         "uno": 0, "unet": 0, "anie_ns": 7}
COHORTS = ("confirmation", "old_test")
GRIDS = (64, 96, 128)


class AuditError(ValueError):
    """A scientific identity, completed artifact or independent numerical check failed."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def plain(value):
    """Serialize receipts without changing numerical arrays or their source files."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Digests:
    """Cache only immutable completed-file hashes during one final read-only audit."""
    def __init__(self):
        self.values = {}

    def __call__(self, path):
        path = Path(path).resolve(strict=True)
        if str(path) not in self.values:
            result = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b""):
                    result.update(block)
            self.values[str(path)] = result.hexdigest()
        return self.values[str(path)]


def expected_ids(cohort):
    return np.arange(100000, 100180, dtype=np.int64) if cohort == "confirmation" else np.arange(1040, 1220, dtype=np.int64)


def check_axes(node, *, ids, points, grid, field_name, context):
    """Check actual identities, array layout and physical nodes before reading fields."""
    require(np.array_equal(node["trajectory_index"][:], ids), context+": trajectory IDs/order differ")
    require(node[field_name].shape == (180, points, grid, grid), context+": field axes differ")
    require(node[field_name].dtype == np.dtype("float32"), context+": field dtype differs")
    if "time" in node:
        require(np.allclose(node["time"][:], np.arange(points)*.04, atol=1e-12, rtol=0), context+": local time labels differ")
    if "absolute_time" in node:
        require(np.allclose(node["absolute_time"][:], 5+np.arange(points)*.04, atol=1e-12, rtol=0), context+": absolute time labels differ")


def source_selection(path, group, cohort, points, grid):
    """Locate real observation nodes independently of the evaluation reader."""
    ids = expected_ids(cohort)
    with h5py.File(path, "r") as handle:
        node = handle[group]
        require(handle.attrs.get("trajectory_id_scheme") == "canonical", "reference requires canonical IDs")
        if cohort == "confirmation":
            require(bool(handle.attrs.get("complete", False)), "confirmation reference is incomplete")
        observed_ids = node["trajectory_index"][:]
        require(len(observed_ids) == 180 and set(observed_ids.tolist()) == set(ids.tolist()), "reference omits/replaces a prescribed trajectory")
        require(node["vorticity"].shape[0] == 180 and node["vorticity"].shape[-2:] == (grid, grid), "reference grid layout differs")
        times = node["absolute_time"][:] if "absolute_time" in node else node["time"][:]
        require(np.isfinite(times).all() and np.all(np.diff(times) > 0), "invalid reference time axis")
        targets = 5+np.arange(points)*.04
        columns = np.asarray([np.argmin(np.abs(times-t)) for t in targets], np.int64)
        require(len(set(columns.tolist())) == points and np.allclose(times[columns], targets, rtol=0, atol=1e-6),
                "reference lacks actual .04 nodes; interpolation/relabeling is forbidden")
        by_id = {int(ident): row for row, ident in enumerate(observed_ids)}
        rows = np.asarray([by_id[int(ident)] for ident in ids], np.int64)
        parameter_key = "initial_parameters" if "initial_parameters" in node else "initial_condition_parameters"
        parameters = node[parameter_key][:][rows]
        require(parameters.shape == (180, 4, 4) and np.isfinite(parameters).all(), "invalid continuous initial parameters")
    return dict(path=Path(path), group=group, rows=rows, columns=columns, parameters=parameters)


def independently_measure(prediction, truth, explicit_failures):
    """Calculate flattened float64 norms without any project scoring function.

Spatial sums use einsum rather than the evaluator's square/sum pipeline.
Every future node is scored. Explicit solver failures invalidate whole-future
scores for that trajectory, and t0 is excluded from the primary mean and MSE.
"""
    require(prediction.shape == truth.shape and prediction.ndim == 4, "raw/reference shapes differ")
    require(np.isfinite(truth).all(), "nonfinite independent truth")
    batch, points = truth.shape[:2]
    require(points > 1 and np.asarray(explicit_failures).shape == (batch,), "invalid numerical audit axes")
    target = truth.astype(np.float64).reshape(batch, points, -1)
    error = prediction.astype(np.float64).reshape(batch, points, -1)-target
    finite = np.isfinite(prediction).all(axis=(-2, -1))
    failures = ~finite[:, 1:].all(axis=1) | np.asarray(explicit_failures, bool)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        energy = np.einsum("bti,bti->bt", target, target, optimize=False)
        error_energy = np.einsum("bti,bti->bt", error, error, optimize=False)
        require(np.all(energy > 0), "relative L2 denominator is zero")
        relative = np.sqrt(error_energy/energy)
        values = dict(future_mean_relative_l2=relative[:, 1:].mean(axis=1),
                      endpoint_relative_l2=relative[:, -1].copy(),
                      future_mse=error_energy[:, 1:].mean(axis=1)/target.shape[-1],
                      future_path_relative_l2=np.sqrt(error_energy[:, 1:].sum(axis=1)/energy[:, 1:].sum(axis=1)),
                      t0_relative_l2=relative[:, 0].copy(),
                      all_points_mse_diagnostic=error_energy.mean(axis=1)/target.shape[-1])
    for key in ("future_mean_relative_l2", "endpoint_relative_l2", "future_mse", "future_path_relative_l2"):
        values[key][failures] = np.inf
    return dict(relative=relative, finite=finite, failed=failures, per_trajectory=values)


def independently_measure_bands(prediction, truth, failed):
    """Recheck W2 supplementary Fourier bands directly from stored physical fields.

Integer mode squares are rebuilt here without the experiment's band helpers.
Float32 projected fields and subtraction preserve the reported band's stated
precision; norms are independently reduced by flattened float64 einsum.
"""
    grid = truth.shape[-1]
    modes = np.rint(np.fft.fftfreq(grid)*grid).astype(np.int64)
    ky, kx = modes[:, None], modes[None, :]
    p21 = (np.abs(ky) <= 21) & (np.abs(kx) <= 21)
    p31 = (np.abs(ky) <= 31) & (np.abs(kx) <= 31)
    masks = dict(P21=p21, Q21=~p21, P22_31=p31 & ~p21, outer_P31=~p31)
    reliable = ~p31 & (np.abs(ky) < grid//2) & (np.abs(kx) < grid//2)
    if reliable.any():
        masks["reliable_outer_P31"] = reliable
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        left_hat, right_hat = np.fft.fft2(prediction, axes=(-2, -1)), np.fft.fft2(truth, axes=(-2, -1))
        scores = {}
        for name, mask in masks.items():
            left = np.fft.ifft2(left_hat*mask, axes=(-2, -1)).real.astype(np.float32)
            right = np.fft.ifft2(right_hat*mask, axes=(-2, -1)).real.astype(np.float32)
            difference = (left-right).astype(np.float64).reshape(len(truth), 26, -1)
            reference = right.astype(np.float64).reshape(len(truth), 26, -1)
            numerator = np.sqrt(np.einsum("bti,bti->bt", difference, difference, optimize=False))
            denominator = np.maximum(np.sqrt(np.einsum("bti,bti->bt", reference, reference, optimize=False)), np.finfo(np.float64).tiny)
            relative = numerator/denominator
            for suffix, values in (("future_mean", relative[:, 1:].mean(axis=1)), ("t6", relative[:, -1].copy())):
                values[failed] = np.inf
                scores[name+"_relative_l2_"+suffix] = values
    return scores


def compare_array(actual, expected, context, *, rtol=5e-11, atol=1e-12):
    """Require finite numerical values and failure/nonfinite patterns to agree."""
    left, right = np.asarray(actual), np.asarray(expected)
    require(left.shape == right.shape, context+": saved/recomputed shape differs")
    if left.dtype.kind == "b" or right.dtype.kind == "b":
        require(np.array_equal(left, right), context+": boolean failure/finite mask differs")
        return 0.
    require(np.array_equal(np.isnan(left), np.isnan(right)), context+": NaN positions differ")
    require(np.array_equal(np.isposinf(left), np.isposinf(right)) and np.array_equal(np.isneginf(left), np.isneginf(right)), context+": infinity positions differ")
    finite = np.isfinite(left) & np.isfinite(right)
    require(np.allclose(left[finite], right[finite], rtol=rtol, atol=atol), context+": saved/recomputed numerical values differ")
    return float(np.max(np.abs(left[finite]-right[finite]))) if finite.any() else 0.


def compare_json_vector(saved, expected, context):
    """JSON null denotes a nonfinite score, not a zero or omitted trajectory."""
    require(len(saved) == len(expected), context+": JSON population length differs")
    for index, (value, target) in enumerate(zip(saved, expected)):
        if np.isfinite(target):
            require(value is not None and np.isclose(value, target, rtol=5e-11, atol=1e-12), f"{context}: JSON score differs at row{index}")
        else:
            require(value is None, f"{context}: nonfinite score must be JSON null at row{index}")


def population_mean(vector):
    return float(np.mean(vector)) if len(vector) and np.isfinite(vector).all() else None


def verify_summary(summary, calculated, ids, points, endpoint, context):
    """Compare full180 means, failed counts/IDs and every reported time curve."""
    require(summary.get("population") == 180, context+": population is not180")
    require(summary.get("failed_count") == int(calculated["failed"].sum()), context+": failed count differs")
    if "failed_ids" in summary:
        require(summary["failed_ids"] == ids[calculated["failed"]].tolist(), context+": failed identities differ")
    if "failure_rate" in summary:
        require(np.isclose(summary["failure_rate"], calculated["failed"].mean(), rtol=0, atol=1e-15), context+": failure rate differs")
    metrics = calculated["per_trajectory"]
    for name in ("future_mean_relative_l2", endpoint, "future_mse", "future_path_relative_l2", "t0_relative_l2"):
        vector = metrics["endpoint_relative_l2" if name == endpoint else name]
        stored = summary["metrics"][name]
        expected = population_mean(vector)
        require(stored.get("population") == 180 and stored.get("finite_count") == int(np.isfinite(vector).sum()), context+": full-population metric count differs at "+name)
        if expected is None:
            require(stored.get("mean") is None, context+": failed population was replaced with finite-subset mean at "+name)
        else:
            require(stored.get("mean") is not None and np.isclose(stored["mean"], expected, rtol=5e-11, atol=1e-12), context+": population mean differs at "+name)
    primary = population_mean(metrics["future_mean_relative_l2"])
    require((primary is None and summary.get("primary_metric") is None) or
            (primary is not None and summary.get("primary_metric") is not None and np.isclose(summary["primary_metric"], primary, rtol=5e-11, atol=1e-12)), context+": primary future mean differs")
    expected_curve = [population_mean(calculated["relative"][:, i]) for i in range(points)]
    require(len(summary["mean_relative_l2_by_time"]) == points, context+": time curve omits actual nodes")
    for i, (saved, actual) in enumerate(zip(summary["mean_relative_l2_by_time"], expected_curve)):
        require((actual is None and saved is None) or (actual is not None and saved is not None and np.isclose(saved, actual, rtol=5e-11, atol=1e-12)), f"{context}: time-mean differs at index{i}")


def audit_measurement(directory, record, source, *, cohort, grid, points, label, checkpoint_sha, digest):
    """Independently check raw, reference, saved metric arrays, and JSON summaries."""
    context, ids = f"{directory.name}/N{grid}/{label}", expected_ids(cohort)
    if record.get("status") == "training_failed":
        require(record["summary"].get("population") == 180 and record["summary"].get("failed_count") == 180
                and record["summary"].get("primary_metric") is None, context+": scientific failed slot was dropped/averaged")
        return dict(method=label, grid=grid, status="scientific_training_failure_retained", population=180, failed_count=180)
    require(record.get("method") == label and record.get("seed") == SEEDS[label], context+": record identity differs")
    require(record.get("checkpoint_sha256") == checkpoint_sha, context+": record checkpoint differs from formal slot")
    raw_path, metric_path = directory / record["raw_prediction_file"], directory / record["metric_file"]
    require(digest(raw_path) == record["raw_prediction_sha256"], context+": raw prediction SHA differs from numerical record")
    truth_path = directory / "raw" / ("truth.h5" if points == 76 else f"N{grid}_truth.h5")
    max_delta = 0.
    pieces = dict(relative=[], finite=[], failed=[], per_trajectory={})
    with h5py.File(raw_path, "r") as raw, h5py.File(truth_path, "r") as stored_truth, h5py.File(source["path"], "r") as original, h5py.File(metric_path, "r") as stored:
        require(raw.attrs.get("status") == stored_truth.attrs.get("status") == stored.attrs.get("status") == "complete", context+": raw/reference/metric artifact incomplete")
        require(int(raw.attrs.get("completed_trajectories", 0)) == 180, context+": raw population not fully committed")
        check_axes(raw, ids=ids, points=points, grid=grid, field_name="prediction", context=context+" raw")
        check_axes(stored_truth, ids=ids, points=points, grid=grid, field_name="vorticity", context=context+" truth")
        if points == 26:
            require(raw.attrs.get("checkpoint_sha256") == checkpoint_sha and raw.attrs.get("method") == label, context+": native W2 raw identity differs")
        require(raw["solver_failed"].shape == (180,), context+": solver-failure population differs")
        metric_group = stored["per_trajectory"] if "per_trajectory" in stored else stored
        require(np.array_equal(metric_group["trajectory_id"][:], ids), context+": saved metric identities differ")
        endpoint = "t8_relative_l2" if points == 76 else "t6_relative_l2"
        require({"future_mean_relative_l2", endpoint, "future_mse", "future_path_relative_l2", "t0_relative_l2"} <= set(metric_group),
                context+": a required per-trajectory numerical array is absent")
        if points == 26:
            bands = ["P21", "Q21", "P22_31", "outer_P31"]+(["reliable_outer_P31"] if grid != 64 else [])
            require({name+"_relative_l2_"+suffix for name in bands for suffix in ("future_mean", "t6")} <= set(metric_group),
                    context+": required W2 supplementary Fourier arrays absent")
        for first in range(0, 180, 4):
            last = min(first+4, 180)
            truth = np.stack([original[source["group"]+"/vorticity"][int(row), source["columns"]] for row in source["rows"][first:last]])
            require(np.array_equal(stored_truth["vorticity"][first:last], truth), context+": saved truth differs from exact independent source nodes")
            predicted = raw["prediction"][first:last]
            values = independently_measure(predicted, truth, raw["solver_failed"][first:last])
            for key, expected in (("relative_l2_by_time", values["relative"]), ("finite_by_time", values["finite"]), ("failed", values["failed"]), ("solver_failed", raw["solver_failed"][first:last])):
                max_delta = max(max_delta, compare_array(stored[key][first:last], expected, context+" "+key))
            for key, expected in values["per_trajectory"].items():
                name = endpoint if key == "endpoint_relative_l2" else key
                if name in metric_group:
                    max_delta = max(max_delta, compare_array(metric_group[name][first:last], expected, context+" "+name))
                if name in record.get("per_trajectory", {}):
                    compare_json_vector(record["per_trajectory"][name][first:last], expected, context+" JSON "+name)
                pieces["per_trajectory"].setdefault(key, []).append(expected)
            if points == 26:
                bands = independently_measure_bands(predicted, truth, values["failed"])
                for key, expected in bands.items():
                    require(key in metric_group, context+": supplementary Fourier metric array absent: "+key)
                    max_delta = max(max_delta, compare_array(metric_group[key][first:last], expected, context+" "+key))
                    pieces["per_trajectory"].setdefault(key, []).append(expected)
            for key in ("relative", "finite", "failed"):
                pieces[key].append(values[key])
    calculated = dict(relative=np.concatenate(pieces["relative"]), finite=np.concatenate(pieces["finite"]),
                      failed=np.concatenate(pieces["failed"]), per_trajectory={key: np.concatenate(values) for key, values in pieces["per_trajectory"].items()})
    verify_summary(record["summary"], calculated, ids, points, endpoint, context)
    for key, vector in calculated["per_trajectory"].items():
        if key.endswith("_relative_l2_t6") or key.endswith("_relative_l2_future_mean"):
            saved = record["summary"]["metrics"].get(key)
            require(saved is not None and saved.get("population") == 180
                    and saved.get("finite_count") == int(np.isfinite(vector).sum()), context+": Fourier metric population omitted")
            target = population_mean(vector)
            require((target is None and saved.get("mean") is None) or
                    (target is not None and saved.get("mean") is not None and np.isclose(target, saved["mean"], rtol=5e-11, atol=1e-12)),
                    context+": supplementary Fourier metric mean differs: "+key)
    cost = record.get("cost", {})
    require(isinstance(cost.get("inference_seconds"), (int, float)) and np.isfinite(cost["inference_seconds"])
            and cost["inference_seconds"] >= 0, context+": actual inference cost is absent/nonfinite")
    if points == 26:
        audit = cost.get("adaptation_audit", {})
        require(audit.get("test_statistics_used", False) is False and audit.get("target_grid_fit", False) is False,
                context+": target-grid fit/test statistics were used")
        if label == "fno3d" and grid != 64:
            require(audit.get("source_grid") == 64 and audit.get("target_grid") == grid and audit.get("input_steps") == 1
                    and audit.get("output_steps") == 75 and audit.get("temporal_interpolation") is False,
                    context+": FNO3D frozen normalizer/full75 time interface differs")
            require(cost.get("forward_time_points") == 76 and cost.get("scored_time_points") == 26,
                    context+": FNO3D inference was not measured with the trained full75 future time axis")
        if grid == 64 and audit.get("no_additional_model_forward"):
            require(cost.get("inference_seconds") == 0., context+": reused W1 prefix inference charged again")
    return dict(method=label, grid=grid, status="independent_raw_recomputation_passed", population=180,
                failed_count=int(calculated["failed"].sum()), time_points=points, future_steps=points-1,
                max_absolute_numerical_delta=max_delta, raw_sha256=digest(raw_path), metric_sha256=digest(metric_path),
                truth_sha256=digest(truth_path), primary_metric=population_mean(calculated["per_trajectory"]["future_mean_relative_l2"]))


def completed_report(directory, digest):
    """Do not inspect fields from an evaluator still producing its artifacts."""
    require((directory / "numeric_report.json").is_file() and (directory / "REPORT_COMPLETE.json").is_file()
            and (directory / "FIGURES_COMPLETE.json").is_file(), str(directory)+": all numerical/report/figure completion receipts required")
    report = read_json(directory / "numeric_report.json")
    require(report.get("status") == "numerical_complete", str(directory)+": numerical status incomplete")
    numeric_sha = digest(directory / "numeric_report.json")
    completion, figures = read_json(directory / "REPORT_COMPLETE.json"), read_json(directory / "FIGURES_COMPLETE.json")
    require(completion.get("numeric_report_sha256") == figures.get("numeric_report_sha256") == numeric_sha, str(directory)+": completion source hash differs")
    require(completion.get("results_sha256") == digest(directory / "RESULTS.md"), str(directory)+": readable report hash differs")
    images = figures.get("figure_sha256", figures.get("file_sha256", {}))
    require(isinstance(images, dict) and images, str(directory)+": figure byte hashes absent")
    for relative, sha in images.items():
        require(digest(directory / relative) == sha, str(directory)+": figure/provenance bytes differ "+relative)
    return report


def read_events(path):
    """An unterminated append-only process record is not a final measurement."""
    text = Path(path).read_text(encoding="utf-8")
    require(not text or text.endswith("\n"), str(path)+": process ledger is still incomplete")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def event_costs(events, expected_training_labels, context):
    """Count actual process attempts once and reject extra/duplicate training."""
    keys, per_label, process_total, training_labels = set(), {}, 0., []
    for event in events:
        require(isinstance(event.get("command"), list) and event.get("started_utc") and event.get("finished_utc"), context+": incomplete process event")
        label, seconds = event.get("label"), event.get("process_seconds")
        require(isinstance(seconds, (int, float)) and np.isfinite(seconds) and seconds >= 0, context+": absent/invalid actual cost")
        key = (label, event["started_utc"], tuple(event["command"]))
        require(key not in keys, context+": duplicated actual process cost event")
        keys.add(key)
        duration = (datetime.fromisoformat(event["finished_utc"])-datetime.fromisoformat(event["started_utc"])).total_seconds()
        require(duration >= 0 and abs(duration-seconds) <= max(1., .001*seconds), context+": measured cost/time labels disagree")
        command = event["command"]
        module = command[command.index("-m")+1] if "-m" in command else ""
        if module.startswith("training."):
            require(label in expected_training_labels, context+": an extra/reused/target-grid training process appeared")
            training_labels.append(label)
            prior = per_label.get(label, {}).get("attempts", 0)
            require(prior == 0 or "--resume" in command, context+": a sole training slot was freshly initialized again")
        process_total += seconds
        entry = per_label.setdefault(label, dict(attempts=0, process_seconds=0., exit_codes=[]))
        entry["attempts"] += 1
        entry["process_seconds"] += seconds
        entry["exit_codes"].append(event.get("exit_code"))
    require(set(training_labels) == set(expected_training_labels), context+": required sole training slot has no actual process receipt")
    return dict(process_seconds=process_total, by_job=per_label, event_count=len(events),
                actual_training_seconds=sum(per_label[label]["process_seconds"] for label in expected_training_labels),
                training_job_labels=sorted(set(training_labels)), duplicated_cost_events=False)


def audit_training_window(path, binding, count):
    """Check locked observations and IDs by streaming independent decoded bytes."""
    fingerprints = {}
    with h5py.File(path, "r") as handle:
        require(handle.attrs.get("status") == "complete" and set(handle.keys()) == {"training", "validation"}, "training cache incomplete/contains test")
        for split, expected_count in (("training", count), ("validation", 20)):
            node = handle[split]
            require(np.array_equal(node["trajectory_index"][:], np.asarray(binding[split]["ids"])), "training/validation IDs differ from lock")
            require(node["vorticity"].shape == (expected_count, 76, 64, 64), "training grid/temporal layout differs")
            require(np.allclose(node["time"][:], np.arange(76)*.04, rtol=0, atol=1e-12)
                    and np.allclose(node["absolute_time"][:], 5+np.arange(76)*.04, rtol=0, atol=1e-12), "training nodes differ")
            result = hashlib.sha256()
            for row in range(expected_count):
                field = np.ascontiguousarray(node["vorticity"][row], np.float32)
                require(np.isfinite(field).all(), "nonfinite observed training field")
                result.update(field.tobytes())
            fingerprints[split] = result.hexdigest()
            require(fingerprints[split] == binding[split]["observations_sha256"], "locked decoded observations differ")
    return fingerprints


def independently_check_normalizer(payload, common_window):
    """Recompute unbiased statistics from training300 by float64 Welford updates.

This is independent of training's Torch mean/std reduction. The tolerance
allows rounding of saved float32 statistics, never validation/test fitting.
"""
    norm = payload.get("normalization", {})
    require(norm.get("kind") == "paper_UnitGaussianNormalizer" and norm.get("validation_or_test_used") is False
            and norm.get("eps") == 1e-5, "FNO3D normalizer is not frozen training-only native statistics")
    mean, second, count = None, None, 0
    with h5py.File(common_window, "r") as handle:
        dataset = handle["training/vorticity"]
        require(dataset.shape == (300, 76, 64, 64), "normalizer source is not locked N300 N64 training")
        for row in range(300):
            sample = dataset[row].astype(np.float64)
            count += 1
            if mean is None:
                mean, second = sample.copy(), np.zeros_like(sample)
            else:
                delta = sample-mean
                mean += delta/count
                second += delta*(sample-mean)
    std = np.sqrt(second/(count-1))
    targets = {"input_mean": mean[:1].transpose(1, 2, 0), "input_std": std[:1].transpose(1, 2, 0),
               "output_mean": mean[1:].transpose(1, 2, 0), "output_std": std[1:].transpose(1, 2, 0)}
    deltas = {}
    for key, target in targets.items():
        observed = norm[key].detach().cpu().numpy()
        deltas[key] = compare_array(observed, target, "FNO3D independently fitted train-only "+key, rtol=3e-6, atol=3e-6)
    return dict(status="independent_training_only_statistics_passed", fitting_population=300,
                fitting_grid=64, validation_or_test_used=False, input_steps=1, output_steps=75,
                independent_algorithm="float64 streaming Welford, unbiased sample SD", max_absolute_delta=deltas)


def audit_training(root, plan, extension, base, formal, digest):
    """Verify fixed recipes, actual sole seeds, imported identities and unique cost."""
    import torch
    require(set(formal) == set(W1_METHODS), "FORMAL must include exactly seven outcomes")
    require(extension.get("schema") == "gift.w1w2-native.extension-lock.v1" and extension.get("configurations_locked") is True
            and extension.get("protocol_locked") is True and extension.get("profile") == "single_native76_n300", "extension scientific lock differs")
    require(set(extension["reused_models"]) == set(W1_METHODS)-set(NEW_METHODS)
            and set(extension["training_configurations"]) == set(NEW_METHODS), "locked reused/new slots differ")
    require(extension.get("anie_in_w2") is False and set(extension["w2_methods"]) == set(W2_METHODS), "ANIE W2 inclusion/scope differs")
    require(plan.get("development_candidates") == 0 and plan.get("hyperparameter_search") is False
            and plan.get("repeated_seed_runs") is False and plan.get("new_runs_per_method") == 1, "unrequested search/repeated seeds appear")
    require(digest(plan["base_lock"]) == extension["base_lock_sha256"], "base protocol lock changed")
    common, reduced = extension["common_data_binding"], extension["n50_data_binding"]
    require(common == base["data_binding"] and common["training_count"] == 300 and reduced["training_count"] == 50,
            "common/reduced training population differs")
    require(reduced["training"]["ids"] == common["training"]["ids"][:50] and reduced["validation"] == common["validation"],
            "GIFT-N50 is not fixed nested50/same validation20")
    windows = dict(common=audit_training_window(plan["common_window"], common, 300),
                   reduced=audit_training_window(plan["n50_window"], reduced, 50))
    slots, payloads = {}, {}
    for label in W1_METHODS:
        slot = formal[label]
        require(slot.get("method") == label and slot.get("status") in ("complete", "failed"), "formal method incomplete: "+label)
        if slot["status"] == "failed":
            failed = read_json(slot["failure"])
            require(failed.get("seed") == SEEDS[label] and failed.get("reason") and failed.get("failure_kind") != "infrastructure",
                    "failed slot unresolved infrastructure/wrong seed: "+label)
            slots[label] = dict(status="scientific_training_failure_retained", seed=SEEDS[label], failure_file=slot["failure"], failure_sha256=digest(slot["failure"]))
            continue
        checkpoint = Path(slot["checkpoint"]).resolve(strict=True)
        sha = digest(checkpoint)
        require(slot.get("checkpoint_sha256") == sha, "formal checkpoint hash changed: "+label)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payloads[label] = payload
        require(payload.get("seed") == SEEDS[label] and payload.get("stage") == "formal"
                and payload.get("model_spec", {}).get("history") == 1, "checkpoint actual seed/stage/input differs: "+label)
        architecture = "GIFT" if label == "GIFT-N50" else label
        require(payload["model_spec"].get("method") == architecture, "checkpoint architecture differs: "+label)
        require(np.allclose(payload["model_spec"]["query_times"], np.arange(76)*.04, rtol=0, atol=1e-12), "checkpoint training/query grid differs: "+label)
        job = extension["training_configurations"][label] if label in NEW_METHODS else base["training_configurations"][label]
        for key, value in job["config"].items():
            require(plain(payload["config"].get(key)) == value, "checkpoint locked recipe differs: "+label+"/"+key)
        binding = payload["data_binding"]
        expected_binding = reduced if label == "GIFT-N50" else common
        for key in ("training", "validation", "query_times", "source_frame_indices", "manifest_sha256", "subset_seed", "time_stride", "training_count"):
            require(plain(binding.get(key)) == plain(expected_binding.get(key)), "checkpoint bound observations differ: "+label+"/"+key)
        state = payload.get("model_state", payload.get("model_state_dict"))
        require(isinstance(state, dict) and state, "checkpoint has no learned state: "+label)
        for value in state.values():
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                require(bool(torch.isfinite(value).all()), "nonfinite trained state must be declared failed: "+label)
        selected_epoch = payload["epoch"]
        if label == "anie_ns":
            status = read_json(checkpoint.parent / "STATUS.json")
            require(status.get("status") == "complete" and status.get("seed") == 7 and status.get("best_epoch") == selected_epoch
                    and Path(status["selected_checkpoint"]).resolve() == checkpoint and np.isfinite(payload["val_metric"]), "ANIE completed original best selection differs")
        elif architecture == "GIFT":
            require(selected_epoch == 500 and payload.get("selection") == "terminal_epoch", "GIFT native terminal selection differs")
            require(payload["model_spec"].get("step") == .02 and payload["model_spec"].get("save_interval") == .04
                    and payload["model_spec"].get("rollout_integration_steps") == [0, 4, 10, 20, 24, 30, 40, 50],
                    "GIFT internal dt/changed .08 and .48 leads differ")
            generator_file = Path(payload["model_spec"]["generator_checkpoint"]).resolve(strict=True)
            generator_payload = torch.load(generator_file, map_location="cpu", weights_only=True)
            from training.checkpoints import state_digest
            require(generator_payload.get("seed") == 2026072301 and generator_payload.get("epoch") == 500
                    and state_digest(generator_payload["model_state_dict"]) == payload["model_spec"]["generator_weights_sha256"],
                    "actual GIFT generator seed/completion/tensor binding differs")
            require(generator_payload["training_data"]["training"] == expected_binding["training"]
                    and generator_payload["training_data"]["validation"] == expected_binding["validation"],
                    "GIFT generator training population differs from its branch")
        else:
            require(selected_epoch == job["config"]["max_epochs"] and payload.get("status") == "terminal", "neural terminal selection differs: "+label)
        if label in extension["reused_models"]:
            reused = extension["reused_models"][label]
            require(Path(reused["checkpoint"]).resolve() == checkpoint and reused["sha256"] == sha
                    and slot.get("reused_training") is True and slot.get("new_training_seconds") == 0., "reused four-model identity/cost differs: "+label)
        elif label in ("uno", "unet"):
            attempt = read_json(checkpoint.parent / "ATTEMPT.json")
            status = read_json(checkpoint.parent / "STATUS.json")
            require(attempt.get("fresh_initialization") is True and attempt["identity"]["config"]["seed"] == 0
                    and status.get("status") == "complete" and status.get("epoch") == job["config"]["max_epochs"], "new sole initialization/completion differs: "+label)
        slots[label] = dict(status="frozen_completed_weights_passed", seed=SEEDS[label], checkpoint=str(checkpoint), checkpoint_sha256=sha,
                            selected_epoch=selected_epoch, training_count=expected_binding["training_count"], reused_training=label not in NEW_METHODS,
                            generator_seed=2026072301 if architecture == "GIFT" else None)
    normalizer = independently_check_normalizer(payloads["fno3d"], plan["common_window"]) if "fno3d" in payloads else dict(status="scientific_failed_slot_no_normalizer_fit")
    base_events_path, events_path = Path(plan["base_root"]) / "JOB_EVENTS.jsonl", root / "JOB_EVENTS.jsonl"
    base_events, events = read_events(base_events_path), read_events(events_path)
    base_cost = event_costs(base_events, {"train_GIFT_seed20260820", "train_fno2d_seed0", "train_fno3d_seed0", "train_anie_ns_seed7"}, "base process ledger")
    new_cost = event_costs(events, {"train_unet_seed0", "train_uno_seed0", "train_GIFT-N50_seed20260820"}, "extension process ledger")
    combined_keys = [(record["label"], record["started_utc"], tuple(record["command"])) for record in base_events+events]
    require(len(set(combined_keys)) == len(combined_keys), "base and extension actual process costs overlap")
    costs = dict(base_run_recorded_once=base_cost, extension_run_recorded_once=new_cost,
                 total_actual_training_seconds=base_cost["actual_training_seconds"]+new_cost["actual_training_seconds"],
                 extension_new_training_seconds=new_cost["actual_training_seconds"], reused_new_training_seconds=0.,
                 repeated_reported_inference_costs_not_added=True,
                 ledgers={str(base_events_path): digest(base_events_path), str(events_path): digest(events_path)})
    return dict(schema="gift.w1w2-native.final-training-audit.v1", status="passed", root=str(root), method_slots=slots,
                unique_training_instances=7, newly_trained_instances=3, reused_instances=4, multi_seed_runs=False,
                target_grid_training=False, target_grid_statistic_fitting=False, training_windows=windows,
                fno3d_frozen_normalizer_audit=normalizer, actual_process_cost=costs), payloads


def save_immutable(path, value):
    """Create a final receipt or verify the same evidence without modifying it."""
    path, value = Path(path), plain(value)
    if path.exists():
        require(read_json(path) == value, str(path)+": existing audit describes different evidence; preserve it and choose a new run")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def audit_run(root):
    """Audit only finalized artifacts and write two new successful final receipts."""
    root, digest = Path(root).resolve(strict=True), Digests()
    # All four completion sets are required before any raw/truth HDF is opened.
    reports = {}
    for experiment in ("w1", "w2"):
        for cohort in COHORTS:
            directory = root / f"{experiment}_evaluation" / cohort
            reports[(experiment, cohort)] = completed_report(directory, digest)
    plan, extension, formal = read_json(root / "RUN_PLAN.json"), read_json(root / "EXTENSION_LOCK.json"), read_json(root / "FORMAL.json")
    base = read_json(plan["base_lock"])
    training_audit, payloads = audit_training(root, plan, extension, base, formal, digest)
    slots = training_audit["method_slots"]
    results, pairs = {}, {}
    for experiment in ("w1", "w2"):
        for cohort in COHORTS:
            directory, report = root / f"{experiment}_evaluation" / cohort, reports[(experiment, cohort)]
            require(report.get("cohort") == cohort and report.get("trajectory_ids") == expected_ids(cohort).tolist(), "report cohort population differs")
            points, methods = (76, W1_METHODS) if experiment == "w1" else (26, W2_METHODS)
            require(report.get("time_points") == points and report.get("future_steps") == points-1
                    and np.allclose(report["query_times"], np.arange(points)*.04, rtol=0, atol=1e-12)
                    and np.allclose(report["absolute_times"], 5+np.arange(points)*.04, rtol=0, atol=1e-12), "report actual grid labels differ")
            if experiment == "w2":
                require(report.get("ANIE_included") is False and report.get("native_zero_shot") is True and report.get("test_statistics_used") is False,
                        "W2 native scope/target-statistic metadata differs")
            eval_plan = read_json(directory / "evaluation_plan.json")
            cohort_records = {}
            for grid in ((64,) if experiment == "w1" else GRIDS):
                if experiment == "w1":
                    expected_path = plan["n64_confirmation"] if cohort == "confirmation" else plan["old_test"]
                    expected_group = "test" if cohort == "confirmation" else "N64/test"
                    require(Path(eval_plan["data_file"]).resolve() == Path(expected_path).resolve() and eval_plan["group"] == expected_group,
                            "W1 source differs from the prescribed same cohort")
                    source = source_selection(eval_plan["data_file"], eval_plan["group"], cohort, points, grid)
                    records = report["methods"]
                else:
                    entry = eval_plan["source_files"][str(grid)]
                    require(digest(entry["path"]) == entry["sha256"], "W2 reference source bytes changed")
                    source = source_selection(entry["path"], entry["group"], cohort, points, grid)
                    records = report["methods"][str(grid)]
                    pairs[(cohort, grid)] = source["parameters"]
                require(set(records) == set(methods), "report required method slots differ or ANIE entered W2")
                cohort_records[str(grid)] = {}
                for label in methods:
                    record = records[label]
                    if slots[label]["status"] == "scientific_training_failure_retained":
                        require(record.get("status") == "training_failed", "failed training slot masquerades as successful prediction")
                        sha = None
                    else:
                        require(record.get("status") != "training_failed", "completed weights dropped as training failure")
                        sha = slots[label]["checkpoint_sha256"]
                    cohort_records[str(grid)][label] = audit_measurement(directory, record, source, cohort=cohort, grid=grid,
                                                                       points=points, label=label, checkpoint_sha=sha, digest=digest)
                    if sha is not None and experiment == "w1" and label not in NEW_METHODS:
                        original_directory = Path(plan["base_root"]) / "evaluation" / cohort
                        original_record = read_json(original_directory / "numeric_report.json")["methods"][label]
                        require(record.get("new_inference_performed") is False
                                and record["raw_prediction_sha256"] == original_record["raw_prediction_sha256"]
                                and record["checkpoint_sha256"] == original_record["checkpoint_sha256"],
                                "base four completed W1 measurements were recomputed/changed rather than reused")
                    if sha is not None and experiment == "w2" and grid == 64:
                        w1_directory = root / "w1_evaluation" / cohort
                        w1_record = reports[("w1", cohort)]["methods"][label]
                        cost = record.get("cost", {})
                        reuse = cost.get("adaptation_audit", {})
                        require(reuse.get("no_additional_model_forward") is True and cost.get("inference_seconds") == 0.
                                and reuse.get("source_raw_sha256") == w1_record["raw_prediction_sha256"],
                                "W2 N64 must reuse the same measured W1 prefix without extra inference costs")
                        with h5py.File(directory / record["raw_prediction_file"], "r") as short, h5py.File(w1_directory / w1_record["raw_prediction_file"], "r") as long:
                            for first in range(0, 180, 4):
                                require(np.array_equal(short["prediction"][first:first+4], long["prediction"][first:first+4, :26], equal_nan=True),
                                        "W2 N64 raw fields differ from their preserved original W1 prefix")
            results[experiment+"_"+cohort] = cohort_records
    for cohort in COHORTS:
        require(np.array_equal(pairs[(cohort, 64)], pairs[(cohort, 96)]) and np.array_equal(pairs[(cohort, 64)], pairs[(cohort, 128)]),
                "W2 paired initial parameters differ across target grids")
    identity = dict(extension_lock_sha256=digest(root / "EXTENSION_LOCK.json"), base_lock_sha256=digest(plan["base_lock"]),
                    formal_sha256=digest(root / "FORMAL.json"), numeric_reports={f"{exp}_{cohort}": digest(root / f"{exp}_evaluation" / cohort / "numeric_report.json")
                    for exp in ("w1", "w2") for cohort in COHORTS})
    result_audit = dict(schema="gift.w1w2-native.final-result-audit.v1", status="passed", root=str(root), identity=identity,
                       independent_recomputation=True, project_scoring_or_prediction_functions_called=False,
                       W1_method_slots_per_cohort=7, W2_method_slots_per_grid_cohort=6,
                       audited_completed_or_failed_measurements=50, population_per_measurement=180,
                       future_steps={"W1": 75, "W2": 25}, ANIE_in_W2=False,
                       paired_exact_initial_parameters=True, repeated_inference_costs_not_added=True,
                       measurements=results, numeric_tolerance=dict(rtol=5e-11, atol=1e-12),
                       reference_sources_and_saved_truth_compared_exactly=True)
    training_audit["identity"] = identity
    save_immutable(root / "FINAL_TRAINING_AUDIT.json", training_audit)
    save_immutable(root / "FINAL_RESULT_AUDIT.json", result_audit)
    return result_audit, training_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result, training = audit_run(args.root)
    except (AuditError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, json.dumps(dict(status="audit_failed_or_incomplete", reason=str(error)), ensure_ascii=False)+"\n")
    print(json.dumps(dict(status="passed", measurements=result["audited_completed_or_failed_measurements"],
                         newly_trained_instances=training["newly_trained_instances"], ANIE_in_W2=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
