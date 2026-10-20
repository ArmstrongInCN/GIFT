"""Seven-slot W1 comparison reusing the four completed N300 measurements.

New U-NO/U-Net and the explicitly reduced-data GIFT-N50 use the same single
initial field, 76 physical nodes and whole180 population. GIFT-N50 is separated
as a sample/cost control. No historical original-W1 weights enter this run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import h5py
import numpy as np

from experiments.additional import w1_single_evaluate as native
from experiments.additional import w1_v2_evaluate as shared

QUERY_TIMES = native.QUERY_TIMES
METHOD_SEEDS = {**native.METHOD_SEEDS, "uno": 0, "unet": 0, "GIFT-N50": 20260820}
DISPLAY_NAMES = {**native.DISPLAY_NAMES, "uno": "U-NO", "unet": "U-Net", "GIFT-N50": "GIFT-N50"}


def _lock(value):
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else value


def _finite_state(value):
    import torch
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        return bool(torch.isfinite(value).all())
    return True


def checkpoint_metadata(label, path, base_lock, extension_lock):
    """Validate an explicit label, unique seed and exact frozen recipe before targets."""
    import torch
    base, extension = _lock(base_lock), _lock(extension_lock)
    if label not in METHOD_SEEDS:
        raise ValueError("unknown extended method label")
    path = Path(path).resolve(strict=True)
    if label in native.METHOD_SEEDS:
        payload = native.checkpoint_metadata(path, base)
        reused = extension["reused_models"][label]
        if Path(reused["checkpoint"]).resolve() != path or shared.file_sha256(path) != reused["sha256"]:
            raise ValueError("reused checkpoint differs from the completed frozen source")
        return payload
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_spec", "config", "data_binding", "seed", "stage", "epoch", "val_metric"):
        if key not in payload:
            raise ValueError("extended checkpoint missing " + key)
    job = extension["training_configurations"][label]
    if payload["stage"] != "formal" or payload["seed"] != METHOD_SEEDS[label]:
        raise ValueError("extended checkpoint seed/stage differs from its only slot")
    if payload["model_spec"].get("history") != 1:
        raise ValueError("extended comparison requires one true initial field")
    for key, value in job["config"].items():
        if shared._json_ready(payload["config"].get(key)) != value:
            raise ValueError("extended checkpoint recipe differs at " + key)
    for key, value in job.get("model_spec", {}).items():
        if shared._json_ready(payload["model_spec"].get(key)) != value:
            raise ValueError("extended model specification differs at " + key)
    spec, binding = payload["model_spec"], payload["data_binding"]
    if label == "GIFT-N50":
        from training.w1_v2_gift import _validate_checkpoint_grid
        _validate_checkpoint_grid(payload)
        if spec.get("method") != "GIFT" or payload.get("selection") != "terminal_epoch" or payload["epoch"] != 500:
            raise ValueError("GIFT-N50 requires its native terminal pair")
        if binding.get("training_count") != 50 or binding["training"]["ids"] != base["data_binding"]["training"]["ids"][:50]:
            raise ValueError("GIFT-N50 must use the predeclared first50 of N300")
        expected = extension["n50_data_binding"]
        for key in ("training", "validation", "query_times", "source_frame_indices", "manifest_sha256", "subset_seed", "time_stride"):
            if shared._json_ready(binding.get(key)) != expected[key]:
                raise ValueError("GIFT-N50 observations differ at " + key)
    else:
        shared.validate_checkpoint_grid(payload, QUERY_TIMES)
        if (spec.get("method") != label or payload.get("status") != "terminal"
                or payload.get("schema") != "gift.w1-extended-native.weights.v1" or payload.get("selection") != "terminal"
                or payload["epoch"] != job["config"]["max_epochs"]):
            raise ValueError("U-NO/U-Net require completed terminal weights")
        expected = shared.common_data_identity({"model_spec": {"method": "anie_ns", "history": 1,
                                                "query_times": QUERY_TIMES.tolist()}, "data_binding": base["data_binding"]})
        if shared.common_data_identity(payload) != expected:
            raise ValueError("extended baselines must use the exact shared N300 observations")
    state = payload.get("model_state", payload.get("model_state_dict"))
    if not isinstance(state, dict) or not state or not _finite_state(state):
        raise ValueError("checkpoint requires a finite model state dictionary")
    return payload


def load_checkpoint(label, path, payload, device):
    """Load an explicit new/old label without changing old checkpoint semantics."""
    if label in native.METHOD_SEEDS or label == "GIFT-N50":
        return native.load_checkpoint(path, payload, device)
    from adapters.w1_extended import build_network
    from experiments.model_locations import runtime_model_spec
    model = build_network(runtime_model_spec(payload["model_spec"]), device="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def _forward(model, initial, spec, normalization, device, *, training=False):
    from experiments.model_locations import runtime_model_spec
    spec = runtime_model_spec(spec)
    if spec["method"] in ("uno", "unet"):
        from adapters.w1_extended import forward_model
        return forward_model(model, initial, spec, device, training=training)
    from training.w1_single_train import forward_model
    return forward_model(model, initial, spec, normalization, device, training=training)


def _link_immutable(source, destination):
    """Reuse complete immutable fields with a hard link, falling back to an exact copy."""
    import shutil
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if shared.file_sha256(source) != shared.file_sha256(destination):
            raise ValueError("existing imported prediction differs from immutable source")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _validate_imported_truth(window, truth_path):
    """Compare exact selected fields, not just IDs, before reusing prior measurements."""
    with h5py.File(truth_path, "r") as saved, h5py.File(window.path, "r") as source:
        if (saved.attrs.get("status") != "complete"
                or not np.array_equal(saved["trajectory_index"][:], window.trajectory_ids)
                or not np.allclose(saved["time"][:], window.local_times, rtol=0, atol=1e-12)
                or not np.allclose(saved["absolute_time"][:], window.absolute_times, rtol=0, atol=1e-12)
                or saved["vorticity"].shape != (len(window.trajectory_ids), len(QUERY_TIMES), 64, 64)):
            raise ValueError("imported truth is incomplete or its declared cohort/time differs")
        field = source[window.group + "/vorticity"]
        for index, source_row in enumerate(window.rows):
            if not np.array_equal(saved["vorticity"][index], np.asarray(field[int(source_row), window.columns], dtype=np.float32)):
                raise ValueError("imported truth values differ from the selected evaluation source")


def _commit_truth(window, path):
    """Save exact observed evaluation nodes after autonomous prediction completes."""
    truth = window.truth()
    with h5py.File(path, "x") as saved:
        saved.attrs.update(status="complete", cohort=window.cohort)
        saved.create_dataset("trajectory_index", data=window.trajectory_ids)
        saved.create_dataset("time", data=QUERY_TIMES)
        saved.create_dataset("absolute_time", data=5 + QUERY_TIMES)
        saved.create_dataset("vorticity", data=truth, chunks=(1, 1, *truth.shape[-2:]), compression="lzf")


def _metric_record(record_path, metric_path, checkpoint_sha256, raw_sha256, population=180):
    """Accept only complete, bound metric arrays; an interrupted pair must be rebuilt."""
    if not record_path.is_file() or not metric_path.is_file():
        return None
    try:
        candidate = _lock(record_path)
    except json.JSONDecodeError:
        return None
    with h5py.File(metric_path, "r") as metrics:
        if metrics.attrs.get("status") != "complete":
            return None
        if ("relative_l2_by_time" not in metrics
                or metrics["relative_l2_by_time"].shape != (population, len(QUERY_TIMES))
                or "time" not in metrics
                or not np.allclose(metrics["time"][:], QUERY_TIMES, rtol=0, atol=1e-12)):
            raise ValueError("completed metric array population/grid differs")
    if (candidate.get("checkpoint_sha256") != checkpoint_sha256
            or candidate.get("raw_prediction_sha256") != raw_sha256):
        raise ValueError("completed metric sources differ from the fixed model/predictions")
    if candidate.get("metric_sha256") != shared.file_sha256(metric_path):
        raise ValueError("completed metric bytes differ from their recorded measurement")
    return candidate


def _numeric_artifacts_complete(output, report, plan):
    """A JSON report is complete only with every slot's committed raw and metric files."""
    if (report.get("status") != "numerical_complete" or report.get("cohort") != plan["cohort"]
            or report.get("trajectory_ids") != plan["trajectory_ids"]
            or report.get("query_times") != plan["query_times"]
            or set(report.get("methods", {})) != set(METHOD_SEEDS)):
        return False
    for label, record in report["methods"].items():
        record_path = output / "metrics" / (label + ".json")
        if record.get("status") == "training_failed":
            if not record_path.is_file() or _lock(record_path) != record:
                return False
            continue
        if not all(key in record for key in ("raw_prediction_file", "raw_prediction_sha256", "metric_file", "checkpoint_sha256")):
            return False
        raw_path, metric_path = output / record["raw_prediction_file"], output / record["metric_file"]
        if not raw_path.is_file() or not metric_path.is_file():
            return False
        with h5py.File(raw_path, "r") as raw:
            if raw.attrs.get("status") != "complete" or int(raw.attrs.get("completed_trajectories", 0)) != len(plan["trajectory_ids"]):
                return False
        if shared.file_sha256(raw_path) != record["raw_prediction_sha256"]:
            raise ValueError("completed report's raw prediction bytes changed")
        cached = _metric_record(record_path, metric_path, plan["checkpoints"][label]["sha256"],
                                record["raw_prediction_sha256"], len(plan["trajectory_ids"]))
        if cached is None:
            return False
        if cached != record:
            raise ValueError("completed report differs from its per-method measurement")
    return True


def evaluate(checkpoints, failed_runs, *, base_lock, extension_lock, data_file, group, cohort, output,
             reuse_evaluation=None, device="cuda", batch_size=4, resume=False, plots=True, job_events=None):
    """Measure all seven fixed models, optionally importing the frozen four outputs.

    With no imported evaluation every method predicts from the single initial
    field using its existing native loader and forward path. Imported evaluations
    retain the original extension experiment's four unchanged measurements.
    """
    extension, base = _lock(extension_lock), _lock(base_lock)
    if len(checkpoints) + len(failed_runs) != 7 or set(checkpoints) | set(failed_runs) != set(METHOD_SEEDS) or set(checkpoints) & set(failed_runs):
        raise ValueError("declare exactly seven unique completed or failed method slots")
    frozen = {label: checkpoint_metadata(label, path, base, extension) for label, path in checkpoints.items()}
    failures = {}
    for label, path in failed_runs.items():
        failed = _lock(path)
        if (failed.get("method") != label or failed.get("stage") != "formal"
                or failed.get("seed") != METHOD_SEEDS[label] or not failed.get("reason")
                or failed.get("failure_kind") not in ("numerical", "resource")):
            raise ValueError("failure slot must be a fixed-seed scientific outcome")
        failures[label] = failed
    window = shared.observation_window(data_file, group, cohort, query_times=QUERY_TIMES)
    output = Path(output).resolve()
    reused = Path(reuse_evaluation).resolve(strict=True) if reuse_evaluation is not None else None
    bindings = {label: {"checkpoint": str(Path(path).resolve()), "sha256": shared.file_sha256(path)} for label, path in checkpoints.items()}
    plan = {"schema": "gift.w1-extended-native.evaluation-plan.v1", "cohort": cohort,
            "checkpoints": bindings, "failures": failures, "base_lock": base, "extension_lock": extension,
            "trajectory_ids": window.trajectory_ids.tolist(), "query_times": QUERY_TIMES.tolist(),
            "reuse_evaluation": str(reused) if reused is not None else None, "data_file": str(window.path), "group": group}
    if output.exists():
        if not resume or _lock(output / "evaluation_plan.json") != plan:
            raise ValueError("extended resume requires the same weights/recipes/cohort")
        if (output / "numeric_report.json").is_file():
            try:
                previous_report = _lock(output / "numeric_report.json")
            except json.JSONDecodeError:
                previous_report = {}
            if _numeric_artifacts_complete(output, previous_report, plan):
                _validate_imported_truth(window, output / "raw/truth.h5")
                render_report(output)
                if plots:
                    draw_figures(output)
                return previous_report
            shared.preserve_interrupted(output / "numeric_report.json")
    else:
        output.mkdir(parents=True)
        (output / "raw").mkdir()
        (output / "metrics").mkdir()
        shared.write_json(output / "evaluation_plan.json", plan)
    reused_report = None
    if reused is not None:
        reused_report = _lock(reused / "numeric_report.json")
        if reused_report["trajectory_ids"] != window.trajectory_ids.tolist() or reused_report["cohort"] != cohort:
            raise ValueError("imported cohort IDs differ")
        np.testing.assert_allclose(reused_report["query_times"], QUERY_TIMES, rtol=0, atol=1e-12)
        _validate_imported_truth(window, reused / "raw/truth.h5")
        _link_immutable(reused / "raw/truth.h5", output / "raw/truth.h5")
    records = {}
    for label in METHOD_SEEDS:
        if label in failures:
            records[label] = {"method": label, "seed": METHOD_SEEDS[label], "status": "training_failed", "reason": failures[label]["reason"],
                              "summary": {"status": "training_failed", "population": 180, "failed_count": 180, "primary_metric": None, "metrics": {}, "failed_ids": window.trajectory_ids.tolist()},
                              "comparison_role": "reduced_data_control" if label == "GIFT-N50" else "common_N300"}
            failure_path = output / "metrics" / (label + ".json")
            if failure_path.exists():
                if _lock(failure_path) != records[label]:
                    raise ValueError("scientific failed slot differs on resume")
            else:
                shared.write_json(failure_path, records[label])
            continue
        path, payload = Path(checkpoints[label]).resolve(), frozen[label]
        raw_path, metric_path = output / "raw" / (label + ".h5"), output / "metrics" / (label + ".h5")
        record_path = output / "metrics" / (label + ".json")
        if reused is not None and label in native.METHOD_SEEDS:
            record = copy.deepcopy(reused_report["methods"][label])
            if record["checkpoint_sha256"] != bindings[label]["sha256"] or record["raw_prediction_sha256"] != shared.file_sha256(reused / record["raw_prediction_file"]):
                raise ValueError("reused model or prediction changed")
            _link_immutable(reused / record["raw_prediction_file"], raw_path)
            _link_immutable(reused / record["metric_file"], metric_path)
            record.update(reused_measurement_from=str(reused), new_inference_performed=False,
                          metric_sha256=shared.file_sha256(metric_path),
                          raw_prediction_file=str(raw_path.relative_to(output)), metric_file=str(metric_path.relative_to(output)))
            with h5py.File(metric_path, "r") as metrics:
                if metrics.attrs.get("status") != "complete":
                    raise ValueError("imported prior measurement has incomplete metric arrays")
        else:
            previous = None
            if raw_path.is_file():
                status, cost = shared.raw_checkpoint_status(raw_path, window, path, payload)
                if status != "complete":
                    previous = shared.preserve_interrupted(raw_path)
            if not raw_path.is_file():
                cost = native.predict_checkpoint(path, payload, window, raw_path, device=device, batch_size=batch_size,
                    previous_raw=previous, loader=lambda p, value, where: load_checkpoint(label, p, value, where), forward=_forward)
            candidate = _metric_record(record_path, metric_path, bindings[label]["sha256"], shared.file_sha256(raw_path))
            if candidate is not None:
                records[label] = candidate
                continue
            if record_path.exists():
                shared.preserve_interrupted(record_path)
            if metric_path.exists():
                shared.preserve_interrupted(metric_path)
            truth_path = output / "raw/truth.h5"
            if not truth_path.is_file():
                # Read future truth only after this autonomous prediction commits.
                _commit_truth(window, truth_path)
            with h5py.File(truth_path, "r") as saved:
                truth = saved["vorticity"][:]
            measured = shared.score_raw_predictions(raw_path, truth, metric_path)
            record = {"method": label, "seed": payload["seed"], "status": measured["summary"]["status"],
                      "checkpoint": str(path), "checkpoint_sha256": bindings[label]["sha256"],
                      "selected_epoch": payload["epoch"], "model_spec": payload["model_spec"], "config": payload["config"],
                      "data_binding": payload["data_binding"], "summary": measured["summary"], "per_trajectory": measured["per_trajectory"],
                      "cost": cost, "new_inference_performed": True, "raw_prediction_sha256": shared.file_sha256(raw_path),
                      "metric_sha256": shared.file_sha256(metric_path),
                      "raw_prediction_file": str(raw_path.relative_to(output)), "metric_file": str(metric_path.relative_to(output)),
                      "source_provenance": payload.get("source_provenance", payload.get("source_metadata"))}
        record["comparison_role"] = "reduced_data_control" if label == "GIFT-N50" else "common_N300"
        if record_path.exists():
            if _lock(record_path) != shared._json_ready(record):
                raise ValueError("completed extended measurement differs on resume")
        else:
            shared.write_json(record_path, record)
        records[label] = record
    if not (output / "raw/truth.h5").is_file():
        # An entirely failed training population still binds its true test cohort.
        _commit_truth(window, output / "raw/truth.h5")
    report = {"schema": "gift.w1-extended-native.evaluation.v1", "status": "numerical_complete", "cohort": cohort,
              "interpretation": ("posthoc_existing_heldout_complete_model_reevaluation" if reused is None else
                                 "existing_independent_heldout_confirmation_known_four_results_before_extension" if cohort == "confirmation" else
                                 "posthoc_exploratory_old_test"),
              "population": 180, "trajectory_ids": window.trajectory_ids.tolist(), "query_times": QUERY_TIMES.tolist(),
              "absolute_times": (5+QUERY_TIMES).tolist(), "time_points": 76, "future_steps": 75, "dt": .04,
              "method_seeds": METHOD_SEEDS, "methods": records, "base_lock": base, "extension_lock": extension,
              "job_events_file": str(job_events) if job_events else None,
              "limitations": ["Single training run per method; no training-seed SD.", "GIFT-N50 is a reduced data and cost control, not the historical GIFT-Lite recipe.",
                              ("The four completed training runs and W1 predictions are reused unchanged." if reused is not None else
                               "All seven existing fixed checkpoints are reevaluated autonomously; no training or model selection is performed."),
                              "Both held-out populations have already been scored; reevaluation is not a new blind test and permits no score-dependent tuning."]}
    shared.write_json(output / "numeric_report.json", report)
    render_report(output)
    if plots:
        draw_figures(output)
    return report


def render_report(output):
    output = Path(output)
    report = _lock(output / "numeric_report.json")
    target, marker = output / "RESULTS.md", output / "REPORT_COMPLETE.json"
    if target.exists() and marker.exists():
        try:
            completion = _lock(marker)
        except json.JSONDecodeError:
            completion = {}
        if (isinstance(completion, dict) and completion.get("numeric_report_sha256") == shared.file_sha256(output / "numeric_report.json")
                and completion.get("results_sha256") == shared.file_sha256(target)):
            return
    for path in (target, marker):
        if path.exists():
            shared.preserve_interrupted(path)
    lines = ["# W1: N300/76 points, single initial field", "", "One training instance per method; future75 excludes the given t5 field.",
             "", "| Method | Role | Future75 relative L2 | t8 relative L2 | Future MSE | Failed/180 |", "|---|---|---:|---:|---:|---:|"]
    def number(value):
        return "unavailable" if value is None else f"{value:.8g}"
    for method, record in report["methods"].items():
        s, m = record["summary"], record["summary"].get("metrics", {})
        lines.append(f"| {DISPLAY_NAMES[method]} | {record['comparison_role']} | {number(s.get('primary_metric'))} | {number(m.get('t8_relative_l2', {}).get('mean'))} | {number(m.get('future_mse', {}).get('mean'))} | {s['failed_count']}/180 |")
    lines += ["", *report["limitations"], "", "Raw predictions and metrics are preserved before rendering figures."]
    target.write_text("\n".join(lines)+"\n", encoding="utf-8")
    shared.write_json(marker, {"numeric_report_sha256": shared.file_sha256(output / "numeric_report.json"), "results_sha256": shared.file_sha256(target)})


def figures_complete(output):
    """Bind completion to the numeric source and actual image bytes."""
    output = Path(output)
    marker = output / "FIGURES_COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        record = _lock(marker)
    except json.JSONDecodeError:
        return False
    if (not isinstance(record, dict) or not record.get("files") or not record.get("file_sha256")
            or record.get("numeric_report_sha256") != shared.file_sha256(output / "numeric_report.json")):
        return False
    return all((output / path).is_file() and record["file_sha256"].get(path) == shared.file_sha256(output / path)
               for path in record["files"])


def draw_figures(output, *, force=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    output = Path(output)
    if not force and figures_complete(output):
        return
    report = _lock(output / "numeric_report.json")
    figures = output / "figures"
    if figures.exists():
        index = 1
        while (output / f"figures_retry_{index:03d}").exists():
            index += 1
        figures = output / f"figures_retry_{index:03d}"
    figures.mkdir()
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for label, record in report["methods"].items():
        if record["status"] == "complete":
            axis.plot(report["absolute_times"][1:], record["summary"]["mean_relative_l2_by_time"][1:],
                      label=DISPLAY_NAMES[label], linestyle="--" if label == "GIFT-N50" else "-")
    axis.set(xlabel="Physical time (s)", ylabel="Mean relative L2 error", title="W1: one initial field; GIFT-N50 is a budget control")
    if axis.lines:
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    else:
        axis.text(.5, .5, "No finite complete method population", ha="center", transform=axis.transAxes)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figures / ("future_error."+ext), dpi=180)
    plt.close(fig)
    keyframes = draw_keyframes(output, figures, report)
    shared.write_json(figures / "provenance.json", {"numeric_report_sha256": shared.file_sha256(output / "numeric_report.json"),
                      "training_seed_shading": False, "t0_excluded": True, "keyframes": keyframes})
    if (output / "FIGURES_COMPLETE.json").exists():
        shared.preserve_interrupted(output / "FIGURES_COMPLETE.json")
    figure_files = [str(p.relative_to(output)) for p in sorted(figures.iterdir())]
    shared.write_json(output / "FIGURES_COMPLETE.json", {"numeric_report_sha256": shared.file_sha256(output / "numeric_report.json"),
                      "files": figure_files, "file_sha256": {path: shared.file_sha256(output / path) for path in figure_files}})


def draw_keyframes(output, figures, report):
    import matplotlib.pyplot as plt
    fixed_id = 100000 if report["cohort"] == "confirmation" else 1040
    indices = np.array([12, 25, 50, 75])
    with h5py.File(output / "raw/truth.h5", "r") as raw:
        row = int(np.flatnonzero(raw["trajectory_index"][:] == fixed_id)[0])
        truth = raw["vorticity"][row, indices]
    predictions = {}
    for label, record in report["methods"].items():
        if "raw_prediction_file" in record:
            with h5py.File(output / record["raw_prediction_file"], "r") as raw:
                row = int(np.flatnonzero(raw["trajectory_index"][:] == fixed_id)[0])
                predictions[label] = raw["prediction"][row, indices]
    limit = max([float(abs(truth).max())] + [float(abs(value)[np.isfinite(value)].max())
                for value in predictions.values() if np.isfinite(value).any()] + [1e-12])
    error_limit = max([float(abs(value-truth)[np.isfinite(value)].max()) for value in predictions.values() if np.isfinite(value).any()] + [1e-12])
    fig, axes = plt.subplots(15, 4, figsize=(11, 29), squeeze=False)
    for col, t in enumerate([5.48, 6, 7, 8]):
        field_image = axes[0, col].imshow(truth[col], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[0, col].set_title(f"t={t:g}")
    axes[0, 0].set_ylabel("Truth")
    for index, (label, record) in enumerate(report["methods"].items()):
        first = 1+2*index
        axes[first, 0].set_ylabel(DISPLAY_NAMES[label]+"\nprediction")
        axes[first+1, 0].set_ylabel("absolute error")
        for col in range(4):
            if label not in predictions:
                axes[first, col].text(.5, .5, "Training failed", ha="center", transform=axes[first, col].transAxes)
                continue
            value = predictions[label][col]
            field_image = axes[first, col].imshow(np.ma.masked_invalid(value), origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            error_image = axes[first+1, col].imshow(np.ma.masked_invalid(abs(value-truth[col])), origin="lower", cmap="magma", vmin=0, vmax=error_limit)
            if not np.isfinite(value).all():
                axes[first, col].text(.5, .5, "Nonfinite prediction", ha="center", transform=axes[first, col].transAxes)
    for axis in axes.flat:
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle(f"W1 prespecified trajectory {fixed_id}; one fixed training run per method", y=.988)
    fig.subplots_adjust(left=.18, right=.85, top=.95, bottom=.02, hspace=.18, wspace=.08)
    fig.colorbar(field_image, cax=fig.add_axes([.88,.6,.015,.3]), label="Vorticity")
    if predictions:
        fig.colorbar(error_image, cax=fig.add_axes([.88,.13,.015,.3]), label="Absolute error")
    for ext in ("png", "svg"):
        fig.savefig(figures / ("fixed_id_keyframes."+ext), dpi=160)
    plt.close(fig)
    return {"trajectory_id": fixed_id, "absolute_time": [5.48,6,7,8], "interpolation": False,
            "method_seeds": METHOD_SEEDS, "selection": "predeclared ID and actual nodes",
            "field_limits": [-limit, limit], "absolute_error_limits": [0., error_limit],
            "colour_limit_rule": "one shared maximum of all finite selected truth/prediction fields and all selected absolute errors; no finite value clipped"}


def labeled_paths(values):
    result = {}
    for value in values:
        label, path = value.split("=", 1)
        if label in result:
            raise ValueError("duplicate method label")
        result[label] = Path(path).resolve(strict=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("base-lock", "extension-lock", "data-file", "output"):
        parser.add_argument("--"+argument, type=Path, required=True)
    parser.add_argument("--reuse-evaluation", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--failed-run", action="append", default=[])
    parser.add_argument("--cohort", choices=("confirmation", "old_test"), required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--job-events", type=Path)
    args = parser.parse_args(argv)
    return evaluate(labeled_paths(args.checkpoint), labeled_paths(args.failed_run), base_lock=args.base_lock,
                    extension_lock=args.extension_lock, data_file=args.data_file, group=args.group, cohort=args.cohort,
                    output=args.output, reuse_evaluation=args.reuse_evaluation, device=args.device, batch_size=args.batch_size,
                    resume=args.resume, plots=not args.skip_plots, job_events=args.job_events)


if __name__ == "__main__":
    main()
