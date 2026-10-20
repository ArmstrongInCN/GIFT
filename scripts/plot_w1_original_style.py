"""Plot completed single-run W1 data with the original paper renderers.

This is a CPU-only adapter for separate per-method HDF5 files. It retains the
original seven-node PCHIP curves and reference/field/signed-residual plate.
Original measurements, weights, and previously completed figure packages are
never written. ANIE's real initial reconstruction is deliberately retained.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import h5py
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from experiments.formal.w1_recursive_prediction import plot_mean_error as curves
from experiments.formal.w1_recursive_prediction import plot_keyframes as plates
from experiments.formal._shared.figure_evidence import finish_figures, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT.parent/"runs"/"w1w2_native_single76_n300_v6_20261005"/"w1_evaluation"
DEFAULT_OUTPUT = ROOT.parent/"runs"/"w1w2_original_style_figures_20261005"/"W1"
PINNED_COMMIT = "eff64d2682dc856e8b35aaae3b7b37320a3f301f"
SPECS = (
    {"label": "GIFT", "slug": "gift", "group": "GIFT"},
    {"label": "GIFT-N50", "slug": "gift_n50", "group": "GIFT-N50"},
    {"label": "FNO-2D", "slug": "fno2d", "group": "fno2d"},
    {"label": "FNO-3D", "slug": "fno3d", "group": "fno3d"},
    {"label": "U-NO", "slug": "uno", "group": "uno"},
    {"label": "U-Net", "slug": "unet", "group": "unet"},
    {"label": "ANIE", "slug": "anie", "group": "anie_ns"},
)
REQUESTED_TIMES = np.asarray([5., 5.5, 6., 6.5, 7., 7.5, 8.])
FIELD_LIMIT, RESIDUAL_LIMIT = 19., 22.


def nearest_times(times, requested):
    """Map to real observation nodes, selecting the earlier node at a midpoint."""
    indices = []
    for value in requested:
        difference = np.abs(times-value)
        candidates = np.flatnonzero(difference <= difference.min()+2e-12)
        indices.append(int(candidates[0]))
    return np.asarray(indices, dtype=np.int64)


def renderer_sources():
    """Bind original committed renderer bytes and actual parameterized functions."""
    records = []
    for module, function in ((curves, "draw"), (plates, "draw_composite")):
        path = Path(module.__file__).resolve()
        relative = path.relative_to(ROOT).as_posix()
        original = subprocess.check_output(["git", "show", PINNED_COMMIT+":"+relative], cwd=ROOT)
        text = path.read_text(encoding="utf-8")
        node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == function)
        function_text = ast.get_source_segment(text, node)
        records.append({"path": str(path), "sha256": sha256(path), "original_commit": PINNED_COMMIT,
                        "original_committed_file_sha256": hashlib.sha256(original).hexdigest().upper(),
                        "function": function, "function_source_utf8_lf_sha256": hashlib.sha256(function_text.encode()).hexdigest().upper()})
    return records


def resolve_raw(directory, record):
    """Follow each report's recorded file; bind the source of reused measurements."""
    declared = Path(record["raw_prediction_file"])
    path = declared if declared.is_absolute() else directory/declared
    if not path.exists() and record.get("reused_measurement_from"):
        path = Path(record["reused_measurement_from"])/declared
    return path.resolve(strict=True)


def file_evidence(path, expected=None):
    """Read-only byte verification, including whole HDF5 files, before plotting."""
    digest = sha256(path)
    if expected is not None and digest.lower() != expected.lower():
        raise ValueError(f"recorded raw SHA256 differs: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest,
            "whole_file_sha256_verified": True}


def read_cohort(directory, cohort, trajectory_id):
    """Load summary means and only the predeclared trajectory's four raw frames."""
    report_path = directory/"numeric_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "numerical_complete" or report["population"] != 180 or report["cohort"] != cohort:
        raise ValueError("W1 figure requires the completed full180 report")
    times = np.asarray(report["absolute_times"], dtype=np.float64)
    ids = np.asarray(report["trajectory_ids"], dtype=np.int64)
    if times.shape != (76,) or not np.allclose(times, 5+np.arange(76)*.04, rtol=0, atol=2e-12):
        raise ValueError("W1 recorded output grid differs from76 actual nodes")
    selected = np.flatnonzero(ids == trajectory_id)
    if selected.size != 1 or int(selected[0]) != 5:
        raise ValueError("predeclared keyframe ID must be unique and sixth")
    row = int(selected[0])
    report_indices = nearest_times(times, REQUESTED_TIMES)
    key_indices = nearest_times(times, plates.KEY_TIMES)
    if report_indices.tolist() != [0, 12, 25, 37, 50, 62, 75] or key_indices.tolist() != [0, 25, 50, 75]:
        raise ValueError("locked W1 time mapping differs")
    truth_path = (directory/"raw"/"truth.h5").resolve(strict=True)
    sources = [file_evidence(truth_path)]
    with h5py.File(truth_path, "r") as handle:
        if (handle.attrs.get("status") != "complete" or handle["vorticity"].shape != (180, 76, 64, 64)
                or not np.array_equal(handle["trajectory_index"][:], ids)
                or not np.allclose(handle["absolute_time"][:], times, rtol=0, atol=2e-12)):
            raise ValueError("truth raw population/grid differs")
        truth = np.asarray(handle["vorticity"][row, key_indices], dtype=np.float32)
    predictions, residuals, errors, curve_rows, provenance = {}, {}, {}, [], []
    for spec in SPECS:
        key, label, slug = spec["group"], spec["label"], spec["slug"]
        record = report["methods"][key]
        raw = resolve_raw(directory, record)
        sources.append(file_evidence(raw, record["raw_prediction_sha256"]))
        with h5py.File(raw, "r") as handle:
            if (handle.attrs.get("status") != "complete" or handle["prediction"].shape != (180, 76, 64, 64)
                    or not np.array_equal(handle["trajectory_index"][:], ids)
                    or not np.allclose(handle["absolute_time"][:], times, rtol=0, atol=2e-12)):
                raise ValueError(f"{label} raw population/grid differs")
            prediction = np.asarray(handle["prediction"][row, key_indices], dtype=np.float32)
            checkpoint_attribute = str(handle.attrs.get("checkpoint", ""))
        if key != "anie_ns" and not np.array_equal(prediction[0], truth[0]):
            raise ValueError(f"{label} no longer shares its recorded initial anchor")
        predictions[slug] = prediction
        residuals[slug] = np.asarray(prediction-truth, dtype=np.float32)
        errors[slug] = plates.relative_l2(prediction, truth)
        summary = record["summary"]
        if summary["population"] != 180:
            raise ValueError(f"{label} summary population differs")
        means = summary["mean_relative_l2_by_time"]
        if len(means) != 76:
            raise ValueError(f"{label} summary time count differs")
        complete = summary["failed_count"] == 0
        for requested, index in zip(REQUESTED_TIMES, report_indices):
            mean = means[index]
            if complete and (mean is None or not np.isfinite(mean) or mean < 0):
                raise ValueError(f"{label} complete-population mean is unavailable")
            curve_rows.append({"cohort": cohort, "method": label, "raw_method_key": key,
                               "requested_absolute_time": float(requested), "absolute_time": float(times[index]),
                               "observation_index": int(index), "mean_relative_l2": mean if complete else None,
                               "training_seed_sd": "", "n_training_seeds": 1,
                               "training_seed": record["seed"], "n_test_trajectories": 180,
                               "finite_count_min": 180-int(summary["failed_count"]), "complete_population": complete})
        provenance.append({"label": label, "raw_method_key": key, "panel_slug": slug,
                           "recorded_raw_prediction_file": record["raw_prediction_file"],
                           "actual_read_path": str(raw), "raw_prediction_sha256": sources[-1]["sha256"],
                           "reused_measurement_from": record.get("reused_measurement_from"),
                           "new_inference_performed_in_measurement": record["new_inference_performed"],
                           "checkpoint_from_report": record["checkpoint"], "checkpoint_from_raw_attribute": checkpoint_attribute,
                           "checkpoint_sha256_recorded": record["checkpoint_sha256"], "seed": record["seed"]})
    ranges = plates.verify_color_limits(truth, predictions, residuals, field_limit=FIELD_LIMIT, residual_limit=RESIDUAL_LIMIT)
    if ranges["field_limit"] != FIELD_LIMIT or ranges["residual_limit"] != RESIDUAL_LIMIT:
        raise ValueError("a selected raw field exceeds the prespecified19/22 color scales")
    return {"report_path": report_path, "report_sha256": sha256(report_path), "report": report,
            "trajectory_id": trajectory_id, "times": times, "report_indices": report_indices,
            "key_indices": key_indices, "truth": truth, "predictions": predictions,
            "residuals": residuals, "errors": errors, "curve_rows": curve_rows,
            "raw_provenance": provenance, "sources": sources, "ranges": ranges}


def plot_cohort(output, cohort, data, source_records, shared_y):
    """Save numeric sources first, then original SVG and same-figure PNG exports."""
    output.mkdir(parents=True, exist_ok=False)
    curves.write_source(output/"source_data.csv", data["curve_rows"])
    plates.save_source_data(output, trajectory_id=data["trajectory_id"], time_indices=data["key_indices"].tolist(),
                            truth=data["truth"], predictions=data["predictions"], residuals=data["residuals"],
                            errors=data["errors"], methods=SPECS)
    styles = {key: value for key, value in curves.STYLES.items() if key != "GIFT-Lite"}
    styles["GIFT-N50"] = curves.STYLES["GIFT-Lite"]
    styles["ANIE"] = ("#D62728", "X", "-", 1.6, 7)
    labels = tuple(spec["label"] for spec in SPECS)
    # rc_context prevents curve styling from changing the original plate's
    # independently configured defaults in this combined plotting process.
    with mpl.rc_context():
        figure = curves.draw(data["curve_rows"], methods=labels, y_limit=shared_y, styles=styles,
                              draw_order=("FNO-3D", "U-Net", "FNO-2D", "U-NO", "ANIE", "GIFT-N50", "GIFT"),
                              report_times=data["times"][data["report_indices"]])
        curve_axes = len(figure.axes)
        curve_size_mm = (figure.get_size_inches()*25.4).tolist()
        curve_y_limits = list(figure.axes[0].get_ylim())
        figure.savefig(output/"mean_relative_l2_vs_time.svg", bbox_inches="tight")
        figure.savefig(output/"mean_relative_l2_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close(figure)
    plates.draw_composite(output, trajectory_id=data["trajectory_id"], truth=data["truth"],
                          predictions=data["predictions"], residuals=data["residuals"], errors=data["errors"],
                          field_norm=TwoSlopeNorm(vmin=-FIELD_LIMIT, vcenter=0., vmax=FIELD_LIMIT),
                          residual_norm=TwoSlopeNorm(vmin=-RESIDUAL_LIMIT, vcenter=0., vmax=RESIDUAL_LIMIT),
                          methods=SPECS, field_ticks=(-19., -10., 0., 10., 19.),
                          residual_ticks=(-22., -10., 0., 10., 22.), save_png=True)
    plate = output/f"traj{data['trajectory_id']}_recursive_keyframes.svg"
    plate_axis_count = len(re.findall(r'<g id="axes_[0-9]+">', plate.read_text(encoding="utf-8")))
    if curve_axes != 1 or plate_axis_count != 74:
        raise RuntimeError("original one-axis curve or eight-row plate structure differs")
    evidence = {"experiment": "W1", "cohort": cohort, "report_sha256": data["report_sha256"],
                "numeric_report_path": str(data["report_path"]), "sources": data["sources"],
                "integrity_verified": True, "experiment_inputs": {"raw_provenance": data["raw_provenance"]}}
    details = {"cohort": cohort, "trajectory_count": 180, "trajectory_id": data["trajectory_id"],
               "renderer_sources": source_records, "methods": list(labels), "method_specs": list(SPECS),
               "requested_report_times": REQUESTED_TIMES.tolist(),
               "actual_report_times": data["times"][data["report_indices"]].tolist(),
               "report_time_indices": data["report_indices"].tolist(), "key_times": plates.KEY_TIMES.tolist(),
               "key_time_indices": data["key_indices"].tolist(), "source_rows": len(data["curve_rows"]),
               "curve": "seven actual measured nodes;601-point PCHIP visual guide; no additional evaluations",
               "curve_canvas_mm": curve_size_mm, "curve_axis_count": curve_axes, "curve_y_limits": curve_y_limits,
               "legend": "original top,ncol3", "styles": styles, "error_bars": "none; one training model per method",
               "primary_metric": "unchanged future75 mean per-frame relative L2; known initial frame excluded",
               "layout": {"row_order": ["Reference", *labels], "axis_count": plate_axis_count,
                          "field_columns": plates.KEY_TIMES.tolist(), "signed_residual_columns": plates.KEY_TIMES.tolist(),
                          "arrangement": "scalar_left_residual_right", "origin": "lower; original pcolormesh(values),ylim(0,64)",
                          "reference_residual": "original grey zero placeholders", "two_horizontal_colorbars": True},
               "color_scales": {"field_limit": FIELD_LIMIT, "residual_limit": RESIDUAL_LIMIT,
                                "field_cmap": plates.FIELD_CMAP, "residual_cmap": plates.RESIDUAL_CMAP,
                                "residual_definition": "signed_prediction_minus_truth", "clipped_values": 0, **data["ranges"]},
               "adaptations": ["explicit additional ANIE style/row; GIFT-N50 reuses original Lite light-blue style",
                               "actual .04-node ticks and markers; unavailable half-second nodes use nearest,ties earlier",
                               "per-method raw-file adapter with full hashes; only ANIE exempted from t5==truth check",
                               "locked shared residual22 instead of21 because old-test FNO3D exceeds21; no clipping",
                               "PNG saved from same original figure; original renderer defaults preserved"],
               "anie_initial_relative_l2": (float(data["errors"]["anie"][0])
                                             if np.isfinite(data["errors"]["anie"][0]) else None),
               "inference_performed_for_plotting": False}
    finish_figures(output, evidence, Path(__file__), details)
    print(json.dumps({"status": "complete", "cohort": cohort, "output": str(output),
                      "axis_count": plate_axis_count, "actual_report_times": details["actual_report_times"],
                      "anie_initial_relative_l2": details["anie_initial_relative_l2"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    cohorts = (("confirmation", 100005), ("old_test", 1045))
    for cohort, _ in cohorts:
        if (args.output/cohort).exists():
            raise FileExistsError(f"refusing to overwrite any existing cohort package:{args.output/cohort}")
    sources = renderer_sources()
    measured = {cohort: read_cohort(args.input/cohort, cohort, trajectory_id) for cohort, trajectory_id in cohorts}
    shared_y = max(1.28, 1.08*max(row["mean_relative_l2"] for data in measured.values()
                                for row in data["curve_rows"] if row["mean_relative_l2"] is not None))
    for cohort, _ in cohorts:
        plot_cohort(args.output/cohort, cohort, measured[cohort], sources, shared_y)


if __name__ == "__main__":
    main()
