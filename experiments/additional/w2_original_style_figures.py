"""Bridge frozen single-model W2 results into the original manuscript plot functions.

This CPU-only renderer loads saved metrics/fields, never models. The legacy
three-panel curves and two-row N128 field/signed-residual plate are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from experiments.formal.w2_cross_resolution import plot_results as legacy
from experiments.formal._shared.figure_evidence import finish_figures, sha256


METHODS = ("GIFT", "GIFT-N50", "FNO-2D", "FNO-3D", "U-NO", "U-Net")
KEYS = ("GIFT", "GIFT-N50", "fno2d", "fno3d", "uno", "unet")
GRIDS = (64, 96, 128)
SAMPLE_INDICES = np.array([0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25])
EXPECTED_TIMES = np.array([5, 5.08, 5.2, 5.28, 5.4, 5.48, 5.6, 5.68, 5.8, 5.88, 6])
FIXED_IDS = {"confirmation": 100110, "old_test": 1150}
ORIGINAL_TEMPLATE_SHA256 = "BE3B0770E9AE8C17D46C620EF7E0000DC1C1E0701D5916DBD0D33DE7415B39DF"
COLORS = {**legacy.COLORS, "GIFT-N50": legacy.COLORS["GIFT-Lite"],
          "U-NO": "#29A82C", "U-Net": "#FF7A00"}
MARKERS = {**legacy.MARKERS, "GIFT-N50": "o", "U-NO": "^", "U-Net": "v"}
LINESTYLES = {**legacy.LINESTYLES, "GIFT-N50": "--", "U-NO": "-", "U-Net": "-"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def source_record(path: Path, root: Path, expected: str | None = None) -> dict:
    digest = sha256(path)
    require(expected is None or digest == expected.upper(), f"source hash changed: {path}")
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest}


def load_frozen_data(root: Path, cohort: str) -> tuple[dict, dict, dict]:
    """Check completed audits and map all 180 trajectories to legacy plotting keys."""
    result = root / "w2_evaluation" / cohort
    report_path = result / "numeric_report.json"
    report = read_json(report_path)
    plan = read_json(result / "evaluation_plan.json")
    result_audit_path = root / "FINAL_RESULT_AUDIT.json"
    training_audit_path = root / "FINAL_TRAINING_AUDIT.json"
    audit = read_json(result_audit_path)
    training_audit = read_json(training_audit_path)
    require(audit["status"] == training_audit["status"] == "passed", "final audits must pass")
    require(sha256(report_path) == audit["identity"]["numeric_reports"][f"w2_{cohort}"].upper(),
            "numeric report changed after independent audit")
    require(report["status"] == "numerical_complete", "W2 numeric evaluation is incomplete")
    require(report["population_per_grid"] == 180, "expected complete 180-trajectory population")
    require(not report["ANIE_included"] and not plan["ANIE_included"], "ANIE must not appear in W2")
    require(plan["training_grid"] == 64 and not plan["target_grid_retraining"]
            and not plan["target_grid_fitting"] and not plan["test_statistics_used"],
            "W2 must use frozen N64 training only")
    require(set(plan["methods"]) == set(KEYS), "W2 method inventory differs")
    times = np.asarray(report["absolute_times"], dtype=np.float64)
    np.testing.assert_allclose(times, 5.0 + np.arange(26) * .04, rtol=0, atol=1e-14)
    np.testing.assert_allclose(times[SAMPLE_INDICES], EXPECTED_TIMES, rtol=0, atol=1e-14)
    ids = np.asarray(report["trajectory_ids"], dtype=np.int64)
    require(len(ids) == 180 and len(np.unique(ids)) == 180, "trajectory identity is incomplete")
    trajectory_id = FIXED_IDS[cohort]
    matches = np.flatnonzero(ids == trajectory_id)
    require(matches.tolist() == [110], "fixed example must preserve original population row 110")
    selected_row = 110
    data: dict[str, Any] = {"times": times[SAMPLE_INDICES], "curves": {}, "curve_seeds": {},
                           "fields": {}, "residual_errors": {}}
    rows, sources, model_seeds, weights, training_counts = [], [], {}, {}, {}
    measurements = audit["measurements"][f"w2_{cohort}"]
    for grid in GRIDS:
        require(set(report["methods"][str(grid)]) == set(KEYS), "grid method inventory differs")
        data["curves"][grid] = {}
        data["curve_seeds"][grid] = {}
        for key, display in zip(KEYS, METHODS):
            rec = report["methods"][str(grid)][key]
            binding = training_audit["method_slots"][key]
            require(rec["status"] == "complete" and rec["summary"]["failed_count"] == 0,
                    "this legacy curve package requires the audited full finite population")
            require(rec["seed"] == binding["seed"] and
                    rec["checkpoint_sha256"].lower() == binding["checkpoint_sha256"].lower(),
                    "metric model identity differs from audited frozen training")
            metric_path = result / rec["metric_file"]
            sources.append(source_record(metric_path, result, measurements[str(grid)][key]["metric_sha256"]))
            with h5py.File(metric_path, "r") as h:
                require(h.attrs["status"] == "complete", "metric marker is incomplete")
                np.testing.assert_array_equal(h["trajectory_id"][:], ids)
                curve_population = np.asarray(h["relative_l2_by_time"][:], dtype=np.float64)
                require(curve_population.shape == (180, 26) and np.isfinite(curve_population).all()
                        and not np.asarray(h["failed"][:]).any(), "curve excludes or fails population members")
                curve_mean = curve_population.mean(axis=0)
                np.testing.assert_allclose(curve_mean, rec["summary"]["mean_relative_l2_by_time"],
                                           rtol=1e-12, atol=1e-14)
                if grid == 128:
                    data["residual_errors"][display] = float(h["t6_relative_l2"][selected_row])
            curve = curve_mean[SAMPLE_INDICES]
            data["curves"][grid][display] = curve
            if display in ("GIFT", "GIFT-N50"):
                data["curve_seeds"][grid][display] = curve[None, :]
            model_seeds[display] = int(rec["seed"])
            weights[display] = rec["checkpoint_sha256"]
            training_counts[display] = int(binding["training_count"])
            for column, time_value in enumerate(data["times"]):
                rows.append({"method": display, "method_key": key, "resolution": grid,
                             "absolute_time": float(time_value), "mean_relative_l2": float(curve[column]),
                             "raw_time_index": int(SAMPLE_INDICES[column]), "population": 180,
                             "failed_count": 0, "training_seed": int(rec["seed"]),
                             "training_count": int(binding["training_count"]),
                             "seed_count": 1, "checkpoint_sha256": rec["checkpoint_sha256"]})
    truth_path = result / "raw" / "N128_truth.h5"
    sources.append(source_record(truth_path, result, measurements["128"]["GIFT"]["truth_sha256"]))
    with h5py.File(truth_path, "r") as h:
        require(h.attrs["status"] == "complete", "truth marker is incomplete")
        np.testing.assert_array_equal(h["trajectory_index"][:], ids)
        np.testing.assert_allclose(h["absolute_time"][:], times, rtol=0, atol=1e-14)
        data["fields"]["Reference"] = h["vorticity"][selected_row, -1]
    for key, display in zip(KEYS, METHODS):
        rec = report["methods"]["128"][key]
        path = result / rec["raw_prediction_file"]
        sources.append(source_record(path, result, measurements["128"][key]["raw_sha256"]))
        with h5py.File(path, "r") as h:
            require(h.attrs["status"] == "complete" and h.attrs["completed_trajectories"] == 180,
                    "prediction population is incomplete")
            require(h.attrs["method"] == key and
                    h.attrs["checkpoint_sha256"].lower() == weights[display].lower(),
                    "raw field identity differs from curve identity")
            np.testing.assert_array_equal(h["trajectory_index"][:], ids)
            np.testing.assert_allclose(h["absolute_time"][:], times, rtol=0, atol=1e-14)
            field = h["prediction"][selected_row, -1]
        require(field.shape == (128, 128) and np.isfinite(field).all(), "fixed field is invalid")
        truth64 = data["fields"]["Reference"].astype(np.float64)
        # The new audited scorer subtracts after conversion to float64. Keeping
        # that precision here binds the legacy caption to the saved metric.
        score = np.linalg.norm(field.astype(np.float64) - truth64) / np.linalg.norm(truth64)
        np.testing.assert_allclose(score, data["residual_errors"][display], rtol=1e-12, atol=1e-14)
        data["fields"][display] = field
    data["source_rows"] = pd.DataFrame(rows)
    metadata = {"cohort": cohort, "trajectory_id": trajectory_id, "selected_population_row": selected_row,
                "training_seeds": model_seeds, "checkpoint_sha256": weights,
                "training_counts": training_counts, "interpretation": report["interpretation"]}
    evidence = {"experiment": "W2", "report_sha256": sha256(report_path), "sources": sources,
                "source_root": str(result), "integrity_verified": True,
                "independent_result_audit_sha256": sha256(result_audit_path),
                "training_audit_sha256": sha256(training_audit_path),
                "evaluation_plan_sha256": sha256(result / "evaluation_plan.json"),
                "scientific_predictions_and_metrics_unchanged": True}
    return data, metadata, evidence


def separate_endpoint_labels(figure) -> dict:
    """Keep the original endpoint annotations, with minimal vertical offsets for collisions."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    offsets = {}
    for grid, axis in zip(GRIDS, figure.axes):
        annotations = [text for text in axis.texts if hasattr(text, "xy")]
        ranked = sorted(annotations, key=lambda a: axis.transData.transform(a.xy)[1])
        original = np.array([axis.transData.transform(a.xy)[1] for a in ranked])
        centers = original.copy()
        heights = np.array([a.get_window_extent(renderer).height for a in ranked])
        lower, upper = axis.bbox.y0 + heights.max()/2, axis.bbox.y1 - heights.max()/2
        for _ in range(64):
            before = centers.copy()
            for index in range(1, len(centers)):
                gap = (heights[index-1] + heights[index])/2 + 1.5
                deficit = gap - (centers[index] - centers[index-1])
                if deficit > 0:
                    centers[index-1] -= deficit/2
                    centers[index] += deficit/2
            centers = np.clip(centers, lower, upper)
            if np.max(np.abs(centers-before)) < 1e-6:
                break
        offsets[str(grid)] = {}
        for a, old_y, new_y in zip(ranked, original, centers):
            offset = float((new_y-old_y) * 72/figure.dpi)
            a.set_position((3.0, offset))
            offsets[str(grid)][a.get_text()] = offset
    return offsets


def expanded_field_width_mm() -> float:
    """Add two columns while preserving the original field-cell width and row structure."""
    def factor(method_count):
        ncols = method_count + 3
        ratio_sum = .6006 + method_count + 1 + .045
        return (ncols + .085*(ncols-1)) / ncols * ratio_sum
    return 205.17 * factor(len(METHODS)) / factor(len(legacy.METHODS))


def render(root: Path, cohort: str, output: Path) -> dict:
    require(not output.exists(), "output already exists; choose a new figure directory")
    data, metadata, evidence = load_frozen_data(root, cohort)
    output.mkdir(parents=True, exist_ok=False)
    data["source_rows"].to_csv(output / "source_data.csv", index=False)
    np.savez_compressed(output / "source_fields.npz", **data["fields"],
                        trajectory_id=metadata["trajectory_id"], population_row=110,
                        absolute_time=6.0, resolution=128,
                        methods=np.array(METHODS),
                        training_seeds_json=json.dumps(metadata["training_seeds"]),
                        checkpoint_sha256_json=json.dumps(metadata["checkpoint_sha256"]))
    curves = legacy.draw_time_curves_only(
        data, methods=METHODS, gift_regimes=("GIFT", "GIFT-N50"),
        colors=COLORS, markers=MARKERS, linestyles=LINESTYLES,
        x_ticks=[5.08, 5.4, 5.68, 6.0],
    )
    label_offsets = separate_endpoint_labels(curves)
    curve_axes = [{"scale": a.get_yscale(), "limits": list(a.get_ylim())} for a in curves.axes]
    for suffix in ("svg", "png"):
        curves.savefig(output / f"cross_resolution_mean_relative_l2.{suffix}",
                       bbox_inches="tight", facecolor="white", dpi=300)
    legacy.plt.close(curves)
    width_mm = expanded_field_width_mm()
    fields, limits = legacy.draw_prediction_fields_only(
        data, metadata["trajectory_id"], metadata["training_seeds"]["GIFT"],
        methods=METHODS, width_mm=width_mm,
    )
    for suffix in ("svg", "png"):
        fields.savefig(output / f"traj{metadata['trajectory_id']}_n128_t6_prediction_fields.{suffix}",
                       bbox_inches="tight", facecolor="white", dpi=300)
    legacy.plt.close(fields)
    function_hashes = {name: hashlib.sha256(inspect.getsource(getattr(legacy, name)).encode()).hexdigest().upper()
                       for name in ("draw_time_curves_only", "draw_prediction_fields_only", "draw_vector_field")}
    details = {**metadata, "methods": list(METHODS), "native_method_keys": list(KEYS),
               "population_per_curve": 180, "single_model_per_method": True,
               "seed_shading": "none; one actual training seed per method",
               "metric_units": "relative L2 ratio; not percentage",
               "curve_grid": "original 1x3 N64/N96/N128", "field_grid": "original 2 rows; reference+6 methods",
               "curve_sample_indices": SAMPLE_INDICES.tolist(), "curve_actual_absolute_times": data["times"].tolist(),
               "time_selection": "nearest original .1-second nodes, ties earlier; real saved .04 nodes only",
               "t0_plotted": False, "t0_in_primary_metric": False,
               "curve_axes": curve_axes, "endpoint_label_y_offsets_points": label_offsets,
               "endpoint_label_policy": "original direct endpoint labels; vertical text spacing only, anchors unchanged",
               "field_canvas_width_mm": width_mm, "field_canvas_height_mm": 96.0,
               "field_width_policy": "two added method columns; original cell width, height, row/column ratios retained",
               "signed_residual_definition": "prediction minus truth",
               "field_rendering": "original upper-origin vector pcolormesh; no smoothing/interpolation/rasterization",
               "field_colormap": "RdBu_r", "residual_colormap": "PuOr",
               "colour_scale_policy": "original max(preferred,ceil(observed)), shared per quantity, no clipping",
               **limits, "field_caption_relative_l2": data["residual_errors"],
               "template_source": "experiments/formal/w2_cross_resolution/plot_results.py",
               "template_source_sha256": sha256(Path(legacy.__file__)),
               "template_before_parameterization_sha256": ORIGINAL_TEMPLATE_SHA256,
               "template_function_source_sha256": function_hashes,
               "template_changes": "optional method/style/tick/canvas-width arguments; defaults preserve old rendering; single-seed bands omitted",
               "model_loading_or_inference": False, "science_run_mutated": False,
               "png_dpi": 300, "visual_qa": "pending actual PNG inspection"}
    finish_figures(output, evidence, Path(__file__), details)
    return {"status": "complete", "output": str(output), "cohort": cohort,
            "trajectory_id": metadata["trajectory_id"], "field_limits": limits,
            "methods": list(METHODS), "svg_embedded_images": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="completed v6 scientific run")
    parser.add_argument("--cohort", choices=tuple(FIXED_IDS), required=True)
    parser.add_argument("--output", type=Path, required=True, help="new original-style figure directory")
    args = parser.parse_args()
    print(json.dumps(render(args.root.resolve(strict=True), args.cohort, args.output.resolve()),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
