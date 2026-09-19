"""Draw the M2 keyframe and signed-residual comparison from formal raw fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.formal._shared.figure_evidence import (
    bind_result, finish_figures,
)


METHODS = (
    {"label": "GIFT", "slug": "gift", "group": "GIFT_seed_20260820"},
    {"label": "GIFT-Lite", "slug": "gift_lite", "group": "GIFT_Lite_seed_20260820"},
    {"label": "FNO-2D", "slug": "fno2d", "group": "FNO_2D"},
    {"label": "FNO-3D", "slug": "fno3d", "group": "FNO_3D"},
    {"label": "U-NO", "slug": "uno", "group": "U_NO"},
    {"label": "U-Net", "slug": "unet", "group": "U_Net"},
)
KEY_TIMES = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
# Prespecified, fixed colour limits, one pair per initial-condition population.
# The four-vortex pair is the paper's constant. The Gaussian pair is declared in
# S4_PROTOCOL.md because that population's reference field genuinely exceeds 19.
# Neither pair is fitted to the plotted data: a keyframe the prespecified range
# would clip is refused, exactly as in the paper's renderer, instead of the range
# being widened to fit whatever trajectory was selected.
COLOUR_LIMITS = {'M2': (19.0, 21.0), 'S4': (25.0, 25.0)}
FIELD_LIMIT, RESIDUAL_LIMIT = COLOUR_LIMITS['M2']
# The paper's own tick tuples are kept verbatim for its constants; any other
# prespecified range uses five evenly spaced ticks on the same symmetric scale.
PRESET_TICKS = {
    19.0: (-19.0, -10.0, 0.0, 10.0, 19.0),
    21.0: (-21.0, -10.0, 0.0, 10.0, 21.0),
}


def colour_ticks(limit: float) -> tuple[float, ...]:
    return PRESET_TICKS.get(float(limit), tuple(np.linspace(-limit, limit, 5)))
FIELD_CMAP = "RdBu_r"
# Face colour of the tiles that carry no field of their own, as in the manuscript.
PLACEHOLDER_FACE = "#f6f6f7"
RESIDUAL_CMAP = "PuOr"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.0,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    difference = np.asarray(prediction - truth, dtype=np.float32)
    numerator = np.linalg.norm(
        difference.reshape(difference.shape[0], -1).astype(np.float64), axis=1
    )
    denominator = np.linalg.norm(
        truth.reshape(truth.shape[0], -1).astype(np.float64), axis=1
    )
    return numerator / denominator


def time_slug(value: float) -> str:
    return f"T{value:.1f}".replace(".", "p")


def style_field_axis(axis: Any) -> None:
    axis.set_xlim(0.0, 64.0)
    axis.set_ylim(0.0, 64.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_mesh(
    axis: Any,
    values: np.ndarray,
    *,
    cmap: str,
    norm: TwoSlopeNorm,
) -> Any:
    edges = np.arange(65, dtype=np.float64)
    mesh = axis.pcolormesh(
        edges,
        edges,
        values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=False,
        snap=True,
    )
    style_field_axis(axis)
    return mesh


def save_panel_svg(
    path: Path,
    values: np.ndarray,
    *,
    cmap: str,
    norm: TwoSlopeNorm,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite panel: {path}")
    figure = plt.figure(figsize=(1.20, 1.20))
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_mesh(axis, values, cmap=cmap, norm=norm)
    figure.savefig(path, format="svg", bbox_inches=None, pad_inches=0.0)
    plt.close(figure)


def load_keyframes(
    input_path: Path, trajectory_id: int, *, experiment: str = 'M2'
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[int],
]:
    with h5py.File(input_path, "r") as handle:
        if experiment not in ('M2', 'S4') or handle.attrs.get("schema") != f"gift.formal.{experiment}.raw.v2":
            raise RuntimeError("formal prediction raw-result schema differs")
        trajectory_ids = np.asarray(handle["trajectory_ids"][:], dtype=np.int64)
        matches = np.flatnonzero(trajectory_ids == trajectory_id)
        if matches.size != 1:
            raise RuntimeError("requested trajectory is not unique in the formal result")
        trajectory_index = int(matches[0])
        all_times = np.asarray(handle["absolute_times"][:], dtype=np.float64)
        time_indices: list[int] = []
        for value in KEY_TIMES:
            candidates = np.flatnonzero(
                np.isclose(all_times, value, rtol=0.0, atol=1e-12)
            )
            if candidates.size != 1:
                raise RuntimeError(f"key time is not unique: {value}")
            time_indices.append(int(candidates[0]))
        truth = np.asarray(
            handle["truth"][trajectory_index, time_indices], dtype=np.float32
        )
        predictions: dict[str, np.ndarray] = {}
        recorded_errors: dict[str, np.ndarray] = {}
        for spec in METHODS:
            slug = str(spec["slug"])
            group = str(spec["group"])
            predictions[slug] = np.asarray(
                handle[f"{group}/prediction"][trajectory_index, time_indices],
                dtype=np.float32,
            )
            recorded_errors[slug] = np.asarray(
                handle[f"{group}/full_relative_l2"][trajectory_index, time_indices],
                dtype=np.float64,
            )
    if truth.shape != (4, 64, 64) or not np.isfinite(truth).all():
        raise RuntimeError("truth keyframes have invalid shape or nonfinite values")
    residuals: dict[str, np.ndarray] = {}
    calculated_errors: dict[str, np.ndarray] = {}
    for spec in METHODS:
        slug = str(spec["slug"])
        prediction = predictions[slug]
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"{spec['label']} prediction differs")
        residuals[slug] = np.asarray(prediction - truth, dtype=np.float32)
        calculated_errors[slug] = relative_l2(prediction, truth)
        if not np.allclose(
            calculated_errors[slug], recorded_errors[slug], rtol=0.0, atol=1e-14
        ):
            maximum = float(
                np.max(np.abs(calculated_errors[slug] - recorded_errors[slug]))
            )
            raise RuntimeError(
                f"{spec['label']} relative-L2 values differ; maximum={maximum}"
            )
        if not np.array_equal(prediction[0], truth[0]):
            raise RuntimeError(f"{spec['label']} T=5.0 field is not the shared anchor")
    return truth, predictions, residuals, calculated_errors, time_indices


def verify_color_limits(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    *,
    field_limit: float,
    residual_limit: float,
) -> dict[str, float]:
    """Refuse a keyframe that the prespecified colour limits would clip."""
    field_maximum = max(
        [float(np.max(np.abs(truth)))]
        + [float(np.max(np.abs(value))) for value in predictions.values()]
    )
    residual_maximum = max(float(np.max(np.abs(value))) for value in residuals.values())
    if field_maximum > field_limit:
        raise RuntimeError("field color limit would clip a selected keyframe")
    if residual_maximum > residual_limit:
        raise RuntimeError("residual color limit would clip a selected keyframe")
    return {
        "selected_field_maximum_absolute_value": field_maximum,
        "selected_residual_maximum_absolute_value": residual_maximum,
        "field_limit": float(field_limit),
        "residual_limit": float(residual_limit),
    }


def save_source_data(
    output: Path,
    *,
    trajectory_id: int,
    time_indices: list[int],
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "trajectory_id": np.asarray(trajectory_id, dtype=np.int64),
        "absolute_times": KEY_TIMES,
        "formal_report_time_indices": np.asarray(time_indices, dtype=np.int64),
        "truth": truth,
    }
    for spec in METHODS:
        slug = str(spec["slug"])
        arrays[f"prediction_{slug}"] = predictions[slug]
        arrays[f"residual_{slug}"] = residuals[slug]
        arrays[f"relative_l2_{slug}"] = errors[slug]
    np.savez_compressed(output / "source_fields.npz", **arrays)

    with (output / "panel_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "trajectory_id",
                "method",
                "formal_group",
                "absolute_time",
                "relative_l2",
                "field_minimum",
                "field_maximum",
                "residual_minimum",
                "residual_maximum",
            ),
        )
        writer.writeheader()
        for spec in METHODS:
            slug = str(spec["slug"])
            for index, absolute_time in enumerate(KEY_TIMES):
                writer.writerow(
                    {
                        "trajectory_id": trajectory_id,
                        "method": spec["label"],
                        "formal_group": spec["group"],
                        "absolute_time": f"{absolute_time:.1f}",
                        "relative_l2": repr(float(errors[slug][index])),
                        "field_minimum": repr(float(predictions[slug][index].min())),
                        "field_maximum": repr(float(predictions[slug][index].max())),
                        "residual_minimum": repr(float(residuals[slug][index].min())),
                        "residual_maximum": repr(float(residuals[slug][index].max())),
                    }
                )


def save_individual_panels(
    output: Path,
    *,
    trajectory_id: int,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    field_norm: TwoSlopeNorm,
    residual_norm: TwoSlopeNorm,
) -> list[dict[str, Any]]:
    panel_root = output / "panels"
    panel_root.mkdir()
    index_rows: list[dict[str, Any]] = []

    reference_dir = panel_root / "reference"
    reference_dir.mkdir()
    for index, absolute_time in enumerate(KEY_TIMES):
        filename = (
            f"traj{trajectory_id}_reference_scalar_{time_slug(absolute_time)}.svg"
        )
        path = reference_dir / filename
        save_panel_svg(path, truth[index], cmap=FIELD_CMAP, norm=field_norm)
        index_rows.append(
            {
                "method": "Reference",
                "panel_type": "scalar_field",
                "absolute_time": float(absolute_time),
                "relative_path": path.relative_to(output).as_posix(),
            }
        )

    for spec in METHODS:
        slug = str(spec["slug"])
        method_dir = panel_root / slug
        method_dir.mkdir()
        for index, absolute_time in enumerate(KEY_TIMES):
            scalar_path = method_dir / (
                f"traj{trajectory_id}_{slug}_scalar_{time_slug(absolute_time)}.svg"
            )
            residual_path = method_dir / (
                f"traj{trajectory_id}_{slug}_residual_{time_slug(absolute_time)}.svg"
            )
            save_panel_svg(
                scalar_path,
                predictions[slug][index],
                cmap=FIELD_CMAP,
                norm=field_norm,
            )
            save_panel_svg(
                residual_path,
                residuals[slug][index],
                cmap=RESIDUAL_CMAP,
                norm=residual_norm,
            )
            index_rows.extend(
                (
                    {
                        "method": spec["label"],
                        "panel_type": "scalar_field",
                        "absolute_time": float(absolute_time),
                        "relative_path": scalar_path.relative_to(output).as_posix(),
                    },
                    {
                        "method": spec["label"],
                        "panel_type": "signed_residual",
                        "absolute_time": float(absolute_time),
                        "relative_path": residual_path.relative_to(output).as_posix(),
                    },
                )
            )

    with (output / "panel_file_index.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("method", "panel_type", "absolute_time", "relative_path"),
        )
        writer.writeheader()
        writer.writerows(index_rows)
    return index_rows


def _grid_group_bounds(figure: Any, grid: Any, start: int, stop: int) -> Any:
    return grid[0, start:stop].get_position(figure)


def draw_composite(
    output: Path,
    *,
    trajectory_id: int,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    field_norm: TwoSlopeNorm,
    residual_norm: TwoSlopeNorm,
) -> None:
    figure = plt.figure(figsize=(183.0 / 25.4, (134.0 * (len(METHODS) + 1) / 6.0) / 25.4))
    grid = figure.add_gridspec(
        len(METHODS) + 1,
        10,
        left=0.018,
        right=0.989,
        bottom=0.112,
        top=0.874,
        width_ratios=(0.76, 1.0, 1.0, 1.0, 1.0, 0.28, 1.0, 1.0, 1.0, 1.0),
        wspace=0.040,
        hspace=0.060,
    )

    row_labels = ("Reference",) + tuple(str(spec["label"]) for spec in METHODS)
    for row, label in enumerate(row_labels):
        label_axis = figure.add_subplot(grid[row, 0])
        label_axis.set_axis_off()
        label_axis.text(
            0.98,
            0.50,
            label,
            ha="right",
            va="center",
            fontsize=6.25,
            fontweight="semibold",
            color="#171717",
        )

    for time_index in range(4):
        axis = figure.add_subplot(grid[0, time_index + 1])
        draw_mesh(axis, truth[time_index], cmap=FIELD_CMAP, norm=field_norm)

    # The reference row has no prediction error of its own, as the manuscript notes
    # in that block.
    reference_note_axis = figure.add_subplot(grid[0, 6:10])
    reference_note_axis.set_axis_off()
    reference_note_axis.text(
        0.50,
        0.50,
        "Prediction error is defined for model predictions",
        ha="center",
        va="center",
        fontsize=5.2,
        color="#777777",
    )

    for row, spec in enumerate(METHODS, start=1):
        slug = str(spec["slug"])
        for time_index in range(4):
            field_axis = figure.add_subplot(grid[row, time_index + 1])
            draw_mesh(
                field_axis,
                predictions[slug][time_index],
                cmap=FIELD_CMAP,
                norm=field_norm,
            )
            residual_axis = figure.add_subplot(grid[row, time_index + 6])
            draw_mesh(
                residual_axis,
                residuals[slug][time_index],
                cmap=RESIDUAL_CMAP,
                norm=residual_norm,
            )
            residual_axis.text(
                0.035,
                0.035,
                f"rel. L2 = {errors[slug][time_index]:.3f}",
                transform=residual_axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=3.75,
                color="#171717",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.80,
                    "pad": 0.40,
                },
            )

    scalar_bounds = _grid_group_bounds(figure, grid, 1, 5)
    residual_bounds = _grid_group_bounds(figure, grid, 6, 10)
    figure.text(
        0.5 * (scalar_bounds.x0 + scalar_bounds.x1),
        0.934,
        "Vorticity field",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="semibold",
    )
    figure.text(
        0.5 * (residual_bounds.x0 + residual_bounds.x1),
        0.934,
        "Prediction error",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="semibold",
    )
    for time_index in range(4):
        scalar_cell = grid[0, time_index + 1].get_position(figure)
        residual_cell = grid[0, time_index + 6].get_position(figure)
        for cell in (scalar_cell, residual_cell):
            figure.text(
                0.5 * (cell.x0 + cell.x1),
                0.899,
                f"t = {KEY_TIMES[time_index]:.1f}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="#333333",
            )

    separator_y = 0.5 * (
        grid[0, 1].get_position(figure).y0
        + grid[1, 1].get_position(figure).y1
    )
    figure.add_artist(
        Line2D(
            [grid[0, 0].get_position(figure).x0, residual_bounds.x1],
            [separator_y, separator_y],
            transform=figure.transFigure,
            color="#D9D9D9",
            linewidth=0.45,
        )
    )

    field_bar_width = 0.66 * scalar_bounds.width
    field_bar_axis = figure.add_axes(
        [
            0.5 * (scalar_bounds.x0 + scalar_bounds.x1 - field_bar_width),
            0.041,
            field_bar_width,
            0.012,
        ]
    )
    field_bar = figure.colorbar(
        ScalarMappable(norm=field_norm, cmap=FIELD_CMAP),
        cax=field_bar_axis,
        orientation="horizontal",
        ticks=colour_ticks(field_norm.vmax),
    )
    field_bar.set_label("Vorticity field", fontsize=5.4, labelpad=1.3)
    field_bar.ax.xaxis.set_label_position("top")
    field_bar.solids.set_rasterized(False)
    field_bar.ax.tick_params(labelsize=4.6, width=0.5, length=2.0, pad=1.2)
    field_bar.outline.set_linewidth(0.5)

    residual_bar_width = 0.66 * residual_bounds.width
    residual_bar_axis = figure.add_axes(
        [
            0.5 * (residual_bounds.x0 + residual_bounds.x1 - residual_bar_width),
            0.041,
            residual_bar_width,
            0.012,
        ]
    )
    residual_bar = figure.colorbar(
        ScalarMappable(norm=residual_norm, cmap=RESIDUAL_CMAP),
        cax=residual_bar_axis,
        orientation="horizontal",
        ticks=colour_ticks(residual_norm.vmax),
    )
    residual_bar.set_label(
        "Prediction error", fontsize=5.4, labelpad=1.3
    )
    residual_bar.ax.xaxis.set_label_position("top")
    residual_bar.solids.set_rasterized(False)
    residual_bar.ax.tick_params(labelsize=4.6, width=0.5, length=2.0, pad=1.2)
    residual_bar.outline.set_linewidth(0.5)

    figure.suptitle(
        f"Recursive prediction keyframes — trajectory {trajectory_id}",
        x=0.508,
        y=0.982,
        fontsize=8.0,
        fontweight="bold",
    )

    path = output / f"traj{trajectory_id}_recursive_keyframes.svg"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite composite: {path}")
    figure.savefig(path, bbox_inches=None, pad_inches=0.0)
    plt.close(figure)


def verify_vector_svgs(output: Path, expected_panel_count: int) -> dict[str, Any]:
    panel_paths = sorted((output / "panels").rglob("*.svg"))
    if len(panel_paths) != expected_panel_count:
        raise RuntimeError(
            f"individual SVG count differs: {len(panel_paths)} != {expected_panel_count}"
        )
    composite_paths = sorted(output.glob("traj*_recursive_keyframes.svg"))
    if len(composite_paths) != 1:
        raise RuntimeError("composite SVG is not unique")
    all_paths = panel_paths + composite_paths
    embedded_images: list[str] = []
    for path in all_paths:
        text = path.read_text(encoding="utf-8")
        if "<image" in text.lower():
            embedded_images.append(path.relative_to(output).as_posix())
    if embedded_images:
        raise RuntimeError(f"SVG contains embedded raster images: {embedded_images}")
    composite_text = composite_paths[0].read_text(encoding="utf-8")
    if "<text" not in composite_text.lower():
        raise RuntimeError("composite SVG lacks editable text nodes")
    return {
        "individual_svg_count": len(panel_paths),
        "composite_svg_count": 1,
        "embedded_raster_image_count": 0,
        "composite_has_editable_text_nodes": True,
    }


def main() -> None:
    global METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trajectory-id", type=int, default=1045)
    parser.add_argument("--experiment", choices=('M2', 'S4'), default='M2')
    parser.add_argument("--gift-regimes", nargs="+", choices=("GIFT", "GIFT-Lite"),
                        default=["GIFT", "GIFT-Lite"],
                        help="explicit completed regimes; never substitute an unfinished model")
    args = parser.parse_args()
    METHODS = tuple(spec for spec in METHODS
                    if spec["label"] not in ("GIFT", "GIFT-Lite") or spec["label"] in args.gift_regimes)

    input_path = args.input.resolve(strict=True)
    if input_path.name != "predictions.h5" or input_path.parent.name != "raw":
        raise ValueError("use the completed experiment's raw/predictions.h5")
    evidence = bind_result(input_path.parent.parent, args.experiment, ("raw/predictions.h5", "summary/metrics.csv"))
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    truth, predictions, residuals, errors, time_indices = load_keyframes(
        input_path, args.trajectory_id, experiment=args.experiment
    )
    field_limit, residual_limit = COLOUR_LIMITS[args.experiment]
    ranges = verify_color_limits(truth, predictions, residuals,
                                field_limit=field_limit, residual_limit=residual_limit)
    field_limit, residual_limit = ranges["field_limit"], ranges["residual_limit"]
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )

    save_source_data(
        output,
        trajectory_id=args.trajectory_id,
        time_indices=time_indices,
        truth=truth,
        predictions=predictions,
        residuals=residuals,
        errors=errors,
    )
    panel_index = save_individual_panels(
        output,
        trajectory_id=args.trajectory_id,
        truth=truth,
        predictions=predictions,
        residuals=residuals,
        field_norm=field_norm,
        residual_norm=residual_norm,
    )
    draw_composite(
        output,
        trajectory_id=args.trajectory_id,
        truth=truth,
        predictions=predictions,
        residuals=residuals,
        errors=errors,
        field_norm=field_norm,
        residual_norm=residual_norm,
    )

    vector_qa = verify_vector_svgs(output, expected_panel_count=len(KEY_TIMES) * (1 + 2 * len(METHODS)))
    project_root = Path(__file__).resolve().parents[3]
    try:
        input_record = input_path.relative_to(project_root).as_posix()
    except ValueError:
        input_record = "raw/predictions.h5"
    metadata = {
        "schema": f"gift.paper-figure.{args.experiment}.keyframes.v1",
        "status": "complete",
        "trajectory_id": args.trajectory_id,
        "absolute_times": KEY_TIMES.tolist(),
        "input": {
            "record": input_record,
            "bytes": int(input_path.stat().st_size),
            "sha256": sha256_file(input_path),
        },
        "methods": [
            {
                "label": spec["label"],
                "panel_slug": spec["slug"],
                "formal_group": spec["group"],
            }
            for spec in METHODS
        ],
        "gift_visualization": {
            "training_seed": 20260820,
            "selection": "prespecified training seed; not selected from evaluation errors",
            "field_level_seed_averaging": False,
        },
        "residual_definition": "prediction_minus_reference",
        "color_scales": {
            "scalar_field": {
                "colormap": FIELD_CMAP,
                "normalization": "linear_centered_at_zero",
                "minimum": -field_limit,
                "maximum": field_limit,
            },
            "signed_residual": {
                "colormap": RESIDUAL_CMAP,
                "normalization": "linear_centered_at_zero",
                "minimum": -residual_limit,
                "maximum": residual_limit,
            },
            **ranges,
        },
        "panel_count": len(panel_index),
        "vector_qa": vector_qa,
        "layout": {
            "archetype": "image_plate_with_residual_evidence",
            "arrangement": "scalar_left_residual_right",
            "row_order": ["Reference", *[spec["label"] for spec in METHODS]],
            "scalar_columns": KEY_TIMES.tolist(),
            "residual_columns": KEY_TIMES.tolist(),
            "scalar_residual_rows_aligned": True,
            "reference_residual_displayed": False,
            "reference_repeated_in_composite": False,
            "individual_reference_panels_stored_once": True,
            "method_name_position": "far_left_centered_on_aligned_scalar_residual_row",
            "column_group_titles": ["Vorticity field", "Prediction error"],
        },
    }
    with (output / "figure_metadata.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Commit one manifest after all SVG display hints are applied. Plotting code
    # and usage instructions live in the project, not as duplicate result files.
    finish_figures(output, evidence, Path(__file__), {
        "trajectory_id": args.trajectory_id, "gift_seed": 20260820,
        "selection": "prespecified trajectory and training seed",
        "color_scales": metadata["color_scales"], "crop": "none", "smoothing": "none",
    })
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "individual_vector_panels": len(panel_index),
                "field_limit": field_limit,
                "residual_limit": residual_limit,
                "vector_qa": vector_qa,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
