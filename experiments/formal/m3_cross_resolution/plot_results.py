"""Render the M3 time curves and prespecified N128 field example as vector SVG."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, SymmetricalLogLocator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.formal._shared.common import relative_l2
from experiments.formal._shared.figure_evidence import (
    bind_result, error_axis, finish_figures, symmetric_limit,
)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "font.size": 7.0, "axes.labelsize": 7.5,
    "axes.titlesize": 8.0, "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
    "legend.fontsize": 6.7, "axes.linewidth": 0.75, "axes.spines.right": False,
    "axes.spines.top": False, "legend.frameon": False, "lines.solid_capstyle": "round",
    "mathtext.fontset": "dejavusans",
})
SEEDS = (20260820, 20260821, 20260822)
RESOLUTIONS = (64, 96, 128)
METHODS = ("GIFT", "GIFT-Lite", "FNO-2D", "FNO-3D")
GIFT_REGIMES = ("GIFT", "GIFT-Lite")
H5_GROUP = {"FNO-2D": "FNO_2D", "FNO-3D": "FNO_3D"}
COLORS = {"GIFT": "#1246D6", "GIFT-Lite": "#19C2F5", "FNO-2D": "#8B3FE0", "FNO-3D": "#00A896", "Truth": "#262626"}
MARKERS = {"GIFT": "o", "GIFT-Lite": "o", "FNO-2D": "s", "FNO-3D": "D"}
LINESTYLES = {"GIFT": "-", "GIFT-Lite": "--", "FNO-2D": (0, (4.0, 2.0)), "FNO-3D": (0, (1.3, 1.5))}
TEXT_COLOR, NOTE_COLOR, GUIDE_COLOR = "#262626", "#687078", "#E7E9EC"
# Face colour of tiles that carry no field of their own, as in the manuscript.
PLACEHOLDER_FACE = "#f6f6f7"


def add_panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", clip_on=False)


def add_prediction_error_label(axis: Any, value: float) -> None:
    """The manuscript's residual caption under a prediction-error panel."""
    axis.text(
        0.50,
        -0.070,
        rf"rel. $L^2$ = {value:.3f}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=5.4,
        color="#3F454B",
        fontweight="normal",
        clip_on=False,
    )


def draw_placeholder_tile(axis: Any) -> None:
    """A tile for the reference column's prediction-error cell.

    The manuscript fills it with the light grey the zero-error tiles carry,
    rather than leaving it blank or putting a prose note in it. Leaving
    ``axison`` on keeps the axes patch, which is what draws the tile.
    """
    axis.set_facecolor(PLACEHOLDER_FACE)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def clean_axis(ax):
    ax.tick_params(axis="both", which="major", direction="out", length=2.7, width=0.7, pad=2.0)
    ax.tick_params(axis="both", which="minor", direction="out", length=1.6, width=0.55)


def format_plain(value, _=None):
    if value == 0 or value >= 1:
        return f"{value:g}"
    if value >= 0.1:
        return f"{value:.1f}"
    return f"{value:.2g}"


def draw_vector_field(axis, values, *, cmap, vmin, vmax):
    """One vector cell per grid point; match upper-origin, nearest-cell display."""
    height, width = values.shape
    mesh = axis.pcolormesh(np.arange(width + 1) - 0.5, np.arange(height + 1) - 0.5,
                          values, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                          edgecolors="none", linewidth=0, antialiased=False, rasterized=False)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_aspect("equal", adjustable="box")
    return mesh


def load_data(result: Path, trajectory_id: int, gift_seed: int) -> dict:
    """Check CSV against raw per-trajectory metrics; do not assert method ranking."""
    frame = pd.read_csv(result / "summary/metrics.csv")
    frame = frame.loc[(frame["cohort"] == "test_1040_1219") &
                      (frame["metric"] == "full_relative_l2")].copy()
    times = 5.0 + np.arange(11) * 0.1
    data = {"times": times, "curves": {}, "curve_seeds": {}, "fields": {},
            "residual_errors": {}, "source_rows": frame}
    frame = frame.loc[frame["method"].isin(METHODS)].copy()
    data["source_rows"] = frame
    if len(frame) != 3 * (3 * len(GIFT_REGIMES) + 2) * 11 or set(frame["experiment"]) != {"M3"}:
        raise ValueError("incomplete M3 full-field metric grid")
    with h5py.File(result / "raw/predictions.h5", "r") as handle:
        if handle.attrs.get("schema") != "gift.formal.M3.raw.v2":
            raise ValueError("unexpected M3 raw-result schema")
        for resolution in RESOLUTIONS:
            grid = handle[f"N{resolution}"]
            ids = np.asarray(grid["trajectory_ids"], dtype=np.int64)
            if not np.array_equal(ids, np.arange(1040, 1220)) or not np.allclose(
                np.asarray(grid["absolute_times"]), times, rtol=0, atol=2e-12
            ):
                raise ValueError("M3 trajectory IDs or time grid differ")
            seed_curves = {name: [] for name in GIFT_REGIMES}
            data["curves"][resolution] = {}
            groups = [(name, str(seed), f"{name.replace('-', '_')}_seed_{seed}")
                      for name in GIFT_REGIMES for seed in SEEDS]
            groups += [(method, "fixed", H5_GROUP[method]) for method in ("FNO-2D", "FNO-3D")]
            for method, seed, name in groups:
                values = np.asarray(grid[f"{name}/full_relative_l2"], dtype=np.float64)
                if values.shape != (180, 11) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("M3 raw metric shape or values differ")
                subset = frame.loc[(frame["resolution"] == f"N{resolution}") &
                                   (frame["method"] == method) &
                                   (frame["seed"].astype(str) == seed)].sort_values("absolute_time")
                if len(subset) != 11 or not np.allclose(subset["absolute_time"], times, rtol=0, atol=2e-12):
                    raise ValueError("M3 method/seed/time rows missing or duplicated")
                if not (subset["population"].eq(180).all() and subset["finite_count"].eq(180).all()):
                    raise ValueError("M3 population or finiteness differs")
                statistics = {"mean": values.mean(0), "sample_sd": values.std(0, ddof=1),
                              "median": np.median(values, axis=0), "p95": np.quantile(values, 0.95, axis=0),
                              "maximum": values.max(0)}
                for column, calculated in statistics.items():
                    if not np.allclose(subset[column], calculated, rtol=1e-12, atol=1e-12):
                        raise ValueError(f"M3 CSV/raw {column} mismatch")
                if method in seed_curves:
                    seed_curves[method].append(statistics["mean"])
                else:
                    data["curves"][resolution][method] = statistics["mean"]
            data["curve_seeds"][resolution] = {name: np.stack(values) for name, values in seed_curves.items()}
            for name, values in seed_curves.items():
                data["curves"][resolution][name] = np.mean(values, axis=0)
        grid = handle["N128"]
        index = np.flatnonzero(np.asarray(grid["trajectory_ids"]) == trajectory_id)
        if len(index) != 1:
            raise ValueError("M3 example trajectory is not unique")
        row = int(index[0])
        truth = np.asarray(grid["truth"][row, -1], dtype=np.float32)
        data["fields"]["Reference"] = truth
        for method in METHODS:
            name = (f"{method.replace('-', '_')}_seed_{gift_seed}"
                    if method in ("GIFT", "GIFT-Lite") else H5_GROUP[method])
            field = np.asarray(grid[f"{name}/prediction"][row, -1], dtype=np.float32)
            if field.shape != (128, 128) or not np.isfinite(field).all() or not np.isfinite(truth).all():
                raise ValueError("M3 example field invalid")
            actual = float(relative_l2(field[None], truth[None])[0])
            recorded = float(grid[f"{name}/full_relative_l2"][row, -1])
            if not np.isclose(actual, recorded, rtol=1e-12, atol=1e-14):
                raise ValueError("M3 field error differs from recorded metric")
            data["fields"][method] = field
            data["residual_errors"][method] = actual
    return data


def draw_time_curves_only(data: dict[str, Any]) -> plt.Figure:
    """Compare all three grids with the same scales and method styles."""
    width_inches = 183.0 / 25.4
    height_inches = 72.0 / 25.4
    fig = plt.figure(figsize=(width_inches, height_inches), facecolor="white")
    curve_grid = fig.add_gridspec(
        1,
        3,
        left=0.073,
        right=0.975,
        bottom=0.205,
        top=0.785,
        wspace=0.15,
    )
    curve_axes = [fig.add_subplot(curve_grid[0, index]) for index in range(3)]
    absolute_time = np.asarray(data["times"])[1:]
    y_ticks = [0.02, 0.05, 0.1, 0.2, 0.5]
    values = np.concatenate([
        data["curve_seeds"][grid][name][:, 1:].ravel() for grid in RESOLUTIONS for name in GIFT_REGIMES
    ] + [data["curves"][grid][method][1:] for grid in RESOLUTIONS for method in METHODS])
    scale = error_axis(values, (0.015, 0.55))

    for index, (axis, resolution) in enumerate(zip(curve_axes, RESOLUTIONS)):
        for name in GIFT_REGIMES:
            seed_curves = data["curve_seeds"][resolution][name][:, 1:]
            axis.fill_between(absolute_time, seed_curves.min(axis=0), seed_curves.max(axis=0),
                              color=COLORS[name], alpha=0.16, linewidth=0, zorder=1)
        for method in METHODS:
            values = data["curves"][resolution][method][1:]
            axis.plot(
                absolute_time,
                values,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.9 if method == "GIFT" else 1.35,
                marker=MARKERS[method],
                markersize=3.2 if method == "GIFT" else 2.8,
                markeredgecolor="white",
                markeredgewidth=0.35,
                markevery=[0, 2, 4, 6, 8, 9],
                zorder=5 if method == "GIFT" else 3,
            )
        axis.set_yscale(scale["scale"], **(
            {"linthresh": scale["linthresh"]} if scale["scale"] == "symlog" else {}
        ))
        # Reserve a right-hand label lane inside every panel so all three
        # resolutions can use the same "method + endpoint value" labels.
        axis.set_xlim(5.075, 6.20)
        axis.set_ylim(*scale["limits"])
        axis.set_xticks([5.1, 5.4, 5.7, 6.0])
        axis.yaxis.set_major_locator(FixedLocator(y_ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(format_plain))
        if scale["expanded"]:
            locator = (SymmetricalLogLocator(base=10, linthresh=scale["linthresh"])
                       if scale["scale"] == "symlog" else LogLocator(base=10))
            axis.yaxis.set_major_locator(locator)
        axis.yaxis.grid(True, which="major", color=GUIDE_COLOR, lw=0.48)
        axis.set_axisbelow(True)
        axis.text(
            0.02,
            1.04,
            rf"$N={resolution}$",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="semibold",
            color=TEXT_COLOR,
            clip_on=False,
        )
        if index == 0:
            axis.set_ylabel(r"Mean full-field relative $L^2$ error")
        else:
            axis.tick_params(axis="y", which="both", left=False, labelleft=False)
            axis.spines["left"].set_visible(False)
        clean_axis(axis)

    curve_axes[1].set_xlabel(r"Time, $t$")
    for axis, resolution in zip(curve_axes, RESOLUTIONS):
        for method in METHODS:
            endpoint = float(data["curves"][resolution][method][-1])
            label = f"{method}  {endpoint:.3f}"
            annotation = axis.annotate(
                label,
                xy=(float(absolute_time[-1]), endpoint),
                xytext=(3.0, 0.0),
                textcoords="offset points",
                color=COLORS[method],
                fontsize=5.25,
                fontweight="semibold" if method == "GIFT" else "normal",
                ha="left",
                va="center",
                annotation_clip=False,
            )
    return fig


def draw_prediction_fields_only(
    data: dict[str, Any],
    trajectory_id: int,
    gift_seed: int,
) -> tuple[plt.Figure, dict[str, float]]:
    """N128 reference/prediction row with aligned signed residual evidence."""
    # The manuscript's plate carries a left label column as wide as its longest
    # row name, so its canvas is 205.17 mm rather than the 183.0 mm the curves
    # use. The panels themselves keep the manuscript's 87.667 pt cell size.
    width_inches = 205.17 / 25.4
    height_inches = 96.0 / 25.4
    fig = plt.figure(figsize=(width_inches, height_inches), facecolor="white")
    grid = fig.add_gridspec(
        2,
        len(METHODS) + 3,
        left=0.052,
        right=0.965,
        bottom=0.115,
        top=0.865,
        height_ratios=(1.0, 1.0),
        width_ratios=(0.6006, *([1.0] * (len(METHODS) + 1)), 0.045),
        hspace=0.10,
        wspace=0.085,
    )

    fields = data["fields"]
    methods = METHODS
    residuals = {method: fields[method] - fields["Reference"] for method in methods}
    observed_field_maximum = max(float(np.max(np.abs(value))) for value in fields.values())
    observed_residual_maximum = max(float(np.max(np.abs(value))) for value in residuals.values())
    field_limit = symmetric_limit(observed_field_maximum, 13.0)
    residual_limit = symmetric_limit(observed_residual_maximum, 7.0)

    field_axes: list[plt.Axes] = []
    image_field = None
    for column, label in enumerate(("Reference", *methods), start=1):
        axis = fig.add_subplot(grid[0, column])
        image_field = draw_vector_field(axis,
            fields[label],
            cmap="RdBu_r",
            vmin=-field_limit,
            vmax=field_limit,
        )
        # Every method name is black in the manuscript; the colour coding is
        # carried by the curves, not by the plate's row and column names.
        axis.set_title(
            label,
            pad=2.8,
            fontsize=7.2,
            fontweight="semibold",
            color=TEXT_COLOR,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        field_axes.append(axis)

    # Left label column: the row names, right aligned seven points before the
    # first panel, centred on their own row, as in the manuscript.
    title_anchor = field_axes[0].get_position(fig).x0 - 7.0 / (width_inches * 72.0)
    for row, row_name in enumerate(("Vorticity field", "Prediction error")):
        row_position = grid[row, 1].get_position(fig)
        fig.text(
            title_anchor,
            0.5 * (row_position.y0 + row_position.y1),
            row_name,
            ha="right",
            va="center",
            fontsize=7.0,
            fontweight="semibold",
            color="#171717",
        )

    # The reference column has no prediction error of its own; the manuscript
    # fills that cell with the light grey zero tile and the 0.000 value a
    # reference-versus-itself comparison gives.
    placeholder_axis = fig.add_subplot(grid[1, 1])
    draw_placeholder_tile(placeholder_axis)
    add_prediction_error_label(placeholder_axis, 0.0)

    residual_axes: list[plt.Axes] = []
    image_residual = None
    for column, method in enumerate(methods, start=2):
        axis = fig.add_subplot(grid[1, column])
        image_residual = draw_vector_field(axis,
            residuals[method],
            cmap="PuOr",
            vmin=-residual_limit,
            vmax=residual_limit,
        )
        add_prediction_error_label(axis, float(data["residual_errors"][method]))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        residual_axes.append(axis)

    if image_field is None or image_residual is None:
        raise AssertionError("field or residual image was not created")
    field_color_axis = fig.add_subplot(grid[0, len(METHODS) + 2])
    residual_color_axis = fig.add_subplot(grid[1, len(METHODS) + 2])
    colorbar_field = fig.colorbar(image_field, cax=field_color_axis, orientation="vertical")
    colorbar_residual = fig.colorbar(image_residual, cax=residual_color_axis, orientation="vertical")
    for colorbar, limit, title in (
        (colorbar_field, field_limit, r"$\omega$"),
        (colorbar_residual, residual_limit, r"$\Delta\omega$"),
    ):
        colorbar.solids.set_rasterized(False)
        colorbar.set_ticks([-limit, 0.0, limit])
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
        colorbar.ax.tick_params(length=1.8, width=0.45, pad=1.4, labelsize=6.0)
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_fontweight("semibold")
        # Keep the row variable beside the continuous colour ramp.  Titles
        # above the two stacked bars read as a glyph interrupting the scale.
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.set_ylabel(
            title,
            fontsize=6.5,
            rotation=0,
            labelpad=3.0,
            fontweight="semibold",
            color=TEXT_COLOR,
        )
        colorbar.outline.set_linewidth(0.35)
        colorbar.outline.set_edgecolor("#8A8F94")

    fig.text(
        0.50,
        0.955,
        rf"$N=128$  ·  $t=6.0$  ·  trajectory {trajectory_id}",
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="normal",
        color=TEXT_COLOR,
    )
    return fig, {
        "field_limit": field_limit,
        "residual_limit": residual_limit,
        "observed_field_maximum": observed_field_maximum,
        "observed_residual_maximum": observed_residual_maximum,
    }



def main() -> None:
    global METHODS, GIFT_REGIMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-id", type=int, default=1150)
    parser.add_argument("--gift-seed", type=int, default=20260820, choices=SEEDS)
    parser.add_argument("--gift-regimes", nargs="+", choices=("GIFT", "GIFT-Lite"),
                        default=["GIFT", "GIFT-Lite"],
                        help="explicit completed regimes; all selected metric rows are required")
    args = parser.parse_args()
    GIFT_REGIMES = tuple(name for name in GIFT_REGIMES if name in args.gift_regimes)
    METHODS = (*GIFT_REGIMES, "FNO-2D", "FNO-3D")
    result = args.result_dir.resolve(strict=True)
    evidence = bind_result(result, "M3", ("summary/metrics.csv", "raw/predictions.h5"))
    data = load_data(result, args.trajectory_id, args.gift_seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data["source_rows"].to_csv(output / "source_data.csv", index=False)
    np.savez_compressed(output / "source_fields.npz", **data["fields"],
                        trajectory_id=args.trajectory_id, gift_seed=args.gift_seed,
                        absolute_time=6.0)
    figure = draw_time_curves_only(data)
    curve_axes = [{"scale": axis.get_yscale(), "limits": list(axis.get_ylim())} for axis in figure.axes]
    figure.savefig(output / "cross_resolution_mean_relative_l2.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    figure, limits = draw_prediction_fields_only(data, args.trajectory_id, args.gift_seed)
    figure.savefig(output / f"traj{args.trajectory_id}_n128_t6_prediction_fields.svg",
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)
    finish_figures(output, evidence, Path(__file__), {
        "trajectory_id": args.trajectory_id, "gift_seed": args.gift_seed,
        "selection": "prespecified example, not selected from these evaluation errors",
        "field_color_map": "RdBu_r", "residual_color_map": "PuOr", "color_limits": limits,
        "residual_definition": "prediction minus reference", "crop": "none",
        "smoothing": "none", "grid_cell_display": "vector, upper origin",
        "curve_axes": curve_axes, "t5_anchor_omitted": True,
        "gift_band": "pointwise min/max across three seed-specific trajectory means",
        "trajectory_count": 180, "baseline_seeds": 1, "hypothesis_tests": "none",
        "methods": list(METHODS),
    })


if __name__ == "__main__":
    main()
