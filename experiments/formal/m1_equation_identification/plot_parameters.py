#!/usr/bin/env python
"""Draw the three-parameter M1 comparison from completed numerical results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.formal._shared.figure_evidence import (
    bind_result, error_axis, finish_figures,
)

METHODS = ["GIFT", "PDE-FIND", "PDE-FIND-KC", "PINN-SR", "PINN-SR-KC"]
NOISE_LEVELS = [0.0, 1.0, 10.0]
PARAMETERS = ["nu", "beta", "gamma"]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stixsans",
        "font.size": 6.6,
        "axes.labelsize": 7.0,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# Restrained method-family palette: one blue anchor, a warm PDE family,
# and a muted violet PINN family. Marker shape remains the independent cue.
METHOD_STYLE = {
    "GIFT": {"color": "#1246D6", "marker": "o"},
    "PDE-FIND": {"color": "#FF7A00", "marker": "s"},
    "PDE-FIND-KC": {"color": "#29A82C", "marker": "D"},
    "PINN-SR": {"color": "#8B3FE0", "marker": "^"},
    "PINN-SR-KC": {"color": "#00A896", "marker": "v"},
}

# A sub-marker-width categorical dodge prevents near-identical APE values from
# hiding one another while preserving the three tested noise conditions.
METHOD_X_OFFSET = {
    "GIFT": -0.055,
    "PDE-FIND": -0.0275,
    "PDE-FIND-KC": 0.0,
    "PINN-SR": 0.0275,
    "PINN-SR-KC": 0.055,
}

PARAMETER_SYMBOL = {
    # Unicode Greek remains editable without STIX private-use font glyphs.
    "nu": "ν",
    "beta": "β",
    "gamma": "γ",
}

TRUE_VALUE_LABEL = {
    "nu": "true value 0.01",
    "beta": "true value 1",
    "gamma": "true value 1",
}

TEXT_DARK = "#273039"
TEXT_MID = "#59616A"
TEXT_LIGHT = "#7A828A"
GRID = "#D9DEE3"
SPINE = "#69717A"


def draw(frame: pd.DataFrame) -> plt.Figure:
    # 183 mm wide, compact double-column height.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.20, 2.90),
        sharey=True,
        gridspec_kw={"wspace": 0.115},
    )
    x = np.arange(len(NOISE_LEVELS), dtype=float)
    scale = error_axis(frame["relative_error_percent"], (0.02, 300.0))

    for panel_index, (ax, parameter) in enumerate(zip(axes, PARAMETERS)):
        parameter_frame = frame.loc[frame["parameter"] == parameter]

        for method in METHODS:
            method_frame = (
                parameter_frame.loc[parameter_frame["method"] == method]
                .set_index("noise_percent")
                .loc[NOISE_LEVELS]
            )
            style = METHOD_STYLE[method]
            method_x = x + METHOD_X_OFFSET[method]
            ax.plot(
                method_x,
                method_frame["relative_error_percent"].to_numpy(),
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=4.15,
                linewidth=1.18,
                alpha=0.96,
                zorder=3,
            )

        ax.set_yscale(scale["scale"], **(
            {"linthresh": scale["linthresh"]} if scale["scale"] == "symlog" else {}
        ))
        ax.set_ylim(*scale["limits"])
        ax.set_xlim(-0.20, 2.20)
        ax.set_xticks(x, ["0%", "1%", "10%"])
        ax.yaxis.set_major_locator(FixedLocator([0.03, 0.1, 1.0, 10.0, 100.0]))
        ax.yaxis.set_major_formatter(
            FixedFormatter(["0.03", "0.1", "1", "10", "100"])
        )
        if scale["expanded"]:
            from matplotlib.ticker import LogLocator, SymmetricalLogLocator
            locator = (SymmetricalLogLocator(base=10, linthresh=scale["linthresh"])
                       if scale["scale"] == "symlog" else LogLocator(base=10))
            ax.yaxis.set_major_locator(locator)
            from matplotlib.ticker import LogFormatterSciNotation
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.yaxis.set_minor_locator(NullLocator())

        for grid_value in (0.1, 1.0, 10.0, 100.0):
            ax.axhline(
                grid_value,
                color=GRID,
                linewidth=0.38,
                alpha=0.78,
                zorder=0,
            )

        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            length=2.8,
            width=0.55,
            color=SPINE,
            labelcolor=TEXT_DARK,
            pad=3.0,
        )
        ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=0,
            width=0.0,
            labelcolor=TEXT_MID,
            pad=3.8,
        )
        ax.spines["bottom"].set_color(SPINE)
        ax.spines["left"].set_color(SPINE)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_linewidth(0.6)

        # One editorial header line per panel, as in the manuscript figure:
        # the italic parameter symbol followed by its true value in parentheses.
        ax.text(
            0.0,
            1.055,
            PARAMETER_SYMBOL[parameter],
            fontstyle="italic",
            transform=ax.transAxes,
            ha="left",
            va="baseline",
            fontsize=9.7,
            fontweight="semibold",
            color=TEXT_DARK,
            clip_on=False,
        )
        ax.text(
            0.115,
            1.055,
            f"({TRUE_VALUE_LABEL[parameter]})",
            transform=ax.transAxes,
            ha="left",
            va="baseline",
            fontsize=6.3,
            color=TEXT_LIGHT,
            clip_on=False,
        )

        if panel_index > 0:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

    axes[0].set_ylabel("APE (%)", color=TEXT_DARK, labelpad=7.5)
    fig.supxlabel("Noise condition", x=0.52, y=0.067, fontsize=7.0, color=TEXT_DARK)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            marker=METHOD_STYLE[method]["marker"],
            markerfacecolor=METHOD_STYLE[method]["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.0,
            linewidth=1.18,
            label=method,
        )
        for method in METHODS
    ]
    legend = fig.legend(
        handles=legend_handles,
        labels=METHODS,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.977),
        ncol=5,
        columnspacing=1.02,
        handlelength=1.35,
        handletextpad=0.34,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_DARK)

    fig.subplots_adjust(left=0.077, right=0.991, bottom=0.225, top=0.785)
    return fig


def load_and_validate(path: Path) -> pd.DataFrame:
    """Check complete factorial coverage and derive errors from the estimates."""
    frame = pd.read_csv(path)
    required = {"experiment", "condition", "noise_percent", "method", "parameter",
                "estimate", "reference", "absolute_error", "relative_error_percent"}
    if not required.issubset(frame.columns):
        raise ValueError("M1 metrics columns are incomplete")
    expected = {(noise, method, parameter) for noise in NOISE_LEVELS
                for method in METHODS for parameter in PARAMETERS}
    actual = set(frame[["noise_percent", "method", "parameter"]].itertuples(index=False, name=None))
    if len(frame) != 45 or actual != expected:
        raise ValueError("M1 requires each of the 45 method/condition/parameter rows exactly once")
    if set(frame["experiment"]) != {"M1"}:
        raise ValueError("wrong experiment in M1 metrics")
    condition = {0.0: "noise_000", 1.0: "noise_001", 10.0: "noise_010"}
    references = {"nu": 0.01, "beta": 1.0, "gamma": 1.0}
    if any(row.condition != condition[row.noise_percent] or
           row.reference != references[row.parameter] for row in frame.itertuples()):
        raise ValueError("M1 condition or true parameter differs")
    numeric = frame[["estimate", "reference", "absolute_error", "relative_error_percent"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("M1 contains non-finite data")
    absolute = np.abs(frame["estimate"] - frame["reference"])
    ape = 100.0 * absolute / np.abs(frame["reference"])
    if not np.allclose(frame["absolute_error"], absolute, rtol=0, atol=1e-12):
        raise ValueError("M1 absolute error does not match the estimate")
    if not np.allclose(frame["relative_error_percent"], ape, rtol=0, atol=1e-10):
        raise ValueError("M1 percentage error does not match the estimate")
    if (frame["relative_error_percent"] < 0).any():
        raise ValueError("negative M1 percentage error")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = args.result_dir.resolve(strict=True)
    evidence = bind_result(result, "M1", ("summary/metrics.csv",))
    frame = load_and_validate(result / "summary/metrics.csv")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "source_data.csv", index=False)
    figure = draw(frame)
    # Tight crop, as in the manuscript figure: no surrounding whitespace.
    figure.savefig(output / "parameter_identification_ape_vs_noise.svg", format="svg",
                   facecolor="white", bbox_inches="tight", pad_inches=0.0,
                   metadata={"Title": "M1 parameter-identification comparison"})
    plt.close(figure)
    finish_figures(output, evidence, Path(__file__), {
        "metric": "absolute percentage error", "error_bars": "none",
        "axis": error_axis(frame["relative_error_percent"], (0.02, 300.0)),
        "method_order": METHODS, "noise_percent": NOISE_LEVELS,
        "source_data": "source_data.csv",
    })


if __name__ == "__main__":
    main()
