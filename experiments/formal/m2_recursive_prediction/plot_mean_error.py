"""Draw the five-method M2 mean-error comparison from a metrics CSV.

Markers are the seven evaluated report times.  The lines are shape-preserving
piecewise cubic Hermite interpolants (PCHIP) that pass exactly through those
markers; they are a visual guide and do not add evaluation times.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402


REPORT_TIMES = np.asarray([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
METHOD_ORDER = ("GIFT", "FNO-2D", "FNO-3D", "U-NO", "U-Net")
COHORT = "trajectories_1000_1199"
POPULATION = 200

STYLES = {
    "GIFT": ("#0F4D92", "o", "-", 2.5, 10),
    "FNO-2D": ("#7884B4", "s", "--", 1.5, 5),
    "FNO-3D": ("#42949E", "D", "-.", 1.5, 4),
    "U-NO": ("#2E9E44", "^", "-", 1.8, 8),
    "U-Net": ("#E28E2C", "v", ":", 1.6, 6),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_source(metrics_path: Path) -> list[dict[str, object]]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["cohort"] == COHORT]
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        method = row["method"]
        if method in METHOD_ORDER:
            grouped[(method, float(row["absolute_time"]))].append(row)

    source: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        for absolute_time in REPORT_TIMES:
            matches = grouped[(method, float(absolute_time))]
            expected = 3 if method == "GIFT" else 1
            if len(matches) != expected:
                raise ValueError(
                    f"{method} at t={absolute_time:g} has {len(matches)} rows; "
                    f"expected {expected}"
                )
            if any(
                int(row["population"]) != POPULATION
                or int(row["finite_count"]) != POPULATION
                for row in matches
            ):
                raise ValueError(f"{method} population or finiteness differs")
            values = np.asarray([float(row["mean"]) for row in matches])
            source.append(
                {
                    "method": method,
                    "absolute_time": float(absolute_time),
                    "mean_relative_l2": float(values.mean()),
                    "training_seed_sd": (
                        float(values.std(ddof=1)) if method == "GIFT" else ""
                    ),
                    "n_training_seeds": expected,
                    "n_test_trajectories": POPULATION,
                }
            )
    return source


def write_source(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict[str, object]]):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    series = {
        method: np.asarray(
            [float(row["mean_relative_l2"]) for row in rows if row["method"] == method]
        )
        for method in METHOD_ORDER
    }
    figure, axis = plt.subplots(figsize=(183.0 / 25.4, 96.0 / 25.4))
    fitted_times = np.linspace(5.0, 8.0, 601)
    handles: dict[str, Line2D] = {}
    for method in ("FNO-3D", "U-Net", "FNO-2D", "U-NO", "GIFT"):
        color, marker, line_style, width, zorder = STYLES[method]
        values = series[method]
        interpolator = PchipInterpolator(REPORT_TIMES, values)
        fitted = interpolator(fitted_times)
        if not np.allclose(interpolator(REPORT_TIMES), values, rtol=0.0, atol=1.0e-12):
            raise AssertionError(f"{method} fitted curve misses a sample point")
        if np.min(fitted) < -1.0e-12 or np.min(np.diff(fitted)) < -1.0e-10:
            raise AssertionError(
                f"{method} fitted curve violates nonnegative monotonicity"
            )
        axis.plot(
            fitted_times,
            fitted,
            color=color,
            linestyle=line_style,
            linewidth=width,
            solid_capstyle="round",
            zorder=zorder,
        )
        axis.plot(
            REPORT_TIMES,
            values,
            linestyle="none",
            color=color,
            marker=marker,
            markersize=4.8 if method == "GIFT" else 4.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=zorder + 0.2,
        )
        handles[method] = Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=line_style,
            linewidth=width,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.55,
        )
    axis.set_xlim(4.96, 8.04)
    axis.set_ylim(0.0, 1.28)
    axis.set_xticks(REPORT_TIMES)
    axis.set_yticks(np.arange(0.0, 1.21, 0.2))
    axis.set_xlabel(r"Time, $t$")
    axis.set_ylabel(r"Mean relative $L^2$ error")
    axis.tick_params(axis="both", direction="out", length=3.0, width=0.8, pad=2.5)
    axis.spines["left"].set_bounds(0.0, 1.2)
    axis.spines["bottom"].set_bounds(5.0, 8.0)
    axis.legend(
        [handles[method] for method in METHOD_ORDER],
        METHOD_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=5,
        handlelength=2.5,
        columnspacing=1.6,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.165, top=0.82)
    return figure


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    result = project_root / "results" / "formal" / "M2_recursive_prediction"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=result / "summary" / "metrics.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=result / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = args.metrics.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "source_data.csv",
        output / "source_manifest.csv",
        output / "mean_relative_l2_vs_time.svg",
        output / "mean_relative_l2_vs_time.pdf",
        output / "mean_relative_l2_vs_time.png",
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite an existing M2 figure file")

    source = read_source(metrics)
    write_source(targets[0], source)
    try:
        metric_display = metrics.relative_to(output.parent).as_posix()
    except ValueError:
        metric_display = metrics.as_posix()
    write_source(
        targets[1],
        [
            {
                "source": metric_display,
                "sha256": sha256_file(metrics),
                "fitted_curve": "PCHIP through all seven evaluated points",
            }
        ],
    )
    figure = draw(source)
    figure.savefig(targets[2], bbox_inches="tight")
    figure.savefig(targets[3], bbox_inches="tight")
    figure.savefig(targets[4], dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
