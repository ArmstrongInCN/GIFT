"""Figures validate their evidence, not a preferred scientific conclusion."""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from experiments.formal._shared.figure_evidence import (
    _crisp_vector_cells, bind_result, error_axis, file_record, finish_figures, symmetric_limit, verify_svg,
)
from experiments.formal._shared.plotting import publish_numeric_then_plot
from experiments.formal.m1_equation_identification import plot_parameters as m1
from experiments.formal.m2_recursive_prediction import plot_mean_error as m2
from experiments.formal.m2_recursive_prediction import plot_keyframes as fields
from experiments.formal.m3_cross_resolution import plot_results as m3
from scripts.verify_published_results import verify_field_source
from scripts.verify_results import verify_figures


def test_completed_result_binds_numeric_files(tmp_path):
    (tmp_path / "summary").mkdir()
    path = tmp_path / "summary/metrics.csv"
    path.write_text("mean\n0.2\n")
    publish_numeric_then_plot(tmp_path, {"experiment": "M2", "status": "complete",
                                      "inputs": {"model": {"path": "private/model.pt", "sha256": "ABC"}}}, [])
    evidence = bind_result(tmp_path, "M2", ("summary/metrics.csv",))
    assert evidence["sources"] == [file_record(path, tmp_path)]
    assert evidence["experiment_inputs"]["model"] == {"sha256": "ABC"}
    path.write_text("mean\n0.1\n")
    with pytest.raises(ValueError, match="differs"):
        bind_result(tmp_path, "M2", ("summary/metrics.csv",))


def test_unfinished_result_rejected(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"experiment": "M3", "status": "running"}))
    with pytest.raises(ValueError, match="completed"):
        bind_result(tmp_path, "M3", ("summary/metrics.csv",))


def test_range_kept_or_expanded_without_clipping():
    assert error_axis([0.02, 0.5], (0.015, 0.55))["limits"] == (0.015, 0.55)
    axis = error_axis([0.001, 20.0], (0.015, 0.55))
    assert axis["limits"][0] < 0.001 and axis["limits"][1] > 20
    assert error_axis([0, 0.2], (0.015, 0.55))["scale"] == "symlog"
    assert symmetric_limit(12.9, 13) == 13
    assert symmetric_limit(13.1, 13) == 14
    for values in ([np.nan], [-1.0], []):
        with pytest.raises(ValueError):
            error_axis(values, (0.015, 0.55))


def test_m2_nonmonotonic_large_errors_are_not_hidden(tmp_path):
    rows = [{"method": method, "absolute_time": float(time), "mean_relative_l2": value}
            for method in m2.METHOD_ORDER
            for time, value in zip(m2.REPORT_TIMES, [0, 0.7, 0.3, 0.5, 1, 2, 1.8])]
    figure = m2.draw(rows)
    assert figure.axes[0].get_ylim()[1] > 2
    output = tmp_path / "mean.svg"
    figure.savefig(output)
    plt.close(figure)
    assert verify_svg(output)["editable_text"]


def test_m2_field_limits_shared_and_outward_only():
    limits = fields.verify_color_limits(np.asarray([23.1]), {"a": np.asarray([-24.2])},
                                        {"a": np.asarray([-25.5])})
    assert limits["field_limit"] == 25
    assert limits["residual_limit"] == 26


def test_m2_keyframe_bundle_uses_only_final_manifest(tmp_path, monkeypatch):
    # A synthetic N64 renderer fixture checks exported artifacts, not accuracy.
    result = tmp_path / "numeric_fixture"
    (result / "raw").mkdir(parents=True)
    (result / "summary").mkdir()
    raw = result / "raw/predictions.h5"
    raw.write_bytes(b"renderer unit fixture; numeric loading is replaced below")
    (result / "summary/metrics.csv").write_text("renderer_unit_fixture\n", encoding="utf-8")
    publish_numeric_then_plot(result, {"experiment": "M2", "status": "complete"}, [])
    evidence = bind_result(result, "M2", ("raw/predictions.h5", "summary/metrics.csv"))
    truth = np.arange(4 * 64 * 64, dtype=np.float64).reshape(4, 64, 64) / 1024 + 1
    predictions, residuals, errors = {}, {}, {}
    for index, spec in enumerate(fields.METHODS):
        slug = str(spec["slug"])
        predictions[slug] = truth + (index + 1) / 10
        predictions[slug][0] = truth[0]
        residuals[slug] = predictions[slug] - truth
        errors[slug] = (np.linalg.norm(residuals[slug], axis=(1, 2))
                        / np.linalg.norm(truth, axis=(1, 2)))
    monkeypatch.setattr(fields, "load_keyframes",
        lambda *_: (truth, predictions, residuals, errors, [0, 2, 4, 6]))
    output = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["plot_keyframes", "--input", str(raw),
                                      "--output-dir", str(output)])
    fields.main()
    assert not (output / "manifest.json").exists()
    assert not (output / "source_data.npz").exists()
    assert not (output / "README.md").exists()
    assert not (output / "plot_keyframes.py").exists()
    assert len(list((output / "panels").rglob("*.svg"))) == len(fields.KEY_TIMES) * (1 + 2 * len(fields.METHODS))
    verify_field_source(output / "source_fields.npz")
    with np.load(output / "source_fields.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["truth"], truth)
        for spec in fields.METHODS:
            slug = str(spec["slug"])
            np.testing.assert_array_equal(archive[f"prediction_{slug}"], predictions[slug])
            np.testing.assert_array_equal(archive[f"residual_{slug}"], residuals[slug])
    verify_figures(output, {"experiment": "M2", "report_sha256": evidence["report_sha256"]})
    manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    assert all(value > 0 for value in manifest["vector_cell_rendering"]["groups_per_svg"].values())
    assert {row["path"] for row in manifest["files"]} == {
        p.relative_to(output).as_posix() for p in output.rglob("*")
        if p.is_file() and p.name != "figure_manifest.json"}


def test_svg_rejects_embedded_raster(tmp_path):
    path = tmp_path / "bad.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><image href="data:image/png;base64,"/></svg>')
    with pytest.raises(ValueError, match="raster"):
        verify_svg(path)


def test_vector_cell_edge_hint_changes_no_paths_colours_or_text(tmp_path):
    path = tmp_path / "cells.svg"
    original = (b'<svg xmlns="http://www.w3.org/2000/svg"><g id="QuadMesh_1">'
        b'<path d="M 0 0 L 1 0 L 1 1 Z" fill="#123456"/></g>'
        b'<g id="line2d_1"><path d="M 0 0 L 1 1"/></g><text>0.04</text></svg>')
    path.write_bytes(original)
    assert _crisp_vector_cells(path) == 1
    modified = path.read_bytes()
    assert modified.replace(b' shape-rendering="crispEdges"', b'') == original
    assert _crisp_vector_cells(path) == 0
    assert path.read_bytes() == modified
    assert verify_svg(path)["editable_text"]


def test_committed_figure_is_not_rewritten(tmp_path):
    path = tmp_path / "cells.svg"
    path.write_bytes(b'<g id="QuadMesh_1">')
    (tmp_path / "figure_manifest.json").write_text('{}', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="committed"):
        finish_figures(tmp_path, {}, Path(m3.__file__), {})
    assert path.read_bytes() == before


def test_m3_vector_cells_preserve_upper_origin(tmp_path):
    figure, axis = plt.subplots()
    array = np.arange(16).reshape(4, 4)
    mesh = m3.draw_vector_field(axis, array, cmap="RdBu_r", vmin=0, vmax=15)
    np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(4, 4), array)
    assert axis.get_ylim() == (3.5, -0.5)
    assert not axis.images
    path = tmp_path / "field.svg"
    figure.savefig(path)
    plt.close(figure)
    assert verify_svg(path)["embedded_raster_images"] == 0


def test_m3_no_method_ordering_gate(tmp_path):
    # Deliberately give GIFT the largest error. This is a renderer unit fixture,
    # not an experimental measurement or a claim about model performance.
    data = {"times": 5 + np.arange(11) * 0.1, "curves": {}, "curve_seeds": {}}
    for grid in m3.RESOLUTIONS:
        seeds = np.stack([np.linspace(0, 2 + index * 0.1, 11) for index in range(3)])
        data["curve_seeds"][grid] = {"GIFT": seeds, "GIFT-Lite": seeds * 0.8}
        data["curves"][grid] = {"GIFT": seeds.mean(0), "GIFT-Lite": seeds.mean(0) * 0.8, "FNO-2D": np.linspace(0, .2, 11),
                                "FNO-3D": np.linspace(0, .3, 11)}
    figure = m3.draw_time_curves_only(data)
    assert all(axis.get_ylim()[1] > 2.2 for axis in figure.axes)
    figure.savefig(tmp_path / "curves.svg")
    plt.close(figure)
    finish_figures(tmp_path, {"test_fixture": True}, Path(m3.__file__), {})
    assert json.loads((tmp_path / "figure_manifest.json").read_text())["status"] == "complete"


def m1_fixture():
    rows = []
    for noise, condition in [(0, "noise_000"), (1, "noise_001"), (10, "noise_010")]:
        for method in m1.METHODS:
            for parameter, reference in [("nu", .01), ("beta", 1.0), ("gamma", 1.0)]:
                rows.append(dict(experiment="M1", condition=condition, noise_percent=noise,
                                 method=method, parameter=parameter, estimate=reference,
                                 reference=reference, absolute_error=0.0, relative_error_percent=0.0))
    return pd.DataFrame(rows)


def test_m1_zero_error_is_visible_and_not_floored(tmp_path):
    frame = m1_fixture()
    frame.to_csv(tmp_path / "metrics.csv", index=False)
    checked = m1.load_and_validate(tmp_path / "metrics.csv")
    figure = m1.draw(checked)
    assert all(axis.get_yscale() == "symlog" for axis in figure.axes)
    assert all(axis.get_ylim()[0] == 0 for axis in figure.axes)
    np.testing.assert_array_equal(checked["relative_error_percent"], np.zeros(45))
    figure.savefig(tmp_path / "zero.svg")
    plt.close(figure)
    verify_svg(tmp_path / "zero.svg")
    # Portable labels must not depend on a viewer's private-use font mapping.
    svg = (tmp_path / "zero.svg").read_text(encoding="utf-8")
    assert all(symbol in svg for symbol in ("ν", "β", "γ"))
    assert not any(0xE000 <= ord(char) <= 0xF8FF for char in svg)


@pytest.mark.parametrize("corruption", ["duplicate", "mismatch", "nan"])
def test_m1_corrupt_metrics_rejected(tmp_path, corruption):
    frame = m1_fixture()
    if corruption == "duplicate":
        frame.iloc[1] = frame.iloc[0]
    elif corruption == "mismatch":
        frame.loc[0, "estimate"] = .5
    else:
        frame.loc[0, "estimate"] = np.nan
    frame.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        m1.load_and_validate(tmp_path / "bad.csv")
