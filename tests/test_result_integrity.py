"""Synthetic integrity fixtures; no experiment values or training are certified."""
import json

import pytest

from scripts.verify_results import verify_figures, verify_result
from experiments.formal._shared.figure_evidence import file_record, finish_figures, sha256


def make_result(root, numeric=True):
    root.mkdir()
    (root / "summary").mkdir()
    metric = root / "summary/metrics.csv"
    metric.write_text("fixture_only\nnot_scientific\n", encoding="utf-8")
    report = dict(experiment="M2", status="numerical_complete" if numeric else "complete")
    if numeric:
        report["numeric_files"] = [file_record(metric, root)]
    filename = "numeric_report.json" if numeric else "report.json"
    (root / filename).write_text(json.dumps(report), encoding="utf-8")
    if not numeric:
        (root / "manifest.json").write_text(json.dumps(dict(experiment="M2", status="complete",
            files=[file_record(metric, root), file_record(root / filename, root)])), encoding="utf-8")
    return root


@pytest.mark.parametrize("numeric", [True, False])
def test_both_completion_formats_are_integrity_only(tmp_path, numeric):
    root = make_result(tmp_path / "result", numeric)
    result = verify_result(root, "M2")
    assert result["status"] == "PASS_INTEGRITY_ONLY"
    assert not result["training_performed"] and not result["numerical_metrics_recomputed"]


@pytest.mark.parametrize("change", ["corrupt", "status", "experiment", "duplicate", "escape", "size"])
def test_invalid_numeric_result_rejected(tmp_path, change):
    root = make_result(tmp_path / "result")
    path = root / "numeric_report.json"
    report = json.loads(path.read_text())
    if change == "corrupt":
        (root / "summary/metrics.csv").write_text("different data")
    elif change == "status":
        report["status"] = "running"
    elif change == "experiment":
        report["experiment"] = "S2"
    elif change == "duplicate":
        report["numeric_files"] *= 2
    elif change == "escape":
        report["numeric_files"][0]["path"] = "../outside.csv"
    else:
        report["numeric_files"][0]["bytes"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_result(root, "M2")


def test_svg_binding_and_structure(tmp_path):
    root = make_result(tmp_path / "result")
    result = verify_result(root, "M2")
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "test.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>fixture</text></svg>')
    finish_figures(figures, {"experiment": "M2", "report_sha256": result["report_sha256"]},
                   root / "numeric_report.json", {"fixture_only": True})
    assert verify_figures(figures, result)["svg_files_verified"] == 1
    with pytest.raises(ValueError, match="another"):
        verify_figures(figures, dict(result, report_sha256="0" * 64))
    (figures / "test.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><image/></svg>')
    path = figures / "figure_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0] = file_record(figures / "test.svg", figures)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="raster"):
        verify_figures(figures, result)
