"""A renderer crash must not discard completed experimental evidence."""
import json
import subprocess

import pytest

from experiments.formal._shared.plotting import publish_numeric_then_plot


def test_numeric_report_and_manifest_survive_plot_failure(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "test renderer")
    monkeypatch.setattr(subprocess, "run", failure)
    with pytest.raises(RuntimeError, match="safely saved"):
        publish_numeric_then_plot(tmp_path, {"experiment": "M2", "status": "complete", "metric": 0.1}, [["test"]])
    numeric = json.loads((tmp_path / "numeric_report.json").read_text())
    final = json.loads((tmp_path / "report.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert numeric["metric"] == 0.1
    assert final["status"] == manifest["status"] == "numerical_complete_plot_failed"


def test_skip_plot_is_explicit(tmp_path):
    publish_numeric_then_plot(tmp_path, {"experiment": "M2", "status": "complete"}, [])
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["plot_execution"]["status"] == "skipped_by_request"
