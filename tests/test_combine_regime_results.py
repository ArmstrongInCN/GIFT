"""Synthetic summary plumbing only; these fixtures contain no experiment data."""

import csv
import json

import pytest

from experiments.formal._shared.common import finish_output, write_csv_new, write_json_new
from scripts.combine_regime_results import combine
from scripts.verify_results import verify_result


def source(tmp_path, regime, experiment="S3", raw_hash="1" * 64):
    path = tmp_path / regime
    (path / "summary").mkdir(parents=True)
    write_csv_new(path / "summary/metrics.csv", [{"experiment": experiment,
        "training_regime": regime, "quantity": "TEST_FIXTURE_NOT_SCIENCE", "seed": 20260820,
        "within_seed_mean": 1.0}])
    write_json_new(path / "summary/summary.json", {"fixture_only": True})
    finish_output(path, {"experiment": experiment, "status": "complete", "training_regime": regime,
        "fixture_only": True, "inputs": {key: {"sha256": raw_hash} for key in
            ("raw_N64", "raw_cross_resolution", "dense_N64_equation_data")},
        "scientific_boundary": {"test_disjoint_from_training_and_validation": True}})
    return path


def test_independent_completed_sources_bind_one_combined_table(tmp_path):
    full, lite = source(tmp_path, "GIFT"), source(tmp_path, "GIFT-Lite")
    output = tmp_path / "combined"
    result = combine("S3", full, lite, output)
    assert result["rows"] == 2
    assert verify_result(output, "S3")["status"] == "PASS_INTEGRITY_ONLY"
    with (output / "summary/metrics.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["training_regime"] for r in rows] == ["GIFT", "GIFT-Lite"]
    report = json.loads((output / "report.json").read_text())
    assert report["inference_performed_by_this_command"] is False
    assert set(report["source_measurements"]) == {"GIFT", "GIFT-Lite"}
    with pytest.raises(FileExistsError):
        combine("S3", full, lite, output)


# Two sources that disagree on the raw test-observation hash must be refused
# rather than silently merged into one summary table.
def test_different_test_inputs_cannot_be_combined(tmp_path):
    full, lite = source(tmp_path, "GIFT"), source(tmp_path, "GIFT-Lite", raw_hash="2" * 64)
    with pytest.raises(ValueError, match="different test observations"):
        combine("S3", full, lite, tmp_path / "rejected")
