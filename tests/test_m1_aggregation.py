"""Synthetic complete-job fixtures test aggregation, not scientific reproduction."""
import csv
import json

import pytest

from experiments.formal.m1_equation_identification import aggregate, run
from experiments.formal._shared.figure_evidence import bind_result


@pytest.fixture
def jobs(tmp_path):
    root = tmp_path / "synthetic_not_trained"
    root.mkdir()
    (root / "TEST_FIXTURE.txt").write_text("Synthetic unit-test values; no model training.\n")
    for condition in aggregate.NOISE:
        for method, folder in run.METHODS.items():
            job = root / condition / folder
            job.mkdir(parents=True)
            identity = {"method": method, "condition": condition,
                        "data": {"sha256": "A"*64, "relative_path": condition, "bytes": 1}}
            parameters = dict(nu=.02, beta=-.5, gamma=1.)
            run.write_json(job / "result.json", {"status": "complete", "method": method,
                "condition": condition, "identity": identity, "parameters": parameters,
                "scientific_acceptance": "synthetic_unit_test_only", "elapsed_seconds": 0.,
                "reference_binding": {"training_cost": {"nadam_updates": 31000}}})
            with (job / "summary.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["condition", "method", "parameter", "estimate"])
                writer.writeheader()
                writer.writerows(dict(condition=condition, method=method, parameter=k, estimate=v)
                                 for k, v in parameters.items())
            (job / "native_coefficients.csv").write_text("term,real,imaginary\n")
            run.write_json(job / "COMPLETE.json", {"schema": "gift.independent-m1-job-commit.v1",
                "identity": identity, "files": {name: run.digest(job/name) for name in
                    ("result.json", "summary.csv", "native_coefficients.csv")}})
    return root


def test_full_numeric_commit_and_figure_binding(jobs, tmp_path):
    output = tmp_path / "test_aggregate"
    result = aggregate.aggregate(jobs, output, skip_plots=True)
    assert result["parameter_rows"] == 45 and result["completed_jobs"] == 15
    assert result["training_executed"] is False
    bind_result(output, "M1", ("summary/metrics.csv",))
    # This validates the renderer's actual numeric interface without making a figure.
    from experiments.formal.m1_equation_identification.plot_parameters import load_and_validate
    frame = load_and_validate(output/"summary/metrics.csv")
    assert set(frame.relative_error_percent) == {100., 150., 0.}
    with pytest.raises(FileExistsError):
        aggregate.aggregate(jobs, output, skip_plots=True)


@pytest.mark.parametrize("defect", ["missing", "tampered", "wrong_data", "wrong_summary"])
def test_incomplete_or_inconsistent_jobs_rejected(jobs, tmp_path, defect):
    job = jobs/"noise_000"/"gift"
    if defect == "missing":
        (job/"COMPLETE.json").rename(job/"retained_completion.json")
    elif defect == "tampered":
        with (job/"summary.csv").open("a") as stream:
            stream.write("tamper\n")
    else:
        record = json.loads((job/"result.json").read_text())
        if defect == "wrong_data":
            record["identity"]["data"]["sha256"] = "B"*64
        else:
            record["parameters"]["nu"] = 10.
        (job/"result.json").write_text(json.dumps(record))
        commit = json.loads((job/"COMPLETE.json").read_text())
        commit["identity"] = record["identity"]
        commit["files"]["result.json"] = run.digest(job/"result.json")
        (job/"COMPLETE.json").write_text(json.dumps(commit))
    with pytest.raises((ValueError, FileNotFoundError)):
        aggregate.aggregate(jobs, tmp_path/"rejected", skip_plots=True)
    assert not (tmp_path/"rejected").exists()


def test_plot_failure_preserves_complete_numbers(jobs, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("test-only plotting failure")
    monkeypatch.setattr(aggregate.subprocess, "run", fail)
    output = tmp_path/"plot_failure_test"
    with pytest.raises(RuntimeError, match="test-only"):
        aggregate.aggregate(jobs, output)
    assert (output/"plot_failure.json").is_file()
    bind_result(output, "M1", ("summary/metrics.csv",))
