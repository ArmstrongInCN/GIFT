"""Fixed-population current W3 aggregation using labelled synthetic job records."""
from argparse import Namespace
import csv
import json

import pytest

from scripts import aggregate_w3_current as aggregate
from scripts import prepare_w3_run as preparation
from scripts import run_w3_parameter_job as job


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    """Create a coherent mock fifty-case proof; no observation generation or training."""
    root = tmp_path / "run"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "parameters.csv").write_text("synthetic_fixture", encoding="utf-8")
    job.fixed_json(data / "INITIAL_CONDITIONS.json", dict(fixture_only=True))
    cases, rows = [], []
    for index in range(50):
        case = dict(case_id=f"parameter_{index:03d}", truth=dict(nu=.01, beta=1., gamma=1.))
        cases.append(case)
        rows.append(dict(case_id=case["case_id"], case_binding=dict(case_id=case["case_id"],
            data_profile=job.SINGLE_PROFILE, data_receipt_sha256=f"{index:064x}")))
    cohort = dict(schema="gift.w3-single-trajectory-data-cohort-qa.v1", status="passed",
        profile=job.SINGLE_PROFILE, case_count=50, trajectory_count=50, cases=cases)
    job.fixed_json(data / "DATA_COHORT_QA.json", cohort)
    monkeypatch.setattr(preparation, "validate_cohort", lambda *args: (cohort, rows))
    preparation.prepare(root)
    for case, prepared in zip(cases, rows):
        for method in job.METHODS:
            folder = root / "jobs" / case["case_id"] / job.METHODS[method]
            identity = job.job_binding(Namespace(method=method), prepared["case_binding"])
            if method in ("GIFT", "PINN-SR", "PINN-SR-KC"):
                training, model, completion, _ = job.training_paths(root, case["case_id"], method)
                job.fixed_json(training / "CASE_JOB_BINDING.json", identity)
                cost = dict(fixture_only=True, process_seconds=10.)
                if method == "GIFT":
                    model.write_bytes(b"synthetic fixture model, not scientific weights")
                    report = training / "training_report.json"
                    job.fixed_json(report, dict(training_cost=cost))
                    job.fixed_json(completion, dict(status="complete", files=[
                        dict(path=path.name, sha256=job.digest(path)) for path in (model, report)]))
                else:
                    state = training / "terminal_state.npz"
                    state.write_bytes(b"synthetic fixture terminal state, not scientific weights")
                    job.fixed_json(model, dict(training_cost=cost, terminal_state_sha256=job.digest(state)))
                    job.fixed_json(completion, dict(schema="gift.pinn-completion.v1",
                        result_sha256=job.digest(model), terminal_state_sha256=job.digest(state),
                        diagnostic_test_only=False))
                identity = dict(identity, checkpoint_sha256=job.digest(model))
            record = dict(schema="gift.w3-parameter-measurement.v1", status="complete",
                case_id=case["case_id"], method=method, identity=identity, truth=case["truth"],
                parameters=case["truth"], APE_percent=dict(nu=0., beta=0., gamma=0.),
                measurement_seconds=1., fixture_only=True)
            job.fixed_json(folder / "result.json", record)
            for name in ("summary.csv", "native_coefficients.csv"):
                (folder / name).write_text("synthetic_fixture\n", encoding="utf-8")
            job.fixed_json(folder / "COMPLETE.json", dict(schema="gift.w3-parameter-measurement-commit.v1",
                status="complete", identity=identity, files={name: job.digest(folder / name) for name in
                    ("result.json", "summary.csv", "native_coefficients.csv")}))
    return root


def test_current_aggregate_requires_all_fifty_cases_per_method_for_main_statistics(prepared_run, tmp_path):
    """All 250 completed synthetic slots produce 750 rows and complete, bound tables."""
    result = aggregate.aggregate(prepared_run, tmp_path / "summary")
    assert result["status"] == "complete" and result["completed_slots"] == 250
    with (tmp_path / "summary" / "parameter_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 750 and {row["status"] for row in rows} == {"complete"}
    with (tmp_path / "summary" / "method_metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 15 and {row["mean_APE_percent"] for row in metrics} == {"0.0"}
    with pytest.raises(FileExistsError):
        aggregate.aggregate(prepared_run, tmp_path / "summary")


def test_aggregate_preserves_all_attempt_costs_and_source_identity(prepared_run, tmp_path):
    """An interrupted training attempt stays present beside the later completed measurement."""
    attempt = prepared_run / "job_attempts/parameter_000/gift/fixture_attempt"
    job.fixed_json(attempt / "START.json", dict(action="train", source_sha256="source-fixture"))
    job.fixed_json(attempt / "FAILURE.json", dict(status="interrupted", process_seconds=123.))
    aggregate.aggregate(prepared_run, tmp_path / "with_attempts")
    report = job.read_json(tmp_path / "with_attempts/numeric_report.json")
    selected = report["slots"][0]
    assert len(selected["attempts"]) == 2
    failed = next(item for item in selected["attempts"] if item["file"]["path"].endswith("FAILURE.json"))
    assert failed["record"]["process_seconds"] == 123.
    assert failed["file"]["sha256"] == job.digest(attempt / "FAILURE.json")


@pytest.mark.parametrize("defect", ["missing_model", "missing_native_commit", "wrong_model_bytes", "wrong_terminal_state"])
def test_neural_measurements_require_actual_own_run_model_and_native_commit(prepared_run, tmp_path, defect):
    """A metadata-only checkpoint claim cannot complete a neural slot or conceal changed weights."""
    method = "PINN-SR" if defect == "wrong_terminal_state" else "GIFT"
    training, model, completion, _ = job.training_paths(prepared_run, "parameter_000", method)
    if defect == "missing_model":
        model.rename(model.with_name("RETAINED_FIXTURE_MODEL.bin"))
    elif defect == "missing_native_commit":
        completion.rename(completion.with_name("RETAINED_FIXTURE_COMMIT.json"))
    elif defect == "wrong_model_bytes":
        model.write_bytes(b"different fixture weights")
    else:
        (training / "terminal_state.npz").write_bytes(b"different fixture terminal state")
    with pytest.raises(ValueError, match="native model|model bytes|PINN cost/model"):
        aggregate.aggregate(prepared_run, tmp_path / "rejected_model")
    assert not (tmp_path / "rejected_model").exists()


def test_failed_and_pending_slots_remain_in_population(prepared_run, tmp_path):
    """A scientific failure and an unfinished slot cannot disappear into available-case means."""
    failed = prepared_run / "jobs" / "parameter_000" / "gift"
    pending = prepared_run / "jobs" / "parameter_001" / "pde_find"
    # Rename fixture markers rather than deleting evidence to simulate two states.
    (failed / "COMPLETE.json").rename(failed / "SYNTHETIC_COMPLETE_RETAINED.json")
    (pending / "COMPLETE.json").rename(pending / "SYNTHETIC_COMPLETE_RETAINED.json")
    binding = job.read_json(prepared_run / "W3_PREPARATION.json")["cases"][0]["case_binding"]
    job.fixed_json(failed / "FAILURE.json", dict(status="failed", message="synthetic nonfinite failure",
        identity=job.job_binding(Namespace(method="GIFT"), binding), process_seconds=2.))
    result = aggregate.aggregate(prepared_run, tmp_path / "partial")
    assert result["status"] == "incomplete"
    assert (result["completed_slots"], result["failed_slots"], result["pending_slots"]) == (248, 1, 1)
    with (tmp_path / "partial" / "parameter_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 750
    assert sum(row["status"] == "failed" for row in rows) == 3
    with (tmp_path / "partial" / "method_metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    for row in metrics:
        if row["method"] in ("GIFT", "PDE-FIND"):
            assert row["mean_APE_percent"] == ""
            assert row["diagnostic_available_case_mean_APE_percent"] == "0.0"


@pytest.mark.parametrize("rehash", [False, True])
def test_changed_or_rehashed_wrong_ape_blocks_aggregation(prepared_run, tmp_path, rehash):
    """Both physical hash damage and internally inconsistent truth/APE prevent publication."""
    folder = prepared_run / "jobs" / "parameter_000" / "gift"
    path = folder / "result.json"
    record = job.read_json(path)
    record["APE_percent"]["nu"] = 42.
    path.write_text(json.dumps(record), encoding="utf-8")
    if rehash:
        commit = job.read_json(folder / "COMPLETE.json")
        commit["files"]["result.json"] = job.digest(path)
        (folder / "COMPLETE.json").write_text(json.dumps(commit), encoding="utf-8")
    with pytest.raises(ValueError, match="APE differs" if rehash else "file changed"):
        aggregate.aggregate(prepared_run, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
