"""CPU-only queue and ETA contracts; children, fields and GPU are never run."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_w1_single as queue
from scripts import estimate_w1_single_eta as eta


def reservation():
    """Declared timing ranges, independent of local historical run/document files."""
    return dict(schema="gift.w1-single-native.cost-forecast.v1", implementation_complete=True,
                methods=dict(anie=dict(max_epochs=5000, estimated_hours_if_full_budget=[29., 35.]),
                             fno2d=dict(estimated_hours=[2.5, 4.5]),
                             fno3d=dict(estimated_hours=[2., 3.5]),
                             gift=dict(estimated_hours=[4., 5.])))


def prepared(tmp_path):
    plan = queue.default_plan(tmp_path)
    queue.atomic_json(tmp_path / "RUN_PLAN.json", plan)
    queue.atomic_json(tmp_path / "COST_FORECAST.json", reservation())
    return plan


def test_fresh_queue_prepares_from_current_protocol_and_external_reservation(tmp_path, monkeypatch):
    """Preparation consumes current documents and canonical observations, no old docs."""
    from training import w1_v2_data as data
    source_root = tmp_path / "evidence"
    window = source_root / "data/train300_validation20_5to8_dt0p04.h5"
    window.parent.mkdir(parents=True)
    window.write_bytes(b"immutable observation contract fixture")
    forecast = source_root / "COST_FORECAST.json"
    queue.atomic_json(forecast, reservation())
    source_bytes = forecast.read_bytes()
    monkeypatch.setattr(queue, "external_source", lambda key: source_root)
    directory = tmp_path / "new_run"
    plan = queue.default_plan(directory)
    expected_binding = dict(schema="gift.w1-sparse.window.v3", points=76,
                            training=dict(ids=list(range(300))), validation=dict(ids=list(range(1000, 1020))))
    monkeypatch.setattr(data, "WindowBank", lambda path: SimpleNamespace(
        binding=dict(expected_binding, file=str(Path(path)))))
    jobs = queue.prepare_run(directory, plan)
    assert list(jobs) == plan["methods"]
    assert queue.PLAN_DOCUMENT == queue.ROOT / "docs/W1_W2_PROTOCOL.md"
    assert (directory / "PROTOCOL_LOCKED.md").read_bytes() == queue.PLAN_DOCUMENT.read_bytes()
    assert Path(plan["immutable_window_source"]) == window
    assert Path(plan["window"]).read_bytes() == window.read_bytes()
    imported = queue.read_json(directory / "COST_FORECAST.json")
    assert imported["methods"] == reservation()["methods"]
    assert imported["forecast_source"]["sha256"] == queue.file_hash(forecast)
    assert imported["plan_file"] == plan["protocol_file"]
    assert forecast.read_bytes() == source_bytes
    assert queue.prepare_run(directory, plan) == jobs


def test_explicit_external_cost_reservation_and_invalid_source_are_visible(tmp_path):
    forecast = tmp_path / "declared.json"
    queue.atomic_json(forecast, reservation())
    plan = queue.default_plan(tmp_path / "new_run", cost_forecast=forecast)
    assert queue.declared_cost_forecast(plan)["methods"] == reservation()["methods"]
    plan["cost_forecast_source"] = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="--cost-forecast"):
        queue.declared_cost_forecast(plan)
    invalid = reservation()
    invalid["methods"]["fno2d"]["estimated_hours"] = [4.5, 2.5]
    queue.atomic_json(forecast, invalid)
    plan["cost_forecast_source"] = str(forecast)
    with pytest.raises(ValueError, match="fno2d"):
        queue.declared_cost_forecast(plan)


def test_only_four_fixed_runs_no_development_or_seed_loops(tmp_path, monkeypatch):
    """One slot per method reaches both cohorts, not fifteen formal repeats."""
    plan = prepared(tmp_path)
    assert plan["developer_runs"] == 0 and plan["training_runs_per_method"] == 1
    assert plan["methods"] == ["fno2d", "fno3d", "GIFT", "anie_ns"]
    assert plan["seeds"] == queue.SEEDS and "anie_f" not in plan["methods"]
    jobs = {method: dict(method=method, seed=queue.SEEDS[method], output=str(tmp_path / method)) for method in plan["methods"]}
    (tmp_path / "JOB_EVENTS.jsonl").write_text(json.dumps({"label": "mocked_child_only", "process_seconds": 0.}) + "\n", encoding="utf-8")
    trained, evaluated = [], []
    def train(directory, job):
        trained.append(job["method"])
        return dict(job=job, status="complete", checkpoint=job["output"] + "/model.pt")
    def child(directory, module, args, label):
        if label == "confirmation_generation":
            assert "PROTOCOL_LOCK.json" in str(args[args.index("--lock") + 1])
            queue.atomic_json(tmp_path / "confirmation/complete.json", {"status": "complete"})
        else:
            assert args.count("--checkpoint") == 4
            evaluated.append(args[args.index("--cohort") + 1])
            queue.atomic_json(Path(args[args.index("--output") + 1]) / "numeric_report.json", {"methods": {}})
        return 0
    monkeypatch.setattr(queue, "execute_job", train)
    monkeypatch.setattr(queue, "call_module", child)
    queue.execute(tmp_path, plan, jobs)
    assert trained == plan["methods"] and evaluated == ["confirmation", "old_test"]
    assert json.loads((tmp_path / "ORCHESTRATOR.json").read_text())["formal_runs"] == 4
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert ".08/.48" in text and "无跨seed均值/SD" in text


def test_scientific_failure_retained_and_infrastructure_blocks(tmp_path):
    """A failed initialization is not silently replaced with another seed."""
    output = tmp_path / "fno2d"
    job = dict(method="fno2d", seed=0)
    queue.atomic_json(output / "FAILURE.json", {"failure_kind": "numerical", "reason": "postAdam nonfinite"})
    assert queue.failure_record(output, job)["status"] == "failed"
    queue.atomic_json(output / "FAILURE.json", {"failure_kind": "infrastructure", "reason": "OOM"})
    import pytest
    with pytest.raises(RuntimeError, match="concrete repair"):
        queue.failure_record(output, job)


def test_eta_reserves_full_anie_cap_without_promised_earlystop(tmp_path):
    """Historical reservations include5000 rather than a saved best epoch525."""
    prepared(tmp_path)
    result = eta.estimate(tmp_path, now=datetime(2026, 10, 4, 4, tzinfo=timezone.utc))
    anie = next(row for row in result["jobs"] if row["method"] == "anie_ns")
    assert anie["max_epochs"] == 5000 and anie["remaining_seconds"] == [29*3600, 35*3600]
    assert result["remaining_hours"] == [38.5, 52.0]
    assert result["estimated_completion_beijing"][0].startswith("2026-10-06")


def test_fno_short_horizon_speed_does_not_become_full75_measurement(tmp_path):
    """Future course stages retain explicit historical assumptions until measured."""
    prepared(tmp_path)
    directory = tmp_path / "training/fno2d_seed0"
    queue.atomic_json(directory / "STATUS.json", {"status": "running", "epoch": 20})
    queue.atomic_json(directory / "HISTORY.json", {"training": [{"epoch": 20, "training_future_steps": 1, "epoch_total_seconds": .2}]})
    result = eta.estimate(tmp_path)
    fno = next(row for row in result["jobs"] if row["method"] == "fno2d")
    full = next(row for row in fno["basis"] if row["horizon"] == 75)
    assert full["epochs_remaining"] == 380 and full["samples"] == 0
    assert full["source"] == "historical_shape_author_recipe_reservation"
    assert full["seconds_per_epoch"][0] > 20


def test_json_running_is_not_proof_of_process_liveness_and_blocked_has_no_date(tmp_path):
    """The monitor must separately inspect PIDs; explicit blockers suppress dates."""
    prepared(tmp_path)
    queue.atomic_json(tmp_path / "ORCHESTRATOR.json", {"status": "needs_attention", "reason": "child interrupted"})
    result = eta.estimate(tmp_path)
    assert result["liveness_requires_external_pid_command_verification"] is True
    assert result["status"] == "blocked_or_unavailable"
    assert result["estimated_completion_beijing"] is None


def test_gift_remaining_phase_budget_not_old_thousand_branch_epochs(tmp_path):
    """Completed native derivative work leaves the declared50/25/25 rollout work."""
    prepared(tmp_path)
    directory = tmp_path / "training/GIFT_seed20260820"
    queue.atomic_json(directory / "generator/COMPLETE.json", {"status": "complete"})
    queue.atomic_json(directory / "branch/history.json", [{"epoch": 400, "phase": 0, "seconds": 1.2}])
    result = eta.estimate(tmp_path)
    gift = next(row for row in result["jobs"] if row["method"] == "GIFT")
    assert gift["generator_epoch"] == 500 and gift["branch_epoch"] == 400
    assert [(row["phase"], row["epochs_remaining"]) for row in gift["basis"]] == [(1, 50), (2, 25), (3, 25)]
