"""Meaningful JSON-only queue-budget checks; no torch, GPU, weights or dataset IO."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.estimate_w1_eta import JsonReader, Rates, estimate, save_estimate, _make_jobs, _gift_cost

NOW = datetime(2026, 10, 3, 13, 0, tzinfo=timezone.utc)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def queue(tmp_path):
    methods = ["fno2d", "anie_ns", "anie_f", "fno3d", "GIFT"]
    plan = dict(methods=methods, candidates={m: [1., .3] for m in methods}, development_seed=7,
                development_epochs=200, formal_seeds=[11, 22, 33], formal_min_epochs=500,
                formal_max_epochs=1500, validation_interval=20, future_steps=75,
                fno2d_curriculum=dict(stage_epochs=20, horizons=[1, 2, 5, 10, 25, 50, 75]),
                gift_development=dict(generator=[50, 50], branch=[80, 10, 5, 5]),
                gift_formal=dict(generator=[250, 250], branch=[800, 100, 50, 50]),
                confirmation=dict(count=180))
    costs = dict(neural_epoch_seconds={m: 10. for m in methods if m != "GIFT"},
                 gift_generator_epoch_seconds_including_measured_refit=1.,
                 gift_phase_epoch_seconds=[2., 20., 100., 110.],
                 neural_validation_seconds={m: 1. for m in methods if m != "GIFT"},
                 overhead_fraction_range=[0., 0.],
                 postprocessing_seconds_ranges=dict(confirmation_generation=[0., 0.],
                     evaluation_total=[0., 0.], report=[0., 0.]),
                 limitations=["Declared probe cost does not establish convergence."])
    _write(tmp_path / "RUN_PLAN.json", plan)
    _write(tmp_path / "COST_FORECAST.json", costs)
    return tmp_path, plan, costs


def test_slots_are_exact_before_and_after_lock_without_double_formal_budget(queue):
    root, plan, _ = queue
    unlocked = estimate(root, now=NOW)
    assert unlocked["slot_counts"] == dict(development=10, formal=15)
    assert sum(job["placeholder"] for job in unlocked["jobs"]) == 15
    _write(root / "DEVELOPMENT_LOCK.json", dict(selected_configurations={m: dict(candidate=.3) for m in plan["methods"]}))
    locked = estimate(root, now=NOW)
    assert locked["slot_counts"] == unlocked["slot_counts"]
    assert locked["remaining_seconds"] == unlocked["remaining_seconds"]
    assert not any(job["placeholder"] for job in locked["jobs"])


def test_short_horizon_speed_never_claims_full_grid_measured_and_last60_updates_rates(queue):
    root, plan, costs = queue
    directory = root / "development/fno2d_seed7_candidate1"
    _write(directory / "HISTORY.json", dict(training=[dict(epoch=e, training_future_steps=1, seconds=.01)
                                                       for e in range(1, 21)], validation=[]))
    _write(directory / "STATUS.json", dict(epoch=20, status="running"))
    result = estimate(root, now=NOW)
    full = result["forecast_basis"]["fno2d:75"]
    assert full["seconds_per_epoch"] == 10. and full["observed_epochs"] == 0
    assert "COST_FORECAST" in full["source"]
    assert result["forecast_basis"]["fno2d:1"]["seconds_per_epoch"] == .01
    rows = [dict(epoch=e, seconds=float(e), training_future_steps=75) for e in range(1, 81)]
    validation = [dict(epoch=e, seconds=2.) for e in (20, 40, 60, 80)]
    _write(root / "development/anie_ns_seed7_candidate1/HISTORY.json", dict(training=rows, validation=validation))
    reader = JsonReader(root)
    rates = Rates(plan, costs, _make_jobs(reader, plan))
    assert rates.rate("anie_ns", 75) == 50.5  # mean21..80, not whole-run mean40.5
    assert rates.basis["anie_ns:75"]["observed_epochs"] == 60
    assert rates.validation("anie_ns") == 2.


def test_failures_override_stale_completion_remain_visible_and_both_failed_block_dates(queue):
    root, _, _ = queue
    registry = []
    for candidate in (1., .3):
        name = f"anie_ns_seed7_candidate{candidate:.8g}"
        _write(root / "development" / name / "STATUS.json", dict(status="complete", epoch=200, best_val=.1))
        _write(root / "development" / name / "FAILURE.json", dict(failure_kind="numerical", reason="gate failure"))
        registry.append(dict(job=dict(method="anie_ns", seed=7, candidate=candidate), status="complete"))
    _write(root / "DEVELOPMENT.json", registry)
    result = estimate(root, now=NOW)
    failed = [job for job in result["jobs"] if job["method"] == "anie_ns" and job["stage"] == "development"]
    assert len(failed) == 2 and all(job["status"] == "failed" for job in failed)
    assert all(job["remaining_core_seconds"] == [0., 0.] and job["failures"] for job in failed)
    assert result["status"] == "blocked_or_unavailable" and result["estimated_completion_beijing"] is None
    assert any("development candidates failed" in reason for reason in result["blockers"])


def test_needs_attention_and_user_pause_have_no_schedule_dates(queue):
    root, _, _ = queue
    for state in ("needs_attention", "paused_by_user"):
        _write(root / "ORCHESTRATOR.json", dict(status=state, reason="requires repair or explicit resume"))
        result = estimate(root, now=NOW)
        assert result["estimated_completion_beijing"] is None and result["remaining_seconds"] is None
        assert result["status"] == "blocked_or_unavailable"


def test_no_finite_validation_terminal_cannot_be_overridden_by_stale_registry(queue):
    root, _, _ = queue
    for candidate in (1., .3):
        _write(root / f"development/anie_ns_seed7_candidate{candidate:.8g}/STATUS.json",
               dict(status="complete", epoch=200, best_val=None))
    _write(root / "DEVELOPMENT.json", [dict(job=dict(method="anie_ns", seed=7, candidate=value), status="complete")
                                       for value in (1., .3)])
    result = estimate(root, now=NOW)
    assert result["status"] == "blocked_or_unavailable"
    jobs = [j for j in result["jobs"] if j["stage"] == "development" and j["method"] == "anie_ns"]
    assert all(j["status"] == "failed" and j["remaining_core_seconds"] == [0., 0.] for j in jobs)


def test_gift_measured_epoch_already_contains_validation_and_complete_stage_is_removed(queue):
    root, plan, costs = queue
    job = dict(method="GIFT", stage="development", generator_complete=False, branch_complete=True,
               generator_history=[dict(epoch=1, phase=0, seconds=10., validation=dict(seconds=999.))],
               branch_history=[])
    rates = Rates(plan, costs, [job])
    assert _gift_cost(job, plan, rates) == 49 * 10. + 50 * 1.
    assert rates.basis["gift_generator:0"]["seconds_per_epoch"] == 10.
    assert "including_validation" in rates.basis["gift_generator:0"]["source"]


def test_terminal_formal_seed_zeroes_its_slot_and_reports_no_retry_budget(queue):
    root, plan, _ = queue
    _write(root / "DEVELOPMENT_LOCK.json", dict(selected_configurations={m: dict(candidate=1.) for m in plan["methods"]}))
    before = estimate(root, now=NOW)
    _write(root / "formal/anie_f_seed22_candidate1/FAILURE.json", dict(failure_kind="numerical", reason="nonfinite gradient"))
    after = estimate(root, now=NOW)
    assert after["status"] == "estimated"
    slot = next(job for job in after["jobs"] if job["method"] == "anie_f" and job["seed"] == 22)
    assert slot["status"] == "failed" and slot["remaining_core_seconds"] == [0., 0.]
    assert after["remaining_seconds"][1] < before["remaining_seconds"][1]
    assert after["slot_counts"]["formal"] == 15


def test_queue_complete_zeroes_every_postprocessing_budget_and_history_is_appended(queue):
    root, _, costs = queue
    costs["postprocessing_seconds_ranges"] = dict(confirmation_generation=[900, 3600], evaluation_total=[1800, 7200], report=[360, 1800])
    _write(root / "COST_FORECAST.json", costs)
    first = save_estimate(root, now=NOW)
    _write(root / "ORCHESTRATOR.json", dict(status="complete"))
    (root / "COST_FORECAST.json").write_text("unreadable deliberately", encoding="utf-8")
    second = save_estimate(root, now=NOW + timedelta(hours=6))
    assert second["status"] == "complete" and second["remaining_seconds"] == [0., 0.]
    assert second["estimated_completion_beijing"] == ["2026-10-04T03:00+08:00"] * 2
    assert second["change_since_previous"]["remaining_hours_change"] == [-v for v in first["remaining_hours"]]
    assert len((root / "ETA_HISTORY.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_postprocessing_markers_and_partial_events_are_safe_without_binary_io(queue, monkeypatch):
    root, _, costs = queue
    costs["postprocessing_seconds_ranges"] = dict(confirmation_generation=[900, 3600], evaluation_total=[1800, 7200], report=[360, 1800])
    _write(root / "COST_FORECAST.json", costs)
    before = estimate(root, now=NOW)
    _write(root / "confirmation/complete.json", dict(status="complete"))
    _write(root / "evaluation/old_test/numeric_report.json", dict(status="numerical_complete"))
    _write(root / "evaluation/old_test/FIGURES_COMPLETE.json", dict(status="complete"))
    (root / "JOB_EVENTS.jsonl").write_text('{"label":"confirmation_generation","exit_code":0,"process_seconds":100}\n{"unfinished":', encoding="utf-8")
    after = estimate(root, now=NOW)
    assert after["status"] == "estimated" and after["warnings"]
    assert after["remaining_seconds"][0] == before["remaining_seconds"][0] - 900 - 900
    assert all(not name.endswith((".pt", ".h5")) for name in after["sources_read"])
    assert after["estimated_completion_beijing"][0].endswith("+08:00")


def test_missing_full_cost_is_explicitly_unavailable_even_with_fast_early_history(queue):
    root, _, costs = queue
    costs["neural_epoch_seconds"].pop("fno2d")
    _write(root / "COST_FORECAST.json", costs)
    _write(root / "development/fno2d_seed7_candidate1/HISTORY.json",
           dict(training=[dict(epoch=1, seconds=.001, training_future_steps=1)], validation=[]))
    result = estimate(root, now=NOW)
    assert result["estimated_completion_beijing"] is None
    assert any("full-grid epoch cost" in reason for reason in result["blockers"])


def test_gift_phase_display_after_generator_completion_does_not_call_it_branch_phase1(queue):
    root, _, _ = queue
    prefix = root / "development/GIFT_seed7_candidate1"
    _write(prefix / "generator/COMPLETE.json", dict(status="complete"))
    _write(prefix / "generator/history.json", [dict(epoch=100, phase=1, seconds=1.)])
    _write(root / "ORCHESTRATOR.json", dict(status="running", active_job="development_GIFT_seed7_candidate1"))
    result = estimate(root, now=NOW)
    assert result["current_phase"]["native_stage"] == "branch"
    assert result["current_phase"]["native_phase"] == "branch_setup_or_first_uncommitted_epoch"
    assert result["current_phase"]["generator_epoch"] == 100
    assert result["current_phase"]["branch_epoch"] == 0
