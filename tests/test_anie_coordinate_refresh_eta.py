"""JSON-only ETA tests: native patience, blocked queues and real audit receipts."""
from datetime import datetime, timezone
import json

from scripts import estimate_anie_coordinate_refresh_eta as eta


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fixture(root):
    write(root, "RUN_PLAN.json", dict(source_profile="aligned_ns_single_v1", newly_trained_methods=["anie_ns"]))
    write(root, "COST_FORECAST.json", dict(historical_seconds_per_epoch=20.,
          postprocessing_reserve_hours=[.5, 2.], final_low_frequency_review_reserve_hours=[0., 4.]))
    write(root, "ORCHESTRATOR.json", dict(status="running", active_job="train_anie_ns_seed7"))


def test_patience_improvement_resets_and_equality_does_not_increment():
    """Native min_delta0 leaves equality unchanged rather than treating it as bad."""
    assert eta.patience_counter([dict(validation_mse=x) for x in [3., 4., 3., 2., 2., 3.]]) == 1


def test_full_budget_and_conditional_earlystop_are_separate(tmp_path):
    """Constant measured speed does not permit promising early stopping."""
    fixture(tmp_path)
    write(tmp_path, "training/anie_ns_seed7/STATUS.json", dict(status="running", epoch=5, best_epoch=1))
    write(tmp_path, "training/anie_ns_seed7/HISTORY.json", dict(
        training=[dict(epoch_total_seconds=10.) for _ in range(5)],
        validation=[dict(validation_mse=x) for x in [1., 2., 2., 2., 2.]]))
    value = eta.estimate(tmp_path, datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert value["remaining_epochs_full_budget"] == 4995
    assert value["patience_counter"] == 4 and value["conditional_patience_remaining_epochs"] == 996
    assert value["rate_basis"].startswith("same_run")
    assert value["earliest_no_further_improvement_completion_beijing"] != value["estimated_completion_beijing"]


def test_blocked_queue_has_no_deterministic_finish_date(tmp_path):
    """A stale training sidecar cannot certify continued execution."""
    fixture(tmp_path)
    write(tmp_path, "ORCHESTRATOR.json", dict(status="needs_attention", reason="trainer exited"))
    value = eta.estimate(tmp_path)
    assert value["status"] == "blocked_or_unavailable"
    assert value["estimated_completion_beijing"] is None


def test_complete_requires_all_passed_content_bound_audits(tmp_path):
    """An empty receipt or changed audit never ends the delivery task."""
    fixture(tmp_path)
    write(tmp_path, "COMPLETION_RECEIPT.json", dict(schema="gift.anie-coordinate-refresh.completion.v1", status="complete"))
    assert not eta.completed_delivery(tmp_path)
    names = ("FINAL_TRAINING_AUDIT.json", "FINAL_RESULT_AUDIT.json", "FINAL_DOCUMENT_AUDIT.json", "VISUAL_QA.json")
    hashes = {name: eta.file_hash(write(tmp_path, name, dict(status="passed"))) for name in names}
    write(tmp_path, "COMPLETION_RECEIPT.json", dict(schema="gift.anie-coordinate-refresh.completion.v1", status="complete", audit_hashes=hashes))
    assert eta.completed_delivery(tmp_path)
    write(tmp_path, "VISUAL_QA.json", dict(status="pending"))
    assert not eta.completed_delivery(tmp_path)
