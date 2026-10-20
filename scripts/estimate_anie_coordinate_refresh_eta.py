"""JSON-only full-budget ETA for the coordinate-aligned ANIE and W1 delivery.

Process liveness is deliberately checked separately. Native patience is reported
only as a conditional scenario; it is not a promised termination time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from scripts.run_w1_single import atomic_json, file_hash, read_json

BEIJING = timezone(timedelta(hours=8))


def dates(now, seconds):
    return [(now + timedelta(seconds=value)).astimezone(BEIJING).isoformat(timespec="minutes")
            for value in seconds]


def patience_counter(validation):
    """Replay the original zero-min-delta stopping rule, including no-op equality."""
    best, counter = math.inf, 0
    for row in validation:
        value = row["validation_mse"]
        if value is None or not math.isfinite(value):
            raise ValueError("nonfinite validation history is a scientific blocker")
        if value < best:
            best, counter = value, 0
        elif value > best:
            counter += 1
    return counter


def completed_delivery(root):
    """A declaration alone is insufficient: verify all final audit receipts."""
    receipt = read_json(root / "COMPLETION_RECEIPT.json", {})
    if receipt.get("schema") != "gift.anie-coordinate-refresh.completion.v1" or receipt.get("status") != "complete":
        return False
    required = {"FINAL_TRAINING_AUDIT.json", "FINAL_RESULT_AUDIT.json", "FINAL_DOCUMENT_AUDIT.json", "VISUAL_QA.json"}
    if set(receipt.get("audit_hashes", {})) != required:
        return False
    for name in required:
        path = root / name
        audit = read_json(path, {})
        if str(audit.get("status", "")).lower() not in {"passed", "pass", "complete"}:
            return False
        if file_hash(path) != receipt["audit_hashes"][name]:
            return False
    return True


def estimate(root, now=None):
    root = Path(root).resolve(strict=True)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("estimate clock must have a timezone")
    plan = read_json(root / "RUN_PLAN.json")
    if plan.get("source_profile") != "aligned_ns_single_v1" or plan.get("newly_trained_methods") != ["anie_ns"]:
        raise ValueError("this estimator requires the coordinate-only refresh protocol")
    orchestration = read_json(root / "ORCHESTRATOR.json", {})
    training = root / "training/anie_ns_seed7"
    current = read_json(training / "STATUS.json", {})
    history = read_json(training / "HISTORY.json", dict(training=[], validation=[]))
    forecast = read_json(root / "COST_FORECAST.json")
    rows = history["training"][-60:]
    samples = [row["epoch_total_seconds"] for row in rows if row.get("epoch_total_seconds", 0) > 0]
    own_rate = len(samples) >= 5
    rate = sum(samples)/len(samples) if own_rate else forecast["historical_seconds_per_epoch"]
    rate_range = [.9 * rate, 1.15 * rate]
    epoch = current.get("epoch", 0)
    finished = current.get("status") == "complete"
    failure = read_json(training / "FAILURE.json", {})
    scientific_failure = failure.get("failure_kind") in ("numerical", "resource")
    result = dict(schema="gift.anie-coordinate-refresh.eta.v1", status="estimated",
                  estimated_utc=now.isoformat(), estimated_beijing=now.astimezone(BEIJING).isoformat(timespec="minutes"),
                  current_phase=dict(queue_status=orchestration.get("status"), active_job=orchestration.get("active_job"),
                                     epoch=epoch, max_epochs=5000, best_epoch=current.get("best_epoch")),
                  rate_basis="same_run_last_up_to60_epoch_wall_totals" if own_rate else "historical_same_shape_native_coordinate_run_extrapolation",
                  own_epoch_samples=len(samples), seconds_per_epoch=rate, seconds_per_epoch_range=rate_range,
                  estimated_completion_beijing=None, remaining_hours=None, blockers=[],
                  interpretation="Normal-execution full5000 budget range; conditional patience scenario is not a promise.",
                  liveness_requires_external_pid_command_verification=True,
                  earliest_no_further_improvement_completion_beijing=None)
    if completed_delivery(root):
        result.update(status="complete", remaining_hours=[0., 0.], estimated_completion_beijing=dates(now, [0., 0.]))
        return result
    if orchestration.get("status") == "needs_attention":
        result["blockers"].append(orchestration.get("reason", "queue needs concrete repair"))
    if failure and not scientific_failure:
        result["blockers"].append(failure.get("reason", "infrastructure interruption"))
    if result["blockers"]:
        result["status"] = "blocked_or_unavailable"
        return result
    remaining_epochs = 0 if finished or scientific_failure else max(0, 5000-epoch)
    post = list(forecast["postprocessing_reserve_hours"])
    if (root / "FINAL_DOCUMENT_AUDIT.json").is_file():
        post = [0., 0.]
    review = list(forecast["final_low_frequency_review_reserve_hours"])
    if (root / "VISUAL_QA.json").is_file():
        review = [0., 0.]
    remaining = [remaining_epochs*rate_range[i] + 3600*(post[i]+review[i]) for i in range(2)]
    result.update(remaining_epochs_full_budget=remaining_epochs, remaining_hours=[v/3600 for v in remaining],
                  estimated_completion_beijing=dates(now, remaining),
                  postprocessing_reserve_hours=post, final_low_frequency_review_reserve_hours=review)
    if not finished and not scientific_failure:
        counter = patience_counter(history["validation"])
        minimum = min(remaining_epochs, max(0, 1000-counter))
        result.update(patience_counter=counter, conditional_patience_remaining_epochs=minimum,
                      earliest_no_further_improvement_completion_beijing=dates(now,
                          [minimum*rate_range[i]+3600*(post[i]+review[i]) for i in range(2)]),
                      patience_scenario_assumption="No future validation improvement or equality; native counter is not reset.")
    return result


def save(root, now=None):
    root = Path(root)
    result = estimate(root, now)
    previous = read_json(root / "ETA_ESTIMATE.json", {})
    shifts = None
    if result["estimated_completion_beijing"] and previous.get("estimated_completion_beijing"):
        shifts = [(datetime.fromisoformat(result["estimated_completion_beijing"][i])-
                   datetime.fromisoformat(previous["estimated_completion_beijing"][i])).total_seconds()/3600 for i in range(2)]
    result["change_since_previous"] = dict(previous_estimated_utc=previous.get("estimated_utc"),
                                           full_budget_completion_shift_hours=shifts)
    atomic_json(root / "ETA_ESTIMATE.json", result)
    with (root / "ETA_HISTORY.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, allow_nan=False, ensure_ascii=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(save(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
