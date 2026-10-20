"""Estimate this extension's remaining work from committed JSON, without GPU use.

This does not test PID liveness: every scheduled check must inspect the actual
parent/wrapper/trainer chain separately before reporting a dated estimate.
Historical reservations are explicit until enough new epoch totals exist.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import hashlib
import math
from pathlib import Path
from statistics import mean

BEIJING = timezone(timedelta(hours=8))


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def atomic_json(path, value):
    """Replace only this run's mutable estimate pointer; history is append only."""
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def observed_rate(rows, predicate=lambda row: True):
    values = [row.get("epoch_total_seconds", row.get("seconds")) for row in rows if predicate(row)]
    values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value) and value > 0][-60:]
    return ([mean(values), mean(values) * (1.3 if len(values) < 10 else 1.2)], len(values)) if values else (None, 0)


def neural_remaining(root, label, forecast):
    directory = root / "training" / (label + "_seed0")
    state, history = read_json(directory / "STATUS.json", {}), read_json(directory / "HISTORY.json", {})
    failure = read_json(directory / "FAILURE.json", {})
    rows = history.get("training", [])
    progress = min(500, max([int(row.get("epoch", 0)) for row in rows] + [int(state.get("epoch", 0))]))
    summary = dict(method=label, epoch=progress, max_epochs=500, status=state.get("status", "pending"))
    if state.get("status") == "complete" or failure and failure.get("failure_kind") != "infrastructure":
        return dict(summary, remaining_seconds=[0., 0.], scientific_failure=bool(failure))
    rate, samples = observed_rate(rows)
    if rate is None:
        rate = [hours * 3600 / 500 for hours in forecast["new_training_hours"][label]]
    summary.update(remaining_seconds=[(500-progress) * value for value in rate], samples=samples,
                   basis="same_run_committed_epoch_total" if samples else "historical_different_input_reservation",
                   seconds_per_epoch=rate)
    if failure:
        summary["blocker"] = failure.get("reason")
    return summary


def gift_remaining(root, forecast):
    directory = root / "training/GIFT-N50_seed20260820"
    pipeline, failure = read_json(directory / "PIPELINE.json", {}), read_json(directory / "FAILURE.json", {})
    summary = dict(method="GIFT-N50", status=pipeline.get("status", "pending"), basis=[])
    if pipeline.get("status") == "complete" or failure and failure.get("failure_kind") != "infrastructure":
        return dict(summary, remaining_seconds=[0., 0.], scientific_failure=bool(failure))
    # Split the upfront fixed reservation according to the full-recipe phase
    # cost. Validation/terminal overhead has a separate generous upper reserve.
    weights = {"generator": [.05, .05], "branch": [.04, .24, .31, .31]}
    totals = [0., 0.]
    for stage, budgets in {"generator": [250, 250], "branch": [400, 50, 25, 25]}.items():
        rows = read_json(directory / stage / "history.json", [])
        if isinstance(rows, dict):
            rows = rows.get("history", rows.get("training", []))
        progress = max([int(row.get("epoch", 0)) for row in rows] + [0])
        summary[stage + "_epoch"] = progress
        start = 0
        for phase, epochs in enumerate(budgets):
            remaining = max(0, start + epochs - max(start, progress))
            start += epochs
            if not remaining:
                continue
            rate, samples = observed_rate(rows, lambda row: row.get("phase") == phase)
            if rate is None:
                rate = [hours * 3600 * weights[stage][phase] / epochs for hours in forecast["new_training_hours"]["GIFT-N50"]]
            summary["basis"].append(dict(stage=stage, phase=phase, remaining_epochs=remaining, samples=samples,
                                         seconds_per_epoch=rate, source="same_run_phase_total" if samples else "historical_phase_reservation"))
            totals = [totals[index] + remaining * rate[index] for index in range(2)]
    summary["remaining_seconds"] = totals
    if failure:
        summary["blocker"] = failure.get("reason")
    return summary


def completed_evaluation(directory):
    """Retire a reserve only for a numeric-bound nonempty JSON export receipt."""
    try:
        report_path = directory / "numeric_report.json"
        report, marker = read_json(report_path, {}), read_json(directory / "FIGURES_COMPLETE.json", {})
        records = marker.get("file_sha256", marker.get("figure_sha256", {}))
        return (report.get("status") == "numerical_complete" and bool(records)
                and all(isinstance(value, str) and len(value) == 64 for value in records.values())
                and marker.get("numeric_report_sha256", "").lower() == hashlib.sha256(report_path.read_bytes()).hexdigest())
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def completed_delivery(root):
    """A filename alone cannot close the remaining scientific/document work."""
    try:
        receipt = read_json(root / "COMPLETION_RECEIPT.json", {})
        if receipt.get("schema") != "gift.w1w2-native.completion-receipt.v1" or receipt.get("status") != "complete":
            return False
        names = ("FINAL_RESULT_AUDIT.json", "FINAL_TRAINING_AUDIT.json", "FINAL_COST_CLOSURE.json",
                 "FINAL_DOCUMENT_AUDIT.json", "VISUAL_QA.json")
        for name in names:
            path = root / name
            record = read_json(path, {})
            if (record.get("status") not in ("pass", "PASS", "passed", "complete") or
                    receipt.get("audit_hashes", {}).get(name, "").lower() != hashlib.sha256(path.read_bytes()).hexdigest()):
                return False
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def estimate(root):
    root = Path(root).resolve(strict=True)
    forecast, orchestrator = read_json(root / "COST_FORECAST.json"), read_json(root / "ORCHESTRATOR.json", {})
    methods = [neural_remaining(root, label, forecast) for label in ("unet", "uno")] + [gift_remaining(root, forecast)]
    generated = read_json(root / "w2_confirmation/complete.json", {})
    truth_hours = [0., 0.] if generated.get("status") == "complete" else forecast["w2_truth_hours"]
    # Generation/evaluation predictions are deliberately conservative reserves,
    # not hidden GPU probes. Full two-cohort markers retire each finished part.
    completed_evaluations = sum(completed_evaluation(root / stage / cohort)
                                for stage in ("w1_evaluation", "w2_evaluation") for cohort in ("confirmation", "old_test"))
    reports_done = completed_delivery(root)
    if reports_done:
        truth_hours = [0., 0.]
        for item in methods:
            item["remaining_seconds"] = [0., 0.]
    finish_hours = [0., 0.] if reports_done else [
        value * (.25 + .75 * (4-completed_evaluations)/4) for value in forecast["evaluation_plot_audit_document_hours"]]
    totals = [sum(item["remaining_seconds"][index] for item in methods) + 3600 * (truth_hours[index]+finish_hours[index]) for index in range(2)]
    moment = datetime.now(timezone.utc)
    previous = read_json(root / "ETA_ESTIMATE.json", {})
    blocked = orchestrator.get("status") == "needs_attention" or any(item.get("blocker") for item in methods)
    result = dict(schema="gift.w1w2-native.eta.v1", generated_utc=moment.isoformat(),
                  generated_beijing=moment.astimezone(BEIJING).isoformat(), active_job=orchestrator.get("active_job"),
                  status=orchestrator.get("status"), normal_operation_conditional=True,
                  process_liveness_verified_by_script=False, methods=methods,
                  w2_truth_remaining_hours=truth_hours, evaluation_plot_audit_document_remaining_hours=finish_hours,
                  completed_evaluations=completed_evaluations, remaining_hours=[value/3600 for value in totals],
                  estimated_finish_beijing=None if blocked else [(moment+timedelta(seconds=value)).astimezone(BEIJING).isoformat() for value in totals],
                  blocker=orchestrator.get("reason") if blocked else None,
                  previous_finish_beijing=previous.get("estimated_finish_beijing"),
                  notes="normal runtime estimate, not a promise; historical different-input reservations remain for unsampled phases")
    result["completion_shift_hours"] = ([(datetime.fromisoformat(result["estimated_finish_beijing"][index]) -
            datetime.fromisoformat(previous["estimated_finish_beijing"][index])).total_seconds()/3600 for index in range(2)]
        if result["estimated_finish_beijing"] and previous.get("estimated_finish_beijing") else None)
    atomic_json(root / "ETA_ESTIMATE.json", result)
    with (root / "ETA_HISTORY.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
