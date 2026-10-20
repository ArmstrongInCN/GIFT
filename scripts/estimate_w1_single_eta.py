"""Estimate remaining single-run W1 work using JSON sidecars, without GPU access.

Author budgets are fixed; ANIE is reserved through5000 epochs unless its native
earlystop actually fires. Observed horizon/phase wall times replace historical
shape-based reservations. This script does not check or control process liveness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import mean

BEIJING = timezone(timedelta(hours=8))
SEEDS = {"fno2d": 0, "fno3d": 0, "GIFT": 20260820, "anie_ns": 7}
HORIZONS = [1, 2, 5, 10, 25, 50, 75]
GIFT_PHASES = {"generator": [250, 250], "branch": [400, 50, 25, 25]}
GIFT_BASE_SECONDS = {"generator": [.536, .536], "branch": [1.194, 44.924, 195.923, 188.573]}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def horizon(epoch):
    """The locked1-based course; the final380 epochs use the full75-step graph."""
    return HORIZONS[min((epoch - 1) // 20, 6)]


def observed_rate(rows, predicate=lambda row: True):
    """Use up to60 matching committed epoch totals, including native monitoring."""
    values = [row.get("epoch_total_seconds", row.get("seconds")) for row in rows if predicate(row)]
    values = [value for value in values if positive(value)][-60:]
    if not values:
        return None, 0
    average = mean(values)
    return [average, average * (1.25 if len(values) < 10 else 1.20)], len(values)


def remaining_neural(root, method, forecast):
    directory = root / "training" / f"{method}_seed{SEEDS[method]}"
    state = read_json(directory / "STATUS.json", {})
    records = read_json(directory / "HISTORY.json", {})
    rows = records.get("training", [])
    maximum = 5000 if method == "anie_ns" else 500
    progress = min(maximum, max([int(row.get("epoch", 0)) for row in rows] + [int(state.get("epoch", 0))]))
    failure = read_json(directory / "FAILURE.json", {})
    summary = dict(method=method, seed=SEEDS[method], epoch=progress, max_epochs=maximum,
                   status=state.get("status", "pending"), basis=[])
    if state.get("status") == "complete" or failure.get("failure_kind") == "numerical":
        summary["remaining_seconds"] = [0., 0.]
        return summary
    if failure.get("failure_kind") == "infrastructure":
        summary["blocker"] = str(directory / "FAILURE.json")
    budget = forecast["methods"]["anie" if method == "anie_ns" else method]
    hours = budget.get("estimated_hours_if_full_budget", budget.get("estimated_hours"))
    totals = [0., 0.]
    groups = HORIZONS if method == "fno2d" else [75]
    for current in groups:
        count = sum(horizon(epoch) == current for epoch in range(progress + 1, maximum + 1)) if method == "fno2d" else maximum - progress
        if not count:
            continue
        rate, samples = observed_rate(rows, lambda row: row.get("training_future_steps", 75) == current)
        if rate is None:
            if method == "fno2d":
                # One second/epoch is an explicit monitoring/IO reserve; the
                # remaining reservation scales with unrolled graph exposure.
                rate = [1. + max(0., value * 3600 - 500.) / 30360 * current for value in hours]
            else:
                rate = [value * 3600 / maximum for value in hours]
        summary["basis"].append(dict(horizon=current, epochs_remaining=count, samples=samples,
                                     seconds_per_epoch=rate, source="same_run_epoch_total" if samples else "historical_shape_author_recipe_reservation"))
        totals = [totals[i] + count * rate[i] for i in range(2)]
    summary["remaining_seconds"] = totals
    return summary


def remaining_gift(root):
    directory = root / "training/GIFT_seed20260820"
    pipeline = read_json(directory / "PIPELINE.json", {})
    failure = read_json(directory / "FAILURE.json", {})
    summary = dict(method="GIFT", seed=20260820, status=pipeline.get("status", "pending"), basis=[])
    if pipeline.get("status") == "complete" or failure.get("failure_kind") == "numerical":
        summary["remaining_seconds"] = [0., 0.]
        return summary
    if failure.get("failure_kind") == "infrastructure":
        summary["blocker"] = str(directory / "FAILURE.json")
    base_total = sum(sum(budget * rate for budget, rate in zip(GIFT_PHASES[stage], GIFT_BASE_SECONDS[stage])) for stage in GIFT_PHASES)
    totals = [0., 0.]
    for stage, budgets in GIFT_PHASES.items():
        rows = read_json(directory / stage / "history.json", [])
        if isinstance(rows, dict):
            rows = rows.get("history", rows.get("training", []))
        complete = read_json(directory / stage / "COMPLETE.json", {}).get("status") == "complete"
        progress = sum(budgets) if complete else max([int(row.get("epoch", 0)) for row in rows] + [0])
        summary[stage + "_epoch"] = progress
        start = 0
        for phase, budget in enumerate(budgets):
            count = max(0, start + budget - max(progress, start))
            start += budget
            if not count:
                continue
            rate, samples = observed_rate(rows, lambda row: row.get("phase") == phase)
            if rate is None:
                rate = [GIFT_BASE_SECONDS[stage][phase] * value * 3600 / base_total for value in (4., 5.)]
            summary["basis"].append(dict(stage=stage, phase=phase, epochs_remaining=count,
                                         seconds_per_epoch=rate, samples=samples,
                                         source="same_run_native_epoch_wall" if samples else "historical_phase_author_budget_reservation"))
            totals = [totals[i] + count * rate[i] for i in range(2)]
    summary["remaining_seconds"] = totals
    return summary


def completion_hours(now, seconds):
    dates = []
    for index, value in enumerate(seconds):
        moment = (now + timedelta(seconds=value)).astimezone(BEIJING)
        rounded = moment.replace(minute=0, second=0, microsecond=0)
        if index == 1 and moment > rounded:
            rounded += timedelta(hours=1)
        dates.append(rounded.isoformat(timespec="minutes"))
    return dates


def estimate(root, *, now=None):
    """Return a conditional full-budget forecast without reading fields or scores."""
    root = Path(root).resolve(strict=True)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone aware")
    plan = read_json(root / "RUN_PLAN.json")
    if plan.get("profile") != "single_native76_n300" or plan.get("developer_runs") != 0 or plan.get("training_runs_per_method") != 1:
        raise ValueError("not the fixed single-native profile")
    state = read_json(root / "ORCHESTRATOR.json", {})
    result = dict(schema="gift.w1-single-native.eta.v1", estimated_utc=now.isoformat(),
                  estimated_beijing=now.astimezone(BEIJING).isoformat(timespec="minutes"),
                  status="estimated", current_phase=dict(queue_status=state.get("status"), active_job=state.get("active_job")),
                  remaining_hours=None, estimated_completion_beijing=None, blockers=[],
                  liveness_requires_external_pid_command_verification=True,
                  interpretation="Conditional normal-execution budget, not confidence bounds; ANIE earlystop is not predicted.")
    if state.get("status") == "needs_attention":
        result["blockers"].append(state.get("reason", "queue needs attention"))
    forecast = read_json(root / "COST_FORECAST.json")
    jobs = [remaining_gift(root) if method == "GIFT" else remaining_neural(root, method, forecast) for method in plan["methods"]]
    total = [sum(job["remaining_seconds"][i] for job in jobs) for i in range(2)]
    for job in jobs:
        if job.get("blocker"):
            result["blockers"].append(job["blocker"])
        if state.get("active_job") == f"train_{job['method']}_seed{job['seed']}":
            result["current_phase"].update(job)
    # Coarse predeclared1–4h postprocessing reserve, allocated by task. These
    # are engineering assumptions until each task completes, not GPU timings.
    weights = [("confirmation_generation", .2, root / "confirmation/complete.json"),
               ("evaluation_confirmation", .35, root / "evaluation/confirmation/FIGURES_COMPLETE.json"),
               ("evaluation_old_test", .35, root / "evaluation/old_test/FIGURES_COMPLETE.json"),
               ("final_report", .1, root / "RESULTS.md")]
    post = []
    for label, fraction, path in weights:
        seconds = [0., 0.] if path.exists() else [value * 3600 * fraction for value in (1., 4.)]
        total = [total[i] + seconds[i] for i in range(2)]
        post.append(dict(task=label, complete=path.exists(), remaining_seconds=seconds, source="rough_engineering_reserve"))
    result.update(jobs=jobs, postprocessing=post,
                  assumptions=["Serial execution, exact locked recipe, no outages or unplanned scientific retries.",
                               "Same-run up to60 matching horizon/phase epoch wall totals replace historical reservations.",
                               "JSON sidecars alone cannot certify that the process is still alive.",
                               "ANIE remaining upper-budget epochs are reserved even if native earlystop may finish earlier.",
                               "Terminal scientific failures retain their slots and have zero remaining training budget."])
    if state.get("status") == "complete":
        result.update(status="complete", remaining_hours=[0., 0.], estimated_completion_beijing=completion_hours(now, [0., 0.]))
    elif result["blockers"]:
        result["status"] = "blocked_or_unavailable"
    else:
        result.update(remaining_hours=[value / 3600 for value in total], estimated_completion_beijing=completion_hours(now, total))
    return result


def save_estimate(root, *, now=None):
    """Replace the latest forecast and append every check to its audit history."""
    from scripts.run_w1_single import atomic_json
    root = Path(root)
    result = estimate(root, now=now)
    previous = read_json(root / "ETA_ESTIMATE.json", {})
    change = dict(previous_estimated_utc=previous.get("estimated_utc"), completion_beijing_shift_hours=None)
    if previous.get("estimated_completion_beijing") and result.get("estimated_completion_beijing"):
        change["completion_beijing_shift_hours"] = [
            (datetime.fromisoformat(result["estimated_completion_beijing"][i]) - datetime.fromisoformat(previous["estimated_completion_beijing"][i])).total_seconds() / 3600 for i in range(2)]
    result["change_since_previous"] = change
    atomic_json(root / "ETA_ESTIMATE.json", result)
    with (root / "ETA_HISTORY.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = save_estimate(args.root)
    print(json.dumps({key: result[key] for key in ("status", "current_phase", "remaining_hours", "estimated_completion_beijing", "change_since_previous", "blockers")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
