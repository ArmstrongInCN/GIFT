"""Estimate this W1 queue's remaining resource budget from JSON sidecars only.

No model, torch import, HDF5 field, test score or process control is involved.
Dates are rounded outward to hours; the interval is a conditional budget,
not a probability interval or a guarantee of future numerical success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean

BEIJING = timezone(timedelta(hours=8))
SCHEMA = "gift.w1.eta-estimate.v1"


class JsonReader:
    """Cache small declared sidecars, recording missing/in-flight JSON honestly."""

    def __init__(self, root):
        self.root, self.cache, self.sources, self.errors, self.warnings = Path(root), {}, [], [], []

    def read(self, relative, default=None):
        path = self.root / relative
        key = str(path)
        if key not in self.cache:
            if not path.exists():
                self.cache[key] = default
            else:
                try:
                    self.cache[key] = json.loads(path.read_text(encoding="utf-8-sig"))
                    self.sources.append(str(path.relative_to(self.root)))
                except (OSError, ValueError) as error:
                    self.cache[key] = default
                    self.errors.append(f"Cannot read JSON {relative}: {error}")
        return self.cache[key]

    def events(self):
        path = self.root / "JOB_EVENTS.jsonl"
        if not path.exists():
            return []
        rows, lines = [], path.read_text(encoding="utf-8-sig").splitlines()
        self.sources.append("JOB_EVENTS.jsonl")
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                if index == len(lines) - 1:
                    self.warnings.append("Ignored an unfinished last JOB_EVENTS line; its event is not committed.")
                else:
                    self.errors.append(f"Invalid committed JOB_EVENTS line {index + 1}")
        return rows


def _positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _rows(value, key=None):
    if key and isinstance(value, dict):
        value = value.get(key, [])
    return value if isinstance(value, list) else []


def _progress(rows, status):
    return max([int(row.get("epoch", 0)) for row in rows] + [int((status or {}).get("epoch", 0)), 0])


def _make_jobs(reader, plan):
    """Make exactly the declared dev slots and one formal slot per method/seed."""
    development = reader.read("DEVELOPMENT.json", [])
    formal = reader.read("FORMAL.json", [])
    lock = reader.read("DEVELOPMENT_LOCK.json", {}) or {}
    selected = lock.get("selected_configurations", {})
    jobs = []
    for stage in ("development", "formal"):
        seeds = [plan["development_seed"]] if stage == "development" else plan["formal_seeds"]
        for seed in seeds:
            for method in plan["methods"]:
                candidates = plan["candidates"][method] if stage == "development" else [
                    selected.get(method, {}).get("candidate")]
                for candidate in candidates:
                    records = development if stage == "development" else formal
                    matching = [row for row in _rows(records) if row.get("job", {}).get("method") == method
                                and row["job"].get("seed") == seed
                                and (candidate is None or row["job"].get("candidate") == candidate)]
                    record = matching[-1] if matching else {}
                    if candidate is None and record:
                        candidate = record["job"].get("candidate")
                    basename = f"{method}_seed{seed}_candidate{candidate:.8g}" if candidate is not None else None
                    if basename is None:
                        possible = sorted((reader.root / stage).glob(f"{method}_seed{seed}_candidate*"))
                        if len(possible) == 1:
                            basename = possible[0].name
                        elif len(possible) > 1:
                            reader.errors.append(f"Multiple unlocked formal directories for {method}/seed{seed}")
                    prefix = Path(stage) / basename if basename else None
                    job = dict(method=method, stage=stage, seed=seed, candidate=candidate,
                               placeholder=basename is None, directory=str(prefix) if prefix else None,
                               label=f"{stage}_{basename}" if basename else f"{stage}_{method}_seed{seed}_unlocked",
                               status="pending", failures=[], registry_status=record.get("status"),
                               registry_failure=record.get("failure"))
                    failure = None
                    if prefix:
                        failure = reader.read(prefix / "FAILURE.json")
                        if failure:
                            job["failures"].append(failure)
                            job["status"] = "needs_attention" if failure.get("failure_kind") == "infrastructure" else "failed"
                        if method == "GIFT":
                            pipeline = reader.read(prefix / "PIPELINE.json", {}) or {}
                            gen_complete = reader.read(prefix / "generator/COMPLETE.json", {}) or {}
                            branch_complete = reader.read(prefix / "branch/COMPLETE.json", {}) or {}
                            job.update(generator_history=_rows(reader.read(prefix / "generator/history.json", [])),
                                       branch_history=_rows(reader.read(prefix / "branch/history.json", [])),
                                       generator_complete=gen_complete.get("status") == "complete",
                                       branch_complete=branch_complete.get("status") == "complete")
                            if not failure and (pipeline.get("status") == "complete" or
                                                job["generator_complete"] and job["branch_complete"]):
                                job["status"] = "complete"
                            elif not failure and (str(pipeline.get("status", "")).startswith("failed") or
                                                  str(gen_complete.get("status", "")).startswith("failed") or
                                                  str(branch_complete.get("status", "")).startswith("failed")):
                                job["status"] = "failed"
                        else:
                            state = reader.read(prefix / "STATUS.json", {}) or {}
                            history = reader.read(prefix / "HISTORY.json", {}) or {}
                            job.update(history=_rows(history, "training"), validation=_rows(history, "validation"),
                                       epoch=_progress(_rows(history, "training"), state), state=state)
                            if not failure and state.get("status") == "complete":
                                job["status"] = "failed" if state.get("best_val", 0.) is None else "complete"
                    if (record.get("status") in ("failed", "complete") and not failure and job["status"] != "needs_attention"
                            and not (job["status"] == "failed" and record.get("status") == "complete")):
                        job["status"] = record["status"]
                    if job["status"] == "pending" and any(job.get(key) for key in ("epoch", "generator_history", "branch_history")):
                        job["status"] = "running"
                    jobs.append(job)
    return jobs


class Rates:
    """Last60 observed epochs per actual horizon/phase; never transfer early FNO speed to full."""

    def __init__(self, plan, forecast, jobs):
        self.plan, self.forecast, self.samples, self.basis = plan, forecast, {}, {}
        self.assumptions = []
        for job in jobs:
            method = job["method"]
            if method == "GIFT":
                histories = [("gift_generator", job.get("generator_history", [])),
                             ("gift_branch", job.get("branch_history", []))]
                for kind, history in histories:
                    for row in history:
                        key = (kind, int(row.get("phase", 0)))
                        if _positive(row.get("seconds")):
                            self.samples.setdefault(key, []).append(row["seconds"])
            else:
                for row in job.get("history", []):
                    key = (method, int(row.get("training_future_steps", plan.get("future_steps", 150))))
                    if _positive(row.get("seconds")):
                        self.samples.setdefault(key, []).append(row["seconds"])
                for row in job.get("validation", []):
                    if _positive(row.get("seconds")):
                        self.samples.setdefault((method, "validation"), []).append(row["seconds"])

    def measured(self, key):
        sample = self.samples.get(key, [])[-60:]
        return (mean(sample), len(sample)) if sample else (None, 0)

    def rate(self, method, horizon=None, *, kind=None, phase=0):
        key = (kind, phase) if kind else (method, horizon)
        observed, count = self.measured(key)
        label = ":".join(map(str, key))
        if observed is not None:
            source = "same_run_last60_epoch_mean_including_validation" if kind else "same_run_last60_horizon_epoch_training_mean"
            value = observed
        elif kind == "gift_generator":
            value = self.forecast.get("gift_generator_epoch_seconds_including_measured_refit")
            source = "COST_FORECAST_disposable_generator_and_affine_probe"
        elif kind == "gift_branch":
            supplied = self.forecast.get("gift_phase_epoch_seconds", [])
            value = supplied[phase] if len(supplied) > phase else None
            source = "COST_FORECAST_disposable_native_phase_probe"
        else:
            full = self.plan.get("future_steps", 150)
            baseline, baseline_count = self.measured((method, full))
            if baseline is None:
                baseline = self.forecast.get("neural_epoch_seconds", {}).get(method)
            if not _positive(baseline):
                raise ValueError(f"No measured or declared full-grid epoch cost for {method}")
            if method == "fno2d" and horizon != full:
                floor = self.forecast.get("fno2d_epoch_floor_seconds", .05 * baseline)
                if not isinstance(floor, (int, float)) or not 0 <= floor <= baseline:
                    raise ValueError("FNO2D epoch floor must lie between0 and full-grid cost")
                value = floor + (baseline - floor) * horizon / full
                source = "unmeasured_curriculum_horizon_extrapolation_with_fixed_floor"
                self.assumptions.append(f"Unmeasured FNO2D horizon{horizon}: floor={floor:g}s + variable full-cost*h/{full}; not a measured rate.")
            else:
                value = baseline
                source = "same_run_last60_full_grid_training_mean" if baseline_count else "COST_FORECAST_full_grid_probe_or_explicit_assumption"
        if not _positive(value):
            raise ValueError(f"Missing/nonpositive cost for {label}")
        self.basis[label] = dict(seconds_per_epoch=value, source=source, observed_epochs=count,
                                timing_scope="GIFT epoch rows already include affine/validation; do not add validation twice" if kind else
                                "neural training excludes scheduled validation")
        return value

    def validation(self, method):
        value, count = self.measured((method, "validation"))
        source = "same_run_last60_validation_mean"
        if value is None:
            value = self.forecast.get("neural_validation_seconds", {}).get(method)
            source = "COST_FORECAST_validation_assumption"
        if value is None:
            value = .05 * self.rate(method, self.plan.get("future_steps", 150))
            source = "unmeasured_check_assumed_5percent_full_training_epoch"
            self.assumptions.append(f"{method}: no measured validation yet; one check assumed0.05 full training epoch.")
        if not _positive(value):
            raise ValueError("Invalid validation-check cost")
        self.basis[method + ":validation"] = dict(seconds_per_check=value, source=source, observed_checks=count)
        return value


def _neural_cost(job, target, plan, rates):
    progress = job.get("epoch", 0)
    total, full = 0., plan.get("future_steps", 150)
    horizons = plan.get("fno2d_curriculum", {}).get("horizons", [1, 5, 10, 20, 50, 100, 150])
    stage_epochs = plan.get("fno2d_curriculum", {}).get("stage_epochs", 20)
    counts = {}
    for epoch in range(progress + 1, target + 1):
        horizon = horizons[min((epoch - 1) // stage_epochs, len(horizons) - 1)] if job["method"] == "fno2d" else full
        counts[horizon] = counts.get(horizon, 0) + 1
    for horizon, count in counts.items():
        total += count * rates.rate(job["method"], horizon)
    checks = {e for e in range(progress + 1, target + 1) if e % plan["validation_interval"] == 0 or e == 500}
    if target > progress:
        checks.add(target)
    if checks:
        total += len(checks) * rates.validation(job["method"])
    return total


def _gift_cost(job, plan, rates):
    config = plan["gift_" + ("development" if job["stage"] == "development" else "formal")]
    total = 0.
    for kind, counts, history, complete in (
            ("gift_generator", config["generator"], job.get("generator_history", []), job.get("generator_complete")),
            ("gift_branch", config["branch"], job.get("branch_history", []), job.get("branch_complete"))):
        if complete:
            continue
        progress, boundary = _progress(history, {}), 0
        for phase, count in enumerate(counts):
            end = boundary + count
            remaining = max(0, end - max(progress, boundary))
            if remaining:
                total += remaining * rates.rate("GIFT", kind=kind, phase=phase)
            boundary = end
    return total


def _postprocessing(reader, plan, forecast, jobs, events):
    """JSON completion wins; HDF-only partial progress is deliberately not guessed."""
    supplied = forecast.get("postprocessing_seconds_ranges", {})
    defaults = dict(confirmation_generation=[900., 3600.], evaluation_total=[1800., 7200.], report=[360., 1800.])
    successful = {event.get("label"): event for event in events if event.get("exit_code") == 0}
    expected = sum(job["stage"] == "formal" and job["status"] != "failed" for job in jobs)
    records, total = [], [0., 0.]
    for task in ("confirmation_generation", "evaluation_old_test", "evaluation_confirmation", "report"):
        direct = supplied.get(task)
        if direct is None and task.startswith("evaluation_"):
            direct = [value / 2 for value in supplied.get("evaluation_total", defaults["evaluation_total"])]
            source = "explicit_or_default_total_evaluation_assumption_divided_between_two_cohorts"
        else:
            source = "COST_FORECAST_explicit_resource_assumption" if direct is not None else "documented_default_resource_assumption"
        bounds = direct if direct is not None else defaults[task]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in bounds) or bounds[1] < bounds[0]):
            raise ValueError("Invalid postprocessing seconds range for " + task)
        done, fraction = task in successful, 0.
        if task == "confirmation_generation":
            done |= (reader.read("confirmation/complete.json", {}) or {}).get("status") == "complete"
        elif task == "report":
            done |= (reader.root / "RESULTS.md").exists()
        else:
            cohort = task.removeprefix("evaluation_")
            prefix = Path("evaluation") / cohort
            numeric = reader.read(prefix / "numeric_report.json")
            figures = reader.read(prefix / "FIGURES_COMPLETE.json")
            done |= bool(numeric and figures)
            completed = 0
            if (reader.root / prefix / "metrics").exists():
                for path in sorted((reader.root / prefix / "metrics").glob("run_[0-9][0-9].json")):
                    record = reader.read(path.relative_to(reader.root)) or {}
                    if record.get("summary") is not None:
                        completed += 1
            # A coarse declared90% inference /10% scoring+figures allocation.
            fraction = .9 * min(1., completed / expected) if expected else .9
            if numeric:
                fraction = .9
        remaining = [0., 0.] if done else [float(v) * (1 - fraction) for v in bounds]
        total = [total[i] + remaining[i] for i in range(2)]
        records.append(dict(task=task, status="complete" if done else "pending_or_in_progress",
                            remaining_seconds=remaining, source=source, completed_json_fraction=fraction,
                            successful_event_seconds=successful.get(task, {}).get("process_seconds")))
    return total, records


def _completion_hours(now, seconds):
    dates = []
    for index, value in enumerate(seconds):
        moment = (now + timedelta(seconds=value)).astimezone(BEIJING)
        rounded = moment.replace(minute=0, second=0, microsecond=0)
        if index == 1 and moment > rounded:
            rounded += timedelta(hours=1)
        dates.append(rounded.isoformat(timespec="minutes"))
    return dates


def estimate(root, *, now=None):
    """Pure JSON estimate; returning a blocker never changes queue state or retries work."""
    root = Path(root).resolve(strict=True)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    reader = JsonReader(root)
    plan = reader.read("RUN_PLAN.json")
    if not isinstance(plan, dict):
        raise ValueError("RUN_PLAN.json is required")
    orchestrator = reader.read("ORCHESTRATOR.json", {}) or {}
    result = dict(schema=SCHEMA, root=str(root), estimated_utc=now.astimezone(timezone.utc).isoformat(),
                  estimated_beijing=now.astimezone(BEIJING).isoformat(timespec="minutes"),
                  status="estimated", current_phase=dict(queue_status=orchestrator.get("status", "not_started"),
                  active_job=orchestrator.get("active_job")), remaining_seconds=None,
                  remaining_hours=None, estimated_completion_beijing=None, blockers=[],
                  interpretation="Conditional resource interval, not confidence bounds, guaranteed ETA or a prediction of numerical success.")
    if orchestrator.get("status") == "complete":
        result.update(status="complete", remaining_seconds=[0., 0.], remaining_hours=[0., 0.],
                      estimated_completion_beijing=[result["estimated_beijing"]] * 2,
                      assumptions=["Queue completion is authoritative; no generation/evaluation/report budget is added again."],
                      sources_read=reader.sources)
        return result
    forecast = reader.read("COST_FORECAST.json", {}) or {}
    jobs, events = _make_jobs(reader, plan), reader.events()
    result["slot_counts"] = {stage: sum(job["stage"] == stage for job in jobs) for stage in ("development", "formal")}
    for job in jobs:
        if job["status"] == "needs_attention":
            result["blockers"].append(f"Infrastructure failure needs attention: {job['label']}")
    if orchestrator.get("status") in ("needs_attention", "paused_by_user", "paused", "stopped", "cancelled"):
        result["blockers"].append(f"Queue {orchestrator['status']}: {orchestrator.get('reason', 'no authorized running schedule')}")
    for method in plan["methods"]:
        candidates = [job for job in jobs if job["stage"] == "development" and job["method"] == method]
        if candidates and all(job["status"] == "failed" for job in candidates):
            result["blockers"].append(f"{method}: all declared development candidates failed; no configuration can be locked.")
    rates, totals = Rates(plan, forecast, jobs), [0., 0.]
    summaries = []
    try:
        for job in jobs:
            cost = [0., 0.]
            if job["status"] not in ("complete", "failed", "needs_attention"):
                if job["method"] == "GIFT":
                    value = _gift_cost(job, plan, rates)
                    cost = [value, value]
                else:
                    progress = job.get("epoch", 0)
                    if job["stage"] == "development":
                        low = high = plan["development_epochs"]
                    else:
                        high = plan["formal_max_epochs"]
                        low = plan["formal_min_epochs"]
                        if progress >= low:
                            low = min(high, ((progress // plan["validation_interval"]) + 1) * plan["validation_interval"])
                    cost = [_neural_cost(job, target, plan, rates) for target in (low, high)]
            totals = [totals[i] + cost[i] for i in range(2)]
            summary = {key: job.get(key) for key in ("label", "method", "stage", "seed", "candidate", "placeholder", "status", "epoch", "failures", "registry_failure")}
            if job["method"] == "GIFT":
                summary.update(generator_epoch=_progress(job.get("generator_history", []), {}),
                               branch_epoch=_progress(job.get("branch_history", []), {}))
            summary["remaining_core_seconds"] = cost
            summaries.append(summary)
            if job["label"] == orchestrator.get("active_job"):
                result["current_phase"].update(summary)
                if job["method"] == "GIFT":
                    native_stage = "branch" if job.get("generator_complete") or job.get("branch_history") else "generator"
                    result["current_phase"]["native_stage"] = native_stage
                    rows = job.get(native_stage + "_history", [])
                    if not rows:
                        result["current_phase"]["native_phase"] = native_stage + "_setup_or_first_uncommitted_epoch"
                    else:
                        names = ["derivative", "short_rollout", "long_rollout_1", "long_rollout_2"] if native_stage == "branch" else ["generator_initial_phase", "generator_second_phase"]
                        result["current_phase"]["native_phase"] = names[rows[-1].get("phase", 0)]
                else:
                    rows = job.get("history")
                if rows:
                    result["current_phase"].update(last_committed_phase=rows[-1].get("phase"),
                                                   training_future_steps=rows[-1].get("training_future_steps"))
        extra, post = _postprocessing(reader, plan, forecast, jobs, events)
        overhead = forecast.get("overhead_fraction_range", [.10, .25])
        if (len(overhead) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in overhead)
                or overhead[1] < overhead[0]):
            raise ValueError("Invalid overhead fraction range")
        remaining = [totals[i] * (1 + overhead[i]) + extra[i] for i in range(2)]
        result.update(remaining_core_seconds=totals, postprocessing=post, overhead_fraction_range=overhead,
                      conditional_remaining_seconds=remaining, forecast_basis=rates.basis)
    except (KeyError, TypeError, ValueError) as error:
        result["blockers"].append("Insufficient or incompatible cost/config JSON: " + str(error))
    result["blockers"].extend(reader.errors)
    result.update(jobs=summaries, sources_read=reader.sources, warnings=reader.warnings,
                  assumptions=list(dict.fromkeys([
                    "Sequential execution, same device/batches/grid; future convergence, infrastructure and numerical success are unknown.",
                    "Neural formal endpoints are budget scenarios500..1500 (or RUN_PLAN limits); the lower date does not promise early stopping.",
                    "An active formal run beyond min_epochs uses the next scheduled check as an optimistic lower scenario; patience may require further checks.",
                    "GIFT uses its complete declared native phase budget; terminal failed/completed jobs have zero remaining work but remain recorded.",
                    "Last60 epoch/horizon or phase arithmetic means replace this run's probe; GIFT epoch seconds already include validation.",
                    "Unmeasured postprocessing ranges are resource assumptions; partial HDF-only confirmation progress is not inspected.",
                    "Evaluation JSON progress allocates90% to model slots and10% to scoring/figures; no same-run dataset or test scores are read.",
                    "Explicit overhead covers unmeasured setup/checkpoint IO and fluctuations; it does not bound outages or retries.",
                    *forecast.get("limitations", []), *rates.assumptions])))
    if result["blockers"]:
        result["status"] = "blocked_or_unavailable"
    else:
        result.update(remaining_seconds=remaining, remaining_hours=[v / 3600 for v in remaining],
                      estimated_completion_beijing=_completion_hours(now, remaining))
    return result


def save_estimate(root, *, now=None):
    """Atomically replace the latest ETA and append an audit record, without training IO."""
    root = Path(root)
    result = estimate(root, now=now)
    path = root / "ETA_ESTIMATE.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    change = dict(previous_estimated_utc=previous.get("estimated_utc") if previous else None,
                  remaining_hours_change=None, completion_beijing_shift_hours=None)
    if previous and result.get("remaining_hours") is not None and previous.get("remaining_hours") is not None:
        change["remaining_hours_change"] = [result["remaining_hours"][i] - previous["remaining_hours"][i] for i in range(2)]
        change["completion_beijing_shift_hours"] = [
            (datetime.fromisoformat(result["estimated_completion_beijing"][i]) -
             datetime.fromisoformat(previous["estimated_completion_beijing"][i])).total_seconds() / 3600 for i in range(2)]
    result["change_since_previous"] = change
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    with (root / "ETA_HISTORY.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = save_estimate(args.root)
    print(json.dumps({key: result[key] for key in ("status", "remaining_hours", "estimated_completion_beijing",
                                                  "change_since_previous", "current_phase", "blockers")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
