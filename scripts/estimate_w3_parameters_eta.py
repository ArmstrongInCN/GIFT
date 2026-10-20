"""Estimate W3 delivery from committed phase records; this tool does not inspect processes."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

from scripts.run_w1_single import atomic_json, read_json
from scripts.run_w3_parameters import CONDITIONS

PHASE_REFERENCE = {
    "known": {"pre_nadam": [562.2, 998.1], "pre_lbfgs": [4.9, 7.7], "ado_stridge": [15.9, 22.8],
              "ado_nadam": [665.0, 1197.3], "post_nadam": [2274.6, 3316.7]},
    "open": {"pre_nadam": [767.7, 792.3], "pre_lbfgs": [7.1, 7.9], "ado_stridge": [233.7, 314.8],
             "ado_nadam": [882.0, 1456.4], "post_nadam": [3067.9, 3118.3]},
}


def pinn_progress(output):
    """Read only the latest committed coupled JSON state, never TensorFlow or model tensors."""
    pointer = read_json(Path(output) / "LATEST.json")
    if not pointer:
        return None
    record = read_json(Path(output) / pointer["directory"] / "state.json")
    return None if not record else record["snapshot"]["fsm"]


def remaining_pinn(mode, progress=None, completed=None):
    """Use phase-local counts and measured seconds, with original same-shape ranges as fallback."""
    progress = progress or {"nadam_total": 0, "phase": "pre_nadam", "phase_step": 0, "round": 0,
                            "cost": {"phase_seconds": {}, "stridge_calls": 0}}
    total = progress["nadam_total"]
    counts = {"pre_nadam": min(total, 5000), "ado_nadam": min(max(total - 5000, 0), 6000),
              "post_nadam": max(total - 11000, 0)}
    budgets = {"pre_nadam": 5000, "ado_nadam": 6000, "post_nadam": 20000}
    measured = progress["cost"]["phase_seconds"]
    result = [0., 0.]
    bases = []
    for phase, budget in budgets.items():
        left = max(0, budget - counts[phase])
        if counts[phase] >= 1000 and measured.get(phase, 0.) > 0:
            point = measured[phase] / counts[phase] * left
            bounds, basis = [point * .8, point * 1.25], "same-run committed " + phase
        else:
            samples = [row["training_cost"]["phase_seconds"].get(phase, 0.) for row in (completed or [])
                       if row.get("method") == ("PINN-SR-KC" if mode == "known" else "PINN-SR")]
            if samples and all(value > 0 for value in samples):
                point = statistics.mean(samples) * left / budget
                bounds, basis = [point * .8, point * 1.25], "same-run completed library phases"
            else:
                bounds = [value * left / budget for value in PHASE_REFERENCE[mode][phase]]
                basis = "historical original full-shape phase external estimate"
        result = [a + b for a, b in zip(result, bounds)]
        if left:
            bases.append(basis)
    if not progress.get("lbfgs_summary") and progress["phase"] in ("pre_nadam", "pre_lbfgs"):
        result = [a + b for a, b in zip(result, PHASE_REFERENCE[mode]["pre_lbfgs"])]
    calls_left = 6 - progress["cost"].get("stridge_calls", 0)
    result = [a + b * calls_left / 6 for a, b in zip(result, PHASE_REFERENCE[mode]["ado_stridge"])]
    return result, bases


def estimate(directory):
    """Produce a normal-operation time range; blocked states deliberately have no finish date."""
    directory = Path(directory)
    state = read_json(directory / "ORCHESTRATOR.json", {})
    forecast = read_json(directory / "COST_FORECAST.json", {})
    slots = read_json(directory / "FORMAL.json", [])
    complete = [item for item in slots if item["status"] == "complete"]
    failed = [item for item in slots if item["status"] == "failed"]
    measurements = read_json(directory / "MEASUREMENTS.json", [])
    current = datetime.now(timezone.utc)
    phase = state.get("status", "unknown")
    remaining = [0., 0.]
    bases = []
    progress_rows = []
    if phase in ("needs_attention", "scientific_failures_documents_pending") or failed:
        completion = None
        bases = ["unresolved infrastructure or scientific outcome; no definite completion estimate"]
    elif phase == "complete":
        completion, remaining = [current.isoformat(), current.isoformat()], [0., 0.]
        bases = ["delivery marked complete; verify its final publication receipt separately"]
    else:
        for condition in CONDITIONS:
            for method, mode in (("GIFT", "gift"), ("PINN-SR-KC", "known"), ("PINN-SR", "open")):
                if any(item["condition"] == condition and item["method"] == method for item in complete):
                    continue
                output = directory / "training" / condition / mode
                if method == "GIFT":
                    report = read_json(output / "training_report.json")
                    bounds = [100., 200.] if not report else [0., 0.]
                    bases.append("historical 12000-step generator range; setup/IO allowance")
                else:
                    progress = pinn_progress(output)
                    bounds, basis = remaining_pinn(mode, progress, complete)
                    bases.extend(basis)
                    if progress:
                        progress_rows.append(dict(condition=condition, method=method, phase=progress["phase"],
                            nadam_total=progress["nadam_total"], phase_step=progress["phase_step"],
                            round=progress["round"], remaining_seconds=bounds))
                remaining = [a + b for a, b in zip(remaining, bounds)]
        # Open90 and known4 regressions have very different measured costs.
        # Native readout timers exclude process setup; retain a small allowance.
        completed_pairs = {(item["condition"], item["method"]) for item in measurements
                           if item["status"] == "complete"}
        for method, historical in (("PDE-FIND", [173., 245.]), ("PDE-FIND-KC", [2., 5.])):
            samples = []
            for item in measurements:
                if item["status"] == "complete" and item["method"] == method:
                    report = read_json(Path(item["job"]) / "result.json", {})
                    if report.get("elapsed_seconds", 0.) > 0:
                        samples.append(report["elapsed_seconds"])
            left = sum((condition, method) not in completed_pairs for condition in CONDITIONS)
            bounds = ([statistics.mean(samples) * .8 + 3., statistics.mean(samples) * 1.25 + 5.]
                      if samples else historical)
            remaining = [value + left * cost for value, cost in zip(remaining, bounds)]
            if left:
                bases.append(method + (" same-run completed regression timers" if samples
                                       else " historical original full-shape regression estimate"))
        other_left = sum((condition, method) not in completed_pairs for condition in CONDITIONS
                         for method in ("GIFT", "PINN-SR", "PINN-SR-KC"))
        remaining = [remaining[0] + other_left * 1. + 900,
                     remaining[1] + other_left * 5. + 3600]
        bases.append("15–60 minute original-style publication, default-readout and document/visual audit allowance")
        # The first prepared estimate includes complete setup, data validation,
        # checkpoint IO and document work, which scientific phase timers omit.
        if not slots and not progress_rows:
            remaining = [hours * 3600 for hours in forecast.get("full_delivery_hours", [8., 11.])]
            bases.append("initial original-protocol process/setup/document forecast, no added GPU probe")
        completion = [(current + timedelta(seconds=value)).isoformat() for value in remaining]
    beijing = timezone(timedelta(hours=8))
    payload = dict(schema="gift.w3-parameters.eta.v1", estimated_utc=current.isoformat(), status=phase,
        active_job=state.get("active_job"), completed_training_slots=len(complete), failed_training_slots=len(failed),
        total_training_slots=9, completed_measurements=sum(item["status"] == "complete" for item in measurements),
        total_measurements=15, remaining_seconds=None if completion is None else remaining,
        completion_utc=completion, completion_beijing=None if completion is None else [
            datetime.fromisoformat(value).astimezone(beijing).strftime("%Y-%m-%d %H:%M") for value in completion],
        basis=list(dict.fromkeys(bases)), committed_progress=progress_rows,
        condition="normal continued operation and final result/figure/document audit", estimate_is_not_promise=True,
        liveness_checked=False, process_verification_required_separately=True)
    atomic_json(directory / "ETA_ESTIMATE.json", payload)
    with (directory / "ETA_HISTORY.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
