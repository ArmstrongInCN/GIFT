"""Aggregate the current fifty-case, five-method W3 run without dropping slots.

All 250 slots and 750 parameter rows remain explicit, including failed and
pending jobs. Full-population means are emitted only when all fifty cases of
that method completed. Available-case statistics are labelled diagnostics.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
for location in (PROJECT, PROJECT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from scripts import run_w3_parameter_job as job


def bound_file(path, root):
    """Record one own-run artifact using a relocatable relative name and real hash."""
    return dict(path=Path(path).relative_to(root).as_posix(), sha256=job.digest(path),
                bytes=Path(path).stat().st_size)


def completed_training_evidence(root, case_id, method, expected, checkpoint_sha256):
    """Bind real own-run model bytes and the native completion before accepting a neural slot."""
    training, model, completion, _ = job.training_paths(root, case_id, method)
    report = training / ("training_report.json" if method == "GIFT" else "result.json")
    required = [training / "CASE_JOB_BINDING.json", model, report, completion]
    if method != "GIFT":
        required.append(training / "terminal_state.npz")
    if any(not path.is_file() for path in required):
        raise ValueError("completed neural W3 slot requires its actual native model, report and completion")
    if (job.read_json(training / "CASE_JOB_BINDING.json") != expected
            or job.digest(model) != checkpoint_sha256):
        raise ValueError("completed neural W3 model bytes or own-run binding differ")
    commit, native = job.read_json(completion), job.read_json(report)
    if method == "GIFT":
        files = {row["path"]: row for row in commit.get("files", [])}
        if (commit.get("status") != "complete"
                or files.get(report.name, {}).get("sha256", "").lower() != job.digest(report)
                or files.get(model.name, {}).get("sha256", "").lower() != checkpoint_sha256):
            raise ValueError("W3 saved training cost/model commit differs")
    else:
        terminal_sha = job.digest(training / "terminal_state.npz")
        if (commit.get("schema") != "gift.pinn-completion.v1"
                or commit.get("diagnostic_test_only") is not False
                or commit.get("result_sha256", "").lower() != checkpoint_sha256
                or commit.get("terminal_state_sha256", "").lower() != terminal_sha
                or native.get("terminal_state_sha256", "").lower() != terminal_sha):
            raise ValueError("W3 saved PINN cost/model commit differs")
    evidence = [bound_file(path, root) for path in dict.fromkeys(required)]
    return native.get("training_cost"), evidence


def slot_record(root, case, method, binding):
    """Validate a committed measurement or retain its terminal failure/pending state."""
    folder = root / "jobs" / case["case_id"] / job.METHODS[method]
    expected = job.job_binding(Namespace(method=method), binding)
    evidence = []
    complete, failed = folder / "COMPLETE.json", folder / "FAILURE.json"
    if method in ("GIFT", "PINN-SR", "PINN-SR-KC"):
        training, _, _, _ = job.training_paths(root, case["case_id"], method)
        training_failure = training / "FAILURE.json"
    else:
        training = training_failure = None
    if complete.exists():
        if failed.exists() or (training_failure is not None and training_failure.exists()):
            raise ValueError("a W3 slot cannot be both complete and scientifically failed")
        commit = job.read_json(complete)
        record = job.read_json(folder / "result.json")
        identity = dict(record.get("identity", {}))
        if method in ("GIFT", "PINN-SR", "PINN-SR-KC"):
            checkpoint = identity.pop("checkpoint_sha256", "")
            if not isinstance(checkpoint, str) or not re.fullmatch("[0-9a-f]{64}", checkpoint):
                raise ValueError("neural W3 measurement must bind its terminal model")
        if (identity != expected or commit.get("identity") != record.get("identity")
                or commit.get("schema") != "gift.w3-parameter-measurement-commit.v1"
                or commit.get("status") != "complete" or record.get("status") != "complete"
                or record.get("case_id") != case["case_id"] or record.get("method") != method
                or record.get("truth") != case["truth"]):
            raise ValueError("current W3 slot scientific identity differs")
        files = commit.get("files", {})
        if set(files) != {"result.json", "summary.csv", "native_coefficients.csv"}:
            raise ValueError("W3 numerical commit must bind all three numeric outputs")
        for name, expected_hash in files.items():
            if job.digest(folder / name) != expected_hash:
                raise ValueError("current W3 numeric file changed: " + name)
            evidence.append(bound_file(folder / name, root))
        errors = job.relative_errors(record["parameters"], case["truth"])
        if record.get("APE_percent") != errors:
            raise ValueError("current W3 APE differs from actual case truth")
        measured = record.get("measurement_seconds")
        if not isinstance(measured, (float, int)) or not math.isfinite(measured) or measured < 0:
            raise ValueError("W3 measurement cost must be finite and nonnegative")
        result = dict(status="complete", parameters=record["parameters"], APE_percent=errors,
                      measurement_seconds=measured)
        evidence.append(bound_file(complete, root))
        if training is not None:
            result["training_cost"], trained_evidence = completed_training_evidence(root,
                case["case_id"], method, expected, record["identity"]["checkpoint_sha256"])
            evidence.extend(trained_evidence)
    else:
        failure_path = failed if failed.exists() else training_failure
        if failure_path is not None and failure_path.exists():
            failure = job.read_json(failure_path)
            if failure.get("status") != "failed" or failure.get("identity") != expected:
                raise ValueError("terminal W3 failure has an incompatible case/method identity")
            result = dict(status="failed", failure=failure)
            evidence.append(bound_file(failure_path, root))
        else:
            result = dict(status="pending")
    if training is not None and result["status"] != "complete":
        report = training / ("training_report.json" if method == "GIFT" else "result.json")
        if report.is_file():
            native = job.read_json(report)
            result["training_cost"] = native.get("training_cost")
            evidence.append(bound_file(report, root))
    attempts = root / "job_attempts" / case["case_id"] / job.METHODS[method]
    attempt_records = []
    if attempts.exists():
        for attempt in sorted(attempts.iterdir()):
            if not attempt.is_dir():
                continue
            for name in ("START.json", "FINISH.json", "FAILURE.json"):
                path = attempt / name
                if path.exists():
                    attempt_records.append(dict(file=bound_file(path, root), record=job.read_json(path)))
    return dict(case_id=case["case_id"], method=method, **result,
                evidence=evidence, attempts=attempt_records)


def write_table(path, columns, rows):
    """Create a UTF-8 numeric table without replacing an earlier aggregate."""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root, output):
    """Verify the prepared input/protocol bindings and retain the entire fixed population."""
    root = Path(root).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output == PROJECT or PROJECT in output.parents or output == root or output in root.parents:
        raise ValueError("aggregation needs a separate new output directory outside the repository")
    prepared = job.read_json(root / "W3_PREPARATION.json")
    package = prepared.get("dataset_root")
    if package:
        source = Path(package).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("aggregate outputs must stay outside the read-only input package")
    if output.exists():
        raise FileExistsError("a new W3 aggregate output directory is required")
    cohort = job.read_json(root / "data" / "DATA_COHORT_QA.json")
    cases = [dict(case_id=row["case_id"], truth=row["truth"]) for row in cohort["cases"]]
    expected_cases = [f"parameter_{index:03d}" for index in range(50)]
    if ([case["case_id"] for case in cases] != expected_cases
            or [row.get("case_id") for row in prepared.get("cases", [])] != expected_cases):
        raise ValueError("aggregation requires the complete fifty-case prepared cohort")
    slots, parameter_rows, metrics = [], [], []
    for case, prepared_case in zip(cases, prepared["cases"]):
        binding = prepared_case["case_binding"]
        job.require_all_inputs(root, binding)
        for method in job.METHODS:
            slot = slot_record(root, case, method, binding)
            slots.append(slot)
            for parameter in job.PARAMETERS:
                parameter_rows.append(dict(case_id=case["case_id"], method=method, parameter=parameter,
                    truth=case["truth"][parameter], estimate=slot.get("parameters", {}).get(parameter),
                    APE_percent=slot.get("APE_percent", {}).get(parameter), status=slot["status"]))
    for method in job.METHODS:
        for parameter in job.PARAMETERS:
            rows = [r for r in parameter_rows if r["method"] == method and r["parameter"] == parameter]
            values = [row["APE_percent"] for row in rows if row["status"] == "complete"]
            failures = sum(row["status"] == "failed" for row in rows)
            pending = sum(row["status"] == "pending" for row in rows)
            full = len(values) == 50
            metrics.append(dict(method=method, parameter=parameter, case_count=50,
                completed_cases=len(values), failed_rows=failures, pending_rows=pending,
                mean_APE_percent=statistics.mean(values) if full else None,
                median_APE_percent=statistics.median(values) if full else None,
                minimum_APE_percent=min(values) if full else None,
                maximum_APE_percent=max(values) if full else None,
                diagnostic_available_case_mean_APE_percent=statistics.mean(values) if values else None))
    pending = sum(slot["status"] == "pending" for slot in slots)
    failures = sum(slot["status"] == "failed" for slot in slots)
    report = dict(schema="gift.w3-current-aggregate.v1", profile=job.SINGLE_PROFILE,
        status="incomplete" if pending else "complete_with_failures" if failures else "complete",
        slot_count=250, parameter_rows=750, completed_slots=250-pending-failures,
        failed_slots=failures, pending_slots=pending, all_prescribed_cases_retained=True,
        full_population_statistics_require_50_completed_cases=True,
        sources=[bound_file(root / name, root) for name in ("W3_PREPARATION.json", "PROTOCOL_LOCK.json",
                 "FULL_INPUT_GATE.json", "data/DATA_COHORT_QA.json")], slots=slots,
        training_performed=False, numerical_outputs_recomputed=False,
        cost_scope="per-slot saved native training_cost, measurement_seconds, and every own-run attempt; missing costs are not estimated")
    output.mkdir(parents=True)
    write_table(output / "parameter_cases.csv", ["case_id", *job.PARAMETERS],
                [dict(case_id=case["case_id"], **case["truth"]) for case in cases])
    write_table(output / "parameter_results.csv", list(parameter_rows[0]), parameter_rows)
    write_table(output / "method_metrics.csv", list(metrics[0]), metrics)
    job.fixed_json(output / "numeric_report.json", report)
    job.fixed_json(output / "COMPLETE.json", dict(schema="gift.w3-current-aggregate-commit.v1",
        status=report["status"], files={name: job.digest(output / name) for name in (
            "parameter_cases.csv", "parameter_results.csv", "method_metrics.csv", "numeric_report.json")}))
    return dict(status=report["status"], slot_count=250, completed_slots=report["completed_slots"],
                failed_slots=failures, pending_slots=pending, output=str(output))


def main(argv=None):
    """Preview the fixed aggregation population or write a new committed summary."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = aggregate(args.root, args.output) if args.execute else dict(status="plan_only",
        slots=250, parameter_rows=750, root=str(args.root.resolve()), output=str(args.output.resolve()))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
