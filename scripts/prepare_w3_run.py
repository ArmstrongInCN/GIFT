"""Validate the complete current W3 cohort and prepare a new local run.

ScienceDB observations, sampling caches and historical receipts are opened
read-only. Only run-local protocol, source provenance and input-gate records
are created. No model is trained and no experiment result is manufactured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
for location in (PROJECT, PROJECT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from scripts import generate_w3_parameter_data as generation
from scripts import run_w3_parameter_job as job


def input_paths(root, dataset_root=None, parameter_csv=None):
    """Select catalogue roles or a newly generated run, never a historic default."""
    root = Path(root).expanduser().resolve()
    if root == PROJECT or PROJECT in root.parents:
        raise ValueError("W3 run outputs must stay outside the project repository")
    if dataset_root is not None:
        from gift.paths import data_path
        source = Path(dataset_root).expanduser().resolve(strict=True)
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("W3 run outputs must stay separate from the read-only input package")
        cases = data_path(source, "w3.cases")
        parameters = data_path(source, "w3.parameters")
        if parameter_csv is not None and Path(parameter_csv).resolve() != parameters:
            raise ValueError("the formal catalogue selects its own parameter CSV")
    else:
        source = None
        cases = root / "data"
        parameters = Path(parameter_csv).resolve() if parameter_csv else cases / "parameters.csv"
    return root, source, cases, parameters


def protocol_lock():
    """Declare the current fifty-case scientific protocol before any fitting."""
    return job.current_protocol()


def validate_cohort(root, source, cases_root, parameters):
    """Verify all bytes, numerical arrays and sampling for all fifty ordered cases.

    Recomputed FD fits are data QA, never algorithm scores. The archived QA
    remains byte-for-byte unchanged; newly measured rounding differences are
    recorded in the new preparation record only.
    """
    import numpy as np
    import torch
    cases = generation.parameter_cases(parameters)
    initial_path = cases_root / "INITIAL_CONDITIONS.json"
    initial_hash = job.digest(initial_path)
    if job.read_json(initial_path) != generation.initial_conditions(job.SINGLE_PROFILE):
        raise ValueError("current W3 initial conditions must contain only the fixed ID0 parameters")
    cohort_path = cases_root / "DATA_COHORT_QA.json"
    cohort = job.read_json(cohort_path)
    if (cohort.get("schema") != "gift.w3-single-trajectory-data-cohort-qa.v1"
            or cohort.get("status") != "passed" or cohort.get("profile") != job.SINGLE_PROFILE
            or cohort.get("case_count") != 50 or cohort.get("trajectory_count") != 50
            or cohort.get("parameter_csv_sha256") != job.digest(parameters)
            or cohort.get("initial_conditions_sha256") != initial_hash
            or [item.get("case_id") for item in cohort.get("cases", [])] != [c["case_id"] for c in cases]):
        raise ValueError("complete current fifty-case cohort proof differs")
    actual_directories = sorted(item.name for item in cases_root.iterdir() if item.is_dir())
    if actual_directories != [c["case_id"] for c in cases]:
        raise ValueError("W3 case directory inventory differs from the complete fixed design")
    measured = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for case, row in zip(cases, cohort["cases"]):
            folder, dataset, truth, binding = job.case_input(root, case["case_id"], source)
            lock = job.read_json(folder / "DATA_LOCK.json")
            if (truth != case["truth"] or row.get("truth") != truth
                    or binding.get("data_profile") != job.SINGLE_PROFILE
                    or binding["initial_conditions_sha256"] != initial_hash
                    or binding["parameter_csv_sha256"] != job.digest(parameters)
                    or lock != generation.data_lock(case, initial_hash, job.SINGLE_PROFILE)):
                raise ValueError("case equation/population/initial-condition lock differs: " + case["case_id"])
            for key, expected in {"data_sha256": binding["observation_sha256"],
                    "data_receipt_sha256": binding["data_receipt_sha256"],
                    "data_lock_sha256": binding["data_lock_sha256"],
                    "data_qa_sha256": binding["data_qa_sha256"]}.items():
                if row.get(key) != expected:
                    raise ValueError("cohort member hash differs: " + case["case_id"] + ":" + key)
            qa = generation.audit_case(dataset, lock)
            sampling, sampling_receipt = job.validate_sampling(folder, dataset, binding, deep=True)
            measured.append(dict(case_id=case["case_id"], case_binding=binding,
                sampling_sha256=job.digest(sampling),
                sampling_receipt_sha256=job.digest(job.sampling_paths(folder)[1]),
                finite_all_fields=qa["splits"]["training"]["finite_all_fields"],
                recomputed_fd_oracle_max_ape_percent=float(np.max(qa["unit_operator_oracle"]
                    ["original_fp32_fourth_order_fd_fit"]["ape_percent"])),
                target_recompute_max_absolute_difference=sampling_receipt["target_recompute_max_absolute_difference"],
                target_recompute_relative_l2=sampling_receipt["target_recompute_relative_l2"]))
            print(json.dumps(dict(event="validated_case", case_id=case["case_id"])), flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    return cohort, measured


def prepare(root, dataset_root=None, parameter_csv=None):
    """Commit reproducible run metadata only after every current input passes."""
    root, source, cases_root, parameters = input_paths(root, dataset_root, parameter_csv)
    if not (root / "W3_PREPARATION.json").exists() and any((root / name).exists()
            for name in ("training", "jobs", "job_attempts")):
        raise ValueError("prepare a fresh run before creating training or measurement slots")
    root.mkdir(parents=True, exist_ok=True)
    cohort, measured = validate_cohort(root, source, cases_root, parameters)
    local_cohort = root / "data" / "DATA_COHORT_QA.json"
    job.fixed_json(local_cohort, cohort)
    job.fixed_json(root / "PROTOCOL_LOCK.json", protocol_lock())
    sources = root / "PREPARATION_SOURCES.json"
    # Preserve the first preparation's source provenance; current executions
    # record their own source in job_attempts. Source hashes are not a licence
    # or compatibility gate for ordinary execution or preparation retries.
    if not sources.exists():
        job.fixed_json(sources, dict(schema="gift.w3-preparation-sources.v1",
            source_hashes_are_provenance_only=True,
            files={name: job.digest(PROJECT / name) for name in (
                "scripts/prepare_w3_run.py", "scripts/run_w3_parameter_job.py",
                "scripts/generate_w3_parameter_data.py", "scripts/generate_sampling.py",
                "src/even_full_spectrum_ns.py")}))
    record = dict(schema="gift.w3-preparation.v1", status="passed", profile=job.SINGLE_PROFILE,
        dataset_root=str(source) if source else None, cases_root=str(cases_root),
        input_package_read_only=True, source_cohort_qa_sha256=job.digest(cases_root / "DATA_COHORT_QA.json"),
        parameter_csv_sha256=job.digest(parameters),
        initial_conditions_sha256=job.digest(cases_root / "INITIAL_CONDITIONS.json"),
        protocol_lock_sha256=job.digest(root / "PROTOCOL_LOCK.json"),
        preparation_sources_sha256=job.digest(sources), cases=measured,
        cross_library_target_rtol=5e-6, cross_library_target_atol=5e-7,
        training_performed=False, experiment_results_created=False)
    preparation_path = root / "W3_PREPARATION.json"
    if preparation_path.exists():
        original = job.read_json(preparation_path)
        def scientific_identity(value):
            """Exclude bounded recomputation diagnostics from stable input identity."""
            result = dict(value)
            diagnostics = {"recomputed_fd_oracle_max_ape_percent",
                "target_recompute_max_absolute_difference", "target_recompute_relative_l2"}
            result["cases"] = [{key: item for key, item in row.items() if key not in diagnostics}
                               for row in value["cases"]]
            return result
        if scientific_identity(original) != scientific_identity(record):
            raise ValueError("prepared W3 input/protocol identity changed")
    else:
        job.fixed_json(preparation_path, record)
    # Each real recheck retains its own bounded floating-point diagnostics.
    # The first passed preparation remains the gate's immutable input identity.
    job.fixed_json(root / "preparation_checks" / (uuid.uuid4().hex + ".json"), dict(
        schema="gift.w3-preparation-recheck.v1", status="passed",
        original_preparation_sha256=job.digest(preparation_path),
        current_preparation_source_sha256=job.digest(__file__),
        diagnostics=[{key: row[key] for key in ("case_id", "recomputed_fd_oracle_max_ape_percent",
            "target_recompute_max_absolute_difference", "target_recompute_relative_l2") if key in row}
            for row in measured]))
    gate = dict(schema="gift.w3-single-trajectory-full-input-gate.v1", status="passed",
        profile=job.SINGLE_PROFILE, case_count=50, trajectory_count=50,
        training_trajectories_per_case=1, validation_trajectories_per_case=0,
        case_receipts={row["case_id"]: row["case_binding"]["data_receipt_sha256"] for row in measured},
        cohort_qa_sha256=job.digest(local_cohort), protocol_lock_sha256=record["protocol_lock_sha256"],
        preparation_sha256=job.digest(preparation_path))
    job.fixed_json(root / "FULL_INPUT_GATE.json", gate)
    for row in measured:
        job.require_all_inputs(root, row["case_binding"])
    return dict(status="passed", root=str(root), case_count=50,
        profile=job.SINGLE_PROFILE, gate_sha256=job.digest(root / "FULL_INPUT_GATE.json"))


def main(argv=None):
    """Plan by default; --execute performs the complete read-only input audit."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--parameter-csv", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        result = prepare(args.root, args.dataset_root, args.parameter_csv)
    else:
        root, source, cases, parameters = input_paths(args.root, args.dataset_root, args.parameter_csv)
        result = dict(status="plan_only", root=str(root), dataset_root=str(source) if source else None,
            cases_root=str(cases), parameter_csv=str(parameters), profile=job.SINGLE_PROFILE,
            training_performed=False, outputs_written=0)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
