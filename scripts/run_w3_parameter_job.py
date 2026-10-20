"""Execute one clean W3 parameter instance with the unchanged native recipes.

The case receipt binds observations before any training. Equation coefficients
are read only as generation provenance and measurement truth; the native
trainers receive their existing clean-data route, seed and fixed budgets.
This generic job interface contains no external algorithm implementation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
import traceback
import uuid

PROJECT = Path(__file__).resolve().parents[1]
for location in (PROJECT, PROJECT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

METHODS = {"GIFT": "gift", "PDE-FIND": "pde_find", "PDE-FIND-KC": "pde_find_kc",
           "PINN-SR": "pinn_sr", "PINN-SR-KC": "pinn_sr_kc"}
PARAMETERS = ("nu", "beta", "gamma")
PINN_BUDGET = {"pre_nadam": 5000, "rounds": 6, "ado_nadam": 1000,
               "post_nadam": 20000, "lbf_maxiter": 10000, "lbf_maxfun": 10000}
GIFT_BUDGET = {"seed": 2026072301, "phase_steps": [6000, 6000],
               "phase_learning_rates": [.01, .003], "batch_size": 16,
               "training_trajectories": 1, "cutoff": 21, "rank": 8,
               "affine_refit_interval": 500, "validation_interval": 250,
               "checkpoint_interval": 250}
SINGLE_PROFILE = "single_trajectory_terminal_v1"
GIFT_SINGLE_BUDGET = dict(GIFT_BUDGET, validation_trajectories=0,
                          checkpoint_selection="terminal")


def current_protocol():
    """Describe the current contract without importing PyTorch into a PINN environment."""
    return dict(schema="gift.w3-single-trajectory-protocol.v1", profile=SINGLE_PROFILE,
        case_ids=[f"parameter_{index:03d}" for index in range(50)],
        methods=list(METHODS), noise_fraction=0., training_trajectory_ids=[0],
        training_trajectories_per_case=1, validation_trajectories_per_case=0,
        independent_validation=False, checkpoint_selection="terminal",
        generator=GIFT_SINGLE_BUDGET,
        pinn=dict(seed=1234, budget=PINN_BUDGET, train_rows=24000,
            holdout_rows=6000, physics_rows=84000, execution_profile="native_fp32",
            algorithm_profile="paper_ado_v1", checkpoint_selection="terminal"),
        readout=dict(training_id=0, times=[5. + index * .1 for index in range(11)],
            oracle_truth_used_only_for_data_QA_and_APE=True),
        failed_slots_retained=True, parameter_search=False, old_weights_reused=False)


def gift_budget(binding):
    """Select the receipt's explicit protocol without changing historical cases."""
    return GIFT_SINGLE_BUDGET if binding.get("data_profile") == SINGLE_PROFILE else GIFT_BUDGET


def digest(path):
    """Hash one explicitly bound input or output without loading it all."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    """Read one required JSON record; a missing record never selects old data."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixed_json(path, value):
    """Commit one immutable record, permitting an identical own-run read."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("existing immutable record differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def case_input(root, case_id, dataset_root=None):
    """Verify the exact generated case, its QA and observation bytes.

    Paths from receipts are descriptive across hosts. Their relative file is
    resolved under this host's selected case root; missing local data fails.
    """
    if not re.fullmatch(r"parameter_0[0-4][0-9]", case_id):
        raise ValueError("case must be parameter_000 through parameter_049")
    root = Path(root).resolve(strict=True)
    if dataset_root is None:
        input_root = root
        data_root = (root / "data" / case_id).resolve(strict=True)
    else:
        from gift.paths import data_path
        input_root = Path(dataset_root).resolve(strict=True)
        data_root = (data_path(input_root, "w3.cases") / case_id).resolve(strict=True)
    if input_root not in data_root.parents:
        raise ValueError("case data root escapes this run")
    receipt_path = data_root / "DATA_RECEIPT.json"
    receipt = read_json(receipt_path)
    single = receipt.get("schema") == "gift.w3-single-trajectory-data-receipt.v1"
    if (receipt.get("schema") not in ("gift.w3-parameter-data-receipt.v1",
                                      "gift.w3-single-trajectory-data-receipt.v1")
            or receipt.get("status") != "passed" or receipt.get("case_id") != case_id):
        raise ValueError("case requires its passed DATA_RECEIPT")
    relative = receipt["data_file"]["relative_path"]
    if relative != "standard_ns_n64_full_spectrum.h5":
        raise ValueError("case observation filename differs")
    dataset = (data_root / relative).resolve(strict=True)
    if data_root not in dataset.parents:
        raise ValueError("case observations escape their data root")
    for name, expected in ((relative, receipt["data_file"]["sha256"]),
                           ("DATA_LOCK.json", receipt["data_lock_sha256"]),
                           ("DATA_QA.json", receipt["data_qa_sha256"])):
        if digest(data_root / name).lower() != expected.lower():
            raise ValueError("case input checksum differs: " + name)
    if dataset.stat().st_size != receipt["data_file"]["bytes"]:
        raise ValueError("case observation byte count differs")
    lock = read_json(data_root / "DATA_LOCK.json")
    qa = read_json(data_root / "DATA_QA.json")
    if (lock.get("case_id") != case_id or lock.get("truth") != receipt["truth"]
            or lock.get("parameter_csv_sha256") != receipt["parameter_csv_sha256"]
            or qa.get("status") != "passed" or qa.get("case_id") != case_id
            or qa.get("truth") != receipt["truth"]):
        raise ValueError("case generation lock and receipt differ")
    if not single and lock.get("profile") == SINGLE_PROFILE:
        raise ValueError("single-trajectory lock requires its matching receipt schema")
    if single:
        import h5py
        if (receipt.get("profile") != SINGLE_PROFILE or lock.get("profile") != SINGLE_PROFILE
                or lock.get("schema") != "gift.w3-single-trajectory-data-lock.v1"
                or qa.get("schema") != "gift.w3-single-trajectory-data-qa.v1"
                or lock.get("splits", {}).get("training", {}).get("population") != 1
                or lock.get("splits", {}).get("validation", {}).get("population") != 0):
            raise ValueError("single-trajectory case profile or population differs")
        with h5py.File(dataset, "r") as handle:
            if (set(handle.keys()) != {"training"}
                    or handle["training/vorticity"].shape != (1, 501, 64, 64)
                    or handle["training/trajectory_index"][:].tolist() != [0]):
                raise ValueError("single-trajectory observations must contain only training ID0")
    truth = receipt["truth"]
    if set(truth) != set(PARAMETERS) or any(not math.isfinite(truth[k]) or truth[k] <= 0 for k in PARAMETERS):
        raise ValueError("case truth requires three positive finite coefficients")
    binding = {"case_id": case_id, "observation_sha256": receipt["data_file"]["sha256"].lower(),
               "data_receipt_sha256": digest(receipt_path),
               "data_lock_sha256": receipt["data_lock_sha256"].lower(),
               "data_qa_sha256": receipt["data_qa_sha256"].lower(),
               "parameter_csv_sha256": receipt["parameter_csv_sha256"].lower(),
               "initial_conditions_sha256": receipt["initial_conditions_sha256"].lower()}
    if single:
        binding["data_profile"] = SINGLE_PROFILE
    return data_root, dataset, truth, binding


def training_paths(root, case_id, method):
    """Map three neural methods to independent native own-run journals."""
    if method == "GIFT":
        key, model, completion, latest = "gift", "gift_main.pt", "manifest.json", "checkpoints/LATEST.json"
    elif method in ("PINN-SR", "PINN-SR-KC"):
        key = "known" if method == "PINN-SR-KC" else "open"
        model, completion, latest = "result.json", "COMPLETED.json", "LATEST.json"
    else:
        raise ValueError("PDE-FIND methods are measured directly; no neural training phase")
    output = Path(root).resolve() / "training" / case_id / key
    return output, output / model, output / completion, output / latest


def require_all_inputs(root, binding):
    """Require the master fifty-case gate before fitting any one instance.

    A remote host receives these unchanged master proofs together with its
    own selected case. It need not download observations for unrelated slots.
    """
    root = Path(root).resolve(strict=True)
    gate = read_json(root / "FULL_INPUT_GATE.json")
    cohort_path = root / "data" / "DATA_COHORT_QA.json"
    cohort = read_json(cohort_path)
    expected_cases = {"parameter_%03d" % index for index in range(50)}
    receipts = gate.get("case_receipts", {})
    if (gate.get("status") != "passed" or gate.get("case_count") != 50
            or set(receipts) != expected_cases or cohort.get("status") != "passed"
            or gate.get("cohort_qa_sha256", "").lower() != digest(cohort_path)
            or receipts[binding["case_id"]].lower() != binding["data_receipt_sha256"]):
        raise ValueError("training requires the passed master fifty-case input gate and matching local receipt")
    if (binding.get("data_profile") == SINGLE_PROFILE
            or gate.get("schema") == "gift.w3-single-trajectory-full-input-gate.v1"
            or cohort.get("schema") == "gift.w3-single-trajectory-data-cohort-qa.v1"):
        if (gate.get("schema") != "gift.w3-single-trajectory-full-input-gate.v1"
                or cohort.get("schema") != "gift.w3-single-trajectory-data-cohort-qa.v1"
                or gate.get("profile") != SINGLE_PROFILE or cohort.get("profile") != SINGLE_PROFILE
                or gate.get("trajectory_count") != 50 or cohort.get("trajectory_count") != 50
                or gate.get("training_trajectories_per_case") != 1
                or gate.get("validation_trajectories_per_case") != 0
                or binding.get("data_profile") != SINGLE_PROFILE):
            raise ValueError("single-trajectory training requires its own fifty-trajectory protocol gate")
    if "preparation_sha256" in gate:
        preparation_path = root / "W3_PREPARATION.json"
        protocol_path = root / "PROTOCOL_LOCK.json"
        if (digest(preparation_path) != gate["preparation_sha256"]
                or digest(protocol_path) != gate.get("protocol_lock_sha256")):
            raise ValueError("prepared W3 protocol or validation record changed")
        preparation = read_json(preparation_path)
        selected = [row for row in preparation.get("cases", []) if row.get("case_id") == binding["case_id"]]
        if (read_json(protocol_path) != current_protocol() or preparation.get("status") != "passed"
                or len(selected) != 1 or selected[0].get("case_binding") != binding):
            raise ValueError("prepared W3 scientific protocol or case identity differs")


def gift_arguments(output, resume, device, *, single_trajectory=False):
    """Use the original W3 one-trajectory generator settings exactly."""
    values = ["--condition", "noise_000", "--data-profile", "regenerated",
            "--training-trajectories", "1", "--output", str(output), "--device", device,
            "--seed", "2026072301", "--phase-steps", "6000", "6000",
            "--phase-learning-rates", ".01", ".003", "--batch-size", "16",
            "--affine-refit-interval", "500", "--validation-interval", "250",
            "--checkpoint-interval", "250", "--resume" if resume else "--run-training"]
    if single_trajectory:
        values.extend(["--validation-trajectories", "0", "--checkpoint-selection", "terminal"])
    return values


def pinn_arguments(output, resume, device):
    """Use full native PINN population and unchanged phase limits."""
    values = ["--execute", "--mode", output.name, "--device", device,
              "--condition", "noise_000", "--data-profile", "regenerated",
              "--output", str(output), "--nadam-interval", "1000"]
    if resume:
        values.append("--resume")
    return values


def sampling_paths(data_root):
    """Keep the existing clean PINN reader's path inside this case only."""
    output = Path(data_root) / "auxiliary" / "pinn_sampling" / "noise_000_seed1234.npz"
    receipt = output.with_suffix(".receipt.json")
    return output, receipt


def validate_sampling(data_root, dataset, binding, *, deep=False):
    """Bind the cache; deep preparation also reconstructs every design row and target.

    Targets are physical [u,v,omega] from the saved ID0 fields. Their float64
    FFT construction is fixed explicitly in generate_sampling; no truth
    coefficient is used to build the velocity or to label a sampled row.
    """
    import numpy as np
    target, receipt_path = sampling_paths(data_root)
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != "gift.w3-parameter-sampling-receipt.v1"
            or receipt.get("status") != "passed"
            or receipt.get("case_binding") != binding or receipt.get("sampling_sha256") != digest(target)
            or any(receipt.get(key) != value for key, value in {
                "seed": 1234, "sensors": 500, "time_samples": 60, "train_rows": 24000,
                "holdout_rows": 6000, "physics_rows": 84000,
                "truth_parameters_passed_to_sampling": False}.items())):
        raise ValueError("case sampling receipt/data identity differs")
    with np.load(target, allow_pickle=False) as arrays:
        if (arrays["measurement_coordinates"].shape != (30000, 3)
                or arrays["targets"].shape != (30000, 3)
                or arrays["train_indices"].shape != (24000,)
                or arrays["validation_indices"].shape != (6000,)
                or arrays["lhs_coordinates"].shape != (60000, 3)):
            raise ValueError("case sampling population differs")
        if deep:
            import h5py
            from scripts.generate_sampling import sampling_design, measurement_targets
            with h5py.File(dataset, "r") as source:
                times = np.asarray(source["training/time"], dtype=np.float32)
                expected = sampling_design(times, 64, np.random.RandomState(1234))
                for key, value in expected.items():
                    if key not in arrays or not np.array_equal(arrays[key], value):
                        raise ValueError("case sampling design differs: " + key)
                targets = measurement_targets(source, expected)
            actual = arrays["targets"]
            # Allow FFT-library rounding only; report the measured difference
            # in preparation evidence rather than rewriting the published cache.
            if not np.isfinite(actual).all() or not np.allclose(actual, targets, rtol=5e-6, atol=5e-7):
                raise ValueError("case sampling targets differ from saved observations")
            receipt = dict(receipt, target_recompute_max_absolute_difference=float(
                np.max(np.abs(actual.astype(np.float64) - targets.astype(np.float64)))),
                target_recompute_relative_l2=float(np.linalg.norm(actual.astype(np.float64)-targets)
                    / max(np.linalg.norm(targets.astype(np.float64)), np.finfo(np.float64).tiny)))
    return target, receipt


def prepare_sampling(root, case_id, data_root, dataset, binding):
    """Generate seed1234 sensors/LHS once, keeping only labels case-dependent."""
    import shutil
    from scripts.generate_sampling import main as generate_sampling
    target, receipt_path = sampling_paths(data_root)
    if receipt_path.exists():
        validate_sampling(data_root, dataset, binding)
        return {"status": "complete", "sampling": str(target), "cached": True}
    preparation = Path(root).resolve() / "preparation" / case_id / "sampling"
    completed = preparation / "COMPLETE.json"
    if not completed.exists():
        arguments = ["--input", str(dataset), "--output", str(preparation), "--execute"]
        if (preparation / "run.json").exists():
            arguments.append("--resume")
        generate_sampling(arguments)
    record = read_json(completed)
    source = preparation / "sampling.npz"
    source_hash = digest(source)
    if record.get("sampling_sha256", "").lower() != source_hash:
        raise ValueError("prepared sampling bytes differ")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != source_hash:
            raise ValueError("existing case sampling differs")
    else:
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".copying")
        shutil.copyfile(source, temporary)
        if digest(temporary) != source_hash:
            raise ValueError("sampling copy differs")
        temporary.rename(target)
    fixed_json(receipt_path, {"schema": "gift.w3-parameter-sampling-receipt.v1", "status": "passed",
        "case_binding": binding, "sampling_sha256": source_hash,
        "generation_complete_sha256": digest(completed), "seed": 1234,
        "sensors": 500, "time_samples": 60, "train_rows": 24000, "holdout_rows": 6000,
        "physics_rows": 84000, "physical_velocity_basis": "unit_Biot_Savart",
        "truth_parameters_passed_to_sampling": False})
    validate_sampling(data_root, dataset, binding)
    return {"status": "complete", "sampling": str(target), "cached": False}


def job_binding(args, binding):
    """Bind scientific inputs independently of device and claim provenance."""
    value = {"schema": "gift.w3-parameter-job-binding.v1", "case_binding": binding,
             "method": args.method, "clean_reader_condition": "noise_000",
             "truth_used_as_training_labels_or_initial_coefficients": False}
    if args.method == "GIFT":
        value["training_configuration"] = gift_budget(binding)
    elif args.method.startswith("PINN"):
        value["training_configuration"] = {"seed": 1234, "budget": PINN_BUDGET,
            "train_rows": 24000, "physics_rows": 84000, "execution_profile": "native_fp32",
            "algorithm_profile": "paper_ado_v1", "terminal_checkpoint": True}
    else:
        value["training_configuration"] = {"noise": 0., "SVD_denoising": False,
            "columns": 4 if args.method == "PDE-FIND-KC" else 90,
            "regression_seed": 0, "native_regression_configuration_unchanged": True}
    return value


def validate_training(root, case_id, method, data_root, dataset, binding):
    """Inspect a completed same-case native model, full budget and observation identity."""
    output, model, completion, latest = training_paths(root, case_id, method)
    if (output / "FAILURE.json").exists():
        raise RuntimeError("this unique training slot has a recorded scientific failure")
    if not model.is_file() or not completion.is_file():
        raise ValueError("case training has no completed native export")
    recorded = read_json(output / "CASE_JOB_BINDING.json")
    if recorded.get("case_binding") != binding or recorded.get("method") != method:
        raise ValueError("case training journal binding differs")
    if method == "GIFT":
        import torch
        report = read_json(output / "training_report.json")
        if (report["training_cost"]["phase_updates"] != [6000, 6000]
                or report["optimization"]["seed"] != 2026072301):
            raise ValueError("case GIFT training budget or seed differs")
        payload = torch.load(model, map_location="cpu", weights_only=True)
        training_data = payload.get("training_data", {})
        recorded_file_hash = training_data.get("sha256")
        if (recorded_file_hash is not None and recorded_file_hash.lower() != binding["observation_sha256"]
                or payload.get("model_configuration", {}).get("cutoff") != 21
                or payload.get("model_configuration", {}).get("rank") != 8):
            raise ValueError("case GIFT checkpoint observations or model differ")
        # Native exports bind the actual derivative/state populations rather
        # than assigning a whole-file SHA. Recompute that existing identity
        # from this case to check a new terminal export without rewriting it.
        if recorded_file_hash is None:
            import h5py
            from experiments.formal._shared.gift_generator_training import GeneratorTrainingConfig, selected_observation_arrays
            with h5py.File(dataset, "r") as handle:
                single = binding.get("data_profile") == SINGLE_PROFILE
                configuration = GeneratorTrainingConfig(training_trajectories=1,
                    validation_trajectories=0 if single else 20,
                    checkpoint_selection="terminal" if single else "validation_best")
                _, _, actual = selected_observation_arrays(handle, configuration)
            if training_data.get("selected_arrays_sha256") != actual:
                raise ValueError("case GIFT learned arrays belong to other observations")
        if binding.get("data_profile") == SINGLE_PROFILE:
            configuration = payload.get("training_configuration", {})
            phases = report["optimization"]["phase_records"]
            if (configuration.get("validation_trajectories") != 0
                    or configuration.get("checkpoint_selection") != "terminal"
                    or training_data.get("trajectory_ids") != [0]
                    or any(row.get("selected_step") != 6000 or row.get("checkpoint_selection") != "terminal"
                           for row in phases)
                    or len(phases) != 2 or report["data"]["validation"] is not None
                    or report["data"]["validation_trajectory_ids"] != []
                    or report["metrics"]["independent_validation_derivative_relative_l2"] is not None
                    or report["information_firewall"]["opened_hdf5_datasets"] != [
                        "training/time", "training/trajectory_index", "training/vorticity"]):
                raise ValueError("single-trajectory GIFT requires fixed terminal phases and no holdout")
    else:
        from adapters.pinn_runtime import read_inputs
        report = read_json(model)
        cost = report["training_cost"]
        if (report["requested_budget"] != PINN_BUDGET or cost["nadam_updates"] != 31000
                or cost["stridge_calls"] != 6 or report["diagnostic_test_only"]):
            raise ValueError("case PINN complete budget differs")
        backend = report["terminal_snapshot"]["identity"]["backend"]
        profile = backend["profile"]
        if (profile["initialization"] != "native_Xavier_seed1234" or profile["tensorflow_tf32"]
                or backend["algorithm_profile"] != "paper_ado_v1"
                or profile.get("execution_profile", "native_fp32") != "native_fp32"):
            raise ValueError("case PINN native profile differs")
        sampling, _ = validate_sampling(data_root, dataset, binding)
        actual_inputs = read_inputs(sampling, dataset, condition="noise_000")[-1]
        if (backend["inputs"]["selected_arrays_sha256"] != actual_inputs["selected_arrays_sha256"]
                or backend["inputs"]["selection"] != actual_inputs["selection"]):
            raise ValueError("case PINN learned arrays belong to other observations")
        if digest(output / "terminal_state.npz").lower() != report["terminal_state_sha256"].lower():
            raise ValueError("case PINN terminal model bytes differ")
    return {"status": "complete", "case_id": case_id, "method": method,
        "checkpoint": str(model), "checkpoint_sha256": digest(model),
        "completion_sha256": digest(completion), "training_cost": report["training_cost"],
        "fresh_initialization": True, "seed": 2026072301 if method == "GIFT" else 1234}


def train_case(args, data_root, dataset, binding):
    """Call the existing native trainer; only its own journal can continue."""
    output, model, completion, latest = training_paths(args.root, args.case, args.method)
    identity_path = output / "CASE_JOB_BINDING.json"
    expected = job_binding(args, binding)
    if (output / "FAILURE.json").exists():
        raise RuntimeError("this unique training slot has a recorded scientific failure")
    pending = output.parent / (output.name + "_CASE_JOB_BINDING.json")
    # A hard process exit can occur after the native journal or terminal export
    # is committed but before its adjacent descriptive identity is installed.
    # Recover that identity only from the pre-existing identical pending record.
    if output.is_dir() and not identity_path.exists() and pending.exists():
        if read_json(pending) != expected:
            raise ValueError("pending case identity differs from the native journal")
        fixed_json(identity_path, expected)
    if completion.exists():
        return validate_training(args.root, args.case, args.method, data_root, dataset, binding)
    resume = latest.exists()
    if args.resume != resume:
        raise ValueError("--resume must match the existence of this slot's native journal")
    if resume:
        if read_json(identity_path) != expected:
            raise ValueError("own case journal has incompatible observations or budget")
    elif output.exists():
        raise FileExistsError("fresh case training requires a new native output directory")
    # The native trainers create the directory before their initial snapshot.
    # Place the binding beside it first, then bind the native journal after creation.
    fixed_json(pending, expected)
    if args.method == "GIFT":
        from experiments.formal.train_low_generator import parse_args, run
        result = run(parse_args(gift_arguments(output, resume, "cpu" if args.device == "cpu" else "cuda",
            single_trajectory=binding.get("data_profile") == SINGLE_PROFILE)))
    else:
        validate_sampling(data_root, dataset, binding)
        from training.train_pinn import main
        old_arguments = sys.argv
        sys.argv = ["training.train_pinn", *pinn_arguments(output, resume,
                      "cpu" if args.device == "cpu" else "gpu")]
        try:
            main()
        finally:
            sys.argv = old_arguments
        result = {"status": "complete"}
    fixed_json(identity_path, expected)
    return validate_training(args.root, args.case, args.method, data_root, dataset, binding)


def relative_errors(parameters, truth):
    """Compute actual-case percentage errors without the old single truth tuple."""
    if set(parameters) != set(PARAMETERS) or any(not math.isfinite(parameters[k]) for k in PARAMETERS):
        raise FloatingPointError("requires exactly three finite reported parameters")
    return {key: 100. * abs(parameters[key] - truth[key]) / abs(truth[key]) for key in PARAMETERS}


def measure_case(args, data_root, dataset, truth, binding):
    """Reuse original readouts, then commit case truth, errors and complete coefficients."""
    from experiments.formal.w3_equation_identification.run import calculate
    output = Path(args.root).resolve() / "jobs" / args.case / METHODS[args.method]
    expected = job_binding(args, binding)
    if args.method in ("GIFT", "PINN-SR", "PINN-SR-KC"):
        trained = validate_training(args.root, args.case, args.method, data_root, dataset, binding)
        checkpoint = Path(trained["checkpoint"])
        expected["checkpoint_sha256"] = trained["checkpoint_sha256"]
    else:
        checkpoint = None
    completion = output / "COMPLETE.json"
    if completion.exists():
        commit = read_json(completion)
        if commit.get("identity") != expected:
            raise ValueError("completed case measurement uses another data/model identity")
        for name, checksum in commit["files"].items():
            if digest(output / name) != checksum:
                raise ValueError("case measurement file changed")
        return read_json(output / "result.json")
    if (output / "FAILURE.json").exists():
        raise RuntimeError("this unique measurement slot has a scientific failure")
    output.mkdir(parents=True, exist_ok=True)
    fixed_json(output / "CASE_JOB_BINDING.json", expected)
    result_path = output / "result.json"
    if result_path.exists():
        record = read_json(result_path)
        if record.get("identity") != expected or record.get("truth") != truth or record.get("status") != "complete":
            raise ValueError("uncommitted measurement belongs to another case/model")
        parameters, measured = record["parameters"], record["raw_measurement"]
        errors = relative_errors(parameters, truth)
        if record.get("APE_percent") != errors or measured.get("parameters") != parameters:
            raise ValueError("uncommitted measurement parameters or actual-case errors differ")
    else:
        began = time.perf_counter()
        measured = calculate(args.method, "noise_000", dataset, checkpoint,
                             "cpu" if args.device == "cpu" else "cuda",
                             data_record={"sha256": binding["observation_sha256"].upper()})
        process_seconds = time.perf_counter() - began
        parameters = measured["parameters"]
        errors = relative_errors(parameters, truth)
        record = {"schema": "gift.w3-parameter-measurement.v1", "status": "complete",
                  "case_id": args.case, "method": args.method, "identity": expected,
                  "truth": truth, "parameters": parameters, "APE_percent": errors,
                  "measurement_seconds": process_seconds, "raw_measurement": measured,
                  "measurement_host": socket.gethostname(), "measurement_device": args.device}
    fixed_json(output / "result.json", record)
    files = {"result.json": digest(output / "result.json")}
    tables = {"summary.csv": (["case_id", "method", "parameter", "truth", "estimate", "APE_percent"],
               [[args.case, args.method, key, truth[key], parameters[key], errors[key]] for key in PARAMETERS]),
              "native_coefficients.csv": (["term", "real", "imaginary"],
               [[row["term"], row["real"], row["imaginary"]] for row in measured.get("native_coefficients", [])])}
    for filename, (header, rows) in tables.items():
        text = io.StringIO(newline="")
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        path = output / filename
        if path.exists():
            if path.read_text(encoding="utf-8") != text.getvalue():
                raise ValueError("existing measurement table differs")
        else:
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(text.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
        files[filename] = digest(path)
    fixed_json(completion, {"schema": "gift.w3-parameter-measurement-commit.v1", "status": "complete",
                           "identity": expected, "files": files})
    return record


def execute(args):
    """Select the exact case before dispatch; never fall back to a released input."""
    dataset_root = getattr(args, "dataset_root", None)
    if dataset_root is not None:
        selected = Path(dataset_root).resolve(strict=True)
        output_root = Path(args.root).resolve()
        if output_root == selected or selected in output_root.parents:
            raise ValueError("W3 run outputs must stay outside the input package")
        if args.action == "prepare-sampling":
            raise ValueError("A packaged W3 input is read-only; prepare sampling in a run directory")
    data_root, dataset, truth, binding = case_input(args.root, args.case, dataset_root)
    if args.action == "train" or (binding.get("data_profile") == SINGLE_PROFILE
            and (Path(args.root) / "W3_PREPARATION.json").exists()):
        require_all_inputs(args.root, binding)
        gate = read_json(Path(args.root) / "FULL_INPUT_GATE.json")
        if "preparation_sha256" in gate:
            prepared = read_json(Path(args.root) / "W3_PREPARATION.json")
            selected = next(row for row in prepared["cases"] if row["case_id"] == args.case)
            sampling, sampling_receipt = sampling_paths(data_root)
            if (digest(sampling) != selected["sampling_sha256"]
                    or digest(sampling_receipt) != selected["sampling_receipt_sha256"]):
                raise ValueError("prepared W3 sampling bytes changed")
    os.environ.update(GIFT_DATA_ROOT=str(data_root), PYTHONDONTWRITEBYTECODE="1",
                      GIFT_PINN_EXECUTION_PROFILE="native_fp32")
    if args.action == "prepare-sampling":
        return prepare_sampling(args.root, args.case, data_root, dataset, binding)
    if args.method is None:
        raise ValueError("train/measure require an explicit method")
    expected = job_binding(args, binding)
    provenance = {"schema": "gift.w3-parameter-job-attempt.v1", "case_id": args.case,
        "method": args.method, "action": args.action, "device": args.device,
        "python": platform.python_version(), "executable": sys.executable,
        "host": socket.gethostname(), "claim_token": args.claim_token,
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "source_sha256": digest(__file__), "scientific_binding": expected}
    attempt = Path(args.root).resolve() / "job_attempts" / args.case / METHODS[args.method] / uuid.uuid4().hex
    fixed_json(attempt / "START.json", provenance)
    began = time.perf_counter()
    try:
        result = (train_case(args, data_root, dataset, binding) if args.action == "train"
                  else measure_case(args, data_root, dataset, truth, binding))
        fixed_json(attempt / "FINISH.json", {"status": "complete", "process_seconds": time.perf_counter() - began,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "result_status": result["status"]})
        return result
    except BaseException as error:
        text = str(error).casefold()
        scientific = (isinstance(error, (FloatingPointError, MemoryError))
            or type(error).__name__.casefold() in ("resourceexhaustederror", "outofmemoryerror")
            or any(term in text for term in
            ("out of memory", "oom when allocating tensor", "resourceexhaustederror", "nonfinite", "metric became nonfinite",
             "terminal arrays must be safe finite")))
        record = {"status": "failed" if scientific else "interrupted", "exception": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "process_seconds": time.perf_counter() - began,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "identity": expected}
        fixed_json(attempt / "FAILURE.json", record)
        if args.action == "train":
            output, _, _, _ = training_paths(args.root, args.case, args.method)
            if output.is_dir() or scientific:
                fixed_json(output / "CASE_JOB_BINDING.json", expected)
                if scientific:
                    fixed_json(output / "FAILURE.json", record)
        elif scientific:
            output = Path(args.root).resolve() / "jobs" / args.case / METHODS[args.method]
            fixed_json(output / "FAILURE.json", record)
        raise


def main(argv=None):
    """Plan or execute one case task using the caller's appropriate Python environment."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        help="read W3 cases from this formal package; --root remains the separate run directory")
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument("--action", choices=("prepare-sampling", "train", "measure"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--claim-token", default=os.environ.get("GIFT_SWEEP_CLAIM_TOKEN", ""))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.action != "prepare-sampling" and args.method is None:
        parser.error("train/measure require --method")
    if args.action == "train" and args.method not in ("GIFT", "PINN-SR", "PINN-SR-KC"):
        parser.error("PDE-FIND executes regression through --action measure")
    if args.resume and args.action != "train":
        parser.error("--resume applies only to a native training journal")
    if not args.execute:
        planned_binding = {}
        if args.method == "GIFT":
            if args.dataset_root is not None:
                from gift.paths import data_path
                receipt = data_path(args.dataset_root, "w3.cases") / args.case / "DATA_RECEIPT.json"
            else:
                receipt = args.root / "data" / args.case / "DATA_RECEIPT.json"
            if receipt.is_file() and read_json(receipt).get("profile") == SINGLE_PROFILE:
                planned_binding["data_profile"] = SINGLE_PROFILE
        print(json.dumps({"status": "plan_only", "case_id": args.case, "method": args.method,
            "action": args.action, "device": args.device, "root": str(args.root.resolve()),
            "dataset_root": str(args.dataset_root.resolve()) if args.dataset_root else None,
            "training_configuration": gift_budget(planned_binding) if args.method == "GIFT" else PINN_BUDGET,
            "truth_parameters_passed_to_training": False}, ensure_ascii=False, indent=2))
        return
    result = execute(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
