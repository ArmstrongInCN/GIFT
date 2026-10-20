"""Case routing and population safeguards without training scientific models."""
from argparse import Namespace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import run_w3_parameter_job as job


def fixture_case(tmp_path, case_id="parameter_000", truth=None):
    """Build explicit byte-binding fixtures, never experimental flow results."""
    truth = truth or {"nu": .012, "beta": .8, "gamma": 1.2}
    case = tmp_path / "data" / case_id
    case.mkdir(parents=True)
    data = case / "standard_ns_n64_full_spectrum.h5"
    data.write_bytes(b"binding fixture; numerical methods are mocked")
    lock = {"case_id": case_id, "truth": truth, "parameter_csv_sha256": "a" * 64}
    job.fixed_json(case / "DATA_LOCK.json", lock)
    job.fixed_json(case / "DATA_QA.json", {"status": "passed", "case_id": case_id, "truth": truth})
    job.fixed_json(case / "DATA_RECEIPT.json", {
        "schema": "gift.w3-parameter-data-receipt.v1", "status": "passed", "case_id": case_id,
        "truth": truth, "parameter_csv_sha256": "a" * 64, "initial_conditions_sha256": "b" * 64,
        "data_file": {"path": "X:/another-host/observations.h5", "relative_path": data.name,
                      "sha256": job.digest(data), "bytes": data.stat().st_size},
        "data_lock_sha256": job.digest(case / "DATA_LOCK.json"),
        "data_qa_sha256": job.digest(case / "DATA_QA.json")})
    return job.case_input(tmp_path, case_id)


def args_for(root, method="PDE-FIND"):
    """Select a declared case and CPU readout route for the protocol fixture."""
    return Namespace(root=root, case="parameter_000", method=method,
                     device="cpu", action="measure", resume=False, claim_token="fixture_claim")


def fixture_cohort_gate(root, binding):
    """Supply an explicit master-gate fixture; it is not an experimental cohort QA."""
    cohort = root / "data/DATA_COHORT_QA.json"
    job.fixed_json(cohort, {"status": "passed", "fixture_only": True})
    receipts = {"parameter_%03d" % index: "0" * 64 for index in range(50)}
    receipts[binding["case_id"]] = binding["data_receipt_sha256"]
    job.fixed_json(root / "FULL_INPUT_GATE.json", {"status": "passed", "case_count": 50,
                   "cohort_qa_sha256": job.digest(cohort), "case_receipts": receipts})


def test_case_is_bound_under_this_host_root_and_actual_truth(tmp_path):
    """An origin machine's absolute path is descriptive; local relative bytes bind execution."""
    data_root, data, truth, binding = fixture_case(tmp_path)
    assert data == data_root / "standard_ns_n64_full_spectrum.h5"
    assert truth == {"nu": .012, "beta": .8, "gamma": 1.2}
    assert binding["observation_sha256"] == job.digest(data)


def test_missing_case_never_uses_released_data(tmp_path, monkeypatch):
    """A configured old data root cannot fill a missing new parameter instance."""
    monkeypatch.setenv("GIFT_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        job.case_input(tmp_path, "parameter_001")


def test_changed_case_observations_fail_before_calculation(tmp_path):
    """Detect observation changes even when the old passed QA record remains."""
    _, data, _, _ = fixture_case(tmp_path)
    data.write_bytes(b"changed observation fixture")
    with pytest.raises(ValueError, match="checksum"):
        job.case_input(tmp_path, "parameter_000")


def test_train_requires_the_complete_master_cohort_and_exact_case_receipt(tmp_path):
    """A single passed case cannot start fitting until the fifty-case proof is closed."""
    _, _, _, binding = fixture_case(tmp_path)
    with pytest.raises(FileNotFoundError):
        job.require_all_inputs(tmp_path, binding)
    fixture_cohort_gate(tmp_path, binding)
    job.require_all_inputs(tmp_path, binding)
    changed = dict(binding, data_receipt_sha256="f" * 64)
    with pytest.raises(ValueError, match="master fifty-case"):
        job.require_all_inputs(tmp_path, changed)


def test_changed_master_cohort_proof_blocks_fit_even_with_passed_side_status(tmp_path):
    """The unchanged passed text cannot authorize a modified population proof."""
    _, _, _, binding = fixture_case(tmp_path)
    fixture_cohort_gate(tmp_path, binding)
    (tmp_path / "data/DATA_COHORT_QA.json").write_text(json.dumps({"status": "passed", "changed": True}))
    with pytest.raises(ValueError, match="master fifty-case"):
        job.require_all_inputs(tmp_path, binding)


@pytest.mark.parametrize("case_id", ["parameter_050", "parameter_100", "../parameter_000", "noise_000"])
def test_only_predeclared_case_population_allowed(tmp_path, case_id):
    """No extra instance, noise alias or escaped path expands the planned population."""
    with pytest.raises(ValueError, match="case must"):
        job.case_input(tmp_path, case_id)


def test_actual_truth_errors_do_not_use_original_reference_coefficients():
    """A nonreference tuple must yield zero APE when estimates equal its own truth."""
    values = {"nu": .014, "beta": .78, "gamma": 1.24}
    assert job.relative_errors(values, values) == {"nu": 0., "beta": 0., "gamma": 0.}
    assert job.relative_errors({"nu": .007, "beta": .78, "gamma": 1.24}, values)["nu"] == pytest.approx(50.)


def test_generator_dispatch_keeps_original_native_budget_and_no_truth_labels(tmp_path):
    """The wrapper supplies only the original observation training configuration."""
    arguments = job.gift_arguments(tmp_path, False, "cuda")
    assert arguments == ["--condition", "noise_000", "--data-profile", "regenerated",
        "--training-trajectories", "1", "--output", str(tmp_path), "--device", "cuda",
        "--seed", "2026072301", "--phase-steps", "6000", "6000",
        "--phase-learning-rates", ".01", ".003", "--batch-size", "16",
        "--affine-refit-interval", "500", "--validation-interval", "250",
        "--checkpoint-interval", "250", "--run-training"]
    assert not any(term in arguments for term in ("--nu", "--beta", "--gamma"))


def test_pinn_dispatch_keeps_full_population_and_native_phase_defaults(tmp_path):
    """No diagnostic, row truncation, coefficient labels or budget overrides reach PINN."""
    output = tmp_path / "known"
    assert job.pinn_arguments(output, True, "gpu") == ["--execute", "--mode", "known",
        "--device", "gpu", "--condition", "noise_000", "--data-profile", "regenerated",
        "--output", str(output), "--nadam-interval", "1000", "--resume"]


def test_measurement_commit_uses_actual_truth_and_cache_avoids_recompute(tmp_path, monkeypatch):
    """Bind real-case APE and all coefficient rows, then reuse exact committed bytes."""
    data_root, data, truth, binding = fixture_case(tmp_path)
    from experiments.formal.w3_equation_identification import run
    calls = []
    def calculate(method, condition, dataset, checkpoint, device, **kwargs):
        """Count mock readouts so lifecycle checks never train or fit science."""
        calls.append((method, condition, dataset, checkpoint, device, kwargs))
        return {"parameters": dict(truth), "native_coefficients": [{"term": "fixture", "real": 2., "imaginary": 0.}]}
    monkeypatch.setattr(run, "calculate", calculate)
    args = args_for(tmp_path)
    first = job.measure_case(args, data_root, data, truth, binding)
    second = job.measure_case(args, data_root, data, truth, binding)
    assert first == second and len(calls) == 1
    assert first["APE_percent"] == {"nu": 0., "beta": 0., "gamma": 0.}
    assert calls[0][1] == "noise_000" and calls[0][2] == data
    path = tmp_path / "jobs/parameter_000/pde_find"
    assert "fixture,2.0,0.0" in (path / "native_coefficients.csv").read_text()


def test_commit_recovery_preserves_readout_cost_and_does_not_repeat_regression(tmp_path, monkeypatch):
    """After a result commit, a missing final marker can be repaired from exact saved data."""
    data_root, data, truth, binding = fixture_case(tmp_path)
    args = args_for(tmp_path)
    output = tmp_path / "jobs/parameter_000/pde_find"
    measured = {"parameters": dict(truth), "native_coefficients": []}
    result = {"schema": "gift.w3-parameter-measurement.v1", "status": "complete",
              "case_id": args.case, "method": args.method, "identity": job.job_binding(args, binding),
              "truth": truth, "parameters": truth, "APE_percent": {k: 0. for k in truth},
              "measurement_seconds": 123., "raw_measurement": measured}
    job.fixed_json(output / "result.json", result)
    from experiments.formal.w3_equation_identification import run
    monkeypatch.setattr(run, "calculate", lambda *a, **k: pytest.fail("partial commit repeated regression"))
    assert job.measure_case(args, data_root, data, truth, binding) == result
    assert (output / "COMPLETE.json").exists()


def test_permanent_scientific_failure_cannot_resume_or_recompute(tmp_path):
    """A failed unique scientific slot stays failed even if native state exists."""
    data_root, data, _, binding = fixture_case(tmp_path)
    args = args_for(tmp_path, "GIFT")
    output, _, _, latest = job.training_paths(tmp_path, args.case, args.method)
    job.fixed_json(output / "FAILURE.json", {"status": "failed"})
    job.fixed_json(latest, {"fixture": "not actual weights"})
    args.resume = True
    with pytest.raises(RuntimeError, match="scientific failure"):
        job.train_case(args, data_root, data, binding)


def test_native_generator_none_file_hash_uses_its_actual_array_identity(tmp_path):
    """A native GIFT export's null whole-file SHA still requires the exact learned observations."""
    import torch
    from experiments.formal._shared.gift_generator_training import _load_observed_group
    from training.checkpoints import state_digest
    dataset = tmp_path / "synthetic_observations.h5"
    with h5py.File(dataset, "x") as handle:
        for split, count, ids in (("training", 1, [0]), ("validation", 20, range(50, 70))):
            group = handle.create_group(split)
            group.create_dataset("trajectory_index", data=np.asarray(ids))
            group.create_dataset("time", data=np.arange(501) * .02)
            group.create_dataset("vorticity", shape=(count, 501, 4, 4), dtype="float32")
    with h5py.File(dataset, "r") as handle:
        train_state, train_derivative, train_time, train_ids, _ = _load_observed_group(handle, "training", 1)
        valid_state, valid_derivative, valid_time, valid_ids, _ = _load_observed_group(handle, "validation", 20)
    native_identity = state_digest({"training_state": train_state, "training_derivative": train_derivative,
        "validation_state": valid_state, "validation_derivative": valid_derivative,
        "training_time": torch.as_tensor(train_time), "training_ids": torch.as_tensor(train_ids),
        "validation_time": torch.as_tensor(valid_time), "validation_ids": torch.as_tensor(valid_ids)})
    args = args_for(tmp_path, "GIFT")
    binding = {"case_id": args.case, "observation_sha256": job.digest(dataset)}
    output, model, completion, _ = job.training_paths(tmp_path, args.case, args.method)
    job.fixed_json(output / "CASE_JOB_BINDING.json", job.job_binding(args, binding))
    job.fixed_json(output / "training_report.json", {"training_cost": {"phase_updates": [6000, 6000]},
                   "optimization": {"seed": 2026072301}})
    job.fixed_json(completion, {"fixture_only": "not a scientific checkpoint"})
    torch.save({"training_data": {"sha256": None, "selected_arrays_sha256": native_identity},
                "model_configuration": {"cutoff": 21, "rank": 8}}, model)
    assert job.validate_training(tmp_path, args.case, args.method, tmp_path, dataset, binding)["status"] == "complete"
    with h5py.File(dataset, "r+") as handle:
        handle["validation/vorticity"][3, 30, 2, 1] = 1.
    with pytest.raises(ValueError, match="learned arrays"):
        job.validate_training(tmp_path, args.case, args.method, tmp_path, dataset, binding)


@pytest.mark.parametrize("error", [MemoryError("synthetic host allocation failure"),
    type("ResourceExhaustedError", (RuntimeError,), {})("synthetic tensor allocation failure"),
    RuntimeError("OOM when allocating tensor: synthetic resource fixture")])
def test_resource_failure_before_initial_journal_stays_permanent(tmp_path, monkeypatch, error):
    """A graph-construction allocation failure must not become a fresh retriable slot."""
    _, _, _, binding = fixture_case(tmp_path)
    fixture_cohort_gate(tmp_path, binding)
    args = args_for(tmp_path, "PINN-SR-KC")
    args.action = "train"
    def fail(*unused):
        """Fail before any native journal/output exists, without importing TensorFlow."""
        raise error
    monkeypatch.setattr(job, "train_case", fail)
    with pytest.raises(type(error)):
        job.execute(args)
    output, _, _, _ = job.training_paths(tmp_path, args.case, args.method)
    assert job.read_json(output / "FAILURE.json")["status"] == "failed"
    attempts = list((tmp_path / "job_attempts/parameter_000/pinn_sr_kc").glob("*/FAILURE.json"))
    assert len(attempts) == 1 and job.read_json(attempts[0])["status"] == "failed"


@pytest.mark.parametrize("count", [1, 50])
def test_pde_container_selection_keeps_identical_id_zero_values(tmp_path, count):
    """The only reader adaptation accepts actual 1/50 containers and reads ID 0 identically."""
    from adapters import pde_find
    path = tmp_path / "observations.h5"
    state = np.broadcast_to(np.arange(64, dtype=np.float32)[None, None, :], (501, 64, 64)).copy()
    with h5py.File(path, "x") as file:
        group = file.create_group("training")
        group.create_dataset("trajectory_index", data=np.arange(count))
        group.create_dataset("time", data=np.arange(501) * .02)
        if count == 1:
            group.create_dataset("initial_condition_parameters", data=np.zeros((1, 4, 4), dtype=np.float64))
        field = group.create_dataset("vorticity", shape=(count, 501, 64, 64), dtype="float32",
                                     chunks=(1, 1, 64, 64), compression="gzip")
        field[0] = state
    raw, times = pde_find.read_observations(path)
    assert raw.dtype == np.float64
    np.testing.assert_array_equal(raw, state.astype(np.float64))
    np.testing.assert_array_equal(times, np.arange(501) * .02)


def test_pde_other_training_populations_are_not_accepted(tmp_path):
    """Neither a truncated unspecified population nor duplicated ID 0 is silently accepted."""
    from adapters import pde_find
    path = tmp_path / "observations.h5"
    with h5py.File(path, "x") as file:
        group = file.create_group("training")
        group.create_dataset("trajectory_index", data=[0, 0])
        group.create_dataset("time", data=np.arange(501) * .02)
        group.create_dataset("vorticity", shape=(2, 501, 64, 64), dtype="float32")
    with pytest.raises(ValueError, match="1- or 50-trajectory"):
        pde_find.read_observations(path)
