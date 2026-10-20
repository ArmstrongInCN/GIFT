"""Current W3 preparation safeguards without full-budget scientific training.

Synthetic observation fixtures test file boundaries and sensor correspondence;
they are never emitted as experimental results or numerical QA evidence.
"""
from argparse import Namespace
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from scripts import generate_w3_parameter_data as generation
from scripts import prepare_w3_run as preparation
from scripts import run_w3_parameter_job as job
from scripts.generate_sampling import sampling_design, measurement_targets


def sampling_fixture(root):
    """Make a deterministic synthetic finite field and its exact 30k sensor labels."""
    folder = root / "data" / "parameter_000"
    folder.mkdir(parents=True)
    dataset = folder / generation.DATA_FILENAME
    with h5py.File(dataset, "x") as handle:
        group = handle.create_group("training")
        group.create_dataset("trajectory_index", data=[0])
        group.create_dataset("time", data=generation.TIMES)
        # A static synthetic Fourier field tests u/v reconstruction and row order.
        field = np.sin(2*np.pi*np.arange(64, dtype=np.float32)/64)
        group.create_dataset("vorticity", data=np.broadcast_to(field, (1, 501, 64, 64)),
                             compression="gzip", chunks=(1, 1, 64, 64))
    with h5py.File(dataset, "r") as handle:
        arrays = sampling_design(np.asarray(handle["training/time"], dtype=np.float32),
                                 64, np.random.RandomState(1234))
        arrays["targets"] = measurement_targets(handle, arrays)
    target, receipt = job.sampling_paths(folder)
    target.parent.mkdir(parents=True)
    np.savez_compressed(target, **arrays)
    binding = dict(case_id="parameter_000", data_profile=job.SINGLE_PROFILE,
                   observation_sha256=job.digest(dataset))
    value = dict(schema="gift.w3-parameter-sampling-receipt.v1", status="passed",
        case_binding=binding, sampling_sha256=job.digest(target), seed=1234,
        sensors=500, time_samples=60, train_rows=24000, holdout_rows=6000,
        physics_rows=84000, truth_parameters_passed_to_sampling=False)
    job.fixed_json(receipt, value)
    return folder, dataset, binding, arrays


def test_current_profile_retains_fixed_id_zero_and_historical_profile():
    """The current generator removes the validation population, preserving the same initial ID0."""
    current = generation.initial_conditions(job.SINGLE_PROFILE)
    historical = generation.initial_conditions()
    assert list(current["splits"]) == ["training"]
    assert current["splits"]["training"] == historical["splits"]["training"]
    assert historical["splits"]["validation"]["ids"] == list(range(50, 70))
    lock = generation.data_lock(dict(case_id="parameter_000", truth=dict(nu=.01, beta=1., gamma=1.)),
                                "initial", job.SINGLE_PROFILE)
    assert lock["splits"]["validation"] == dict(ids=[], population=0,
        vorticity_shape=[0, 501, 64, 64], hdf_group_present=False)
    assert lock["monitor_is_training_data"] and not lock["independent_validation"]
    with pytest.raises(ValueError, match="unknown W3"):
        generation.initial_conditions("unrecognised")


def test_current_generation_cli_plan_writes_nothing(tmp_path, monkeypatch, capsys):
    """Explicit current-profile planning declares one trajectory without creating output."""
    monkeypatch.setattr(generation, "parameter_cases", lambda _: [])
    output = tmp_path / "new_integration"
    generation.main(["--root", str(output), "--parameter-csv", "unused.csv",
                     "--profile", job.SINGLE_PROFILE])
    plan = json.loads(capsys.readouterr().out)
    assert plan["observations_per_case"] == dict(training=1, validation=0)
    assert plan["profile"] == job.SINGLE_PROFILE and not output.exists()


def test_current_generation_dispatch_creates_only_one_training_group(tmp_path, monkeypatch):
    """Mock integration checks population/schema wiring; numeric QA is covered separately."""
    truth = dict(nu=.01, beta=1., gamma=1.)
    case = dict(case_id="parameter_000", truth=truth)
    calls = []
    def integration(parameters, coefficients, **kwargs):
        calls.append((parameters, coefficients))
        return np.zeros((len(parameters), 501, 64, 64), dtype=np.float32), {}
    monkeypatch.setattr(generation, "integrate_batch", integration)
    monkeypatch.setattr(generation, "audit_case", lambda dataset, lock: dict(
        schema="gift.w3-single-trajectory-data-qa.v1", status="passed",
        case_id=case["case_id"], truth=truth, fixture_only=True))
    receipt = generation.generate_case(tmp_path, case, "initial-fixture", profile=job.SINGLE_PROFILE)
    assert len(calls) == 1 and len(calls[0][0]) == 1
    assert receipt["new_integration_performed"]
    assert receipt["validation_trajectories_per_case"] == 0
    with h5py.File(tmp_path / case["case_id"] / generation.DATA_FILENAME, "r") as handle:
        assert set(handle) == {"training"}
        assert handle["training/vorticity"].shape == (1, 501, 64, 64)
    # A completed current run cannot be silently resumed as the older profile.
    with pytest.raises(ValueError, match="receipt"):
        generation.generate_case(tmp_path, case, "initial-fixture", resume=True)


def test_deep_sampling_reconstructs_all_real_sensor_labels(tmp_path):
    """Actual FFT target reconstruction must preserve the full design and all 90k target values."""
    folder, dataset, binding, _ = sampling_fixture(tmp_path)
    _, receipt = job.validate_sampling(folder, dataset, binding, deep=True)
    assert receipt["target_recompute_max_absolute_difference"] == 0.
    assert receipt["target_recompute_relative_l2"] == 0.


@pytest.mark.parametrize("damage", ["design", "targets"])
def test_rehashed_sampling_cannot_disguise_wrong_design_or_targets(tmp_path, damage):
    """A new cache hash alone cannot approve scientifically mismatched sensor data."""
    folder, dataset, binding, arrays = sampling_fixture(tmp_path)
    if damage == "design":
        arrays["train_indices"][[0, 1]] = arrays["train_indices"][[1, 0]]
    else:
        arrays["targets"][0, 0] += .05
    target, receipt_path = job.sampling_paths(folder)
    np.savez_compressed(target, **arrays)
    receipt = job.read_json(receipt_path)
    receipt["sampling_sha256"] = job.digest(target)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling " + damage):
        job.validate_sampling(folder, dataset, binding, deep=True)


def mocked_preparation_inputs(root, monkeypatch):
    """Provide clearly synthetic all-case records to test immutable gate transactions."""
    data = root / "data"
    data.mkdir(parents=True)
    parameters = data / "parameters.csv"
    parameters.write_text("fixture only", encoding="utf-8")
    job.fixed_json(data / "INITIAL_CONDITIONS.json", {"fixture_only": True})
    cohort = dict(schema="gift.w3-single-trajectory-data-cohort-qa.v1", status="passed",
                  profile=job.SINGLE_PROFILE, case_count=50, trajectory_count=50)
    job.fixed_json(data / "DATA_COHORT_QA.json", cohort)
    rows = []
    for index in range(50):
        case_id = f"parameter_{index:03d}"
        binding = dict(case_id=case_id, data_profile=job.SINGLE_PROFILE,
                       data_receipt_sha256=f"{index:064x}")
        rows.append(dict(case_id=case_id, case_binding=binding,
                         sampling_sha256="s" * 64, sampling_receipt_sha256="r" * 64))
    monkeypatch.setattr(preparation, "validate_cohort", lambda *args: (cohort, rows))
    return rows


def test_preparation_commits_complete_gate_and_preserves_source_provenance(tmp_path, monkeypatch):
    """The master gate binds validated inputs/protocol and does not require current source hashes."""
    rows = mocked_preparation_inputs(tmp_path, monkeypatch)
    result = preparation.prepare(tmp_path)
    assert result["status"] == "passed"
    old_sources = (tmp_path / "PREPARATION_SOURCES.json").read_bytes()
    old_preparation = (tmp_path / "W3_PREPARATION.json").read_bytes()
    original_digest = job.digest
    def changed_sources(path):
        return "f" * 64 if str(path).endswith(".py") else original_digest(path)
    monkeypatch.setattr(job, "digest", changed_sources)
    preparation.prepare(tmp_path)
    assert (tmp_path / "PREPARATION_SOURCES.json").read_bytes() == old_sources
    assert (tmp_path / "W3_PREPARATION.json").read_bytes() == old_preparation
    job.require_all_inputs(tmp_path, rows[-1]["case_binding"])
    changed = dict(rows[0]["case_binding"], data_receipt_sha256="bad")
    with pytest.raises(ValueError, match="master fifty-case"):
        job.require_all_inputs(tmp_path, changed)


def test_preparation_retry_accepts_bounded_rounding_and_keeps_both_diagnostics(tmp_path, monkeypatch):
    """Identical input bytes may produce tiny allowed diagnostic rounding on a later machine."""
    rows = mocked_preparation_inputs(tmp_path, monkeypatch)
    for row in rows:
        row.update(target_recompute_max_absolute_difference=0., target_recompute_relative_l2=0.,
                   recomputed_fd_oracle_max_ape_percent=.01)
    preparation.prepare(tmp_path)
    original = (tmp_path / "W3_PREPARATION.json").read_bytes()
    gate = (tmp_path / "FULL_INPUT_GATE.json").read_bytes()
    for row in rows:
        row.update(target_recompute_max_absolute_difference=1e-8, target_recompute_relative_l2=1e-9,
                   recomputed_fd_oracle_max_ape_percent=.010000000000001)
    preparation.prepare(tmp_path)
    assert (tmp_path / "W3_PREPARATION.json").read_bytes() == original
    assert (tmp_path / "FULL_INPUT_GATE.json").read_bytes() == gate
    checks = [job.read_json(path) for path in (tmp_path / "preparation_checks").glob("*.json")]
    assert len(checks) == 2
    assert {row["diagnostics"][0]["target_recompute_max_absolute_difference"] for row in checks} == {0., 1e-8}


def test_prepared_protocol_change_blocks_training_before_dispatch(tmp_path, monkeypatch):
    """Changing the terminal protocol after preparation invalidates the committed gate."""
    rows = mocked_preparation_inputs(tmp_path, monkeypatch)
    preparation.prepare(tmp_path)
    protocol = tmp_path / "PROTOCOL_LOCK.json"
    value = job.read_json(protocol)
    value["generator"]["phase_steps"] = [1, 1]
    protocol.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol or validation record"):
        job.require_all_inputs(tmp_path, rows[0]["case_binding"])


def test_prepared_gate_does_not_import_torch_in_separate_pinn_process(tmp_path, monkeypatch):
    """The separate TensorFlow environment can inspect current gates without installing PyTorch."""
    mocked_preparation_inputs(tmp_path, monkeypatch)
    preparation.prepare(tmp_path)
    code = ("from pathlib import Path; import sys; from scripts import run_w3_parameter_job as j; "
            "r=Path(sys.argv[1]); b=j.read_json(r/'W3_PREPARATION.json')['cases'][0]['case_binding']; "
            "j.require_all_inputs(r,b); assert 'torch' not in sys.modules")
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)],
        cwd=preparation.PROJECT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_failed_full_validation_never_creates_a_passed_gate(tmp_path, monkeypatch):
    """One bad case stops preparation with no master proof that could authorize training."""
    def failed(*args):
        raise ValueError("case observation checksum differs")
    monkeypatch.setattr(preparation, "validate_cohort", failed)
    with pytest.raises(ValueError, match="checksum"):
        preparation.prepare(tmp_path / "run")
    assert not (tmp_path / "run" / "FULL_INPUT_GATE.json").exists()


def test_full_cohort_rejects_missing_case_directory_before_any_numerical_fit(tmp_path, monkeypatch):
    """A passed-looking cohort JSON cannot authorize a 49-case input directory."""
    data = tmp_path / "data"
    data.mkdir()
    cases = [dict(case_id=f"parameter_{index:03d}", truth=dict(nu=.01, beta=1., gamma=1.))
             for index in range(50)]
    parameters = data / "parameters.csv"
    parameters.write_text("fixture-design-only", encoding="utf-8")
    job.fixed_json(data / "INITIAL_CONDITIONS.json", generation.initial_conditions(job.SINGLE_PROFILE))
    job.fixed_json(data / "DATA_COHORT_QA.json", dict(
        schema="gift.w3-single-trajectory-data-cohort-qa.v1", status="passed",
        profile=job.SINGLE_PROFILE, case_count=50, trajectory_count=50, cases=cases,
        parameter_csv_sha256=job.digest(parameters),
        initial_conditions_sha256=job.digest(data / "INITIAL_CONDITIONS.json")))
    for case in cases[:-1]:
        (data / case["case_id"]).mkdir()
    monkeypatch.setattr(generation, "parameter_cases", lambda _: cases)
    monkeypatch.setattr(job, "case_input", lambda *args: pytest.fail("must reject the subset before fitting"))
    with pytest.raises(ValueError, match="directory inventory"):
        preparation.validate_cohort(tmp_path, None, data, parameters)


def test_preparation_output_boundaries_and_plan_are_read_only(tmp_path):
    """Neither project-local nor package-local preparation outputs are permitted."""
    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(ValueError, match="outside the project"):
        preparation.input_paths(preparation.PROJECT / "new_run")
    with pytest.raises(ValueError, match="read-only input"):
        preparation.input_paths(package / "new_run", package)


def test_terminal_training_dispatch_retains_native_full_budget(tmp_path, monkeypatch):
    """A prepared current case reaches the real native argument/configuration boundary."""
    from experiments.formal import train_low_generator as native
    args = Namespace(root=tmp_path, case="parameter_000", method="GIFT", resume=False, device="cpu")
    binding = dict(case_id=args.case, data_profile=job.SINGLE_PROFILE)
    observed = []
    def intercepted_run(parsed):
        observed.append(native._configuration(parsed))
        # Mimic only directory creation so identity recovery can be checked.
        parsed.output.mkdir(parents=True)
        return {"status": "fixture_only"}
    monkeypatch.setattr(native, "run", intercepted_run)
    monkeypatch.setattr(job, "validate_training", lambda *args: {"status": "fixture_only"})
    assert job.train_case(args, tmp_path, tmp_path / "data.h5", binding)["status"] == "fixture_only"
    config = observed[0]
    assert config.phase_steps == (6000, 6000)
    assert config.validation_trajectories == 0 and config.checkpoint_selection == "terminal"
    assert config.training_trajectories == 1 and config.seed == 2026072301
