"""Single-trajectory terminal checkpoints and observation boundaries on CPU."""
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from experiments.formal import train_low_generator as cli
from experiments.formal._shared import gift_generator_training as trainer
from scripts import run_w3_parameter_job as job
from training.gift_data import validate_input


def observations(path, *, validation=False):
    """Write one small observed trajectory, optionally a distinct holdout."""
    rng = np.random.RandomState(831)
    with h5py.File(path, "x") as handle:
        for name, identity in (("training", 0), ("validation", 50)) if validation else (("training", 0),):
            handle.create_dataset(name + "/time", data=np.arange(7) * .02)
            handle.create_dataset(name + "/trajectory_index", data=[identity])
            handle.create_dataset(name + "/vorticity", data=rng.normal(size=(1, 7, 8, 8)).astype("float32"))


def configuration():
    """Use a bounded native CPU budget; it supplies no published metrics."""
    return trainer.GeneratorTrainingConfig(training_trajectories=1, validation_trajectories=0,
        checkpoint_selection="terminal", cutoff=1, rank=1, phase_steps=(3, 2),
        phase_learning_rates=(.01, .003), batch_size=2, affine_refit_interval=2,
        validation_interval=1, seed=811)


def test_zero_validation_requires_explicit_terminal_selection():
    """A missing holdout cannot silently keep the validation-best policy."""
    with pytest.raises(ValueError, match="require terminal"):
        trainer.GeneratorTrainingConfig(validation_trajectories=0).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        replace(configuration(), validation_trajectories=-1).validate()
    configuration().validate()
    assert trainer.GeneratorTrainingConfig().validation_trajectories == 20
    assert trainer.GeneratorTrainingConfig().checkpoint_selection == "validation_best"


def test_no_validation_group_is_required_or_hashed(tmp_path):
    """The only ID0 group is sufficient; unused validation bytes cannot change its digest."""
    path = tmp_path / "standard_ns_n64_full_spectrum.h5"
    observations(path)
    config = configuration()
    layout = trainer.inspect_observation_dataset(path, config)
    assert layout["validation"] == dict(trajectory_ids=[], trajectory_count=0, observations_opened=False)
    validate_input(tmp_path, "noise_000", "regenerated", full=True, include_validation=False)
    with h5py.File(path, "r") as handle:
        train, validation, expected = trainer.selected_observation_arrays(handle, config)
        assert validation is None and train[3].tolist() == [0]
    with h5py.File(path, "a") as handle:
        handle.create_dataset("validation/unread_diagnostic", data=np.nan)
    with h5py.File(path, "r") as handle:
        assert trainer.selected_observation_arrays(handle, config)[2] == expected
    with pytest.raises(KeyError):
        validate_input(tmp_path, "noise_000", "regenerated", full=False)


def test_cli_and_job_select_terminal_only_for_explicit_single_profile(tmp_path):
    """Historical CLI recipes keep holdouts, while the new receipt profile requests zero."""
    args = cli.parse_args(["--dry-run", "--training-trajectories", "1",
        "--validation-trajectories", "0", "--checkpoint-selection", "terminal"])
    config = cli._configuration(args)
    assert config.validation_trajectories == 0 and config.checkpoint_selection == "terminal"
    legacy = job.gift_arguments(tmp_path, False, "cuda")
    assert "--checkpoint-selection" not in legacy and "--validation-trajectories" not in legacy
    current = job.gift_arguments(tmp_path, False, "cuda", single_trajectory=True)
    assert current[current.index("--validation-trajectories") + 1] == "0"
    assert current[current.index("--checkpoint-selection") + 1] == "terminal"
    assert job.gift_budget({}) == job.GIFT_BUDGET
    assert job.gift_budget(dict(data_profile=job.SINGLE_PROFILE)) == job.GIFT_SINGLE_BUDGET


def test_single_population_gate_cannot_use_legacy_protocol_or_an_extra_trajectory(tmp_path):
    """Actual one-trajectory HDF and a new fifty-case gate must agree in both directions."""
    case = tmp_path / "data/parameter_000"
    case.mkdir(parents=True)
    dataset = case / "standard_ns_n64_full_spectrum.h5"
    with h5py.File(dataset, "x") as handle:
        handle.create_dataset("training/time", data=np.arange(501) * .02)
        handle.create_dataset("training/trajectory_index", data=[0])
        handle.create_dataset("training/vorticity", shape=(1, 501, 64, 64), dtype="float32")
    truth = dict(nu=.01, beta=1., gamma=1.)
    job.fixed_json(case / "DATA_LOCK.json", dict(schema="gift.w3-single-trajectory-data-lock.v1",
        profile=job.SINGLE_PROFILE, case_id="parameter_000", truth=truth, parameter_csv_sha256="a" * 64,
        splits=dict(training=dict(population=1), validation=dict(population=0))))
    job.fixed_json(case / "DATA_QA.json", dict(schema="gift.w3-single-trajectory-data-qa.v1",
        status="passed", case_id="parameter_000", truth=truth))
    job.fixed_json(case / "DATA_RECEIPT.json", dict(schema="gift.w3-single-trajectory-data-receipt.v1",
        profile=job.SINGLE_PROFILE, status="passed", case_id="parameter_000", truth=truth,
        parameter_csv_sha256="a" * 64, initial_conditions_sha256="b" * 64,
        data_file=dict(relative_path=dataset.name, sha256=job.digest(dataset), bytes=dataset.stat().st_size),
        data_lock_sha256=job.digest(case / "DATA_LOCK.json"), data_qa_sha256=job.digest(case / "DATA_QA.json")))
    binding = job.case_input(tmp_path, "parameter_000")[3]
    cohort = tmp_path / "data/DATA_COHORT_QA.json"
    job.fixed_json(cohort, dict(schema="gift.w3-single-trajectory-data-cohort-qa.v1",
        profile=job.SINGLE_PROFILE, status="passed", trajectory_count=50))
    receipts = {"parameter_%03d" % i: "0" * 64 for i in range(50)}
    receipts["parameter_000"] = binding["data_receipt_sha256"]
    job.fixed_json(tmp_path / "FULL_INPUT_GATE.json", dict(schema="gift.w3-single-trajectory-full-input-gate.v1",
        profile=job.SINGLE_PROFILE, status="passed", case_count=50, trajectory_count=50,
        training_trajectories_per_case=1, validation_trajectories_per_case=0,
        case_receipts=receipts, cohort_qa_sha256=job.digest(cohort)))
    job.require_all_inputs(tmp_path, binding)
    legacy = dict(binding)
    legacy.pop("data_profile")
    with pytest.raises(ValueError, match="own fifty-trajectory protocol gate"):
        job.require_all_inputs(tmp_path, legacy)
    with h5py.File(dataset, "a") as handle:
        handle["validation"] = handle["training"]
    # A newly hashed receipt cannot make an extra observation group admissible.
    import json
    receipt_path = case / "DATA_RECEIPT.json"
    receipt = job.read_json(receipt_path)
    receipt["data_file"].update(sha256=job.digest(dataset), bytes=dataset.stat().st_size)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="only training ID0"):
        job.case_input(tmp_path, "parameter_000")


@pytest.mark.parametrize("boundary", ["phase_1_step_1", "phase_1_step_3", "phase_2_step_1"])
def test_terminal_resume_keeps_final_updates_and_never_labels_monitor_as_holdout(tmp_path, monkeypatch, boundary):
    """Resume a native committed step bitwise with zero validation datasets."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = tmp_path / "observations.h5"
        observations(dataset)
        config = configuration()
        def run(directory, *, resume=False):
            if not resume:
                directory.mkdir()
            return trainer.train_fresh_generator(dataset=dataset, checkpoint=directory / "gift_main.pt",
                report_path=directory / "report.json", history_path=directory / "history.csv",
                condition="noise_000", device_name="cpu", config=config,
                project_root=cli.PROJECT_ROOT, entry_script=Path(cli.__file__),
                checkpoint_interval=1, resume=resume)
        continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
        expected_report = run(continuous)
        original = trainer.CheckpointStore.save
        def commit_and_stop(store, name, payload):
            result = original(store, name, payload)
            if name == boundary:
                raise RuntimeError("simulated committed interruption")
            return result
        with monkeypatch.context() as patch:
            patch.setattr(trainer.CheckpointStore, "save", commit_and_stop)
            with pytest.raises(RuntimeError, match="committed interruption"):
                run(interrupted)
        observed_report = run(interrupted, resume=True)
        expected = torch.load(continuous / "gift_main.pt", weights_only=True)
        actual = torch.load(interrupted / "gift_main.pt", weights_only=True)
        assert all(torch.equal(value, actual["model_state_dict"][key]) for key, value in expected["model_state_dict"].items())
        assert expected_report["optimization"]["phase_records"] == observed_report["optimization"]["phase_records"]
        assert [row["selected_step"] for row in observed_report["optimization"]["phase_records"]] == [3, 2]
        assert observed_report["data"]["validation"] is None
        assert observed_report["data"]["validation_trajectory_ids"] == []
        assert observed_report["metrics"]["independent_validation_derivative_relative_l2"] is None
        assert observed_report["information_firewall"]["opened_hdf5_datasets"] == [
            "training/time", "training/trajectory_index", "training/vorticity"]
        history = (interrupted / "history.csv").read_text(encoding="utf-8-sig")
        assert "training_monitor_relative_l2" in history and "validation_relative_l2" not in history
    finally:
        torch.set_num_threads(old_threads)
