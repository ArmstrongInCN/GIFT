"""Tiny CPU continuation tests; not full-budget or published-metric evidence."""
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from experiments.formal import train_low_generator as cli
from experiments.formal._shared import gift_generator_training as trainer


def make_observations(path):
    """Create one training and one validation trajectory on an 8-grid for bounded CPU continuation."""
    rng = np.random.RandomState(741)
    with h5py.File(path, "x") as handle:
        for group, trajectory_id in (("training", 0), ("validation", 50)):
            handle.create_dataset(group + "/time", data=np.arange(7) * 0.02)
            handle.create_dataset(group + "/trajectory_index", data=[trajectory_id])
            handle.create_dataset(group + "/vorticity",
                                  data=rng.standard_normal((1, 7, 8, 8)).astype("float32"))


@pytest.mark.parametrize("boundary", ["phase_1_step_1", "phase_1_step_3", "phase_2_step_1"])
def test_resume_matches_continuous_across_phase_and_best_selection(tmp_path, monkeypatch, boundary):
    """Require bitwise factors and matching selected-phase metrics after committed interruptions; changed arrays must fail."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = tmp_path / "observations.h5"
        make_observations(dataset)
        config = trainer.GeneratorTrainingConfig(
            training_trajectories=1, validation_trajectories=1, cutoff=1, rank=1,
            phase_steps=(3, 2), batch_size=2, affine_refit_interval=2,
            validation_interval=1, seed=811,
        )
        def run(directory, *, resume=False, configuration=config):
            """Run the actual small generator controller in a new or explicitly resumed output directory."""
            if not resume:
                directory.mkdir()
            return trainer.train_fresh_generator(
                dataset=dataset, checkpoint=directory / "gift_main.pt",
                report_path=directory / "report.json", history_path=directory / "history.csv",
                condition="noise_000", device_name="cpu", config=configuration,
                project_root=cli.PROJECT_ROOT, entry_script=Path(cli.__file__),
                checkpoint_interval=1, resume=resume,
            )

        continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
        reference = run(continuous)
        original_save = trainer.CheckpointStore.save
        def save_then_interrupt(store, name, payload):
            """Commit the requested boundary, then simulate process loss before later work or exports."""
            value = original_save(store, name, payload)
            if name == boundary:
                raise RuntimeError("simulated process loss after durable boundary")
            return value
        with monkeypatch.context() as patch:
            patch.setattr(trainer.CheckpointStore, "save", save_then_interrupt)
            with pytest.raises(RuntimeError, match="simulated process loss"):
                run(interrupted)
        assert not (interrupted / "gift_main.pt").exists()
        with pytest.raises(ValueError, match="configuration differs"):
            run(interrupted, resume=True, configuration=replace(config, seed=812))
        resumed = run(interrupted, resume=True)
        expected = torch.load(continuous / "gift_main.pt", weights_only=True)
        observed = torch.load(interrupted / "gift_main.pt", weights_only=True)
        assert expected["model_state_dict"].keys() == observed["model_state_dict"].keys()
        assert all(torch.equal(value, observed["model_state_dict"][key])
                   for key, value in expected["model_state_dict"].items())
        assert reference["optimization"]["phase_records"] == resumed["optimization"]["phase_records"]
        assert reference["metrics"] == resumed["metrics"]
        assert reference["identifiability_final"] == resumed["identifiability_final"]
        assert resumed["freshness_contract"]["same_attempt_continuation"] is True
        assert resumed["freshness_contract"]["pretrained_model_loaded"] is False
        original_exports = {name: (interrupted / name).read_bytes()
                            for name in ("gift_main.pt", "report.json", "history.csv")}
        assert run(interrupted, resume=True) == resumed
        assert all((interrupted / name).read_bytes() == content
                   for name, content in original_exports.items())
        with h5py.File(dataset, "r+") as handle:
            handle["training/vorticity"][0, 3, 0, 0] += 1.
        with pytest.raises(ValueError, match="scientific inputs differ"):
            run(interrupted, resume=True)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("condition", ["noise_000", "noise_001", "noise_005"])
def test_dry_plan_condition_defaults_and_no_writes(tmp_path, monkeypatch, condition):
    """Planning each noise condition must retain defaults, mark missing input and leave the directory empty."""
    monkeypatch.setenv("GIFT_DATA_ROOT", str(tmp_path))
    plan = cli.build_plan(cli.parse_args(["--dry-run", "--condition", condition]))
    assert plan["dataset"] == str(tmp_path / cli.DATASETS[condition])
    assert plan["configuration"]["seed"] == 2026072301
    assert plan["configuration"]["phase_steps"] == (6000, 6000)
    assert plan["outputs_written"] == 0
    assert plan["dataset_exists"] is False
    assert plan["input_provenance"] is None
    assert plan["input_verification"] == "not_performed_missing_input"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("profile", ["released", "regenerated"])
def test_missing_input_plan_does_not_qualify_for_execution(tmp_path, monkeypatch, profile):
    """A readable missing-input plan must not let execution create output or start optimization."""
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "fresh"
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    common = ["--data-profile", profile, "--output", str(output), "--device", "cpu"]
    plan = cli.build_plan(cli.parse_args(["--dry-run", *common]))
    assert plan["input_verification"] == "not_performed_missing_input"
    monkeypatch.setattr(cli, "train_fresh_generator", lambda **kwargs: pytest.fail("training must not start"))
    with pytest.raises(FileNotFoundError):
        cli.run(cli.parse_args(["--run-training", *common]))
    assert not output.exists()


def test_resume_requires_explicit_own_output():
    """Reject a resume request without the journal directory it is intended to reopen."""
    with pytest.raises(SystemExit):
        cli.parse_args(["--resume"])


def test_interrupted_csv_write_does_not_publish_partial_history(tmp_path, monkeypatch):
    """A mid-write exception must leave no published CSV, allowing a later complete atomic write."""
    from experiments.formal._shared import common
    destination = tmp_path / "history.csv"
    rows = [{"step": 1, "loss": 2.}, {"step": 2, "loss": 1.}]
    def partial_write(writer, values):
        """Write the first row and raise to simulate interruption before the history file is complete."""
        writer.writerow(values[0])
        raise OSError("mid-file interruption")
    with monkeypatch.context() as patch:
        patch.setattr(common.csv.DictWriter, "writerows", partial_write)
        with pytest.raises(OSError, match="mid-file interruption"):
            common.write_csv_new(destination, rows)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))
    common.write_csv_new(destination, rows)
    assert "2,1.0" in destination.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("failed_export", ["model", "history", "report"])
def test_completed_generator_can_finish_interrupted_export(tmp_path, monkeypatch, failed_export):
    """Recover each missing terminal product without repeating affine fits/evaluation or overwriting completed products."""
    dataset = tmp_path / "observations.h5"
    make_observations(dataset)
    output = tmp_path / "run"
    output.mkdir()
    config = trainer.GeneratorTrainingConfig(
        training_trajectories=1, validation_trajectories=1, cutoff=1, rank=1,
        phase_steps=(1,), phase_learning_rates=(.01,), batch_size=2,
        affine_refit_interval=1, validation_interval=1, seed=811,
    )
    def run(resume=False):
        """Use a one-phase native generator fixture to reach export quickly on CPU."""
        return trainer.train_fresh_generator(
            dataset=dataset, checkpoint=output / "gift_main.pt", report_path=output / "report.json",
            history_path=output / "history.csv", condition="noise_000", device_name="cpu",
            config=config, project_root=cli.PROJECT_ROOT, entry_script=Path(cli.__file__),
            checkpoint_interval=1, resume=resume,
        )
    export_function = {"model": "write_model_export", "history": "write_csv_new",
                       "report": "write_json_export"}[failed_export]
    def interrupt(*args, **kwargs):
        """Inject an export-time IO failure after the final training checkpoint has been committed."""
        raise OSError("interrupted export")
    with monkeypatch.context() as patch:
        patch.setattr(trainer, export_function, interrupt)
        with pytest.raises(OSError, match="interrupted export"):
            run()
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    monkeypatch.setattr(trainer, "fit_affine_minimum_norm",
                        lambda *args, **kwargs: pytest.fail("completed export must not refit"))
    monkeypatch.setattr(trainer, "evaluate",
                        lambda *args, **kwargs: pytest.fail("completed export must not repeat evaluation"))
    report = run(resume=True)
    assert report["status"] == "complete"
    assert all((output / name).read_bytes() == content for name, content in before.items())
    assert all((output / name).is_file() for name in ("gift_main.pt", "report.json", "history.csv"))
