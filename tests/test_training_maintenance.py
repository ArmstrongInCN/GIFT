"""Small CPU fixtures for valid checkpoint commits and interrupted terminal exports."""
import json
import numpy as np
import pytest
import torch
from training.checkpoints import CheckpointStore, write_model_export, restore_optimizer
from training.train_pinn import CoupledJournal
from test_train_pinn_fsm import coordinator


def test_nonfinite_pinn_snapshot_never_advances_latest(tmp_path):
    """A NaN state must fail before publication, leaving the preceding finite PINN boundary resumable."""
    control = coordinator()
    journal = CoupledJournal(tmp_path / "run", control)
    journal.save()
    before = (journal.directory / "LATEST.json").read_bytes()
    control.advance()
    control.backend.values[0][0] = np.nan
    with pytest.raises(FloatingPointError, match="previous checkpoint"):
        journal.save()
    assert (journal.directory / "LATEST.json").read_bytes() == before
    restored = coordinator()
    CoupledJournal(journal.directory, restored, resume=True)
    assert restored.state["nadam_total"] == 0
    assert np.isfinite(restored.backend.values[0]).all()


def test_terminal_model_export_is_atomic_and_idempotent(tmp_path):
    """Matching model export retains bytes; differing learned tensors are rejected without replacement."""
    payload = {"model_state_dict": {"w": torch.arange(3)}}
    path = tmp_path / "model.pt"
    write_model_export(path, payload)
    before = path.read_bytes()
    write_model_export(path, payload)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="differs"):
        write_model_export(path, {"model_state_dict": {"w": torch.zeros(3)}})
    assert path.read_bytes() == before


def test_terminal_export_checks_normalization_without_overwriting(tmp_path):
    """Equal weights with different fitted normalization must fail and preserve the original export."""
    path = tmp_path / "model.pt"
    payload = {"model_state_dict": {"w": torch.ones(1)},
               "normalization": {"input_mean": torch.ones(2), "eps": 1e-5}}
    write_model_export(path, payload)
    before = path.read_bytes()
    changed = dict(payload, normalization={"input_mean": torch.zeros(2), "eps": 1e-5})
    with pytest.raises(ValueError, match="configuration differs"):
        write_model_export(path, changed)
    assert path.read_bytes() == before


def test_native_optimizer_rejects_wrong_parameter_slot_shape():
    """An Adam momentum tensor with a different parameter shape must be rejected during restore."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward(); optimizer.step()
    state = optimizer.state_dict()
    state["state"][0]["exp_avg"] = torch.zeros(3)
    with pytest.raises(ValueError, match="shape differs"):
        restore_optimizer(torch.optim.Adam(model.parameters()), state)


def test_resume_checks_loaded_scientific_arrays_but_accepts_legacy_missing_digest(tmp_path):
    """Compare fingerprints when both exist while allowing a legacy journal lacking that field."""
    saved = {"seed": 2, "data": {"training_ids": [1, 2], "observations_sha256": "first"}}
    store = CheckpointStore(tmp_path / "run", saved)
    store.save("initial", {"model": torch.ones(1)})
    changed = {"seed": 2, "data": {"training_ids": [1, 2], "observations_sha256": "changed"}}
    with pytest.raises(ValueError, match="scientific inputs"):
        CheckpointStore(store.directory, changed, resume=True)
    CheckpointStore(store.directory, {"seed": 2, "data": {"training_ids": [1, 2]}}, resume=True)


def test_generator_resume_ignores_branch_only_configuration():
    """Changing future branch budgets is harmless to a generator; changing its own phase counts must fail."""
    from training.configuration import validate_resume
    saved = {"role": "gift_prediction_generator", "configuration": {"generator_phases": [1, 1], "branch_phases": [4, 1, 1, 1]}}
    current = {"role": "gift_prediction_generator", "configuration": {"generator_phases": [1, 1], "branch_phases": [2, 1, 1, 1]}}
    validate_resume(current, saved)
    current["configuration"]["generator_phases"] = [2, 1]
    with pytest.raises(ValueError, match="training configuration"):
        validate_resume(current, saved)


def test_pinn_complete_resume_preserves_committed_provenance(tmp_path):
    """Repairing complete PINN output after comment/IO changes must retain original metadata and file bytes."""
    from test_train_pinn_fsm import drive
    from training.train_pinn import terminal_export
    original = coordinator(); drive(original)
    journal = CoupledJournal(tmp_path / "run", original); journal.save()
    first = terminal_export(journal)
    before = {path.name: path.read_bytes() for path in journal.directory.iterdir() if path.is_file()}
    changed = coordinator()
    changed.identity["source_sha256"] = "comment edit"
    changed.identity["nadam_checkpoint_interval"] = 1
    resumed = CoupledJournal(journal.directory, changed, resume=True)
    assert terminal_export(resumed) == first
    assert before == {path.name: path.read_bytes() for path in journal.directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("phase,change", [
    ("pre_nadam", {"phase_step": 2}),
    ("ado_nadam", {"phase_step": 2}),
    ("post_nadam", {"phase_step": 3}),
    ("ado_stridge", {"nadam_total": 99}),
    ("complete", {"serial": 0}),
])
def test_pinn_restore_rejects_impossible_phase_progress(phase, change):
    """Corrupt actual phase cursors, cumulative updates or serial counts and require restore failure."""
    from test_train_pinn_fsm import drive
    control = coordinator()
    if phase != "pre_nadam":
        drive(control, lambda value: value.state["phase"] == phase)
    arrays, metadata = control.snapshot()
    # Corrupt one actual progress field while keeping identities/layout valid.
    metadata["fsm"] = dict(metadata["fsm"], **change)
    with pytest.raises(ValueError):
        coordinator().restore(arrays, metadata)


@pytest.mark.parametrize("method,schema", [("anie", "gift.anie-terminal-weights.v1"),
                                           ("anie_dense151", "gift.anie-dense-terminal-weights.v1")])
def test_shared_anie_export_preserves_method_schema(tmp_path, method, schema):
    """Sparse and dense151 ANIE exports must retain their distinct method/schema pairs."""
    from training.anie_control import export_terminal
    config = {"method": method, "epochs": 1, "trajectories": 1, "formal": False}
    history = [{"epoch": 1, "processed_trajectories": 1, "optimizer_updates_this_epoch": 1, "epoch_seconds": 0.0}]
    identity = {"configuration": config}
    store = CheckpointStore(tmp_path / "run", identity)
    store.save("epoch_0001", {"epoch": 1, "history": history})
    export_terminal(store.directory, torch.nn.Linear(1, 1), config, {"grid": 64}, identity, history, 1, False)
    payload = torch.load(store.directory / "model.pt", weights_only=True)
    assert payload["schema"] == schema and payload["method"] == method


def test_generator_finishes_interrupted_export_without_training_again(tmp_path, monkeypatch):
    """Completed generator recovery must supply missing metadata without another AdamW update or weight rewrite."""
    from training import gift_prediction_control as control
    from test_gift_prediction_training import make_observations
    data = tmp_path / "observations.h5"; make_observations(data)
    output = tmp_path / "run"
    config = control.PredictionTrainingConfig(generator_phases=(1, 1), branch_phases=(1, 1, 1, 1),
                                              batch_size=3, cutoff=2, rank=2, fixture_only=True)
    original = control.write_json_export
    def interrupted(path, value):
        """Fail specifically at history export, after the final model file has already been published."""
        if path.name == "history.json": raise RuntimeError("export interrupted")
        return original(path, value)
    monkeypatch.setattr(control, "write_json_export", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        control.run_generator(data, output, device="cpu", config=config)
    before = (output / "model.pt").read_bytes()
    monkeypatch.setattr(control, "write_json_export", original)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("completed run must not update"))
    report = control.run_generator(data, output, device="cpu", config=config, resume=True)
    assert report["status"] == "complete"
    assert (output / "model.pt").read_bytes() == before
    assert (output / "history.json").exists() and (output / "COMPLETE.json").exists()


def test_branch_cutoff_is_checked_before_input_loading(tmp_path):
    """An unsupported cutoff must raise before opening dataset/generator files or creating output."""
    from training.train_gift_predictor import run_branch
    from training.gift_prediction_control import PredictionTrainingConfig
    with pytest.raises(ValueError, match="cutoff=21"):
        run_branch(tmp_path / "not-read", tmp_path / "not-read", tmp_path / "output", seed=9,
                   config=PredictionTrainingConfig(cutoff=2))
