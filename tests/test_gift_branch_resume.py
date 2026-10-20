"""Exercise all GIFT high-branch phase transitions using a tiny CPU fixture.

The fixture checks control flow, optimizer/loader/RNG restoration and selection,
not the scientific accuracy or full-budget training of the real branch.
"""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from experiments.formal import train_gift_branches as trainer


class TinyBranch(torch.nn.Module):
    """Three scalar branch weights preserve real optimizer/loader state paths without costly spectral training."""
    def __init__(self, *args, **kwargs):
        """Initialize random weights and the four scale buffers expected by branch checkpoint export."""
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(3))
        for name in ("state_scale", "high_scale", "low_rhs_scale", "high_rhs_scale"):
            self.register_buffer(name, torch.tensor(1.))

    def forward_with_generator_output(self, state, frozen):
        """Provide a differentiable three-channel RHS, ignoring frozen physics in this control fixture."""
        return state * self.weights

    def export_config(self):
        """Label the three-dimensional surrogate explicitly so exports are not scientific model evidence."""
        return {"fixture_only": True, "dimension": 3}


def metric(model, loader):
    """Compute a deterministic loader-wide scalar score for comparing continuation selection/history."""
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            total += float((batch[0] * model.weights).square().mean())
    return total


@pytest.mark.parametrize("boundary", ["derivative_2", "short_1", "long_1"])
def test_high_branch_resume_preserves_all_phases(tmp_path, monkeypatch, boundary):
    """Continuous and resumed derivative/short/long stages must agree; changed data or frozen weights must fail."""
    monkeypatch.setattr(trainer, "HighFrequencyBranch", TinyBranch)
    monkeypatch.setattr(trainer, "load_low_generator", lambda *args: SimpleNamespace(model=torch.nn.Linear(3, 3)))
    monkeypatch.setattr(trainer, "DERIVATIVE_EPOCHS", 3)
    monkeypatch.setattr(trainer, "SHORT_ROLLOUT_EPOCHS", 2)
    monkeypatch.setattr(trainer, "LONG_ROLLOUT_STAGES", 2)
    monkeypatch.setattr(trainer, "_validate_derivative", lambda model, loader, *args: {"normalized_mse": metric(model, loader)})
    monkeypatch.setattr(trainer, "_validate_rollout", lambda model, rhs, loader, device: {"lead_1_full_relative_l2": metric(model, loader)})

    def epoch(model, rhs, loader, optimizer, device):
        """Perform cheap surrogate updates while consuming both global RNG and loader ordering."""
        model.train()
        loss_sum = 0.0
        for (state,) in loader:
            optimizer.zero_grad()
            # Include a global RNG draw as well as the loader's private generator.
            loss = ((state + torch.randn_like(state)) * model.weights).square().mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        return loss_sum

    monkeypatch.setattr(trainer, "_short_epoch", epoch)
    monkeypatch.setattr(trainer, "_long_epoch", epoch)
    data = tmp_path / "observations.bin"
    low = tmp_path / "low.bin"
    data.write_bytes(b"synthetic test data identity")
    low.write_bytes(b"synthetic frozen model identity")
    paths = SimpleNamespace(standard_n64=data, low_model=low)
    state = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
    arrays = (state, state * 0.5, state.copy(), np.zeros_like(state))
    kwargs = dict(seed=20260820, paths=paths, training=arrays, validation=arrays,
                  scales=(1.0, 1.0, 1.0, 1.0), device=torch.device("cpu"))
    expected, expected_selection, expected_history = trainer._train_seed(
        **kwargs, checkpoint_directory=tmp_path / "uninterrupted")

    def interrupt(point):
        """Raise at the named durable training boundary to imitate process interruption."""
        if point == boundary:
            raise InterruptedError("intentional test boundary")

    with pytest.raises(InterruptedError):
        trainer._train_seed(**kwargs, checkpoint_directory=tmp_path / "interrupted", boundary_hook=interrupt)
    changed_arrays = tuple(value.copy() for value in arrays)
    changed_arrays[0][0, 0] += .125
    with pytest.raises(ValueError, match="scientific inputs"):
        trainer._train_seed(**{**kwargs, "training": changed_arrays},
                            checkpoint_directory=tmp_path / "interrupted", resume=True)
    original_loader = trainer.load_low_generator
    def changed_frozen(*arguments):
        """Perturb one loaded frozen weight to verify scientific-input compatibility is enforced."""
        result = original_loader(*arguments)
        with torch.no_grad():
            result.model.weight[0, 0] += .125
        return result
    monkeypatch.setattr(trainer, "load_low_generator", changed_frozen)
    with pytest.raises(ValueError, match="scientific inputs"):
        trainer._train_seed(**kwargs, checkpoint_directory=tmp_path / "interrupted", resume=True)
    monkeypatch.setattr(trainer, "load_low_generator", original_loader)
    actual, actual_selection, actual_history = trainer._train_seed(
        **kwargs, checkpoint_directory=tmp_path / "interrupted", resume=True)
    assert actual_selection == expected_selection
    def numerical_history(rows):
        """Remove measured wall time while retaining all values that continuation must reproduce."""
        return [{key: value for key, value in row.items() if key != "training_validation_seconds"} for row in rows]
    assert numerical_history(actual_history) == numerical_history(expected_history)
    expected_cost = trainer.branch_training_cost(expected_history)
    actual_cost = trainer.branch_training_cost(actual_history)
    assert actual_cost["optimizer_updates"] == expected_cost["optimizer_updates"] == 7
    assert actual_cost["committed_training_validation_seconds"] >= 0
    assert actual_cost["shared_generator_pretraining_included"] is False
    assert all(torch.equal(value, expected.state_dict()[key]) for key, value in actual.state_dict().items())


@pytest.mark.parametrize("interruption", ["model", "history", "report", "manifest"])
def test_cpu_main_completes_interrupted_terminal_exports_without_optimization(tmp_path, monkeypatch, interruption):
    """Compose the real journal, tiny optimizer, atomic model export and reports."""
    import h5py
    from training import checkpoints

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    state = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
    observations = data / "standard_ns_n64_full_spectrum.h5"
    with h5py.File(observations, "x") as handle:
        handle.attrs["fixture_only"] = True
        handle["training/vorticity"] = state
        handle["validation/vorticity"] = state
    low = tmp_path / "synthetic_frozen.pt"
    frozen = torch.nn.Linear(3, 3)
    torch.save({"model_state_dict": frozen.state_dict(), "fixture_only": True}, low)
    def low_fixture(*args):
        """Rebuild the same frozen linear model from its tiny checkpoint for each controller invocation."""
        model = torch.nn.Linear(3, 3)
        model.load_state_dict(torch.load(low, weights_only=True)["model_state_dict"])
        return SimpleNamespace(model=model)
    monkeypatch.setattr(trainer, "HighFrequencyBranch", TinyBranch)
    monkeypatch.setattr(trainer, "load_low_generator", low_fixture)
    monkeypatch.setattr(trainer, "DERIVATIVE_EPOCHS", 1)
    monkeypatch.setattr(trainer, "SHORT_ROLLOUT_EPOCHS", 1)
    monkeypatch.setattr(trainer, "LONG_ROLLOUT_STAGES", 1)
    monkeypatch.setattr(trainer, "_validate_training_source_metadata", lambda _: {
        name: {"trajectory_ids": [0, 3]} for name in ("training", "validation")})
    monkeypatch.setattr(trainer, "validate_input", lambda *a, **k: {"profile": "released", "sha256": None})
    monkeypatch.setattr(trainer, "validate_low_prerequisite", lambda *a: None)
    monkeypatch.setattr(trainer, "_load_raw_training_data", lambda _: ((state, state * .5, state), (state, state * .5, state)))
    monkeypatch.setattr(trainer, "_compute_low_rhs", lambda rhs, values, device: np.zeros_like(values))
    monkeypatch.setattr(trainer, "_training_scales", lambda *a: (1., 1., 1., 1.))
    monkeypatch.setattr(trainer, "_validate_derivative", lambda model, loader, *a: {"normalized_mse": metric(model, loader)})
    monkeypatch.setattr(trainer, "_validate_rollout", lambda model, rhs, loader, device: {"lead_1_full_relative_l2": metric(model, loader)})
    def epoch(model, rhs, loader, optimizer, device):
        """Use deterministic scalar updates to reach real final-journal/export paths in three stages."""
        total = 0.
        for (values,) in loader:
            optimizer.zero_grad()
            loss = (values * model.weights).square().mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        return total
    monkeypatch.setattr(trainer, "_short_epoch", epoch)
    monkeypatch.setattr(trainer, "_long_epoch", epoch)
    output = tmp_path / "run"
    args = SimpleNamespace(project_root=trainer.PROJECT_ROOT, data_profile="released", low_model=low,
        run_training=True, dry_run=False, seed=20260820, device="cpu", execution="auto", output=output, resume=False)
    monkeypatch.setattr(trainer, "parse_args", lambda: args)

    original_model = trainer._write_checkpoint
    original_csv = trainer.write_csv_new
    original_json = checkpoints.write_json_export
    def model_export(*a, **k):
        """Write the real model product, then optionally interrupt before subsequent export bookkeeping."""
        original_model(*a, **k)
        if interruption == "model":
            raise InterruptedError("synthetic export interruption")
    def history_export(*a, **k):
        """Publish history before injecting the requested interruption in the export sequence."""
        original_csv(*a, **k)
        if interruption == "history":
            raise InterruptedError("synthetic export interruption")
    def metadata_export(path, value):
        """Write create-only metadata and fail immediately after the selected report/manifest product."""
        original_json(path, value)
        if path.name == interruption + ".json":
            raise InterruptedError("synthetic export interruption")
    monkeypatch.setattr(trainer, "_write_checkpoint", model_export)
    monkeypatch.setattr(trainer, "write_csv_new", history_export)
    monkeypatch.setattr(checkpoints, "write_json_export", metadata_export)
    with pytest.raises(InterruptedError, match="synthetic export interruption"):
        trainer.main()
    pointer = output / "checkpoints/LATEST.json"
    latest_before = pointer.read_bytes()
    saved = torch.load(pointer.parent / json.loads(latest_before)["file"], weights_only=True)
    assert saved["boundary"] == "long_1"
    preserved = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert output / "gift_seed_20260820.pt" in preserved

    monkeypatch.setattr(trainer, "_write_checkpoint", original_model)
    monkeypatch.setattr(trainer, "write_csv_new", original_csv)
    monkeypatch.setattr(checkpoints, "write_json_export", original_json)
    def no_optimizer(*a, **k):
        """Fail if completed-run recovery constructs an optimizer instead of only completing export."""
        pytest.fail("terminal resume constructed an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", no_optimizer)
    args.resume = True
    trainer.main()
    assert pointer.read_bytes() == latest_before
    assert all(path.read_bytes() == content for path, content in preserved.items())
    assert {"gift_seed_20260820.pt", "training_history.csv", "report.json", "manifest.json"} <= {p.name for p in output.iterdir()}
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete" and report["execution"] == "eager"
    assert report["per_seed"]["20260820"]["training_cost"]["optimizer_updates"] == 3
    assert not list(output.glob("*.tmp"))
    # A second completion is create-only too, including metadata and model bytes.
    completed = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    trainer.main()
    assert all(path.read_bytes() == content for path, content in completed.items())
