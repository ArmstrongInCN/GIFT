"""Test phase/optimizer/resume control with a tiny surrogate, not a science run."""

from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training import train_gift_predictor as control
from training.gift_prediction_control import PredictionTrainingConfig


class TinyBranch(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(()))

    def forward_with_generator_output(self, state, frozen):
        return self.weight * state


def toy_loader(*args, **kwargs):
    values = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
    return DataLoader(TensorDataset(values, 2 * values, values), batch_size=3), (values, values, values)


def install_tiny_control(monkeypatch):
    class Bank:
        def __init__(self, *args, **kwargs):
            self.training = np.zeros((4, 1), dtype=np.float32)
            self.binding = {"observations_sha256": "toy", "training_ids": [0, 1, 2, 3]}
    monkeypatch.setattr(control, "ObservationBank", Bank)
    monkeypatch.setattr(control, "qualify_generator", lambda *args: {})
    monkeypatch.setattr(control, "load_generator", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(control, "FixedBandwidthGridGenerator", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "derivative_loader", toy_loader)
    monkeypatch.setattr(control, "_training_scales", lambda *args: (1.0,) * 4)
    monkeypatch.setattr(control, "HighFrequencyBranch", TinyBranch)
    monkeypatch.setattr(control, "sequence_loader", lambda *args, **kwargs: toy_loader()[0])
    monkeypatch.setattr(control, "_validate_derivative", lambda *args: {"normalized_mse": 1.0})
    monkeypatch.setattr(control, "_validate_rollout", lambda *args: {"lead_1_full_relative_l2": 1.0})
    def toy_epoch(model, frozen, loader, optimizer, device):
        return control.derivative_epoch(model, loader, optimizer, device, 1.0)
    monkeypatch.setattr(control, "_short_epoch", toy_epoch)
    monkeypatch.setattr(control, "_long_epoch", toy_epoch)
    def toy_export(path, model, **kwargs):
        torch.save({"state": model.state_dict(), "selection": kwargs["phase"],
                    "cost": kwargs["training_cost"]}, path)
    monkeypatch.setattr(control, "_write_checkpoint", toy_export)


@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5, 6, 7])
def test_phase_optimizer_and_rng_resume_are_exact(tmp_path, monkeypatch, boundary):
    torch.set_num_threads(1)
    install_tiny_control(monkeypatch)
    prerequisite = tmp_path / "fixture.bin"
    prerequisite.write_bytes(b"explicit test surrogate, not model parameters")
    config = PredictionTrainingConfig(generator_phases=(2, 2), branch_phases=(2, 2, 2, 2),
                                      batch_size=3, rollout_batch_size=3, fixture_only=True)
    full, split = tmp_path / "full", tmp_path / "split"
    kwargs = dict(seed=20260820, device="cpu", config=config, checkpoint_interval=2)
    control.run_branch(prerequisite, prerequisite, full, **kwargs)
    control.run_branch(prerequisite, prerequisite, split, stop_after_epoch=boundary, **kwargs)
    assert not (split / "model.pt").exists()
    control.run_branch(prerequisite, prerequisite, split, resume=True, **kwargs)
    one, two = [torch.load(path / "model.pt", weights_only=True) for path in (full, split)]
    assert torch.equal(one["state"]["weight"], two["state"]["weight"])
    assert one["cost"]["optimizer_updates"] == two["cost"]["optimizer_updates"] == 16
    assert one["selection"] == two["selection"] == "terminal_epoch"
    left, right = [json.loads((path / "history.json").read_text()) for path in (full, split)]
    assert len(left) == len(right) == 8
    assert [{k: v for k, v in row.items() if k != "seconds"} for row in left] == [
        {k: v for k, v in row.items() if k != "seconds"} for row in right]


def test_unrelated_generator_is_rejected(tmp_path):
    class Bank:
        training = [None] * 1000
        binding = {"observations_sha256": "test", "training_ids": list(range(1000))}
    path = tmp_path / "wrong.pt"
    torch.save({"format_version": 3, "training_configuration": asdict(PredictionTrainingConfig())}, path)
    with pytest.raises(ValueError, match="completed fresh"):
        control.qualify_generator(path, Bank(), PredictionTrainingConfig())
