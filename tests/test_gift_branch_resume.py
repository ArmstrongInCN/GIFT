"""Exercise all GIFT high-branch phase transitions using a tiny CPU fixture.

The fixture checks control flow, optimizer/loader/RNG restoration and selection,
not the scientific accuracy or full-budget training of the real branch.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.formal import train_gift_branches as trainer


class TinyBranch(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(3))

    def forward_with_generator_output(self, state, frozen):
        return state * self.weights


def metric(model, loader):
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            total += float((batch[0] * model.weights).square().mean())
    return total


@pytest.mark.parametrize("boundary", ["derivative_2", "short_1", "long_1"])
def test_high_branch_resume_preserves_all_phases(tmp_path, monkeypatch, boundary):
    monkeypatch.setattr(trainer, "HighFrequencyBranch", TinyBranch)
    monkeypatch.setattr(trainer, "load_low_generator", lambda *args: None)
    monkeypatch.setattr(trainer, "DERIVATIVE_EPOCHS", 3)
    monkeypatch.setattr(trainer, "SHORT_ROLLOUT_EPOCHS", 2)
    monkeypatch.setattr(trainer, "LONG_ROLLOUT_STAGES", 2)
    monkeypatch.setattr(trainer, "_validate_derivative", lambda model, loader, *args: {"normalized_mse": metric(model, loader)})
    monkeypatch.setattr(trainer, "_validate_rollout", lambda model, rhs, loader, device: {"lead_1_full_relative_l2": metric(model, loader)})

    def epoch(model, rhs, loader, optimizer, device):
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
        if point == boundary:
            raise InterruptedError("intentional test boundary")

    with pytest.raises(InterruptedError):
        trainer._train_seed(**kwargs, checkpoint_directory=tmp_path / "interrupted", boundary_hook=interrupt)
    actual, actual_selection, actual_history = trainer._train_seed(
        **kwargs, checkpoint_directory=tmp_path / "interrupted", resume=True)
    assert actual_selection == expected_selection
    def numerical_history(rows):
        return [{key: value for key, value in row.items() if key != "training_validation_seconds"} for row in rows]
    assert numerical_history(actual_history) == numerical_history(expected_history)
    expected_cost = trainer.branch_training_cost(expected_history)
    actual_cost = trainer.branch_training_cost(actual_history)
    assert actual_cost["optimizer_updates"] == expected_cost["optimizer_updates"] == 7
    assert actual_cost["committed_training_validation_seconds"] >= 0
    assert actual_cost["shared_generator_pretraining_included"] is False
    assert all(torch.equal(value, expected.state_dict()[key]) for key, value in actual.state_dict().items())
