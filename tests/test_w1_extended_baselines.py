"""CPU contracts for fixed short-horizon native baselines and continuation."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from adapters import w1_extended as adapter
from training import w1_extended_baselines as control


@pytest.fixture(autouse=True)
def cpu_only_runtime(monkeypatch):
    """Fixtures must never initialize or inspect a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(control, "runtime_identity", lambda: {"fixture_only": True, "device": "cpu"})
    torch.set_num_threads(1)


def test_fixed_recipes_and_physical_batches():
    for method, batch, horizon, decay in (("uno", 16, 20, 1e-5), ("unet", 20, 4, 1e-4)):
        config = control.native_training_config(method)
        assert config["effective_batch"] == config["micro_batch"] == batch
        assert config["training_future_steps"] == horizon
        assert config["max_epochs"] == 500 and config["seed"] == 0
        assert config["learning_rate"] == .001 and config["weight_decay"] == decay
        assert config["selection"] == "terminal" and not config["closed_loop_curriculum"]
        with pytest.raises(ValueError, match="micro_batch"):
            control._validate_config(method, dict(config, micro_batch=1))
        with pytest.raises(ValueError, match="learning_rate"):
            control._validate_config(method, dict(config, learning_rate=.0001))


def relative_sum(prediction, truth):
    return ((prediction - truth).norm(dim=1) / truth.norm(dim=1)).sum()


def test_native_batch_time_sum_and_known_initial_exclusion():
    truth = torch.ones((2, 76, 3, 3))
    prediction = torch.ones((2, 21, 3, 3)) * 2
    prediction[:, 0] = 1000
    assert control.native_loss(prediction, truth, "uno", relative_sum).item() == pytest.approx(40)
    assert control.native_loss(prediction[:1], truth[:1], "uno", relative_sum).item() == pytest.approx(20)
    assert control.native_loss(prediction[:, :5], truth, "unet").item() == pytest.approx(8)
    truth[:, 1] = 0
    with pytest.raises(FloatingPointError, match="zero-norm"):
        control.native_loss(prediction, truth, "unet")


class ScaleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.1))

    def forward(self, value):
        return value * self.weight


@pytest.mark.parametrize("method", ["uno", "unet"])
def test_feedback_keeps_full_prediction_graph_and_checkpoint_recomputation(method):
    """A k-step scale recurrence has derivative k*a**(k-1), including all steps."""
    model = ScaleModel()
    spec = dict(method=method, history=1, future_steps=75)
    initial = torch.ones((2, 3, 3))
    direct = adapter.forward_model(model, initial, spec, steps=4, training=True)
    direct[:, -1].sum().backward()
    expected = torch.tensor(18 * 4 * 1.1 ** 3)
    assert torch.allclose(model.weight.grad, expected)
    model.zero_grad()
    recomputed = adapter.forward_model(model, initial, spec, steps=4, training=True, checkpoint_uno=True)
    recomputed[:, -1].sum().backward()
    assert torch.equal(direct, recomputed) and torch.allclose(model.weight.grad, expected)
    assert torch.equal(direct[:, 0], initial)


def test_public_constructors_and_n64_n96_shapes():
    """Validate actual unchanged public networks and one-field channel dimensions."""
    source = os.environ.get("GIFT_EXTERNAL_ROOT")
    if not source:
        pytest.skip("configure pinned public repositories for native CPU checks")
    for method in ("uno", "unet"):
        model = adapter.build_network(dict(method=method, history=1, source_root=source)).eval()
        if method == "uno":
            assert model.fc.in_features == 5 and model.width == 32
        else:
            assert model.conv1[0].in_channels == 1
            assert model.output_layer.in_channels == 17
        with torch.no_grad():
            for size in (64, 96):
                initial = torch.ones((1, size, size))
                result = adapter.forward_model(model, initial, dict(method=method, history=1, future_steps=75), steps=1)
                assert result.shape == (1, 2, size, size) and torch.isfinite(result).all()


class ToyBank:
    """Marked 76-time, three-trajectory data with a tail batch of one."""
    def __init__(self, path, fixture=False):
        self.training = np.random.default_rng(5).normal(size=(3, 76, 2, 2)).astype("f4") + 2
        self.validation = self.training[:2].copy()
        self.binding = dict(fixture=True, points=76, future_steps=75,
                            query_times=(np.arange(76) * .04).tolist())

    def batch(self, rows, device, validation=False):
        truth = torch.tensor((self.validation if validation else self.training)[rows], device=device)
        return truth[:, 0], truth


class ToyModel(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(()) * .1 + .8)
        self.bias = torch.nn.Parameter(torch.rand(()) * .01)
        self.gift_external_source = dict(directory=str(Path(source) / "uno"))

    def forward(self, value):
        return value * self.weight + self.bias


def toy_run(monkeypatch, method="uno", epochs=3):
    source = os.environ.get("GIFT_EXTERNAL_ROOT")
    if not source:
        pytest.skip("configure the original UNO optimizer/loss for CPU continuation")
    monkeypatch.setattr(control, "WindowBank", ToyBank)
    monkeypatch.setattr(control, "build_network", lambda spec, device: ToyModel(source).to(device))
    monkeypatch.setattr(control, "_source_provenance", lambda *args: {"fixture_only": True})
    config = control.native_training_config(method)
    config.update(fixture=True, max_epochs=epochs, effective_batch=2, micro_batch=2,
                  checkpoint_interval=1, validation_interval=1)
    return config, dict(method=method, history=1, source_root=source)


@pytest.mark.parametrize("method", ["uno", "unet"])
def test_same_run_resume_restores_native_adam_scheduler_rng_and_tail(tmp_path, monkeypatch, method):
    config, spec = toy_run(monkeypatch, method)
    whole = control.run_training("fixture", tmp_path / "whole", model_spec=spec, config=config, device="cpu")
    control.run_training("fixture", tmp_path / "split", model_spec=spec, config=config, device="cpu", stop_after_epoch=2)
    resumed = control.run_training("fixture", tmp_path / "split", model_spec=spec, config=config, device="cpu", resume=True)
    assert whole["updates"] == resumed["updates"] == 6
    assert whole["scheduler"] == resumed["scheduler"]
    assert torch.equal(whole["rng"]["torch_cpu"], resumed["rng"]["torch_cpu"])
    assert [row["loss"] for row in whole["history"]] == [row["loss"] for row in resumed["history"]]
    for key, value in whole["model_state"].items():
        assert torch.equal(value, resumed["model_state"][key])
    weights = torch.load(tmp_path / "split" / "terminal_weights.pt", weights_only=True)
    assert weights["schema"] == control.WEIGHTS_SCHEMA and weights["selection"] == "terminal"
    assert weights["epoch"] == 3 and weights["normalization"] == {"kind": "none"}
    assert len(list((tmp_path / "split").glob("resume_evidence_*"))) == 1
    assert not (tmp_path / "split" / "best.pt").exists()


def test_validation_failures_do_not_change_finite_terminal_policy(tmp_path, monkeypatch):
    config, spec = toy_run(monkeypatch, "unet")
    monkeypatch.setattr(control, "validation_measurement", lambda *args: dict(
        validation_mse=float("inf"), future_mse=float("inf"), failed_trajectories=2))
    result = control.run_training("fixture", tmp_path / "finite", model_spec=spec, config=config, device="cpu")
    assert result["status"] == "complete"
    weights = torch.load(tmp_path / "finite" / "terminal_weights.pt", weights_only=True)
    assert all(torch.isfinite(value).all() for value in weights["model_state"].values())
    status = json.loads((tmp_path / "finite" / "STATUS.json").read_text())
    assert status["last_validation"]["validation_mse"] is None and status["status"] == "complete"


def test_true_numerical_failure_and_oom_are_visible_and_cannot_retry(tmp_path, monkeypatch):
    config, spec = toy_run(monkeypatch, "unet")
    original = torch.optim.Adam
    class PoisonAdam(original):
        def step(self, closure=None):
            value = super().step(closure)
            self.param_groups[0]["params"][0].data.fill_(float("inf"))
            return value
    monkeypatch.setattr(torch.optim, "Adam", PoisonAdam)
    with pytest.raises(FloatingPointError, match="after update"):
        control.run_training("fixture", tmp_path / "failed", model_spec=spec, config=config, device="cpu")
    failure = json.loads((tmp_path / "failed" / "FAILURE.json").read_text())
    assert failure["failure_kind"] == "numerical" and failure["physical_batch"] == 2
    with pytest.raises(ValueError, match="unique slot"):
        control.run_training("fixture", tmp_path / "failed", model_spec=spec, config=config, device="cpu", resume=True)
    assert control._failure_kind(torch.cuda.OutOfMemoryError("CPU fixture OOM")) == "resource"
