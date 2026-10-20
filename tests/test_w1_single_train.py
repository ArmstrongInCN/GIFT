"""CPU contracts for the single native run, loss scale, schedules and true resume."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from training import w1_single_train as control


@pytest.fixture(autouse=True)
def cpu_only_runtime(monkeypatch):
    """No fixture test may inspect or initialize the CUDA runtime."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(control, "runtime_identity", lambda: {"fixture_only": True, "device": "cpu"})


def relative_sum(predicted, actual):
    return ((predicted - actual).norm(dim=1) / actual.norm(dim=1)).sum()


def test_official_fixed_budgets_batches_seeds_and_curriculum():
    configs = {method: control.native_training_config(method) for method in control.METHODS}
    assert configs["anie_ns"]["seed"] == 7 and configs["anie_ns"]["max_epochs"] == 5000
    assert configs["fno2d"]["effective_batch"] == 20 and configs["fno3d"]["effective_batch"] == 10
    assert configs["fno2d"]["max_epochs"] == configs["fno3d"]["max_epochs"] == 500
    assert configs["fno2d"]["seed"] == configs["fno3d"]["seed"] == 0
    assert [control.training_horizon("fno2d", epoch, configs["fno2d"])
            for epoch in (1, 21, 41, 61, 81, 101, 121, 500)] == [1, 2, 5, 10, 25, 50, 75, 75]
    bad = dict(configs["fno2d"], learning_rate=.0003)
    with pytest.raises(ValueError, match="learning_rate"):
        control._validate_config("fno2d", bad)
    bad = dict(configs["fno2d"], micro_batch=10)
    with pytest.raises(ValueError):
        control._validate_config("fno2d", bad)


def test_native_batch_sum_loss_and_initial_mse_are_distinct():
    truth = torch.ones((2, 76, 2, 2))
    prediction = truth.clone()
    prediction[:, 0] = 2
    assert control.native_loss(prediction, truth, "anie_ns").item() == pytest.approx(1 / 76)
    prediction[:, 1:] = 2
    assert control.native_loss(prediction, truth, "fno2d", relative_sum).item() == 150
    assert control.native_loss(prediction, truth, "fno3d", relative_sum).item() == 2
    # Sum loss changes proportionally with actual batch size, not with a hidden mean.
    assert control.native_loss(prediction[:1], truth[:1], "fno2d", relative_sum).item() == 75


def test_delayed_native_cosine_cycles_and_step_schedule():
    anie_config = control.native_training_config("anie_ns")
    parameter = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler = control.make_scheduler(optimizer, "anie_ns", anie_config)
    rates = []
    for epoch in range(1, 220):
        optimizer.step()
        control.advance_scheduler(scheduler, "anie_ns", epoch, anie_config)
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[:16] == [1e-4] * 16
    assert rates[16] < 1e-4
    assert rates[116] == pytest.approx(1e-9, abs=1e-16)  # 101 native scheduler steps.
    assert rates[117] > rates[116]
    assert rates[217] == pytest.approx(1e-4)  # Full native cosine return, not a floor clamp.
    fno_config = control.native_training_config("fno2d")
    optimizer = torch.optim.Adam([parameter], lr=.001)
    scheduler = control.make_scheduler(optimizer, "fno2d", fno_config)
    rates = []
    for epoch in range(1, 201):
        optimizer.step()
        control.advance_scheduler(scheduler, "fno2d", epoch, fno_config)
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[98] == .001 and rates[99] == .0005 and rates[199] == .00025


def test_native_early_stopping_ties_do_not_increment_or_reset():
    source = os.environ.get("GIFT_ANIE_SOURCE_ROOT")
    if not source:
        pytest.skip("configure the official ANIE source")
    stopper = control.native_early_stopping(source, patience=2)
    stopper(1.)
    stopper(2.)
    assert stopper.counter == 1
    stopper(1.)
    assert stopper.counter == 1 and not stopper.early_stop
    stopper(.5)
    assert stopper.counter == 0 and stopper.best_loss == .5
    stopper(1.)
    stopper(1.)
    assert stopper.early_stop


class ToyBank:
    """Tiny marked 76-time observations, never a formal data file."""
    def __init__(self, path, fixture=False):
        generator = np.random.default_rng(10)
        self.training = generator.normal(size=(3, 76, 2, 2)).astype("f4") + 2
        self.validation = generator.normal(size=(2, 76, 2, 2)).astype("f4") + 2
        self.binding = dict(fixture=True, points=76, future_steps=75,
                            query_times=(np.arange(76) * .04).tolist())

    def batch(self, rows, device, validation=False):
        truth = torch.tensor((self.validation if validation else self.training)[rows], device=device)
        return truth[:, 0], truth

    def normalization(self, method):
        return {"kind": "none"}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.bias = torch.nn.Parameter(torch.randn(()))
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, initial, times=None):
        result = self.weight * self.dropout(initial) + self.bias
        return result[:, None].expand(-1, len(times), -1, -1)


def toy_run(monkeypatch, max_epochs=3):
    source = os.environ.get("GIFT_ANIE_SOURCE_ROOT")
    if not source:
        pytest.skip("configure the official EarlyStopping source")
    monkeypatch.setattr(control, "WindowBank", ToyBank)
    monkeypatch.setattr(control, "build_network", lambda spec, device: ToyModel().to(device))
    monkeypatch.setattr(control, "_source_provenance", lambda model, spec: {"fixture_only": True})
    config = control.native_training_config("anie_ns")
    config.update(fixture=True, max_epochs=max_epochs, micro_batch=1, effective_batch=2, checkpoint_interval=1)
    spec = dict(method="anie_ns", history=1, condition_mode="native", source_profile=control.SOURCE_PROFILE,
                source_root=source)
    return config, spec


def test_resume_restores_native_optimizer_rng_scheduler_early_stopping_and_best(tmp_path, monkeypatch):
    config, spec = toy_run(monkeypatch, max_epochs=18)
    complete = control.run_training("fixture", tmp_path / "whole", model_spec=spec, config=config, device="cpu")
    control.run_training("fixture", tmp_path / "split", model_spec=spec, config=config,
                         device="cpu", stop_after_epoch=17)
    resumed = control.run_training("fixture", tmp_path / "split", model_spec=spec, config=config, device="cpu", resume=True)
    for name, value in complete["model_state"].items():
        assert torch.equal(value, resumed["model_state"][name])
    assert complete["scheduler"] == resumed["scheduler"]
    assert complete["early_stopping"] == resumed["early_stopping"]
    assert complete["selection"] == resumed["selection"]
    assert complete["updates"] == resumed["updates"] == 36
    assert [row["loss"] for row in complete["history"]] == [row["loss"] for row in resumed["history"]]
    assert torch.equal(complete["rng"]["torch_cpu"], resumed["rng"]["torch_cpu"])
    evidence = list((tmp_path / "split").glob("resume_evidence_*"))
    assert len(evidence) == 1 and (evidence[0] / "HISTORY.json").exists()
    weights = torch.load(tmp_path / "split" / "best.pt", weights_only=True)
    assert weights["schema"] == control.WEIGHTS_SCHEMA and weights["config"]["selection"] == "validation_mse_best"
    status = json.loads((tmp_path / "split" / "STATUS.json").read_text())
    assert Path(status["selected_checkpoint"]).is_file()
    assert status["validation_seconds"] > 0 and status["epoch_total_seconds"] >= status["training_seconds"]


def test_resume_retains_leading_uncommitted_history_bytes(tmp_path, monkeypatch):
    config, spec = toy_run(monkeypatch)
    directory = tmp_path / "ahead"
    control.run_training("fixture", directory, model_spec=spec, config=config, device="cpu", stop_after_epoch=1)
    history = json.loads((directory / "HISTORY.json").read_text())
    history["training"].append(dict(epoch=2, evidence="uncommitted previous attempt"))
    (directory / "HISTORY.json").write_text(json.dumps(history), encoding="utf-8")
    before = (directory / "HISTORY.json").read_bytes()
    control.run_training("fixture", directory, model_spec=spec, config=config, device="cpu", resume=True)
    snapshots = list(directory.glob("resume_evidence_*"))
    assert len(snapshots) == 1 and (snapshots[0] / "HISTORY.json").read_bytes() == before
    assert json.loads((snapshots[0] / "RESUME.json").read_text())["checkpoint_epoch"] == 1


def test_post_adam_nonfinite_is_scientific_failure_and_cannot_retry(tmp_path, monkeypatch):
    config, spec = toy_run(monkeypatch)
    original = torch.optim.Adam

    class PoisonAdam(original):
        def step(self, closure=None):
            result = super().step(closure)
            self.param_groups[0]["params"][0].data.fill_(float("inf"))
            return result

    monkeypatch.setattr(torch.optim, "Adam", PoisonAdam)
    with pytest.raises(FloatingPointError, match="after update"):
        control.run_training("fixture", tmp_path / "failed", model_spec=spec, config=config, device="cpu")
    failure = json.loads((tmp_path / "failed" / "FAILURE.json").read_text())
    assert failure["failure_kind"] == "numerical" and failure["selected_checkpoint"] is None
    assert failure["updates"] == 0 and (tmp_path / "failed" / "latest.pt").exists()
    with pytest.raises(ValueError, match="unique slot"):
        control.run_training("fixture", tmp_path / "failed", model_spec=spec, config=config, device="cpu", resume=True)


def test_anie_nonfinite_validation_records_numerical_failure(tmp_path, monkeypatch):
    config, spec = toy_run(monkeypatch)
    monkeypatch.setattr(control, "validation_measurement", lambda *args, **kwargs: dict(
        validation_mse=float("inf"), failed_trajectories=1))
    with pytest.raises(FloatingPointError, match="validation MSE"):
        control.run_training("fixture", tmp_path / "failed_val", model_spec=spec, config=config, device="cpu")
    failure = json.loads((tmp_path / "failed_val" / "FAILURE.json").read_text())
    assert failure["failure_kind"] == "numerical" and failure["failed_epoch"] == 1
    assert not (tmp_path / "failed_val" / "best.pt").exists()


def test_fno_terminal_weights_are_used_even_when_validation_worsens(tmp_path, monkeypatch):
    """The FNO endpoint is the budget terminal, never a validation best selection."""
    monkeypatch.setattr(control, "WindowBank", ToyBank)
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(control, "build_network", lambda spec, device: model.to(device))
    monkeypatch.setattr(control, "_source_provenance", lambda model, spec: {"fixture_only": True})
    monkeypatch.setattr(control, "fno_utilities", lambda root: (None, lambda **kwargs: relative_sum))
    def forecast(model, initial, spec, normalization, device=None, training=False, horizon=None):
        count = 76 if horizon is None else horizon + 1
        return model(initial.unsqueeze(-1)).squeeze(-1)[:, None].expand(-1, count, -1, -1)
    monkeypatch.setattr(control, "forward_model", forecast)
    monkeypatch.setattr(control, "validation_measurement", lambda *args, **kwargs: dict(
        validation_mse=float("inf"), failed_trajectories=2))
    config = control.native_training_config("fno2d")
    config.update(fixture=True, max_epochs=3, effective_batch=3, micro_batch=3, checkpoint_interval=1)
    spec = dict(method="fno2d", history=1, source_root="fixture")
    result = control.run_training("fixture", tmp_path / "fno", model_spec=spec, config=config, device="cpu")
    terminal = torch.load(tmp_path / "fno" / "terminal_weights.pt", weights_only=True)
    assert terminal["epoch"] == 3 and terminal["selection"] == "terminal"
    assert result["selection"]["best_epoch"] is None and not (tmp_path / "fno" / "best.pt").exists()
    assert result["updates"] == 3 and result["early_stopping"] is None
    status = json.loads((tmp_path / "fno" / "STATUS.json").read_text())
    assert status["status"] == "complete" and status["last_validation"]["validation_mse"] is None
    assert status["last_validation"]["failed_trajectories"] == 2
