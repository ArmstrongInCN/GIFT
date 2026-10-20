"""Check true resume, condition-only forecasts, tail weighting and failed scores."""
from pathlib import Path

import numpy as np
import pytest
import torch

from training import w1_v2_train as control
from training.w1_v2_data import epoch_order
from training.w1_v2_data import window_manifest


class ToyBank:
    """Three observed trajectories with all 151 targets; no external/test IO."""
    points = 151

    def __init__(self, path, fixture=False):
        rng = np.random.default_rng(10)
        self.training = rng.normal(size=(3, self.points, 2, 2)).astype("f4")
        self.validation = rng.normal(size=(2, self.points, 2, 2)).astype("f4")
        self.binding = {"fixture": True, "ids": [0, 1, 2]}

    def batch(self, rows, device, validation=False):
        truth = torch.tensor((self.validation if validation else self.training)[rows], device=device)
        return truth[:, 0], truth

    def normalization(self, method):
        return {"kind": "none"}


class ToyModel(torch.nn.Module):
    """Stochastic forecast makes omitted RNG restoration numerically observable."""
    points = 151

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.bias = torch.nn.Parameter(torch.randn(()))
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, initial, times=None):
        value = self.weight * self.dropout(initial) + self.bias
        points = self.points if times is None else len(times)
        return value[:, None].expand(-1, points, -1, -1)

    def source_identity(self):
        return {"fixture_only": True}


def test_resume_matches_uninterrupted_native_optimizer_rng_and_selection(tmp_path, monkeypatch):
    """Pause after epoch1; complete Adam+cosine+dropout state reproduces epoch3."""
    monkeypatch.setattr(control, "WindowBank", ToyBank)
    monkeypatch.setattr(control, "build_network", lambda spec, device: ToyModel().to(device))
    spec = dict(method="anie_ns", history=1, condition_mode="native", source_root="fixture")
    config = dict(seed=11, sample_seed=2026100301, fixture=True, learning_rate=.01,
                  max_epochs=3, micro_batch=1, effective_batch=2, validation_interval=1,
                  stage="development", min_epochs=2, patience=10)
    whole = control.run_training("fixture", tmp_path / "whole", model_spec=spec, config=config, device="cpu")
    control.run_training("fixture", tmp_path / "segmented", model_spec=spec, config=config,
                         device="cpu", stop_after_epoch=1)
    split = control.run_training("fixture", tmp_path / "segmented", model_spec=spec, config=config,
                                 device="cpu", resume=True)
    for name, value in whole["model_state"].items():
        assert torch.equal(value, split["model_state"][name])
    assert whole["scheduler"] == split["scheduler"]
    assert whole["selection"] == split["selection"]
    assert whole["updates"] == split["updates"] == 6
    assert [row["loss"] for row in whole["history"]] == [row["loss"] for row in split["history"]]
    assert torch.equal(whole["rng"]["torch_cpu"], split["rng"]["torch_cpu"])


def test_sampling_is_common_and_independent_of_model_rng():
    """Changing model seeds cannot change training exposures or omit the tail8."""
    first = epoch_order(1000, 19)
    torch.manual_seed(33)
    np.random.seed(888)
    assert np.array_equal(first, epoch_order(1000, 19))
    assert np.array_equal(np.sort(first), np.arange(1000))
    assert len([first[start:start + 32] for start in range(0, 1000, 32)]) == 32
    assert len(first[992:]) == 8


def test_selection_retains_earlier_ties_and_accumulates_small_improvements():
    """Best and plateau rules have different purposes; failed scores never win."""
    selected = dict(best_metric=float("inf"), best_epoch=None, plateau_anchor=float("inf"), stale_checks=0)
    assert control.update_selection(selected, 1., 20)
    assert not control.update_selection(selected, 1., 40)
    assert selected["best_epoch"] == 20
    assert control.update_selection(selected, .9995, 60)
    assert selected["stale_checks"] == 2
    assert control.update_selection(selected, .9989, 80)
    assert selected["stale_checks"] == 0
    assert not control.update_selection(selected, float("inf"), 100)


def test_future_failure_invalidates_whole_validation_and_t0_is_excluded():
    """A failed trajectory cannot be hidden by nanmean or a finite subset."""
    truth = torch.ones((2, 151, 2, 2))
    prediction = truth.clone()
    prediction[:, 0] = 30
    curves = control.relative_curves(prediction, truth)
    assert float(curves[:, 1:].mean()) == 0
    prediction[1, 150, 0, 0] = float("nan")
    failed = control.relative_curves(prediction, truth)
    assert not bool(torch.isfinite(failed[:, 1:]).all())
    assert control.finite_json(dict(future=float("inf"), failures=1)) == dict(future=None, failures=1)


def test_effective_tail_gradient_uses_actual_samples():
    """A 3-sample tail accumulated in microbatches matches its full mean."""
    values = torch.tensor([1., 3., 7.])
    full = torch.tensor(.4, requires_grad=True)
    ((full * values - 2).square().mean()).backward()
    accumulated = torch.tensor(.4, requires_grad=True)
    for part in (values[:2], values[2:]):
        ((accumulated * part - 2).square().mean() * len(part) / len(values)).backward()
    assert torch.allclose(full.grad, accumulated.grad, rtol=1e-6)


@pytest.mark.parametrize("training_count,time_stride,grid_points", [(200, 5, 31), (300, 2, 76)])
def test_sparse_resume_binds_grid_and_restores_rng_optimizer_and_scheduler(
        tmp_path, monkeypatch, training_count, time_stride, grid_points):
    """Both sparse grids have the same exact pause/resume guarantee as legacy151."""
    class SparseBank(ToyBank):
        points = grid_points

        def __init__(self, path, fixture=False):
            super().__init__(path, fixture)
            self.binding.update(query_times=window_manifest(training_count, time_stride)["query_times"],
                                future_steps=grid_points - 1, points=grid_points)

    class SparseModel(ToyModel):
        points = grid_points

    monkeypatch.setattr(control, "WindowBank", SparseBank)
    monkeypatch.setattr(control, "build_network", lambda spec, device: SparseModel().to(device))
    spec = dict(method="anie_ns", history=1, condition_mode="native", source_root="fixture")
    config = dict(seed=11, sample_seed=2026100301, fixture=True, learning_rate=.01,
                  max_epochs=3, micro_batch=1, effective_batch=2, validation_interval=1,
                  stage="development", min_epochs=2, patience=10, future_steps=grid_points - 1)
    whole = control.run_training("fixture", tmp_path / "whole31", model_spec=spec, config=config, device="cpu")
    control.run_training("fixture", tmp_path / "segmented31", model_spec=spec, config=config,
                         device="cpu", stop_after_epoch=1)
    split = control.run_training("fixture", tmp_path / "segmented31", model_spec=spec, config=config,
                                 device="cpu", resume=True)
    assert whole["identity"]["model_spec"]["future_steps"] == grid_points - 1
    assert whole["history"][0]["training_future_steps"] == grid_points - 1
    for name, value in whole["model_state"].items():
        assert torch.equal(value, split["model_state"][name])
    assert whole["scheduler"] == split["scheduler"]
    assert whole["selection"] == split["selection"]
    assert whole["updates"] == split["updates"] == 6
    assert [row["loss"] for row in whole["history"]] == [row["loss"] for row in split["history"]]
    assert torch.equal(whole["rng"]["torch_cpu"], split["rng"]["torch_cpu"])


@pytest.mark.parametrize("population,updates,tail", [(200, 7, 8), (300, 10, 12)])
def test_sparse_population_has_complete_updates_and_actual_tail(population, updates, tail):
    """Every selected row is seen exactly once, including the actual trailing group."""
    order = epoch_order(population, 19)
    groups = [order[start:start + 32] for start in range(0, population, 32)]
    assert len(groups) == updates and len(groups[-1]) == tail
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(population))


@pytest.mark.parametrize("training_count,time_stride,steps", [(200, 5, 30), (300, 2, 75)])
def test_sparse_fno2d_closed_loop_keeps_complete_multistep_gradient(training_count, time_stride, steps):
    """Both sparse defaults are recursive chains; no truth, detach or teacher forcing."""
    class Multiply(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.9))

        def forward(self, context):
            return self.weight * context

    model = Multiply()
    initial = torch.ones((1, 2, 2))
    spec = dict(method="fno2d", query_times=window_manifest(training_count, time_stride)["query_times"], future_steps=steps)
    prediction = control.forward_model(model, initial, spec, {}, training=True)
    assert prediction.shape == (1, steps + 1, 2, 2)
    prediction[:, -1].mean().backward()
    assert torch.allclose(model.weight.grad, steps * model.weight.detach() ** (steps - 1), rtol=1e-5)
    assert torch.allclose(prediction[0, :, 0, 0], model.weight.detach() ** torch.arange(steps + 1), rtol=1e-6)


@pytest.mark.parametrize("points", [31, 76])
def test_sparse_native_losses_keep_initial_mse_and_future_sum_conventions(points):
    """Grid changes are explicit objective changes, not an implicit loss rescaling."""
    truth = torch.ones((2, points, 2, 2))
    prediction = truth.clone()
    prediction[:, 0] = 2
    assert torch.allclose(control.native_loss(prediction, truth, "anie_ns"), torch.tensor(1 / points))
    prediction[:, 1:] = 2

    def relative_sum(predicted, actual):
        return ((predicted - actual).norm(dim=1) / actual.norm(dim=1)).sum()

    assert control.native_loss(prediction, truth, "fno2d", relative_sum).item() == points - 1
    assert control.native_loss(prediction, truth, "fno3d", relative_sum).item() == 1
