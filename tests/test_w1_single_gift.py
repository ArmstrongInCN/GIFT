"""CPU checks of the fixed GIFT recipe, terminal selection and own-run recovery."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import training.w1_single_gift as native
from training import w1_v2_gift as common


def make_slice(path, grid=8):
    """A tiny dense fixture tests mechanics; it never qualifies as the N300 run."""
    rng = np.random.default_rng(12)
    with h5py.File(path, "x") as handle:
        handle.attrs.update(schema=common.SCHEMA, status="complete", fixture_only=True, trajectory_id_scheme="canonical")
        for name, ids in (("training", np.arange(4)), ("validation", np.arange(1000, 1002))):
            group = handle.create_group(name)
            times = np.arange(151)*.02
            group.create_dataset("time", data=times)
            group.create_dataset("absolute_time", data=times+5)
            group.create_dataset("trajectory_index", data=ids)
            a = rng.normal(size=(len(ids), 1, grid, grid)).astype(np.float32)
            b = rng.normal(size=a.shape).astype(np.float32)
            group.create_dataset("vorticity", data=(a+times[None, :, None, None]*b).astype(np.float32))


def fixture_config(**values):
    return native.NativeGIFTConfig(generator_phases=(2, 2), branch_phases=(2, 1, 1, 1),
              generator_batch=3, derivative_batch=3, rollout_batch=2, cutoff=2, rank=2,
              fixture_only=True, branch_monitor_interval=1, **values)


def diagnostic(low, branch, bank, device, config):
    """Deliberately unavailable future validation cannot substitute old weights."""
    return {"val_metric": None, "seconds": 0., "failed_trajectory_ids": bank.binding["validation_ids"],
            "failed_trajectory_count": len(bank.validation)}


def test_locked_recipe_and_native_update_counts():
    """N300 produces physical19 and60 batches, not the old10 accumulated groups."""
    config = native.NativeGIFTConfig()
    config.validate()
    assert config.generator_phases == (250, 250)
    assert config.branch_phases == (400, 50, 25, 25)
    assert 500*math.ceil(300/config.generator_batch) == 9500
    assert 400*math.ceil(300/config.derivative_batch)+100*math.ceil(300/config.rollout_batch) == 13600
    fixed = native.native_training_config()
    assert fixed["selection"] == "terminal_epoch" and not fixed["early_stopping"]
    assert not fixed["gradient_accumulation"]
    with pytest.raises(ValueError, match="pinned native"):
        replace(config, generator_batch=32).validate()
    # Historical W1-v2 behavior must remain its independent profile.
    assert common.GIFTTrainingConfig().branch_phases == (800, 100, 50, 50)
    assert np.array_equal(common._rollout_steps(.04), [0, 4, 10, 20, 24, 30, 40, 50])


def test_phase_local_generator_scheduler_reset_and_branch_constant_rates():
    """Both generator phases restart their own cosine; only branch phase0 has one."""
    config = fixture_config()
    p = torch.nn.Parameter(torch.tensor([1.]))
    for phase, rate in enumerate(config.generator_learning_rates):
        optimizer, scheduler = native._optimizer([p], "generator", phase, config, 4)
        assert scheduler.T_max == 4
        assert optimizer.param_groups[0]["lr"] == rate
        assert not optimizer.state
        for _ in range(4):
            optimizer.zero_grad(); p.square().sum().backward(); optimizer.step(); scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(rate*.02)
    for phase, rate in enumerate(config.branch_learning_rates):
        optimizer, scheduler = native._optimizer([p], "branch", phase, config, 4)
        assert not optimizer.state and optimizer.param_groups[0]["lr"] == rate
        if phase == 0:
            assert scheduler.T_max == 2
            for _ in range(2):
                optimizer.zero_grad(); p.square().sum().backward(); optimizer.step(); scheduler.step()
            assert optimizer.param_groups[0]["lr"] == pytest.approx(.0001)
        else:
            assert scheduler is None


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_real_generator_restore_and_terminal_export_even_failed_future_validation(tmp_path, monkeypatch, boundary):
    """Real factors/affine fits and per-update cosine match at and inside phases."""
    torch.set_num_threads(1)
    monkeypatch.setattr(common, "_validation", diagnostic)
    path = tmp_path/"slice.h5"
    make_slice(path)
    config = fixture_config()
    full, split = tmp_path/"full", tmp_path/"split"
    native.run_generator(path, full, config=config, device="cpu")
    native.run_generator(path, split, config=config, device="cpu", stop_after_epoch=boundary)
    result = native.run_generator(path, split, config=config, device="cpu", resume=True)
    assert result["status"] == "complete" and result["validation_status"] == "failed"
    a, b = (torch.load(folder/"latest.pt", weights_only=True)["payload"] for folder in (full, split))
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert a["scheduler"] == b["scheduler"]
    assert a["optimizer_updates"] == b["optimizer_updates"] == 8
    assert b["best"] is None
    assert b["scheduler"]["last_epoch"] == b["scheduler"]["T_max"] == 4
    for aa, bb in zip(a["history"], b["history"]):
        assert {key: value for key, value in aa.items() if key not in ("seconds", "validation")} == {
            key: value for key, value in bb.items() if key not in ("seconds", "validation")}
    model, terminal = (torch.load(split/name, weights_only=True) for name in ("model.pt", "terminal.pt"))
    assert model["epoch"] == 4 and model["selection"] == "terminal_epoch" and model["val_metric"] is None
    assert model["training_cost"]["affine_fits"] == 5
    assert all(torch.equal(model["model_state"][key], terminal["model_state"][key]) for key in model["model_state"])
    assert not (split/"best.pt").exists()
    before = (split/"model.pt").read_bytes()
    native.run_generator(path, split, config=config, device="cpu", resume=True)
    assert before == (split/"model.pt").read_bytes()


class SmallBranch(torch.nn.Module):
    """A stochastic CPU stand-in isolates controller state from expensive RK4."""
    def __init__(self, config, *, frozen_generator, **scales):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3))
        for name, value in scales.items():
            self.register_buffer(name, torch.tensor(value))

    def export_config(self):
        return {}


@pytest.mark.parametrize("boundary", [1, 2, 3, 4])
def test_branch_original_phase_reset_and_stochastic_resume(tmp_path, monkeypatch, boundary):
    """Check complete RNG/AdamW recovery and branch16/5 routing through all phases."""
    torch.set_num_threads(1)
    path = tmp_path/"slice.h5"
    make_slice(path, 64)
    config = replace(fixture_config(), cutoff=21, rank=8)
    bank = native._bank(path, config)
    generator = common.GIFTGenerator(21, 8, bank.state_scale, False)
    gp = tmp_path/"generator.pt"
    torch.save({"format_version": 3, "model_configuration": generator.configuration(),
                "model_state_dict": common.clone_state(generator), "epoch": 4,
                "selection": "terminal_epoch", "training_regime": "w1_single_native_t5_t8",
                "val_metric": None, "training_data": bank.binding,
                "training_cost": {"trajectory_epochs": 4, "optimizer_updates": 8,
                                  "committed_seconds": 0., "total_training_seconds": 0., "parameter_count": 1}}, gp)
    monkeypatch.setattr(common, "HighFrequencyBranch", SmallBranch)
    monkeypatch.setattr(common, "_compute_low_rhs", lambda frozen, states, device: np.ones_like(states))
    monkeypatch.setattr(common, "_training_scales", lambda *values: (1., 1., 1., 1.))
    monkeypatch.setattr(common, "_validation", diagnostic)
    monkeypatch.setattr(native, "_validate_derivative", lambda *args: {"normalized_mse": 1.})
    monkeypatch.setattr(native, "_validate_rollout", lambda *args: {"lead_1_full_relative_l2": 1.})
    used = []
    def epoch(model, frozen, bank, optimizer, loss_config, seed, global_epoch, phase, device, scales):
        batch = config.derivative_batch if phase == 0 else config.rollout_batch
        assert loss_config.effective_batch == loss_config.branch_micro_batch == batch
        used.append((phase, global_epoch, optimizer.param_groups[0]["lr"], len(optimizer.state)))
        for first in range(0, len(bank.training), batch):
            optimizer.zero_grad(set_to_none=True)
            loss = (model.weight-torch.rand_like(model.weight)).square().mean()
            loss.backward(); optimizer.step()
        return float(loss.detach()), math.ceil(len(bank.training)/batch)
    monkeypatch.setattr(common, "_branch_epoch", epoch)
    full, split = tmp_path/"full", tmp_path/"split"
    native.run_branch(path, gp, full, config=config, device="cpu")
    native.run_branch(path, gp, split, config=config, device="cpu", stop_after_epoch=boundary)
    result = native.run_branch(path, gp, split, config=config, device="cpu", resume=True)
    a, b = (torch.load(folder/"latest.pt", weights_only=True)["payload"] for folder in (full, split))
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert a["scheduler"] is b["scheduler"] is None
    assert b["best"] is None and b["optimizer_updates"] == 10
    assert result["status"] == "complete" and result["val_metric"] is None
    assert result["training_cost"]["total_optimizer_updates"] == 18
    exported = torch.load(split/"model.pt", weights_only=True)
    assert exported["epoch"] == 5 and exported["selection"] == "terminal_epoch"
    assert exported["seed"] == 20260820 and exported["stage"] == "formal"
    assert [row["global_trajectory_epoch"] for row in b["history"]] == [5, 6, 7, 8, 9]
    assert not (split/"best.pt").exists()
    # At each fresh phase, Adam moments reset, including the phase resumed
    # immediately after the previous phase's committed terminal epoch.
    for phase, epoch_key, rate, state_count in used:
        if epoch_key in (4, 6, 7, 8):
            assert state_count == 0


def test_boundary_rejects_old_whole_stage_cosine_and_wrong_updates():
    """A source-compatible old W1-v2 clock is scientifically incompatible here."""
    config = fixture_config()
    p = torch.nn.Linear(1, 1)
    optimizer, scheduler = native._optimizer(p.parameters(), "generator", 0, config, 4)
    saved = native._boundary(p, optimizer, scheduler, 0, 0, [], False)
    native._validate_boundary(saved, "generator", config, 4)
    saved["scheduler"]["T_max"] = 8
    with pytest.raises(ValueError, match="phase-local cosine"):
        native._validate_boundary(saved, "generator", config, 4)


def test_monitor_gate_rejection_preserves_rng_and_does_not_weaken_training_gate():
    """Validation rejection is diagnostic; unrelated infrastructure errors propagate."""
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    def gate():
        torch.rand(7)
        raise RuntimeError("recursive local correction gate failed during training")
    value = native._monitor(gate, 20)
    assert value["status"] == "failed_native_validation_gate" and value["population"] == 20
    assert value["individual_failure_ids_known"] is False
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(RuntimeError, match="different infrastructure"):
        native._monitor(lambda: (_ for _ in ()).throw(RuntimeError("different infrastructure")), 20)


def test_real_nonfinite_gradient_clipping_is_scientific_and_finite_clip_unchanged():
    """A real nonfinite gradient is rejected before Adam rather than retried as infrastructure."""
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    p.grad = torch.tensor([float("inf"), 1.])
    with pytest.raises(FloatingPointError, match="non-finite"):
        native._clip_generator_gradients([p])
    assert torch.equal(p, torch.tensor([1., 2.]))
    p.grad = torch.tensor([10., 20.])
    q = torch.nn.Parameter(p.detach().clone())
    q.grad = p.grad.clone()
    native._clip_generator_gradients([p])
    torch.nn.utils.clip_grad_norm_([q], 5., error_if_nonfinite=True)
    assert torch.equal(p.grad, q.grad)
    assert not native._nonfinite_gradient_error(RuntimeError("CUDA out of memory"))


def test_export_is_existing_evaluator_compatible_on76_grid(tmp_path):
    """Load real native model states through the untouched paired inference loader."""
    torch.set_num_threads(1)
    config = native.NativeGIFTConfig()
    generator = common.GIFTGenerator(21, 8, 1., False)
    bank = SimpleNamespace(state_scale=1., binding={"query_times": (np.arange(76)*.04).tolist(), "points": 76, "future_steps": 75})
    gp = tmp_path/"generator.pt"
    torch.save(native._export_generator(generator, config, bank, 500, {"val_metric": None}, {}), gp)
    low = common.FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    branch = common.HighFrequencyBranch(common.HighFrequencyConfig(), state_scale=1., high_scale=1.,
                  low_rhs_scale=1., high_rhs_scale=1., frozen_generator=low)
    payload = common._branch_export(branch, common.clone_state(branch), gp, common.state_digest(generator.state_dict()),
                                    config, config.branch_seed, bank, 500, None, {})
    payload.update(phase="terminal_epoch", selection="terminal_epoch", profile=native.PROFILE)
    bp = tmp_path/"branch.pt"
    torch.save(payload, bp)
    loaded_low, loaded_branch, loaded = common.load_gift_checkpoint(bp)
    assert loaded["model_spec"]["points"] == 76
    assert loaded["model_spec"]["save_interval"] == .04 and loaded["model_spec"]["step"] == .02
    assert loaded["model_spec"]["rollout_integration_steps"] == [0, 4, 10, 20, 24, 30, 40, 50]
    assert common.state_digest(loaded_low.model.state_dict()) == common.state_digest(generator.state_dict())
    assert common.state_digest(loaded_branch.state_dict()) == common.state_digest(branch.state_dict())


def test_cli_preserves_typed_numerical_failure(tmp_path, monkeypatch):
    """The runner receives a nonzero exception and a scientific failure receipt."""
    job = tmp_path/"job.json"
    output = tmp_path/"out"
    job.write_text(json.dumps({"method": "GIFT", "window": str(tmp_path/"unused.h5"),
                               "output": str(output), "device": "cpu", "profile": native.PROFILE}))
    def failure(*args, **kwargs):
        raise FloatingPointError("training gate rejection")
    monkeypatch.setattr(native, "run_pipeline", failure)
    with pytest.raises(FloatingPointError):
        native.main(["--run-config", str(job)])
    receipt = json.loads((output/"FAILURE.json").read_text())
    assert receipt["failure_kind"] == "numerical" and receipt["seed"] == 20260820
    assert receipt["generator_seed"] == 2026072301 and receipt["stage"] == "formal"
