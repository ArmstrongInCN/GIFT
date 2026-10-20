"""CPU checks of window isolation, native losses, selection and multi-phase resume.

Small fixtures test controller mechanics without claiming model performance;
the full-grid runtime check still uses the real failure-preserving GIFT rollout.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import training.w1_v2_gift as training


def make_slice(path, *, grid=8):
    """Write four train/two validation151-frame polynomial fields, with no outside frames."""
    rng = np.random.default_rng(1729)
    with h5py.File(path, "x") as h:
        h.attrs.update(schema=training.SCHEMA, status="complete", trajectory_id_scheme="canonical", fixture_only=True)
        for name, ids in (("training", np.arange(4)), ("validation", np.arange(1000, 1002))):
            g = h.create_group(name)
            t = np.arange(151)*.02
            g.create_dataset("time", data=t)
            g.create_dataset("absolute_time", data=t+5)
            g.create_dataset("trajectory_index", data=ids)
            a = rng.normal(size=(len(ids), 1, grid, grid)).astype(np.float32)
            b = rng.normal(size=a.shape).astype(np.float32)
            w = a + t[None, :, None, None]*b
            g.create_dataset("vorticity", data=w.astype(np.float32))


def fixture_config(**kw):
    """Use positive tiny phase counts and a small generator, retaining controller semantics."""
    return training.GIFTTrainingConfig(generator_phases=(2, 2), branch_phases=(1, 1, 1, 1),
                                       effective_batch=3, generator_micro_batch=2, branch_micro_batch=2,
                                       cutoff=2, rank=2, fixture_only=True, validation_interval=1,
                                       early_stopping=False, **kw)


def fake_validation(low, branch, bank, device, config):
    """Deterministic state-derived validation stand-in for tiny-grid training recovery checks."""
    model = low.model if branch is None else branch
    value = float(sum(v.detach().abs().double().square().sum() for v in model.parameters()))
    return {"val_metric": value, "seconds": 0.0, "failed_trajectory_ids": []}


def test_sampling_order_and_boundary_stencils(tmp_path):
    """Keep the common order independent of sampling/model seeds and every stencil inside5..8."""
    path = tmp_path / "slice.h5"
    make_slice(path)
    config = fixture_config()
    bank = training.PredictionObservationBank(path, config)
    changed = training.PredictionObservationBank(path, replace(config, sampling_seed=998))
    for epoch in (0, 1, 50, 500, 1500):
        reference = np.random.default_rng(np.random.SeedSequence([config.order_seed, epoch])).permutation(4)
        for rollout in (False, True):
            order, positions = bank.locations(epoch, rollout=rollout)
            assert np.array_equal(order, reference)
            assert np.array_equal(changed.locations(epoch, rollout=rollout)[0], reference)
            assert positions.min() >= (0 if rollout else 2)
            assert positions.max() <= (100 if rollout else 148)
    order, positions = np.array([3, 1, 0, 2]), np.array([2, 148, 2, 148])
    state, rhs = training.observed_pairs(bank.training, order, positions)
    assert torch.equal(state[0], torch.from_numpy(bank.training[3, 148]))
    expected = (bank.training[3, 146]-8*bank.training[3, 147]
                +8*bank.training[3, 149]-bank.training[3, 150])/.24
    assert np.array_equal(rhs[0].numpy(), expected)
    sequence = training.observed_sequences(bank.training, order, np.full(4, 100))
    assert np.array_equal(sequence[0, -1], bank.training[3, 150])
    assert bank.binding["read_extra_boundary_frames"] is False


def test_rejects_full_trajectory_or_shifted_absolute_time(tmp_path):
    """An old or mislabelled time window must fail before it can supply extra observations."""
    path = tmp_path / "slice.h5"
    make_slice(path)
    with h5py.File(path, "r+") as h:
        h["training/absolute_time"][0] = 4.98
    with pytest.raises(ValueError, match="exactly t5..8"):
        training.PredictionObservationBank(path, fixture_config())


def test_cosine_phase_change_has_no_time_restart_and_exact_resume():
    """Compare native scheduler rates with a single global cosine across both phase boundaries."""
    config = fixture_config()
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer, scheduler, rates = training._optimizer([p], "generator", config)
    states = []
    for epoch in range(1, 5):
        if epoch == 3:
            training._change_phase(optimizer, scheduler, rates[1])
            assert len(optimizer.state) == 0
            assert scheduler.last_epoch == 2
        phase = 0 if epoch <= 2 else 1
        expected = rates[phase]*(.01+.99*(1+np.cos(np.pi*(epoch-1)/4))/2)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-12)
        optimizer.zero_grad(); p.square().sum().backward(); optimizer.step(); scheduler.step()
        states.append((p.detach().clone(), optimizer.state_dict(), scheduler.state_dict()))
    q = torch.nn.Parameter(states[1][0].clone())
    resumed, resumed_scheduler, _ = training._optimizer([q], "generator", config)
    training.restore_optimizer(resumed, states[1][1]); resumed_scheduler.load_state_dict(states[1][2])
    training._change_phase(resumed, resumed_scheduler, rates[1])
    for _ in range(2):
        resumed.zero_grad(); q.square().sum().backward(); resumed.step(); resumed_scheduler.step()
    assert torch.equal(p, q)
    assert scheduler.state_dict() == resumed_scheduler.state_dict()


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_real_generator_resume_inside_and_across_phases(tmp_path, monkeypatch, boundary):
    """Native factors/affine refits/AdamW match through inside-phase and phase-boundary pauses."""
    torch.set_num_threads(1)
    monkeypatch.setattr(training, "_validation", fake_validation)
    path = tmp_path / "slice.h5"
    make_slice(path)
    full, split = tmp_path / "full", tmp_path / "split"
    config = fixture_config()
    training.run_generator(path, full, seed=7, config=config, device="cpu", checkpoint_interval=1)
    result = training.run_generator(path, split, seed=7, config=config, device="cpu",
                                    checkpoint_interval=1, stop_after_epoch=boundary)
    assert result["status"] == "paused_at_committed_epoch"
    training.run_generator(path, split, seed=7, config=config, device="cpu", checkpoint_interval=1, resume=True)
    a, b = (torch.load(p/"latest.pt", weights_only=True)["payload"] for p in (full, split))
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
        assert torch.equal(a["best"]["model"][key], b["best"]["model"][key])
    assert a["best"]["epoch"] >= 3
    assert a["scheduler"] == b["scheduler"]
    assert a["optimizer_updates"] == b["optimizer_updates"] == 8
    for aa, bb in zip(a["history"], b["history"]):
        assert {k:v for k,v in aa.items() if k!="seconds"} == {k:v for k,v in bb.items() if k!="seconds"}
    exported = torch.load(split/"model.pt", weights_only=True)
    assert exported["training_cost"]["affine_fits"] == 5
    assert training.state_digest(training.load_generator(exported).state_dict()) == training.state_digest(exported["model_state"])
    before = (split/"model.pt").read_bytes()
    assert training.run_generator(path, split, seed=7, config=config, device="cpu", resume=True)["status"] == "complete"
    assert before == (split/"model.pt").read_bytes()


def test_selection_excludes_coarse_generator_and_keeps_earliest_tie():
    """Validation selection must be finite/eligible, with no score threshold or failure exclusion."""
    model = torch.nn.Linear(1, 1)
    assert training._select(None, {"val_metric": 0.0}, 1, model, eligible=False) is None
    assert training._select(None, {"val_metric": None}, 2, model) is None
    best = training._select(None, {"val_metric": 3.0}, 3, model)
    assert training._select(best, {"val_metric": 3.0}, 4, model) is best
    assert training._select(best, {"val_metric": 2.0}, 5, model)["epoch"] == 5


def test_native_initial_only_validation_and_failure_population(monkeypatch):
    """Real zero-RHS GIFT rollout preserves151 fields; any failed path makes the whole score unavailable."""
    torch.set_num_threads(1)
    initial = np.full((2, 64, 64), .25, dtype=np.float32)
    bank = SimpleNamespace(validation=np.repeat(initial[:, None], 151, axis=1),
                           binding={"validation_ids": [1000, 1001]})
    low = lambda x: torch.zeros_like(x)
    measured = training._validation(low, None, bank, torch.device("cpu"), fixture_config())
    assert measured["val_metric"] < 1e-6
    assert measured["failed_trajectory_count"] == 0
    result = SimpleNamespace(prediction=bank.validation.copy(), failure_step=np.array([-1, 4]))
    result.prediction[1, 4:] = np.nan
    monkeypatch.setattr(training, "rollout_gift", lambda **kw: result)
    measured = training._validation(low, None, bank, torch.device("cpu"), fixture_config())
    assert measured["val_metric"] is None
    assert measured["failed_trajectory_ids"] == [1001]
    assert len(measured["relative_l2"]) == 2
    assert all(v is None for v in measured["relative_l2"][1])


class SmallBranch(torch.nn.Module):
    """Tiny CPU model for controller RNG/phase tests, without replacing production branch code."""
    def __init__(self, config, *, state_scale, high_scale, low_rhs_scale, high_rhs_scale, frozen_generator):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3))
        self.config = config
        for name, value in dict(state_scale=state_scale, high_scale=high_scale,
                                low_rhs_scale=low_rhs_scale, high_rhs_scale=high_rhs_scale).items():
            self.register_buffer(name, torch.tensor(value))
    def export_config(self):
        return {}


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_branch_phase_optimizer_scheduler_rng_resume(tmp_path, monkeypatch, boundary):
    """A stochastic native-AdamW stand-in checks every branch phase switch and rolling state restoration."""
    torch.set_num_threads(1)
    path = tmp_path / "slice.h5"
    make_slice(path, grid=64)
    config = replace(fixture_config(), cutoff=21, rank=8)
    generator = training.GIFTGenerator(21, 8, 1.0, False)
    gp = tmp_path / "generator.pt"
    binding = training.PredictionObservationBank(path, config).binding
    torch.save({"format_version": 3, "model_configuration": generator.configuration(),
                "model_state_dict": training.clone_state(generator), "val_metric": 1.0,
                "training_regime": "w1_v2_common_t5_t8",
                "training_data": binding,
                "training_cost": {"trajectory_epochs": 4, "optimizer_updates": 8, "committed_seconds": 0.0}}, gp)
    monkeypatch.setattr(training, "HighFrequencyBranch", SmallBranch)
    monkeypatch.setattr(training, "_training_scales", lambda *v: (1.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(training, "_compute_low_rhs", lambda frozen, states, device: np.ones_like(states))
    monkeypatch.setattr(training, "_validation", fake_validation)
    def epoch(model, frozen, bank, optimizer, config, seed, global_epoch, phase, device, scales):
        optimizer.zero_grad(set_to_none=True)
        target = torch.rand_like(model.weight)
        loss = (model.weight-target).square().mean()
        loss.backward(); optimizer.step()
        return float(loss.detach()), 1
    monkeypatch.setattr(training, "_branch_epoch", epoch)
    full, split = tmp_path / "full", tmp_path / "split"
    training.run_branch(path, gp, full, seed=11, config=config, device="cpu", checkpoint_interval=1)
    training.run_branch(path, gp, split, seed=11, config=config, device="cpu", checkpoint_interval=1, stop_after_epoch=boundary)
    training.run_branch(path, gp, split, seed=11, config=config, device="cpu", checkpoint_interval=1, resume=True)
    a, b = (torch.load(p/"latest.pt", weights_only=True)["payload"] for p in (full, split))
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["scheduler"] == b["scheduler"]
    assert a["best"]["epoch"] == b["best"]["epoch"]
    assert a["best"]["val_metric"] == b["best"]["val_metric"]
    assert [r["global_trajectory_epoch"] for r in a["history"]] == [5, 6, 7, 8]


@pytest.mark.parametrize("observation_dt", [.02, .04, .1])
def test_native_pair_export_load_without_future_access(tmp_path, observation_dt):
    """The new model specification must strictly load through both original inference loaders."""
    torch.set_num_threads(1)
    generator = training.GIFTGenerator(21, 8, 1.0, False)
    gp = tmp_path / "generator.pt"
    state = training.clone_state(generator)
    torch.save({"format_version": 3, "model_configuration": generator.configuration(), "model_state_dict": state}, gp)
    low = training.FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    branch = training.HighFrequencyBranch(training.HighFrequencyConfig(), state_scale=1.0,
                                          high_scale=1.0, low_rhs_scale=1.0, high_rhs_scale=1.0, frozen_generator=low)
    times = np.arange(round(3/observation_dt)+1)*observation_dt
    bank = SimpleNamespace(binding={"schema": training.SCHEMA if observation_dt == .02 else "gift.w1-sparse.window.v3",
                                    "points": len(times), "future_steps": len(times)-1, "query_times": times.tolist()})
    payload = training._branch_export(branch, training.clone_state(branch), gp, training.state_digest(state),
                                      training.GIFTTrainingConfig(), 11, bank, 1000, .5, {})
    bp = tmp_path / "branch.pt"
    torch.save(payload, bp)
    loaded_low, loaded_branch, loaded = training.load_gift_checkpoint(bp)
    assert loaded["model_spec"]["history"] == 1
    assert loaded["model_spec"]["save_interval"] == observation_dt
    assert loaded["model_spec"]["step"] == .02
    assert loaded["model_spec"]["points"] == len(times)
    assert loaded["data_binding"]["query_times"] == times.tolist()
    assert training.state_digest(loaded_low.model.state_dict()) == training.state_digest(state)
    assert training.state_digest(loaded_branch.state_dict()) == training.state_digest(branch.state_dict())


def test_cli_failure_record_and_nonzero_exception(tmp_path, monkeypatch):
    """A numerical failure must be retained for the orchestrator and propagated, never treated as success."""
    def failure(*args, **kwargs):
        raise FloatingPointError("declared fixture numerical failure")
    monkeypatch.setattr(training, "run_pipeline", failure)
    output = tmp_path / "failure"
    with pytest.raises(FloatingPointError):
        training.main(["--run-training", "--dataset", str(tmp_path/"unused.h5"),
                       "--output", str(output), "--seed", "7", "--profile", "development", "--device", "cpu"])
    result = json.loads((output/"FAILURE.json").read_text())
    assert result["failure_kind"] == "numerical" and result["stage"] == "development"
    assert result["seed"] == 7 and result["status"] == "failed"


def test_derivative_effective_group_matches_microbatch_and_tail(monkeypatch):
    """Native derivative MSE weighted across microbatches must preserve both full and tail update groups."""
    class DerivativeBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.2))
        def forward_with_generator_output(self, state, low):
            return self.weight*state
    rng = np.random.default_rng(18)
    states = torch.from_numpy(rng.normal(size=(5, 64, 64)).astype(np.float32))
    targets = torch.from_numpy(rng.normal(size=(5, 64, 64)).astype(np.float32))
    bank = SimpleNamespace(pairs=lambda seed, epoch: (states, targets))
    monkeypatch.setattr(training, "_compute_low_rhs", lambda frozen, raw, device: np.zeros_like(raw))
    config = replace(fixture_config(), cutoff=21)
    results = []
    for microbatch in (1, 3):
        model = DerivativeBranch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=1e-6)
        loss, updates = training._branch_epoch(model, None, bank, optimizer,
                        replace(config, branch_micro_batch=microbatch), 7, 0, 0, torch.device("cpu"), (1.,1.,1.,1.))
        results.append((model.weight.detach(), loss, updates))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-7, rtol=0)
    assert results[0][1] == pytest.approx(results[1][1], rel=1e-6)
    assert results[0][2] == results[1][2] == 2


def test_sparse_stencil_units_and_exact_physical_rollout_leads():
    """The .1-second stencil differentiates quartics and preserves native .1..1s target leads."""
    times = np.arange(31)*.1
    raw = np.stack([1+times+times**4, 3+2*times+times**4]).astype(np.float32)[:, :, None, None]
    order, centres = np.array([1, 0]), np.array([2, 28])
    state, rhs = training._grid_pairs(raw, order, centres, .1)
    used = centres[order]
    exact = np.array([2., 1.])+4*times[used]**3
    assert np.allclose(rhs[:, 0, 0], exact, atol=2e-4, rtol=0)
    assert np.array_equal(state.numpy(), raw[order, used])
    offsets = training._observation_offsets(.1)
    assert np.array_equal(offsets, [0, 1, 2, 4, 5, 6, 8, 10])
    assert np.allclose(offsets*.1, training.ROLLOUT_OFFSETS*.02, atol=2e-12, rtol=0)
    sequence = training._grid_sequences(raw, order, np.array([0, 20]), offsets)
    assert np.array_equal(sequence[0], raw[1, 20+offsets])
    assert sequence[0, -1, 0, 0] == raw[1, 30, 0, 0]
    with pytest.raises(ValueError, match="supports"):
        training._observation_offsets(.06)


def test_sparse76_declared_nearest_leads_tie_earlier_and_stencil_units():
    """The predeclared .04 grid moves only .1/.5s to .08/.48s, without extra dense observations."""
    steps = training._rollout_steps(.04)
    offsets = training._observation_offsets(.04)
    assert np.array_equal(steps, [0, 4, 10, 20, 24, 30, 40, 50])
    assert np.array_equal(offsets, [0, 2, 5, 10, 12, 15, 20, 25])
    assert np.allclose(steps*.02, offsets*.04, atol=2e-12, rtol=0)
    requested = training.ROLLOUT_OFFSETS*.02
    actual = steps*.02
    assert np.allclose(actual-requested, [0, -.02, 0, 0, -.02, 0, 0, 0], atol=2e-12, rtol=0)
    times = np.arange(76)*.04
    raw = (1+2*times+times**4).astype(np.float32)[None, :, None, None]
    for centre in (2, 73):
        _, rhs = training._grid_pairs(raw, np.array([0]), np.array([centre]), .04)
        assert float(rhs[0, 0, 0]) == pytest.approx(2+4*times[centre]**3, abs=7e-4)
    sequence = training._grid_sequences(raw, np.array([0]), np.array([50]), offsets)
    assert np.array_equal(sequence[0, -1], raw[0, 75])


def make_sparse_slice(path, *, validation_multiplier=1., time_stride=5):
    """Write a shared-reader-compatible small sparse64-grid fixture with exact decoded-data receipts."""
    from training.w1_v2_data import window_manifest, SPARSE_SCHEMA
    manifest = window_manifest(training_count=4, time_stride=time_stride)
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    rng = np.random.default_rng(1729)
    times = np.asarray(manifest["query_times"])
    with h5py.File(path, "x") as h:
        h.attrs.update(schema=SPARSE_SCHEMA, status="complete", trajectory_id_scheme="canonical", fixture_only=True,
                       source_file="fixture", manifest_json=text,
                       manifest_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        for name, ids in (("training", manifest["training_ids"]), ("validation", [1000, 1001])):
            group = h.create_group(name)
            group["time"], group["absolute_time"], group["trajectory_index"] = times, 5+times, ids
            slope = rng.uniform(.1, .2, size=(len(ids), 1, 64, 64)).astype(np.float32)
            fields = (1+times[None, :, None, None]*slope).astype(np.float32)
            if name == "validation":
                fields *= validation_multiplier
            group["vorticity"] = fields
            group.attrs["observations_sha256"] = hashlib.sha256(fields.tobytes()).hexdigest()


@pytest.mark.parametrize("time_stride", [2, 5])
def test_shared_sparse_bank_sampling_receipts_and_training_only_scale(tmp_path, time_stride):
    """Real shared-bank loading preserves seeded row order, legal31/76-point windows and training-only RMS."""
    from training.w1_v2_data import window_manifest
    path, changed = tmp_path/"sparse.h5", tmp_path/"large_val.h5"
    make_sparse_slice(path, time_stride=time_stride)
    make_sparse_slice(changed, validation_multiplier=100., time_stride=time_stride)
    config = fixture_config()
    bank, other = training.PredictionObservationBank(path, config), training.PredictionObservationBank(changed, config)
    manifest = window_manifest(4, time_stride)
    points, dt = manifest["points"], .02*time_stride
    offsets = training._observation_offsets(dt)
    assert bank.training.shape == (4, points, 64, 64)
    assert bank.binding["training_ids"] == manifest["training_ids"]
    assert bank.binding["derivative_centres"] == [2, points-3]
    assert bank.binding["rollout_anchors"] == [0, points-1-int(offsets[-1])]
    assert bank.binding["rollout_observation_offsets"] == offsets.tolist()
    assert bank.binding["rollout_integration_steps"] == training._rollout_steps(dt).tolist()
    assert bank.state_scale == other.state_scale
    assert bank.binding["observations_sha256"] != other.binding["observations_sha256"]
    assert bank.state_scale == float(np.sqrt(np.square(bank.training, dtype=np.float64).mean()))
    for epoch in (0, 500, 1500):
        expected = np.random.default_rng(np.random.SeedSequence([config.order_seed, epoch])).permutation(4)
        for rollout in (False, True):
            order, positions = bank.locations(epoch, rollout=rollout)
            assert np.array_equal(order, expected)
            assert positions.min() >= (0 if rollout else 2)
            assert positions.max() <= (points-1-int(offsets[-1]) if rollout else points-3)
    _, rhs = bank.pairs(7, 0)
    # The same linear field gives the same derivative at every legal centre.
    order, positions = bank.locations(0)
    expected_slope = (bank.training[order, 1]-bank.training[order, 0])/dt
    assert np.allclose(rhs.numpy(), expected_slope, atol=6e-6, rtol=0)
    assert bank.sequences(7, 0).shape == (4, 8, 64, 64)
    assert bank.binding["read_extra_boundary_frames"] is False
    with h5py.File(path, "r+") as h:
        h["training/vorticity"][0, 0, 0, 0] += .25
    with pytest.raises(ValueError, match="copy receipt"):
        training.PredictionObservationBank(path, config)


@pytest.mark.parametrize("observation_dt", [.04, .1])
def test_sparse_validation_saves_declared_grid_but_integrates150_native_steps(observation_dt):
    """A real zero-RHS rollout makes600 native RHS calls while saving31/76 frames and scoring30/75 futures."""
    torch.set_num_threads(1)
    initial = np.full((1, 64, 64), .25, dtype=np.float32)
    points = round(3/observation_dt)+1
    bank = SimpleNamespace(validation=np.repeat(initial[:, None], points, axis=1),
                           binding={"validation_ids": [1000], "query_times": (np.arange(points)*observation_dt).tolist()})
    calls = []
    def low(state):
        calls.append(tuple(state.shape))
        return torch.zeros_like(state)
    measured = training._validation(low, None, bank, torch.device("cpu"), fixture_config())
    assert len(calls) == 600
    assert measured["metric"] == f"future{points-1}_mean_per_frame_relative_l2"
    assert len(measured["relative_l2"][0]) == points
    assert measured["val_metric"] < 1e-6
    assert measured[f"future{points-1}_physical_mse"] < 1e-12


@pytest.mark.parametrize("phase,steps", [(1, 10), (2, 50), (3, 50)])
@pytest.mark.parametrize("observation_dt", [.04, .1])
def test_sparse_branch_objectives_keep_native_integrator_steps(monkeypatch, phase, steps, observation_dt):
    """Sparse observed frames supply the eight declared leads without shortening native rollout/truncation."""
    class LinearBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.register_buffer("high_scale", torch.tensor(1.))
        def project_q21(self, value):
            return value
    calls = []
    def rk4(frozen, model, low, high):
        calls.append(1)
        return low, high+model.weight*.02
    monkeypatch.setattr(training, "_dual_rk4_step", rk4)
    monkeypatch.setattr(training, "_correct_low", lambda value: value)
    monkeypatch.setattr(training, "_initialize_tracks", lambda model, initial: (initial*0, initial+model.weight*0))
    points = round(3/observation_dt)+1
    raw = np.broadcast_to((np.arange(points)*observation_dt).astype(np.float32)[None, :, None, None], (1, points, 2, 2)).copy()
    offsets = training._observation_offsets(observation_dt)
    bank = SimpleNamespace(sequences=lambda seed, epoch: training._grid_sequences(raw, np.array([0]), np.array([0]), offsets),
                           integration_steps=training._rollout_steps(observation_dt))
    model = LinearBranch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
    loss, updates = training._branch_epoch(model, None, bank, optimizer, fixture_config(), 7, 0,
                                          phase, torch.device("cpu"), (1., 1., 1., 1.))
    assert len(calls) == steps
    assert loss < 1e-12
    assert updates == 1


@pytest.mark.parametrize("observation_dt", [.04, .1])
def test_sparse_checkpoint_grid_binding_and_initial_only_output(monkeypatch, observation_dt):
    """Inference uses the recorded coarse save grid and fixed .02 integrator, rejecting mismatched declarations."""
    points = round(3/observation_dt)+1
    spec = {"family": "gift", "history": 1, "duration": 3., "save_interval": observation_dt, "step": .02,
            "points": points, "future_steps": points-1, "query_times": (np.arange(points)*observation_dt).tolist(),
            "observation_dt": observation_dt, "integration_dt": .02,
            "rollout_integration_steps": training._rollout_steps(observation_dt).tolist(),
            "rollout_observation_offsets": training._observation_offsets(observation_dt).tolist()}
    payload = {"model_spec": spec, "data_binding": dict(spec)}
    assert len(training._validate_checkpoint_grid(payload)) == points
    bad = {**payload, "data_binding": {**spec, "query_times": (np.arange(points)*.02).tolist()}}
    with pytest.raises(ValueError, match="query-time"):
        training._validate_checkpoint_grid(bad)
    bad = {**payload, "data_binding": {**spec, "rollout_integration_steps": [0, 6, 10, 20, 26, 30, 40, 50]}}
    with pytest.raises(ValueError, match="supervision step"):
        training._validate_checkpoint_grid(bad)
    bad = {**payload, "data_binding": {**spec, "rollout_target_policy": "unrecorded_rounding"}}
    with pytest.raises(ValueError, match="mapping policy"):
        training._validate_checkpoint_grid(bad)
    captured = {}
    monkeypatch.setattr(training, "load_gift_checkpoint", lambda *a: (None, None, payload))
    def rollout(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(prediction=np.repeat(kwargs["initial_state"][:, None], points, axis=1))
    monkeypatch.setattr(training, "rollout_gift", rollout)
    initial = np.ones((2, 64, 64), dtype=np.float32)
    result = training.predict_gift_checkpoint("unused.pt", initial, [1000, 1001], device="cpu")
    assert captured["step"] == .02 and captured["save_interval"] == observation_dt
    assert result.prediction.shape == (2, points, 64, 64)
    assert "truth" not in captured and "future" not in captured


@pytest.mark.parametrize("reason,error_type", [
    ("recursive local correction gate failed during training", FloatingPointError),
    ("CUDA out of memory", RuntimeError),
    ("recursive local correction gate failed during training: unexpected infrastructure detail", RuntimeError),
])
def test_native_gate_failure_is_numerical_but_other_runtime_failures_keep_type(monkeypatch, reason, error_type):
    """Only the native gate's exact failure becomes nonretryable numerical status; no gate or error is suppressed."""
    model = torch.nn.Linear(1, 1)
    model.project_q21 = lambda value: value
    bank = SimpleNamespace(sequences=lambda seed, epoch: np.ones((1, 8, 2, 2), dtype=np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    monkeypatch.setattr(training, "_initialize_tracks", lambda model, initial: (initial, initial))
    monkeypatch.setattr(training, "_dual_rk4_step", lambda *args: args[-2:])
    def failed_gate(value):
        raise RuntimeError(reason)
    monkeypatch.setattr(training, "_correct_low", failed_gate)
    with pytest.raises(error_type, match=reason) as caught:
        training._branch_epoch(model, None, bank, optimizer, fixture_config(), 7, 0, 2,
                               torch.device("cpu"), (1., 1., 1., 1.))
    if error_type is FloatingPointError:
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert str(caught.value.__cause__) == reason


def test_real_native_gate_failure_reaches_cli_numerical_record(tmp_path, monkeypatch):
    """An actual native gate rejection reaches the training CLI as numerical failure, retaining its cause."""
    import experiments.formal.train_gift_branches as native
    model = torch.nn.Linear(1, 1)
    model.project_q21 = lambda value: value
    bank = SimpleNamespace(sequences=lambda seed, epoch: np.ones((1, 8, 2, 2), dtype=np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    monkeypatch.setattr(training, "_initialize_tracks", lambda model, initial: (initial, initial))
    monkeypatch.setattr(training, "_dual_rk4_step", lambda *args: args[-2:])
    monkeypatch.setattr(native, "apply_local_correction", lambda state, policy:
                        SimpleNamespace(state=state, gate_failed=torch.tensor([True])))
    def pipeline(*args, **kwargs):
        return training._branch_epoch(model, None, bank, optimizer, fixture_config(), 7, 0, 2,
                                      torch.device("cpu"), (1., 1., 1., 1.))
    monkeypatch.setattr(training, "run_pipeline", pipeline)
    output = tmp_path/"failed_seed"
    with pytest.raises(FloatingPointError, match="correction gate failed") as caught:
        training.main(["--run-training", "--dataset", str(tmp_path/"unused.h5"), "--output", str(output),
                       "--seed", "7", "--profile", "development", "--device", "cpu"])
    assert isinstance(caught.value.__cause__, RuntimeError)
    failure = json.loads((output/"FAILURE.json").read_text())
    assert failure["failure_kind"] == "numerical" and failure["status"] == "failed"
    assert failure["seed"] == 7 and failure["stage"] == "development"


@pytest.mark.parametrize("measurement_rows,setup_rows", [(2, None), (2, 4)])
@pytest.mark.parametrize("time_stride", [2, 5])
def test_cost_preflight_setup_population_and_real_common_tail(tmp_path, monkeypatch, measurement_rows, setup_rows, time_stride):
    """CPU stand-ins verify full-setup fit/scales with an unchanged default or the exact epoch0 tail, without GPU training."""
    torch.set_num_threads(1)
    path = tmp_path/"sparse.h5"
    make_sparse_slice(path, time_stride=time_stride)
    affine_counts, scale_counts, constructor_scales, measured_states = [], [], [], []
    class ToyGenerator(torch.nn.Module):
        def __init__(self, cutoff, rank, state_scale, flag):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.1))
            constructor_scales.append(state_scale)
        def forward(self, state):
            return self.weight*state
    monkeypatch.setattr(training, "GIFTGenerator", ToyGenerator)
    monkeypatch.setattr(training, "configure_variable_projection_parameters", lambda model: list(model.parameters()))
    monkeypatch.setattr(training, "fit_affine_minimum_norm", lambda model, states, targets, **kw: affine_counts.append(len(states)))
    monkeypatch.setattr(training, "FixedBandwidthGridGenerator", lambda model, *a, **kw: model)
    monkeypatch.setattr(training, "_compute_low_rhs", lambda frozen, states, device: np.ones_like(states))
    def scales(states, targets, low):
        scale_counts.append(len(states))
        return (1., 1., 1., 1.)
    monkeypatch.setattr(training, "_training_scales", scales)
    monkeypatch.setattr(training, "HighFrequencyBranch", SmallBranch)
    def epoch(model, frozen, bank, optimizer, config, seed, global_epoch, phase, device, scales):
        states, targets = bank.pairs(seed, global_epoch)
        measured_states.append(states.numpy().copy())
        assert np.array_equal(bank.integration_steps, training._rollout_steps(.02*time_stride))
        assert len(bank.sequences(seed, global_epoch)) == measurement_rows
        return 0., 1
    monkeypatch.setattr(training, "_branch_epoch", epoch)
    result = training.preflight(path, device="cpu", rows=measurement_rows, setup_rows=setup_rows)
    setup_count = measurement_rows if setup_rows is None else setup_rows
    assert affine_counts == [setup_count, setup_count]
    assert scale_counts == [setup_count]
    assert result["generator"]["affine_rows"] == setup_count
    assert result["generator"]["full_training_affine_time_measured"] == (setup_count == 4)
    assert result["setup_rows"] == setup_count and result["measurement_rows"] == measurement_rows
    assert result["validation_read"] is False and result["test_read"] is False
    with h5py.File(path, "r") as handle:
        raw = handle["training/vorticity"][:setup_count]
        ids = handle["training/trajectory_index"][:setup_count]
    order = np.random.default_rng(np.random.SeedSequence([2026100301, 0])).permutation(setup_count)
    centres = np.random.default_rng(np.random.SeedSequence([2026100303, 0, 0, 0])).integers(2, raw.shape[1]-2, setup_count)
    expected, _ = training._grid_pairs(raw, order, centres, .02*time_stride)
    expected = expected[-measurement_rows:].numpy()
    assert all(np.array_equal(value, expected) for value in measured_states)
    assert result["measurement_training_ids"] == ids[order[-measurement_rows:]].tolist()
    assert constructor_scales[0] == float(np.sqrt(np.square(raw, dtype=np.float64).mean()))
    assert len(result["branch"]) == 4
    assert all(row["rows"] == measurement_rows and row["updates"] == 1 for row in result["branch"])
