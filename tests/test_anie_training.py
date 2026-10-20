"""ANIE observation, budget and native continuation contracts; no formal training."""
import copy
import random

import h5py
import numpy as np
import pytest
import torch

from training.anie_data import ObservationBank, TRAIN_INDICES, epoch_indices
from training.checkpoints import CheckpointStore
from training.train_anie import (arguments, configuration, configure_random,
                                 native_training_objects, objective, step_scheduler,
                                 train_epoch, validation_measurement)


def fixture_observations(tmp_path, count=3):
    """Create row/time-identifiable 151-frame observations and an intentionally invalid test group."""
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["fixture_only"] = True
        for name, ids in (("training", np.arange(count)), ("validation", np.arange(1000, 1002))):
            group = handle.create_group(name)
            group["trajectory_index"] = ids
            group["time"] = np.arange(151) * .02
            values = np.empty((len(ids), 151, 64, 64), dtype=np.float32)
            for row in range(len(ids)):
                values[row] = (row + 1) + np.arange(151, dtype=np.float32)[:, None, None] / 150
            group["vorticity"] = values
        # Invalid independent test content must never be read by the trainer.
        handle["test/vorticity"] = np.asarray([np.nan], dtype=np.float32)
    return path


def test_formal_budget_sparse_times_and_tail_are_explicit():
    """Check the default budget, ten actual frame offsets, effective batch and terminal selection."""
    config = configuration(arguments([]))
    assert config["formal"] and not config["fixture_only"]
    assert config["epochs"] == 500 and config["trajectories"] == 1000
    assert config["effective_batch"] == 32 and config["accumulation"] == 32
    assert config["drop_last"] is False and config["selection"] == "terminal_epoch_not_validation"
    assert config["training_time_points"] == 10 and config["prediction_time_points"] == 151
    assert config["training_frame_indices"] == [0, 16, 33, 50, 66, 83, 100, 116, 133, 150]
    assert np.array_equal(config["training_local_times"], TRAIN_INDICES * .02)
    # Actual integer observation times differ from an invented uniform grid.
    assert not np.allclose(config["training_local_times"], np.linspace(0, 3, 10))
    assert configuration(arguments(["--micro-batch", "2", "--accumulation", "2"]))["effective_batch"] == 4


def test_sampling_visits_every_identity_and_is_independent_of_global_rng():
    """Require a full trajectory permutation and repeatable bounded anchors independent of global NumPy draws."""
    first = epoch_indices(1000, 501, 20260820, 17)
    np.random.seed(999)
    np.random.random(100)
    again = epoch_indices(1000, 501, 20260820, 17)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert np.array_equal(np.sort(first[0]), np.arange(1000))
    assert first[1].min() >= 0 and first[1].max() <= 350
    assert not np.array_equal(first[1], epoch_indices(1000, 501, 20260820, 18)[1])


def test_bank_reads_only_train_validation_and_preserves_sparse_frames(tmp_path, monkeypatch):
    """Verify selected row order, ten-frame targets and scientific fingerprints without touching the test split."""
    path = fixture_observations(tmp_path)
    original = h5py.Group.__getitem__

    def guarded(self, key):
        """Reject any test-group access while forwarding legitimate HDF5 lookup."""
        if str(key).startswith("test"):
            raise AssertionError("ANIE training must not access an independent test group")
        return original(self, key)

    monkeypatch.setattr(h5py.Group, "__getitem__", guarded)
    with ObservationBank(tmp_path, path, formal=False, trajectories=3,
                         cache=True, verify_full=True) as bank:
        initial, target = bank.batch([2, 0], np.zeros(3, dtype=np.int64), "cpu", indices=TRAIN_INDICES)
        assert initial.shape == (2, 64, 64) and target.shape == (2, 10, 64, 64)
        assert torch.equal(initial, target[:, 0])
        np.testing.assert_allclose(target[0, :, 0, 0], 3 + TRAIN_INDICES / 150, rtol=1e-6)
        assert bank.binding["source_fixture_flag"] and bank.binding["fixture_only"]
        assert not bank.binding["sha256_verified"] and len(bank.binding["observations_sha256"]) == 64


class StochasticFixture(torch.nn.Module):
    """Small stand-in tests loop/RNG plumbing, never ANIE numerical behavior."""
    def __init__(self):
        """Create one randomly initialized trainable scale for inexpensive state-restoration checks."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(()) * .01 + .8)

    def forward(self, initial, times=None):
        """Emit time-indexed fields with jitter from Python, NumPy and Torch RNG streams."""
        jitter = torch.rand(()) * .01 + random.random() * .01 + float(np.random.random()) * .01
        return (initial[:, None] * self.scale + times[None, :, None, None] * .2 + jitter)


def test_native_resume_replays_training_validation_and_mixed_rng(tmp_path):
    """Require exact next-epoch loss, validation, parameters, Adam slots and cosine state after resume."""
    path = fixture_observations(tmp_path)
    config = configuration(arguments(["--tiny", "--tiny-trajectories", "3", "--micro-batch", "2"]))
    configure_random(config["seed"])
    with ObservationBank(tmp_path, path, formal=False, trajectories=3, cache=True) as bank:
        model = StochasticFixture()
        optimizer, scheduler = native_training_objects(model, config)
        # Epoch17 is the first native scheduler step; use its boundary to test
        # scheduler state as well as Adam and Python/NumPy/Torch randomness.
        record = train_epoch(model, bank, optimizer, config, 16, torch.device("cpu"))
        assert record["processed_trajectories"] == 3 and record["optimizer_updates_this_epoch"] == 2
        validation_measurement(model, bank, config, torch.device("cpu"))
        step_scheduler(scheduler, 16)
        store = CheckpointStore(tmp_path / "run", {"fixture_only": True, "configuration": config})
        store.save("epoch_0017", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                     scheduler=scheduler.state_dict()))
        expected_record = train_epoch(model, bank, optimizer, config, 17, torch.device("cpu"))
        expected_validation = validation_measurement(model, bank, config, torch.device("cpu"))
        step_scheduler(scheduler, 17)
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())

        configure_random(999)
        replay = StochasticFixture()
        replay_optimizer, replay_scheduler = native_training_objects(replay, config)
        resumed = CheckpointStore(tmp_path / "run", {"fixture_only": True, "configuration": config}, resume=True)
        replay.load_state_dict(resumed.payload["model"])
        replay_optimizer.load_state_dict(resumed.payload["optimizer"])
        replay_scheduler.load_state_dict(resumed.payload["scheduler"])
        resumed.restore_random_state()
        assert train_epoch(replay, bank, replay_optimizer, config, 17, torch.device("cpu")) == expected_record
        assert validation_measurement(replay, bank, config, torch.device("cpu")) == expected_validation
        step_scheduler(replay_scheduler, 17)
        assert all(torch.equal(value, expected_model[key]) for key, value in replay.state_dict().items())
        assert replay_scheduler.state_dict() == expected_scheduler
        for parameter, state in replay_optimizer.state_dict()["state"].items():
            for key, value in state.items():
                assert torch.equal(value, expected_optimizer["state"][parameter][key])


def test_objective_uses_initial_reconstruction_and_native_mse():
    """An error confined to time zero must contribute one tenth of the observed-window MSE."""
    class WrongInitial(torch.nn.Module):
        """Synthetic predictor with only its initial reconstruction wrong; isolates time-zero loss weighting."""
        def forward(self, initial, times=None):
            """Return zero future fields and a unit initial error for the requested observation count."""
            result = torch.zeros((len(initial), len(times), 64, 64))
            result[:, 0] = 1
            return result
    loss = objective(WrongInitial(), torch.zeros(1, 64, 64), torch.zeros(1, 10, 64, 64), torch.arange(10))
    assert float(loss) == pytest.approx(.1)


def test_scheduler_preserves_official_epoch_gate():
    """Require sixteen unchanged scheduler boundaries and the first cosine step after epoch seventeen."""
    model = torch.nn.Linear(1, 1)
    config = configuration(arguments([]))
    optimizer, scheduler = native_training_objects(model, config)
    initial = copy.deepcopy(scheduler.state_dict())
    for epoch in range(16):
        optimizer.step()
        step_scheduler(scheduler, epoch)
    assert scheduler.state_dict() == initial
    optimizer.step()
    step_scheduler(scheduler, 16)
    assert scheduler.state_dict()["last_epoch"] == initial["last_epoch"] + 1
