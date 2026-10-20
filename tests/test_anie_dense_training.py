"""Dense151 ANIE observation, budget and native continuation contracts; no formal training."""
import copy
import random

import h5py
import numpy as np
import pytest
import torch

from training.anie_data import ObservationBank, epoch_indices
from training.checkpoints import CheckpointStore
from training.train_anie_dense import (DENSE_INDICES, dense_time_grid_identity, source_hashes, arguments, configuration, configure_random,
                                 native_training_objects, objective, step_scheduler,
                                 train_epoch, validation_measurement)


def fixture_observations(tmp_path, count=3):
    """Write synthetic 64-grid trajectories whose row/frame values make dense indexing observable."""
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


def test_formal_budget_dense_times_and_tail_are_explicit():
    """Check the dense151 method, every observed offset and the default 16,000-update budget arithmetic."""
    config = configuration(arguments(["--micro-batch", "4", "--accumulation", "8"]))
    assert config["formal"] and not config["fixture_only"]
    assert config["epochs"] == 500 and config["trajectories"] == 1000
    assert config["effective_batch"] == 32 and config["accumulation"] == 8
    assert config["drop_last"] is False and config["selection"] == "terminal_epoch_not_validation"
    assert config["training_time_points"] == 151 and config["prediction_time_points"] == 151
    assert config["training_frame_indices"] == list(range(151))
    assert np.array_equal(config["training_local_times"], DENSE_INDICES * .02)
    assert config["method"] == "anie_dense151" and config["protocol"] == "anie.dense151.v1"
    assert config["micro_batch"] == 4 and config["prediction_time_points"] == 151
    assert config["loss"].endswith("151_observed_times_including_initial")
    assert (1000 + 31) // 32 * 500 == 16000
    assert configuration(arguments(["--micro-batch", "2", "--accumulation", "2"]))["effective_batch"] == 4


def test_sampling_visits_every_identity_and_is_independent_of_global_rng():
    """Require reproducible full-population sampling with legal 3-second anchors and epoch-local variation."""
    first = epoch_indices(1000, 501, 20260820, 17)
    np.random.seed(999)
    np.random.random(100)
    again = epoch_indices(1000, 501, 20260820, 17)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert np.array_equal(np.sort(first[0]), np.arange(1000))
    assert first[1].min() >= 0 and first[1].max() <= 350
    assert not np.array_equal(first[1], epoch_indices(1000, 501, 20260820, 18)[1])


def test_bank_reads_only_train_validation_and_preserves_all_dense_frames(tmp_path, monkeypatch):
    """Read reordered trajectories and retain all 151 frames while guarding the independent test split."""
    path = fixture_observations(tmp_path)
    original = h5py.Group.__getitem__

    def guarded(self, key):
        """Fail on test-group access so accidental training/test leakage is visible."""
        if str(key).startswith("test"):
            raise AssertionError("ANIE training must not access an independent test group")
        return original(self, key)

    monkeypatch.setattr(h5py.Group, "__getitem__", guarded)
    with ObservationBank(tmp_path, path, formal=False, trajectories=3,
                         cache=True, verify_full=True) as bank:
        initial, target = bank.batch([2, 0], np.zeros(3, dtype=np.int64), "cpu", indices=DENSE_INDICES)
        assert initial.shape == (2, 64, 64) and target.shape == (2, 151, 64, 64)
        assert torch.equal(initial, target[:, 0])
        np.testing.assert_allclose(target[0, :, 0, 0], 3 + DENSE_INDICES / 150, rtol=1e-6)
        assert bank.binding["source_fixture_flag"] and bank.binding["fixture_only"]
        assert not bank.binding["sha256_verified"] and len(bank.binding["observations_sha256"]) == 64


class StochasticFixture(torch.nn.Module):
    """Small stand-in tests loop/RNG plumbing, never ANIE numerical behavior."""
    def __init__(self):
        """Initialize one scale parameter; full ANIE solving is unnecessary for RNG/controller tests."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(()) * .01 + .8)

    def forward(self, initial, times=None):
        """Provide dense fields, mixed-RNG jitter and a reported two-iteration solver count."""
        self.last_solver_iterations = 2
        jitter = torch.rand(()) * .01 + random.random() * .01 + float(np.random.random()) * .01
        return (initial[:, None] * self.scale + times[None, :, None, None] * .2 + jitter)


def test_native_resume_replays_training_validation_and_mixed_rng(tmp_path):
    """Replay the next dense epoch exactly, including validation, RNG, Adam, cosine and iteration histograms."""
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
        assert record["training_solver_iterations"] == {"2": 2}
        measured = validation_measurement(model, bank, config, torch.device("cpu"))
        assert measured["validation_solver_iterations"] == {"2": 1}
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
    """A unit error only at time zero must yield MSE 1/151 across the dense window."""
    class WrongInitial(torch.nn.Module):
        """Time-zero-only error fixture for dense observation weighting."""
        def forward(self, initial, times=None):
            """Return a dense zero field except for the deliberately wrong initial slice."""
            result = torch.zeros((len(initial), len(times), 64, 64))
            result[:, 0] = 1
            return result
    loss = objective(WrongInitial(), torch.zeros(1, 64, 64), torch.zeros(1, 151, 64, 64), torch.arange(151))
    assert float(loss) == pytest.approx(1 / 151)


def test_scheduler_preserves_official_epoch_gate():
    """Confirm dense training retains the native epoch-17 cosine scheduling gate."""
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



def test_exact_native_query_time_bytes_are_bound():
    """Compare recorded float32 time vectors/digests with actual query construction and distinguish rounding orders."""
    import hashlib
    config = configuration(arguments([]))
    physical = torch.as_tensor(config["training_local_times"], dtype=torch.float32)
    actual = dense_time_grid_identity(torch.device("cpu"))
    assert actual["physical_float32_sha256"] == hashlib.sha256(physical.numpy().tobytes()).hexdigest()
    assert actual["normalized_float32_sha256"] == hashlib.sha256((physical / 3.0).numpy().tobytes()).hexdigest()
    assert actual["frames"] == 151 and actual["window_seconds"] == 3.0
    assert actual["declared_backend"] == "cpu" and actual["device"] == "cpu" and actual["byte_order"] == "little"
    assert hashlib.sha256(np.asarray(actual["physical_float32_values"], dtype=np.float32).tobytes()).hexdigest() == actual["physical_float32_sha256"]
    assert hashlib.sha256(np.asarray(actual["normalized_float32_values"], dtype=np.float32).tobytes()).hexdigest() == actual["normalized_float32_sha256"]
    # Adapter's default float32 multiply rounds differently; dense train/eval
    # deliberately pass the same explicit vector instead of changing adapter.
    assert not torch.equal(physical, torch.arange(151, dtype=torch.float32) * .02)
    files = source_hashes()
    assert "training/train_anie_dense.py" in files and "training/train_anie.py" not in files
    assert "training/anie_data.py" in files and "adapters/anie.py" in files


def test_dense_accumulation_weights_partial_tail_and_records_native_forward_counts(tmp_path):
    """Two microbatch layouts must give matching updates for seven rows, including the three-row tail."""
    path = fixture_observations(tmp_path, count=7)

    class DeterministicFixture(torch.nn.Module):
        """One-parameter predictor isolates microbatch weighting from stochastic solver behavior."""
        def __init__(self):
            """Use an identical scalar initialization for both accumulation layouts."""
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.8))

        def forward(self, initial, times=None):
            """Evaluate all 151 times and report three iterations per native forward call."""
            self.last_solver_iterations = 3
            assert len(times) == 151
            return initial[:, None] * self.scale + times[None, :, None, None] * .2

    with ObservationBank(tmp_path, path, formal=False, trajectories=7, cache=True) as bank:
        configs = [configuration(arguments(["--tiny", "--tiny-trajectories", "7",
                   "--micro-batch", str(micro), "--accumulation", str(acc)])) for micro, acc in ((2, 2), (4, 1))]
        models = [DeterministicFixture(), DeterministicFixture()]
        records = []
        for model, config in zip(models, configs):
            optimizer, _ = native_training_objects(model, config)
            records.append(train_epoch(model, bank, optimizer, config, 0, torch.device("cpu")))
        assert all(r["processed_trajectories"] == 7 and r["optimizer_updates_this_epoch"] == 2 for r in records)
        assert records[0]["training_solver_iterations"] == {"3": 4}
        assert records[1]["training_solver_iterations"] == {"3": 2}
        assert records[0]["training_mse"] == pytest.approx(records[1]["training_mse"], rel=2e-6)
        torch.testing.assert_close(models[0].scale, models[1].scale, rtol=1e-6, atol=1e-7)
