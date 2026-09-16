"""Bounded CPU checks for sampling, terminal selection and exact continuation."""

import json

import h5py
import numpy as np
import pytest
import torch

from training.gift_prediction_control import PredictionTrainingConfig, phase_at, run_generator
from training.gift_prediction_data import epoch_indices, observed_pairs, observed_sequences


def make_observations(path, *, gaussian=False):
    rng = np.random.default_rng(881)
    with h5py.File(path, "x") as handle:
        handle.attrs["trajectory_id_scheme"] = "canonical"
        handle.attrs["fixture_only"] = True
        if gaussian:
            handle.attrs['prediction_cohort'] = 'gaussian_s4'
        groups = (("training", [1220, 1221, 1222, 1223]), ("validation", [2220, 2221])) if gaussian else (
            ("training", [0, 1, 2, 3]), ("validation", [1000, 1001]))
        for name, ids in groups:
            group = handle.create_group(name)
            group.create_dataset("trajectory_index", data=np.asarray(ids, dtype=np.int64))
            group.create_dataset("time", data=np.arange(55) * 0.02)
            initial = rng.normal(size=(len(ids), 1, 8, 8)).astype(np.float32)
            trend = rng.normal(size=(len(ids), 1, 8, 8)).astype(np.float32)
            time = (np.arange(55, dtype=np.float32) * 0.02)[None, :, None, None]
            group.create_dataset("vorticity", data=(initial + time * trend).astype(np.float32))


def test_one_window_per_trajectory_and_reproducible_sampling():
    for rollout in (False, True):
        for epoch in (0, 1, 250, 500):
            order, positions = epoch_indices(1000, 501, 7, epoch, rollout=rollout)
            assert sorted(order.tolist()) == list(range(1000))
            again = epoch_indices(1000, 501, 7, epoch, rollout=rollout)
            assert np.array_equal(order, again[0]) and np.array_equal(positions, again[1])
            assert positions.min() >= (0 if rollout else 2)
            assert positions.max() <= (450 if rollout else 498)


def test_five_point_derivative_uses_observations_and_row_order():
    raw = np.arange(2 * 55 * 8 * 8, dtype=np.float32).reshape(2, 55, 8, 8)
    order, centres = np.asarray([1, 0]), np.asarray([3, 7])
    state, target = observed_pairs(raw, order, centres)
    assert torch.equal(state[0], torch.from_numpy(raw[1, 7]))
    expected = (raw[1, 5] - 8.0 * raw[1, 6] + 8.0 * raw[1, 8] - raw[1, 9]) / 0.24
    assert np.array_equal(target[0].numpy(), expected)
    sequences = observed_sequences(raw, order, np.asarray([0, 4]))
    assert np.array_equal(sequences[0, -1], raw[1, 54])


def test_phase_boundaries_count_total_not_each_phase_twice():
    assert phase_at(250, (250, 250)) == (0, 250)
    assert phase_at(251, (250, 250)) == (1, 1)
    assert phase_at(500, (250, 250)) == (1, 250)
    assert phase_at(451, (400, 50, 25, 25)) == (2, 1)
    with pytest.raises(ValueError):
        phase_at(501, (250, 250))


@pytest.mark.parametrize("boundary", [1, 2, 3])
@pytest.mark.parametrize("gaussian", [False, True])
def test_generator_exact_resume_inside_and_across_phases(tmp_path, boundary, gaussian):
    torch.set_num_threads(1)
    dataset = tmp_path / "observations.h5"
    make_observations(dataset, gaussian=gaussian)
    config = PredictionTrainingConfig(generator_phases=(2, 2), branch_phases=(1, 1, 1, 1),
                                      batch_size=3, cutoff=2, rank=2, fixture_only=True)
    full, split = tmp_path / "full", tmp_path / "split"
    run_generator(dataset, full, device="cpu", config=config, checkpoint_interval=2)
    paused = run_generator(dataset, split, device="cpu", config=config,
                           checkpoint_interval=2, stop_after_epoch=boundary)
    assert paused["status"] == "paused_at_committed_epoch"
    assert not (split / "model.pt").exists()
    run_generator(dataset, split, device="cpu", config=config, checkpoint_interval=2, resume=True)
    one = torch.load(full / "model.pt", weights_only=True)
    two = torch.load(split / "model.pt", weights_only=True)
    assert one["selection"] == two["selection"] == "terminal_epoch"
    assert one["training_cost"]["optimizer_updates"] == 8
    assert one["training_cost"]["affine_fits"] == 5
    assert all(torch.equal(one["model_state_dict"][key], two["model_state_dict"][key])
               for key in one["model_state_dict"])
    histories = [json.loads((folder / "history.json").read_text()) for folder in (full, split)]
    for left, right in zip(*histories):
        assert {k: v for k, v in left.items() if k != "seconds"} == {
            k: v for k, v in right.items() if k != "seconds"}
    with pytest.raises(FileExistsError):
        run_generator(dataset, split, device="cpu", config=config, resume=True)


def test_fixture_cannot_be_advertised_as_formal():
    with pytest.raises(ValueError, match="formal"):
        PredictionTrainingConfig(generator_phases=(2, 2)).validate()
