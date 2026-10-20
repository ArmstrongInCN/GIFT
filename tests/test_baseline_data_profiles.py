"""Ordinary observed training files need schema checks, not package receipts."""
import h5py
import numpy as np
import pytest
import torch
from training import baseline_control as control


def observations(tmp_path):
    """Write a minimal finite training population with unique IDs and DT=0.02 time coordinates."""
    path = tmp_path / "observations.h5"
    with h5py.File(path, "w") as h:
        h["training/vorticity"] = np.ones((2, 48, 64, 64), np.float32)
        h["training/trajectory_index"] = [3, 8]
        h["training/time"] = np.arange(48) * .02
    return path


@pytest.mark.parametrize("profile", ["released", "regenerated", "canonical", "gaussian"])
def test_raw_observations_work_without_manifest_or_hash(tmp_path, profile, monkeypatch):
    """Accept valid observed arrays for every profile without invoking a file-digest gate."""
    path = observations(tmp_path)
    config = control.configuration("uno", control._arguments("uno", ["--tiny", "--data-profile", profile]))
    monkeypatch.setattr(control, "digest_file", lambda *a: (_ for _ in ()).throw(AssertionError("no ordinary hash gate")))
    record = control._validate_data(path, config, hash_bytes=True)
    assert record["shape"] == [2, 48, 64, 64]
    assert record["sha256"] is None and not record["sha256_verified"]


@pytest.mark.parametrize("change", ["ids", "time", "population"])
def test_actual_schema_and_population_remain_checked(tmp_path, change):
    """Reject duplicate IDs, a changed physical time step or an oversized selected population."""
    path = observations(tmp_path)
    config = control.configuration("uno", control._arguments("uno", ["--tiny"]))
    with h5py.File(path, "r+") as h:
        if change == "ids": h["training/trajectory_index"][1] = 3
        if change == "time": h["training/time"][1] = .03
    if change == "population": config["trajectories"] = 3
    with pytest.raises(ValueError):
        control._validate_data(path, config, hash_bytes=False)


def test_nonfinite_sample_is_rejected_before_training(tmp_path):
    """A NaN in the sampled history must stop batch creation before forward/backward."""
    path = observations(tmp_path)
    config = control.configuration("uno", control._arguments("uno", ["--tiny"]))
    with h5py.File(path, "r+") as h:
        h["training/vorticity"][0, 0, 0, 0] = np.nan
        with pytest.raises(FloatingPointError):
            control._batch(h["training/vorticity"], [0], np.array([0, 0]), config, torch.device("cpu"))


def test_custom_budget_seed_and_explicit_zero_rollout():
    """Honor explicit counts/seeds and reject a zero rollout rather than silently replacing it."""
    config = control.configuration("uno", control._arguments("uno", ["--epochs", "3", "--trajectories", "7", "--seed", "91"]))
    assert (config["epochs"], config["trajectories"], config["seed"]) == (3, 7, 91)
    with pytest.raises(ValueError, match="positive"):
        control.configuration("uno", control._arguments("uno", ["--tiny", "--tiny-rollout", "0"]))
