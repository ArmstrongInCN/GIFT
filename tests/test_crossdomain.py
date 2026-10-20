"""Check cross-domain CLI paths and input limits with stubbed model inference.

Synthetic HDF files exercise output routing and validation before training.
The saved model bytes and rollout stand-ins are not scientific predictions.
"""
import json

import h5py
import numpy as np
import pytest

from gift import identified
from scripts import generate_crossdomain_data as generation
from scripts import run_crossdomain as runner


def observations(path, frames):
    """Write one zero-filled test trajectory with the runner's N64/time metadata."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("test")
        group.create_dataset("vorticity", shape=(1, frames, 64, 64), dtype="float32")
        group.create_dataset("trajectory_index", data=np.array([2000], dtype=np.int64))
        group.create_dataset("time", data=np.arange(frames) * .02)


def test_evaluate_saved_model_into_new_directory(tmp_path, monkeypatch):
    """Evaluate an explicitly selected artifact without modifying its weight bytes.

    The stub returns repeated zero fields, so assertions cover model-path
    dispatch and saved reports rather than cross-domain model accuracy.
    """
    dataset, model, output = tmp_path / "heat.h5", tmp_path / "model.pt", tmp_path / "evaluation"
    observations(dataset, 401)
    model.write_bytes(b"saved weights")
    loaded = []
    monkeypatch.setattr(identified, "load_generator", lambda path, **kwargs: loaded.append(path) or object())
    monkeypatch.setattr(identified, "FixedBandwidthGridGenerator", lambda *args, **kwargs: object())
    monkeypatch.setattr(identified, "rollout_rk4", lambda model, initial, **kwargs: initial[None].expand(16, *initial.shape))
    runner.main(["--kind", "heat", "--data", str(dataset), "--stage", "evaluate",
                 "--model", str(model), "--output", str(output), "--device", "cpu", "--execute"])
    assert loaded == [model]
    report = json.loads((output / "summary/summary.json").read_text())
    assert report["model"] == str(model.resolve())
    assert report["final_mean_relative_l2"] == 0
    assert model.read_bytes() == b"saved weights"
    assert (output / "summary/metrics.csv").is_file()


def test_short_dataset_rejected_before_training_or_output(tmp_path, monkeypatch):
    """Reject an undersized horizon before entering training or creating output."""
    dataset, output = tmp_path / "short.h5", tmp_path / "all"
    observations(dataset, 51)
    monkeypatch.setattr(runner, "train_stage", lambda *args: pytest.fail("training was started"))
    with pytest.raises(ValueError, match="401 frames"):
        runner.main(["--kind", "heat", "--data", str(dataset), "--output", str(output),
                     "--device", "cpu", "--execute"])
    assert not output.exists()


@pytest.mark.parametrize("argument,count", [("--train-count", 251), ("--validation-count", 21), ("--test-count", 21)])
def test_population_cannot_exceed_id_pool(tmp_path, argument, count):
    """Reject train/validation/test counts beyond their fixed generator ID pools."""
    output = tmp_path / "data"
    with pytest.raises(SystemExit) as error:
        generation.main(["--kind", "heat", "--output", str(output), argument, str(count), "--execute"])
    assert error.value.code == 2
    assert not output.exists()
