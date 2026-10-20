"""CPU contracts for the explicit N50 control without altering the N300 recipe."""
import hashlib
import inspect
import json
import math
from types import SimpleNamespace

import numpy as np
import h5py
import pytest
import torch

from training import w1_extended_gift_n50 as control
from training import w1_single_gift as native


@pytest.fixture(autouse=True)
def cpu_only_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def job(tmp_path):
    """A fixed job receipt contains no physical fixture observations or GPU calls."""
    return dict(method="GIFT-N50", expected_training_count=50, seed=20260820,
                generator_seed=2026072301, stage="formal", profile=native.PROFILE,
                config=native.native_training_config(), window=str(tmp_path / "readonly.h5"),
                output=str(tmp_path / "out"), device="cpu")


def test_n50_population_is_explicit_and_all_original_defaults_remain_n300(monkeypatch):
    for name in ("_bank", "run_generator", "run_branch", "run_pipeline"):
        assert inspect.signature(getattr(native, name)).parameters["expected_training_count"].default == 300
    count = 50
    bank = SimpleNamespace(training=np.empty((count, 76, 1, 1)), validation=np.empty((20, 76, 1, 1)),
                           binding=dict(time_stride=2, subset_seed=2026100304))
    monkeypatch.setattr(native.common, "PredictionObservationBank", lambda *args: bank)
    monkeypatch.setattr(native.common, "_grid_spec", lambda value: dict(points=76, save_interval=.04))
    assert native._bank("synthetic", native.NativeGIFTConfig(), expected_training_count=50) is bank
    with pytest.raises(ValueError, match="locked population"):
        native._bank("synthetic", native.NativeGIFTConfig())
    with pytest.raises(ValueError, match="N300 or N50"):
        native._bank("synthetic", native.NativeGIFTConfig(), expected_training_count=100)
    config = native.NativeGIFTConfig()
    assert 500 * math.ceil(50 / config.generator_batch) == 2000
    assert 400 * math.ceil(50 / config.derivative_batch) + 100 * math.ceil(50 / config.rollout_batch) == 2600


def test_pipeline_forwards_n50_to_both_native_components(tmp_path, monkeypatch):
    received = []
    def component(dataset, output, **kwargs):
        received.append(("generator", kwargs["expected_training_count"], kwargs["config"]))
        return dict(status="complete")
    def branch(dataset, generator, output, **kwargs):
        received.append(("branch", kwargs["expected_training_count"], kwargs["config"]))
        return dict(status="complete", training_cost={}, validation_status="finite")
    monkeypatch.setattr(native, "run_generator", component)
    monkeypatch.setattr(native, "run_branch", branch)
    value = native.run_pipeline("synthetic", tmp_path / "new", device="cpu", expected_training_count=50)
    assert value["status"] == "complete"
    assert [item[:2] for item in received] == [("generator", 50), ("branch", 50)]
    assert all(item[2] == native.NativeGIFTConfig() for item in received)


def test_wrapper_frozen_recipe_and_original_seed_forwarding(tmp_path, monkeypatch):
    value = job(tmp_path)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    received = {}
    def pipeline(window, output, **kwargs):
        received.update(kwargs)
        return dict(status="complete", paths=dict(branch_model=str(output / "branch/model.pt")))
    monkeypatch.setattr(native, "run_pipeline", pipeline)
    control.main(["--run-config", str(path)])
    assert received["expected_training_count"] == 50 and received["device"] == "cpu"
    assert received["config"] == native.NativeGIFTConfig()
    label = json.loads((tmp_path / "out/COMPARISON_LABEL.json").read_text())
    assert label["method"] == "GIFT-N50" and label["architecture_method"] == "GIFT"
    assert label["training_count"] == 50 and "not original GIFT-Lite" in label["interpretation"]
    value["config"]["branch_phases"] = [300, 50, 25, 25]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed GIFT"):
        control.main(["--run-config", str(path)])


def test_wrapper_propagates_numerical_failure_without_retry_or_parameter_change(tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job(tmp_path)), encoding="utf-8")
    def failure(*args, **kwargs):
        raise FloatingPointError("native training correction gate rejected")
    monkeypatch.setattr(native, "run_pipeline", failure)
    with pytest.raises(FloatingPointError):
        control.main(["--run-config", str(path)])
    saved = json.loads((tmp_path / "out/FAILURE.json").read_text())
    assert saved["failure_kind"] == "numerical" and saved["method"] == "GIFT-N50"
    assert saved["seed"] == 20260820


def test_wrapper_oom_is_a_retained_resource_slot_with_original_seed(tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job(tmp_path)), encoding="utf-8")
    calls = []
    def out_of_memory(*args, **kwargs):
        calls.append(kwargs)
        raise torch.cuda.OutOfMemoryError("CPU synthetic fixture: out of memory")
    monkeypatch.setattr(native, "run_pipeline", out_of_memory)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        control.main(["--run-config", str(path)])
    saved = json.loads((tmp_path / "out/FAILURE.json").read_text())
    assert len(calls) == 1 and calls[0]["config"] == native.NativeGIFTConfig()
    assert saved["failure_kind"] == "resource" and saved["method"] == "GIFT-N50" and saved["seed"] == 20260820


def test_real_n50_sparse_receipt_loads_and_retains_unchanged_fd_and_native_leads(tmp_path):
    """The actual shared reader accepts N50 IDs/hashes while the default rejects it."""
    from training import w1_v2_data as data
    manifest = data.window_manifest(training_count=50, time_stride=2)
    text = data._manifest_text(manifest)
    times = np.asarray(manifest["query_times"])
    path = tmp_path / "n50.h5"
    with h5py.File(path, "x") as saved:
        saved.attrs.update(schema=data.SPARSE_SCHEMA, status="complete", fixture_only=False,
                           source_file="synthetic_exact_window", manifest_json=text,
                           manifest_sha256=hashlib.sha256(text.encode()).hexdigest())
        for split, ids in (("training", manifest["training_ids"]), ("validation", manifest["validation_ids"])):
            node = saved.create_group(split)
            node.create_dataset("trajectory_index", data=ids)
            node.create_dataset("time", data=times)
            node.create_dataset("absolute_time", data=times+5)
            values = np.broadcast_to((1+times*.1).astype("f4")[None, :, None, None], (len(ids), 76, 64, 64)).copy()
            node.create_dataset("vorticity", data=values, chunks=(1, 76, 64, 64), compression="lzf")
            node.attrs["observations_sha256"] = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
    bank = native._bank(path, native.NativeGIFTConfig(), expected_training_count=50)
    assert bank.binding["training_count"] == 50 and len(bank.validation) == 20
    assert bank.observation_dt == .04
    assert bank.integration_steps.tolist() == [0, 4, 10, 20, 24, 30, 40, 50]
    assert bank.observation_offsets.tolist() == [0, 2, 5, 10, 12, 15, 20, 25]
    states, derivative = bank.pairs(2026072301, 0)
    assert torch.allclose(derivative, torch.full_like(derivative, .1), atol=2e-5)
