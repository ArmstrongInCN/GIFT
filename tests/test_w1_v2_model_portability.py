"""Legacy V2 evaluation relocates construction/forward sources without rewriting evidence."""
import copy
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from experiments.additional import w1_v2_evaluate as evaluation
from training import w1_v2_train as training


@pytest.mark.parametrize("method,variable", (("fno2d", "GIFT_EXTERNAL_ROOT"),
                                           ("anie_ns", "GIFT_ANIE_SOURCE_ROOT"),
                                           ("anie_f", "GIFT_ANIE_SOURCE_ROOT")))
def test_relocated_legacy_construction_and_prediction_preserve_recorded_spec(tmp_path, monkeypatch, method, variable):
    """The real loader and predictor use current roots; checkpoint/raw metadata keep the locked root."""
    current = tmp_path / "current_upstream"
    current.mkdir()
    monkeypatch.setenv(variable, str(current))
    grid = np.arange(151, dtype=np.float64) * .02
    spec = dict(method=method, history=1, source_root="unavailable-original-machine",
                query_times=grid.tolist(), future_steps=150, observation_dt=.02)
    if method.startswith("anie"):
        spec["condition_mode"] = "native" if method == "anie_ns" else "fixed_f"
    reference = torch.nn.Linear(1, 1)
    payload = dict(model_spec=copy.deepcopy(spec), config={}, stage="formal", seed=11,
                   epoch=1, val_metric=.1, model_state=reference.state_dict())
    checkpoint = tmp_path / "best.pt"
    torch.save(payload, checkpoint)
    original_bytes = checkpoint.read_bytes()
    built, forwarded = [], []

    def build(actual_spec, device):
        built.append(copy.deepcopy(actual_spec))
        assert actual_spec == dict(spec, source_root=str(current))
        return torch.nn.Linear(1, 1)

    def forward(model, initial, actual_spec, normalization, device, *, training):
        forwarded.append(copy.deepcopy(actual_spec))
        assert actual_spec == dict(spec, source_root=str(current))
        assert not training and initial.shape == (1, 64, 64)
        return initial[:, None].expand(-1, len(grid), -1, -1)

    monkeypatch.setattr(training, "build_network", build)
    monkeypatch.setattr(training, "forward_model", forward)
    _, loaded = evaluation._load_checkpoint(checkpoint, "cpu")
    assert loaded["model_spec"] == spec
    window = SimpleNamespace(local_times=grid, trajectory_ids=np.asarray([100000]), cohort="confirmation",
                             initial=lambda first, last: np.ones((last-first, 64, 64), np.float32))
    raw = tmp_path / "predictions.h5"
    returned, _ = evaluation.predict_checkpoint(checkpoint, window, raw, device="cpu", batch_size=1)
    assert len(built) == 2 and len(forwarded) == 1
    assert returned["model_spec"] == spec and checkpoint.read_bytes() == original_bytes
    with h5py.File(raw, "r") as handle:
        assert handle.attrs["status"] == "complete"
        assert json.loads(handle.attrs["model_spec_json"]) == spec
        assert handle["prediction"].shape == (1, 151, 64, 64)
        assert np.isfinite(handle["prediction"][:]).all()
