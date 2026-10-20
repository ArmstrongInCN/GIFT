"""PINN input routing keeps one numerical reader and no manifest gate."""
from pathlib import Path
import numpy as np
import pytest
from adapters import pinn_runtime
from training import pinn_data


def test_regenerated_route_uses_same_observation_reader(tmp_path, monkeypatch):
    """Regenerated routing must pass row selection unchanged and augment only provenance metadata."""
    sampling, data = tmp_path / "sampling.npz", tmp_path / "observations.h5"
    sampling.touch(); data.touch()
    arrays = (np.zeros((2, 3), np.float32),) * 3 + (np.zeros(3), np.ones(3))
    calls = []
    def read(*args, **kwargs):
        """Record adapter arguments and return synthetic selected arrays without actual PINN IO."""
        calls.append((args, kwargs))
        return (*arrays, {"selection": {"train_rows": 2}})
    monkeypatch.setattr(pinn_runtime, "read_inputs", read)
    actual = pinn_data.read_regenerated_inputs(sampling, data, tmp_path, "noise_001", 1, 2)
    assert actual[:5] == arrays
    assert actual[-1]["provenance"]["profile"] == "regenerated"
    assert calls == [((sampling, data, 1, 2), {"condition": "noise_001"})]


def test_input_root_boundary_remains_checked(tmp_path, monkeypatch):
    """An escaped sampling file must fail before the shared reader is called."""
    root = tmp_path / "collection"; root.mkdir()
    sampling = tmp_path / "outside.npz"; sampling.touch()
    data = root / "data.h5"; data.touch()
    monkeypatch.setattr(pinn_runtime, "read_inputs", lambda *a, **k: pytest.fail("escaped input was consumed"))
    with pytest.raises(ValueError, match="data root"):
        pinn_data.read_regenerated_inputs(sampling, data, root, "noise_000")
