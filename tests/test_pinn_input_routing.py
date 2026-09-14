"""Input selection only; mocked runtime does not construct a TensorFlow graph."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from adapters import pinn_runtime
from training import pinn_data, train_pinn


@pytest.fixture
def routes(monkeypatch):
    observed = []
    selected = (np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32),
                np.zeros((4, 3), np.float32), np.zeros(3, np.float32),
                np.ones(3, np.float32), {"unit_test_only": True})
    def read_released(*args):
        observed.append(("released", args))
        return selected
    def read_regenerated(*args):
        observed.append(("regenerated", args))
        return selected
    def runtime(*args):
        observed.append(("runtime", args))
        return SimpleNamespace(profile={"unit_test_only": True}, source_records={})
    monkeypatch.setattr(pinn_runtime, "read_inputs", read_released)
    monkeypatch.setattr(pinn_data, "read_regenerated_inputs", read_regenerated)
    monkeypatch.setattr(pinn_runtime, "PINNRuntime", runtime)
    return observed, selected


def test_default_keeps_released_reader_and_identity(routes):
    observed, selected = routes
    backend = train_pinn.TensorFlowBackend("sampling", "data", train_rows=2, physics_rows=4)
    assert observed[0] == ("released", ("sampling", "data", 2, 4))
    assert [item[0] for item in observed] == ["released", "runtime"]
    assert backend.coordinates is selected[0] and backend.targets is selected[1]
    assert backend.identity["inputs"] is selected[-1]
    assert "input_gate_sha256" not in backend.identity


def test_regenerated_uses_explicit_collection_gate(routes):
    observed, selected = routes
    backend = train_pinn.TensorFlowBackend("sampling", "data", data_profile="regenerated",
        data_root="collection", condition="noise_001", train_rows=2, physics_rows=4)
    assert observed[0] == ("regenerated", ("sampling", "data", "collection", "noise_001", 2, 4))
    assert [item[0] for item in observed] == ["regenerated", "runtime"]
    assert backend.identity["inputs"] is selected[-1]
    assert backend.identity["input_gate_sha256"] == train_pinn.sha256(pinn_data.__file__)


@pytest.mark.parametrize("arguments", [dict(data_profile="unknown"),
    dict(data_profile="regenerated"), dict(data_profile="regenerated", data_root="collection")])
def test_bad_profile_is_rejected_before_graph_or_input_io(routes, arguments):
    with pytest.raises(ValueError):
        train_pinn.TensorFlowBackend("sampling", "data", **arguments)
    assert routes[0] == []


def test_regenerated_cli_plan_has_no_output_or_tensorflow(tmp_path, monkeypatch, capsys):
    import sys
    output = tmp_path / "not_created"
    monkeypatch.setattr(sys, "argv", ["train_pinn", "--data-profile", "regenerated", "--output", str(output)])
    train_pinn.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["plan_only"] and plan["data_profile"] == "regenerated"
    assert not plan["training_executed"] and not output.exists()
    assert "tensorflow" not in sys.modules
