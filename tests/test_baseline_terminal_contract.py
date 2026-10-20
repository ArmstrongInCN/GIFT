"""Actual exporter -> consumer metadata tests, not model training/acceptance.

The exported scalar state is NOT a baseline model. Model loading is explicitly
stubbed; all fixture paths/provenance say METADATA_ONLY_MOCK_NOT_TRAINING. These
tests do not establish that a full-budget training campaign has completed.
"""
import json

import pytest
import torch

from adapters import prediction
from experiments.formal._shared.fno_runtime import _terminal_contract
from training import baseline_control as control


class MetadataOnlyModel:
    """Minimal exporter interface; the scalar tensor is not a baseline architecture or trained model."""
    def state_dict(self):
        """Supply a labeled scalar tensor solely to test serialization and consumer metadata."""
        return {"METADATA_ONLY_MOCK_NOT_TRAINING": torch.ones(1)}

    def requires_grad_(self, enabled):
        """Accept inference freezing without constructing or running a real neural network."""
        assert enabled is False
        return self


def export_fixture(tmp_path, model, tiny=False):
    """Build a real terminal export around synthetic state and shape-correct, unfitted normalization."""
    output = tmp_path / ("METADATA_ONLY_MOCK_NOT_TRAINING_" + model + ("_tiny" if tiny else ""))
    output.mkdir()
    (output / "ATTEMPT.json").write_text(json.dumps({"run_id": output.name}), encoding="utf-8")
    (output / "LATEST.json").write_text(json.dumps({"file": "NO_REAL_TRAINING_CHECKPOINT",
        "sha256": "0" * 64, "fixture_only": True}), encoding="utf-8")
    config = control.configuration(model, control._arguments(model, ["--tiny"] if tiny else []))
    identity = {"purpose": "METADATA_ONLY_MOCK_NOT_TRAINING", "configuration": config,
                "data": {"fixture_only": True}, "sources": {"fixture_only": True}}
    normalization = {"kind": "none"}
    if model == "fno3d":
        # Correct metadata/tensor shapes only; these are NOT fitted statistics.
        normalization = dict(kind="paper_UnitGaussianNormalizer", eps=1e-5,
            input_mean=torch.zeros(64, 64, 46), input_std=torch.ones(64, 64, 46),
            output_mean=torch.zeros(64, 64, config["rollout"]),
            output_std=torch.ones(64, 64, config["rollout"]))
    control._export_terminal(output, MetadataOnlyModel(), normalization, config, identity, resumed=False)
    payload = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    return output, payload, config


@pytest.mark.parametrize("model", ["fno2d", "fno3d", "uno", "unet"])
def test_exported_full_budget_metadata_matches_consumers(tmp_path, monkeypatch, model):
    """Round-trip family/configuration/normalizer metadata while stubbing any actual baseline load."""
    output, payload, config = export_fixture(tmp_path, model)
    assert payload["schema"] == "gift.independent-baseline-weights.v1"
    assert payload["status"] == "complete"
    assert payload["method"] == model
    assert payload["terminal_epoch"] == control.PROTOCOLS[model]["epochs"]
    assert payload["formal_configuration"] == config and config["formal"] is True
    assert payload["artifact_role"] == "fresh_terminal"
    assert payload["initialization"] == "random_from_seed_no_published_checkpoint"
    assert payload["fresh_training"] is True and payload["same_run_resume_used"] is False
    assert payload["run_provenance"]["identity"]["purpose"] == "METADATA_ONLY_MOCK_NOT_TRAINING"
    assert "optimizer_state_dict" not in payload
    assert set(payload["model_state_dict"]) == {"METADATA_ONLY_MOCK_NOT_TRAINING"}
    if model == "fno3d":
        assert payload["normalization"]["kind"] == "paper_UnitGaussianNormalizer"
        assert payload["normalization"]["eps"] == 1e-5
        assert payload["normalization"]["input_mean"].shape == (64, 64, 46)
        assert payload["normalization"]["output_mean"].shape == (64, 64, 150)
    else:
        assert payload["normalization"] == {"kind": "none"}
    if model.startswith("fno"):
        _terminal_contract(payload, "FNO-2D" if model == "fno2d" else "FNO-3D")
    else:
        calls = []
        def metadata_loader(method, path, device):
            """Record inference dispatch and return synthetic state instead of external model construction."""
            calls.append((method, path, device))
            return MetadataOnlyModel(), payload
        monkeypatch.setattr(prediction, "load_checkpoint", metadata_loader)
        assert prediction._load(model, output / "model.pt", "cpu")[1] is payload
        assert calls == [(model, output / "model.pt", "cpu")]


@pytest.mark.parametrize("model", ["uno", "unet"])
def test_custom_budget_export_is_readable_and_keeps_its_metadata(tmp_path, monkeypatch, model):
    """Read a diagnostic budget without turning artifact metadata into inference permission."""
    output, payload, config = export_fixture(tmp_path, model, tiny=True)
    assert payload["artifact_role"] == "test_only" and config["formal"] is False
    monkeypatch.setattr(prediction, "load_checkpoint", lambda *args: (MetadataOnlyModel(), payload))
    assert prediction._load(model, output / "model.pt", "cpu")[1] is payload


def test_matching_export_is_idempotent(tmp_path):
    """Re-exporting identical model/configuration must retain the original file bytes."""
    output, _, config = export_fixture(tmp_path, "fno2d")
    before = (output / "model.pt").read_bytes()
    control._export_terminal(output, MetadataOnlyModel(), {"kind": "none"}, config,
                             {"purpose": "METADATA_ONLY_MOCK_NOT_TRAINING"}, resumed=True)
    assert (output / "model.pt").read_bytes() == before


def test_uno_short_budget_is_descriptive_metadata(tmp_path, monkeypatch):
    """A recorded shorter terminal epoch is readable without pretending it completed the default budget."""
    output, payload, _ = export_fixture(tmp_path, "uno")
    short = dict(payload, terminal_epoch=150)
    monkeypatch.setattr(prediction, "load_checkpoint", lambda *args: (MetadataOnlyModel(), short))
    assert prediction._load("uno", output / "model.pt", "cpu")[1]["terminal_epoch"] == 150


@pytest.mark.parametrize("method", ["FNO-2D", "FNO-3D"])
def test_fno_custom_budget_is_readable_without_formal_permission(method):
    """FNO consumers must accept custom-budget metadata rather than require a formal training label."""
    _terminal_contract({"method": method, "status": "diagnostic_complete", "terminal_epoch": 7,
                        "formal_configuration": False, "artifact_role": "test_only"}, method)
