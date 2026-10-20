"""Crash-boundary recovery repairs progress JSON without another update or weight write."""
import hashlib
import json

import numpy as np
import pytest
import torch

from training import w1_single_train as native
from training import w1_extended_baselines as extended


class FixtureBank:
    """Two synthetic trajectories exercise the real journal commit ordering."""
    def __init__(self, path, fixture=False):
        self.training = np.random.default_rng(10).normal(size=(2, 76, 2, 2)).astype("f4") + 2
        self.validation = self.training.copy()
        self.binding = dict(fixture=True, points=76, future_steps=75,
                            query_times=(np.arange(76) * .04).tolist())

    def batch(self, rows, device, validation=False):
        truth = torch.tensor((self.validation if validation else self.training)[rows], device=device)
        return truth[:, 0], truth

    def normalization(self, method):
        return {"kind": "none"}


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.8))


class Stopper:
    """A deterministic stand-in leaves ANIE's selected checkpoint protocol active."""
    early_stop = False

    def __call__(self, metric):
        self.last_metric = metric


def forecast(model, initial, spec, normalization=None, device=None, training=False,
             horizon=None, steps=None, **kwargs):
    points = (horizon if horizon is not None else steps if steps is not None else 75) + 1
    return (model.weight * initial)[:, None].expand(-1, points, -1, -1)


def relative_sum(prediction, truth):
    return ((prediction - truth).norm(dim=1) / truth.norm(dim=1)).sum()


def fixture_run(monkeypatch, control, method):
    """Keep trainer/loss/journal code real; replace only costly models and inputs."""
    monkeypatch.setattr(control, "WindowBank", FixtureBank)
    monkeypatch.setattr(control, "build_network", lambda spec, device: Scale().to(device))
    monkeypatch.setattr(control, "_source_provenance", lambda *args: {"fixture_only": True})
    monkeypatch.setattr(control, "forward_model", forecast)
    monkeypatch.setattr(control, "runtime_identity", lambda: {"fixture_only": True})
    if control is native:
        monkeypatch.setattr(control, "fno_utilities", lambda root: (None, lambda **kw: relative_sum))
        monkeypatch.setattr(control, "native_early_stopping", lambda *args: Stopper())
    else:
        monkeypatch.setattr(control, "_configure_profile", lambda *args: torch.manual_seed(0))
        monkeypatch.setattr(control, "make_optimizer", lambda model, spec, config:
            (torch.optim.Adam(model.parameters(), lr=config["learning_rate"]), relative_sum))
    config = control.native_training_config(method)
    config.update(fixture=True, max_epochs=1, effective_batch=2, micro_batch=2, checkpoint_interval=1)
    spec = dict(method=method, history=1, source_root="synthetic-fixture-only")
    if method == "anie_ns":
        spec.update(condition_mode="native", source_profile=native.SOURCE_PROFILE)
    return config, spec


@pytest.mark.parametrize("control,method", [(native, "fno2d"), (native, "fno3d"),
                                           (native, "anie_ns"), (extended, "uno"), (extended, "unet")])
@pytest.mark.parametrize("missing", [False, True])
def test_terminal_journal_repairs_lagging_or_absent_json_without_touching_weights(
        tmp_path, monkeypatch, control, method, missing):
    """A hard exit after latest.pt cannot leave a completed queue slot running."""
    config, spec = fixture_run(monkeypatch, control, method)
    output = tmp_path / method
    control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu")
    protected = {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                 for path in output.glob("*.pt")}
    for name in ("STATUS.json", "HISTORY.json"):
        path = output / name
        if missing:
            path.rename(output / (name + ".before_crash"))
        else:
            path.write_text(json.dumps(dict(status="running", epoch=0, selected_checkpoint=None)
                                       if name == "STATUS.json" else dict(training=[], validation=[])), encoding="utf-8")
    lagging = {name: (output / name).read_bytes() for name in ("STATUS.json", "HISTORY.json")
               if (output / name).exists()}
    monkeypatch.setattr(control, "forward_model", lambda *args, **kw: pytest.fail("terminal resume cannot train"))
    resumed = control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu", resume=True)
    status = json.loads((output / "STATUS.json").read_text(encoding="utf-8"))
    assert resumed["status"] == status["status"] == "complete"
    assert status["epoch"] == 1 and status["updates"] == 1
    assert status["operational_progress_recovered_from"] == "latest.pt"
    assert status["selected_checkpoint"].endswith("best.pt" if method == "anie_ns" else "terminal_weights.pt")
    history = json.loads((output / "HISTORY.json").read_text(encoding="utf-8"))
    assert len(history["training"]) == len(history["validation"]) == 1
    evidence = list(output.glob("resume_evidence_*"))
    assert len(evidence) == 1
    for name, content in lagging.items():
        assert (evidence[0] / name).read_bytes() == content
    for name, snapshot in protected.items():
        path = output / name
        assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == snapshot
    recovered_status = (output / "STATUS.json").read_bytes()
    control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu", resume=True)
    assert (output / "STATUS.json").read_bytes() == recovered_status
    assert len(list(output.glob("resume_evidence_*"))) == 1


@pytest.mark.parametrize("control,method", [(native, "fno2d"), (extended, "unet")])
def test_terminal_resume_does_not_synthesize_missing_selected_model(tmp_path, monkeypatch, control, method):
    """A terminal journal cannot conceal a lost selected weight file."""
    config, spec = fixture_run(monkeypatch, control, method)
    output = tmp_path / method
    control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu")
    (output / "terminal_weights.pt").rename(output / "terminal_weights.retained.pt")
    with pytest.raises(ValueError, match="existing selected model weights"):
        control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu", resume=True)
    assert not (output / "terminal_weights.pt").exists()


@pytest.mark.parametrize("control,method", [(native, "fno3d"), (native, "anie_ns"), (extended, "unet")])
def test_relocated_sources_are_runtime_only_and_keep_locked_weight_metadata(tmp_path, monkeypatch, control, method):
    """Changing an upstream location must reach all loaders without editing the scientific spec."""
    config, spec = fixture_run(monkeypatch, control, method)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    monkeypatch.setenv("GIFT_ANIE_SOURCE_ROOT" if method == "anie_ns" else "GIFT_EXTERNAL_ROOT", str(upstream))
    seen = []

    def build(runtime_spec, device):
        seen.append(runtime_spec["source_root"])
        return Scale().to(device)

    def predict(model, initial, runtime_spec, *args, **kwargs):
        seen.append(runtime_spec["source_root"])
        return forecast(model, initial, runtime_spec, *args, **kwargs)

    monkeypatch.setattr(control, "build_network", build)
    monkeypatch.setattr(control, "forward_model", predict)
    monkeypatch.setattr(control, "_source_provenance", lambda model, runtime_spec:
                        dict(actual_source_root=runtime_spec["source_root"]))
    if control is native:
        def utilities(root):
            seen.append(root)
            return None, lambda **kw: relative_sum
        def stopper(root, *args):
            seen.append(root)
            return Stopper()
        monkeypatch.setattr(control, "fno_utilities", utilities)
        monkeypatch.setattr(control, "native_early_stopping", stopper)
    output = tmp_path / method
    saved = control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu")
    assert len(seen) >= 3 and set(seen) == {str(upstream)}
    assert spec["source_root"] == saved["identity"]["model_spec"]["source_root"] == "synthetic-fixture-only"
    assert saved["source_provenance"]["actual_source_root"] == str(upstream)
    selected = "best.pt" if method == "anie_ns" else "terminal_weights.pt"
    weights = torch.load(output / selected, weights_only=True)
    assert weights["model_spec"]["source_root"] == "synthetic-fixture-only"
    assert weights["source_provenance"]["actual_source_root"] == str(upstream)


@pytest.mark.parametrize("method", ["anie_ns", "anie_f", "fno3d"])
def test_historical_v2_trainer_relocates_sources_without_editing_its_locked_spec(tmp_path, monkeypatch, method):
    """Historical queues retain their original ANIE branch/grid and weight metadata after relocation."""
    from training import w1_v2_train as control
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    monkeypatch.setenv("GIFT_ANIE_SOURCE_ROOT" if method.startswith("anie") else "GIFT_EXTERNAL_ROOT", str(upstream))
    seen = []

    class Model(Scale):
        def source_identity(self):
            return {"fixture_only": True, "directory": str(upstream)}

    def build(spec, device):
        seen.append(spec["source_root"])
        model = Model().to(device)
        model.gift_external_source = dict(directory=str(upstream))
        return model

    def predict(model, initial, spec, *args, **kwargs):
        seen.append(spec["source_root"])
        return forecast(model, initial, spec, *args, **kwargs)

    def utilities(root):
        seen.append(root)
        return None, lambda **kwargs: relative_sum

    monkeypatch.setattr(control, "WindowBank", FixtureBank)
    monkeypatch.setattr(control, "build_network", build)
    monkeypatch.setattr(control, "forward_model", predict)
    monkeypatch.setattr(control, "fno_utilities", utilities)
    spec = dict(method=method, history=1, source_root="historical-upstream-location")
    if method.startswith("anie"):
        spec["condition_mode"] = "fixed_f" if method == "anie_f" else "native"
    config = dict(seed=11, sample_seed=2026100301, fixture=True, learning_rate=.01,
        max_epochs=1, micro_batch=2, effective_batch=2, validation_interval=1,
        stage="development", min_epochs=1, patience=10, future_steps=75)
    output = tmp_path / method
    saved = control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu")
    assert len(seen) >= 3 and set(seen) == {str(upstream)}
    assert spec["source_root"] == saved["identity"]["model_spec"]["source_root"] == "historical-upstream-location"
    weights = torch.load(output / "best.pt", weights_only=True)
    assert weights["model_spec"]["source_root"] == "historical-upstream-location"
    assert weights["source_provenance"]["runtime_source_root"] == str(upstream)
