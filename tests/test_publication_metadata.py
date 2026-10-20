"""Public exports omit placeholder source text while training state stays exact."""
from collections import OrderedDict, namedtuple
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from gift.publication_metadata import public_metadata, source_metadata_findings


def provenance_fixture():
    """Use descriptive placeholders, never copied upstream implementation text."""
    return {
        "repository": "https://example.org/upstream", "commit": "recorded-revision",
        "source_code_sha256": "a" * 64, "adaptation": "Apply the documented coordinate ordering.",
        "before": "measurement label", "after": "measurement label",
        "metric": {"before": .125, "after": .25},
        "patches": [{"id": "coordinate_order", "before": "OLD_PLACEHOLDER",
                     "after": "NEW_PLACEHOLDER", "reason": "Align state coordinates.",
                     "nested": {"patches": [{"before": 4, "after": 5,
                                               "sourceCode": ["SOURCE_PLACEHOLDER", 7]}]}}],
        "source_code": {"body": "BODY_PLACEHOLDER", "version": 2},
        "nested": ({"patch-code": "TEXT_PLACEHOLDER"},),
    }


def test_nested_source_fields_are_removed_without_editing_original_metadata():
    original = provenance_fixture()
    snapshot = copy.deepcopy(original)
    findings = source_metadata_findings(original)
    assert {item["path"] for item in findings} == {
        "$.patches[0].before", "$.patches[0].after",
        "$.patches[0].nested.patches[0].sourceCode[0]",
        "$.source_code.body", "$.nested[0]['patch-code']",
    }
    assert all(item["reason"] for item in findings)
    json.dumps(findings)
    published = public_metadata(original)
    assert original == snapshot
    assert source_metadata_findings(published) == []
    assert published["before"] == published["after"] == "measurement label"
    assert published["metric"] == {"before": .125, "after": .25}
    assert published["source_code_sha256"] == original["source_code_sha256"]
    assert published["patches"][0]["reason"] == original["patches"][0]["reason"]
    assert published["patches"][0]["nested"]["patches"][0] == {"before": 4, "after": 5, "sourceCode": [7]}
    assert published["source_code"] == {"version": 2}
    assert isinstance(published["nested"], tuple)
    assert public_metadata({"patch_code": b"BYTES_PLACEHOLDER", "after": 7}) == {"after": 7}


def test_state_dict_order_attributes_and_tensor_bits_survive_public_copy(tmp_path):
    model = torch.nn.Linear(3, 2)
    state = model.state_dict()
    state._metadata[""]["source_code"] = "VERSION_METADATA_PLACEHOLDER"
    state._metadata[""]["before"] = 2
    state.patch_code = "STATE_ATTRIBUTE_PLACEHOLDER"
    state.source_code = "SOURCE_ATTRIBUTE_PLACEHOLDER"
    source = OrderedDict([("model_state", state), ("metadata", provenance_fixture())])
    published = public_metadata(source)
    assert isinstance(published, OrderedDict)
    assert isinstance(published["model_state"], OrderedDict)
    assert list(published) == list(source)
    assert list(published["model_state"]) == list(state)
    assert published["model_state"]._metadata[""] == {"version": 1, "before": 2}
    assert "source_code" in state._metadata[""]
    assert state.patch_code == "STATE_ATTRIBUTE_PLACEHOLDER"
    assert state.source_code == "SOURCE_ATTRIBUTE_PLACEHOLDER"
    assert not hasattr(published["model_state"], "patch_code")
    assert not hasattr(published["model_state"], "source_code")
    for key, value in state.items():
        assert published["model_state"][key] is value
        assert torch.equal(published["model_state"][key].view(torch.uint8), value.view(torch.uint8))
    assert any(item["path"].endswith(".@_metadata[''].source_code") for item in source_metadata_findings(source))
    assert source_metadata_findings(published) == []
    path = tmp_path / "PUBLIC_FIXTURE_ONLY.pt"
    torch.save(published, path)
    restored = torch.load(path, weights_only=True)
    model.load_state_dict(restored["model_state"], strict=True)
    for key, value in state.items():
        assert torch.equal(restored["model_state"][key].view(torch.uint8), value.view(torch.uint8))


def test_named_tuples_aliases_and_numeric_patch_values_are_retained():
    Record = namedtuple("Record", "metric metadata")
    shared = {"before": 1, "after": 2}
    value = {"patches": [shared], "records": Record(3, shared)}
    result = public_metadata(value)
    assert type(result["records"]) is Record
    assert result["records"].metric == 3
    assert result["patches"] == [{"before": 1, "after": 2}]
    assert value["patches"][0] is shared


@pytest.mark.parametrize("placeholder", [
    "def placeholder_only(value):\n    return value\n",
    "class PlaceholderOnly:\n    pass\n",
    "import placeholder_only\n",
    "first_placeholder = 1\nsecond_placeholder = 2\n",
    "if placeholder_ready:\n    placeholder_action()\n",
])
def test_renamed_source_fields_are_reported_and_rejected_without_execution(placeholder):
    """Original placeholder syntax demonstrates that renaming keys cannot bypass export checks."""
    value = {"notes": {"details": placeholder}, "metric": .125}
    findings = source_metadata_findings(value)
    assert findings == [{"path": "$.notes.details", "reason": "unregistered Python source text"}]
    with pytest.raises(ValueError, match=r"\$\.notes\.details"):
        public_metadata(value)
    assert value["notes"]["details"] == placeholder and value["metric"] == .125


@pytest.mark.parametrize("metadata", [
    "https://example.org/upstream/tree/recorded-revision", "a" * 64, "FNO2d",
    "Match the token coordinates to the state ordering.", "q(x, y, t)",
    "nu * laplacian_w", "terminal_epoch_not_validation", "training_complete; equation_readout_not_yet_evaluated",
])
def test_prose_references_hashes_names_and_formulas_are_not_source(metadata):
    value = {"notes": metadata, "metric": .125}
    assert source_metadata_findings(value) == []
    assert public_metadata(value) == value


def test_metadata_and_bridge_imports_do_not_load_numerical_frameworks():
    project = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "sys.path[:0] = " + repr([str(project / "src"), str(project)]) + "; "
        "import gift; import adapters; "
        "from gift.publication_metadata import public_metadata, source_metadata_findings; "
        "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
        "sample = {'patches': [{'before': 'placeholder', 'reason': 'layout'}], 'before': 2}; "
        "assert adapters.public_metadata(sample) == public_metadata(sample); "
        "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
        "assert source_metadata_findings({'before': 2, 'after': 3}) == []; "
        "assert gift.__version__ == '1.0.0'"
    )
    subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], cwd=project, check=True)


def test_lazy_gift_public_symbols_remain_the_original_objects():
    import gift
    for module_name, names in gift._MODULES.items():
        module = importlib.import_module("gift." + module_name)
        for name in names:
            assert name in gift.__all__ and name in dir(gift)
            assert getattr(gift, name) is getattr(module, name)
    with pytest.raises(AttributeError):
        getattr(gift, "unknown_public_symbol")


@pytest.mark.parametrize("method", ["anie", "anie_dense151"])
def test_anie_public_terminal_export_keeps_resume_identity_complete(tmp_path, method):
    from training.anie_control import export_terminal
    from training.checkpoints import CheckpointStore
    model = torch.nn.Linear(1, 1)
    config = {"method": method, "epochs": 1, "trajectories": 1, "formal": False}
    history = [{"epoch": 1, "processed_trajectories": 1, "optimizer_updates_this_epoch": 1, "epoch_seconds": 0.0}]
    identity = {"configuration": config, "upstream": provenance_fixture()}
    snapshot = copy.deepcopy(identity)
    store = CheckpointStore(tmp_path / "PRIVATE_JOURNAL_FIXTURE", identity)
    checkpoint = store.save("epoch_0001", {"epoch": 1, "history": history,
                                          "model_state_dict": model.state_dict()})
    checkpoint_bytes = checkpoint.read_bytes()
    export_terminal(store.directory, model, config, {"grid": 64}, identity, history, 1, False)
    exported = torch.load(store.directory / "model.pt", weights_only=True)
    assert source_metadata_findings(exported) == []
    assert identity == snapshot
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert source_metadata_findings(torch.load(checkpoint, weights_only=True)["identity"])
    resumed = CheckpointStore(store.directory, identity, resume=True)
    assert resumed.original_identity == json.loads(json.dumps(identity))
    assert sorted(source_metadata_findings(resumed.original_identity), key=lambda item: item["path"]) == sorted(
        source_metadata_findings(identity), key=lambda item: item["path"])
    assert resumed.payload["epoch"] == 1
    for key, value in model.state_dict().items():
        assert torch.equal(value, exported["model_state_dict"][key])


@pytest.mark.parametrize("family,method", [("native", "anie_ns"), ("native", "fno2d"), ("extended", "unet")])
def test_selected_weights_are_public_but_paused_journal_resumes_with_full_source_identity(
        tmp_path, monkeypatch, family, method):
    from test_terminal_status_recovery import fixture_run, native, extended
    control = native if family == "native" else extended
    config, spec = fixture_run(monkeypatch, control, method)
    config["max_epochs"] = 2
    provenance = provenance_fixture()
    monkeypatch.setattr(control, "_source_provenance", lambda *args: provenance)
    output = tmp_path / ("PRIVATE_TRAINING_FIXTURE_" + method)
    control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu", stop_after_epoch=1)
    paused = torch.load(output / "latest.pt", weights_only=True)
    assert paused["epoch"] == 1 and source_metadata_findings(paused["source_provenance"])
    complete = control.run_training("synthetic", output, model_spec=spec, config=config, device="cpu", resume=True)
    assert complete["epoch"] == 2 and complete["status"] == "complete"
    assert complete["source_provenance"] == provenance
    selected = "best.pt" if method == "anie_ns" else "terminal_weights.pt"
    weights = torch.load(output / selected, weights_only=True)
    assert source_metadata_findings(weights) == []
    assert weights["source_provenance"]["metric"] == provenance["metric"]
    if method != "anie_ns":
        for key, value in complete["model_state"].items():
            assert torch.equal(weights["model_state"][key], value)
