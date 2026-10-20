"""Relocation must retain exact weights, locked recipes and strict file inventory."""
import copy
import json
import shutil

import pytest

from experiments.model_locations import generator_checkpoint, runtime_model_spec
from scripts import package_current_models as bundle


def test_runtime_source_location_does_not_mutate_recorded_scientific_spec(tmp_path, monkeypatch):
    """A user root changes construction location and leaves all recorded settings intact."""
    source = tmp_path / "upstreams"
    source.mkdir()
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", str(source))
    spec = {"method": "unet", "source_root": "old-machine", "history": 1, "query_times": [0., .04]}
    original = copy.deepcopy(spec)
    actual = runtime_model_spec(spec)
    assert actual == dict(spec, source_root=str(source)) and spec == original


def test_gift_sibling_is_authoritative_and_missing_pair_is_an_error(tmp_path):
    """A moved pair never silently switches to another recorded generator."""
    branch = tmp_path / "branch/model.pt"
    branch.parent.mkdir()
    branch.write_bytes(b"immutable branch fixture")
    sibling = tmp_path / "generator/model.pt"
    sibling.parent.mkdir()
    sibling.write_bytes(b"paired generator fixture")
    payload = {"model_spec": {"generator_checkpoint": "unavailable-original-machine.pt"}}
    assert generator_checkpoint(branch, payload) == sibling
    missing_branch = tmp_path / "other/branch/model.pt"
    missing_branch.parent.mkdir(parents=True)
    missing_branch.write_bytes(b"fixture without generator")
    with pytest.raises(FileNotFoundError, match="paired GIFT generator"):
        generator_checkpoint(missing_branch, payload)


def model_bundle(root):
    """Build byte fixtures for packaging checks, with no inference accuracy claim."""
    root.mkdir()
    models = {}
    for method in bundle.METHODS:
        path = root / "models" / method / ("branch/model.pt" if method.startswith("GIFT") else "best.pt")
        path.parent.mkdir(parents=True)
        path.write_bytes((method + " frozen weight bytes").encode())
        models[method] = path.relative_to(root).as_posix()
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "base_lock.json").write_text(json.dumps({"protocol_file": "old-protocol"}))
    (metadata / "extension_lock.json").write_text(json.dumps({"reused_models": {}, "training_configurations": {}}))
    manifest = dict(schema=bundle.SCHEMA, models=models,
        observations_included=False, external_algorithm_source_included=False,
        complete_training_resume_state_included=False,
        files=[bundle.record(path, root) for path in root.rglob("*") if path.is_file()])
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("change", ["corrupt_weight", "extra_file", "escape", "missing_role", "duplicate"])
def test_bundle_rejects_corruption_and_undeclared_files_before_activation(tmp_path, change):
    """A transferable folder is accepted only when all declared bytes and roles agree."""
    root = tmp_path / "bundle"
    manifest = model_bundle(root)
    assert bundle.verify_bundle(root)["schema"] == bundle.SCHEMA
    if change == "corrupt_weight":
        (root / manifest["models"]["GIFT"]).write_bytes(b"changed")
    elif change == "extra_file":
        (root / "unexpected.pt").write_bytes(b"unbound")
    elif change == "escape":
        manifest["files"][0]["path"] = "../outside.pt"
    elif change == "missing_role":
        manifest["models"].pop("anie_ns")
    else:
        manifest["files"].append(manifest["files"][0])
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        bundle.verify_bundle(root)


def test_bundle_can_move_without_weight_or_recipe_rewriting(tmp_path, monkeypatch):
    """Activation derives paths separately; immutable bundle bytes survive relocation."""
    from experiments.additional import w1_extended_evaluate
    original = tmp_path / "download"
    model_bundle(original)
    moved = tmp_path / "other-machine/download"
    shutil.copytree(original, moved)
    before = {p.relative_to(moved): p.read_bytes() for p in moved.rglob("*") if p.is_file()}
    calls = []
    monkeypatch.setattr(w1_extended_evaluate, "checkpoint_metadata", lambda label, path, base, extension:
        calls.append((label, path, extension["reused_models"])))
    monkeypatch.setattr(bundle, "validate_gift_pairs", lambda models: None)
    output = tmp_path / "runtime"
    bundle.activate_bundle(moved, output)
    receipt = json.loads((output / "MODELS.json").read_text())
    assert set(receipt["checkpoints"]) == set(bundle.METHODS) and len(calls) == 7
    assert all(str(moved) in path for path in receipt["checkpoints"].values())
    assert before == {p.relative_to(moved): p.read_bytes() for p in moved.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        bundle.activate_bundle(moved, output)
