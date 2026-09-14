"""NumPy-only input-gate tests; tiny fixtures never certify scientific data."""
import json

import numpy as np
import pytest

from training import pinn_data as inputs
from scripts import assemble_generated_data as assembly
from scripts import generate_data as gen


@pytest.fixture
def inventory(tmp_path):
    base = tmp_path / "collection"
    base.mkdir()
    (base / "metadata.txt").write_text("unit gate only", encoding="utf-8")
    allowed = set(assembly.CLEAN_PATHS) | set(assembly.SAMPLING) | {"noise"}
    manifest = dict(schema="gift.generated-data-collection.v1", data_profile="regenerated",
        status="ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED", scientific_arrays_modified=False,
        released_truth_used_to_fill_gaps=False, full_hash_and_finite_checks_performed=True,
        assembler_sha256=gen.sha256(assembly.__file__), available_jobs=["standard"],
        missing_jobs=sorted(allowed - {"standard"}), files=[dict(path="metadata.txt", category="unit_fixture",
            bytes=(base / "metadata.txt").stat().st_size, sha256=gen.sha256(base / "metadata.txt"))])
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base, manifest


def test_inventory_is_not_scientific_acceptance(inventory):
    base, manifest = inventory
    actual, files, digest = inputs._inventory(base)
    assert actual == manifest
    assert set(files) == {"metadata.txt"}
    assert digest == gen.sha256(base / "manifest.json")
    assert not actual.get("training_profile_integration_verified", False)


@pytest.mark.parametrize("change", ["status", "profile", "assembler", "fill", "scan", "duplicate", "escape",
                                  "size", "sha", "job_overlap", "job_missing"])
def test_inventory_rejects_invalid_declarations(inventory, change):
    base, manifest = inventory
    if change == "status":
        manifest["status"] = "COMPLETE_PILOT"
    elif change == "profile":
        manifest["data_profile"] = "released"
    elif change == "assembler":
        manifest["assembler_sha256"] = "0" * 64
    elif change == "fill":
        manifest["released_truth_used_to_fill_gaps"] = True
    elif change == "scan":
        manifest["full_hash_and_finite_checks_performed"] = False
    elif change == "duplicate":
        manifest["files"].append(dict(manifest["files"][0]))
    elif change == "escape":
        manifest["files"][0]["path"] = "../metadata.txt"
    elif change == "size":
        manifest["files"][0]["bytes"] += 1
    elif change == "sha":
        manifest["files"][0]["sha256"] = None
    elif change == "job_overlap":
        manifest["missing_jobs"].append("standard")
    else:
        manifest["missing_jobs"].pop()
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        inputs._inventory(base)


def test_unknown_condition_rejected_before_io(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        inputs.verify_regenerated_inputs(tmp_path, tmp_path, tmp_path, "noise_999")


@pytest.mark.parametrize("limits", [(None, None), (1, 2), (2, 3)])
def test_selected_arrays_preserve_order_and_boundaries(tmp_path, monkeypatch, limits):
    # Mock only provenance; this test isolates IO ordering, not scientific acceptance.
    monkeypatch.setattr(inputs, "verify_regenerated_inputs", lambda *a: {"unit_fixture_only": True})
    source = tmp_path / "sampling.npz"
    coordinates = np.arange(9, dtype=np.float32).reshape(3, 3)
    targets = coordinates + np.float32(20)
    lhs = coordinates + np.float32(40)
    train = np.array([2, 0])
    np.savez(source, measurement_coordinates=coordinates, targets=targets, train_indices=train,
             lhs_coordinates=lhs, lower=np.zeros(3, np.float32), upper=np.ones(3, np.float32))
    n, p = limits
    actual = inputs.read_regenerated_inputs(source, tmp_path, tmp_path, "noise_000", n, p)
    assert np.array_equal(actual[0], coordinates[train][:n])
    assert np.array_equal(actual[1], targets[train][:n])
    assert np.array_equal(actual[2], np.concatenate([lhs, coordinates[train]])[:p])
    assert actual[-1]["selection"] == dict(train_rows=len(actual[0]), physics_rows=len(actual[2]),
                                           diagnostic_subset=n is not None or p is not None)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, 10])
def test_invalid_row_limits_rejected(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(inputs, "verify_regenerated_inputs", lambda *a: {})
    source = tmp_path / "sampling.npz"
    data = np.ones((2, 3), np.float32)
    np.savez(source, measurement_coordinates=data, targets=data, train_indices=np.arange(2),
             lhs_coordinates=data, lower=np.zeros(3, np.float32), upper=np.ones(3, np.float32))
    with pytest.raises(ValueError, match="row limit"):
        inputs.read_regenerated_inputs(source, tmp_path, tmp_path, "noise_000", invalid)
