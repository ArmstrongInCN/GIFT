"""CPU input-gate tests only: fixtures are NOT generated scientific trajectories.

The collection's success case tests metadata/provenance plumbing with an
explicitly mocked scientific validator; it never qualifies a training input.
The unmocked assembler finite-field rejection is tested separately.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import assemble_generated_data as assembly
from scripts import generate_data as gen
from training import baseline_control as control


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def collection(tmp_path):
    base = tmp_path / "collection"
    base.mkdir()
    provenance = base / "provenance" / "fno-training"
    provenance.mkdir(parents=True)
    plan = gen.make_plan(argparse.Namespace(dataset="standard", initial_conditions=None,
        subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))
    plan.update(dataset="fno-training", initial_conditions_file_sha256="1" * 64,
                extra950_origin="specified_initial_conditions_seed_unknown")
    # These are deliberately not valid full FNO scientific groups.
    binding = {"plan": plan, "runtime": {"test_fixture_only": True}}
    binding_hash = hashlib.sha256(gen.canonical(binding).encode()).hexdigest()
    run = dict(attempt_id="MOCK_NOT_SCIENCE", binding=binding, binding_sha256=binding_hash)
    data = base / control.TRAIN_FILE
    data.parent.mkdir()
    with h5py.File(data, "x") as handle:
        handle.attrs.update(fixture_only=True, purpose="MOCK_NOT_SCIENCE")
        handle.create_dataset("training/vorticity", data=np.ones((1, 47, 64, 64), dtype=np.float32))
    write_json(provenance / "run.json", run)
    write_json(provenance / "COMPLETE.json", dict(status="COMPLETE_GENERATION",
        attempt_id=run["attempt_id"], binding_sha256=binding_hash, data_sha256=gen.sha256(data)))
    write_json(base / "splits.json", {"schema": "gift.generated-splits.v1", "files": {control.TRAIN_FILE: {}}})
    manifest = dict(schema="gift.generated-data-collection.v1", data_profile="regenerated",
        status="ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED", scientific_arrays_modified=False,
        released_truth_used_to_fill_gaps=False, full_hash_and_finite_checks_performed=True,
        assembler_sha256=gen.sha256(assembly.__file__), available_jobs=["fno-training"],
        missing_jobs=sorted((set(assembly.CLEAN_PATHS) | set(assembly.SAMPLING) | {"noise"}) - {"fno-training"}),
        files=[])
    for relative, category in ((control.TRAIN_FILE, "generated_scientific_input"),
            ("splits.json", "split_metadata"), ("provenance/fno-training/run.json", "generation_provenance"),
            ("provenance/fno-training/COMPLETE.json", "generation_provenance")):
        file = base / relative
        manifest["files"].append(dict(path=relative, category=category, bytes=file.stat().st_size,
            sha256=gen.sha256(file), generation_job="fno-training", attempt_id=run["attempt_id"]))
    write_json(base / "manifest.json", manifest)
    return base, data, manifest


def test_profile_defaults_and_explicit_switch():
    for model in control.PROTOCOLS:
        assert control.configuration(model, control._arguments(model, []))["data_profile"] == "released"
        assert control.configuration(model, control._arguments(model,
            ["--data-profile", "regenerated"]))["data_profile"] == "regenerated"


def test_schema_plan_binding_success_is_explicitly_not_scientific_acceptance(collection, monkeypatch):
    base, data, manifest = collection
    calls = []
    def metadata_boundary(attempt, *, full, data_path):
        calls.append((attempt["name"], full, data_path))
        return [(data_path, control.TRAIN_FILE, gen.sha256(data_path), {})]
    monkeypatch.setattr(assembly, "validate_clean", metadata_boundary)
    result = control._regenerated_input(base, data, full=False)
    assert calls == [("fno-training", False, data)]
    assert result["manifest_sha256"] == gen.sha256(base / "manifest.json")
    assert result["consumed_files"][control.TRAIN_FILE] == gen.sha256(data)
    assert result["generation_sources"] == gen.make_plan(argparse.Namespace(dataset="standard",
        initial_conditions=None, subset=None, split="all", pilot_steps=None,
        batch_size=None, device="cpu"))["sources"]
    assert result["profile"] == "regenerated"
    assert not result["full_input_checks_performed"]
    assert not result["published_dataset_identity"]
    assert not result["numerical_reproduction_verified"]


@pytest.mark.parametrize("change", ["status", "assembler", "duplicate", "escape", "missing_receipt", "parent_fill", "hash"])
def test_collection_gate_rejections(collection, change):
    base, data, manifest = collection
    if change == "status":
        manifest["status"] = "COMPLETE_PILOT"
    elif change == "assembler":
        manifest["assembler_sha256"] = "0" * 64
    elif change == "duplicate":
        manifest["files"].append(dict(manifest["files"][0]))
    elif change == "escape":
        manifest["files"][0]["path"] = "../outside.h5"
    elif change == "missing_receipt":
        manifest["files"] = manifest["files"][:-1]
    elif change == "parent_fill":
        manifest["released_truth_used_to_fill_gaps"] = True
    else:
        manifest["files"][1]["sha256"] = "0" * 64
    write_json(base / "manifest.json", manifest)
    with pytest.raises(ValueError):
        control._regenerated_input(base, data, full=False)


def test_mock_cannot_qualify_even_with_consistent_receipts(collection):
    base, data, _ = collection
    with pytest.raises(ValueError, match="mock/pilot"):
        control._regenerated_input(base, data, full=True)


@pytest.mark.parametrize("change", ["solver_source", "pde", "receipt_binding"])
def test_consistent_manifest_cannot_hide_wrong_generation_protocol(collection, change):
    base, data, manifest = collection
    run_path = base / "provenance/fno-training/run.json"
    receipt_path = base / "provenance/fno-training/COMPLETE.json"
    run, receipt = json.loads(run_path.read_text()), json.loads(receipt_path.read_text())
    if change == "solver_source":
        run["binding"]["plan"]["sources"].pop("src/even_full_spectrum_ns.py")
    elif change == "pde":
        run["binding"]["plan"]["physical_protocol"]["viscosity"] = .02
    else:
        receipt["attempt_id"] = "different_attempt"
    run["binding_sha256"] = hashlib.sha256(gen.canonical(run["binding"]).encode()).hexdigest()
    receipt["binding_sha256"] = run["binding_sha256"]
    write_json(run_path, run)
    write_json(receipt_path, receipt)
    for record in manifest["files"]:
        file = base / record["path"]
        record.update(bytes=file.stat().st_size, sha256=gen.sha256(file))
    write_json(base / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source/PDE|Physical/numerical|identity mismatch"):
        control._regenerated_input(base, data, full=False)


def test_real_science_validator_is_not_bypassed_by_metadata_manifest(collection):
    base, data, _ = collection
    with pytest.raises(ValueError, match="groups"):
        control._regenerated_input(base, data, full=False)


def test_raw_h5_cannot_select_regenerated_without_collection(collection):
    base, data, _ = collection
    config = control.configuration("uno", control._arguments("uno", ["--tiny", "--data-profile", "regenerated"]))
    with pytest.raises(ValueError, match="collection root"):
        control._validate_data(data, config, hash_bytes=False)


def test_released_formal_hash_gate_unchanged_and_tiny_not_released(tmp_path):
    path = tmp_path / "metadata_only.h5"
    with h5py.File(path, "x") as handle:
        handle.create_dataset("training/vorticity", shape=(1000, 501, 64, 64), dtype="f4",
                              chunks=(1, 1, 64, 64), fillvalue=np.nan)
        handle.create_dataset("training/time", data=np.arange(501) * .02)
        handle.create_dataset("training/trajectory_index", data=np.r_[np.arange(50), np.arange(1200, 2150)])
    formal = control.configuration("uno", control._arguments("uno", []))
    with pytest.raises(ValueError, match="formal dataset SHA256 differs"):
        control._validate_data(path, formal, hash_bytes=True)
    tiny = control.configuration("uno", control._arguments("uno", ["--tiny"]))
    result = control._validate_data(path, tiny, hash_bytes=True)
    assert result["provenance"]["profile"] == "nonformal_test"
    assert not result["provenance"]["published_dataset_identity"]


@pytest.mark.parametrize("change", ["unwritten", "ids", "time", "parameters"])
def test_collection_path_override_keeps_scientific_guards(tmp_path, change):
    plan = gen.make_plan(argparse.Namespace(dataset="standard", initial_conditions=None,
        subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))
    path = tmp_path / "collection_file.h5"
    with h5py.File(path, "x") as handle:
        handle.attrs.update(status="COMPLETE_GENERATION", attempt_id="fixture", binding_sha256="unit",
                            generated_from="initial_conditions_not_saved_truth")
        for info in plan["groups"]:
            group = handle.create_group(info["name"])
            group.create_dataset("trajectory_index", data=np.asarray(info["ids"], dtype=np.int64))
            group.create_dataset("time", data=gen.stored_times(plan, info))
            group.create_dataset("initial_condition_parameters", data=np.asarray(info["parameters"], dtype=np.float64))
            group.create_dataset("vorticity", shape=(len(info["ids"]), len(info["steps"]), 64, 64),
                                 dtype="f4", chunks=(1, 1, 64, 64), fillvalue=np.nan)
        if change == "ids":
            handle["training/trajectory_index"][0] = -1
        elif change == "time":
            handle["training/time"][1] = .01
        elif change == "parameters":
            handle["training/initial_condition_parameters"][0, 0, 0] = 99.
    attempt = dict(name="standard", root=tmp_path / "nonexistent_attempt_data", scientific=plan,
        run=dict(attempt_id="fixture", binding_sha256="unit"), receipt=dict(data_sha256=gen.sha256(path)))
    if change == "unwritten":
        assert assembly.validate_clean(attempt, full=False, data_path=path)[0][0] == path
        with pytest.raises(ValueError, match="Nonfinite/unwritten"):
            assembly.validate_clean(attempt, full=True, data_path=path)
    else:
        with pytest.raises(ValueError, match="identity/time|initial parameters"):
            assembly.validate_clean(attempt, full=False, data_path=path)
