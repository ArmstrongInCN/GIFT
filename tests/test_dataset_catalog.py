"""Portable role resolution and packaged-case boundaries on synthetic files."""
import json
import shutil
from argparse import Namespace

import pytest

from gift.paths import data_path, dataset_catalog
from experiments.formal._shared.common import ProjectPaths
from scripts import run_w3_parameter_job as job
from scripts.verify_data import formal_catalog, verify_formal_inputs, verify_formal_splits


def catalog(root, roles, **fields):
    """Write a small routing catalogue, without claiming scientific observations."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.json").write_text(json.dumps({
        "schema": "gift.dataset.v1", "roles": roles, **fields}), encoding="utf-8")


def test_roles_remain_portable_and_absent_role_does_not_use_old_file(tmp_path):
    """A relocated role resolves locally; an incomplete catalogue cannot hide missing inputs."""
    root = tmp_path / "download"
    catalog(root, {"four_vortex.standard_n64": "prediction/standard.h5"})
    target = root / "prediction/standard.h5"
    target.parent.mkdir()
    target.write_bytes(b"routing fixture")
    assert data_path(root, "four_vortex.standard_n64", "old.h5") == target
    (root / "old.h5").write_bytes(b"wrong population")
    with pytest.raises(ValueError, match="Missing dataset input role"):
        data_path(root, "s3.lite", "old.h5")
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    assert data_path(copied, "four_vortex.standard_n64") == copied / "prediction/standard.h5"


def test_legacy_layout_constructs_unused_paths_without_requiring_files(tmp_path):
    """Independent generated roots retain their existing no-catalogue contract."""
    assert dataset_catalog(tmp_path) is None
    assert data_path(tmp_path, "four_vortex.standard_n64", "old.h5") == tmp_path / "old.h5"


@pytest.mark.parametrize("relative", ["../outside", "/outside", "C:/outside", "a\\b", "a//b", ""])
def test_role_paths_cannot_escape_package(tmp_path, relative):
    """Reject ambiguous or host-specific paths before accessing another directory."""
    catalog(tmp_path, {"s3.lite": relative})
    with pytest.raises(ValueError):
        data_path(tmp_path, "s3.lite")


def test_malformed_catalogue_does_not_fall_back(tmp_path):
    """An existing incompatible catalogue must be repaired explicitly."""
    (tmp_path / "dataset.json").write_text('{"schema":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="catalogue"):
        data_path(tmp_path, "s3.lite", "old.h5")


def test_project_paths_use_all_six_reference_roles(tmp_path, monkeypatch):
    """Reference experiments resolve role files without imposing the historical directory names."""
    inputs = tmp_path / "inputs"
    names = ("standard_n64", "dense_n64", "cross_resolution", "fno_training",
             "fno_training_dt0p02", "fno_test_dt0p02")
    catalog(inputs, {"four_vortex." + name: name + ".h5" for name in names})
    for name in names:
        (inputs / (name + ".h5")).write_bytes(b"path fixture")
    monkeypatch.setenv("GIFT_DATA_ROOT", str(inputs))
    paths = ProjectPaths.from_root(tmp_path)
    assert all(getattr(paths, name) == inputs / (name + ".h5") for name in names)


def test_formal_catalogue_rejects_incomplete_metadata_and_unlisted_role(tmp_path):
    """Formal verification requires the whole declared package rather than an S3-only root."""
    catalog(tmp_path, {"s3.lite": "missing.h5"})
    with pytest.raises(ValueError, match="roles"):
        formal_catalog(tmp_path, [])


def test_source_archive_is_excluded_from_numeric_scan(tmp_path, monkeypatch):
    """Historical source bytes stay hash-bound without a finite-value assumption."""
    from scripts import verify_data
    paths = ["source_data/old.npz", "prediction/input.h5"]
    entries = [{"path": path} for path in paths]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"scan routing fixture")
    scanned = []
    monkeypatch.setattr(verify_data, "inspect_arrays", lambda path, **kw: scanned.append(path) or {})
    monkeypatch.setattr(verify_data, "formal_observations", lambda *args: None)
    monkeypatch.setattr(verify_data, "verify_formal_splits", lambda *args: None)
    verify_formal_inputs(tmp_path, {}, entries, {
        "schema": "gift.observation-schema.v1", "files": {paths[1]: {}}}, full=True)
    assert scanned == [tmp_path / paths[1]]


def test_split_metadata_is_checked_against_hdf_ids_and_absolute_times(tmp_path):
    """A manifest-valid but scientifically incorrect split document must fail."""
    import h5py
    import numpy as np
    inputs = tmp_path / "input.h5"
    with h5py.File(inputs, "w") as handle:
        group = handle.create_group("test")
        group.create_dataset("trajectory_index", data=np.array([1040, 1041], dtype=np.int64))
        group.create_dataset("time", data=np.array([0., .04]))
        group.create_dataset("absolute_time", data=np.array([5., 5.04]))
        group.create_dataset("vorticity", data=np.zeros((2, 2, 4, 4), dtype=np.float32))
    splits = {"schema": "gift.formal-dataset-splits.v1", "files": {"input.h5": {"test": {
        "trajectory_ids": [1040, 1041], "vorticity_shape": [2, 2, 4, 4],
        "time_start": 5., "time_stop": 5.04, "time_spacing": .04, "time_count": 2}}}}
    target = tmp_path / "splits.json"
    target.write_text(json.dumps(splits), encoding="utf-8")
    verify_formal_splits(tmp_path, {"splits_file": target.name}, {inputs.name})
    splits["files"]["input.h5"]["test"]["trajectory_ids"] = [1040, 1042]
    target.write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(ValueError, match="split identities"):
        verify_formal_splits(tmp_path, {"splits_file": target.name}, {inputs.name})


def test_future_queue_plans_use_selected_catalogue_without_editing_prior_locks(tmp_path, monkeypatch):
    """A new plan uses the reorganized input roles; existing run locks are not opened."""
    from scripts import run_w1_single, run_w1_w2_single, run_w1_v2
    inputs = tmp_path / "inputs"
    roles = {"four_vortex.fno_test_dt0p02": "test.h5",
             "four_vortex.fno_training_dt0p02": "training.h5",
             "w1_w2.confirmation_n64": "confirmation.h5"}
    catalog(inputs, roles)
    for name in roles.values():
        (inputs / name).write_bytes(b"future plan fixture")
    monkeypatch.setenv("GIFT_DATA_ROOT", str(inputs))
    plans = [module.default_plan(tmp_path / "future")
             for module in (run_w1_single, run_w1_w2_single, run_w1_v2)]
    assert all(plan["old_test"] == str(inputs / "test.h5") for plan in plans)
    assert plans[1]["source_training"] == str(inputs / "training.h5")
    assert plans[1]["n64_confirmation"] == str(inputs / "confirmation.h5")


def test_packaged_w3_case_reads_new_input_root_and_preserves_run_root(tmp_path):
    """Original absolute receipt paths are descriptive after inputs move to a package."""
    from test_w3_parameter_job import fixture_case
    original = tmp_path / "original"
    fixture_case(original)
    package = tmp_path / "package"
    catalog(package, {"w3.cases": "identification/w3"})
    shutil.copytree(original / "data", package / "identification/w3")
    run = tmp_path / "fresh_run"
    run.mkdir()
    data_root, dataset, _, _ = job.case_input(run, "parameter_000", package)
    assert data_root == package / "identification/w3/parameter_000"
    assert dataset.parent == data_root and list(run.iterdir()) == []


@pytest.mark.parametrize("action", ["measure", "prepare-sampling"])
def test_w3_packaged_inputs_cannot_be_run_output(tmp_path, action):
    """Reject output placement inside the package before any receipt or model is read."""
    with pytest.raises(ValueError, match="outside the input package"):
        job.execute(Namespace(root=tmp_path, dataset_root=tmp_path, action=action))
