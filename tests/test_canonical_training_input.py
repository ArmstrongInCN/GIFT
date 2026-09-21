"""Sparse metadata fixtures test input gates; none are simulation trajectories."""

import json

import h5py
import numpy as np
import pytest

from gift.canonical_package import training_input
from gift.data_splits import prediction_split_manifest
from training.checkpoints import digest_file


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "canonical"
    (root / "fno").mkdir(parents=True)
    relative = "fno/fno1000_n64_t0_t10_dt0p02.h5"
    path = root / relative
    fields = {}
    with h5py.File(path, "x") as handle:
        handle.attrs["trajectory_id_scheme"] = "canonical"
        for name, ids in (("training", np.arange(1000)), ("validation", np.arange(1000, 1020))):
            shape = (len(ids), 501, 64, 64)
            handle.create_dataset(f"{name}/trajectory_index", data=ids.astype(np.int64))
            handle.create_dataset(f"{name}/time", data=np.arange(501) * .02)
            # Unwritten NaN fields deliberately fail full scientific validation.
            handle.create_dataset(f"{name}/vorticity", shape=shape, dtype="f4", fillvalue=np.nan,
                                  chunks=(1, 1, 64, 64))
            # The decoded field hash is a placeholder; only the metadata contract
            # is checked, never the (unwritten NaN) scientific contents.
            fields[f"/{name}/vorticity"] = {"shape": list(shape), "dtype": "float32", "decoded_sha256": "0" * 64}
    (root / "splits.json").write_text(json.dumps(prediction_split_manifest()), encoding="utf-8")
    (root / "schema.json").write_text(json.dumps({"schema": "gift.observation-schema.v1", "files": {relative: fields}}), encoding="utf-8")
    manifest = {"schema": "gift.data-package-manifest.v1", "data_profile": "canonical",
                "prediction_observations_lossless": True, "files": []}
    for name in (relative, "splits.json", "schema.json"):
        file = root / name
        manifest["files"].append({"path": name, "bytes": file.stat().st_size, "sha256": digest_file(file)})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, path, manifest


def test_read_only_metadata_check_does_not_claim_scientific_acceptance(package):
    root, path, _ = package
    result = training_input(root, path, full=False)
    assert result["profile"] == "canonical"
    assert not result["full_input_checks_performed"]
    assert not result["numerical_reproduction_verified"]


def test_full_check_rejects_unwritten_observations(package):
    root, path, _ = package
    with pytest.raises(ValueError, match="nonfinite"):
        training_input(root, path, full=True)


@pytest.mark.parametrize("change", ["profile", "duplicate", "escape", "split_hash", "missing"])
def test_package_integrity_rejections(package, change):
    root, path, manifest = package
    if change == "profile":
        manifest["data_profile"] = "released"
    elif change == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif change == "escape":
        manifest["files"][0]["path"] = "../outside.h5"
    elif change == "split_hash":
        manifest["files"][1]["sha256"] = "0" * 64
    else:
        manifest["files"].pop()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        training_input(root, path, full=False)


def test_formal_baseline_requires_explicit_canonical_profile():
    from training.baseline_control import _arguments, configuration
    for name in ("fno2d", "fno3d", "uno", "unet"):
        config = configuration(name, _arguments(name, ["--data-profile", "canonical"]))
        assert config["data_profile"] == "canonical"
        assert config["epochs"] == 500 and config["trajectories"] == 1000
