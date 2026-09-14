"""Small synthetic packages test validation logic, not experimental reproduction."""
import json

import h5py
import numpy as np
import pytest

from scripts.verify_data import inspect_arrays, package_path, sha256, verify


def build_package(root, *, nonfinite=False):
    root.mkdir()
    for name in ("README.md", "DATA_DICTIONARY.md", "LICENSE.txt", "splits.json"):
        (root / name).write_text("{}", encoding="utf-8")
    with h5py.File(root / "data.h5", "w") as handle:
        values = np.ones((1, 2, 2, 2), dtype=np.float32)
        if nonfinite:
            values[0, 0, 0, 0] = np.nan
        handle.create_dataset("test/vorticity", data=values)
    schema = {"files": {"data.h5": inspect_arrays(root / "data.h5")}}
    (root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    entries = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in root.iterdir()]
    manifest = {"schema": "gift.data-package-manifest.v1", "files": entries,
                "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_is_read_only(tmp_path):
    root = tmp_path / "package"
    build_package(root)
    before = {p.name: sha256(p) for p in root.iterdir()}
    result = verify(root, full_array_scan=True)
    assert result["all_numeric_values_finite_checked"]
    assert not result["experiment_results_verified"]
    assert before == {p.name: sha256(p) for p in root.iterdir()}


def test_hash_mismatch(tmp_path):
    root = tmp_path / "package"
    build_package(root)
    (root / "README.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify(root)


def test_unlisted_file(tmp_path):
    root = tmp_path / "package"
    build_package(root)
    (root / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="Unlisted"):
        verify(root)


def test_finite_scan_is_explicit(tmp_path):
    root = tmp_path / "package"
    build_package(root, nonfinite=True)
    assert not verify(root)["all_numeric_values_finite_checked"]
    with pytest.raises(ValueError, match="Nonfinite"):
        verify(root, full_array_scan=True)


@pytest.mark.parametrize("relative", ["../outside", "/outside", "C:/outside", "a\\b", "a/../b", "a//b", ""])
def test_path_rejection(tmp_path, relative):
    with pytest.raises(ValueError):
        package_path(tmp_path, relative)


def test_external_h5_link_rejected(tmp_path):
    path = tmp_path / "linked.h5"
    with h5py.File(path, "w") as handle:
        handle["data"] = h5py.ExternalLink("other.h5", "/data")
    with pytest.raises(ValueError, match="links"):
        inspect_arrays(path)
