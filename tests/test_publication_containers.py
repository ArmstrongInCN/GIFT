"""Publication checks inspect bytes even when an attacker refreshes all hashes.

Fixtures contain synthetic placeholder metadata, never upstream source code.
The unsafe-pickle fixture would create a marker only if an unsafe load occurred.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from gift.publication_files import audit_public_file, audit_public_files
from scripts.audit_repository import external_boundary_failures
from training.weight_files import split_weights


def _checkpoint(path, *, source=False):
    value = {"model_state_dict": {"weight": torch.arange(8)},
             "metrics": {"before": 2., "after": 1., "relative_l2": 1e-7},
             "source_provenance": {"repository": "https://example.invalid/upstream",
                                   "source_code_hash": "f" * 64}}
    if source:
        value["source_provenance"]["upstream"] = {
            "patches": [{"before": "placeholder original", "after": "placeholder replacement"}]}
    torch.save(value, path)


def test_real_checkpoint_metadata_not_manifest_claim_controls_audit(tmp_path):
    """A newly hashed checkpoint still fails if its actual nested metadata has code."""
    path = tmp_path / "best.pt"
    _checkpoint(path, source=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"contains_upstream_code": False, "files": [
        {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    failures = audit_public_files([path, manifest])
    assert any("$.source_provenance.upstream.patches[0].before" in item for item in failures)
    assert any("$.source_provenance.upstream.patches[0].after" in item for item in failures)


def test_numeric_metadata_and_state_dict_are_allowed(tmp_path):
    """Numbers, tensor state and provenance hashes are not copied source text."""
    path = tmp_path / "clean.pt"
    _checkpoint(path)
    assert audit_public_file(path) == []
    path = tmp_path / "clean.json"
    path.write_text(json.dumps({"patches": [{"before": 3, "after": 2}],
                                "source_code_hash": "a" * 64}))
    assert audit_public_file(path) == []


@pytest.mark.parametrize("source", [False, True])
def test_shard_index_reconstructs_actual_safe_metadata(tmp_path, source):
    """Part hashes protect reconstruction; the reconstructed metadata is also audited."""
    original = tmp_path / "original.pt"
    _checkpoint(original, source=source)
    index = split_weights(original, tmp_path / "shards", part_bytes=512)
    failures = audit_public_files([index.parent])
    assert bool(failures) == source
    if source:
        assert any("patches[0].before" in item for item in failures)
    else:
        assert audit_public_file(index) == []
    part = next(index.parent.glob("*.part"))
    assert any("requires" in item for item in audit_public_file(part))


@pytest.mark.parametrize("hidden", ["checkpoint", "json", "python", "nested_zip", "traversal"])
def test_zip_members_cannot_hide_source_or_unsafe_paths(tmp_path, hidden):
    """Release archives are audited recursively without trusting their inventories."""
    path = tmp_path / "bundle.zip"
    buffer = io.BytesIO()
    _checkpoint(buffer, source=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"contains_upstream_code": False}))
        if hidden == "checkpoint":
            archive.writestr("models/anie/best.pt", buffer.getvalue())
        elif hidden == "json":
            archive.writestr("metadata.json", json.dumps({"nested": {"source_code": "placeholder"}}))
        elif hidden == "python":
            archive.writestr("metadata/hidden.py", "# synthetic source placeholder\n")
        elif hidden == "nested_zip":
            inner = io.BytesIO()
            with zipfile.ZipFile(inner, "w") as nested:
                nested.writestr("best.pt", buffer.getvalue())
            archive.writestr("nested.zip", inner.getvalue())
        else:
            archive.writestr("../outside.json", "{}")
    failures = audit_public_file(path)
    assert failures
    assert any("!" in item or "Unsafe" in item for item in failures)
    assert not (tmp_path.parent / "outside.json").exists()


def test_zip_shard_group_is_checked_without_extraction(tmp_path):
    original = tmp_path / "original.pt"
    _checkpoint(original, source=True)
    index = split_weights(original, tmp_path / "shards", part_bytes=512)
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for member in index.parent.iterdir():
            archive.write(member, "models/" + member.name)
    assert any("patches[0].before" in item for item in audit_public_file(path))


@pytest.mark.parametrize("hidden", ["extra.py", "extra.json"])
def test_checkpoint_zip_extra_members_are_not_ignored(tmp_path, hidden):
    """PyTorch's ordinary payload loader may ignore extras; publication must not."""
    path = tmp_path / "clean.pt"
    _checkpoint(path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("clean/" + hidden, json.dumps({"source_code": "placeholder"}))
    assert audit_public_file(path)


@pytest.mark.parametrize("names", [("same.pt", "SAME.PT"), ("extra.json.",),
                                   ("extra.json ",), ("folder:payload.json",)])
def test_zip_windows_path_aliases_are_rejected(tmp_path, names):
    path = tmp_path / "aliases.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "{}")
    assert any("Unsafe" in item for item in audit_public_file(path))


def test_zip_magic_is_checked_even_with_unrecognized_extension(tmp_path):
    path = tmp_path / "hidden.dat"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("payload.json", json.dumps({"source_code": "placeholder"}))
    assert any("source_code" in item for item in audit_public_file(path))


def test_checkpoint_zip_extra_nested_archive_is_checked(tmp_path):
    path = tmp_path / "clean.pt"
    _checkpoint(path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as nested:
        nested.writestr("metadata.json", json.dumps({"source_code": "placeholder"}))
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("clean/extra.dat", buffer.getvalue())
    assert any("source_code" in item for item in audit_public_file(path))


@pytest.mark.parametrize("kind", ["numeric", "text_metadata", "object"])
def test_numpy_metadata_and_object_safety(tmp_path, kind):
    path = tmp_path / "arrays.npz"
    if kind == "numeric":
        np.savez(path, before=np.arange(4), after=np.ones(4), failed=np.asarray([False]))
    elif kind == "text_metadata":
        np.savez(path, metadata=np.asarray(json.dumps({"source_code": "placeholder"})))
    else:
        np.savez(path, opaque=np.asarray([{"source_code": "placeholder"}], dtype=object))
    failures = audit_public_file(path)
    assert bool(failures) == (kind != "numeric")


def test_plain_numpy_array_can_be_audited(tmp_path):
    path = tmp_path / "array.npy"
    np.save(path, np.arange(4))
    assert audit_public_file(path) == []


def test_duplicate_numpy_members_do_not_hide_metadata(tmp_path):
    path = tmp_path / "duplicate.npz"
    buffer = io.BytesIO()
    np.save(buffer, np.arange(4))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("metadata.npy", buffer.getvalue())
            archive.writestr("metadata.npy", buffer.getvalue())
    assert audit_public_file(path)


class _UnsafeFixture:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return os.mkdir, (self.marker,)


def test_unknown_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "must_not_exist"
    path = tmp_path / "unsafe.pt"
    torch.save({"unexpected": _UnsafeFixture(marker)}, path)
    failures = audit_public_file(path)
    assert any("Unsafe or unreadable checkpoint rejected" in item for item in failures)
    assert not marker.exists()


def test_repository_boundary_reads_checkpoint_metadata(tmp_path):
    path = tmp_path / "catalogued.pt"
    _checkpoint(path, source=True)
    assert any("patches[0].before" in item
               for item in external_boundary_failures(tmp_path, [path.name]))


def test_external_container_cli_checks_content_without_a_repository(tmp_path, capsys):
    from scripts import audit_repository

    path = tmp_path / "external.json"
    path.write_text(json.dumps({"source_code": "placeholder"}))
    assert audit_repository.main(["--public-only", "--public-file", str(path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["files_checked"] == 0
    assert any("source_code" in item for item in result["failures"])


def test_optional_history_report_uses_existing_local_objects_only(monkeypatch):
    from scripts import audit_repository

    def local_tree(command, **kwargs):
        assert command[3] == "ls-tree"
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        return b"adapters/__init__.py\0adapters/models.py\0src/gift/model.py\0"
    monkeypatch.setattr(audit_repository.subprocess, "check_output", local_tree)
    result = audit_repository.history_path_risks(Path("example"), "old-tag")
    assert result["status"] == "RISK_PRESENT"
    assert result["paths"] == ["adapters/models.py"]
    assert result["fetch_performed"] is False
    assert result["history_modified"] is False


def test_requested_history_risk_makes_cli_fail(tmp_path, monkeypatch, capsys):
    from scripts import audit_repository

    path = tmp_path / "clean.json"
    path.write_text("{}")
    monkeypatch.setattr(audit_repository, "history_path_risks", lambda root, ref: {
        "reference": ref, "status": "RISK_PRESENT", "paths": ["adapters/models.py"]})
    assert audit_repository.main(["--public-only", "--public-file", str(path),
                                  "--history-risk-ref", "old-ref"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert "known-path check only" in result["failures"][0]


def test_repository_reference_checkpoint_containers_are_clean():
    """The actual catalogued reference weights should not be rejected by the new gate."""
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "artifacts/CHECKPOINTS.json").read_text(encoding="utf-8"))
    paths = [root / row["path"] for row in catalog["files"]
             if row["artifact_role"] in {"trained_weights", "trained_weights_index"}]
    assert paths
    assert audit_public_files(paths) == []
