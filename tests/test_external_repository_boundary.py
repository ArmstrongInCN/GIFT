"""Enforce source placement without changing external numerical interfaces."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from adapters import adapter_root, public_metadata, source_file
from scripts.audit_repository import external_boundary_failures, selected_files


def test_runtime_must_remain_outside_repository(tmp_path, monkeypatch):
    project = tmp_path / "GIFT"
    project.mkdir()
    monkeypatch.setenv("GIFT_EXTERNAL_ADAPTER_ROOT", str(project / "vendor"))
    with pytest.raises(ValueError, match="outside"):
        adapter_root(project)
    monkeypatch.delenv("GIFT_EXTERNAL_ADAPTER_ROOT")
    assert adapter_root(project) == tmp_path / "gift-external-runtime"


def test_source_records_resolve_to_actual_external_implementation(tmp_path, monkeypatch):
    project, runtime = tmp_path / "GIFT", tmp_path / "runtime"
    project.mkdir()
    (runtime / "adapters").mkdir(parents=True)
    implementation = runtime / "adapters" / "models.py"
    implementation.write_text("# User-prepared interface\n", encoding="utf-8")
    (project / "own.py").write_text("value = 3\n", encoding="utf-8")
    monkeypatch.setenv("GIFT_EXTERNAL_ADAPTER_ROOT", str(runtime))
    assert source_file(project, "adapters/models.py") == implementation
    assert source_file(project, "own.py") == project / "own.py"
    with pytest.raises(ValueError, match="relative"):
        source_file(project, "../outside.py")


def test_metadata_removes_only_patch_source_fields():
    original = {"metric": .125, "before": 2., "after": 1.,
                "patches": [{"id": "layout", "before": "original source",
                             "after": "replacement source", "reason": "same tensor labels"}]}
    published = public_metadata(original)
    assert published == {"metric": .125, "before": 2., "after": 1.,
                         "patches": [{"id": "layout", "reason": "same tensor labels"}]}
    assert "before" in original["patches"][0]


def test_audit_rejects_algorithm_files_and_executable_patch_metadata(tmp_path):
    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters/__init__.py").write_text("# Generic interface\n")
    (tmp_path / "adapters/models.py").write_text("# Implementation belongs outside\n")
    (tmp_path / "copied.py").write_text("class FNO2d:\n    pass\n")
    (tmp_path / "patch.json").write_text(json.dumps({"patches": [{"before": "x", "after": "y"}]}))
    failures = external_boundary_failures(tmp_path, ["adapters/__init__.py", "adapters/models.py",
                                                     "copied.py", "patch.json"])
    # Both source-bearing fields are reported with their exact nested paths.
    assert len(failures) == 4
    assert any("algorithm definition" in item for item in failures)
    assert any("patch source text" in item for item in failures)
    assert external_boundary_failures(tmp_path, ["adapters/__init__.py"]) == []


def test_bridge_and_metadata_import_do_not_load_torch():
    project = Path(__file__).resolve().parents[1]
    if not (adapter_root(project) / "adapters/anie_config.py").is_file():
        pytest.skip("User-prepared external metadata interface is required")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    code = ("import sys; import adapters; from adapters.anie_config import ANIEConfig; "
            "assert 'torch' not in sys.modules; assert ANIEConfig().frames == 151")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=project, env=environment, check=True)


def test_working_tree_selection_omits_pending_deletions_but_index_mode_keeps_them(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    module = tmp_path / "moved.py"
    module.write_text("# Before relocation\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "moved.py"], check=True)
    module.rename(tmp_path.parent / (tmp_path.name + "_external.py"))
    assert selected_files(tmp_path) == []
    assert selected_files(tmp_path, tracked=True) == ["moved.py"]


def test_catalog_reads_direct_published_numeric_evidence(tmp_path, monkeypatch):
    from scripts import audit_repository as audit
    package = tmp_path / "results/formal/W1"
    package.mkdir(parents=True)
    copied = {"path": "figures/source_fields.npz", "bytes": 17, "sha256": "a" * 64}
    (package / "published.json").write_text(json.dumps({
        "schema": "gift.published-results.v1", "experiment": "W1", "files": [copied],
    }))
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    assert audit.figure_source_allowlist() == {"results/formal/W1/" + copied["path"]: copied}
