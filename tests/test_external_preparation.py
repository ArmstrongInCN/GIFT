"""Read-only existing-checkout and path guards; no network operations."""
import hashlib
import json

import pytest

from scripts import prepare_external as prepare


@pytest.fixture
def checkouts(tmp_path, monkeypatch):
    project, external = tmp_path/"project", tmp_path/"external"
    project.mkdir()
    source = external/"test_source"
    source.mkdir(parents=True)
    raw = b"TEST ONLY: no third-party implementation\n"
    (source/"source.txt").write_bytes(raw)
    registry = {"sources": {"sample": {"directory": "test_source", "repository": "unused-test-url",
        "commit": "test_commit", "files": {"source.txt": {"sha256": hashlib.sha256(raw).hexdigest().upper()}}}}}
    (project/"external_sources.json").write_text(json.dumps(registry))
    monkeypatch.setattr(prepare, "ROOT", project)
    monkeypatch.delenv("GIFT_DATA_ROOT", raising=False)
    calls = []
    def read_git(directory, *arguments):
        calls.append(arguments)
        assert arguments == ("rev-parse", "HEAD"), "existing checkout must never be modified"
        return "test_commit"
    monkeypatch.setattr(prepare, "git", read_git)
    return project, external, source, calls


def test_existing_sources_verified_without_any_writes(checkouts):
    _, external, source, calls = checkouts
    before = (source/"source.txt").read_bytes()
    result = prepare.prepare(external)
    assert result[0]["verified"] and result[0]["existing_directory_unchanged"]
    assert (source/"source.txt").read_bytes() == before
    assert calls == [("rev-parse", "HEAD")]


def test_changed_source_is_rejected_not_repaired(checkouts):
    _, external, source, _ = checkouts
    (source/"source.txt").write_bytes(b"changed test-only source")
    with pytest.raises(ValueError, match="bytes differ"):
        prepare.prepare(external)
    assert (source/"source.txt").read_bytes() == b"changed test-only source"


def test_project_data_and_missing_roots_guarded(checkouts, tmp_path, monkeypatch):
    project, _, _, calls = checkouts
    data = tmp_path/"data"
    data.mkdir()
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    for root in (project, project/"vendor", data/"external"):
        with pytest.raises(ValueError, match="outside"):
            prepare.prepare(root)
    with pytest.raises(FileNotFoundError):
        prepare.prepare(tmp_path/"missing", verify_only=True)
    assert not (tmp_path/"missing").exists() and not calls
