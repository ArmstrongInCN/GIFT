"""Catalog binding for lossless model parts, with no Git or training invocation."""
import json

import pytest
import torch

from scripts import audit_repository as audit
from training.weight_files import read_index, split_weights


def package(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    torch.save({"status": "complete", "model_state_dict": {"x": torch.ones(16)}}, source)
    root = tmp_path / "project"
    index_path = split_weights(source, root / "artifacts/uno", part_bytes=512)
    index = read_index(index_path)
    relative = "artifacts/uno/weights.json"
    records = [{"path": relative, "artifact_role": "trained_weights_index",
                "bytes": index_path.stat().st_size, "sha256": audit.digest(index_path)}]
    for part in index["parts"]:
        records.append({**part, "path": "artifacts/uno/" + part["path"],
                        "artifact_role": "trained_weights_part", "checkpoint_index": relative})
    catalog = root / "artifacts/CHECKPOINTS.json"
    catalog.write_text(json.dumps({"files": records}))
    monkeypatch.setattr(audit, "ROOT", root)
    return catalog, records


def test_complete_part_catalog(tmp_path, monkeypatch):
    _, records = package(tmp_path, monkeypatch)
    assert set(audit.numeric_checkpoint_allowlist()) == {r["path"] for r in records}


@pytest.mark.parametrize("change", ["missing_part", "wrong_binding", "orphan_part"])
def test_inconsistent_part_catalog(tmp_path, monkeypatch, change):
    catalog, records = package(tmp_path, monkeypatch)
    if change == "missing_part":
        records.pop()
    elif change == "wrong_binding":
        records[1]["checkpoint_index"] = "wrong.json"
    else:
        records = records[1:]
    catalog.write_text(json.dumps({"files": records}))
    with pytest.raises(ValueError):
        audit.numeric_checkpoint_allowlist()
