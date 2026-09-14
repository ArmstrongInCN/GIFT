"""Tiny synthetic terminal weights exercise storage only, not scientific training."""
import io
import json

import pytest
import torch

from training.weight_files import load_weights, read_index, reconstruct, split_weights


def fixture(tmp_path):
    source = tmp_path / "model.pt"
    payload = {"status": "complete", "model_state_dict": {"w": torch.arange(80).reshape(8, 10),
               "complex": torch.complex(torch.arange(10.).float(), torch.ones(10))}}
    torch.save(payload, source)
    index = split_weights(source, tmp_path / "parts", part_bytes=512)
    return source, index, payload


def test_reconstructed_file_and_tensors_exact(tmp_path):
    source, index, original = fixture(tmp_path)
    buffer = io.BytesIO()
    reconstruct(index, buffer)
    assert buffer.getvalue() == source.read_bytes()
    for path in (source, index):
        loaded = load_weights(path)
        assert all(torch.equal(value, loaded["model_state_dict"][key]) for key, value in original["model_state_dict"].items())


def test_existing_output_retained(tmp_path):
    source, index, _ = fixture(tmp_path)
    before = index.read_bytes()
    with pytest.raises(FileExistsError):
        split_weights(source, index.parent)
    assert index.read_bytes() == before


def test_corrupt_part_rejected(tmp_path):
    _, index, _ = fixture(tmp_path)
    part = index.parent / "weights-0001.part"
    value = part.read_bytes()
    part.write_bytes(bytes([value[0] ^ 1]) + value[1:])
    with pytest.raises(ValueError, match="SHA-256"):
        load_weights(index)


@pytest.mark.parametrize("change", ["escape", "duplicate", "total", "order"])
def test_bad_index_rejected(tmp_path, change):
    _, index, _ = fixture(tmp_path)
    doc = json.loads(index.read_text())
    if change == "escape":
        doc["parts"][0]["path"] = "../model.pt"
    elif change == "duplicate":
        doc["parts"][1] = doc["parts"][0]
    elif change == "total":
        doc["bytes"] += 1
    else:
        doc["parts"] = list(reversed(doc["parts"]))
    index.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_weights(index)


def test_refuses_nonmodel_archive(tmp_path):
    source = tmp_path / "data.pt"
    torch.save({"observations": torch.ones(3)}, source)
    with pytest.raises(ValueError, match="terminal"):
        split_weights(source, tmp_path / "parts")
    assert not (tmp_path / "parts").exists()


def test_excess_parts_rejected_before_creation(tmp_path):
    source, _, _ = fixture(tmp_path)
    output = tmp_path / "too_many"
    with pytest.raises(ValueError, match="2048"):
        split_weights(source, output, part_bytes=1)
    assert not output.exists()


@pytest.mark.parametrize("name", ["GIFT_DATA_ROOT", "GIFT_EXTERNAL_ROOT"])
def test_protected_roots_unchanged(tmp_path, monkeypatch, name):
    source, _, _ = fixture(tmp_path)
    protected = tmp_path / "protected"
    protected.mkdir()
    monkeypatch.setenv(name, str(protected))
    with pytest.raises(ValueError, match="roots"):
        split_weights(source, protected / "weights")
    assert list(protected.iterdir()) == []


@pytest.mark.parametrize("doc", [[], {"schema": "gift.sharded-checkpoint.v1", "bytes": 2,
                                      "sha256": None, "parts": []}])
def test_malformed_types_rejected(tmp_path, doc):
    index = tmp_path / "weights.json"
    index.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        read_index(index)
