"""Publishing may omit dependency patch text, but must retain measured values."""
import copy
import json

import pytest

from scripts.publish_w1_w2_single import portable
from scripts import publish_original_style_figures as original
from adapters import public_metadata


def metadata_fixture():
    """Placeholder provenance contains no downloaded algorithm implementation."""
    return {
        "summary": {"mean_relative_l2": .125, "failures": 0, "before": 4, "after": 5},
        "source": {"repository": "https://example.org/upstream", "commit": "fixed-revision",
                   "patches": [{"id": "coordinate_order", "file": "external-source.py",
                                "before": "old-placeholder", "after": "new-placeholder",
                                "reason": "Match token coordinates to state order."}]},
    }


def test_portable_keeps_measurements_and_textual_patch_provenance():
    payload = metadata_fixture()
    before = copy.deepcopy(payload)
    result = portable(payload)
    assert payload == before
    assert result["summary"] == payload["summary"]
    assert result["source"]["repository"] == payload["source"]["repository"]
    assert result["source"]["patches"] == [{
        "id": "coordinate_order", "reason": "Match token coordinates to state order."}]


def test_json_copy_omits_only_patch_code_and_preserves_numeric_csv_bytes(tmp_path):
    publisher = original
    source, destination = tmp_path / "source.json", tmp_path / "destination.json"
    payload = metadata_fixture()
    source.write_text(json.dumps(payload), encoding="utf-8")
    publisher.copy_once(source, destination)
    cleaned = json.loads(destination.read_text(encoding="utf-8"))
    assert cleaned["summary"] == payload["summary"]
    assert cleaned["source"]["patches"] == [{
        key: value for key, value in payload["source"]["patches"][0].items()
        if key not in {"before", "after"}}]
    assert json.loads(source.read_text(encoding="utf-8")) == payload
    publisher.copy_once(source, destination)
    csv_source, csv_destination = tmp_path / "source.csv", tmp_path / "destination.csv"
    csv_source.write_bytes(b"method,error\r\nmethodA,0.125\r\n")
    publisher.copy_once(csv_source, csv_destination)
    assert csv_source.read_bytes() == csv_destination.read_bytes()


def test_public_metadata_omits_only_registered_patch_fields(tmp_path):
    source, destination = tmp_path / "source.json", tmp_path / "destination.json"
    source.write_text(json.dumps(metadata_fixture()), encoding="utf-8")
    original.copy_once(source, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == public_metadata(metadata_fixture())
    cleaned = json.loads(destination.read_text(encoding="utf-8"))
    cleaned["summary"]["mean_relative_l2"] = .25
    destination.write_text(json.dumps(cleaned), encoding="utf-8")
    assert cleaned != public_metadata(metadata_fixture())
    with pytest.raises(FileExistsError):
        original.copy_once(source, destination)
