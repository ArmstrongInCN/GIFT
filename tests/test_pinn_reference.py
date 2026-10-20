"""PINN coefficient readers use only completed JSON results."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from adapters import pinn_reference as reference


@pytest.fixture
def record(tmp_path):
    """Supply a minimal completed JSON writer with valid sparse coefficient arithmetic."""
    def write(mode="known"):
        """Write known/open ordered coefficients with the first term masked out."""
        names = reference.expected_library(mode)
        raw = np.arange(1, len(names) + 1, dtype=np.float32) / 100
        mask = np.ones(len(names), dtype=np.float32)
        mask[0] = 0
        value = dict(mode=mode, condition="noise_000", training_completed=True,
                     diagnostic_test_only=False, library=names, raw_coefficients=raw.tolist(),
                     mask=mask.tolist(), effective_coefficients=(raw * mask).tolist())
        path = tmp_path / "result.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, value
    return write


@pytest.mark.parametrize("mode", ["known", "open"])
def test_json_readout_does_not_need_resume_state(record, mode):
    """Read signed/averaged parameters from JSON alone without an optimizer archive."""
    path, value = record(mode)
    result = reference.read_checkpoint(path, "noise_000", mode)
    coefficients = dict(zip(value["library"], value["effective_coefficients"]))
    expected = (dict(nu=coefficients["laplacian"], beta=-coefficients["advection"], gamma=coefficients["q"])
                if mode == "known" else
                dict(nu=(coefficients["w_{xx}"] + coefficients["w_{yy}"]) / 2,
                     beta=-(coefficients["u**1w_{x}"] + coefficients["v**1w_{y}"]) / 2,
                     gamma=coefficients["q"]))
    assert result["parameters"] == expected
    assert not result["training"] and not result["forward"]
    assert not (path.parent / "terminal_state.npz").exists()


def test_legacy_result_metadata_still_reads(record):
    """Preserve descriptive old input/archive metadata without requiring its files."""
    path, value = record()
    condition = value.pop("condition")
    value["terminal_snapshot"] = {"identity": {"backend": {"inputs": {
        "condition": condition, "data": {"sha256": "old_declared_data_hash"},
        "sampling": {"sha256": "old_declared_sampling_hash"}}}}}
    value["terminal_state_sha256"] = "old_declared_archive_hash"
    path.write_text(json.dumps(value))
    result = reference.read_checkpoint(path, condition, "known")
    assert result["source_data_protocol"]["data_sha256"] == "old_declared_data_hash"
    assert result["archive_sha256"] == "old_declared_archive_hash"


@pytest.mark.parametrize("defect", ["incomplete", "diagnostic", "condition", "library", "nonfinite", "mask", "effective", "shape"])
def test_invalid_coefficient_contract_rejected(record, defect):
    """Reject incomplete, mismatched or numerically inconsistent coefficient readouts."""
    path, value = record()
    if defect == "incomplete": value["training_completed"] = False
    elif defect == "diagnostic": value["diagnostic_test_only"] = True
    elif defect == "condition": value["condition"] = "noise_005"
    elif defect == "library": value["library"].reverse()
    elif defect == "nonfinite": value["raw_coefficients"][1] = float("nan")
    elif defect == "mask": value["mask"][0] = .5
    elif defect == "effective": value["effective_coefficients"][0] = 1
    else: value["mask"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        reference.read_checkpoint(path, "noise_000", "known")


def test_no_tensorflow_or_torch_import(record):
    """Read result coefficients in a fresh process without either model framework."""
    path, _ = record()
    root = str(Path(__file__).resolve().parents[1])
    code = ("import sys; sys.path.insert(0, %r); from adapters.pinn_reference import read_checkpoint; "
            "read_checkpoint(%r, 'noise_000', 'known'); "
            "assert 'tensorflow' not in sys.modules and 'torch' not in sys.modules") % (root, str(path))
    subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True, timeout=30)
