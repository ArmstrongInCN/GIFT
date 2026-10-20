"""Baseline export tests using small CPU checkpoints."""
import copy
import json
import os
import subprocess
import sys

import pytest
import torch

from scripts import recover_baseline_export as recovery


def checkpoint():
    """Build a two-epoch completed UNO fixture with two finite CPU weight values."""
    config = {"model": "uno", "epochs": 2, "formal": False}
    saved = {
        "schema": "gift.training-boundary.v1",
        "identity": {"configuration": config, "sources": {"model.py": "old"}, "data": {}},
        "runtime": {"python": "older"}, "boundary": "epoch_0002",
        "payload": {
            "status": "complete", "completed_epochs": 2,
            "history": [{"epoch": 1}, {"epoch": 2}],
            "model_state_dict": {"weight": torch.tensor([1., 2.])},
            "normalization": {"kind": "none"},
        },
    }
    return saved


def test_plan_does_not_import_models_or_change_environment():
    """Run the empty export plan in a fresh interpreter without reading files or importing Torch."""
    code = """
import contextlib, io, json, os, sys
before = dict(os.environ)
from scripts import recover_baseline_export as recovery
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    assert recovery.main([]) == 0
assert json.loads(stream.getvalue())["files_read"] == 0
assert "torch" not in sys.modules
assert dict(os.environ) == before
"""
    run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


def test_terminal_metadata_uses_actual_budget():
    """Accept the saved two-epoch budget and reject incomplete status/progress/history."""
    saved = checkpoint()
    assert recovery.validate_metadata(saved)["epochs"] == 2
    for change in ("status", "epochs", "history"):
        invalid = copy.deepcopy(saved)
        if change == "status":
            invalid["payload"]["status"] = "running"
        elif change == "epochs":
            invalid["payload"]["completed_epochs"] = 1
        else:
            invalid["payload"]["history"].pop()
        with pytest.raises(ValueError):
            recovery.validate_metadata(invalid)


def test_export_without_attempt_or_hashes(tmp_path):
    """Export complete weights independently of a training directory or optional checksums.

    The source and process environment stay unchanged; the export contains
    model metadata but omits optimizer state and cannot replace its own output.
    """
    saved = checkpoint()
    source, target = tmp_path / "checkpoint.pt", tmp_path / "export/model.pt"
    torch.save(saved, source)
    before = source.read_bytes()
    environment = dict(os.environ)
    assert recovery.main(["--execute", "--checkpoint", str(source), "--output", str(target)]) == 0
    exported = torch.load(target, map_location="cpu", weights_only=True)
    assert exported["terminal_epoch"] == 2
    assert torch.equal(exported["model_state_dict"]["weight"], saved["payload"]["model_state_dict"]["weight"])
    assert "optimizer_state_dict" not in exported
    assert exported["fresh_training"] is None
    assert source.read_bytes() == before and dict(os.environ) == environment
    with pytest.raises(ValueError, match="new weights"):
        recovery.main(["--execute", "--checkpoint", str(source), "--output", str(target)])


def test_optional_checksum_rejects_wrong_archive(tmp_path):
    """Honor a supplied checkpoint checksum when the caller explicitly requests it."""
    source = tmp_path / "checkpoint.pt"
    torch.save(checkpoint(), source)
    with pytest.raises(ValueError, match="SHA"):
        recovery.main(["--execute", "--checkpoint", str(source), "--checkpoint-sha256", "0" * 64,
                       "--output", str(tmp_path / "model.pt")])


def test_nonfinite_weights_rejected(tmp_path):
    """Prevent a completed checkpoint containing NaN model weights from being exported."""
    saved = checkpoint()
    saved["payload"]["model_state_dict"]["weight"][0] = float("nan")
    source = tmp_path / "checkpoint.pt"
    torch.save(saved, source)
    with pytest.raises(ValueError, match="model tensors"):
        recovery.main(["--execute", "--checkpoint", str(source), "--output", str(tmp_path / "model.pt")])


def test_atomic_creation_preserves_existing_and_failed_files(tmp_path):
    """Keep existing weights intact and leave a failed staging file for inspection."""
    target = tmp_path / "weights.pt"
    recovery.atomic_create(target, lambda stream: stream.write(b"weights"), lambda path: None)
    with pytest.raises(ValueError):
        recovery.atomic_create(target, lambda stream: stream.write(b"other"), lambda path: None)
    assert target.read_bytes() == b"weights"
    failed = tmp_path / "failed.pt"

    def reject(path):
        """Simulate verification failure after writing the candidate staging file."""
        raise ValueError("verification interrupted")

    with pytest.raises(ValueError):
        recovery.atomic_create(failed, lambda stream: stream.write(b"partial"), reject)
    assert not failed.exists()
    assert len(list(tmp_path.glob(".failed.pt.recovery-*.tmp"))) == 1
