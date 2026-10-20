"""Optional U-Net launch defaults do not restrict normal CPU/module use."""
import os
from unittest.mock import Mock
import pytest


def test_standard_parser_accepts_cpu_and_help(capsys, monkeypatch):
    """U-Net CPU planning/help must use argparse without changing environment or requiring an import-order gate."""
    from training.baseline_control import _arguments
    environment = {"PYTHONWARNINGS": "error", "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "4"}
    before = dict(environment)
    monkeypatch.setattr(os, "environ", environment)
    assert _arguments("unet", ["--dry-run", "--device", "cpu"]).device == "cpu"
    assert environment == before
    with pytest.raises(SystemExit) as stopped:
        _arguments("unet", ["--help"])
    assert stopped.value.code == 0
    assert "--epochs" in capsys.readouterr().out


def test_entry_dispatches_without_import_order_license(monkeypatch):
    """The U-Net entry must forward arguments directly to the shared baseline controller."""
    from training import baseline_control, train_unet
    train = Mock()
    monkeypatch.setattr(baseline_control, "main", train)
    train_unet.main(["--dry-run", "--device", "cpu"])
    train.assert_called_once_with("unet", ["--dry-run", "--device", "cpu"])


def test_complete_parser_rejects_long_option_abbreviations():
    """Reject an ambiguous abbreviated long option instead of silently enabling tiny training."""
    from training.baseline_control import _arguments
    with pytest.raises(SystemExit):
        _arguments("unet", ["--tin"])
