"""Stdlib launcher tests: no training, downloads, popups or parent-env mutation."""
import os
import sys
from types import SimpleNamespace

import pytest

from scripts import run_training as launcher


def test_child_environment_does_not_change_parent():
    """Child defaults must preserve parent settings and apply model-specific thread/device defaults locally."""
    parent = {"PATH": "test-only-path", "CUDA_VISIBLE_DEVICES": "1",
              "pythonpath": "case-variant", "UNRELATED": "preserved"}
    before = parent.copy()
    env = launcher.child_environment("unet", sys.executable, "cuda", parent)
    assert env["UNRELATED"] == "preserved" and parent == before
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    for model in ("fno2d", "fno3d", "uno", "gift_low", "gift_branch", "gift_generator", "gift_predictor"):
        env = launcher.child_environment(model, sys.executable, "cuda", {})
        if model == "uno":
            assert env["OMP_NUM_THREADS"] == "24"
            assert env["MKL_NUM_THREADS"] == "16"
        else:
            assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "16"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        if model.startswith("gift_"):
            assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        else:
            assert "CUBLAS_WORKSPACE_CONFIG" not in env


@pytest.mark.parametrize("model", list(launcher.MODULES))
def test_dispatches_exactly_one_module_without_shell(model, monkeypatch):
    """Each model name must dispatch one explicit Python module with forwarded arguments and shell disabled."""
    calls = []
    def child(command, **kwargs):
        """Capture the child command/options and return success without starting any training process."""
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, "run", child)
    with pytest.raises(SystemExit) as exit_info:
        launcher.main(["--device", "cpu", model, "--dry-run", "--output", "test-only-output"])
    assert exit_info.value.code == 0 and len(calls) == 1
    command, options = calls[0]
    assert command[:4] == [sys.executable, "-B", "-m", launcher.MODULES[model]]
    assert command[4:] == ["--device", "cpu", "--dry-run", "--output", "test-only-output"]
    assert options.get("shell", False) is False


def test_pinn_maps_cuda_to_gpu_and_uses_selected_python(monkeypatch):
    """Translate PINN device naming, preserve selected interpreter and propagate the child exit code."""
    calls = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or SimpleNamespace(returncode=3))
    with pytest.raises(SystemExit) as exit_info:
        launcher.main(["--python", sys.executable, "pinn", "--mode", "open"])
    assert exit_info.value.code == 3
    assert calls[0][0][0][4:6] == ["--device", "gpu"]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[0][1]["env"]["TF_ENABLE_ONEDNN_OPTS"] == "0"
