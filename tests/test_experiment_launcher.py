"""Launcher contracts only; no scientific inference or child process is started."""
import sys
from types import SimpleNamespace

import pytest

from scripts import run_experiment as launcher


@pytest.mark.parametrize("experiment", tuple(launcher.MODULES))
def test_one_independent_child_no_shell(experiment, monkeypatch):
    """Forward arguments to exactly one explicit Python module with CPU selection."""
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs:
                        calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    with pytest.raises(SystemExit) as error:
        launcher.main(["--device", "cpu", experiment, "--output", "unit-output", "--resume"])
    assert error.value.code == 0 and len(calls) == 1
    command, options = calls[0]
    module = launcher.CURRENT_MODULES.get(experiment, "experiments.formal." + launcher.MODULES[experiment] + ".run")
    assert command == [sys.executable, "-B", "-m", module,
                       "--device", "cpu", "--output", "unit-output", "--resume"]
    assert options.get("shell", False) is False
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    if experiment != "W3":
        assert str(launcher.ROOT / "src") in options["env"]["PYTHONPATH"]


def test_parent_environment_and_visible_gpu_are_preserved():
    """Set child defaults without changing parent options or visible-GPU mappings."""
    parent = {"Path": "ordinary-test-path", "nvidia_tf32_override": "0", "OMP_NUM_THREADS": "99",
              "CUDA_VISIBLE_DEVICES": "2,3", "CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    original = dict(parent)
    for experiment, args, default_threads in (("W3", ["--method", "GIFT"], "2"),
            ("W3", ["--method=PINN-SR"], "1"), ("W3", ["--method", "PDE-FIND"], "1"), ("W1", [], "2")):
        env = launcher.child_environment(experiment, args, "cuda", parent)
        assert env["OMP_NUM_THREADS"] == "99" and env["MKL_NUM_THREADS"] == default_threads
        assert env["nvidia_tf32_override"] == "0" and env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert env["Path"] == parent["Path"] and parent == original
        assert launcher.child_environment(experiment, args, "cpu", parent)["CUDA_VISIBLE_DEVICES"] == "-1"


def test_device_must_precede_name(monkeypatch):
    """Reject misplaced launcher device options before any child dispatch."""
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(SystemExit) as error:
        launcher.main(["W1", "--device", "cpu"])
    assert error.value.code != 0
