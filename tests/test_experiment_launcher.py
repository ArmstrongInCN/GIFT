"""Launcher contracts only; no scientific inference or child process is started."""
import sys
from types import SimpleNamespace

import pytest

from scripts import run_experiment as launcher


@pytest.mark.parametrize("experiment", tuple(launcher.MODULES))
def test_one_independent_child_no_shell(experiment, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs:
                        calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    with pytest.raises(SystemExit) as error:
        launcher.main(["--device", "cpu", experiment, "--output", "unit-output", "--resume"])
    assert error.value.code == 0 and len(calls) == 1
    command, options = calls[0]
    assert command == [sys.executable, "-B", "-m", "experiments.formal." + launcher.MODULES[experiment] + ".run",
                       "--device", "cpu", "--output", "unit-output", "--resume"]
    assert options.get("shell", False) is False
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    if experiment != "M1":
        assert str(launcher.ROOT / "src") in options["env"]["PYTHONPATH"]


def test_parent_unchanged_and_recorded_threads():
    parent = {"Path": "ordinary-test-path", "nvidia_tf32_override": "0", "OMP_NUM_THREADS": "99"}
    original = dict(parent)
    for experiment, args, threads in (("M1", ["--method", "GIFT"], "2"),
            ("M1", ["--method=PINN-SR"], "1"), ("M1", ["--method", "PDE-FIND"], "1"), ("M2", [], "2")):
        env = launcher.child_environment(experiment, args, "cuda", parent)
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == threads
        assert "nvidia_tf32_override" not in env and "CUBLAS_WORKSPACE_CONFIG" not in env
        assert env["Path"] == parent["Path"] and parent == original


def test_device_must_precede_name(monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(SystemExit) as error:
        launcher.main(["M2", "--device", "cpu"])
    assert error.value.code != 0
