"""B7 startup contracts only: no models, dataset reads, GPU or training.

Fresh children stop at an import sentinel instead of loading the baseline
runtime. Parser/controller AST probes exercise only the guarded entry boundary.
"""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from training import unet_environment as gate


ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import importlib.abc
import json
import os
import runpy
import sys

class StopAtRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "training.baseline_control":
            print("DOWNSTREAM_SENTINEL_REACHED", flush=True)
            raise SystemExit(73)
        if fullname.split(".", 1)[0] in {"numpy", "torch", "h5py"}:
            print("SCIENTIFIC_IMPORT_ATTEMPT", flush=True)
            raise SystemExit(74)

before = dict(os.environ)
sys.meta_path.insert(0, StopAtRuntime())
sys.argv = ["training.train_unet"] + sys.argv[1:]
try:
    runpy.run_module("training.train_unet", run_name="__main__")
finally:
    print(json.dumps({"environment_unchanged": before == dict(os.environ),
                      "scientific_modules_loaded": any(name.split(".", 1)[0] in
                         {"numpy", "torch", "h5py"} for name in sys.modules)}), flush=True)
'''


def _environment():
    """Configure only the test child; the parent environment is never modified."""
    controlled = set(gate.ABSENT_ENVIRONMENT) | set(gate.FIXED_ENVIRONMENT)
    result = {key: value for key, value in os.environ.items() if key.upper() not in controlled}
    result.update(gate.FIXED_ENVIRONMENT)
    return result


def _child(arguments, environment):
    result = subprocess.run([sys.executable, "-s", "-B", "-c", CHILD, *arguments],
                            env=environment, cwd=ROOT, capture_output=True, text=True, timeout=30)
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt == {"environment_unchanged": True, "scientific_modules_loaded": False}
    assert "SCIENTIFIC_IMPORT_ATTEMPT" not in result.stdout
    return result


def test_exact_environment_is_read_only_and_ignores_unrelated_values():
    environment = dict(gate.FIXED_ENVIRONMENT, UNRELATED_PRIVATE_VALUE="not-for-output")
    before = dict(environment)
    gate.validate_environment(environment)
    assert environment == before


def test_each_fixed_key_requires_exact_value_and_canonical_spelling():
    for key in gate.FIXED_ENVIRONMENT:
        for mode in ("missing", "wrong", "variant_only", "variant_duplicate"):
            environment = dict(gate.FIXED_ENVIRONMENT)
            if mode == "missing":
                environment.pop(key)
            elif mode == "wrong":
                environment[key] = "do-not-print-this-value"
            elif mode == "variant_only":
                environment[key.lower()] = environment.pop(key)
            else:
                environment[key.lower()] = environment[key]
            before = dict(environment)
            with pytest.raises(ValueError) as caught:
                gate.validate_environment(environment)
            assert key in str(caught.value)
            assert "do-not-print-this-value" not in str(caught.value)
            assert environment == before


def test_each_absent_key_rejects_presence_empty_and_case_variant():
    for key in gate.ABSENT_ENVIRONMENT:
        for value in ("", "0", "1", "do-not-print-this-value"):
            for spelling in (key, key.lower()):
                environment = dict(gate.FIXED_ENVIRONMENT, **{spelling: value})
                with pytest.raises(ValueError) as caught:
                    gate.validate_environment(environment)
                assert key in str(caught.value)
                assert "do-not-print-this-value" not in str(caught.value)


@pytest.mark.parametrize("value", ["", "0", "1"])
def test_formal_override_rejected_before_runtime_import(value):
    environment = _environment()
    environment["NVIDIA_TF32_OVERRIDE"] = value
    result = _child(["--dry-run"], environment)
    assert result.returncode == 1
    assert "NVIDIA_TF32_OVERRIDE" in result.stderr
    assert "DOWNSTREAM_SENTINEL_REACHED" not in result.stdout


def test_formal_wrong_thread_profile_rejected_before_runtime_import():
    environment = _environment()
    environment["OMP_NUM_THREADS"] = "16"
    result = _child(["--run-training", "--output", "unused-not-created"], environment)
    assert result.returncode == 1 and "OMP_NUM_THREADS" in result.stderr
    assert "DOWNSTREAM_SENTINEL_REACHED" not in result.stdout


def test_valid_formal_gate_reaches_only_sentinel_not_scientific_execution():
    result = _child(["--dry-run"], _environment())
    assert result.returncode == 73
    assert "DOWNSTREAM_SENTINEL_REACHED" in result.stdout


def test_help_works_without_formal_environment_or_runtime():
    environment = _environment()
    environment["NVIDIA_TF32_OVERRIDE"] = "private-value-not-for-output"
    result = _child(["--help"], environment)
    assert result.returncode == 0 and "Independent U-Net" in result.stdout
    assert "DOWNSTREAM_SENTINEL_REACHED" not in result.stdout
    assert "private-value-not-for-output" not in result.stdout + result.stderr


def test_explicit_tiny_cpu_can_reach_nonformal_sentinel():
    environment = _environment()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = _child(["--tiny", "--device", "cpu", "--dry-run"], environment)
    assert result.returncode == 73 and "DOWNSTREAM_SENTINEL_REACHED" in result.stdout


def test_abbreviated_tiny_does_not_bypass_formal_gate():
    environment = _environment()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = _child(["--tin", "--dry-run"], environment)
    assert result.returncode == 1 and "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "DOWNSTREAM_SENTINEL_REACHED" not in result.stdout


def _controller_function(name, namespace):
    """Compile only an entry function, not its scientific imports or operations."""
    parsed = ast.parse((ROOT / "training/baseline_control.py").read_text(encoding="utf-8"))
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<entry-contract-only>", "exec"), namespace)
    return namespace[name]


def test_complete_parser_rejects_long_option_abbreviations():
    parse = _controller_function("_arguments", {"argparse": argparse, "Path": Path, "TRAIN_FILE": "fixture-only"})
    with contextlib.redirect_stderr(io.StringIO()):
        for arguments in (["--tin"], ["--tiny", "--tiny-ep", "1"]):
            with pytest.raises(SystemExit) as caught:
                parse("unet", arguments)
            assert caught.value.code == 2
    assert parse("unet", ["--tiny", "--device", "cpu"]).tiny is True


def test_direct_formal_controller_cannot_bypass_preimport_gate():
    namespace = {"_arguments": lambda *args: None,
                 "configuration": lambda *args: {"formal": True},
                 "_data_path": mock.Mock(side_effect=AssertionError("Input access must not occur"))}
    main = _controller_function("main", namespace)
    with mock.patch.object(gate, "_PREIMPORT_VALIDATED", False):
        with pytest.raises(ValueError, match="pre-import"):
            main("unet", [])
    namespace["_data_path"].assert_not_called()


def test_environment_change_after_preflight_is_rejected():
    environment = _environment()
    environment["MKL_NUM_THREADS"] = "16"
    with mock.patch.object(gate, "_PREIMPORT_VALIDATED", True), mock.patch.object(gate.os, "environ", environment):
        with pytest.raises(ValueError, match="MKL_NUM_THREADS"):
            gate.require_validated_startup()
