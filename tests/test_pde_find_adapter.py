"""Small external-source equivalence tests, never full M1 scientific acceptance.

Set GIFT_EXTERNAL_ROOT for upstream tests. Maintainers may additionally set
GIFT_LEGACY_PROJECT_ROOT to their read-only historical audit copy; no local
absolute path or old implementation is bundled in this test.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from adapters import pde_find as pde
from experiments.formal.m1_equation_identification import run


@pytest.fixture
def upstream():
    if not os.environ.get("GIFT_EXTERNAL_ROOT"):
        pytest.skip("external pinned PDE-FIND checkout not configured")
    return pde.load_upstream()


@pytest.fixture
def small_rows():
    rng = np.random.RandomState(314159)
    return {name: rng.normal(size=(128, 1))
            for name in ("w", "u", "v", "q", "wt", "wx", "wy", "wxx", "wxy", "wyy")}


@pytest.mark.parametrize("known", [False, True])
def test_external_solver_sampling_and_coefficients_exact(upstream, small_rows, known):
    theta, names = pde.library(upstream, small_rows, known=known)
    assert theta.dtype == np.complex64
    assert theta.shape == (128, 4 if known else 90)
    before = np.random.get_state()
    observed = pde.regress(upstream, theta, small_rows["wt"])
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    expected_indices = np.random.RandomState(0).choice(128, int(128 * .8), replace=False)
    np.testing.assert_array_equal(upstream.np.random.last_indices, expected_indices)
    # The same external function bodies, without our local RNG/membership proxy.
    reference = pde.load_upstream()
    reference.__dict__["np"] = np
    try:
        expected = pde.regress(reference, theta, small_rows["wt"])
    finally:
        np.random.set_state(before)
    np.testing.assert_array_equal(observed, expected)
    assert set(pde.parameter_readout(names, observed, known=known)) == {"nu", "beta", "gamma"}
    assert Path(upstream.TrainSTRidge.__code__.co_filename).name == "PDE_FIND.py"


def test_external_polynomial_derivative_small_reference(upstream):
    values = np.random.RandomState(4).normal(size=9)
    positions = np.arange(9, dtype=np.float64) * .02
    for order in (1, 2):
        expected = upstream.PolyDiffPoint(values, positions, deg=5, diff=order)[order - 1]
        observed = values @ pde.derivative_weights(9, order, .02)
        assert abs(expected - observed) <= 1e-10


def load_legacy_oracle():
    configured = os.environ.get("GIFT_LEGACY_PROJECT_ROOT")
    if not configured:
        pytest.skip("optional historical read-only oracle not configured")
    root = Path(configured).resolve(strict=True)
    path = root / "benchmarks/pde_find/locked_runtime/run_pdefind_batch.py"
    spec = importlib.util.spec_from_file_location("readonly_pdefind_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Read only one project-owned data transformation, not legacy optimizer code.
    native = root / "experiments/formal/m1_equation_identification/benchmark_sources/pdefind_native.py"
    parsed = ast.parse(native.read_text(encoding="utf-8-sig"))
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                and node.name == "truncated_svd_field"]
    assert len(selected) == 1
    namespace = {"np": np, "Any": object, "sha256_array": pde.array_digest}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(native), "exec"), namespace)
    return module, namespace["truncated_svd_field"]


def test_all_library_columns_match_historical_oracle(upstream, small_rows):
    oracle, _ = load_legacy_oracle()
    for known in (False, True):
        expected, expected_names = (oracle.build_kc_theta(small_rows) if known else
                                    oracle.build_open_theta(small_rows))
        observed, names = pde.library(upstream, small_rows, known=known)
        assert names == expected_names
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("condition", pde.CONDITIONS)
def test_project_data_derivatives_and_noise_svd_match_historical_oracle(condition):
    oracle, svd = load_legacy_oracle()
    raw = np.random.RandomState(27).normal(size=(101, 64, 64))
    times = np.arange(101) * .02
    observed, protocol = pde.prepare_rows(raw, times, condition)
    u, v = oracle.velocity_from_vorticity(raw)
    state = raw
    if condition != "noise_000":
        state, _ = svd(raw, 26)
        u, _ = svd(u, 20)
        v, _ = svd(v, 20)
    oracle.velocity_from_vorticity = lambda _: (u, v)
    expected, expected_protocol = oracle.collect_rows(state, times)
    assert protocol["time_indices"] == expected_protocol["time_indices"]
    assert observed.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(observed[name], expected[name], err_msg=name)


def synthetic_args(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = inputs / run.DATASETS["noise_000"]
    data.write_bytes(b"synthetic job-state fixture, not an experimental HDF5")
    (inputs / "manifest.json").write_text(json.dumps({"files": [
        {"path": data.name, "bytes": data.stat().st_size, "sha256": pde.digest(data)}]}))
    monkeypatch.setenv("GIFT_DATA_ROOT", str(inputs))
    monkeypatch.setattr(pde, "source_record", lambda: {"sha256": "fixture_source_only"})
    return run.parse_args(["--execute", "--method", "PDE-FIND", "--condition", "noise_000",
                           "--output", str(tmp_path / "jobs")])


def test_job_commit_and_cached_read_do_not_recompute(tmp_path, monkeypatch):
    args = synthetic_args(tmp_path, monkeypatch)
    calls = []
    def fake_calculation(*_):
        calls.append(True)
        return {"parameters": {"nu": .123, "beta": .456, "gamma": .789},
                "native_coefficients": [], "synthetic_test_fixture": True}
    monkeypatch.setattr(run, "calculate", fake_calculation)
    result = run.execute(args)
    assert result["status"] == "complete" and len(calls) == 1
    reused = run.execute(args)
    assert reused["status"] == "cached_result_read" and len(calls) == 1
    assert reused["scientific_acceptance"] == "not_evaluated_against_published_reference"
    monkeypatch.delenv("GIFT_DATA_ROOT")
    assert run.read_completed(Path(result["job"]), args.method, args.condition)["parameters"] == result["parameters"]
    (Path(result["job"]) / "summary.csv").write_text("tampered test fixture")
    with pytest.raises(ValueError, match="SHA256"):
        run.read_completed(Path(result["job"]), args.method, args.condition)


def test_failed_job_is_retained_and_not_rerun(tmp_path, monkeypatch):
    args = synthetic_args(tmp_path, monkeypatch)
    def fail(*_):
        raise RuntimeError("synthetic failed calculation")
    monkeypatch.setattr(run, "calculate", fail)
    with pytest.raises(RuntimeError, match="synthetic failed"):
        run.execute(args)
    pending = run._job_path(args).with_name("pde_find.partial")
    assert (pending / "FAILURE.json").is_file()
    assert not run._job_path(args).exists()
    with pytest.raises(FileExistsError):
        run.execute(args)


def test_pinn_plan_is_reference_readout_not_fresh_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GIFT_DATA_ROOT", str(tmp_path))
    run.main(["--dry-run", "--method", "PINN-SR", "--condition", "noise_000"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["implemented"] and plan["scope"] == "trained_coefficients_readout"
    assert plan["training"] is False and plan["forward"] is False
    assert plan["fresh_training"] is False and plan["writes"] == 0
