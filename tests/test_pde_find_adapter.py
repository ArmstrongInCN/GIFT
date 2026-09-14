"""Small external-source equivalence tests, never full M1 scientific acceptance.

Set GIFT_EXTERNAL_ROOT for pinned upstream tests. Analytic periodic fields test
the data adapter independently, without another project checkout.
"""
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
    for window in (9, 19):
        values = np.random.RandomState(4).normal(size=window)
        positions = np.arange(window, dtype=np.float64) * .02
        for order in (1, 2):
            expected = upstream.PolyDiffPoint(values, positions, deg=5, diff=order)[order - 1]
            observed = values @ pde.derivative_weights(window, order, .02)
            assert abs(expected - observed) <= 1e-10


def test_known_physical_columns_match_declared_equation(upstream, small_rows):
    observed, names = pde.library(upstream, small_rows, known=True)
    expected = np.column_stack((small_rows["w"][:, 0],
        (small_rows["u"]*small_rows["wx"] + small_rows["v"]*small_rows["wy"])[:, 0],
        (small_rows["wxx"]+small_rows["wyy"])[:, 0], small_rows["q"][:, 0])).astype(np.complex64)
    np.testing.assert_array_equal(observed, expected)
    assert len(names) == 4


def test_periodic_velocity_matches_single_fourier_mode():
    n = 64
    yy, xx = np.meshgrid(2*np.pi*np.arange(n)/n, 2*np.pi*np.arange(n)/n, indexing="ij")
    amplitude = np.array([1., 2., 3.])[:, None, None]
    w = amplitude*np.sin(xx)*np.cos(2*yy)
    u, v = pde.velocity(w)
    np.testing.assert_allclose(u, -.4*amplitude*np.sin(xx)*np.sin(2*yy), atol=1e-14)
    np.testing.assert_allclose(v, -.2*amplitude*np.cos(xx)*np.cos(2*yy), atol=1e-14)


@pytest.mark.parametrize("condition", pde.CONDITIONS)
def test_derivative_rows_and_svd_preserve_analytic_low_rank_field(condition):
    n, times = 64, np.arange(101)*.02
    yy, xx = np.meshgrid(2*np.pi*np.arange(n)/n, 2*np.pi*np.arange(n)/n, indexing="ij")
    amplitude = 1 + .1*times[:, None, None]
    raw = amplitude*np.sin(xx)*np.cos(2*yy)
    rows, protocol = pde.prepare_rows(raw, times, condition)
    selected = np.asarray(protocol["time_indices"])
    assert len(np.unique(selected)) == 60 and protocol["row_order"] == "y,x,time"
    def cube(name):
        return rows[name].reshape(n, n, 60).transpose(2, 0, 1)
    # SVD of the large rank-one field has ordinary double-precision roundoff.
    np.testing.assert_allclose(cube("w"), raw[selected], atol=1e-12)
    np.testing.assert_allclose(cube("wt"), np.broadcast_to(.1*np.sin(xx)*np.cos(2*yy), (60,n,n)), atol=1e-10)
    # A polynomial stencil is not an exact spectral derivative. Its Fourier
    # symbol predicts the discrete answer, including its truncation error.
    def response(window, order, frequency):
        offsets = np.arange(-(window//2), window//2+1)
        return np.dot(pde.derivative_weights(window, order, 2*np.pi/n),
                      np.exp(1j*frequency*offsets*2*np.pi/n))
    np.testing.assert_allclose(cube("wx"), response(19,1,1).imag*amplitude[selected]*np.cos(xx)*np.cos(2*yy), atol=1e-10)
    np.testing.assert_allclose(cube("wy"), -response(9,1,2).imag*amplitude[selected]*np.sin(xx)*np.sin(2*yy), atol=1e-10)
    np.testing.assert_allclose(cube("wxx"), response(19,2,1).real*raw[selected], atol=1e-10)
    np.testing.assert_allclose(cube("wyy"), response(9,2,2).real*raw[selected], atol=1e-10)
    assert abs(response(9,2,2).real/(-4)-1) < 1e-3
    np.testing.assert_allclose(cube("q"), np.broadcast_to(-4*np.cos(4*yy), (60,n,n)), atol=1e-14)


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
