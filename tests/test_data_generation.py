"""Bounded CPU tests, never full-budget data generation or model training."""
import importlib.util
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gift_generate_data", ROOT / "scripts/generate_data.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_seed_parameter_protocol():
    groups = generator.seed_parameters()
    assert generator.array_hash(groups["validation"][1]) == "7b8e4cb4174a3efb5e8933a9c8f976bf492fd7ee7f7c95e6656a67a1c698569b"
    assert generator.array_hash(groups["test"][1]) == "94d1ce16d6beb4cf92c2985253661ff978cb9af80297dd97818ecf437a935604"
    assert groups["training"][0].tolist() == list(range(50))
    assert generator.parse_ids("0,2,50:52") == {0, 2, 50, 51, 52}
    with pytest.raises(ValueError):
        generator.parse_ids("0,0")


def test_short_integration_and_own_resume(tmp_path):
    straight, paused = tmp_path / "straight", tmp_path / "paused"
    common = ["--dataset", "standard", "--split", "training", "--subset", "0",
              "--pilot-steps", "8", "--checkpoint-every", "2", "--execute"]
    assert generator.main(common + ["--output", str(straight)]) == 0
    assert generator.main(common + ["--output", str(paused), "--stop-after-steps", "3"]) == 0
    assert not (paused / "COMPLETE.json").exists()
    with pytest.raises(ValueError, match="mismatch"):
        generator.main(common + ["--output", str(paused), "--resume", "--batch-size", "2"])
    assert generator.main(common + ["--output", str(paused), "--resume"]) == 0
    with h5py.File(straight / "data.h5", "r") as a, h5py.File(paused / "data.h5", "r") as b:
        values = a["training/vorticity"][:]
        assert np.isfinite(values).all()
        assert not np.array_equal(values[:, 0], values[:, -1])
        np.testing.assert_array_equal(values, b["training/vorticity"][:])
        np.testing.assert_array_equal(a["training/time"][:], [0., .02, .04])
        assert a.attrs["status"] == "COMPLETE_PILOT"
    with pytest.raises(ValueError, match="already complete"):
        generator.main(common + ["--output", str(paused), "--resume"])
    with pytest.raises(FileExistsError):
        generator.main(common + ["--output", str(paused)])


def test_output_guards_and_dry_plan(tmp_path, monkeypatch, capsys):
    data = tmp_path / "readonly_data"
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    for bad in (ROOT / "outputs", data / "outputs", tmp_path):
        with pytest.raises(ValueError, match="outside"):
            generator.safe_output(bad)
    out = tmp_path / "planned"
    generator.main(["--dataset", "cross-resolution", "--subset", "1000", "--output", str(out)])
    plan = json.loads(capsys.readouterr().out)
    assert not out.exists()
    assert [(g["name"], len(g["steps"])) for g in plan["groups"]] == [("N96/test", 20), ("N128/test", 20)]


def test_reference_parameters_and_t0():
    """Read only the downloaded input package; no path to an author's project."""
    configured = os.environ.get("GIFT_TEST_REFERENCE_DATA_ROOT")
    if not configured:
        pytest.skip("Optional read-only comparison requires GIFT_TEST_REFERENCE_DATA_ROOT")
    from src.even_full_spectrum_ns import build_initial_hat
    data = Path(configured)
    groups = generator.seed_parameters()
    with h5py.File(data / "fno/fno1000_n64_t0_t10_dt0p02.h5", "r") as handle:
        np.testing.assert_array_equal(groups["training"][1], handle["training/initial_condition_parameters"][:50])
    with h5py.File(data / "fno/fno_test_n64_n96_n128_dt0p02.h5", "r") as handle:
        for grid in (64, 96, 128):
            np.testing.assert_array_equal(groups["test"][1], handle[f"N{grid}/test/initial_parameters"][:])
    extra_ids, extra = generator.extra_parameters(data / "initial_conditions/baseline_extra950.json")
    assert extra_ids.tolist() == list(range(1200, 2150))
    errors = {}
    with torch.inference_mode(), h5py.File(data / "standard_ns_n64_full_spectrum.h5", "r") as handle:
        for split in ("training", "validation"):
            # Full original group shape; CPU versus original CUDA may differ in last bits.
            generated = torch.fft.ifft2(build_initial_hat(groups[split][1], device="cpu")).real.numpy()
            reference = handle[f"{split}/vorticity"][:, 0]
            # Declared before execution; this is not an altered experimental acceptance gate.
            np.testing.assert_allclose(generated, reference, atol=1e-5, rtol=1e-6)
            errors[split] = {"max_absolute": float(np.max(np.abs(generated-reference))),
                             "bitwise_equal": bool(np.array_equal(generated, reference))}
    with torch.inference_mode(), h5py.File(data / "fno/fno1000_n64_t0_t10_dt0p02.h5", "r") as handle:
        generated = torch.fft.ifft2(build_initial_hat(extra[:1], device="cpu")).real.numpy()
        reference = handle["training/vorticity"][50:51, 0]
        np.testing.assert_allclose(generated, reference, atol=1e-5, rtol=1e-6)
        errors["extra950_first"] = {"max_absolute": float(np.max(np.abs(generated-reference))),
                                   "bitwise_equal": bool(np.array_equal(generated, reference))}
    print("READ_ONLY_T0_COMPARISON=" + json.dumps(errors, sort_keys=True))


def test_cross_resolution_short_pilot(tmp_path):
    out = tmp_path / "cross"
    generator.main(["--dataset", "cross-resolution", "--subset", "1000",
                    "--pilot-steps", "1", "--output", str(out), "--execute"])
    with h5py.File(out / "data.h5", "r") as handle:
        for grid in (96, 128):
            values = handle[f"N{grid}/test/vorticity"][:]
            assert values.shape == (1, 2, grid, grid)
            assert np.isfinite(values).all()


def full_plan(dataset):
    return generator.make_plan(argparse.Namespace(dataset=dataset, initial_conditions=None,
        subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))


def test_formal_short_time_labels_match_reader_exactly():
    for dataset in ("short-test", "cross-resolution"):
        plan = full_plan(dataset)
        for group in plan["groups"]:
            np.testing.assert_array_equal(generator.stored_times(plan, group),
                                          np.round(np.arange(41, 61, dtype=np.float64)/10., 12))
            assert group["steps"] == list(range(820, 1201, 20))


def test_native_fno_metadata_reader_interface(tmp_path):
    """Sparse zero-filled MOCK fields test the reader only; not scientific output."""
    from experiments.formal._shared.common import load_fno_test_dt0p02
    plan = full_plan("fno-test")
    path = tmp_path / "MOCK_reader_only.h5"
    group_plan = plan["groups"][0]
    with h5py.File(path, "x") as handle:
        handle.attrs["role"] = "METADATA_ONLY_MOCK_NOT_SCIENCE"
        group = handle.create_group("N64/test")
        group.create_dataset("trajectory_index", data=np.arange(1000, 1200, dtype=np.int64))
        group.create_dataset("time", data=generator.stored_times(plan, group_plan))
        group.create_dataset("vorticity", shape=(200,196,64,64), dtype="f4", chunks=(1,1,64,64), fillvalue=0.)
        generator.write_generation_metadata(handle, plan, complete=False)
    paths = SimpleNamespace(fno_test_dt0p02=path, require=lambda *names: None)
    with pytest.raises(ValueError, match="metadata differs"):
        load_fno_test_dt0p02(paths, 64, [5., 6.])
    with h5py.File(path, "r+") as handle:
        generator.write_generation_metadata(handle, plan, complete=True)
    ids, times, context, report_times, truth = load_fno_test_dt0p02(paths,64,[5.,6.])
    assert context.shape == (200,46,64,64)
    assert truth.shape == (200,2,64,64)
    assert len(ids) == 200
    np.testing.assert_allclose(report_times,[5.,6.],atol=2e-12,rtol=0)
    with h5py.File(path, "r+") as handle:
        plan["pilot"] = True
        generator.write_generation_metadata(handle, plan, complete=True)
    with pytest.raises(ValueError, match="metadata differs"):
        load_fno_test_dt0p02(paths,64,[5.,6.])


def test_single_trajectory_initial_path_diagnostic(tmp_path, monkeypatch):
    """One CPU trajectory, eight steps; reference comparison is diagnostic only.

    Never treats stored vorticity as the initial condition. No assertion of
    cross-device bitwise equality or changed experimental tolerance is made.
    """
    from src.even_full_spectrum_ns import (
        EvenFullSpectrumNSSolver, build_initial_hat, periodic_gaussian_native_hat,
    )
    parameters = generator.seed_parameters()["training"][1][:1]
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        raise AssertionError("The clean Torch generation path must not call NumPy FFT")

    for name in ("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "rfft", "irfft", "rfft2", "irfft2"):
        monkeypatch.setattr(np.fft, name, forbidden)
    with torch.inference_mode():
        state = build_initial_hat(parameters, device="cpu")
        reconstructed = torch.zeros_like(state[0])
        vortex_dtypes = []
        for values in parameters[0]:
            vortex = periodic_gaussian_native_hat(*values, device="cpu")
            vortex_dtypes.append(str(vortex.dtype))
            reconstructed += vortex
        before_zero = complex(reconstructed[0, 0])
        reconstructed[0, 0] = 0.
        assert torch.equal(state[0], reconstructed)
        solver = EvenFullSpectrumNSSolver(device="cpu")
        fields = {}
        for step in range(9):
            if step in (0, 4, 8):
                fields[step] = torch.fft.ifft2(state).real.numpy().copy()
            if step < 8:
                state = solver.advance(state)
    assert not forbidden_calls
    report = dict(trajectory_id=0, device="cpu", maximum_integrator_steps=8,
                  numpy_fft_calls=0, initial_vortex_dtypes=vortex_dtypes,
                  initial_manual_sum_and_zero_mode_exact=True,
                  pre_zero_mode=[before_zero.real, before_zero.imag],
                  initial_field_dtype=str(fields[0].dtype), solver_sha256=generator.sha256(
                      ROOT / "src/even_full_spectrum_ns.py"),
                  reference_used_as_generation_input=False, reference_comparison=None)
    configured = os.environ.get("GIFT_TEST_REFERENCE_DATA_ROOT")
    if configured:
        data = Path(configured)
        with h5py.File(data / "fno/fno1000_n64_t0_t10_dt0p02.h5", "r") as handle:
            np.testing.assert_array_equal(parameters, handle["training/initial_condition_parameters"][:1])
        comparisons = {}
        with h5py.File(data / "standard_ns_n64_full_spectrum.h5", "r") as handle:
            for step, field in fields.items():
                expected = handle["training/vorticity"][0:1, step // 4]
                difference = field.astype(np.float64) - expected.astype(np.float64)
                comparisons[str(step)] = dict(max_absolute=float(np.abs(difference).max()),
                    relative_l2=float(np.linalg.norm(difference) / np.linalg.norm(expected.astype(np.float64))),
                    unequal_values=int(np.count_nonzero(field != expected)),
                    total_values=field.size, bitwise_equal=bool(np.array_equal(field, expected)))
        report["reference_comparison"] = comparisons
        report["parameter_values_bitwise_equal"] = True
    (tmp_path / "initial_path_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SINGLE_TRAJECTORY_INITIAL_PATH_DIAGNOSTIC=" + json.dumps(report, sort_keys=True))
