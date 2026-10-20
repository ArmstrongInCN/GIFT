"""Small tensor checks for generator scaling, initialization and ETDRK4."""
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import numpy as np

from gift.model import FreeHermitianTable, GIFTGenerator, make_wave_grid
from gift.identifiability import _minimum_norm_from_statistics
from gift.identified import load_generator
from src.even_full_spectrum_ns import EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver


@pytest.mark.parametrize("scale", [0., -1., math.inf, math.nan])
def test_invalid_generator_scale_rejected(scale):
    """Reject zero, negative and non-finite quadratic scaling at construction."""
    with pytest.raises(ValueError, match="state_scale"):
        GIFTGenerator(2, rank=2, state_scale=scale)


def test_small_positive_scale_is_used_as_declared():
    """Keep a valid tiny scale rather than silently replacing it with an epsilon."""
    model = GIFTGenerator(1, rank=1, state_scale=1e-14)
    assert model.state_scale == 1e-14


@pytest.mark.parametrize("n", [8, 16])
def test_fourier_initialization_matches_previous_count_formula(n):
    """Show active modes map uniquely to table entries on both small test grids."""
    grid = make_wave_grid(n, 2, torch.device("cpu"), torch.float32)
    value = torch.fft.fft2(torch.arange(n * n, dtype=torch.float32).reshape(n, n))
    model = FreeHermitianTable(2)
    expected = torch.zeros_like(model.values)
    counts = torch.zeros(expected.shape[:2])
    iy, ix = grid.table_y[grid.mask], grid.table_x[grid.mask]
    expected[iy, ix, 0] += value[grid.mask].real
    expected[iy, ix, 1] += value[grid.mask].imag
    counts[iy, ix] += 1
    expected /= counts.clamp_min(1)[..., None]
    model.initialize_from_fft_array(value, grid)
    assert torch.equal(model.values, expected)


def test_affine_fit_rejects_final_precision_overflow():
    """Reject finite float64 statistics that overflow the exported complex64 constant."""
    zero = torch.zeros(4, 4, dtype=torch.complex128)
    with pytest.raises(FloatingPointError, match="fitted constant"):
        _minimum_norm_from_statistics(sample_count=2, sum_x=zero,
            sum_y=torch.full_like(zero, 1e50), sum_xx=torch.zeros(4, 4, dtype=torch.float64),
            sum_xy=zero, state_scale=1., active_mask=torch.ones(4, 4, dtype=torch.bool),
            rcond=1e-6, storage_dtype=torch.float32, subtract_quadratic=False)


def test_internal_etdrk4_matches_public_checked_stages():
    """Require unchecked internal stages to equal the public checked RK4 arithmetic."""
    solver = EvenFullSpectrumNSSolver(EvenFullSpectrumConfiguration(grid=8, padding_grid=13))
    field = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) / 64
    state = torch.fft.fft2(field)
    c = solver.coefficients
    n0 = solver.remainder(state)
    a = c["e2"] * state + c["q"] * n0
    na = solver.remainder(a)
    b = c["e2"] * state + c["q"] * na
    nb = solver.remainder(b)
    stage = c["e2"] * a + c["q"] * (2 * nb - n0)
    nc = solver.remainder(stage)
    expected = c["e"] * state + c["f1"] * n0 + 2 * c["f2"] * (na + nb) + c["f3"] * nc
    expected[..., 0, 0] = 0
    assert torch.equal(solver.advance(state), expected)
    with pytest.raises(TypeError, match="complex64"):
        solver.advance(field)


def test_loaded_payload_does_not_read_weights_twice(monkeypatch):
    """Build from an in-memory payload without reopening or rehashing its artifact."""
    model = GIFTGenerator(2, rank=2)
    payload = dict(format_version=3, model_configuration=model.configuration(), model_state_dict=model.state_dict())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unexpected second weight read"))
    loaded = load_generator(payload)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])


def test_configuration_import_requires_no_torch():
    """Import architecture planning metadata in a fresh process without Torch."""
    root = str(Path(__file__).resolve().parents[1])
    code = ("import sys; sys.path.insert(0, %r); "
            "from adapters.anie_config import ANIEConfig; assert ANIEConfig().frames == 151; "
            "assert 'torch' not in sys.modules") % root
    subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True, timeout=30)


def test_pinn_binding_records_consumed_observations(tmp_path):
    """Changing a selected target must change the loaded-array resume fingerprint."""
    from adapters.pinn_runtime import read_inputs
    sampling, data = tmp_path / "noise_000_seed1234.npz", tmp_path / "data.h5"
    data.write_bytes(b"input metadata")
    arrays = dict(measurement_coordinates=np.arange(9, dtype=np.float32).reshape(3, 3),
                  targets=np.ones((3, 3), np.float32), train_indices=np.array([0, 2]),
                  lhs_coordinates=np.zeros((2, 3), np.float32),
                  lower=np.zeros(3, np.float32), upper=np.ones(3, np.float32))
    np.savez(sampling, **arrays)
    first = read_inputs(sampling, data, condition="noise_000")
    np.testing.assert_array_equal(first[0], arrays["measurement_coordinates"][[0, 2]])
    arrays["targets"][2, 0] = 2
    np.savez(sampling, **arrays)
    second = read_inputs(sampling, data, condition="noise_000")
    assert first[-1]["selected_arrays_sha256"] != second[-1]["selected_arrays_sha256"]
