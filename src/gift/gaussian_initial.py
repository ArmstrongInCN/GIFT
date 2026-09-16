"""A real, periodic, zero-mean smooth Gaussian random vorticity field.

The construction is a fixed linear map of independent standard normals, not
four randomly placed vortices. No per-realization normalization or rejection
sampling is used: both would change the declared Gaussian distribution.
"""
import numpy as np

ROOT_SEED = 2026091601
LENGTH_SCALE = 0.4
# sqrt(mean(omega(0)**2)) over the ORIGINAL 1,000 training trajectories only.
# This fixed ensemble amplitude does not use any validation/test observation.
ENSEMBLE_RMS = 1.0908839162005806
GRID = 64


def specification():
    return {
        "family": "periodic_zero_mean_smooth_gaussian_random_field",
        "root_seed": ROOT_SEED, "rng": "NumPy PCG64 with SeedSequence([root_seed, trajectory_id])",
        "grid": GRID, "correlation_length": LENGTH_SCALE,
        "ensemble_pointwise_standard_deviation": ENSEMBLE_RMS,
        "amplitude_source": "RMS of initial fields in four-vortex training IDs 0-999 only",
        "amplitude_source_file_sha256": "23f5a0af459633b620f486624e6802b012b4fb7684b4d41a186b15c4eeab3244",
        "spectral_filter": "h(k)=exp(-ell^2*|k|^2/4), h(0)=0",
        "construction": "omega_hat=a*h*FFT(z); z~N(0,I), a=sigma*N/sqrt(sum(h^2))",
        "samplewise_rms_normalization": False, "rejection_sampling": False,
        "zero_spatial_mean": True, "parameters_selected_using_test_results": False,
    }


def initial_hat(trajectory_ids):
    """Return native FFT coefficients; row order/batch size cannot change draws."""
    ids = list(trajectory_ids)
    if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Initial-condition IDs must be distinct nonnegative integers")
    frequency = np.fft.fftfreq(GRID, d=1 / GRID)
    ky, kx = np.meshgrid(frequency, frequency, indexing="ij")
    envelope = np.exp(-LENGTH_SCALE**2 * (kx*kx + ky*ky) / 4)
    envelope[0, 0] = 0
    amplitude = ENSEMBLE_RMS * GRID / np.sqrt(np.square(envelope).sum())
    fields = []
    for identifier in ids:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([ROOT_SEED, identifier])))
        values = amplitude * envelope * np.fft.fft2(rng.standard_normal((GRID, GRID)))
        fields.append(values.astype(np.complex64))
    return np.stack(fields)
