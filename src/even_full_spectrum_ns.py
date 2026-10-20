"""Minimal-difference full-spectrum N=64 Navier--Stokes truth solver.

This module deliberately follows the released standard-NS generator's
unnormalised ``torch.fft`` convention, float32/complex64 arithmetic and
Cox--Matthews ETDRK4 implementation.  Its sole scientific change is removal
of the square K=21 state mask.  Quadratic products are evaluated on an odd
``P >= 3*(N/2)+1`` workspace.  Odd derivatives are first formed with the
standard real even-grid collocation convention (their self-conjugate Nyquist
multiplier is zero).  The resulting Hermitian field spectra are then split
between the continuous ``-N/2`` and ``+N/2`` modes for padded multiplication
and merged back after the product.  Padding is therefore alias control, not a
state-space cutoff: every one of the N x N representable DFT coefficients is
retained.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


GRID = 64
PADDING_GRID = 99
NU = 0.01
FORCE_AMPLITUDE = 1.0
FORCE_WAVENUMBER = 4
INTERNAL_DT = 0.005
CONTOUR_POINTS = 32


def _validate_even_grid(n: int) -> int:
    """Return an even grid width supporting the real collocation convention."""
    n = int(n)
    if n < 4 or n % 2:
        raise ValueError("state grid must be an even integer >= 4")
    return n


def _validate_padding(n: int, p: int) -> int:
    """Require an odd product workspace large enough for all retained modes."""
    p = int(p)
    if p < 3 * (n // 2) + 1 or p % 2 != 1:
        raise ValueError(
            f"padding grid must be odd and at least {3 * (n // 2) + 1}"
        )
    return p


def split_even_native_dft(value_hat: torch.Tensor, padding_grid: int) -> torch.Tensor:
    """Embed an even-grid native DFT into an odd padded native DFT.

    In shifted ordering, the coarse Nyquist entry represents the sum of the
    continuous ``-K`` and ``+K`` coefficients.  It is divided equally along
    each Nyquist axis (and consequently into quarters at the corner).  The
    factor ``(P/N)^2`` converts unnormalised N-grid DFT coefficients to the
    same Fourier-series coefficients under a P-grid unnormalised DFT.
    """

    if value_hat.ndim < 2 or value_hat.shape[-2] != value_hat.shape[-1]:
        raise ValueError("value_hat must end in square spatial dimensions")
    n = _validate_even_grid(int(value_hat.shape[-1]))
    p = _validate_padding(n, padding_grid)
    return _split_even_native_dft(value_hat, n, p)


def _split_even_native_dft(value_hat: torch.Tensor, n: int, p: int) -> torch.Tensor:
    """Split checked ``[..., N, N]`` native spectra into ``[..., P, P]`` storage.

    The caller has validated grid sizes. Leading axes, complex dtype and
    device are preserved; Nyquist splitting and DFT amplitude conversion are
    performed without imposing an additional state cutoff.
    """
    k = n // 2
    kp = (p - 1) // 2
    shifted = torch.fft.fftshift(value_hat, dim=(-2, -1))

    # First split the x-Nyquist column, producing modes -K,...,+K.
    split_x = torch.empty(
        (*value_hat.shape[:-2], n, n + 1),
        dtype=value_hat.dtype,
        device=value_hat.device,
    )
    split_x[..., 0] = 0.5 * shifted[..., 0]
    split_x[..., 1:n] = shifted[..., 1:n]
    split_x[..., n] = 0.5 * shifted[..., 0]

    # Then split the y-Nyquist row.  The double-Nyquist corner is now quartered.
    canonical = torch.empty(
        (*value_hat.shape[:-2], n + 1, n + 1),
        dtype=value_hat.dtype,
        device=value_hat.device,
    )
    canonical[..., 0, :] = 0.5 * split_x[..., 0, :]
    canonical[..., 1:n, :] = split_x[..., 1:n, :]
    canonical[..., n, :] = 0.5 * split_x[..., 0, :]

    padded_shifted = torch.zeros(
        (*value_hat.shape[:-2], p, p),
        dtype=value_hat.dtype,
        device=value_hat.device,
    )
    start = kp - k
    padded_shifted[..., start : start + n + 1, start : start + n + 1] = (
        canonical * float(p / n) ** 2
    )
    return torch.fft.ifftshift(padded_shifted, dim=(-2, -1))


def merge_odd_dft_to_even_native(value_hat: torch.Tensor, output_grid: int) -> torch.Tensor:
    """Crop odd ``[..., P, P]`` spectra and merge +/- Nyquist modes to ``N``.

    ``output_grid`` is even and must meet the padding relation with ``P``.
    The factor ``(N/P)^2`` restores native N-grid DFT normalisation. Leading
    dimensions, dtype and device are preserved.
    """

    if value_hat.ndim < 2 or value_hat.shape[-2] != value_hat.shape[-1]:
        raise ValueError("value_hat must end in square spatial dimensions")
    p = int(value_hat.shape[-1])
    n = _validate_even_grid(output_grid)
    _validate_padding(n, p)
    return _merge_odd_dft(value_hat, n, p)


def _merge_odd_dft(value_hat: torch.Tensor, n: int, p: int) -> torch.Tensor:
    """Merge checked padded spectra using the inverse Nyquist split convention."""
    k = n // 2
    kp = (p - 1) // 2
    start = kp - k
    shifted = torch.fft.fftshift(value_hat, dim=(-2, -1))
    canonical = shifted[..., start : start + n + 1, start : start + n + 1]

    # Merge x=+/-K, then y=+/-K.  This reverses the split convention.
    merged_x = torch.empty(
        (*value_hat.shape[:-2], n + 1, n),
        dtype=value_hat.dtype,
        device=value_hat.device,
    )
    merged_x[..., 0] = canonical[..., 0] + canonical[..., n]
    merged_x[..., 1:n] = canonical[..., 1:n]
    merged = torch.empty(
        (*value_hat.shape[:-2], n, n),
        dtype=value_hat.dtype,
        device=value_hat.device,
    )
    merged[..., 0, :] = merged_x[..., 0, :] + merged_x[..., n, :]
    merged[..., 1:n, :] = merged_x[..., 1:n, :]
    merged = merged * float(n / p) ** 2
    return torch.fft.ifftshift(merged, dim=(-2, -1))


def merge_canonical_nyquist_modes(canonical_shifted: torch.Tensor) -> torch.Tensor:
    """Merge shifted modes ``[-K,K]^2`` into an unnormalised even-grid DFT.

    Unlike :func:`merge_odd_dft_to_even_native`, the input already uses the
    target N-grid DFT normalization, so no grid-size conversion is applied.
    """

    if canonical_shifted.ndim < 2 or canonical_shifted.shape[-2] != canonical_shifted.shape[-1]:
        raise ValueError("canonical array must be square")
    width = int(canonical_shifted.shape[-1])
    if width < 5 or width % 2 != 1:
        raise ValueError("canonical width must be odd and at least five")
    n = width - 1
    merged_x = torch.empty(
        (*canonical_shifted.shape[:-2], width, n),
        dtype=canonical_shifted.dtype,
        device=canonical_shifted.device,
    )
    merged_x[..., 0] = canonical_shifted[..., 0] + canonical_shifted[..., n]
    merged_x[..., 1:n] = canonical_shifted[..., 1:n]
    merged = torch.empty(
        (*canonical_shifted.shape[:-2], n, n),
        dtype=canonical_shifted.dtype,
        device=canonical_shifted.device,
    )
    merged[..., 0, :] = merged_x[..., 0, :] + merged_x[..., n, :]
    merged[..., 1:n, :] = merged_x[..., 1:n, :]
    return torch.fft.ifftshift(merged, dim=(-2, -1))


def alias_free_product_hat(
    left_hat: torch.Tensor,
    right_hat: torch.Tensor,
    *,
    padding_grid: int = PADDING_GRID,
) -> torch.Tensor:
    """Return the native DFT of two real fields' alias-controlled product.

    Inputs are equal-layout Hermitian ``[..., N, N]`` spectra. They are split
    onto the odd ``padding_grid`` before multiplication in physical space;
    cropping retains every representable N-grid coefficient, including the
    merged Nyquist entries. Output dtype/device follow the inputs.
    """

    if left_hat.shape != right_hat.shape:
        raise ValueError("product inputs must have equal shapes")
    n = _validate_even_grid(int(left_hat.shape[-1]))
    left_padded = split_even_native_dft(left_hat, padding_grid)
    right_padded = split_even_native_dft(right_hat, padding_grid)
    left = torch.fft.ifft2(left_padded, dim=(-2, -1)).real
    right = torch.fft.ifft2(right_padded, dim=(-2, -1)).real
    product_padded = torch.fft.fft2(left * right, dim=(-2, -1))
    return merge_odd_dft_to_even_native(product_padded, n)


def etdrk4_coefficients(linear: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
    """Build Cox--Matthews ETDRK4 multipliers for a real spectral linear operator.

    ``linear`` normally has ``[N, N]`` shape and inverse-time units; ``dt`` is
    the fixed time step. Exponential coefficients ``e``/``e2`` and contour
    means ``q``/``f1``/``f2``/``f3`` share that shape. The 32 complex64 contour
    points stabilise the quotients near a zero linear eigenvalue.
    """

    points = CONTOUR_POINTS
    roots = torch.exp(
        1j
        * math.pi
        * (torch.arange(points, device=linear.device) + 0.5)
        / points
    )
    lr = dt * linear.to(torch.complex64)[..., None] + roots
    exp_lr = torch.exp(lr)
    return {
        "e": torch.exp(dt * linear),
        "e2": torch.exp(0.5 * dt * linear),
        "q": dt * torch.mean((torch.exp(lr / 2) - 1) / lr, dim=-1).real,
        "f1": dt
        * torch.mean(
            (-4 - lr + exp_lr * (4 - 3 * lr + lr.square())) / lr.pow(3),
            dim=-1,
        ).real,
        "f2": dt
        * torch.mean((2 + lr + exp_lr * (-2 + lr)) / lr.pow(3), dim=-1).real,
        "f3": dt
        * torch.mean(
            (-4 - 3 * lr - lr.square() + exp_lr * (4 - lr)) / lr.pow(3),
            dim=-1,
        ).real,
    }


def periodic_gaussian_native_hat(
    circulation: float,
    sigma: float,
    centre_x: float,
    centre_y: float,
    *,
    grid: int = GRID,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return complex64 ``[grid, grid]`` coefficients of one periodic Gaussian.

    ``circulation`` is the spatial integral; ``sigma`` and the centre use
    coordinates on the ``2*pi`` periodic domain. Analytic Fourier-series
    coefficients are scaled to the native DFT, with +/- Nyquist entries
    combined using the same convention as the product solver.
    """

    n = _validate_even_grid(grid)
    k = n // 2
    frequency = torch.arange(
        -k, k + 1, device=torch.device(device), dtype=torch.float32
    )
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    scale = n * n / ((2 * math.pi) ** 2)
    envelope = torch.exp(-0.5 * float(sigma) ** 2 * (kx.square() + ky.square()))
    phase = torch.exp(-1j * (kx * float(centre_x) + ky * float(centre_y)))
    canonical = float(circulation) * scale * envelope * phase
    return merge_canonical_nyquist_modes(canonical)


def build_initial_hat(
    parameters: Sequence[Sequence[Sequence[float]]],
    *,
    grid: int = GRID,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build complex64 ``[trajectories, grid, grid]`` four-vortex initial spectra.

    Each trajectory supplies four ``(circulation, sigma, centre_x, centre_y)``
    tuples in the periodic ``2*pi`` coordinates. Vortex spectra are added,
    then the zero Fourier mode is set to zero to enforce zero spatial mean.
    """

    values = []
    for trajectory in parameters:
        if len(trajectory) != 4:
            raise ValueError("every trajectory must contain four vortices")
        value = torch.zeros(
            (grid, grid), dtype=torch.complex64, device=torch.device(device)
        )
        for circulation, sigma, centre_x, centre_y in trajectory:
            value += periodic_gaussian_native_hat(
                circulation,
                sigma,
                centre_x,
                centre_y,
                grid=grid,
                device=device,
            )
        value[0, 0] = 0.0
        values.append(value)
    return torch.stack(values)


@dataclass(frozen=True)
class EvenFullSpectrumConfiguration:
    """Physical coefficients and fixed discretisation for the periodic solver.

    ``grid`` is the even state width, ``padding_grid`` the odd product width,
    and ``dt`` the internal integration interval. Viscosity multiplies the
    Laplacian, and ``advection_coefficient`` multiplies the unit transport
    ``u*omega_x+v*omega_y`` without changing the velocity reconstruction.
    The vorticity forcing is
    ``-forcing_amplitude*forcing_wavenumber*cos(forcing_wavenumber*y)``.
    """

    grid: int = GRID
    padding_grid: int = PADDING_GRID
    viscosity: float = NU
    forcing_amplitude: float = FORCE_AMPLITUDE
    forcing_wavenumber: int = FORCE_WAVENUMBER
    dt: float = INTERNAL_DT
    advection_coefficient: float = 1.0


class EvenFullSpectrumNSSolver:
    """Float32/complex64 ETDRK4 solver with no sub-Nyquist state mask."""

    def __init__(
        self,
        configuration: EvenFullSpectrumConfiguration = EvenFullSpectrumConfiguration(),
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        """Prepare float32 geometry and time-step coefficients on one device.

        States supplied later must be complex64 native spectra on this
        device. The streamfunction's zero mode is defined as zero, and only
        odd derivatives suppress the self-conjugate Nyquist multiplier.
        """
        self.configuration = configuration
        if not math.isfinite(float(configuration.advection_coefficient)):
            raise ValueError("advection_coefficient must be finite")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.n = _validate_even_grid(configuration.grid)
        self.p = _validate_padding(self.n, configuration.padding_grid)
        frequency = torch.fft.fftfreq(
            self.n, d=1 / self.n, device=self.device, dtype=torch.float32
        )
        self.ky, self.kx = torch.meshgrid(frequency, frequency, indexing="ij")
        self.k2 = self.kx.square() + self.ky.square()
        # On an even collocation grid the first derivative of the self-conjugate
        # Nyquist basis cos(N*x/2) is not representable as a real grid field.
        # The standard real spectral-collocation convention therefore assigns
        # zero to that *odd* derivative multiplier.  Even derivatives retain
        # (-N/2)^2.  Derivatives are formed before padding; only their products
        # use the split/merge workspace.
        self.derivative_kx = torch.where(
            self.kx == -self.n // 2, torch.zeros_like(self.kx), self.kx
        )
        self.derivative_ky = torch.where(
            self.ky == -self.n // 2, torch.zeros_like(self.ky), self.ky
        )
        self.safe_k2 = self.k2.clone()
        self.safe_k2[0, 0] = 1.0
        self.linear = -float(configuration.viscosity) * self.k2
        self.coefficients = etdrk4_coefficients(self.linear, float(configuration.dt))

        coordinate = (
            2
            * math.pi
            * torch.arange(self.n, device=self.device, dtype=torch.float32)
            / self.n
        )
        y, _x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        forcing = (
            -float(configuration.forcing_amplitude)
            * int(configuration.forcing_wavenumber)
            * torch.cos(int(configuration.forcing_wavenumber) * y)
        )
        self.forcing_hat = torch.fft.fft2(forcing)

    @property
    def metadata(self) -> dict[str, object]:
        """Describe represented modes, product padding and arithmetic conventions."""
        return {
            "grid": self.n,
            "native_axis_wavenumbers": [-self.n // 2, self.n // 2 - 1],
            "represented_axis_wavenumbers": [-self.n // 2, self.n // 2 - 1],
            "represented_mode_count": self.n * self.n,
            "all_native_dft_coefficients_retained": True,
            "state_mask_applied": False,
            "padding_grid": self.p,
            "padding_minimum": 3 * (self.n // 2) + 1,
            "even_grid_nyquist_convention": (
                "form odd derivatives on the even grid with the Nyquist "
                "multiplier set to zero; split +/-N/2 only for padded "
                "multiplication; merge after product"
            ),
            "solver": "full-spectrum Fourier pseudospectral ETDRK4",
            "internal_dt": float(self.configuration.dt),
            "viscosity": float(self.configuration.viscosity),
            "advection_coefficient": float(self.configuration.advection_coefficient),
            "forcing_amplitude": float(self.configuration.forcing_amplitude),
            "forcing_wavenumber": int(self.configuration.forcing_wavenumber),
            "contour_points": CONTOUR_POINTS,
            "real_dtype": "float32",
            "complex_dtype": "complex64",
            "zero_fourier_mode_enforced_by_solver": True,
        }

    def _check(self, value_hat: torch.Tensor) -> None:
        """Check trailing state dimensions, complex64 arithmetic and solver device."""
        if value_hat.shape[-2:] != (self.n, self.n):
            raise ValueError(f"state must end in [{self.n},{self.n}]")
        if value_hat.dtype != torch.complex64:
            raise TypeError("state must be complex64")
        if value_hat.device != self.device:
            raise ValueError("state and solver devices differ")

    def transport_hat(self, value_hat: torch.Tensor) -> torch.Tensor:
        """Evaluate u dot grad(omega) using real even-grid collocation.

        First derivatives are formed on N=64 with the odd-derivative Nyquist
        multiplier set to zero.  The four resulting Hermitian spectra are
        subsequently split/padded for their quadratic products.  Splitting
        the raw state before differentiation would incorrectly assign a sine
        derivative to the self-conjugate coarse-grid Nyquist cosine.
        """

        self._check(value_hat)
        return self._transport_hat(value_hat)

    def _transport_hat(self, value_hat: torch.Tensor) -> torch.Tensor:
        """Evaluate checked batched spectra with ``u=psi_y`` and ``v=-psi_x``.

        ``psi_hat=omega_hat/|k|^2`` solves ``-laplacian(psi)=omega``. Derivatives
        are formed on the even state grid; velocities and vorticity gradients
        then share the padded real workspace for ``u*omega_x+v*omega_y``.
        """
        psi_hat = value_hat / self.safe_k2
        psi_hat[..., 0, 0] = 0.0
        spectral_fields = (
            1j * self.derivative_ky * psi_hat,
            -1j * self.derivative_kx * psi_hat,
            1j * self.derivative_kx * value_hat,
            1j * self.derivative_ky * value_hat,
        )
        padded_fields = [_split_even_native_dft(value, self.n, self.p) for value in spectral_fields]
        velocity_u, velocity_v, omega_x, omega_y = [
            torch.fft.ifft2(value).real for value in padded_fields
        ]
        transport_padded = torch.fft.fft2(
            velocity_u * omega_x + velocity_v * omega_y
        )
        return _merge_odd_dft(transport_padded, self.n, self.p)

    def remainder(self, value_hat: torch.Tensor) -> torch.Tensor:
        """Return ``-beta*unit_transport + forcing`` after checking the state.

        Beta=1 follows the original operation path exactly, preserving existing
        experiments. Other beta values change the physical transport coefficient.
        """
        transport = self.transport_hat(value_hat)
        if self.configuration.advection_coefficient == 1.0:
            return -transport + self.forcing_hat
        return -float(self.configuration.advection_coefficient) * transport + self.forcing_hat

    def _remainder(self, value_hat: torch.Tensor) -> torch.Tensor:
        """Evaluate the same RHS for internal ETDRK4 stages already on this grid."""
        transport = self._transport_hat(value_hat)
        if self.configuration.advection_coefficient == 1.0:
            return -transport + self.forcing_hat
        return -float(self.configuration.advection_coefficient) * transport + self.forcing_hat

    def advance(self, value_hat: torch.Tensor) -> torch.Tensor:
        """Return one fixed ETDRK4 step for complex64 ``[..., N, N]`` spectra.

        Viscosity is integrated through the exponential coefficients, while
        transport and forcing are evaluated at four stages. No spatial
        coefficient is masked; the result's zero mode alone is set to zero
        to preserve the declared zero-mean state space.
        """

        self._check(value_hat)
        coefficient = self.coefficients
        n0 = self._remainder(value_hat)
        a = coefficient["e2"] * value_hat + coefficient["q"] * n0
        na = self._remainder(a)
        b = coefficient["e2"] * value_hat + coefficient["q"] * na
        nb = self._remainder(b)
        c = coefficient["e2"] * a + coefficient["q"] * (2 * nb - n0)
        nc = self._remainder(c)
        result = (
            coefficient["e"] * value_hat
            + coefficient["f1"] * n0
            + 2 * coefficient["f2"] * (na + nb)
            + coefficient["f3"] * nc
        )
        result[..., 0, 0] = 0.0
        return result


def hermitian_imaginary_defect(value_hat: torch.Tensor) -> float:
    """Relative imaginary defect of the physical field represented by a DFT."""

    physical = torch.fft.ifft2(value_hat)
    scale = max(float(physical.real.abs().amax()), torch.finfo(torch.float32).tiny)
    return float(physical.imag.abs().amax()) / scale
