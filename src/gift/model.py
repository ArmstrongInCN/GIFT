from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class WaveGrid:
    kx: torch.Tensor
    ky: torch.Tensor
    k2: torch.Tensor
    mask: torch.Tensor
    table_y: torch.Tensor
    table_x: torch.Tensor
    table_negative_y: torch.Tensor
    table_negative_x: torch.Tensor


def make_wave_grid(
    n: int,
    cutoff: int,
    device: torch.device,
    dtype: torch.dtype,
) -> WaveGrid:
    frequency = torch.fft.fftfreq(n, d=1.0 / float(n), device=device, dtype=dtype)
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    ix = kx.round().long()
    iy = ky.round().long()
    mask = (ix.abs() <= int(cutoff)) & (iy.abs() <= int(cutoff))
    return WaveGrid(
        kx=kx,
        ky=ky,
        k2=kx.square() + ky.square(),
        mask=mask,
        table_y=(iy + cutoff).clamp(0, 2 * cutoff),
        table_x=(ix + cutoff).clamp(0, 2 * cutoff),
        table_negative_y=(-iy + cutoff).clamp(0, 2 * cutoff),
        table_negative_x=(-ix + cutoff).clamp(0, 2 * cutoff),
    )


class FreeHermitianTable(nn.Module):
    """A free Fourier table constrained only by real-field Hermitian symmetry."""

    def __init__(self, cutoff: int, standard_deviation: float = 0.02):
        super().__init__()
        self.cutoff = int(cutoff)
        width = 2 * self.cutoff + 1
        self.values = nn.Parameter(torch.empty(width, width, 2))
        nn.init.normal_(self.values, std=float(standard_deviation))

    def forward(self, grid: WaveGrid) -> torch.Tensor:
        plus = self.values[grid.table_y, grid.table_x]
        minus = self.values[grid.table_negative_y, grid.table_negative_x]
        real = 0.5 * (plus[..., 0] + minus[..., 0])
        imaginary = 0.5 * (plus[..., 1] - minus[..., 1])
        return torch.complex(real, imaginary) * grid.mask

    @torch.no_grad()
    def initialize_from_fft_array(self, value: torch.Tensor, grid: WaveGrid) -> None:
        if value.ndim != 2:
            raise ValueError("expected a two-dimensional Fourier array")
        width = 2 * self.cutoff + 1
        table = torch.zeros(width, width, 2, dtype=self.values.dtype, device=self.values.device)
        counts = torch.zeros(width, width, dtype=self.values.dtype, device=self.values.device)
        active = grid.mask
        iy = grid.table_y[active]
        ix = grid.table_x[active]
        table[iy, ix, 0] += value[active].real.to(table.dtype)
        table[iy, ix, 1] += value[active].imag.to(table.dtype)
        counts[iy, ix] += 1.0
        self.values.copy_(table / counts.clamp_min(1.0)[..., None])


class GIFTGenerator(nn.Module):
    """Form-free constant, linear and homogeneous-quadratic field generator.

    The model is

        G(w) = c + A w + Q(w, w).

    ``c`` and the Fourier multiplier ``A`` are free Hermitian tables. ``Q`` is
    a learned low-rank spectral triad operator. No Navier--Stokes derivative
    term, isotropy law, dissipative sign, forcing support, or PDE dictionary is
    encoded in this class.
    """

    def __init__(
        self,
        cutoff: int,
        rank: int = 8,
        state_scale: float = 1.0,
        conserve_spatial_mean: bool = False,
    ):
        super().__init__()
        self.cutoff = int(cutoff)
        self.rank = int(rank)
        self.state_scale = float(state_scale)
        self.conserve_spatial_mean = bool(conserve_spatial_mean)
        self.linear_table = FreeHermitianTable(cutoff, standard_deviation=0.01)
        self.constant_table = FreeHermitianTable(cutoff, standard_deviation=0.0)
        self.left_tables = nn.ModuleList(
            [FreeHermitianTable(cutoff, standard_deviation=0.08) for _ in range(rank)]
        )
        self.right_tables = nn.ModuleList(
            [FreeHermitianTable(cutoff, standard_deviation=0.08) for _ in range(rank)]
        )
        self.log_left_gain = nn.Parameter(torch.full((rank,), -0.3))
        self.log_right_gain = nn.Parameter(torch.full((rank,), -0.3))
        self.nonlinear_scale = nn.Parameter(torch.full((rank,), 0.25))
        self._cache_wave_grids = False
        self._wave_grids = {}

    def enable_static_cache(self, enabled: bool = True) -> None:
        """Cache geometry only; learned tables are recomputed on every forward.

        This opt-in execution setting is not part of a model checkpoint. The
        default path remains the reference implementation, including M1.
        """
        self._cache_wave_grids = bool(enabled)
        self._wave_grids.clear()

    def configuration(self) -> dict[str, int | float | bool]:
        return {
            "cutoff": self.cutoff,
            "rank": self.rank,
            "state_scale": self.state_scale,
            "conserve_spatial_mean": self.conserve_spatial_mean,
        }

    def prepare(self, n: int, device: torch.device, dtype: torch.dtype) -> dict:
        key = (n, self.cutoff, device, dtype)
        grid = self._wave_grids.get(key) if self._cache_wave_grids else None
        if grid is None:
            # Validation may initialize a cache before training. Ordinary
            # tensors can subsequently be saved by autograd; inference tensors cannot.
            with torch.inference_mode(False), torch.no_grad():
                grid = make_wave_grid(n, self.cutoff, device, dtype)
            if self._cache_wave_grids:
                self._wave_grids[key] = grid
        return {
            "grid": grid,
            "linear": self.linear_table(grid),
            "constant": self.constant_table(grid),
            "left": torch.stack([module(grid) for module in self.left_tables]),
            "right": torch.stack([module(grid) for module in self.right_tables]),
        }

    def nonlinear_hat(self, state_hat: torch.Tensor, prepared: dict) -> torch.Tensor:
        grid: WaveGrid = prepared["grid"]
        scale = max(self.state_scale, 1e-12)
        left = torch.fft.ifft2(
            state_hat[:, None] * prepared["left"][None], dim=(-2, -1)
        ).real / scale
        right = torch.fft.ifft2(
            state_hat[:, None] * prepared["right"][None], dim=(-2, -1)
        ).real / scale
        left = self.log_left_gain.exp()[None, :, None, None] * left
        right = self.log_right_gain.exp()[None, :, None, None] * right
        product_hat = torch.fft.fft2(left * right, dim=(-2, -1))
        result = (
            self.nonlinear_scale[None, :, None, None] * product_hat
        ).sum(dim=1)
        return result * grid.mask

    def components_hat(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if state.ndim == 2:
            state = state[None]
        prepared = self.prepare(int(state.shape[-1]), state.device, state.dtype)
        grid: WaveGrid = prepared["grid"]
        state_hat = torch.fft.fft2(state) * grid.mask
        return {
            "constant": prepared["constant"].expand_as(state_hat),
            "linear": prepared["linear"] * state_hat,
            "quadratic": self.nonlinear_hat(state_hat, prepared),
        }

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeeze = state.ndim == 2
        components = self.components_hat(state)
        result = components["constant"] + components["linear"] + components["quadratic"]
        if self.conserve_spatial_mean:
            result[..., 0, 0] = 0.0
        value = torch.fft.ifft2(result).real
        return value[0] if squeeze else value
