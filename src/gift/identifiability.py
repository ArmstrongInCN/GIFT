from __future__ import annotations

from dataclasses import dataclass
import torch

from .model import GIFTGenerator, WaveGrid, make_wave_grid


@dataclass(frozen=True)
class AffineMinimumNormFit:
    """Result of a rank-revealing affine fit in fixed dimensionless coordinates.

    ``constant_hat`` and ``linear_hat`` use the same Fourier convention as
    :class:`GIFTGenerator`.  The two singular values at every Fourier mode are
    ordered from largest to smallest.  A numerical rank below two means that
    at least one affine coefficient direction is not supported by the supplied
    states at the declared precision.
    """

    constant_hat: torch.Tensor
    linear_hat: torch.Tensor
    singular_values: torch.Tensor
    tolerance: torch.Tensor
    numerical_rank: torch.Tensor
    intercept_retained: torch.Tensor
    slope_retained: torch.Tensor
    sample_count: int
    state_scale: float
    rcond: float
    storage_dtype: str
    subtract_quadratic: bool = True

    def summary(self, grid: WaveGrid) -> dict:
        active = grid.mask.detach().cpu()
        rank = self.numerical_rank.detach().cpu()
        intercept_retained = self.intercept_retained.detach().cpu()
        slope_retained = self.slope_retained.detach().cpu()
        singular = self.singular_values.detach().double().cpu()
        tolerance = self.tolerance.detach().double().cpu()
        kx = grid.kx.detach().cpu()
        ky = grid.ky.detach().cpu()
        histogram = {
            str(value): int(((rank == value) & active).sum()) for value in (0, 1, 2)
        }
        unresolved = []
        for index in torch.nonzero(active & (rank < 2), as_tuple=False):
            iy, ix = (int(index[0]), int(index[1]))
            unresolved.append(
                {
                    "ky": int(round(float(ky[iy, ix]))),
                    "kx": int(round(float(kx[iy, ix]))),
                    "numerical_rank": int(rank[iy, ix]),
                    "singular_values": [
                        float(singular[iy, ix, 0]),
                        float(singular[iy, ix, 1]),
                    ],
                    "tolerance": float(tolerance[iy, ix]),
                    "intercept_retained": bool(intercept_retained[iy, ix]),
                    "slope_retained": bool(slope_retained[iy, ix]),
                    "status": "discarded TSVD directions use the centered minimum-norm gauge",
                }
            )
        return {
            "method": "rank-revealing TSVD Moore-Penrose affine fit",
            "design_coordinates": "[1, fft(state)/N^2 - pooled_sample_mean]",
            "target_coordinates": (
                "fft(derivative-Q)/(N^2)"
                if self.subtract_quadratic
                else "fft(derivative)/(N^2)"
            ),
            "subtract_quadratic": self.subtract_quadratic,
            "singular_value_precision": "complex128/float64",
            "storage_precision_assumption": self.storage_dtype,
            "rcond": self.rcond,
            "threshold_rule": "sigma_j > rcond * sigma_max",
            "sample_count": self.sample_count,
            "state_scale": self.state_scale,
            "active_mode_count": int(active.sum()),
            "rank_histogram": histogram,
            "unresolved_modes": unresolved,
        }


def default_rank_rcond(
    sample_count: int,
    storage_dtype: torch.dtype = torch.float32,
) -> float:
    """Precision-aware relative cutoff used by GIFT.

    This is the standard matrix-rank rule ``max(M, N) * eps`` for an
    ``M x 2`` design.  It reproduces the independently preregistered zero-mode
    audit.  It must be recorded with every fitted artifact because a
    minimum-norm gauge depends on the declared numerical precision and sample
    support.
    """

    if storage_dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError("storage_dtype must be a real floating-point dtype")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return max(int(sample_count), 2) * float(torch.finfo(storage_dtype).eps)


def _minimum_norm_from_statistics(
    *,
    sample_count: int,
    sum_x: torch.Tensor,
    sum_y: torch.Tensor,
    sum_xx: torch.Tensor,
    sum_xy: torch.Tensor,
    state_scale: float,
    active_mask: torch.Tensor,
    rcond: float,
    storage_dtype: torch.dtype,
    subtract_quadratic: bool,
) -> AffineMinimumNormFit:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if state_scale <= 0.0:
        raise ValueError("state_scale must be positive")
    if not 0.0 < rcond < 1.0:
        raise ValueError("rcond must lie strictly between zero and one")
    if sum_x.shape != sum_y.shape or sum_x.shape != sum_xx.shape:
        raise ValueError("all sufficient statistics must have equal spatial shape")

    shape = sum_x.shape
    mean_x = sum_x / float(sample_count)
    mean_y = sum_y / float(sample_count)
    centered_xx = (
        sum_xx - float(sample_count) * mean_x.abs().square()
    ).clamp_min(0.0)
    centered_xy = sum_xy - float(sample_count) * mean_x.conj() * mean_y
    for name, value in {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "centered_xx": centered_xx,
        "centered_xy": centered_xy,
    }.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite sufficient statistic: {name}")

    intercept_singular = torch.full_like(centered_xx, float(sample_count) ** 0.5)
    slope_singular = centered_xx.sqrt()
    largest = torch.maximum(intercept_singular, slope_singular)
    tolerance = float(rcond) * largest
    keep_intercept = intercept_singular > tolerance
    keep_slope = slope_singular > tolerance

    centered_intercept = torch.where(
        keep_intercept, mean_y, torch.zeros_like(mean_y)
    )
    slope = torch.where(
        keep_slope,
        centered_xy / centered_xx.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(centered_xy),
    )
    intercept = centered_intercept - slope * mean_x

    n = int(shape[-1])
    constant_hat = intercept * float(n * n)
    linear_hat = slope
    active = active_mask.to(device=sum_x.device)
    constant_hat = torch.where(active, constant_hat, torch.zeros_like(constant_hat))
    linear_hat = torch.where(active, linear_hat, torch.zeros_like(linear_hat))
    if not bool(torch.isfinite(constant_hat).all()):
        raise FloatingPointError("non-finite fitted constant")
    if not bool(torch.isfinite(linear_hat).all()):
        raise FloatingPointError("non-finite fitted linear multiplier")
    constant_hat = constant_hat.to(torch.complex64)
    linear_hat = linear_hat.to(torch.complex64)

    return AffineMinimumNormFit(
        constant_hat=constant_hat,
        linear_hat=linear_hat,
        singular_values=torch.stack(
            (largest, torch.minimum(intercept_singular, slope_singular)), dim=-1
        ).to(torch.float64),
        tolerance=tolerance.to(torch.float64),
        numerical_rank=(keep_intercept.to(torch.int8) + keep_slope.to(torch.int8)),
        intercept_retained=keep_intercept,
        slope_retained=keep_slope,
        sample_count=int(sample_count),
        state_scale=float(state_scale),
        rcond=float(rcond),
        storage_dtype=str(storage_dtype).removeprefix("torch."),
        subtract_quadratic=bool(subtract_quadratic),
    )


@torch.no_grad()
def fit_affine_minimum_norm(
    model: GIFTGenerator,
    state: torch.Tensor,
    derivative: torch.Tensor,
    *,
    count: int | None = None,
    device: str | torch.device | None = None,
    chunk_size: int = 32,
    rcond: float | None = None,
    storage_dtype: torch.dtype = torch.float32,
    subtract_quadratic: bool = True,
) -> AffineMinimumNormFit:
    """Fit and install ``c`` and ``A`` with a uniform per-mode TSVD rule.

    The nonlinear branch is frozen.  When ``subtract_quadratic`` is true the
    regression target is ``dstate/dt - Q(state,state)``; setting it false is
    useful for the initial affine fit before nonlinear training.  No Fourier
    mode, conservation law, PDE term, or known operator is singled out.
    """

    if state.ndim != 3 or derivative.shape != state.shape:
        raise ValueError("state and derivative must have shape [samples, N, N]")
    if state.shape[-2] != state.shape[-1]:
        raise ValueError("only square spatial grids are supported")
    sample_count = int(state.shape[0] if count is None else count)
    if not (1 <= sample_count <= int(state.shape[0])):
        raise ValueError("count is outside the available sample range")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if storage_dtype is not torch.float32:
        raise ValueError(
            "the rank-aware solver currently requires float32 observation storage"
        )

    fit_device = torch.device(device) if device is not None else next(model.parameters()).device
    n = int(state.shape[-1])
    prepared = model.prepare(n, fit_device, torch.float32)
    grid: WaveGrid = prepared["grid"]
    sum_x = torch.zeros(n, n, dtype=torch.complex128, device=fit_device)
    sum_y = torch.zeros_like(sum_x)
    sum_xx = torch.zeros(n, n, dtype=torch.float64, device=fit_device)
    sum_xy = torch.zeros_like(sum_x)
    coordinate_scale = float(n * n)
    target_scale = float(n * n)

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        batch_state = state[start:stop].to(fit_device, dtype=torch.float32)
        state_hat = torch.fft.fft2(batch_state) * grid.mask
        target_hat = torch.fft.fft2(
            derivative[start:stop].to(fit_device, dtype=torch.float32)
        ) * grid.mask
        if subtract_quadratic:
            target_hat = target_hat - model.nonlinear_hat(state_hat, prepared)
        x = (state_hat / coordinate_scale).to(torch.complex128)
        y = (target_hat / target_scale).to(torch.complex128)
        sum_x += x.sum(0)
        sum_y += y.sum(0)
        sum_xx += x.abs().square().sum(0)
        sum_xy += (x.conj() * y).sum(0)

    cutoff = (
        default_rank_rcond(sample_count, storage_dtype)
        if rcond is None
        else float(rcond)
    )
    fit = _minimum_norm_from_statistics(
        sample_count=sample_count,
        sum_x=sum_x,
        sum_y=sum_y,
        sum_xx=sum_xx,
        sum_xy=sum_xy,
        state_scale=float(model.state_scale),
        active_mask=grid.mask,
        rcond=cutoff,
        storage_dtype=storage_dtype,
        subtract_quadratic=subtract_quadratic,
    )
    model.linear_table.initialize_from_fft_array(fit.linear_hat, grid)
    model.constant_table.initialize_from_fft_array(fit.constant_hat, grid)
    return fit


def identifiability_grid(
    model: GIFTGenerator,
    n: int,
    *,
    device: str | torch.device = "cpu",
) -> WaveGrid:
    """Return the wave grid used when serializing an affine-fit report."""

    return make_wave_grid(n, model.cutoff, torch.device(device), torch.float32)
