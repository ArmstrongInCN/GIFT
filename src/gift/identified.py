"""Load frozen generators and integrate their continuous-time field dynamics.

Artifact loading checks model format and tensor layout. The grid adapter below
also preserves the physical amplitude of a constant term when native FFT
coefficients move between resolutions.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .model import GIFTGenerator


def load_release_payload(artifact: str | Path | dict) -> dict:
    """Read format-2/3 model metadata and weights without optimizer recovery.

    ``artifact`` may be an already loaded dictionary or a Torch artifact path.
    Paths are deserialised on CPU with ``weights_only=True``; dictionaries are
    returned directly after required model fields are checked.
    """

    payload = artifact if isinstance(artifact, dict) else torch.load(
        Path(artifact), map_location="cpu", weights_only=True)
    if payload.get("format_version") not in (2, 3):
        raise ValueError("unsupported GIFT artifact format")
    if "model_configuration" not in payload or "model_state_dict" not in payload:
        raise ValueError("incomplete GIFT artifact")
    return payload


def load_generator(
    artifact: str | Path | dict,
    *,
    device: str | torch.device = "cpu",
) -> GIFTGenerator:
    """Rebuild a generator, load exact state keys/shapes, and enter eval mode.

    The returned model lives on ``device``. Eval mode does not disable
    autograd; inference callers choose their own gradient context.
    """
    payload = load_release_payload(artifact)
    model = GIFTGenerator(**payload["model_configuration"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(torch.device(device)).eval()


class FixedBandwidthGridGenerator:
    """Evaluate a frozen generator on another grid with fixed Fourier support.

    ``reference_grid`` is the grid on which the model's unnormalised constant
    DFT table was identified.  Only that complete state-independent table is
    converted by ``(grid_size / reference_grid) ** 2``; the linear multiplier
    and quadratic factors are unchanged.
    """

    def __init__(
        self,
        model: GIFTGenerator,
        grid_size: int,
        *,
        reference_grid: int,
    ) -> None:
        """Prepare fixed weights on ``grid_size`` using the training FFT scale.

        Both grids must resolve all modes through ``model.cutoff``. Table
        values are prepared once, so changing weights, dtype or device later
        requires rebuilding this adapter.
        """
        if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 1:
            raise ValueError("grid_size must be a positive integer")
        if (
            isinstance(reference_grid, bool)
            or int(reference_grid) != reference_grid
            or reference_grid < 1
        ):
            raise ValueError("reference_grid must be a positive integer")
        self.model = model
        self.grid_size = int(grid_size)
        self.reference_grid = int(reference_grid)
        minimum_grid = 2 * int(model.cutoff) + 1
        if self.grid_size < minimum_grid or self.reference_grid < minimum_grid:
            raise ValueError(
                f"grid_size and reference_grid must be at least {minimum_grid} "
                "for the model cutoff"
            )
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        with torch.no_grad():
            self.prepared = model.prepare(self.grid_size, self.device, self.dtype)
        self.constant_factor = (self.grid_size / self.reference_grid) ** 2
        self._parameter_versions = tuple(
            parameter._version for parameter in self.model.parameters()
        )

    @torch.inference_mode()
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Return a derivative for real ``[N, N]`` or ``[batch, N, N]`` fields.

        Inputs must use the prepared model's device and real dtype. The
        constant spectrum scales by ``(N/reference_grid)^2`` because its
        coefficients are unnormalised DFT values; linear multipliers and
        pointwise quadratic factors already have grid-independent amplitudes.
        """
        if state.ndim not in (2, 3):
            raise ValueError("state must have shape [N, N] or [batch, N, N]")
        if tuple(state.shape[-2:]) != (self.grid_size, self.grid_size):
            raise ValueError(
                f"state spatial shape must be [{self.grid_size}, {self.grid_size}]"
            )
        parameter = next(self.model.parameters())
        if parameter.device != self.device or parameter.dtype != self.dtype:
            raise RuntimeError(
                "model device or dtype changed after the fixed-bandwidth adapter "
                "was prepared; construct a new adapter"
            )
        parameter_versions = tuple(
            parameter._version for parameter in self.model.parameters()
        )
        if parameter_versions != self._parameter_versions:
            raise RuntimeError(
                "model parameters changed after the fixed-bandwidth adapter was "
                "prepared; construct a new adapter"
            )
        if state.device != self.device:
            raise ValueError(
                f"state device {state.device} does not match model device {self.device}"
            )
        if state.dtype != self.dtype:
            raise ValueError(
                f"state dtype {state.dtype} does not match model dtype {self.dtype}"
            )

        squeeze = state.ndim == 2
        batch = state[None] if squeeze else state
        grid = self.prepared["grid"]
        state_hat = torch.fft.fft2(batch) * grid.mask
        constant = self.constant_factor * self.prepared["constant"].expand_as(state_hat)
        linear = self.prepared["linear"] * state_hat
        quadratic = self.model.nonlinear_hat(state_hat, self.prepared)
        result_hat = constant + linear + quadratic
        if self.model.conserve_spatial_mean:
            result_hat[..., 0, 0] = 0.0
        value = torch.fft.ifft2(result_hat).real
        return value[0] if squeeze else value


@torch.inference_mode()
def rollout_rk4(
    model: GIFTGenerator | FixedBandwidthGridGenerator,
    initial_state: torch.Tensor,
    *,
    dt: float,
    steps: int,
    save_every: int = 1,
) -> torch.Tensor:
    """Integrate a real initial field with explicit, fixed-step classical RK4.

    ``dt`` uses the generator's time unit; ``steps`` counts integration updates.
    The tensor returned is ``[saved_times, *initial_state.shape]`` and includes
    the initial field plus every ``save_every``-th step. A final step that is
    not a save multiple is not appended. Saved states are detached copies.
    """
    if steps < 0 or save_every < 1:
        raise ValueError("steps must be nonnegative and save_every must be positive")
    state = initial_state
    saved = [state.detach().clone()]
    for step in range(1, steps + 1):
        k1 = model(state)
        k2 = model(state + 0.5 * dt * k1)
        k3 = model(state + 0.5 * dt * k2)
        k4 = model(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if step % save_every == 0:
            saved.append(state.detach().clone())
    return torch.stack(saved)
