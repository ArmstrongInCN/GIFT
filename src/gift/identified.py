from __future__ import annotations

from pathlib import Path

import torch

from .model import GIFTGenerator


def load_release_payload(artifact: str | Path) -> dict:
    """Load and validate a frozen GIFT artifact payload on CPU."""

    payload = torch.load(Path(artifact), map_location="cpu", weights_only=True)
    if payload.get("format_version") not in (2, 3):
        raise ValueError("unsupported GIFT artifact format")
    if "model_configuration" not in payload or "model_state_dict" not in payload:
        raise ValueError("incomplete GIFT artifact")
    return payload


def load_generator(
    artifact: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> GIFTGenerator:
    """Load a frozen GIFT generator artifact."""
    payload = load_release_payload(artifact)
    model = GIFTGenerator(**payload["model_configuration"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(torch.device(device)).eval()


class FixedBandwidthGridGenerator:
    """Evaluate a frozen generator on another grid with fixed Fourier support.

    ``reference_grid`` is the grid on which the model's unnormalised constant
    DFT table was identified.  Only that complete state-independent table is
    converted by ``(grid_size / reference_grid) ** 2``; the linear multiplier
    and quadratic factors are unchanged.  ``constant_factor`` is an explicit
    override reserved for controlled representation-unit ablations; normal
    inference should leave it as ``None``.
    """

    def __init__(
        self,
        model: GIFTGenerator,
        grid_size: int,
        *,
        reference_grid: int,
        constant_factor: float | None = None,
    ) -> None:
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
        expected_factor = (self.grid_size / self.reference_grid) ** 2
        factor = expected_factor if constant_factor is None else float(constant_factor)
        if not bool(torch.isfinite(torch.tensor(factor))) or factor < 0.0:
            raise ValueError("constant_factor must be finite and nonnegative")
        self.constant_factor = factor
        self._parameter_versions = tuple(
            parameter._version for parameter in self.model.parameters()
        )

    @property
    def grid(self) -> int:
        """Compatibility alias for the evaluation grid size."""

        return self.grid_size

    @torch.inference_mode()
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
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
    """Integrate the learned continuous-time generator with explicit RK4."""
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
