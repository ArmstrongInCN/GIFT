"""Resolution-adaptive high-frequency branch used by formal GIFT experiments.

The branch receives the complete RK4 stage state together with the output of
the frozen P21 GIFT generator evaluated at that same state. Its spectral-
convolution backbone decodes smooth coefficient fields for a compact local
differential action on Q21(state), plus a source on the N64-common band
P22--P32. The final
right-hand side is projected only by Q21: there is no model-fixed upper output
cutoff, and the same learned parameters operate on every supported grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HighFrequencyConfig:
    """Fixed architecture of the released GIFT high-frequency branch."""

    cutoff: int = 21
    common_source_cutoff: int = 32
    coefficient_cutoff: int = 12
    dealiased_products: bool = False
    modes: int = 21
    width: int = 20
    layers: int = 4
    decoder_width: int = 64
    head: str = "transport"

    def __post_init__(self) -> None:
        """Require the implemented transport-head architecture before constructing its layers."""
        expected = (21, 32, 12, False, 21, 20, 4, 64, "transport")
        actual = (
            self.cutoff,
            self.common_source_cutoff,
            self.coefficient_cutoff,
            self.dealiased_products,
            self.modes,
            self.width,
            self.layers,
            self.decoder_width,
            self.head,
        )
        if actual != expected:
            raise ValueError("high-frequency model architecture differs from GIFT")


def square_mask(
    grid: int,
    cutoff: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the incoming grid's square integer-wavenumber support as a tensor mask."""
    modes = torch.fft.fftfreq(grid, d=1.0 / grid, device=device, dtype=dtype)
    ky, kx = torch.meshgrid(modes, modes, indexing="ij")
    return ((kx.abs() <= cutoff) & (ky.abs() <= cutoff)).to(dtype)


def project(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a periodic Fourier projection on the final two spatial axes."""
    return torch.fft.ifft2(torch.fft.fft2(field) * mask).real


class SpectralConvolution2d(nn.Module):
    """Low-mode Fourier integral operator for the spectral backbone."""

    def __init__(self, channels: int, modes: int) -> None:
        """Allocate independent complex channel-mixing weights for positive and negative y modes."""
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / (channels * channels)
        shape = (channels, channels, modes, modes)
        self.weight_positive = nn.Parameter(
            scale * torch.rand(*shape, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.rand(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def multiply(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Contract input channels [batch,in,ky,kx] with [in,out,ky,kx] complex weights."""
        return torch.einsum("bixy,ioxy->boxy", values, weights)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Transform [batch,channels,Ny,Nx] fields, mix retained modes and return the original grid."""
        grid_y, grid_x = field.shape[-2:]
        if 2 * self.modes > grid_y or self.modes > grid_x // 2 + 1:
            raise ValueError("configured spectral modes exceed the incoming grid")
        field_hat = torch.fft.rfft2(field)
        output_hat = torch.zeros(
            field.shape[0],
            self.channels,
            grid_y,
            grid_x // 2 + 1,
            device=field.device,
            dtype=field_hat.dtype,
        )
        output_hat[:, :, : self.modes, : self.modes] = self.multiply(
            field_hat[:, :, : self.modes, : self.modes], self.weight_positive
        )
        output_hat[:, :, -self.modes :, : self.modes] = self.multiply(
            field_hat[:, :, -self.modes :, : self.modes], self.weight_negative
        )
        return torch.fft.irfft2(output_hat, s=(grid_y, grid_x))


class SpectralResidualBlock(nn.Module):
    """Combine a low-mode spectral operator with a pointwise channel map."""
    def __init__(self, width: int, modes: int, *, activate: bool) -> None:
        """Create spectral/pointwise paths and choose whether this block applies GELU."""
        super().__init__()
        self.spectral = SpectralConvolution2d(width, modes)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.activate = activate

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Add both channel maps, applying GELU only on configured intermediate blocks."""
        value = self.spectral(field) + self.pointwise(field)
        return F.gelu(value) if self.activate else value


class HighFrequencyBranch(nn.Module):
    """Generator-conditioned, grid-adaptive Q21 right-hand-side model."""

    def __init__(
        self,
        config: HighFrequencyConfig,
        *,
        state_scale: float,
        high_scale: float,
        low_rhs_scale: float,
        high_rhs_scale: float,
        frozen_generator: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Construct the conditioned Q21 branch with scales measured from training fields.

        The frozen generator is deliberately outside state_dict: branch exports store
        its weight binding, while trainable layers and scale buffers belong to the branch.
        """
        super().__init__()
        self.config = config
        for name, value in (
            ("state_scale", state_scale),
            ("high_scale", high_scale),
            ("low_rhs_scale", low_rhs_scale),
            ("high_rhs_scale", high_rhs_scale),
        ):
            if not float(value) > 0.0:
                raise ValueError(f"{name} must be positive")
            self.register_buffer(name, torch.tensor(float(value)))
        self.register_buffer(
            "rate_scale", torch.tensor(float(high_rhs_scale) / float(high_scale))
        )
        # The identified generator is an external frozen dependency and is not
        # embedded in this branch's trainable state_dict.
        self.__dict__["_frozen_generator"] = frozen_generator
        self.lift = nn.Conv2d(3, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                SpectralResidualBlock(
                    config.width,
                    config.modes,
                    activate=index + 1 < config.layers,
                )
                for index in range(config.layers)
            ]
        )
        self.decode1 = nn.Conv2d(config.width, config.decoder_width, 1)
        self.decode2 = nn.Conv2d(config.decoder_width, 5, 1)
        self._mask_cache: dict[
            tuple[int, str, int | None, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._cache_transport_grids = False
        self._transport_grids = {}

    def masks(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache P21, Q21 and common-source masks for [batch,N,N] state tensors.

        The cache key includes grid, device and dtype; training-safe tensors are built
        outside inference mode so reuse cannot invalidate autograd.
        """
        if state.ndim != 3 or state.shape[-1] != state.shape[-2]:
            raise ValueError("state must have shape [batch,N,N]")
        grid = int(state.shape[-1])
        if grid < 2 * self.config.cutoff + 1:
            raise ValueError("grid cannot represent P21")
        key = (grid, state.device.type, state.device.index, state.dtype)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        with torch.inference_mode(False), torch.no_grad():
            low = square_mask(
                grid, self.config.cutoff, device=state.device, dtype=state.dtype
            )
            common = square_mask(
                grid,
                self.config.common_source_cutoff,
                device=state.device,
                dtype=state.dtype,
            ) * (1.0 - low)
            result = (low, 1.0 - low, common)
        self._mask_cache[key] = result
        return result

    def split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the P21 and Q21 components of a full [batch,N,N] state."""
        low_mask, high_mask, _ = self.masks(state)
        state_hat = torch.fft.fft2(state)
        return (
            torch.fft.ifft2(state_hat * low_mask).real,
            torch.fft.ifft2(state_hat * high_mask).real,
        )

    def enable_static_cache(self, enabled: bool = True) -> None:
        """Reuse constant transport grids, never learned coefficient fields."""
        self._cache_transport_grids = bool(enabled)
        self._transport_grids = {}
        # Discard masks possibly created under inference_mode before training.
        self._mask_cache.clear()

    def transport_grid(self, state: torch.Tensor):
        """Return integer wave numbers and the P12 coefficient mask on the state's grid."""
        grid = int(state.shape[-1])
        key = (grid, state.device, state.dtype)
        enabled = self._cache_transport_grids
        cached = self._transport_grids.get(key) if enabled else None
        if cached is None:
            with torch.inference_mode(False), torch.no_grad():
                frequencies = torch.fft.fftfreq(
                    grid, d=1.0 / grid, device=state.device, dtype=state.dtype
                )
                ky, kx = torch.meshgrid(frequencies, frequencies, indexing="ij")
                mask = square_mask(grid, self.config.coefficient_cutoff,
                                   device=state.device, dtype=state.dtype)
            cached = ky, kx, mask
            if enabled:
                self._transport_grids[key] = cached
        return cached

    def project_q21(self, field: torch.Tensor) -> torch.Tensor:
        """Remove P21 modes from a physical field while retaining the incoming Q21 support."""
        return project(field, self.masks(field)[1])

    def frozen_generator_output(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the fixed generator without gradients and project its output onto P21."""
        with torch.no_grad():
            value = self.__dict__["_frozen_generator"](state.detach())
        if value.shape != state.shape or value.device != state.device:
            raise ValueError("frozen generator output is incompatible with state")
        if value.is_inference():
            value = value.clone()
        return project(value, self.masks(state)[0])

    def forward_with_generator_output(
        self, state: torch.Tensor, low_rhs: torch.Tensor, *, _training_kernels=None
    ) -> torch.Tensor:
        """Map same-state vorticity and frozen P21 RHS to a [batch,N,N] Q21 derivative.

        The backbone decodes bounded rate, velocity and diffusivity fields plus a
        common-band source. Gradients/Laplacian act on Q21(state), so the learned local
        action extends to modes above the source band on larger grids. Training kernels
        change execution of the same operators, not their spectral support.
        """
        if low_rhs.shape != state.shape:
            raise ValueError("low_rhs shape differs from state")
        low_mask, high_mask, common_mask = self.masks(state)
        if low_rhs.is_inference():
            low_rhs = low_rhs.clone()
        low_rhs = project(low_rhs, low_mask)
        high_state = project(state, high_mask)
        inputs = torch.stack(
            (
                state / self.state_scale,
                high_state / self.high_scale,
                low_rhs / self.low_rhs_scale,
            ),
            dim=1,
        )
        features = self.lift(inputs)
        for block in self.blocks:
            features = block(features) if _training_kernels is None else _training_kernels.block(block, features)
        decoded = self.decode2(F.gelu(self.decode1(features)))

        ky, kx, coefficient_mask = self.transport_grid(state)
        high_hat = torch.fft.fft2(high_state)
        normalizer = float(self.config.common_source_cutoff)
        gradient_x = torch.fft.ifft2(1j * kx * high_hat).real / (
            normalizer * self.high_scale
        )
        gradient_y = torch.fft.ifft2(1j * ky * high_hat).real / (
            normalizer * self.high_scale
        )
        laplacian = torch.fft.ifft2(-(kx * kx + ky * ky) * high_hat).real / (
            normalizer * normalizer * self.high_scale
        )
        if _training_kernels is None:
            rate = torch.tanh(project(decoded[:, 0], coefficient_mask))
            velocity_x = torch.tanh(project(decoded[:, 1], coefficient_mask))
            velocity_y = torch.tanh(project(decoded[:, 2], coefficient_mask))
            diffusivity = torch.sigmoid(project(decoded[:, 3], coefficient_mask))
        else:
            # Four independent real coefficient fields share one batched FFT.
            rate, velocity_x, velocity_y, diffusivity = project(decoded[:, :4], coefficient_mask).unbind(1)
            rate, velocity_x, velocity_y = torch.tanh(rate), torch.tanh(velocity_x), torch.tanh(velocity_y)
            diffusivity = torch.sigmoid(diffusivity)
        common_source = project(decoded[:, 4] * self.high_rhs_scale, common_mask)
        transported = self.high_rhs_scale * (
            rate * (high_state / self.high_scale)
            + velocity_x * gradient_x
            + velocity_y * gradient_y
            + diffusivity * laplacian
        )
        return project(transported + common_source, high_mask)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the frozen same-state conditioning RHS before the high-frequency action."""
        return self.forward_with_generator_output(
            state, self.frozen_generator_output(state)
        )

    def export_config(self) -> dict[str, Any]:
        """Return the architecture settings needed to reconstruct the branch from its weights."""
        return asdict(self.config)


def load_high_frequency_model(
    path: Path,
    device: torch.device,
    frozen_generator: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[HighFrequencyBranch, dict[str, Any]]:
    """Strictly load a released generator-conditioned GIFT branch."""

    resolved = Path(path).resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("GIFT model parameter file must contain a mapping")
    required = {
        "schema",
        "method",
        "model_config",
        "state_scale",
        "high_scale",
        "low_rhs_scale",
        "high_rhs_scale",
        "model_state",
        "support",
        "selection_data_resolution",
        "fixed_upper_output_cutoff",
        "equation_terms",
        "memory_or_hidden_state",
        "seed",
        "training_seed",
        "phase",
        "epoch",
        "selection_metric",
    }
    if not required.issubset(payload):
        missing = sorted(required - set(payload))
        raise ValueError(f"GIFT model parameter file is missing: {missing}")
    if payload["schema"] != "gift.high_frequency_branch.parameters.v2":
        raise ValueError("GIFT high-frequency model schema differs")
    if payload["method"] != "GIFT":
        raise ValueError("high-frequency parameter file method differs from GIFT")
    if payload["fixed_upper_output_cutoff"] is not None:
        raise ValueError("GIFT high-frequency model has a fixed upper output cutoff")
    if int(payload["selection_data_resolution"]) != 64:
        raise ValueError("GIFT high-frequency model was not selected on N64 data")
    if payload["equation_terms"] != []:
        raise ValueError("high-frequency model contains an equation-specific term")
    if payload["memory_or_hidden_state"] is not False:
        raise ValueError("high-frequency model contains an unsupported memory state")
    expected_support = {
        "low_generator": "P21",
        "learned_rhs": "complete Q21 through incoming-grid Nyquist support",
        "common_explicit_source": "P22-P32",
        "coefficient_logits": "P12 before nonlinear activation",
        "modes_above_32": "state-conditioned local differential action",
    }
    if (
        not isinstance(payload["support"], Mapping)
        or dict(payload["support"]) != expected_support
    ):
        raise ValueError("GIFT high-frequency model support metadata differs")
    frozen_hash = payload.get("generator_weights_sha256") or payload.get("frozen_generator_sha256")
    if frozen_hash is None:
        raise ValueError("GIFT model parameter file is missing its frozen-generator binding")
    frozen_hash = str(frozen_hash).upper()
    if len(frozen_hash) != 64 or any(
        character not in "0123456789ABCDEF" for character in frozen_hash
    ):
        raise ValueError("GIFT high-frequency model frozen-generator hash is invalid")
    if int(payload["seed"]) != int(payload["training_seed"]):
        raise ValueError("GIFT high-frequency model seed metadata differs")
    config = HighFrequencyConfig(**dict(payload["model_config"]))
    model = HighFrequencyBranch(
        config,
        state_scale=float(payload["state_scale"]),
        high_scale=float(payload["high_scale"]),
        low_rhs_scale=float(payload["low_rhs_scale"]),
        high_rhs_scale=float(payload["high_rhs_scale"]),
        frozen_generator=frozen_generator,
    )
    model.load_state_dict(dict(payload["model_state"]), strict=True)
    model.requires_grad_(False).to(device).eval()
    if any(
        not bool(torch.isfinite(value).all()) for value in model.state_dict().values()
    ):
        raise ValueError("GIFT high-frequency model contains nonfinite values")
    return model, dict(payload)
