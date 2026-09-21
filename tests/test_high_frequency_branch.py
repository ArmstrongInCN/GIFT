"""Unit tests for the high-frequency branch.

The branch predicts the complementary spectral band and is coupled with the
frozen generator inside one RK4 integration. These tests cover the released
configuration and its loader guards, the requirement that the branch output
lives in the complementary band on every supported grid, the state-dependent
action above the retained mode count, preservation of the anchor when the branch
is disabled, and reuse of the generator output within one stage.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from experiments.formal._shared.gift_runtime import rollout_gift
from experiments.formal._shared.high_frequency import (
    HighFrequencyBranch,
    HighFrequencyConfig,
    load_high_frequency_model,
    square_mask,
)


def _zero_generator(state: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(state)


def _model() -> HighFrequencyBranch:
    torch.manual_seed(7)
    return HighFrequencyBranch(
        HighFrequencyConfig(),
        state_scale=4.0,
        high_scale=0.1,
        low_rhs_scale=10.0,
        high_rhs_scale=1.0,
        frozen_generator=_zero_generator,
    ).eval()


def _released_payload() -> dict[str, object]:
    model = _model()
    return {
        "schema": "gift.high_frequency_branch.parameters.v2",
        "method": "GIFT",
        "model_config": model.export_config(),
        "state_scale": 4.0,
        "high_scale": 0.1,
        "low_rhs_scale": 10.0,
        "high_rhs_scale": 1.0,
        "model_state": model.state_dict(),
        "support": {
            "low_generator": "P21",
            "learned_rhs": "complete Q21 through incoming-grid Nyquist support",
            "common_explicit_source": "P22-P32",
            "coefficient_logits": "P12 before nonlinear activation",
            "modes_above_32": "state-conditioned local differential action",
        },
        "selection_data_resolution": 64,
        "fixed_upper_output_cutoff": None,
        "equation_terms": [],
        "memory_or_hidden_state": False,
        "frozen_generator_sha256": "A" * 64,
        "seed": 7,
        "training_seed": 7,
        "phase": "test",
        "epoch": 1,
        "selection_metric": "test",
    }


def test_released_configuration_is_fixed() -> None:
    HighFrequencyConfig()
    with pytest.raises(ValueError):
        HighFrequencyConfig(modes=20)
    with pytest.raises(ValueError):
        HighFrequencyConfig(coefficient_cutoff=21)


@pytest.mark.parametrize(
    "missing_key",
    ("schema", "fixed_upper_output_cutoff", "frozen_generator_sha256"),
)
def test_loader_rejects_incomplete_release_metadata(tmp_path, missing_key: str) -> None:
    payload = _released_payload()
    payload.pop(missing_key)
    path = tmp_path / f"missing_{missing_key}.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing"):
        load_high_frequency_model(path, torch.device("cpu"), _zero_generator)


@pytest.mark.parametrize("grid", (64, 96, 128))
def test_output_is_q21_on_each_supported_grid(grid: int) -> None:
    model = _model()
    state = torch.randn(2, grid, grid)
    value = model.forward_with_generator_output(state, torch.zeros_like(state))
    p21 = square_mask(grid, 21, device=state.device, dtype=state.dtype)
    low_part = torch.fft.ifft2(torch.fft.fft2(value) * p21).real
    assert value.shape == state.shape
    assert torch.isfinite(value).all()
    assert float(low_part.detach().abs().max()) < 2.0e-5


def test_modes_above_31_receive_state_dependent_action() -> None:
    grid = 96
    model = _model()
    coordinate = torch.arange(grid, dtype=torch.float32) / grid
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    state = torch.sin(2.0 * torch.pi * 40.0 * xx)[None]
    value = model.forward_with_generator_output(state, torch.zeros_like(state))
    p31 = square_mask(grid, 31, device=state.device, dtype=state.dtype)
    outer = torch.fft.ifft2(torch.fft.fft2(value) * (1.0 - p31)).real
    assert float(outer.detach().square().mean()) > 1.0e-12


def test_disabled_branch_preserves_q21_anchor() -> None:
    grid = 64
    coordinate = np.arange(grid, dtype=np.float32) / grid
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    anchor = (
        np.sin(2.0 * np.pi * 3.0 * xx)
        + 0.1 * np.sin(2.0 * np.pi * 25.0 * yy)
    )[None].astype(np.float32)
    result = rollout_gift(
        low=_zero_generator,
        branch=None,
        initial_state=anchor,
        trajectory_ids=np.asarray([1], dtype=np.int64),
        duration=0.02,
        save_interval=0.02,
        device=torch.device("cpu"),
        batch_size=1,
        correction=False,
    )
    np.testing.assert_allclose(result.prediction[:, 1], anchor, rtol=0.0, atol=2e-6)
    assert result.runtime["low_rhs_evaluations_by_trajectory"] == [4]
    assert result.runtime["high_frequency_rhs_evaluations_by_trajectory"] == [0]


def test_same_stage_generator_output_is_reused_by_enabled_branch() -> None:
    class CountingGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, state: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return state.square() + 1.0

    class ZeroBranch:
        config = SimpleNamespace(cutoff=21)

        def __init__(self) -> None:
            self.calls = 0

        def forward_with_generator_output(
            self, state: torch.Tensor, low_rhs: torch.Tensor
        ) -> torch.Tensor:
            self.calls += 1
            p21 = square_mask(
                int(state.shape[-1]),
                21,
                device=state.device,
                dtype=state.dtype,
            )
            expected = torch.fft.ifft2(
                torch.fft.fft2(state.square() + 1.0) * p21
            ).real
            torch.testing.assert_close(low_rhs, expected, rtol=0.0, atol=2.0e-6)
            return torch.zeros_like(state)

    generator = CountingGenerator()
    branch = ZeroBranch()
    anchor = np.zeros((1, 64, 64), dtype=np.float32)
    result = rollout_gift(
        low=generator,
        branch=branch,  # type: ignore[arg-type]
        initial_state=anchor,
        trajectory_ids=np.asarray([1], dtype=np.int64),
        duration=0.02,
        save_interval=0.02,
        device=torch.device("cpu"),
        batch_size=1,
        correction=False,
    )
    assert generator.calls == 4
    assert branch.calls == 4
    assert result.branch_audit[
        "single_frozen_generator_evaluation_reused_per_stage"
    ] is True
