"""ANIE source guards, native coordinate/dropout fixes, and direct-forecast gradients."""
import copy
from dataclasses import asdict
import inspect
import os
from pathlib import Path

import pytest

from adapters import anie


@pytest.fixture
def source_root():
    """Use only the explicitly configured native ANIE checkout, otherwise skip."""
    configured = os.environ.get("GIFT_ANIE_SOURCE_ROOT")
    if not configured:
        pytest.skip("set GIFT_ANIE_SOURCE_ROOT to run pinned upstream ANIE checks")
    return Path(configured)


@pytest.fixture
def native(source_root):
    """Load the patched native modules with a CPU solver device for tensor checks."""
    torch = pytest.importorskip("torch")
    identity = anie.source_identity(source_root)
    modules = anie._native_components(identity)
    modules[2].device = torch.device("cpu")
    return modules


def test_read_only_source_identity_and_exact_patch_guards(source_root):
    """Keep source bytes untouched and reject changed anchors before in-memory patching."""
    lock = anie._lock()
    files = [name for name, entry in lock["files"].items() if entry["definitions"]]
    before = {name: (source_root / name).read_bytes() for name in files}
    identity = anie.source_identity(source_root)
    assert identity["commit"] == anie.PINNED_COMMIT
    assert len(identity["files"]) == 3
    assert identity == anie.source_identity(source_root)
    assert before == {name: (source_root / name).read_bytes() for name in files}
    texts = {name: (source_root / name).read_text(encoding="utf-8") for name in files}
    damaged = dict(texts)
    patch = lock["patches"][0]
    damaged[patch["file"]] = damaged[patch["file"]].replace(patch["before"], "changed_anchor", 1)
    with pytest.raises(ValueError, match="exact patch anchor"):
        anie._patch_sources(damaged, lock)
    assert [p["id"] for p in identity["patches"]] == [
        "spatial_time_token_alignment", "linear_attention_eval_dropout_mode"]


def test_fixed_configuration_and_initial_field_only_interface():
    """Keep the reference layout fixed and expose initial fields plus query times only."""
    config = anie.ANIEConfig()
    assert config.frames == 151 and asdict(config)["window_seconds"] == 3.0
    with pytest.raises(ValueError, match="fixed"):
        anie.ANIEConfig(window_seconds=1.0)
    assert tuple(inspect.signature(anie.ANIEPredictor.forward).parameters) == ("self", "initial", "times")


def test_native_space_time_coordinates_match_state_token_order(native):
    """Inspect token coordinates with distinguishable values at every space-time slot.

    The capture encoder returns zero so only alignment and the native solver
    layout are tested; it supplies no learned attention or accuracy result.
    """
    import torch

    class Capture(torch.nn.Module):
        """Record native encoder tokens while contributing a zero attention update."""

        def forward(self, values, dynamical_mask=None):
            """Retain detached tokens for inspection and return a same-layout zero tensor."""
            assert dynamical_mask is None
            self.values = values.detach().clone()
            return torch.zeros_like(values)

    capture = Capture()
    state = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3, 1)
    spatial = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    times = torch.tensor([0., 0.4, 1.])
    solver = native[2].Integral_spatial_attention_solver_multbatch(
        times, state[..., :1, :], y_init=state, c=lambda _: torch.zeros_like(state),
        Encoder=capture, sampling_points=3, max_iterations=3, spatial_integration=True,
        spatial_domain=spatial, spatial_domain_dim=2, use_support=False,
        accumulate_grads=True, initialization=True)
    output = solver._solve_rhs_with_known_y(state)
    expected_coordinates = spatial.repeat_interleave(3, dim=0).unsqueeze(0).repeat(2, 1, 1)
    expected_times = times.repeat(4).reshape(1, 12, 1).repeat(2, 1, 1)
    assert torch.equal(capture.values[:, :, :1], state.reshape(2, 12, 1))
    assert torch.equal(capture.values[:, :, 1:3], expected_coordinates)
    assert torch.equal(capture.values[:, :, 3:], expected_times)
    assert torch.equal(output, state)


def test_native_eval_dropout_disabled_and_training_probability_preserved(native):
    """Preserve native training dropout while making eval deterministic and RNG-neutral."""
    import torch

    q = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + 1
    k, v = q + 1, q + 2
    dropout = torch.nn.Dropout(0.1)
    base = k.transpose(-2, -1) @ v / 4
    dropout.train()
    torch.manual_seed(73)
    expected = torch.nn.functional.dropout(base, p=0.5, training=True)
    torch.manual_seed(73)
    _, actual = native[1].linear_attention(q, k, v, dropout=dropout)
    assert torch.equal(actual, expected)
    dropout.eval()
    rng = torch.get_rng_state().clone()
    first, weights = native[1].linear_attention(q, k, v, dropout=dropout)
    second, _ = native[1].linear_attention(q, k, v, dropout=dropout)
    assert torch.equal(weights, base) and torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng)


def test_native_sparse_grid_shape_and_original_autograd(source_root):
    """Run a ten-time CPU forecast and require finite nonzero gradients in all branches.

    This uses native encoder, attention and decoder bodies; random fields
    exercise differentiability rather than fitting or reproducing a dataset.
    """
    import torch

    torch.manual_seed(17)
    model = anie.ANIEPredictor(source_root=source_root).cpu().train()
    initial = torch.randn(1, 64, 64) * 0.1
    indices = torch.tensor([0, 16, 33, 50, 66, 83, 100, 116, 133, 150])
    physical_times = indices.float() * 0.02
    predicted = model(initial, times=physical_times)
    assert predicted.shape == (1, 10, 64, 64) and torch.isfinite(predicted).all()
    loss = torch.nn.functional.mse_loss(predicted, torch.zeros_like(predicted))
    loss.backward()
    for branch in (model.integral_operator, model.encoder, model.decoder):
        gradients = [p.grad for p in branch.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(bool((g != 0).any()) for g in gradients)
    # Native attention contains an allocated position-projection layer unused
    # when coordinates are concatenated in the input; its grad=None is retained.
    assert model.last_solver_iterations <= model.config.max_iterations
    with pytest.raises(ValueError, match="include 0 and 3"):
        model(initial, times=torch.tensor([0., 1.]))


def test_native_full_grid_eval_determinism_and_separate_initial_reconstruction(source_root):
    """Check a native 151-time CUDA forecast without overwriting its raw initial output."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("151-frame native integration smoke uses CUDA; sparse-grid CPU test covers gradients")
    torch.manual_seed(23)
    model = anie.ANIEPredictor(source_root=source_root).cuda().eval()
    initial = torch.randn(1, 64, 64, device="cuda") * 0.1
    saved = initial.clone()
    with torch.no_grad():
        first = model(initial)
        second = model(initial)
        reconstructed = model.initial_decoder(initial)
    assert first.shape == (1, 151, 64, 64) and torch.isfinite(first).all()
    assert torch.equal(first, second) and torch.equal(initial, saved)
    assert reconstructed.shape == (1, 64, 64)
    assert not torch.equal(first[:, 0], initial), "raw ANIE t=0 must not be replaced by the observed anchor"
