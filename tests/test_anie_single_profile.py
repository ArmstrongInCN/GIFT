"""Keep old ANIE coordinate behavior while the single-native run restores author tokens."""
import os
from pathlib import Path

import pytest
import torch

from adapters import anie
from adapters.anie_config import ANIEConfig, source_identity


@pytest.fixture
def upstream():
    path = os.environ.get("GIFT_ANIE_SOURCE_ROOT")
    if not path:
        pytest.skip("configure the read-only official ANIE checkout")
    return Path(path)


def test_single_profile_removes_only_alignment_and_preserves_legacy(upstream):
    """New explicit metadata names one patch; old default retains its two patches."""
    before = {name: (upstream / name).read_bytes() for name in anie._lock()["files"]}
    legacy = source_identity(upstream)
    native = source_identity(upstream, source_profile="native_ns_single_v1")
    assert "source_profile" not in legacy
    assert [patch["id"] for patch in legacy["patches"]] == [
        "spatial_time_token_alignment", "linear_attention_eval_dropout_mode"]
    assert [patch["id"] for patch in native["patches"]] == ["linear_attention_eval_dropout_mode"]
    assert native["source_profile"] == "native_ns_single_v1"
    assert anie._native_components(legacy)[2] is not anie._native_components(native)[2]
    assert before == {name: (upstream / name).read_bytes() for name in before}
    with pytest.raises(ValueError, match="source_profile"):
        source_identity(upstream, source_profile="unknown")
    with pytest.raises(ValueError, match="native NS"):
        ANIEConfig(source_profile="native_ns_single_v1", condition_mode="fixed_f")


def test_single_native_solver_receives_exact_author_coordinate_order(upstream):
    """Capture original repeated space and sorted-time tokens, without training a model."""
    modules = anie._native_components(source_identity(upstream, source_profile="native_ns_single_v1"))
    modules[2].device = torch.device("cpu")

    class Capture(torch.nn.Module):
        def forward(self, values, dynamical_mask=None):
            self.values = values.detach().clone()
            return torch.zeros_like(values)

    capture = Capture()
    state = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3, 1)
    spatial = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    times = torch.tensor([0., .4, 1.])
    solver = modules[2].Integral_spatial_attention_solver_multbatch(
        times, state[..., :1, :], y_init=state, c=lambda _: torch.zeros_like(state),
        Encoder=capture, sampling_points=3, max_iterations=3, spatial_integration=True,
        spatial_domain=spatial, spatial_domain_dim=2, use_support=False,
        accumulate_grads=True, initialization=True)
    result = solver._solve_rhs_with_known_y(state)
    assert torch.equal(capture.values[:, :, :1], state.reshape(2, 12, 1))
    assert torch.equal(capture.values[:, :, 1:3], spatial.repeat(3, 1).unsqueeze(0).repeat(2, 1, 1))
    assert torch.equal(capture.values[:, :, 3:], times.repeat(4).sort()[0].reshape(1, 12, 1).repeat(2, 1, 1))
    assert torch.equal(result, state)


def test_single_profile_keeps_eval_dropout_deterministic_and_native_training_p(upstream):
    """Only eval randomness changes; training uses the author's default dropout .5."""
    galerkin = anie._native_components(source_identity(upstream, source_profile="native_ns_single_v1"))[1]
    query = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + 1
    key, value = query + 1, query + 2
    dropout = torch.nn.Dropout(.1)
    base = key.transpose(-2, -1) @ value / 4
    torch.manual_seed(73)
    expected = torch.nn.functional.dropout(base)
    torch.manual_seed(73)
    _, actual = galerkin.linear_attention(query, key, value, dropout=dropout)
    assert torch.equal(actual, expected)
    dropout.eval()
    rng = torch.get_rng_state().clone()
    first, weights = galerkin.linear_attention(query, key, value, dropout=dropout)
    second, _ = galerkin.linear_attention(query, key, value, dropout=dropout)
    assert torch.equal(first, second) and torch.equal(weights, base)
    assert torch.equal(rng, torch.get_rng_state())


def test_single_native_sparse_cpu_forecast_retains_all_branch_gradients(upstream):
    """Exercise the new build path on CPU without any training data or CUDA call."""
    from training.w1_single_train import build_network
    torch.manual_seed(7)
    model = build_network(dict(method="anie_ns", history=1, condition_mode="native",
                          source_profile="native_ns_single_v1", source_root=str(upstream)), "cpu")
    initial = torch.randn(1, 64, 64) * .1
    predicted = model(initial, times=torch.tensor([0., 1.5, 3.]))
    predicted.square().mean().backward()
    assert predicted.shape == (1, 3, 64, 64) and torch.isfinite(predicted).all()
    assert model.source_identity()["source_profile"] == "native_ns_single_v1"
    for branch in (model.encoder, model.integral_operator, model.decoder):
        gradients = [parameter.grad for parameter in branch.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        assert any(bool((value != 0).any()) for value in gradients)
