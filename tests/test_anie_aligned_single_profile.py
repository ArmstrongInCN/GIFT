"""CPU regressions for an explicit coordinate-only NS source profile.

Distinct values at every formal76-grid token prove labels correspond to state.
Real native modules cover gradients, original dropout, network equality and
unchanged historical source metadata, without using experiment data or weights.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from adapters.anie_config import ANIEConfig, source_identity


PROFILE = "aligned_ns_single_v1"
# Canonical historical identities captured before introducing this profile.
# The caller's absolute checkout directory is excluded for portability.
HISTORICAL_IDENTITY_DIGEST = {
    "aligned_legacy": "63450cd810887c1cad79438bc1d56e561bcf0ace9dafed0f7c83ec4e3e8b5ec5",
    "native_ns_single_v1": "ae7b3ef70e569c64d7bc4d6e43ddde89fbbfa1c398629d947b5506c6a808e36a",
}


@pytest.fixture(scope="module")
def torch_cpu():
    """Keep native test forwards and derivatives on four CPU threads."""
    torch = pytest.importorskip("torch")
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield torch
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def source_root():
    """Use an explicitly configured official checkout, without modifying it."""
    root = os.environ.get("GIFT_ANIE_SOURCE_ROOT")
    if not root:
        pytest.skip("set GIFT_ANIE_SOURCE_ROOT for native ANIE source tests")
    return Path(root)


@pytest.fixture(scope="module")
def aligned_native(source_root, torch_cpu):
    """Compile the existing two in-memory patches under the new explicit profile."""
    from adapters.anie import _native_components

    modules = _native_components(source_identity(source_root, source_profile=PROFILE))
    modules[2].device = torch_cpu.device("cpu")
    return modules


def test_config_preserves_defaults_and_requires_native_condition():
    """The new profile changes coordinates, not the NS conditioning equation."""
    assert ANIEConfig().source_profile == "aligned_legacy"
    assert ANIEConfig(source_profile=PROFILE).condition_mode == "native"
    assert ANIEConfig(source_profile=PROFILE).frames == 151
    for profile in (PROFILE, "native_ns_single_v1"):
        with pytest.raises(ValueError, match="requires the native NS condition branch"):
            ANIEConfig(source_profile=profile, condition_mode="fixed_f")
    assert ANIEConfig(source_profile="aligned_legacy", condition_mode="fixed_f").condition_mode == "fixed_f"


@pytest.mark.parametrize("profile", tuple(HISTORICAL_IDENTITY_DIGEST))
def test_old_profile_metadata_is_exactly_unchanged(source_root, profile):
    """Protect checkpoint provenance for both legacy and original-coordinate runs."""
    identity = source_identity(source_root, source_profile=profile)
    identity.pop("directory")
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == HISTORICAL_IDENTITY_DIGEST[profile]


def test_aligned_profile_selects_only_existing_patches_and_native_control(source_root):
    """Require the exact registered patches and unchanged native training source."""
    from adapters.anie_config import _lock

    lock = _lock()
    identity = source_identity(source_root, source_profile=PROFILE)
    native = source_identity(source_root, source_profile="native_ns_single_v1")
    legacy = source_identity(source_root, source_profile="aligned_legacy")
    assert identity["schema"] == "gift.anie-source-metadata.v3"
    assert identity["source_profile"] == PROFILE
    assert identity["patches"] == legacy["patches"] == lock["patches"]
    assert [patch["id"] for patch in identity["patches"]] == [
        "spatial_time_token_alignment", "linear_attention_eval_dropout_mode"]
    assert identity["training_control_source"] == native["training_control_source"]
    assert identity["files"] == native["files"]


def test_every_formal76_token_matches_state_space_time_and_keeps_gradients(aligned_native, torch_cpu):
    """Verify all19456 space/time slots per batch, plus state and coordinate derivatives.

    A simple differentiable native-operator substitute makes the exact RHS
    derivative known. It tests token preparation and the existing residual;
    the separate native-network test covers actual attention autograd.
    """
    torch = torch_cpu
    batch, side, count, channels = 2, 16, 76, 2
    volume = side**2
    state = torch.arange(batch*volume*count*channels, dtype=torch.float32).reshape(
        batch, side, side, count, channels).requires_grad_()
    axis = torch.arange(side, dtype=torch.float32) / (side-1)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    space = torch.stack((y.flatten(), x.flatten()), dim=-1).requires_grad_()
    times = (torch.arange(count, dtype=torch.float32)/(count-1)).requires_grad_()

    class Capture(torch.nn.Module):
        def forward(self, values, dynamical_mask=None):
            self.values = values
            assert dynamical_mask is None
            return .1 * values.sum(-1, keepdim=True).expand_as(values)

    capture = Capture()
    solver = aligned_native[2].Integral_spatial_attention_solver_multbatch(
        times, state[..., :1, :], y_init=state, c=lambda _: torch.zeros_like(state),
        Encoder=capture, sampling_points=count, max_iterations=3, spatial_integration=True,
        spatial_domain=space, spatial_domain_dim=2, use_support=False,
        accumulate_grads=True, initialization=True)
    result = solver._solve_rhs_with_known_y(state)
    expected_state = state.reshape(batch, volume*count, channels)
    expected_space = space[:, None, :].expand(volume, count, 2).reshape(volume*count, 2)
    expected_time = times[None, :, None].expand(volume, count, 1).reshape(volume*count, 1)
    assert torch.equal(capture.values[..., :channels], expected_state)
    assert torch.equal(capture.values[..., channels:channels+2], expected_space[None].expand(batch, -1, -1))
    assert torch.equal(capture.values[..., channels+2:], expected_time[None].expand(batch, -1, -1))
    # Explicit indices disambiguate batch, flattened spatial location, time,
    # and channel: coordinate correction must not permute the actual state.
    for b in range(batch):
        for p in range(volume):
            begin = p*count
            assert torch.equal(capture.values[b, begin:begin+count, :channels], state[b, p//side, p%side])
    result.sum().backward()
    assert torch.allclose(state.grad, torch.full_like(state, 1.+.1*channels), rtol=0, atol=1e-6)
    assert torch.allclose(space.grad, torch.full_like(space, .1*channels*batch*count), rtol=0, atol=1e-4)
    assert torch.allclose(times.grad, torch.full_like(times, .1*channels*batch*volume), rtol=0, atol=1e-3)


def test_original_training_dropout_probability_and_eval_determinism(aligned_native, torch_cpu):
    """Training uses functionalp=.5 even with modulep=.1; eval is RNG-neutral."""
    torch = torch_cpu
    q = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)+1
    k, v = q+1, q+2
    dropout = torch.nn.Dropout(.1)
    base = k.transpose(-2,-1) @ v / 4
    dropout.train()
    torch.manual_seed(73)
    expected = torch.nn.functional.dropout(base, p=.5, training=True)
    torch.manual_seed(73)
    _, actual = aligned_native[1].linear_attention(q, k, v, dropout=dropout)
    assert torch.equal(actual, expected)
    dropout.eval()
    rng = torch.get_rng_state().clone()
    first, weights = aligned_native[1].linear_attention(q, k, v, dropout=dropout)
    second, _ = aligned_native[1].linear_attention(q, k, v, dropout=dropout)
    assert torch.equal(weights, base) and torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("training", [False, True])
def test_real_native_network_matches_legacy_alignment_and_reaches_encoder(source_root, torch_cpu, training):
    """Keep networks and native c0 computation equal to the established aligned profile."""
    from adapters.anie import ANIEPredictor

    torch = torch_cpu
    paths = [source_root/name for name in source_identity(source_root, source_profile=PROFILE)["files"]]
    original_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    torch.manual_seed(7)
    model = ANIEPredictor(ANIEConfig(source_profile=PROFILE), source_root=source_root).cpu().train(training)
    torch.manual_seed(7)
    legacy = ANIEPredictor(ANIEConfig(source_profile="aligned_legacy"), source_root=source_root).cpu().train(training)
    torch.manual_seed(7)
    original = ANIEPredictor(ANIEConfig(source_profile="native_ns_single_v1"), source_root=source_root).cpu()
    assert sum(p.numel() for p in model.parameters()) == 1737891
    assert model.state_dict().keys() == legacy.state_dict().keys() == original.state_dict().keys()
    assert all(torch.equal(value, legacy.state_dict()[name]) and torch.equal(value, original.state_dict()[name])
               for name,value in model.state_dict().items())
    initial = (torch.randn(1,64,64)*.1).requires_grad_()
    times = torch.tensor([0.,.6,1.2,1.8,2.4,3.])
    torch.manual_seed(82)
    actual = model(initial, times=times)
    with torch.no_grad():
        torch.manual_seed(82)
        reference = legacy(initial.detach(), times=times)
    assert torch.equal(actual, reference)
    assert actual.shape == (1,6,64,64) and bool(torch.isfinite(actual).all())
    actual[:,1:].square().mean().backward()
    assert initial.grad is not None and bool(torch.isfinite(initial.grad).all()) and bool((initial.grad!=0).any())
    for branch in (model.encoder,model.integral_operator,model.decoder):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
        assert any(bool((g!=0).any()) for g in grads)
    assert original_hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
