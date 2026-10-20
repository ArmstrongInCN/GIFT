"""Native CPU recurrence verifies checkpoint values, gradients and BN state."""
import copy
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_fno():
    """Build only the explicitly selected native single-field FNO2D on CPU."""
    torch = pytest.importorskip("torch")
    configured = os.environ.get("GIFT_EXTERNAL_ROOT")
    if not configured:
        pytest.skip("set GIFT_EXTERNAL_ROOT for native FNO2D checkpoint checks")
    from adapters.models import build_model
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    model = build_model("fno2d", external_root=Path(configured), history=1).cpu().train()
    yield torch, model
    torch.set_num_threads(previous)


def _closed_loop(model, initial, call, torch):
    """Retain all three native dependencies; no target is used as context."""
    values, context = [], initial
    for _ in range(3):
        context = call(model, context)
        values.append(context)
    return torch.cat(values, dim=-1)


def test_native_checkpoint_matches_values_gradients_and_all_state(native_fno):
    """Replay must preserve both differentiable recurrence and BN evaluation buffers."""
    from training.w1_v2_checkpoint import checkpoint_native_model

    torch, prototype = native_fno
    eager, replay = copy.deepcopy(prototype), copy.deepcopy(prototype)
    initial = torch.randn(2, 32, 32, 1)
    first, second = initial.clone().requires_grad_(), initial.clone().requires_grad_()
    expected = _closed_loop(eager, first, lambda model, value: model(value), torch)
    actual = _closed_loop(replay, second, checkpoint_native_model, torch)
    assert torch.equal(actual, expected)
    target = torch.ones_like(expected)
    (expected - target).square().mean().backward()
    (actual - target).square().mean().backward()
    torch.testing.assert_close(second.grad, first.grad, rtol=1e-6, atol=1e-7)
    for name, parameter in eager.named_parameters():
        other = dict(replay.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            assert torch.isfinite(parameter.grad).all() and torch.isfinite(other.grad).all()
            torch.testing.assert_close(other.grad, parameter.grad, rtol=1e-6, atol=1e-7)
    for name, value in eager.state_dict().items():
        assert torch.equal(value, replay.state_dict()[name]), name
    assert int(replay.bn0.num_batches_tracked) == 3
    eager.eval()
    replay.eval()
    with torch.no_grad():
        assert torch.equal(eager(initial), replay(initial))


def test_unprotected_checkpoint_replays_running_updates(native_fno):
    """Show why matching forward/grad alone cannot establish native checkpoint equivalence."""
    from torch.utils.checkpoint import checkpoint

    torch, prototype = native_fno
    eager, replay = copy.deepcopy(prototype), copy.deepcopy(prototype)
    initial = torch.randn(2, 32, 32, 1)
    expected = _closed_loop(eager, initial.clone().requires_grad_(), lambda model, value: model(value), torch)
    actual = _closed_loop(replay, initial.clone().requires_grad_(),
        lambda model, value: checkpoint(model, value, use_reentrant=False), torch)
    assert torch.equal(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert int(eager.bn0.num_batches_tracked) == 3
    assert int(replay.bn0.num_batches_tracked) == 6
    assert not torch.equal(eager.bn0.running_mean, replay.bn0.running_mean)
