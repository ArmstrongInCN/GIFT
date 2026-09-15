"""Segment lifetimes and native-autograd invariants of training-only layouts."""
import copy
import os
from types import SimpleNamespace

import pytest
import torch

from training.gift_branch_kernels import BranchTrainingView
from experiments.formal._shared.high_frequency import SpectralResidualBlock, HighFrequencyBranch, HighFrequencyConfig


def test_segment_lifetime_parameter_storage_and_native_gradients():
    torch.manual_seed(12)
    old = SpectralResidualBlock(2, 2, activate=True)
    new = copy.deepcopy(old)
    view = BranchTrainingView(SimpleNamespace(blocks=[new]))
    field = torch.randn(2, 8, 8).unsqueeze(0)
    pointers = [p.data_ptr() for p in new.parameters()]
    with pytest.raises(RuntimeError, match="begin a training segment"):
        view.block(new, field)
    for _ in range(2):
        view.begin_segment()
        with pytest.raises(RuntimeError, match="previous training segment"):
            view.begin_segment()
        old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
        expected, actual = old(field), view.block(new, field)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward(); actual.square().mean().backward()
        for p, q in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        view.end_segment()
        assert view.packed is None and pointers == [p.data_ptr() for p in new.parameters()]
        with torch.no_grad():
            for model in (old, new): model.spectral.weight_positive.add_(.03)


@pytest.mark.skipif(os.environ.get("GIFT_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
                    reason="explicit CUDA test opt-in required")
def test_training_view_keeps_outputs_gradients_updates_and_prediction_exact():
    from training.gift_acceleration import TrainingEngine
    from experiments.formal._shared.gift_generator_training import configure_determinism
    configure_determinism(20260821, strict=False)
    old = HighFrequencyBranch(HighFrequencyConfig(), state_scale=1., high_scale=1.,
        low_rhs_scale=1., high_rhs_scale=1., frozen_generator=torch.zeros_like).cuda()
    new = copy.deepcopy(old)
    opts = [torch.optim.AdamW(m.parameters(), lr=.0015, weight_decay=1e-6) for m in (old, new)]
    engine = TrainingEngine(new, kind="derivative", scale=1., frozen=torch.zeros_like)
    pointers = [p.data_ptr() for p in new.parameters()]
    rng_before = torch.cuda.get_rng_state().clone()
    for size in (16, 1, 16):
        state = torch.randn(size, 64, 64, device="cuda")*.1
        target = torch.randn_like(state)*.01
        low = torch.zeros_like(state)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        opts[0].zero_grad(set_to_none=True)
        loss = (old.forward_with_generator_output(state, low)-target).square().mean()
        loss.backward(); actual = engine.gradients(state, target, low)
        assert torch.equal(loss.detach().reshape(1), actual)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert all(torch.equal(p.grad,q.grad) for p,q in zip(old.parameters(),new.parameters()))
        for model,opt in zip((old,new),opts):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.equal(p,q) for p,q in zip(old.parameters(),new.parameters()))
        assert pointers == [p.data_ptr() for p in new.parameters()]
        assert engine.branch_view.packed is None
        with torch.no_grad(): assert torch.equal(old(state),new(state))
    assert not torch.equal(rng_before,torch.cuda.get_rng_state())  # Only explicit inputs drew randomness.


@pytest.mark.skipif(os.environ.get("GIFT_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
                    reason="explicit CUDA test opt-in required")
def test_bad_batch_releases_views_and_cannot_update_optimizer():
    from training.gift_acceleration import TrainingEngine
    model = HighFrequencyBranch(HighFrequencyConfig(),state_scale=1.,high_scale=1.,
        low_rhs_scale=1.,high_rhs_scale=1.,frozen_generator=torch.zeros_like).cuda()
    engine = TrainingEngine(model,kind="derivative",scale=1.,frozen=torch.zeros_like)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0015)
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    state = torch.full((16,64,64),float('nan'),device="cuda")
    with pytest.raises(FloatingPointError,match="no optimizer update"):
        engine.step(optimizer,state,torch.zeros_like(state),torch.zeros_like(state))
    assert engine.branch_view.packed is None and not optimizer.state
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
