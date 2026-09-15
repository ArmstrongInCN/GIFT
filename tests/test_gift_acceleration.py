"""Execution invariants; small fixtures are not published accuracy evidence."""
import os

import pytest
import torch

from gift import GIFTGenerator
from training.gift_acceleration import TrainingEngine
from experiments.formal._shared.gift_runtime import (
    apply_local_correction, correction_tensors, square_selected_mask,
    DEFAULT_P21_CORRECTION_POLICY,
)


def test_static_grid_cache_keeps_state_dict_and_gradients():
    torch.manual_seed(44)
    reference = GIFTGenerator(2, rank=2)
    candidate = GIFTGenerator(2, rank=2)
    candidate.load_state_dict(reference.state_dict())
    candidate.enable_static_cache()
    state = torch.randn(3, 8, 8)
    with torch.inference_mode():
        candidate(state)  # Cache created during validation must remain trainable.
    for model in (reference, candidate):
        model(state).square().mean().backward()
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for left, right in zip(reference.parameters(), candidate.parameters()):
        assert torch.equal(left.grad, right.grad)
    # Trainable table values must not be cached.
    with torch.no_grad():
        for model in (reference, candidate): model.left_tables[0].values.add_(0.1)
    assert torch.equal(reference(state), candidate(state))


def test_tensor_correction_preserves_state_counts_gates():
    state = torch.zeros(2, 64, 64)
    state[0, 13, 17] = 1000
    state[1] = torch.arange(64 * 64).reshape(64, 64).remainder(3) * 1000
    mask = square_selected_mask(64, cutoff=21, device=state.device)
    expected = apply_local_correction(state)
    actual = correction_tensors(state, DEFAULT_P21_CORRECTION_POLICY, mask)
    assert torch.equal(expected.state, actual[0])
    assert torch.equal(expected.trigger_count, actual[1])
    assert torch.equal(expected.gate_failed, actual[2])


def test_cpu_fallback_and_rejected_update():
    model = GIFTGenerator(2, rank=1)
    engine = TrainingEngine(model, kind="generator", scale=1., backend="cuda-graph")
    assert engine.backend == "eager"
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(FloatingPointError, match="no optimizer update"):
        engine.step(opt, torch.full((2, 8, 8), float('nan')), torch.zeros(2, 8, 8))
    assert not opt.state
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())


def test_rollout_gate_rejects_before_optimizer_update():
    class SafetyBranch(torch.nn.Module):
        """Small control-flow fixture; correction uses the real tensor core."""
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.high_scale = 1.

        def enable_static_cache(self): pass
        def split_state(self, value): return value, torch.zeros_like(value)
        def project_q21(self, value): return value
        def forward_with_generator_output(self, state, low): return state * self.weight

    torch.manual_seed(100)
    sequence = torch.zeros(1, 8, 64, 64)
    sequence[:, 0] = torch.randn(1, 64, 64) * 1000
    model = SafetyBranch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    engine = TrainingEngine(model, kind="short", frozen=torch.zeros_like, backend="eager")
    with pytest.raises(FloatingPointError, match="no optimizer update"):
        engine.step(optimizer, sequence)
    assert not optimizer.state and model.weight.item() == 0.


def test_frozen_generator_mutation_is_rejected():
    from types import SimpleNamespace
    model = GIFTGenerator(2, rank=1)
    frozen = SimpleNamespace(model=GIFTGenerator(2, rank=1))
    engine = TrainingEngine(model, kind="generator", scale=1., frozen=frozen, backend="eager")
    with torch.no_grad(): next(frozen.model.parameters()).add_(1.)
    with pytest.raises(RuntimeError, match="frozen generator changed"):
        engine.gradients(torch.zeros(1, 8, 8), torch.zeros(1, 8, 8))


@pytest.mark.parametrize("module_name", [
    "m2_recursive_prediction", "m3_cross_resolution", "s1_high_frequency_branch",
    "s2_recursive_local_correction", "s3_seed_stability",
])
def test_prediction_cli_exposes_opt_in_backend(monkeypatch, module_name):
    import importlib
    import sys
    module = importlib.import_module(f"experiments.formal.{module_name}.run")
    monkeypatch.setattr(sys, "argv", ["run.py", "--gift-execution", "cuda-graph"])
    assert module.parse_args().gift_execution == "cuda-graph"


@pytest.mark.skipif(os.environ.get("GIFT_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
                    reason="explicit CUDA test opt-in required")
def test_graph_shape_switch_rng_and_parameter_storage_guard():
    torch.manual_seed(88)
    model = GIFTGenerator(2, rank=1).cuda()
    model.constant_table.requires_grad_(False)
    model.linear_table.requires_grad_(False)
    engine = TrainingEngine(model, kind="generator", scale=1.)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    batches = [(torch.randn(n, 8, 8, device="cuda"), torch.randn(n, 8, 8, device="cuda")) for n in (3, 1)]
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    for batch in batches:
        engine.gradients(*batch)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    for batch in (batches[0], batches[1], batches[0]):
        engine.gradients(*batch)
        actual = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        ((model(batch[0]) - batch[1]).square().mean()).backward()
        assert all((a is None and p.grad is None) or
                   (a is not None and p.grad is not None and torch.equal(a, p.grad))
                   for a, p in zip(actual, model.parameters()))
    graph = next(iter(engine.graphs.values()))
    with torch.no_grad(): model.constant_table.values.set_(model.constant_table.values.clone())
    with pytest.raises(RuntimeError, match="storage changed"):
        graph(*batches[0])
