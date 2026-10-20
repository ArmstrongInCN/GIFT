"""CPU contracts for fixed ANIE conditioning and genuinely single-field FNO models.

These tests use fresh native models and artificial fields, never experiment
weights or observations. Optional source checkouts must be explicitly configured.
"""
import os
from pathlib import Path

import pytest

from adapters.anie_config import ANIEConfig


@pytest.fixture(scope="module")
def torch_cpu():
    """Bound these native forward/gradient checks to four CPU worker threads."""
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield torch
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def anie_root():
    """Require an explicitly selected read-only native ANIE source checkout."""
    configured = os.environ.get("GIFT_ANIE_SOURCE_ROOT")
    if not configured:
        pytest.skip("set GIFT_ANIE_SOURCE_ROOT for native ANIE model tests")
    return Path(configured)


@pytest.fixture(scope="module")
def external_root():
    """Require an explicitly selected upstream root containing the native FNO source."""
    configured = os.environ.get("GIFT_EXTERNAL_ROOT")
    if not configured:
        pytest.skip("set GIFT_EXTERNAL_ROOT for native FNO model tests")
    return Path(configured)


def _historical_native_forward(model, initial, times, torch):
    """Execute the pre-v2 c0/y_init computation as a backward-compatibility reference."""
    model._validate_initial(initial)
    normalized = model._query_times(times, initial)
    count = normalized.numel()
    model._solver_module.device = initial.device
    y0 = model.encoder(initial.unsqueeze(1)).permute(0, 2, 3, 1).unsqueeze(-2).contiguous()
    yi = y0.repeat(1, 1, 1, count, 1)
    solver = model._solver_module.Integral_spatial_attention_solver_multbatch(
        normalized, y0, y_init=yi, c=lambda _: torch.zeros_like(yi),
        Encoder=model.integral_operator, sampling_points=count, mask=None,
        max_iterations=model.config.max_iterations, spatial_integration=True,
        spatial_domain=model.spatial_domain, spatial_domain_dim=2,
        smoothing_factor=model.config.smoothing_factor,
        global_error_tolerance=model.config.global_error_tolerance,
        use_support=False, accumulate_grads=True, initialization=True)
    return model.decoder(solver.solve()).permute(0, 3, 1, 2).contiguous()


def test_condition_mode_keeps_architecture_fixed():
    """Allow only the two declared condition equations, with native as the old default."""
    assert ANIEConfig().condition_mode == "native"
    assert ANIEConfig(condition_mode="fixed_f").frames == 151
    with pytest.raises(ValueError, match="condition_mode"):
        ANIEConfig(condition_mode="test_only_f")
    with pytest.raises(ValueError, match="fixed"):
        ANIEConfig(condition_mode="fixed_f", latent_dim=16)


@pytest.mark.parametrize("training", [False, True])
def test_default_anie_matches_historical_forward_exactly(torch_cpu, anie_root, training):
    """Keep c0 predictions bit-identical in eval and with seeded native training dropout."""
    from adapters.anie import ANIEPredictor

    torch = torch_cpu
    torch.manual_seed(41)
    model = ANIEPredictor(source_root=anie_root).cpu().train(training)
    initial = torch.randn(1, 64, 64) * .1
    times = torch.tensor([0., .6, 1.2, 1.8, 2.4, 3.])
    with torch.no_grad():
        torch.manual_seed(42)
        actual = model(initial, times=times)
        torch.manual_seed(42)
        expected = _historical_native_forward(model, initial, times, torch)
    assert torch.equal(actual, expected)


def test_fixed_free_term_reused_and_gradient_reaches_encoder(torch_cpu, anie_root, monkeypatch):
    """Isolate the added RHS gradient from the initial-iterate gradient in the native solver.

    Only this diagnostic detaches the solver's initial iterate. The free
    term stays attached, so a nonzero encoder gradient must pass through c.
    One fresh-model CPU optimizer step verifies that gradient is usable.
    """
    from adapters.anie import ANIEPredictor

    torch = torch_cpu
    torch.manual_seed(43)
    model = ANIEPredictor(ANIEConfig(condition_mode="fixed_f"), source_root=anie_root).cpu().eval()
    native = ANIEPredictor(source_root=anie_root).cpu()
    native.load_state_dict(model.state_dict(), strict=True)
    assert model.state_dict().keys() == native.state_dict().keys()
    original_solver = model._solver_module.Integral_spatial_attention_solver_multbatch
    calls, encoded_calls = [], []
    handle = model.encoder.register_forward_hook(lambda *args: encoded_calls.append(1))

    def isolated_solver(times, y0, **kwargs):
        """Keep the actual native solver while separating its two conditioning paths."""
        free_function = kwargs["c"]
        free = free_function(times)
        assert free is kwargs["y_init"] and free.requires_grad
        free.retain_grad()

        def recorded_free(query):
            """Require every RHS to use the same encoder tensor, regardless of query argument."""
            value = free_function(query)
            assert value is free
            calls.append(value)
            return value

        kwargs["c"] = recorded_free
        kwargs["y_init"] = kwargs["y_init"].detach()
        return original_solver(times, y0.detach(), **kwargs)

    monkeypatch.setattr(model._solver_module, "Integral_spatial_attention_solver_multbatch", isolated_solver)
    initial = torch.randn(1, 64, 64) * .1
    times = torch.tensor([0., .6, 1.2, 1.8, 2.4, 3.])
    prediction = model(initial, times=times)
    assert prediction.shape == (1, 6, 64, 64) and torch.isfinite(prediction).all()
    prediction[:, 1:].square().mean().backward()
    handle.remove()
    assert len(encoded_calls) == 1 and len(calls) >= 2
    assert calls[0].grad is not None and bool((calls[0].grad != 0).any())
    gradients = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(bool((g != 0).any()) for g in gradients)
    before = model.encoder.conv_layer1.weight.detach().clone()
    torch.optim.Adam(model.encoder.parameters(), lr=1e-4).step()
    assert not torch.equal(before, model.encoder.conv_layer1.weight.detach())


@pytest.mark.parametrize("method", ["fno2d", "fno3d"])
def test_fno_single_field_projection_and_native_gradient(torch_cpu, external_root, method):
    """Use one state channel, preserve every other layer shape, and execute native CPU autograd."""
    from adapters.models import build_model

    torch = torch_cpu
    torch.manual_seed(44)
    legacy = build_model(method, external_root=external_root)
    torch.manual_seed(44)
    explicit_legacy = build_model(method, external_root=external_root, history=46)
    assert all(torch.equal(value, explicit_legacy.state_dict()[key])
               for key, value in legacy.state_dict().items())
    single = build_model(method, external_root=external_root, history=1)
    assert single.fc0.in_features == (3 if method == "fno2d" else 4)
    assert legacy.fc0.in_features == (48 if method == "fno2d" else 49)
    assert single.state_dict().keys() == legacy.state_dict().keys()
    for key, value in single.state_dict().items():
        if key != "fc0.weight":
            assert value.shape == legacy.state_dict()[key].shape
    assert sum(p.numel() for p in legacy.parameters()) - sum(p.numel() for p in single.parameters()) == 900
    if method == "fno2d":
        inputs = torch.randn(1, 32, 32, 1, requires_grad=True)
    else:
        # Native FNO3D expects caller-supplied xy-time coordinates. The only
        # observed state channel is repeated along its query-time axis.
        state = torch.randn(1, 16, 16, 1, 1).repeat(1, 1, 1, 16, 1)
        row, column, time = torch.meshgrid(torch.linspace(0., 1., 16),
            torch.linspace(0., 1., 16), torch.linspace(0., 1., 16), indexing="ij")
        coordinates = torch.stack((row, column, time), dim=-1).unsqueeze(0)
        inputs = torch.cat((state, coordinates), dim=-1).requires_grad_()
    prediction = single(inputs)
    assert prediction.shape == (*inputs.shape[:-1], 1) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert bool((inputs.grad[..., :1] != 0).any())
    assert single.fc0.weight.grad is not None and torch.isfinite(single.fc0.weight.grad).all()
    assert bool((single.fc0.weight.grad != 0).any())


def test_fno_history_is_explicit_and_supported():
    """Reject implicit history reuse and unsupported single-field changes to other families."""
    from adapters.models import build_model

    for history in (0, 2, True):
        with pytest.raises(ValueError, match="history must"):
            build_model("fno2d", history=history)
    with pytest.raises(ValueError, match="only for FNO"):
        build_model("uno", history=1)


@pytest.mark.parametrize("method", ["anie_ns", "anie_f"])
@pytest.mark.parametrize("training_count,time_stride", [(200, 5), (300, 2)])
def test_native_anie_sparse_is_explicit_and_differentiable(torch_cpu, anie_root, method, training_count, time_stride):
    """Use the same official modules with declared physical queries and encoder gradients."""
    from adapters.anie import ANIEPredictor
    from training.w1_v2_data import window_manifest
    from training.w1_v2_train import forward_model

    torch = torch_cpu
    torch.manual_seed(45)
    condition = "native" if method == "anie_ns" else "fixed_f"
    model = ANIEPredictor(ANIEConfig(condition_mode=condition), source_root=anie_root).cpu().eval()
    manifest = window_manifest(training_count, time_stride)
    points = manifest["points"]
    spec = dict(method=method, query_times=manifest["query_times"], future_steps=points - 1)
    initial = torch.randn(1, 64, 64) * .1
    prediction = forward_model(model, initial, spec, {}, training=True)
    assert prediction.shape == (1, points, 64, 64) and torch.isfinite(prediction).all()
    prediction[:, 1:].square().mean().backward()
    gradients = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(bool((g != 0).any()) for g in gradients)
    # Stored native defaults remain151; explicit queries alone define this run.
    assert model.physical_times.numel() == 151


@pytest.mark.parametrize("method", ["fno2d", "fno3d"])
@pytest.mark.parametrize("training_count,time_stride", [(200, 5), (300, 2)])
def test_native_fno_sparse_controller_shapes_gradients_and_temporal_modes(
        torch_cpu, external_root, method, training_count, time_stride):
    """The declared future axis preserves native layers, single-field inputs and full autograd."""
    from adapters.models import build_model
    from training.w1_v2_data import window_manifest
    from training.w1_v2_train import forward_model

    torch = torch_cpu
    torch.manual_seed(46)
    model = build_model(method, external_root=external_root, history=1).cpu().eval()
    manifest = window_manifest(training_count, time_stride)
    points = manifest["points"]
    steps = points - 1
    spec = dict(method=method, source_root=str(external_root),
                query_times=manifest["query_times"], future_steps=steps)
    # FNO2D retains12 spatial modes, requiring at least22 real-axis samples;
    # FNO3D retains8 and therefore permits the smaller16-square CPU fixture.
    size = 32 if method == "fno2d" else 16
    initial = torch.randn(1, size, size, requires_grad=True)
    normalization = dict(input_mean=torch.zeros(size, size, 1), input_std=torch.ones(size, size, 1),
                         output_mean=torch.zeros(size, size, steps), output_std=torch.ones(size, size, steps), eps=1e-5)
    prediction = forward_model(model, initial, spec, normalization, training=True)
    assert prediction.shape == (1, points, size, size) and torch.isfinite(prediction).all()
    prediction[:, 1:].square().mean().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all() and bool((initial.grad != 0).any())
    assert model.fc0.weight.grad is not None and torch.isfinite(model.fc0.weight.grad).all()
    if method == "fno3d":
        assert model.modes3 == 8 and steps // 2 + 1 >= model.modes3
        too_short = dict(spec, query_times=(torch.arange(11).double() * .3).tolist(), future_steps=10)
        with pytest.raises(ValueError, match="temporal mode count"):
            forward_model(model, initial, too_short, normalization)
