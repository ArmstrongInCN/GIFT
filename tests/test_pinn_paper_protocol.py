"""PINN orchestration boundaries; no substitute neural loss or PDE algorithm."""
from types import SimpleNamespace

import numpy as np
import pytest

from training.pinn_lbfgsb import run_lbfgs
from training.train_pinn import TensorFlowBackend
from test_train_pinn_fsm import coordinator, drive


@pytest.mark.parametrize("phase,expected", [("pre_nadam", "pre"), ("ado_nadam", "ado"),
                                           ("post_nadam", "post")])
def test_backend_routes_native_update_phase(phase, expected):
    """The fixed-coefficient ADO operation must be selected by the schedule phase."""
    observed = []
    runtime = SimpleNamespace(profile={"variable_semantics": "TF1_RefVariable"},
        apply_sparse_result=lambda value: None, finalize_sparse_support=lambda: None,
        lbfgs_result=lambda: {}, step=lambda feed, phase: observed.append((feed, phase)))
    backend = TensorFlowBackend.__new__(TensorFlowBackend)
    backend.runtime, backend.phase = runtime, phase
    backend.feed = lambda *args: {"shared_population": True}
    backend.nadam(.001, 1., 0.)
    assert observed == [({"shared_population": True}, expected)]


def test_old_runtime_rejected_before_an_update():
    """The previous ResourceVariable runtime cannot silently run the repaired profile."""
    observed = []
    backend = TensorFlowBackend.__new__(TensorFlowBackend)
    backend.runtime = SimpleNamespace(profile={}, step=lambda *args: observed.append(args))
    backend.phase = "pre_nadam"
    backend.feed = lambda *args: {}
    with pytest.raises(RuntimeError, match="repaired outside runtime"):
        backend.nadam(.001, 1., 0.)
    assert observed == []


@pytest.mark.parametrize("mode", ["known", "open"])
def test_final_mask_uses_recorded_regression_not_revived_raw_values(mode):
    """Raw coefficient regrowth must not replace the independently selected sparse basis."""
    control = coordinator(mode)
    drive(control, lambda c: c.state["phase"] == "final_mask")
    selected = np.asarray(control.state["stridge_records"][-1]["coefficients"], np.float32)
    control.backend.assign_coefficients(np.ones(control.columns, np.float32))
    control.advance()
    np.testing.assert_array_equal(control.backend.coefficients(), selected)
    np.testing.assert_array_equal(control.backend.mask(), (selected != 0).astype(np.float32))


def test_resume_rejects_mask_not_bound_to_regression():
    """A checkpoint cannot pair a posttraining cursor with an unrelated dense mask."""
    control = coordinator()
    drive(control, lambda c: c.state["phase"] == "post_nadam")
    arrays, metadata = control.snapshot()
    arrays["model_mask"] = np.ones_like(arrays["model_mask"])
    with pytest.raises(ValueError, match="coupled regression support"):
        coordinator().restore(arrays, metadata)


def test_native_scipy_limit_is_not_reported_as_convergence():
    """Save the actual iteration-limit status while preserving observed callback counts."""
    class Optimizer:
        optimizer_kwargs = {"options": {"maxcor": 50}}
        def minimize(self, session, feed_dict, fetches, step_callback, loss_callback):
            loss_callback(1.)
            loss_callback(.5)
            step_callback(None)
    native = {"success": False, "status": 1, "message": "iteration limit",
              "iterations": 1, "function_evaluations": 2, "gradient_evaluations": 2,
              "objective": .5, "gradient_inf_norm": .2}
    losses = iter((1., .5))
    runtime = SimpleNamespace(model=SimpleNamespace(optimizer_BFGS_Pre=Optimizer(), loss="loss"),
                              session=None, metrics=lambda feed: {"loss": next(losses)},
                              lbfgs_result=lambda: native)
    record = run_lbfgs(runtime, {}, maxiter=1, maxfun=2)
    assert record["initial_loss"] == 1. and record["final_loss"] == .5
    assert record["loss_change"] == -.5
    assert record["scipy_result"] == native
    assert record["callback_counts_match_scipy"]
    assert not record["convergence_reported_by_scipy"]
    assert record["termination"] == "iteration limit"
