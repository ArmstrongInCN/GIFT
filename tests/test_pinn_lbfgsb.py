"""Check public-interface invocation, not a reimplemented L-BFGS solver."""
from types import SimpleNamespace
import unittest
import numpy as np
from training.pinn_lbfgsb import run_lbfgs

class InterfaceTests(unittest.TestCase):
    def test_callbacks_and_options_pass_through(self):
        class Optimizer:
            optimizer_kwargs = {"options": {"maxiter": 10000, "maxfun": 10000, "maxcor": 50, "maxls": 50}}
            def minimize(self, session, feed_dict, fetches, step_callback, loss_callback):
                self.received = (session, feed_dict, fetches)
                loss_callback(1.)
                loss_callback(.5)
                step_callback(None)
        optimizer = Optimizer()
        model = SimpleNamespace(optimizer_BFGS_Pre=optimizer, loss="native_loss")
        runtime = SimpleNamespace(model=model, session="native_session", metrics=lambda feed: {"loss": .5})
        result = run_lbfgs(runtime, {"fixture": True}, maxiter=2, maxfun=5)
        self.assertEqual(optimizer.received, ("native_session", {"fixture": True}, ["native_loss"]))
        self.assertEqual(optimizer.optimizer_kwargs["options"], {"maxiter": 2, "maxfun": 5, "maxcor": 50, "maxls": 50})
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["function_evaluations"], 2)
        self.assertIn("convergence_not_exposed", result["termination"])
        self.assertEqual(result["resume_boundary"], "phase_start_if_interrupted")

    def test_native_exception_is_not_hidden_or_marked_complete(self):
        class Optimizer:
            optimizer_kwargs = {"options": {}}
            def minimize(self, *args, **kwargs):
                raise RuntimeError("external solver interrupted")
        runtime = SimpleNamespace(model=SimpleNamespace(optimizer_BFGS_Pre=Optimizer(), loss=None), session=None)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            run_lbfgs(runtime, {})

if __name__ == "__main__":
    unittest.main()
