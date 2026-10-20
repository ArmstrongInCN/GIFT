"""Check PINN planning and device selection without constructing a TF graph.

The import sentinel stops at the actual TensorFlow boundary: tests verify that
literal environment values no longer gate entry, not that a GPU is available.
"""
import contextlib
import io
import json
import os
import unittest
from unittest import mock
from adapters import pinn_runtime as runtime
from training import train_pinn as training

class DeviceContractTests(unittest.TestCase):
    """Validate plans and budgets independently from native TensorFlow execution."""

    def test_plans_accept_both_libraries_and_devices(self):
        """Report requested library/device pairs with no training execution."""
        for mode in ("known", "open"):
            for device in ("cpu", "gpu"):
                output = io.StringIO()
                with mock.patch("sys.argv", ["train_pinn", "--output", "unused-plan-only",
                                             "--mode", mode, "--device", device]), contextlib.redirect_stdout(output):
                    training.main()
                plan = json.loads(output.getvalue())
                self.assertTrue(plan["plan_only"])
                self.assertFalse(plan["training_executed"])
                self.assertEqual((plan["mode"], plan["device"]), (mode, device))
                self.assertEqual(plan["initialization"], "native_Xavier_seed1234")

    def test_physical_device_selection_is_not_a_literal_environment_gate(self):
        """Reach TensorFlow import for valid devices regardless of visible-device text."""
        import builtins
        original_import = builtins.__import__
        def import_boundary(name, *args, **kwargs):
            """Stop before real TensorFlow import so device planning remains lightweight."""
            if name == "tensorflow":
                raise RuntimeError("tensorflow import reached")
            return original_import(name, *args, **kwargs)
        threads = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        for device, visible in (("cpu", "0"), ("gpu", "1")):
            with mock.patch.dict(os.environ, dict(threads, CUDA_VISIBLE_DEVICES=visible)):
                with mock.patch("builtins.__import__", side_effect=import_boundary), self.assertRaisesRegex(RuntimeError, "tensorflow import reached"):
                    runtime.PINNRuntime("known", None, None, device=device)
        with self.assertRaises(ValueError):
            runtime.PINNRuntime("known", None, None, device="invalid")

    def test_requested_budget_is_preserved(self):
        """Keep an explicit NAdam budget for both ordinary and diagnostic plans."""
        self.assertEqual(training.budget_for({"pre_nadam": 2})["pre_nadam"], 2)
        self.assertEqual(training.budget_for({"pre_nadam": 2}, diagnostic=True)["pre_nadam"], 2)

if __name__ == "__main__":
    unittest.main()
