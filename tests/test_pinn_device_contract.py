"""Plans and environment gates do not construct a TensorFlow model."""
import contextlib
import io
import json
import os
import unittest
from unittest import mock
from adapters import pinn_runtime as runtime
from training import train_pinn as training

class DeviceContractTests(unittest.TestCase):
    def test_plans_accept_both_libraries_and_devices(self):
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

    def test_wrong_device_environment_rejected_before_tf_import(self):
        threads = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        for device, visible in (("cpu", "0"), ("gpu", "-1"), ("invalid", "-1")):
            with mock.patch.dict(os.environ, dict(threads, CUDA_VISIBLE_DEVICES=visible)):
                with self.assertRaises(ValueError):
                    runtime.PINNRuntime("known", None, None, device=device)

    def test_nonformal_budget_requires_explicit_diagnostic_flag(self):
        with self.assertRaises(ValueError):
            training.budget_for({"pre_nadam": 2})
        self.assertEqual(training.budget_for({"pre_nadam": 2}, diagnostic=True)["pre_nadam"], 2)

if __name__ == "__main__":
    unittest.main()
