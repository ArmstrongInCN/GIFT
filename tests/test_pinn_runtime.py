"""Opt-in native PINN tests in the separate Python3.8/TensorFlow environment.

Use GIFT_PINN_NATIVE_TEST=1 and configure GIFT_DATA_ROOT/GIFT_EXTERNAL_ROOT.
No other author's working directory, checkpoint or initialization file is read.
"""
import os
from pathlib import Path
import unittest
import numpy as np
from adapters import pinn_runtime as pinn

@unittest.skipUnless(os.environ.get("GIFT_PINN_NATIVE_TEST") == "1", "native TensorFlow test is opt-in")
class PINNBoundaryTests(unittest.TestCase):
    def check_mode(self, mode):
        root = Path(os.environ["GIFT_DATA_ROOT"])
        coordinates, targets, physics, lower, upper, _ = pinn.read_inputs(
            root / "auxiliary/pinn_sampling/noise_000_seed1234.npz",
            root / "standard_ns_n64_full_spectrum.h5", 8, 12)
        runtime = pinn.PINNRuntime(mode, lower, upper, device="cpu")
        try:
            feed = runtime.feed(coordinates, targets, physics)
            before = runtime.state()
            runtime.step(feed)
            trained = runtime.state()
            runtime.restore(before)
            feed[runtime.model.NonZeroMask_w_tf] = runtime.mask
            runtime.step(feed)
            replay = runtime.state()
            self.assertEqual(trained["iteration"], replay["iteration"])
            for left, right in zip(trained["variables"], replay["variables"]):
                np.testing.assert_array_equal(left, right)
            matrix, target = runtime.library(physics)
            self.assertEqual(matrix.shape, (12, 4 if mode == "known" else 90))
            self.assertEqual(target.shape, (12, 1))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertEqual(len(runtime.variables), 59)
            compatibility = runtime.model.net_U.__globals__["tf"]
            self.assertIs(compatibility.gradients, runtime.tf.compat.v1.gradients)
            self.assertNotIn("gradients", vars(type(compatibility)))
            self.assertFalse(any("gift_gradient" in variable.name for variable in runtime.variables))
        finally:
            runtime.close()

    def test_known_native_operation_and_restore(self):
        self.check_mode("known")

    def test_open_native_operation_and_restore(self):
        self.check_mode("open")

if __name__ == "__main__":
    unittest.main()
