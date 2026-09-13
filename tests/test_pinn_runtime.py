"""Small CPU equivalence tests; stdlib unittest works in the separate Py3.8 env.

Optional files are user supplied via GIFT_DATA_ROOT/GIFT_EXTERNAL_ROOT and
GIFT_LEGACY_PROJECT_ROOT. No legacy implementation or absolute path is shipped.
Temporary test artifacts are retained, never recursively deleted.
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from adapters import pinn_runtime as pinn


class PINNBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(os.environ.get(k) for k in ("GIFT_DATA_ROOT", "GIFT_EXTERNAL_ROOT", "GIFT_LEGACY_PROJECT_ROOT")):
            raise unittest.SkipTest("External data/source/read-only legacy oracle not configured")
        try:
            import tensorflow as tf
        except ImportError:
            raise unittest.SkipTest("Run in the separate documented PINN environment")
        cls.tf = tf
        data = Path(os.environ["GIFT_DATA_ROOT"])
        cls.values = pinn.read_inputs(data / "auxiliary/pinn_initialization/network_initialization_tf115_seed1234.npz",
                                      data / "auxiliary/pinn_sampling/noise_001_seed1234.npz",
                                      data / "m1_parameter_identification/noise_001.h5", 8, 12)
        configured = os.environ.get("GIFT_PINN_TEST_OUTPUT")
        cls.output = Path(configured) if configured else Path(tempfile.mkdtemp(prefix="gift_pinn_test_"))
        if configured:
            cls.output.mkdir(parents=True, exist_ok=False)
        path = Path(os.environ["GIFT_LEGACY_PROJECT_ROOT"]) / "benchmarks/pinn_sr/locked_runtime/pinnsr_singletraj.py"
        if pinn.sha256(path) != "E0CCF4E680E35331531BEEF05E118BE43A845FDD7CAE1C2E7790DAFCA821374D":
            raise ValueError("Unexpected read-only legacy oracle")
        cls.oracle = pinn._module(path, "readonly_pinn_test_oracle")

    def runtime(self, mode):
        initial, _, _, _, lower, upper, _ = self.values
        return pinn.PINNRuntime(mode, lower, upper, initial)

    def assert_state(self, left, right):
        self.assertEqual(left["iteration"], right["iteration"])
        np.testing.assert_array_equal(left["mask"], right["mask"])
        self.assertEqual(len(left["variables"]), len(right["variables"]))
        for a, b in zip(left["variables"], right["variables"]):
            np.testing.assert_array_equal(a, b)

    def check_mode(self, mode):
        initial, coordinates, targets, physics, lower, upper, bound = self.values
        runtime = self.runtime(mode)
        try:
            start = runtime.state()
            runtime.step(coordinates, targets, physics)
            runtime.step(coordinates, targets, physics)
            continuous = runtime.state()
            runtime.restore(start)
            runtime.step(coordinates, targets, physics)
            directory = self.output / (mode + "_same_run")
            identity = {"inputs": bound, "config": {"chunk": 32768, "lr": .001, "physics": 1.0, "l1": 1e-7, "seed": 1234}}
            store = pinn.NAdamCheckpoints(directory, runtime, identity)
            store.save()
            expected_python, expected_numpy = random.random(), np.random.random()
            # A duplicate commit cannot replace the first immutable state.
            with self.assertRaises(FileExistsError):
                store.save()
        finally:
            runtime.close()
        resumed = self.runtime(mode)
        try:
            with self.assertRaises(ValueError):
                pinn.NAdamCheckpoints(directory, resumed, {**identity, "wrong_input": True}, resume=True)
            resumed_store = pinn.NAdamCheckpoints(directory, resumed, identity, resume=True)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            resumed.step(coordinates, targets, physics)
            self.assert_state(continuous, resumed.state())
            resumed_store.save()
            # The old handle must fail before writing another checkpoint.
            with self.assertRaises(ValueError):
                store.save()
            self.assertTrue((directory / "step_00000001/state.npz").exists())
            self.assertTrue((directory / "step_00000002/state.npz").exists())
            # Nonzero coefficients, sparse mask, and exactly two chunks per
            # data/physics part exercise the previously untested gradient path.
            resumed.restore(start)
            count = 90 if mode == "open" else 4
            coefficients = np.linspace(-.02, .03, count, dtype=np.float32).reshape(-1, 1)
            mask = np.ones_like(coefficients)
            mask[::3] = 0
            resumed.set_coefficients(coefficients, mask)
            legacy = self.oracle.PINNSRGraph(mode, lower, upper)
            config = self.tf.compat.v1.ConfigProto(allow_soft_placement=True, device_count={"GPU": 0},
                                                  intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            session = self.tf.compat.v1.Session(graph=legacy.graph, config=config)
            try:
                session.run(legacy.initializer)
                with legacy.graph.as_default():
                    session.run([v.assign(a) for v, a in zip(legacy.network_variables, initial)])
                session.run(legacy.assign_coefficients, {legacy.new_coefficients: coefficients})
                session.run(legacy.assign_mask, {legacy.new_mask: mask})
                experiment = self.oracle.Experiment.__new__(self.oracle.Experiment)
                experiment.model, experiment.session, experiment.chunk = legacy, session, 4
                experiment.data = SimpleNamespace(train_coordinates=coordinates, train_targets=targets, physics_coordinates=physics[:8])
                expected_loss, expected_gradient, _, expected_parts = experiment.objective_gradient(1.0, 1e-7, True)
                actual_parts, actual_gradients = resumed.objective(coordinates, targets, physics[:8], chunk=4)
                self.assertEqual(actual_parts["loss"], expected_loss)
                self.assertEqual(actual_parts["total"], expected_parts["unlogged_total"])
                np.testing.assert_array_equal(np.concatenate([v.reshape(-1) for v in actual_gradients]), expected_gradient)
                np.testing.assert_array_equal(resumed.predict(coordinates), session.run(legacy.prediction, {legacy.coordinates: coordinates}))
                for actual, expected in zip(resumed.library(physics), session.run([legacy.library, legacy.omega_t], {legacy.coordinates: physics})):
                    np.testing.assert_array_equal(actual, expected)
                session.run(legacy.apply_all, {legacy.learning_rate: .001})
                resumed.session.run(resumed.update, {resumed.model.lr_tf: .001})
                for actual, expected in zip(resumed.session.run(resumed.parameters), session.run(legacy.train_variables)):
                    np.testing.assert_array_equal(actual, expected)
            finally:
                session.close()
            report = {"mode": mode, "continuous_2_vs_1_resume_1": "all_81_graph_variables_plus_mask_exact",
                      "python_numpy_rng_restored": True, "nonzero_sparse_mask_two_chunks": "loss_full_gradient_prediction_library_one_update_exact",
                      "full_pinn_acceptance": False, "source_sha256": pinn.sha256(Path(pinn.__file__)), "inputs": bound}
            with (self.output / (mode + "_RESULT.json")).open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
        finally:
            resumed.close()

    def test_open_resume_and_nonzero_two_chunk_oracle(self):
        self.check_mode("open")

    def test_known_resume_and_nonzero_two_chunk_oracle(self):
        self.check_mode("known")


if __name__ == "__main__":
    unittest.main(verbosity=2)
