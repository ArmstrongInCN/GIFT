"""Three bounded NumPy cases; optional read-only oracle, no TF/Torch or training."""
import ast
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from adapters import pinn_stridge as adapter


class STRidgeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.version_info[:2] != (3, 8) or np.__version__ != "1.21.6":
            raise unittest.SkipTest("Use the separate documented Python3.8/NumPy1.21.6 PINN environment")
        if not all(os.environ.get(name) for name in ("GIFT_EXTERNAL_ROOT", "GIFT_LEGACY_PROJECT_ROOT")):
            raise unittest.SkipTest("Pinned external clone and explicit read-only oracle not configured")
        path = Path(os.environ["GIFT_LEGACY_PROJECT_ROOT"]) / "benchmarks/pinn_sr/locked_runtime/pinnsr_singletraj.py"
        if adapter._sha(path) != "e0ccf4e680e35331531beef05e118be43a845fdd7cae1c2e7790dafca821374d":
            raise ValueError("Read-only oracle source differs")
        selected = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name in ("stridge", "train_stridge")]
        if len(selected) != 2:
            raise ValueError("Expected exactly two oracle functions")
        cls.oracle = dict(np=np, RIDGE=1e-5, STRIDGE_D_TOL=1., STRIDGE_TOL_STEPS=100, STRIDGE_INNER=10)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), cls.oracle)
        cls.reports = []

    def compare(self, matrix, target, inherited, l0=None, iteration=0):
        before = np.random.get_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            expected = self.oracle["train_stridge"](matrix, target, inherited, l0)
            observed = adapter.fit_stridge(matrix, target, inherited, l0, iteration=iteration)
        after = np.random.get_state()
        np.testing.assert_array_equal(expected[0], observed["coefficients"])
        self.assertEqual(expected[1], observed["l0_penalty"])
        self.assertEqual(expected[2]["best_tolerance"], observed["best_tolerance"])
        self.assertEqual(expected[2]["best_error"], observed["best_error"])
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(len(observed["accepted_history"]), 1 + sum(row["accepted"] for row in expected[2]["history"]))
        self.assertTrue(observed["audit"]["inputs_unchanged"])
        self.assertEqual(observed["input_binding"]["inherited"], adapter._array_record(inherited))
        self.assertFalse(any(name.startswith("tensorflow") or name == "torch" for name in sys.modules))
        report = dict(case=self.id().split(".")[-1], shape=list(matrix.shape), coefficient_max_error=0.,
                      l0_exact=True, best_tolerance_exact=True, best_score_exact=True, global_numpy_rng_unchanged=True,
                      accepted_plus_initial=len(observed["accepted_history"]), full_trial_count=100,
                      input_binding=observed["input_binding"], source_binding=observed["source_binding"])
        self.reports.append(report)
        print("STRIDGE_CASE=" + json.dumps(report, sort_keys=True), flush=True)
        return observed

    @staticmethod
    def four_case():
        rng = np.random.RandomState(442)
        matrix = rng.standard_normal((128, 4)).astype(np.float32)
        target = (matrix @ np.array([.2, -.4, 0., .8], np.float32)
                  + .03 * rng.standard_normal(128)).astype(np.float32).reshape(-1, 1)
        return matrix, target, np.array([.1, -.25, 0., .55], np.float32)

    def test_128_by_4(self):
        self.compare(*self.four_case())

    def test_192_by_90(self):
        rng = np.random.RandomState(443)
        matrix = rng.standard_normal((192, 90)).astype(np.float32)
        truth = np.zeros(90, np.float32)
        truth[[3, 8, 16]] = [-.5, .01, 1.]
        target = (matrix @ truth + .01 * rng.standard_normal(192)).astype(np.float32).reshape(-1, 1)
        self.compare(matrix, target, truth * np.float32(.7))

    def test_persisted_l0_and_fixed_round_inheritance(self):
        matrix, target, initial = self.four_case()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            l0 = self.oracle["train_stridge"](matrix, target, initial, None)[1]
        inherited = np.array([.17, -.3, .002, .7], np.float32)
        observed_reads = []
        original = adapter._native_methods

        def monitor(path, columns):
            native = original(path, columns)
            regression = native.STRidge

            def call(*args, **kwargs):
                observed_reads.append(native.sess.run(native.lambda_w))
                return regression(*args, **kwargs)

            native.STRidge = call
            return native

        with patch.object(adapter, "_native_methods", monitor):
            observed = self.compare(matrix, target, inherited, l0, iteration=1)
        self.assertEqual(observed["l0_penalty"], l0)
        self.assertEqual(len(observed_reads), 100)
        for value in observed_reads:
            np.testing.assert_array_equal(value, inherited.astype(np.float64).reshape(-1, 1))
        with self.assertRaisesRegex(ValueError, "persisted"):
            adapter.fit_stridge(matrix, target, inherited, iteration=1)

    @classmethod
    def tearDownClass(cls):
        configured = os.environ.get("GIFT_PINN_STRIDGE_TEST_OUTPUT")
        if configured:
            directory = Path(configured).resolve()
            if directory == adapter.ROOT or adapter.ROOT in directory.parents:
                raise ValueError("Test evidence must stay outside candidate")
            directory.mkdir(parents=True, exist_ok=False)
            with (directory / "REPORT.json").open("x", encoding="utf-8") as stream:
                json.dump(dict(scope="three_synthetic_numpy_cases_not_full_PINN", cases=cls.reports), stream, indent=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
