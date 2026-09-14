"""Synthetic sparse-regression contracts using only the pinned external source.

Known synthetic coefficients check recovery; repeated calls check determinism.
Neither check certifies a full PINN experiment or imports another working copy.
"""
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
        if not os.environ.get("GIFT_EXTERNAL_ROOT"):
            raise unittest.SkipTest("Pinned external clone not configured")
        cls.reports = []

    def check(self, matrix, target, inherited, truth, l0=None, iteration=0, repeat=True, recovery_atol=None):
        before = np.random.get_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            observed = adapter.fit_stridge(matrix, target, inherited, l0, iteration=iteration)
            if repeat:
                repeated = adapter.fit_stridge(matrix, target, inherited, l0, iteration=iteration)
                np.testing.assert_array_equal(repeated["coefficients"], observed["coefficients"])
                for key in ("l0_penalty", "best_tolerance", "best_error", "accepted_history"):
                    self.assertEqual(repeated[key], observed[key])
        after = np.random.get_state()
        # Sparse penalized selection can correctly discard a nonzero true term.
        # Require coefficient recovery only in the separate unpenalized case.
        if recovery_atol is not None:
            np.testing.assert_allclose(observed["coefficients"], truth, rtol=0, atol=recovery_atol)
        train = np.sort(np.random.RandomState(0).choice(len(matrix), int(.8 * len(matrix)), replace=False))
        test = np.setdiff1d(np.arange(len(matrix)), train)
        residual = target.reshape(-1)[test] - matrix[test] @ observed["coefficients"]
        score = np.mean(residual ** 2) + observed["l0_penalty"] * np.count_nonzero(observed["coefficients"])
        np.testing.assert_allclose(observed["best_error"], score, rtol=1e-5, atol=1e-7)
        history = observed["accepted_history"]
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["kind"], "initial")
        self.assertEqual(history[-1]["score"], observed["best_error"])
        self.assertTrue(all(row["score"] <= previous["score"] for previous, row in zip(history, history[1:])))
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertTrue(observed["audit"]["inputs_unchanged"])
        self.assertEqual(observed["input_binding"]["inherited"], adapter._array_record(inherited))
        self.assertFalse(any(name.startswith("tensorflow") or name == "torch" for name in sys.modules))
        report = dict(case=self.id().split(".")[-1], shape=list(matrix.shape),
                      coefficient_max_error=float(np.max(np.abs(observed["coefficients"] - truth))),
                      recovery_absolute_tolerance=recovery_atol, repeated_call_exact=repeat,
                      heldout_penalized_objective_recomputed=True,
                      global_numpy_rng_unchanged=True, accepted_plus_initial=len(history),
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
        self.check(*self.four_case(), truth=np.array([.2, -.4, 0., .8]))

    def test_noiseless_unpenalized_coefficient_recovery(self):
        matrix, _, inherited = self.four_case()
        truth = np.array([.2, -.4, 0., .8], np.float32)
        target = (matrix @ truth).reshape(-1, 1)
        self.check(matrix, target, inherited, truth, l0=0., recovery_atol=1e-5)

    def test_192_by_90(self):
        rng = np.random.RandomState(443)
        matrix = rng.standard_normal((192, 90)).astype(np.float32)
        truth = np.zeros(90, np.float32)
        truth[[3, 8, 16]] = [-.5, .01, 1.]
        target = (matrix @ truth + .01 * rng.standard_normal(192)).astype(np.float32).reshape(-1, 1)
        self.check(matrix, target, truth * np.float32(.7), truth)

    def test_persisted_l0_and_fixed_round_inheritance(self):
        matrix, target, initial = self.four_case()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            l0 = adapter.fit_stridge(matrix, target, initial)["l0_penalty"]
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
            observed = self.check(matrix, target, inherited, np.array([.2, -.4, 0., .8]),
                                  l0, iteration=1, repeat=False)
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
                raise ValueError("Test evidence must stay outside the project")
            directory.mkdir(parents=True, exist_ok=False)
            with (directory / "REPORT.json").open("x", encoding="utf-8") as stream:
                json.dump(dict(scope="synthetic_numpy_contracts_not_full_PINN", cases=cls.reports), stream, indent=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
