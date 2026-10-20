"""Synthetic sparse-regression contracts using the configured external source.

Known synthetic coefficients check recovery; repeated calls check determinism.
Neither check certifies a full PINN experiment or imports another working copy.
"""
import json
import os
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from adapters import pinn_stridge as adapter


class STRidgeBoundaryTests(unittest.TestCase):
    """Check native sparse search on synthetic four/90-column systems and saved l0 values."""

    @classmethod
    def setUpClass(cls):
        """Require the configured external source and initialise optional case reports."""
        if not os.environ.get("GIFT_EXTERNAL_ROOT"):
            raise unittest.SkipTest("External PINN clone not configured")
        cls.reports = []

    def check(self, matrix, target, inherited, truth, l0=None, iteration=0, repeat=True, recovery_atol=None):
        """Check deterministic coefficients, held-out score and unchanged caller RNG.

        ``truth`` is used for reporting, with equality required only when an
        explicit recovery tolerance is supplied. Penalised noisy cases may
        legitimately remove a nonzero synthetic term. No TF graph is built.
        """
        before = np.random.get_state()
        imported_before = set(sys.modules)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
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
        self.assertFalse(any(name.startswith("tensorflow") or name == "torch"
                             for name in set(sys.modules) - imported_before))
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
        """Create a seeded noisy 128-by-four regression and inherited coefficients."""
        rng = np.random.RandomState(442)
        matrix = rng.standard_normal((128, 4)).astype(np.float32)
        target = (matrix @ np.array([.2, -.4, 0., .8], np.float32)
                  + .03 * rng.standard_normal(128)).astype(np.float32).reshape(-1, 1)
        return matrix, target, np.array([.1, -.25, 0., .55], np.float32)

    def test_128_by_4(self):
        """Check the native penalised search on a noisy four-column synthetic system."""
        self.check(*self.four_case(), truth=np.array([.2, -.4, 0., .8]))

    def test_sparse_adapter_is_numpy_only_in_a_clean_process(self):
        """Other pytest modules may use torch; this adapter itself must not import it."""
        program = """import sys
import numpy as np
from adapters.pinn_stridge import fit_stridge
matrix = np.random.RandomState(31).normal(size=(32, 4)).astype(np.float32)
truth = np.asarray([.2, -.4, 0., .8], np.float32)
fit_stridge(matrix, (matrix @ truth).reshape(-1, 1), truth, 0.)
assert not any(name == 'torch' or name.startswith('tensorflow') for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-B", "-c", program], cwd=adapter.ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_noiseless_unpenalized_coefficient_recovery(self):
        """Recover declared coefficients when neither observation noise nor l0 penalty acts."""
        matrix, _, inherited = self.four_case()
        truth = np.array([.2, -.4, 0., .8], np.float32)
        target = (matrix @ truth).reshape(-1, 1)
        self.check(matrix, target, inherited, truth, l0=0., recovery_atol=1e-5)

    def test_192_by_90(self):
        """Check the open-library-sized search with three planted nonzero coefficients."""
        rng = np.random.RandomState(443)
        matrix = rng.standard_normal((192, 90)).astype(np.float32)
        truth = np.zeros(90, np.float32)
        truth[[3, 8, 16]] = [-.5, .01, 1.]
        target = (matrix @ truth + .01 * rng.standard_normal(192)).astype(np.float32).reshape(-1, 1)
        self.check(matrix, target, truth * np.float32(.7), truth)

    def test_persisted_l0_and_fixed_round_inheritance(self):
        """Reuse first-round l0 and observe one immutable inherited vector in all 100 trials."""
        matrix, target, initial = self.four_case()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            l0 = adapter.fit_stridge(matrix, target, initial)["l0_penalty"]
        inherited = np.array([.17, -.3, .002, .7], np.float32)
        observed_reads = []
        original = adapter._native_methods

        def monitor(path, columns):
            """Wrap only native STRidge calls to inspect the emulated session snapshot."""
            native = original(path, columns)
            regression = native.STRidge

            def call(*args, **kwargs):
                """Record the inherited vector before delegating the unchanged native trial."""
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
        """Optionally write synthetic case evidence to the explicitly configured directory."""
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
