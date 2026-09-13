"""CPU analytic quadratics only; no TensorFlow, PINN graph or trained state.

Fresh-process resume proves that the private kernel is not relying on unsaved
Python-process state. Test output directories are retained, never deleted.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training import pinn_lbfgsb as driver


def problem(case):
    if case == "diagonal":
        matrix = np.diag([1.0, 3.0, 17.0, 80.0])
        center = np.array([.5, -1.0, 2.0, -.25])
    elif case == "coupled":
        base = np.array([[2.0, .5, 0, 1], [0, 3, -.25, 0], [.5, 0, 1, 0], [0, .2, 0, 2]])
        matrix = base.T @ base + .3*np.eye(4)
        center = np.array([1.0, -.4, .7, -2.0])
    else:
        raise ValueError(case)
    x0 = np.array([-2.0, 3.0, -1.0, 4.0])
    def objective(x):
        error = x-center
        return float(.5*np.dot(error, matrix @ error)), matrix @ error
    identity = {"kind": "analytic_quadratic_test_only", "case": case, "matrix": matrix.tolist(), "center": center.tolist()}
    return x0, objective, identity


def array_hashes(solver):
    return {k: driver.hashlib.sha256(v.tobytes()).hexdigest() for k, v in solver.arrays.items()}


def advance_until(solver, objective, stop=None):
    events, evaluations, accepted = [], [], []
    def evaluated(x):
        f, g = objective(x)
        evaluations.append({"x": x.tolist(), "f": f, "g": g.tolist()})
        return f, g
    while not solver.done:
        event = solver.advance(evaluated)
        events.append({"task": event["task"], "x": event["x"].tolist(), "f": event["f"]})
        if event["task"].startswith("NEW_X"):
            accepted.append(event["x"].tolist())
        if stop == "FG" and event["task"].startswith("FG") and event["fg_requests"] >= 3:
            break
        if stop == "NEW_X" and event["task"].startswith("NEW_X") and event["iterations"] == 2:
            break
    return {"events": events, "evaluations": evaluations, "accepted": accepted}


def worker(directory):
    config = json.loads((directory / "TEST_CONFIG.json").read_text(encoding="utf-8"))
    x0, objective, identity = problem(config["case"])
    solver = driver.LBFGSB(x0, identity)
    journal = driver.Journal(directory, solver, resume=True)
    restored = {"hashes": array_hashes(solver), "metadata": json.loads(json.dumps(solver.metadata()))}
    trace = advance_until(solver, objective)
    journal.save()
    report = {"restored": restored, "trace": trace, "x": solver.x.tolist(), "f": float(solver.arrays["f"]),
              "task": solver.task, "counts": solver.counts, "tensorflow_imported": "tensorflow" in sys.modules}
    with (directory / "CHILD_RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


class LBFGSBBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (scipy.__version__, np.__version__) != ("1.7.3", "1.21.6"):
            raise unittest.SkipTest("Use the separate pinned SciPy1.7.3/NumPy1.21.6 environment")
        configured = os.environ.get("GIFT_LBFGSB_TEST_OUTPUT")
        cls.output = Path(configured) if configured else Path(tempfile.mkdtemp(prefix="gift_lbfgsb_"))
        if configured:
            cls.output.mkdir(parents=True, exist_ok=False)
        cls.report = {"scope": "analytic_quadratic_private_api_only", "full_pinn_test": False, "cases": []}

    @classmethod
    def tearDownClass(cls):
        cls.report["source_sha256"] = driver.digest(Path(driver.__file__))
        cls.report["runtime"] = driver.runtime_profile()
        with (cls.output / "RESULTS.json").open("x", encoding="utf-8") as stream:
            json.dump(cls.report, stream, indent=2, allow_nan=False)

    def compare_public(self, case, options=None):
        x0, objective, identity = problem(case)
        solver = driver.LBFGSB(x0, identity, options)
        actual = advance_until(solver, objective)
        official_evaluations, official_accepted = [], []
        def official_fg(x):
            f, g = objective(x)
            official_evaluations.append({"x": x.tolist(), "f": f, "g": g.tolist()})
            return f, g
        expected = minimize(official_fg, x0, jac=True, method="L-BFGS-B",
                            callback=lambda x: official_accepted.append(x.tolist()), options=solver.options)
        np.testing.assert_array_equal(solver.x, expected.x)
        self.assertEqual(float(solver.arrays["f"]), float(expected.fun))
        self.assertEqual(solver.task, expected.message)
        self.assertEqual(solver.counts["iterations"], expected.nit)
        self.assertEqual(solver.counts["evaluations"], expected.nfev)
        self.assertEqual(actual["evaluations"], official_evaluations)
        self.assertEqual(actual["accepted"], official_accepted)
        return solver, actual

    def test_quadratics_match_public_minimize(self):
        for case in ("diagonal", "coupled"):
            with self.subTest(case=case):
                solver, _ = self.compare_public(case)
                self.assertTrue(solver.task.startswith("CONV"))
                self.report["cases"].append({"case": case, "public_endpoint_accepted_and_evaluation_trace": "exact",
                                             "counts": dict(solver.counts), "task": solver.task})

    def test_fresh_process_fg_and_new_x_resume(self):
        for case in ("diagonal", "coupled"):
            continuous, expected = self.compare_public(case)
            for stop in ("FG", "NEW_X"):
                with self.subTest(case=case, boundary=stop):
                    x0, objective, identity = problem(case)
                    interrupted = driver.LBFGSB(x0, identity)
                    prefix = advance_until(interrupted, objective, stop)
                    self.assertFalse(interrupted.done)
                    directory = self.output / (case + "_" + stop)
                    journal = driver.Journal(directory, interrupted)
                    journal.save()
                    driver._json(directory / "TEST_CONFIG.json", {"case": case, "boundary": stop})
                    saved_hashes = array_hashes(interrupted)
                    saved_metadata = interrupted.metadata()
                    process = subprocess.run([sys.executable, "-s", "-B", str(Path(__file__).resolve()), "--resume-worker", str(directory)],
                                             cwd=str(ROOT), capture_output=True, text=True, timeout=30)
                    (directory / "child.stdout.log").write_text(process.stdout, encoding="utf-8")
                    (directory / "child.stderr.log").write_text(process.stderr, encoding="utf-8")
                    self.assertEqual(process.returncode, 0, process.stderr)
                    child = json.loads((directory / "CHILD_RESULT.json").read_text(encoding="utf-8"))
                    self.assertFalse(child["tensorflow_imported"])
                    self.assertEqual(child["restored"]["hashes"], saved_hashes)
                    self.assertEqual(child["restored"]["metadata"], saved_metadata)
                    for key in expected:
                        self.assertEqual(prefix[key] + child["trace"][key], expected[key])
                    np.testing.assert_array_equal(child["x"], continuous.x)
                    self.assertEqual(child["f"], float(continuous.arrays["f"]))
                    self.assertEqual(child["task"], continuous.task)
                    self.assertEqual(child["counts"], continuous.counts)
                    self.report["cases"].append({"case": case, "boundary": stop, "fresh_process_returncode": process.returncode,
                                                 "all_saved_arrays_metadata": "exact", "full_event_evaluation_and_iteration_trace": "exact"})

    def test_budget_stops_match_public(self):
        for options in ({"maxiter": 2}, {"maxfun": 1}):
            with self.subTest(options=options):
                solver, _ = self.compare_public("diagonal", options)
                self.assertTrue(solver.task.startswith("STOP"))
                self.report["cases"].append({"budget_test": options, "public_trace_and_stop": "exact", "counts": dict(solver.counts)})

    def test_identity_integrity_and_stale_writer(self):
        x0, objective, identity = problem("coupled")
        first = driver.LBFGSB(x0, identity)
        first.advance(objective)
        directory = self.output / "negative_journal"
        writer = driver.Journal(directory, first)
        writer.save()
        with self.assertRaises(FileExistsError):
            writer.save()
        with self.assertRaises(ValueError):
            driver.Journal(directory, driver.LBFGSB(x0, {"wrong": True}), resume=True)
        second = driver.LBFGSB(x0, identity)
        stale = driver.Journal(directory, second, resume=True)
        first.advance(objective)
        latest_folder = writer.save()
        with self.assertRaises(ValueError):
            stale.save()
        # Isolated negative fixture, never a science checkpoint: corrupt its
        # bytes to verify hash rejection before arrays reach the Fortran kernel.
        with (latest_folder / "state.npz").open("ab") as stream:
            stream.write(b"intentional-test-corruption")
        with self.assertRaises(ValueError):
            driver.Journal(directory, driver.LBFGSB(x0, identity), resume=True)
        self.report["negative_tests"] = "duplicate, wrong identity, stale writer, corrupted NPZ rejected"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--resume-worker":
        worker(Path(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
