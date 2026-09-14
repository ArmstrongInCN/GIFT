"""Mock scheduling/commit tests only; these fixtures are not trained PINN models."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from training.train_pinn import Coordinator, CoupledJournal, DEFAULT_BUDGET, budget_for

TINY = {"pre_nadam": 2, "rounds": 6, "ado_nadam": 2, "post_nadam": 3,
        "lbf_maxiter": 5, "lbf_maxfun": 6}

class MockBackend:
    kind = "mock_test_only"
    def __init__(self, columns=4):
        self.columns = columns
        self.identity = {"type": "mock_not_PINN", "columns": columns,
                         "inputs": {"selection": {"diagnostic_subset": True}}}
        self.layout = [{"name": name, "shape": [columns], "dtype": "float32"}
                       for name in ("weights", "coefficients", "slots")]
        self.values = [np.arange(columns, dtype=np.float32),
                       np.zeros(columns, np.float32), np.zeros(columns, np.float32)]
        self.support, self.iteration = np.ones((columns, 1), np.float32), 0
        names = ["w", "advection", "laplacian", "q"] if columns == 4 else ["fixture_%d" % i for i in range(columns)]
        self.runtime = SimpleNamespace(library_names=names)
    def state(self):
        return {"variables": [x.copy() for x in self.values], "mask": self.support.copy(), "iteration": self.iteration}
    def restore(self, state):
        self.values = [x.copy() for x in state["variables"]]
        self.support, self.iteration = state["mask"].copy(), state["iteration"]
    def nadam(self, lr, physics, l1):
        self.values[0] += np.float32(lr)
        self.values[2] += np.float32(1)
        self.values[1][::2] += np.float32(.01) * self.support.reshape(-1)[::2]
        self.iteration += 1
    def coefficients(self):
        return self.values[1].copy()
    def mask(self):
        return self.support.reshape(-1).copy()
    def assign_coefficients(self, values, mask=None):
        self.values[1] = np.asarray(values, np.float32).reshape(self.columns).copy()
        if mask is not None:
            self.support = np.asarray(mask, np.float32).reshape(self.columns, 1).copy()
    def collect(self):
        return np.ones((128, self.columns), np.float32), np.ones((128, 1), np.float32)
    def lbfgs(self, budget):
        self.values[0] += np.float32(.25)
        return {"iterations": 2, "function_evaluations": 3, "termination": "mock_not_solver"}

def mock_stridge(matrix, target, inherited, l0_penalty=None, *, iteration=0, external_root=None):
    assert l0_penalty is None if iteration == 0 else l0_penalty == .125
    coefficients = np.zeros(matrix.shape[1], np.float64)
    coefficients[0], coefficients[-1] = inherited[0] + .1, .4
    return {"coefficients": coefficients, "l0_penalty": .125, "input_binding": {"mock_iteration": iteration}}

def coordinator(mode="known"):
    return Coordinator(MockBackend(4 if mode == "known" else 90), mock_stridge,
                       budget=TINY, diagnostic=True, mode=mode)

def drive(control, predicate=None):
    trace = []
    while control.state["phase"] != "complete":
        trace.append(control.advance())
        if predicate and predicate(control):
            break
    return trace

def semantic_metadata(metadata):
    result = json.loads(json.dumps(metadata))
    result["fsm"].pop("cost")
    return result

class PhaseStateTests(unittest.TestCase):
    def assert_equal(self, left, right):
        a, am = left.snapshot()
        b, bm = right.snapshot()
        self.assertEqual(semantic_metadata(am), semantic_metadata(bm))
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_budgets_and_diagnostic_guard(self):
        self.assertEqual(DEFAULT_BUDGET["pre_nadam"] + DEFAULT_BUDGET["rounds"]*DEFAULT_BUDGET["ado_nadam"]
                         + DEFAULT_BUDGET["post_nadam"], 31000)
        with self.assertRaises(ValueError):
            budget_for(TINY)
        with self.assertRaises(ValueError):
            budget_for({"rounds": 7}, diagnostic=True)
        with self.assertRaises(ValueError):
            Coordinator(MockBackend(), mock_stridge)

    def test_both_libraries_complete_with_joint_updates(self):
        for mode in ("known", "open"):
            control = coordinator(mode)
            trace = drive(control)
            self.assertEqual(control.state["nadam_total"], 17)
            self.assertEqual(control.state["cost"]["nadam_updates"], 17)
            self.assertEqual(control.state["cost"]["stridge_calls"], 6)
            self.assertEqual(len(control.backend.mask()), 4 if mode == "known" else 90)
            self.assertEqual(sum(e["phase"] == "pre_lbfgs" for e in trace), 1)
            self.assertEqual(sum(e["phase"] == "final_mask" for e in trace), 1)
            self.assertEqual([r["iteration"] for r in control.state["stridge_records"]], list(range(6)))
            np.testing.assert_array_equal(control.backend.values[2], np.full(control.columns, 17, np.float32))
            with self.assertRaises(ValueError):
                control.advance()

    def test_resume_at_every_committable_phase(self):
        root = Path(tempfile.mkdtemp(prefix="gift_native_phase_tests_"))
        for mode in ("known", "open"):
            baseline = coordinator(mode)
            drive(baseline)
            for phase in ("pre_nadam", "pre_lbfgs", "ado_stridge", "ado_nadam", "final_mask", "post_nadam"):
                with self.subTest(mode=mode, phase=phase):
                    partial = coordinator(mode)
                    if phase != "pre_nadam":
                        drive(partial, lambda c: c.state["phase"] == phase)
                    folder = root / (mode + "_" + phase)
                    journal = CoupledJournal(folder, partial)
                    journal.save()
                    py_next, np_next = random.random(), np.random.random()
                    resumed = coordinator(mode)
                    CoupledJournal(folder, resumed, resume=True)
                    self.assertEqual(random.random(), py_next)
                    self.assertEqual(np.random.random(), np_next)
                    self.assert_equal(partial, resumed)
                    drive(resumed)
                    self.assert_equal(baseline, resumed)

    def test_uncommitted_lbfgs_restarts_from_phase_start(self):
        folder = Path(tempfile.mkdtemp(prefix="gift_native_lbf_tests_")) / "run"
        partial = coordinator()
        drive(partial, lambda c: c.state["phase"] == "pre_lbfgs")
        journal = CoupledJournal(folder, partial)
        journal.save()
        initial, _ = partial.snapshot()
        partial.backend.values[0] += np.float32(100)  # Simulated interrupted external call.
        resumed = coordinator()
        CoupledJournal(folder, resumed, resume=True)
        actual, _ = resumed.snapshot()
        for key in initial:
            np.testing.assert_array_equal(initial[key], actual[key])
        self.assertEqual(resumed.state["phase"], "pre_lbfgs")
        resumed.advance()
        self.assertEqual(resumed.state["phase"], "ado_stridge")
        self.assertFalse(any("lbfgsb" in key for key in resumed.snapshot()[0]))

    def test_wrong_identity_duplicate_and_stale_writes_rejected(self):
        folder = Path(tempfile.mkdtemp(prefix="gift_native_commit_tests_")) / "run"
        first = coordinator()
        journal = CoupledJournal(folder, first)
        journal.save()
        with self.assertRaises(ValueError):
            journal.save()
        stale = CoupledJournal(folder, coordinator(), resume=True)
        with self.assertRaises(ValueError):
            CoupledJournal(folder, coordinator("open"), resume=True)
        first.advance()
        journal.save()
        with self.assertRaises(ValueError):
            stale.save()

if __name__ == "__main__":
    unittest.main()
