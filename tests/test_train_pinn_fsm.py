"""Lightweight mock FSM/state tests only: no TF/Torch/SciPy algorithm execution."""
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from training.train_pinn import Coordinator, CoupledJournal, DEFAULT_BUDGET, budget_for


TINY = {"pre_nadam": 2, "rounds": 6, "ado_nadam": 2, "post_nadam": 3,
        "lbf_maxiter": 5, "lbf_maxfun": 6}


class MockBackend:
    kind = "mock_test_only"
    identity = {"type": "fake_state_machine_fixture_not_PINN", "inputs": {"selection": {"diagnostic_subset": True}}}
    layout = [{"name": name, "shape": [4], "dtype": "float32"} for name in ("fake_weights", "coefficients", "slots", "beta")]

    def __init__(self):
        self.values = [np.arange(4, dtype=np.float32), np.zeros(4, np.float32), np.zeros(4, np.float32), np.ones(4, np.float32)]
        self.support = np.ones((4, 1), np.float32)
        self.iteration = 0

    def state(self):
        return {"variables": [v.copy() for v in self.values], "mask": self.support.copy(), "iteration": self.iteration}

    def restore(self, state):
        assert len(state["variables"]) == 4
        self.values = [v.copy() for v in state["variables"]]
        self.support, self.iteration = state["mask"].copy(), state["iteration"]

    def flat(self):
        return np.concatenate(self.values[:2]).astype(np.float64)

    def assign_flat(self, value):
        self.values[0] = np.asarray(value[:4], np.float32).copy()
        self.values[1] = np.asarray(value[4:], np.float32).copy()

    def objective(self, value):
        self.assign_flat(value)
        return float(np.sum(value)), np.ones_like(value)

    def nadam(self, lr, physics, l1):
        # Deliberately NOT an optimizer implementation; exercises live slots,
        # coefficient regrowth, masks and counters through phase transitions.
        self.values[0] += np.float32(lr)
        self.values[2] += np.float32(1)
        self.values[3] *= np.float32(.99)
        self.values[1] += np.array([.01, 0, .03, 0], np.float32) * self.support.reshape(-1)
        self.iteration += 1
        return {"iteration": self.iteration, "lr": lr, "physics": physics, "l1": l1}

    def coefficients(self):
        return self.values[1].copy()

    def mask(self):
        return self.support.reshape(-1).copy()

    def assign_coefficients(self, value, mask=None):
        self.values[1] = np.asarray(value, np.float32).reshape(4).copy()
        if mask is not None:
            self.support = np.asarray(mask, np.float32).reshape(4, 1).copy()

    def collect(self):
        assert np.array_equal(self.mask(), np.ones(4, np.float32))
        return np.ones((8, 4), np.float32), np.ones((8, 1), np.float32)


def mock_stridge(matrix, target, inherited, l0_penalty=None, *, iteration=0, external_root=None):
    assert matrix.shape == (8, 4) and target.shape == (8, 1)
    assert (l0_penalty is None) if iteration == 0 else (l0_penalty == .125)
    return {"coefficients": np.array([inherited[0]+.1, 0, 0, .4], np.float64),
            "l0_penalty": .125, "best_tolerance": 1.0, "best_error": .25,
            "accepted_history": [{"test_only": True}], "source_binding": {"mock": True},
            "input_binding": {"mock_iteration": iteration}}


class MockSolver:
    accepted_limit = 2

    def __init__(self, x0, identity, options=None):
        self.identity = {"test_only": True, "identity": identity, "options": options}
        self.arrays = {"x": np.asarray(x0, np.float64).copy(), "f": np.asarray(0.0, np.float64)}
        self.counts = {"kernel_calls": 0, "fg_requests": 0, "iterations": 0, "evaluations": 0}
        self.task, self.done = "START", False

    @property
    def x(self):
        return self.arrays["x"].copy()

    def advance(self, objective):
        if self.task.startswith("FG"):
            self.arrays["f"][...] = objective(self.x)[0]
            self.counts["evaluations"] += 1
            if self.counts["iterations"] == self.accepted_limit:
                self.task, self.done = "CONVERGENCE: MOCK_ONLY", True
            else:
                self.arrays["x"] += .25
                self.task = "NEW_X"
                self.counts["iterations"] += 1
        else:
            self.task = "FG_MOCK"
            self.counts["fg_requests"] += 1
        self.counts["kernel_calls"] += 1
        return {"task": self.task, "test_only": True}

    def metadata(self):
        return {"identity": self.identity, "counts": dict(self.counts), "task": self.task, "done": self.done}

    def restore(self, arrays, metadata):
        assert metadata["identity"] == self.identity
        self.arrays = {k: v.copy() for k, v in arrays.items()}
        self.counts, self.task, self.done = dict(metadata["counts"]), metadata["task"], metadata["done"]


class ManyBoundaries(MockSolver):
    accepted_limit = 101


def coordinator(factory=MockSolver):
    return Coordinator(MockBackend(), mock_stridge, factory, budget=TINY, diagnostic=True)


def drive(control, predicate=None):
    trace = []
    while control.state["phase"] != "complete":
        event = control.advance()
        trace.append(event)
        if predicate and predicate(control):
            break
    return trace


class PhaseStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("GIFT_PINN_FSM_TEST_OUTPUT")
        cls.output = Path(configured) if configured else Path(tempfile.mkdtemp(prefix="gift_pinn_fsm_"))
        if configured:
            cls.output.mkdir(parents=True, exist_ok=False)
        cls.completed = []

    @classmethod
    def tearDownClass(cls):
        import sys
        with (cls.output / "RESULTS.json").open("x", encoding="utf-8") as stream:
            json.dump({"kind": "mock_only_not_scientific_result", "checks": cls.completed,
                       "tensorflow_imported": "tensorflow" in sys.modules,
                       "torch_imported": "torch" in sys.modules, "scipy_imported": "scipy" in sys.modules}, stream, indent=2)

    def assert_equal(self, left, right):
        a, am = left.snapshot()
        b, bm = right.snapshot()
        self.assertEqual(am, bm)
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_default_protocol_and_guards(self):
        self.assertEqual(DEFAULT_BUDGET, {"pre_nadam": 5000, "rounds": 6, "ado_nadam": 1000,
                                         "post_nadam": 20000, "lbf_maxiter": 10000, "lbf_maxfun": 10000})
        with self.assertRaises(ValueError):
            budget_for(TINY)
        with self.assertRaises(ValueError):
            budget_for({"rounds": 7}, diagnostic=True)
        with self.assertRaises(ValueError):
            Coordinator(MockBackend(), mock_stridge, MockSolver, diagnostic=True, mode="open")
        with self.assertRaises(ValueError):
            Coordinator(MockBackend(), mock_stridge, MockSolver)
        self.completed.append("original budgets; tiny requires diagnostic; open/mock formal refused")

    def test_joint_state_resume_at_all_selected_phases(self):
        baseline = coordinator()
        expected_trace = drive(baseline)
        predicates = {
            "pre_nadam": lambda c: c.state["phase"] == "pre_nadam" and c.state["phase_step"] == 1,
            "lbfgs_fg": lambda c: c.solver is not None and c.solver.task.startswith("FG"),
            "lbfgs_new_x": lambda c: c.solver is not None and c.solver.task == "NEW_X",
            "before_stridge": lambda c: c.state["phase"] == "ado_stridge" and c.state["round"] == 0,
            "after_stridge": lambda c: c.state["phase"] == "ado_nadam" and c.state["round"] == 0 and c.state["phase_step"] == 0,
            "mid_ado": lambda c: c.state["phase"] == "ado_nadam" and c.state["round"] == 2 and c.state["phase_step"] == 1,
            "before_final_mask": lambda c: c.state["phase"] == "final_mask",
            "post_nadam": lambda c: c.state["phase"] == "post_nadam" and c.state["phase_step"] == 1,
        }
        for name, predicate in predicates.items():
            with self.subTest(boundary=name):
                original = coordinator()
                prefix = drive(original, predicate)
                self.assertTrue(predicate(original))
                journal = CoupledJournal(self.output / name, original)
                journal.save()
                py_next, np_next = random.random(), np.random.random()
                resumed = coordinator()
                CoupledJournal(self.output / name, resumed, resume=True)
                self.assert_equal(original, resumed)
                self.assertEqual(random.random(), py_next)
                self.assertEqual(np.random.random(), np_next)
                self.assertEqual(prefix + drive(resumed), expected_trace)
                self.assert_equal(baseline, resumed)
                self.completed.append("coupled model/slots/mask/l0/FSM/LBF exact at " + name)

    def test_amendment_order_mask_and_slots(self):
        control = coordinator()
        trace = drive(control)
        nadam = [event["metrics"] for event in trace if "metrics" in event]
        self.assertEqual(len(nadam), 17)
        self.assertEqual([m["iteration"] for m in nadam], list(range(1, 18)))
        self.assertEqual([record["iteration"] for record in control.state["stridge_records"]], list(range(6)))
        self.assertEqual(control.state["l0_penalty"], .125)
        np.testing.assert_array_equal(control.backend.mask(), [1, 0, 1, 1])
        # Index2 was zeroed by the last STRidge but regrew during its two joint
        # mock NAdam updates; final mask must include it, not old STRidge support.
        self.assertEqual(control.state["stridge_records"][-1]["coefficients"][2], 0.0)
        self.assertGreater(control.backend.coefficients()[2], 0.0)
        np.testing.assert_array_equal(control.backend.values[2], np.full(4, 17, np.float32))
        self.assertEqual(sum(event["phase"] == "final_mask" for event in trace), 1)
        self.assertTrue(all(event["scientific_acceptance"] is False for event in trace))
        self.assertEqual(sum(event["phase"] == "ado_stridge" for event in trace), 6)
        with self.assertRaises(RuntimeError):
            control.advance()
        self.completed.append("six STRidge; all NAdam joint; same slots; final mask after last ADO NAdam; no post-LBF")

    def test_stridge_call_atomic_before_and_after(self):
        baseline = coordinator()
        drive(baseline)
        original = coordinator()
        drive(original, lambda c: c.state["phase"] == "ado_stridge")
        journal = CoupledJournal(self.output / "stridge_uncommitted", original)
        journal.save()
        original.advance()  # Simulated interruption after return, before commit.
        resumed = coordinator()
        resumed_journal = CoupledJournal(self.output / "stridge_uncommitted", resumed, resume=True)
        self.assertEqual(resumed.state["phase"], "ado_stridge")
        resumed.advance()
        resumed_journal.save()
        again = coordinator()
        CoupledJournal(self.output / "stridge_uncommitted", again, resume=True)
        self.assertEqual(again.state["phase"], "ado_nadam")
        drive(again)
        self.assert_equal(baseline, again)
        self.completed.append("uncommitted STRidge repeats inherited state; committed STRidge not repeated")

    def test_sparse_lbfgs_checkpoint_policy(self):
        control = coordinator(ManyBoundaries)
        selected = []
        while control.state["phase"] in ("pre_nadam", "pre_lbfgs"):
            event = control.advance()
            if control.solver is not None and event["checkpoint_due"]:
                selected.append((control.solver.task, dict(control.solver.counts)))
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0][1]["fg_requests"], 100)
        self.assertEqual(selected[1][1]["iterations"], 100)
        self.completed.append("LBF default every100 FG/NEW_X plus phase boundaries, not each kernel call")

    def test_wrong_identity_stale_writer_and_orphan(self):
        first = coordinator()
        journal = CoupledJournal(self.output / "negative", first)
        journal.save()
        with self.assertRaises(ValueError):
            journal.save()
        second = coordinator()
        stale = CoupledJournal(self.output / "negative", second, resume=True)
        wrong = Coordinator(MockBackend(), mock_stridge, MockSolver, budget={**TINY, "post_nadam": 4}, diagnostic=True)
        with self.assertRaises(ValueError):
            CoupledJournal(self.output / "negative", wrong, resume=True)
        orphan = self.output / "negative" / ("state_000000001_" + "0"*32)
        orphan.mkdir()
        first.advance()
        journal.save()
        self.assertTrue(orphan.exists())
        with self.assertRaises(ValueError):
            stale.save()
        self.completed.append("wrong budget/duplicate/stale writes refused; orphan preserved without blocking new nonce")


if __name__ == "__main__":
    unittest.main(verbosity=2)
