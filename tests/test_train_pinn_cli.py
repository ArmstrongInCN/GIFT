"""No graph/forward: CLI path and create-only terminal contracts using mocks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_train_pinn_fsm import coordinator, drive
from training.train_pinn import (CoupledJournal, ROOT, configured_inputs,
                                 execute_schedule, terminal_export, validate_output)


class TerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = Path(tempfile.mkdtemp(prefix="gift_pinn_terminal_contract_"))

    def test_canonical_path_and_protected_output(self):
        data = self.output / "data"
        data.mkdir()
        _, inputs = configured_inputs("noise_000", data)
        self.assertEqual(inputs["data"], data / "standard_ns_n64_full_spectrum.h5")
        _, noisy = configured_inputs("noise_001", data)
        self.assertEqual(noisy["data"], data / "m1_parameter_identification/noise_001.h5")
        for output in (ROOT / "no_writes", data / "no_writes", data, self.output):
            with self.assertRaises((ValueError, FileExistsError)):
                validate_output(output, data, inputs)
        allowed = self.output / "new_run"
        self.assertEqual(validate_output(allowed, data, inputs), allowed)
        self.assertFalse(allowed.exists())
        with self.assertRaises(ValueError):
            validate_output(allowed, data, inputs, resume=True)

    def test_terminal_numeric_commit_and_completed_resume(self):
        output = self.output / "complete"
        with contextlib.redirect_stdout(io.StringIO()):
            first = execute_schedule(coordinator(), output)
        self.assertFalse(first["already_complete"])
        result = json.loads((output / "result.json").read_text())
        self.assertFalse(result["scientific_acceptance"])
        self.assertTrue(result["diagnostic_test_only"])
        self.assertFalse(result["eligible_for_formal_M1"])
        self.assertEqual(result["library"], ["w", "advection", "laplacian", "q"])
        with np.load(output / "terminal_state.npz", allow_pickle=False) as archive:
            self.assertIn("model_000", archive.files)
            self.assertEqual(archive["raw_coefficients"].shape, (4,))
            np.testing.assert_array_equal(archive["effective_coefficients"], archive["raw_coefficients"] * archive["model_mask"].reshape(-1))
        before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            second = execute_schedule(coordinator(), output, resume=True)
        self.assertTrue(second["already_complete"])
        self.assertEqual(first["completion"], second["completion"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()})
        with self.assertRaises(FileExistsError):
            execute_schedule(coordinator(), output)

    def test_unfinished_or_uncommitted_terminal_refused(self):
        control = coordinator()
        journal = CoupledJournal(self.output / "unfinished", control)
        journal.save()
        with self.assertRaises(ValueError):
            terminal_export(journal)
        drive(control)
        with self.assertRaises(ValueError):
            terminal_export(journal)

    def test_mismatched_existing_numeric_not_overwritten(self):
        control = coordinator()
        drive(control)
        journal = CoupledJournal(self.output / "mismatch", control)
        journal.save()
        target = journal.directory / "terminal_state.npz"
        with target.open("xb") as stream:
            np.savez(stream, unrelated_test_fixture=np.array([1], dtype=np.int64))
        original = target.read_bytes()
        with self.assertRaises(ValueError):
            terminal_export(journal)
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse((journal.directory / "COMPLETED.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
