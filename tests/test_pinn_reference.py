"""Synthetic native-terminal schema tests, never scientific training evidence.

Every fixture is explicitly marked and kept in an external temporary directory.
Positive tests exercise the formal schema using synthetic arrays; negative tests
rebind checksums only to reach the deeper consistency gates.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from adapters import pinn_reference as reference


def write_fixture(root, mode="known"):
    root.mkdir(parents=True, exist_ok=False)
    (root / "TEST_FIXTURE.txt").write_text("UNIT TEST ONLY: synthetic arrays; no training was executed.\n")
    names = reference.expected_library(mode)
    layout = reference.expected_layout(len(names))
    arrays = {"model_%03d" % index: np.zeros(item["shape"], np.float32)
              for index, item in enumerate(layout)}
    coefficients = np.arange(1, len(names)+1, dtype=np.float32) / 100
    mask = np.ones((len(names), 1), np.float32)
    mask[0] = 0
    arrays.update(model_mask=mask, raw_coefficients=coefficients,
                  effective_coefficients=coefficients * mask.reshape(-1))
    arrays["model_018"] = coefficients.reshape(-1, 1).copy()
    cost = {"nadam_updates": 31000, "stridge_calls": 6, "phase_seconds": {}}
    record = {"schema": "gift.pinn-terminal.v1", "run_id": "UNIT_TEST_SYNTHETIC_NOT_TRAINED",
        "mode": mode, "training_completed": True, "diagnostic_test_only": False,
        "eligible_for_formal_M1": True, "requested_budget": reference.BUDGET.copy(),
        "library": names, "raw_coefficients": coefficients.tolist(), "mask": mask.reshape(-1).tolist(),
        "effective_coefficients": arrays["effective_coefficients"].tolist(), "training_cost": cost,
        "terminal_snapshot": {"identity": {"mode": mode, "budget": reference.BUDGET.copy(),
            "diagnostic_test_only": False, "backend": {"inputs": {"condition": "noise_000",
                "selection": {"diagnostic_subset": False, "train_rows": 24000, "physics_rows": 84000},
                "data": {"sha256": "A"*64}, "sampling": {"sha256": "B"*64}}}},
            "fsm": {"phase": "complete", "nadam_total": 31000, "round": 5,
                    "stridge_records": [{} for _ in range(6)], "final_mask": True, "cost": cost},
            "model_iteration": 31000, "model_layout": layout}}
    save_fixture(root, record, arrays)
    return record, arrays


def save_fixture(root, record, arrays):
    np.savez_compressed(root / "terminal_state.npz", **arrays)
    record["terminal_state_sha256"] = reference.digest((root / "terminal_state.npz").read_bytes())
    raw = json.dumps(record, allow_nan=False).encode()
    (root / "result.json").write_bytes(raw)
    (root / "COMPLETED.json").write_text(json.dumps({
        "schema": "gift.pinn-completion.v1", "run_id": record["run_id"],
        "result_sha256": reference.digest(raw),
        "terminal_state_sha256": record["terminal_state_sha256"], "diagnostic_test_only": False}))


class ReferenceTests(unittest.TestCase):
    def fixture(self, mode="known"):
        parent = Path(tempfile.mkdtemp(prefix="gift-native-reader-test-only-"))
        root = parent / "synthetic"
        record, arrays = write_fixture(root, mode)
        return root, record, arrays

    def test_known_and_open_read_coefficients_and_physical_forcing(self):
        for mode in ("known", "open"):
            root, record, arrays = self.fixture(mode)
            result = reference.read_checkpoint(root / "result.json", "noise_000", mode)
            values = dict(zip(record["library"], arrays["effective_coefficients"].astype(float)))
            expected = ({"nu": values["laplacian"], "beta": -values["advection"], "gamma": values["q"]}
                        if mode == "known" else {"nu": (values["w_{xx}"]+values["w_{yy}"])/2,
                        "beta": -(values["u**1w_{x}"]+values["v**1w_{y}"])/2, "gamma": values["q"]})
            self.assertEqual(result["parameters"], expected)
            self.assertEqual(result["verified_tensor_count"], 59)
            self.assertFalse(result["training"])
            self.assertFalse(result["forward"])
            self.assertFalse(result["sparse_structure_recovery_claimed"])

    def test_no_tensorflow_or_torch_import(self):
        root, _, _ = self.fixture()
        project = str(Path(__file__).resolve().parents[1])
        code = ("import sys; sys.path.insert(0, %r); "
                "from adapters.pinn_reference import read_checkpoint; "
                "read_checkpoint(%r, 'noise_000', 'known'); "
                "assert 'tensorflow' not in sys.modules and 'torch' not in sys.modules; print('PASS')") % (project, str(root/"result.json"))
        self.assertEqual(subprocess.check_output([sys.executable, "-I", "-B", "-c", code], timeout=30).strip(), b"PASS")

    def test_incomplete_diagnostic_and_wrong_condition_rejected(self):
        for defect in ("diagnostic", "budget", "rows", "phase", "steps", "rounds", "mask", "condition", "cost"):
            root, record, arrays = self.fixture()
            snapshot = record["terminal_snapshot"]
            if defect == "diagnostic":
                record["diagnostic_test_only"] = True
            elif defect == "budget":
                record["requested_budget"]["pre_nadam"] = 1
            elif defect == "rows":
                snapshot["identity"]["backend"]["inputs"]["selection"]["train_rows"] = 64
            elif defect == "phase":
                snapshot["fsm"]["phase"] = "post_nadam"
            elif defect == "steps":
                snapshot["model_iteration"] = 6
            elif defect == "rounds":
                snapshot["fsm"]["stridge_records"].pop()
            elif defect == "mask":
                snapshot["fsm"]["final_mask"] = False
            elif defect == "condition":
                snapshot["identity"]["backend"]["inputs"]["condition"] = "noise_010"
            else:
                snapshot["fsm"]["cost"] = {"nadam_updates": 10}
            save_fixture(root, record, arrays)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                reference.read_checkpoint(root/"result.json", "noise_000", "known")

    def test_archive_state_library_and_json_consistency(self):
        for defect in ("dtype", "nonfinite", "mask_dtype", "mask_shape", "mask_value", "lambda",
                       "layout_name", "layout_shape", "library", "missing", "effective", "json"):
            root, record, arrays = self.fixture()
            if defect == "dtype":
                arrays["model_000"] = arrays["model_000"].astype(np.float64)
            elif defect == "nonfinite":
                arrays["model_000"][0, 0] = np.nan
            elif defect == "mask_dtype":
                arrays["model_mask"] = arrays["model_mask"].astype(np.float64)
            elif defect == "mask_shape":
                arrays["model_mask"] = arrays["model_mask"].reshape(-1)
            elif defect == "mask_value":
                arrays["model_mask"][0, 0] = .5
            elif defect == "lambda":
                arrays["model_018"][0, 0] = 5
            elif defect == "layout_name":
                record["terminal_snapshot"]["model_layout"][0]["name"] = "not_native:0"
            elif defect == "layout_shape":
                record["terminal_snapshot"]["model_layout"][0]["shape"] = [180]
                arrays["model_000"] = arrays["model_000"].reshape(-1)
            elif defect == "library":
                record["library"][0], record["library"][1] = record["library"][1], record["library"][0]
            elif defect == "missing":
                del arrays["model_058"]
            elif defect == "effective":
                arrays["effective_coefficients"][0] = 1
            else:
                record["raw_coefficients"][0] = 5
            save_fixture(root, record, arrays)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                reference.read_checkpoint(root/"result.json", "noise_000", "known")

    def test_unbound_result_archive_and_duplicate_json_rejected(self):
        for defect in ("result", "archive", "duplicate"):
            root, record, arrays = self.fixture()
            path = root/("terminal_state.npz" if defect == "archive" else "result.json")
            raw = path.read_bytes()
            if defect == "duplicate":
                raw = raw[:-1] + b', "mode": "known"}'
                path.write_bytes(raw)
                completion = json.loads((root/"COMPLETED.json").read_text())
                completion["result_sha256"] = reference.digest(raw)
                (root/"COMPLETED.json").write_text(json.dumps(completion))
            else:
                path.write_bytes(raw+b" ")
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                reference.read_checkpoint(root/"result.json", "noise_000", "known")


if __name__ == "__main__":
    unittest.main()
