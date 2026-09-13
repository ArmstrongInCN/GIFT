"""NumPy-only reference readout/integrity tests; no graph or training fixtures.

Negative cases mutate only new external test copies. Patching the catalog pin
inside a test isolates deeper schema gates; production exposes no such bypass.
Test directories are retained rather than automatically deleted.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from adapters import pinn_reference as reference


EXPECTED = {
    ("noise_000", "open"): (0.024630296044051647, 0.6550853848457336, 0.6280219554901123),
    ("noise_000", "known"): (0.03012167103588581, 0.6795651912689209, 0.7922863960266113),
    ("noise_001", "open"): (0.024698439985513687, 0.6387129724025726, 0.6868330836296082),
    ("noise_001", "known"): (0.031230388209223747, 0.6807405352592468, 0.7674525380134583),
    ("noise_010", "open"): (0.02113440167158842, 0.5910610258579254, 0.5980175137519836),
    ("noise_010", "known"): (0.031767163425683975, 0.6828765273094177, 0.8023972511291504),
}


class ReferenceTests(unittest.TestCase):
    def fixture(self):
        parent = os.environ.get("GIFT_PINN_REFERENCE_TEST_OUTPUT")
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="pinn-reference-test-only-", dir=parent))
        raw = (reference.REFERENCE_ROOT / "manifest.json").read_bytes()
        (root / "manifest.json").write_bytes(raw)
        catalog = json.loads(raw)
        row = next(item for item in catalog["runs"]
                   if (item["condition"], item["mode"]) == ("noise_000", "known"))
        target = root / row["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes((reference.REFERENCE_ROOT / row["path"]).read_bytes())
        return root, catalog, row

    def fixture_catalog_pin(self, root, catalog):
        raw = (json.dumps(catalog, indent=2) + "\n").encode()
        (root / "manifest.json").write_bytes(raw)
        return mock.patch.object(reference, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())

    def test_all_six_real_reference_readouts(self):
        for pair, expected in EXPECTED.items():
            with self.subTest(condition=pair[0], mode=pair[1]):
                result = reference.read_reference(*pair)
                self.assertEqual(tuple(result["parameters"][key] for key in ("nu", "beta", "gamma")), expected)
                self.assertEqual(result["verified_tensor_count"], 82)
                self.assertEqual(result["readout_scope"], "trained_coefficients_readout")
                self.assertEqual(result["artifact_role"], "reference_not_resume")
                for flag in ("training", "forward", "fresh_training", "full_budget_retraining_verified",
                             "is_resume_checkpoint", "sparse_structure_recovery_claimed"):
                    self.assertIs(result[flag], False)
                self.assertEqual(len(result["coefficients"]), 90 if pair[1] == "open" else 4)
                self.assertEqual(result["nonzero_effective_coefficients"], len(result["coefficients"]))

    def test_no_framework_import_in_clean_child(self):
        project = str(Path(__file__).resolve().parents[1])
        code = ("import sys; sys.path.insert(0, %r); "
                "from adapters.pinn_reference import read_reference; "
                "read_reference('noise_000','known'); "
                "assert 'tensorflow' not in sys.modules and 'torch' not in sys.modules; print('PASS')") % project
        output = subprocess.check_output([sys.executable, "-I", "-B", "-c", code], timeout=30)
        self.assertEqual(output.strip(), b"PASS")

    def test_changed_catalog_refused(self):
        root, _, _ = self.fixture()
        with (root / "manifest.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            reference.read_reference("noise_000", "known", root)

    def test_changed_archive_refused(self):
        root, _, row = self.fixture()
        target = root / row["path"]
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 1
        target.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "archive size/SHA256"):
            reference.read_reference("noise_000", "known", root)

    def test_condition_mode_path_and_schema_refused(self):
        for condition, mode in (("noise_100", "known"), ("noise_000", "PINN-SR")):
            with self.subTest(condition=condition, mode=mode), self.assertRaises(ValueError):
                reference.read_reference(condition, mode)
        for field in ("path", "shape", "role"):
            root, catalog, row = self.fixture()
            if field == "path":
                row["path"] = "../outside.npz"
            elif field == "shape":
                catalog["tensor_schemas"]["known"][0]["shape"] = [1]
            else:
                row["artifact_role"] = "resume_checkpoint"
            with self.subTest(field=field), self.fixture_catalog_pin(root, catalog), self.assertRaises(ValueError):
                reference.read_reference("noise_000", "known", root)

    def test_numeric_array_gates_even_with_fixture_checksums(self):
        for defect in ("float64", "nonfinite", "mask", "object", "missing"):
            root, catalog, row = self.fixture()
            target = root / row["path"]
            with np.load(target, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            mask_key = next(item["key"] for item in catalog["tensor_schemas"]["known"]
                            if item["name"] == "coefficient_mask")
            if defect == "float64":
                arrays[mask_key] = arrays[mask_key].astype(np.float64)
            elif defect == "nonfinite":
                arrays[mask_key][0, 0] = np.nan
            elif defect == "mask":
                arrays[mask_key][0, 0] = 0.5
            elif defect == "object":
                arrays[mask_key] = np.zeros((4, 1), dtype=object)
            else:
                del arrays[mask_key]
            with target.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            raw = target.read_bytes()
            row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            for key, array in arrays.items():
                row["array_sha256"][key] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
            with self.subTest(defect=defect), self.fixture_catalog_pin(root, catalog), self.assertRaises(ValueError):
                reference.read_reference("noise_000", "known", root)

    def test_parameters_are_computed_not_trusted_metadata(self):
        root, catalog, row = self.fixture()
        row["parameters"] = {"nu": 999, "beta": 999, "gamma": 999}
        with self.fixture_catalog_pin(root, catalog):
            result = reference.read_reference("noise_000", "known", root)
        self.assertEqual(tuple(result["parameters"].values()), EXPECTED[("noise_000", "known")])


if __name__ == "__main__":
    unittest.main()
