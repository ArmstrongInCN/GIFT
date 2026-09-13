"""Test-only small NPZ/manifest fixtures; no released states/downloads required.

The strict production 82-array reader has separate tests. These mocks isolate
M1 routing, original-data binding, independent commit and cached-result behavior.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from adapters import pinn_reference as reference
from experiments.formal.m1_equation_identification import run


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gift_m1_pinn_fixture_"))
        self.data = self.root / "data"
        self.data.mkdir()
        self.observations = self.data / run.DATASETS["noise_000"]
        self.observations.write_bytes(b"test-only observation binding; not experimental data")
        self.data_sha = run.digest(self.observations)
        (self.data / "manifest.json").write_text(json.dumps({"files": [dict(path=self.observations.name,
            bytes=self.observations.stat().st_size, sha256=self.data_sha)]}))
        self.catalog_root = self.root / "reference"
        self.catalog_root.mkdir()
        self.catalog = {"runs": []}
        for mode in ("open", "known"):
            relative = "noise_000/" + mode + "/terminal_state.npz"
            path = self.catalog_root / relative
            path.parent.mkdir(parents=True)
            coefficients = np.arange(1, (90 if mode == "open" else 4)+1, dtype=np.float32).reshape(-1, 1) / 100
            mask = np.ones_like(coefficients)
            mask[0] = 0
            with path.open("xb") as stream:
                np.savez(stream, coefficients=coefficients, mask=mask)
            self.catalog["runs"].append(dict(condition="noise_000", mode=mode, path=relative,
                bytes=path.stat().st_size, sha256=run.digest(path).lower(),
                source_data_protocol={"data_sha256": self.data_sha.lower()},
                parameters={"nu": 999, "beta": 999, "gamma": 999}))
        self.calls = []
        patches = [mock.patch.dict(os.environ, {"GIFT_DATA_ROOT": str(self.data)}),
                   mock.patch.object(reference, "REFERENCE_ROOT", self.catalog_root),
                   mock.patch.object(reference, "_manifest", return_value=self.catalog),
                   mock.patch.object(reference, "read_reference", side_effect=self.fixture_read)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def args(self, method="PINN-SR-KC"):
        return run.parse_args(["--execute", "--method", method, "--condition", "noise_000",
                               "--output", str(self.root / "jobs"), "--device", "cpu"])

    def fixture_read(self, condition, mode):
        self.calls.append((condition, mode))
        row = next(item for item in self.catalog["runs"] if item["mode"] == mode)
        with np.load(self.catalog_root / row["path"], allow_pickle=False) as archive:
            effective = (archive["coefficients"] * archive["mask"]).reshape(-1).astype(np.float64).tolist()
        # Routing fixture, NOT the scientific physical-coordinate projection.
        return dict(method="PINN-SR-KC" if mode == "known" else "PINN-SR", condition=condition,
            mode=mode, parameters=dict(nu=effective[1], beta=effective[2], gamma=effective[3]),
            coefficients=effective, coefficient_names=["fixture_%d" % i for i in range(len(effective))],
            source_data_protocol=row["source_data_protocol"], test_only_mock_not_scientific=True)

    def test_each_mode_reads_numeric_values_and_commits(self):
        for method in ("PINN-SR", "PINN-SR-KC"):
            result = run.execute(self.args(method))
            record = json.loads((Path(result["job"]) / "result.json").read_text())
            self.assertEqual(record["readout_scope"], "trained_coefficients_readout")
            self.assertNotEqual(record["parameters"], {"nu": 999, "beta": 999, "gamma": 999})
            self.assertTrue(record["reference_binding"]["test_only_mock_not_scientific"])
            self.assertEqual(len(record["native_coefficients"]), 90 if method == "PINN-SR" else 4)
            self.assertEqual(record["native_coefficients"][0]["real"], 0.0)
            for name in ("training", "forward", "training_executed", "forward_executed", "fresh_training"):
                self.assertIs(record[name], False)
            self.assertEqual(record["identity"]["checkpoint"]["artifact_role"], "reference_not_resume")
        self.assertEqual(self.calls, [("noise_000", "open"), ("noise_000", "known")])

    def test_cached_result_and_strict_reexecute_do_not_load_numpy(self):
        args = self.args()
        result = run.execute(args)
        with mock.patch.object(np, "load", side_effect=AssertionError("must not load NPZ again")):
            reused = run.execute(args)
            self.assertEqual(reused["status"], "cached_result_read")
            with mock.patch.object(run, "_dataset_binding", side_effect=AssertionError("cached read must not bind data")):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    run.main(["--use-results", "--method", args.method, "--condition", args.condition,
                              "--output", str(args.output)])
                cached = json.loads(output.getvalue())
        self.assertEqual(cached["parameters"], result["parameters"])
        self.assertFalse(cached["trained_archive_read"])
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(ValueError, "different source"):
            run.read_completed(Path(result["job"]), args.method, args.condition, identity={"test_wrong_source": True})

    def test_wrong_original_data_refused_before_job_creation(self):
        self.catalog["runs"][1]["source_data_protocol"]["data_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "selected original data"):
            run.execute(self.args())
        self.assertFalse((self.root / "jobs").exists())
        self.assertEqual(self.calls, [])

    def test_failed_numeric_reader_preserved_not_retried(self):
        args = self.args()
        with mock.patch.object(reference, "read_reference", side_effect=ValueError("invalid fixture archive")) as reader:
            with self.assertRaisesRegex(ValueError, "invalid fixture archive"):
                run.execute(args)
            pending = run._job_path(args).with_name("pinn_sr_kc.partial")
            self.assertTrue((pending / "FAILURE.json").exists())
            self.assertFalse(run._job_path(args).exists())
            with self.assertRaises(FileExistsError):
                run.execute(args)
            self.assertEqual(reader.call_count, 1)

    def test_dry_run_honestly_scopes_reference_readout(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run.main(["--dry-run", "--method", "PINN-SR", "--condition", "noise_000"])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["scope"], "trained_coefficients_readout")
        self.assertFalse(plan["training"])
        self.assertFalse(plan["forward"])
        self.assertFalse(plan["archive_availability_checked"])
        self.assertEqual(plan["writes"], 0)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
