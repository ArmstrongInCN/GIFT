"""Test-only routing fixtures; real native-terminal integrity has separate tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from adapters import pinn_reference as reference
from experiments.formal.m1_equation_identification import run


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gift-m1-routing-test-only-"))
        self.data = self.root / "data"
        self.data.mkdir()
        self.observations = self.data / run.DATASETS["noise_000"]
        self.observations.write_bytes(b"test-only observation binding; not experimental data")
        self.data_sha = run.digest(self.observations)
        (self.data / "manifest.json").write_text(json.dumps({"files": [dict(path=self.observations.name,
            bytes=self.observations.stat().st_size, sha256=self.data_sha)]}))
        self.checkpoints = self.root / "checkpoints"
        self.records, self.calls = {}, []
        for mode in ("open", "known"):
            path = self.checkpoints / "noise_000" / mode / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"test_only_mock_not_scientific": true}')
            count = 90 if mode == "open" else 4
            self.records[mode] = dict(method=reference.METHODS[mode], condition="noise_000", mode=mode,
                parameters=dict(nu=.01, beta=.9, gamma=.8), coefficients=[0.]+[.1]*(count-1),
                coefficient_names=["fixture_%d" % i for i in range(count)], archive_sha256="A"*64,
                source_data_protocol={"data_sha256": self.data_sha}, test_only_mock_not_scientific=True)
        for patch in (mock.patch.dict(os.environ, {"GIFT_DATA_ROOT": str(self.data)}),
                      mock.patch.object(reference, "REFERENCE_ROOT", self.checkpoints),
                      mock.patch.object(reference, "read_checkpoint", side_effect=self.fixture_read)):
            patch.start()
            self.addCleanup(patch.stop)

    def args(self, method="PINN-SR-KC"):
        return run.parse_args(["--execute", "--method", method, "--condition", "noise_000",
                               "--output", str(self.root / "jobs"), "--device", "cpu"])

    def fixture_read(self, path, condition, mode):
        self.calls.append((str(path), condition, mode))
        self.assertEqual(Path(path), self.checkpoints/condition/mode/"result.json")
        return self.records[mode]

    def test_each_mode_commits_and_binds_completed_native_checkpoint(self):
        for method in ("PINN-SR", "PINN-SR-KC"):
            result = run.execute(self.args(method))
            record = json.loads((Path(result["job"])/"result.json").read_text())
            self.assertEqual(record["readout_scope"], "trained_coefficients_readout")
            self.assertTrue(record["reference_binding"]["test_only_mock_not_scientific"])
            self.assertEqual(len(record["native_coefficients"]), 90 if method == "PINN-SR" else 4)
            self.assertEqual(record["native_coefficients"][0]["real"], 0.)
            for name in ("training", "forward", "training_executed", "forward_executed", "fresh_training"):
                self.assertIs(record[name], False)
            self.assertEqual(record["identity"]["checkpoint"]["artifact_role"], "completed_native_training")
        self.assertEqual(len(self.calls), 6)

    def test_reexecute_revalidates_checkpoint_but_cached_read_needs_neither_data_nor_archive(self):
        args = self.args()
        result = run.execute(args)
        reused = run.execute(args)
        self.assertEqual(reused["status"], "cached_result_read")
        self.assertEqual(len(self.calls), 4)
        with mock.patch.object(reference, "read_checkpoint", side_effect=AssertionError("must not read NPZ")), \
             mock.patch.object(run, "_dataset_binding", side_effect=AssertionError("must not bind data")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run.main(["--use-results", "--method", args.method, "--condition", args.condition,
                          "--output", str(args.output)])
            cached = json.loads(output.getvalue())
        self.assertEqual(cached["parameters"], result["parameters"])
        self.assertFalse(cached["trained_archive_read"])
        with self.assertRaisesRegex(ValueError, "different source"):
            run.read_completed(Path(result["job"]), args.method, args.condition, identity={"wrong": True})

    def test_wrong_data_and_invalid_archive_refused_before_job_creation(self):
        self.records["known"]["source_data_protocol"]["data_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "another dataset"):
            run.execute(self.args())
        self.assertFalse((self.root/"jobs").exists())
        with mock.patch.object(reference, "read_checkpoint", side_effect=ValueError("invalid test archive")):
            with self.assertRaisesRegex(ValueError, "invalid test archive"):
                run.execute(self.args())
        self.assertFalse((self.root/"jobs").exists())

    def test_calculation_failure_is_preserved_and_not_overwritten(self):
        args = self.args()
        with mock.patch.object(run, "calculate", side_effect=ValueError("test-only computation failure")) as calculate:
            with self.assertRaisesRegex(ValueError, "test-only computation failure"):
                run.execute(args)
            pending = run._job_path(args).with_name("pinn_sr_kc.partial")
            self.assertTrue((pending/"FAILURE.json").exists())
            self.assertFalse(run._job_path(args).exists())
            with self.assertRaises(FileExistsError):
                run.execute(args)
            self.assertEqual(calculate.call_count, 1)

    def test_explicit_checkpoint_sha256_checked(self):
        args = self.args()
        args.checkpoint = self.checkpoints/"noise_000"/"known"/"result.json"
        args.checkpoint_sha256 = "0"*64
        with self.assertRaisesRegex(ValueError, "explicit PINN result checksum"):
            run.execute(args)
        args.checkpoint_sha256 = run.digest(args.checkpoint)
        self.assertEqual(run.execute(args)["status"], "complete")

    def test_dry_run_honestly_scopes_checkpoint_readout(self):
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
