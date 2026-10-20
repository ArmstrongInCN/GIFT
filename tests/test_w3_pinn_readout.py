"""Test-only routing fixtures; real native-terminal integrity has separate tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pytest

from adapters import pinn_reference as reference
from experiments.formal.w3_equation_identification import run


class ReadoutTests(unittest.TestCase):
    """Exercise completed-PINN coefficient routing with labelled mock records.

    These fixtures do not run TensorFlow training or prove native optimizer
    completion; archive reading and native-terminal integrity have separate tests.
    """
    def setUp(self):
        """Create temporary observation bytes and open/known-mode coefficient records, with scoped adapter patches."""
        self.root = Path(tempfile.mkdtemp(prefix="gift-w3-routing-test-only-"))
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
        """Build an explicit CPU W3 execution command rooted in this test's temporary jobs directory."""
        return run.parse_args(["--execute", "--method", method, "--condition", "noise_000",
                               "--output", str(self.root / "jobs"), "--device", "cpu"])

    def fixture_read(self, path, condition, mode):
        """Track coefficient-result reads and enforce the expected method/condition path."""
        self.calls.append((str(path), condition, mode))
        self.assertEqual(Path(path), self.checkpoints/condition/mode/"result.json")
        return self.records[mode]

    def test_each_mode_commits_and_binds_completed_native_checkpoint(self):
        """Both PINN modes must commit their learned coefficient readout without claiming training, forward or sparse regression."""
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
        """Re-execution validates current scientific inputs, while direct cached reading needs only committed job files."""
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
        identity = json.loads((Path(result["job"]) / "result.json").read_text())["identity"]
        identity["sources"] = {"updated.py": "NEW"}
        identity["runtime"] = {"python": "new"}
        self.assertEqual(run.read_completed(Path(result["job"]), args.method, args.condition,
                                          identity=identity)["status"], "cached_result_read")
        identity["data"]["sha256"] = "changed-input"
        with self.assertRaisesRegex(ValueError, "different method, data or checkpoint"):
            run.read_completed(Path(result["job"]), args.method, args.condition, identity=identity)

    def test_missing_historical_data_hash_allows_actual_checkpoint_read(self):
        """Absent old observation fingerprints must not prohibit reading a valid completed coefficient result."""
        for mode in ("open", "known"):
            self.records[mode]["source_data_protocol"]["data_sha256"] = None
        for method in ("PINN-SR", "PINN-SR-KC"):
            result = run.execute(self.args(method))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["parameters"], self.records["known" if method.endswith("KC") else "open"]["parameters"])

    def test_wrong_data_and_invalid_archive_refused_before_job_creation(self):
        """Explicit input mismatches or adapter-invalid results must fail before job output creation."""
        self.records["known"]["source_data_protocol"]["data_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "another dataset"):
            run.execute(self.args())
        self.assertFalse((self.root/"jobs").exists())
        with mock.patch.object(reference, "read_checkpoint", side_effect=ValueError("invalid test archive")):
            with self.assertRaisesRegex(ValueError, "invalid test archive"):
                run.execute(self.args())
        self.assertFalse((self.root/"jobs").exists())

    def test_calculation_failure_is_preserved_and_not_overwritten(self):
        """A failed calculation leaves its partial/failure record and a rerun cannot overwrite that evidence."""
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
        """An explicitly requested coefficient-result checksum is honored and a wrong value is rejected."""
        args = self.args()
        args.checkpoint = self.checkpoints/"noise_000"/"known"/"result.json"
        args.checkpoint_sha256 = "0"*64
        with self.assertRaisesRegex(ValueError, "explicit PINN result checksum"):
            run.execute(args)
        args.checkpoint_sha256 = run.digest(args.checkpoint)
        self.assertEqual(run.execute(args)["status"], "complete")

    def test_explicit_checkpoint_without_checksum_is_readable(self):
        """An explicit completed result may be read without requiring the caller to supply its digest."""
        args = self.args()
        args.checkpoint = self.checkpoints / "noise_000" / "known" / "result.json"
        self.assertIsNone(args.checkpoint_sha256)
        self.assertEqual(run.execute(args)["status"], "complete")

    def test_dry_run_honestly_scopes_checkpoint_readout(self):
        """Planning must claim coefficient reading only and avoid output writes or archive inspection."""
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


@pytest.mark.parametrize("mode", ["known", "open"])
def test_execute_without_manifest_reads_coefficients_and_rejects_changed_input(tmp_path, monkeypatch, mode):
    """Real readers on synthetic HDF/coefficient fixtures; no native training."""
    import h5py
    import numpy as np

    data = tmp_path / "data"
    data.mkdir()
    observations = data / run.DATASETS["noise_000"]
    with h5py.File(observations, "x") as handle:
        handle.attrs["fixture_only"] = True
        group = handle.create_group("training")
        group["trajectory_index"] = [0]
        group["time"] = np.arange(3) * .02
        group["vorticity"] = np.ones((1, 3, 4, 4), dtype=np.float32)
    assert not (data / "manifest.json").exists()
    names = reference.expected_library(mode)
    values = np.zeros(len(names))
    coefficients = ({"laplacian": .01, "advection": -1., "q": 1.} if mode == "known"
                    else {"w_{xx}": .01, "w_{yy}": .01,
                          "u**1w_{x}": -1., "v**1w_{y}": -1., "q": 1.})
    for name, value in coefficients.items():
        values[names.index(name)] = value
    checkpoint = tmp_path / "synthetic_coefficients.json"
    checkpoint.write_text(json.dumps(dict(mode=mode, condition="noise_000", training_completed=True,
        library=names, raw_coefficients=values.tolist(), mask=np.ones(len(names)).tolist(),
        effective_coefficients=values.tolist(), inputs={"data": {"sha256": None}},
        fixture_only=True)), encoding="utf-8")
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    calls = []
    actual_digest = run.digest
    def measured(path):
        """Count observation fingerprint reads while retaining the real file digest implementation."""
        if Path(path) == observations:
            calls.append(path)
        return actual_digest(path)
    monkeypatch.setattr(run, "digest", measured)
    args = run.parse_args(["--execute", "--method", reference.METHODS[mode],
        "--condition", "noise_000", "--checkpoint", str(checkpoint),
        "--output", str(tmp_path / "jobs"), "--device", "cpu"])
    result = run.execute(args)
    assert result["parameters"] == dict(nu=.01, beta=1., gamma=1.)
    assert len(calls) == 1
    committed = Path(result["job"]) / "result.json"
    original = committed.read_bytes()
    recorded = json.loads(original)
    assert recorded["identity"]["data"]["integrity_scope"] == "selected_input_file; package_audit_not_performed"
    assert run.execute(args)["status"] == "cached_result_read"
    assert len(calls) == 2  # One input fingerprint per invocation, including cache reuse.
    with h5py.File(observations, "r+") as handle:
        handle["training/vorticity"][0, 0, 0, 0] = 2.
    with pytest.raises(ValueError, match="different method, data or checkpoint"):
        run.execute(args)
    assert len(calls) == 3
    assert committed.read_bytes() == original


if __name__ == "__main__":
    unittest.main(verbosity=2)
