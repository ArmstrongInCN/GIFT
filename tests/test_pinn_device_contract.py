"""CPU-only parser/AST/identity contracts; no TensorFlow graph or model calls."""
import ast
import contextlib
import copy
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

from adapters import pinn_runtime as runtime
from training import train_pinn as training


class DeviceContractTests(unittest.TestCase):
    def test_cli_cpu_default_and_explicit_gpu_plan(self):
        for option, expected in (([], "cpu"), (["--device", "cpu"], "cpu"), (["--device", "gpu"], "gpu")):
            output = io.StringIO()
            with mock.patch("sys.argv", ["train_pinn", "--output", "unused-plan-only"] + option), contextlib.redirect_stdout(output):
                training.main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["device"], expected)
            self.assertTrue(plan["plan_only"])
            self.assertFalse(plan["full_budget_validated"])
        self.assertEqual(inspect.signature(runtime.PINNRuntime).parameters["device"].default, "cpu")
        self.assertEqual(inspect.signature(training.TensorFlowBackend).parameters["device"].default, "cpu")

    def test_device_environment_and_open_rejected_before_tf_import(self):
        thread_environment = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        for device, visible in (("cpu", "0"), ("gpu", "-1"), ("invalid", "-1")):
            with mock.patch.dict(os.environ, dict(thread_environment, CUDA_VISIBLE_DEVICES=visible)):
                with self.assertRaises(ValueError):
                    runtime.PINNRuntime("known", None, None, None, device=device)
        with mock.patch.dict(os.environ, dict(thread_environment, CUDA_VISIBLE_DEVICES="0")):
            with self.assertRaises(ValueError):
                runtime.PINNRuntime("open", None, None, None, device="gpu")
        with mock.patch("sys.argv", ["train_pinn", "--mode", "open", "--device", "gpu", "--output", "unused-plan-only"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                training.main()

    def test_known_mask_exact_ast_only_and_changed_declaration_refused(self):
        source = inspect.getsource(runtime._class)
        start = source.index("    # GPU02_KNOWN_MASK_SHAPE_BEGIN:")
        end = source.index("    # GPU02_KNOWN_MASK_SHAPE_END")
        edit = compile(ast.parse(textwrap.dedent(source[start:end])), "<first-party-mask-shape-contract>", "exec")
        selected = ast.parse("class Fixture:\n    self.NonZeroMask_w_tf = tf.placeholder(tf.float32)\n").body[0]
        known = copy.deepcopy(selected)
        exec(edit, dict(ast=ast, selected=known, mode="known"))
        expected = ast.parse("tf.placeholder(tf.float32, shape=[4,1])", mode="eval").body
        self.assertEqual(ast.dump(known.body[0].value), ast.dump(expected))
        unchanged = copy.deepcopy(selected)
        exec(edit, dict(ast=ast, selected=unchanged, mode="open"))
        self.assertEqual(ast.dump(unchanged), ast.dump(selected))
        for changed in (ast.parse("class Fixture:\n    self.NonZeroMask_w_tf = tf.placeholder(tf.float32, shape=None)\n").body[0],
                        ast.ClassDef(name="Fixture", bases=[], keywords=[], body=selected.body*2, decorator_list=[])):
            with self.assertRaises(ValueError):
                exec(edit, dict(ast=ast, selected=changed, mode="known"))

    def test_old_cpu_source_identity_cannot_resume_new_source(self):
        # Deliberately a metadata fixture, not the archived real CPU state.
        directory = Path(tempfile.mkdtemp(prefix="gift_pinn_old_source_rejection_"))
        old = {"source_sha256": "8B902415FEE75AE09B982212D21B6FFA85AAA8114069822C450C5718CE5920D3",
               "backend": {"source_sha256": "02A2FF0C8005964F4D932A9FB67E3A83A9B090FB756967B96AE4A1CB7D3ADD59"}}
        with (directory / "RUN.json").open("x", encoding="utf-8") as stream:
            json.dump({"run_id": "metadata_fixture_not_real_training", "identity": old}, stream)
        current = {"source_sha256": runtime.sha256(training.__file__),
                   "backend": {"source_sha256": runtime.sha256(runtime.__file__)}}
        self.assertNotEqual(old, current)
        control = type("MetadataOnlyCoordinator", (), {"identity": current})()
        with mock.patch.object(training.CoupledJournal, "_load", side_effect=AssertionError("Must reject before loading state")):
            with self.assertRaisesRegex(ValueError, "same phase schedule/source/data/numeric run"):
                training.CoupledJournal(directory, control, resume=True)
        self.assertEqual({path.name for path in directory.iterdir()}, {"RUN.json"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
