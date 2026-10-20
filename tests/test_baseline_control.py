"""Fast protocol tests; real-model CPU continuation tests require explicit opt-in.

GIFT_BASELINE_TEST_ROOT must be a NEW external directory. Generated fixtures and
all checkpoints are retained, never cleaned up or mixed with scientific runs.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import h5py
import numpy as np
import torch

from training.baseline_control import PROTOCOLS, _arguments, configuration, epoch_schedule, objective, save_boundary, training_cost


class ProtocolTests(unittest.TestCase):
    """Fast scheduling/loss tests that do not need native external baseline checkouts."""
    def test_training_cost_is_sum_of_committed_epochs(self):
        """Only committed epoch durations contribute to reported training time; update totals remain explicit."""
        actual = training_cost([{"epoch_seconds": 1.5}, {"epoch_seconds": 2.25}], 126)
        self.assertEqual(actual["completed_epochs"], 2)
        self.assertEqual(actual["optimizer_updates"], 126)
        self.assertEqual(actual["committed_epoch_seconds"], 3.75)

    def test_formal_budgets(self):
        """Keep each family's documented epoch, microbatch, accumulation and rollout defaults."""
        self.assertEqual([(PROTOCOLS[name]["epochs"], PROTOCOLS[name]["micro_batch"],
                           PROTOCOLS[name]["accumulation"], PROTOCOLS[name]["rollout"])
                          for name in ("fno2d", "fno3d", "uno", "unet")],
                         [(500, 10, 2, 150), (500, 5, 2, 150), (500, 16, 1, 20), (500, 20, 1, 4)])

    def test_tiny_is_distinct_budget(self):
        """Diagnostic U-NO settings must change both artifact status and actual training counts."""
        config = configuration("uno", _arguments("uno", ["--tiny"]))
        self.assertFalse(config["formal"])
        self.assertEqual((config["epochs"], config["micro_batch"], config["rollout"]), (2, 1, 1))

    def test_uno_schedule_is_epoch_local(self):
        """Global NumPy reseeding must not change a seed/epoch-specific U-NO window schedule."""
        config = dict(PROTOCOLS["uno"], model="uno")
        first = epoch_schedule(config, 0, 501)
        np.random.seed(891)
        second = epoch_schedule(config, 0, 501)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(len(first[0]), 1000)
        self.assertLessEqual(first[1].max(), 435)

    def test_unet_closed_loop_keeps_gradient(self):
        """Verify the second rollout prediction retains its gradient path through the first prediction."""
        class Toy(torch.nn.Module):
            """Scalar autoregressive fixture with an analytic two-step gradient."""
            def __init__(self):
                """Choose scale 0.4 so the expected undetached rollout gradient is -1.8."""
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.4))
            def forward(self, history):
                """Predict one frame from the last history channel while preserving autograd."""
                return self.scale * history[:, -1:]
        network = Toy()
        loss, _ = objective(network, torch.ones(1, 46, 64, 64), torch.ones(1, 2, 64, 64), "unet")
        loss.backward()
        self.assertAlmostEqual(float(network.scale.grad), -1.8, places=5)

    def test_checkpoint_interval_and_forced_boundaries(self):
        """Intervals, explicit pauses, final epochs and diagnostics must all select durable boundaries."""
        config = configuration("unet", _arguments("unet", []))
        self.assertEqual(config["checkpoint_interval"], 10)
        self.assertFalse(save_boundary(9, config, 500))
        self.assertTrue(save_boundary(10, config, 500))
        self.assertTrue(save_boundary(4, config, 4))
        self.assertTrue(save_boundary(500, config, 500))
        self.assertTrue(save_boundary(1, dict(config, formal=False), 2))


def _assert_tree_equal(test, left, right, *, float_roundoff=False, path="state"):
    """Check full state, allowing bounded U-Net CPU float accumulation roundoff.

    Only floating model/Adam arrays may differ. The measured two-epoch CPU
    comparison had max tensor-scaled error 108 eps and relative L2 4.5 eps;
    limits of 128 eps and 8 eps respectively allow that backend variation.
    Discrete steps, RNG, schedules, history and exported bytes stay exact.
    """
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertEqual((left.dtype, left.shape), (right.dtype, right.shape), path)
        if left.is_floating_point() or left.is_complex():
            test.assertTrue(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), path)
        if float_roundoff and left.is_floating_point() and not path.endswith("/step") and left.numel():
            scale = max(float(left.abs().max()), float(right.abs().max()))
            eps = torch.finfo(left.dtype).eps
            difference = (left - right).double()
            test.assertLessEqual(float(difference.abs().max()), 128 * eps * scale, path)
            norm = max(float(torch.linalg.vector_norm(left.double())),
                       float(torch.linalg.vector_norm(right.double())))
            test.assertLessEqual(float(torch.linalg.vector_norm(difference)), 8 * eps * norm, path)
        else:
            test.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, dict):
        test.assertIsInstance(right, dict, path)
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_tree_equal(test, left[key], right[key], float_roundoff=float_roundoff, path=path+"/"+str(key))
    elif isinstance(left, (list, tuple)):
        test.assertIsInstance(right, type(left), path)
        test.assertEqual(len(left), len(right))
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_tree_equal(test, a, b, float_roundoff=float_roundoff, path=path+"/"+str(index))
    else:
        test.assertEqual(left, right)


class ContinuationComparisonTests(unittest.TestCase):
    """Float32 roundoff never permits absent Adam state, nonfinite values or lost steps."""

    def test_bounded_float32_roundoff_retains_state(self):
        """A small accumulated perturbation is accepted only by the explicit float mode."""
        left = {"weights": torch.ones(1024, dtype=torch.float32) * .01, "step": torch.tensor(2.)}
        right = {"weights": left["weights"].clone(), "step": torch.tensor(2.)}
        right["weights"][0] += 1e-7
        _assert_tree_equal(self, left, right, float_roundoff=True)
        with self.assertRaises(AssertionError):
            _assert_tree_equal(self, left, right)

    def test_roundoff_mode_rejects_state_loss_and_nonfinite(self):
        """Optimizer steps stay exact and missing/malformed arrays fail the same comparison."""
        for left, right in (({"step": torch.tensor(2.)}, {"step": torch.tensor(1.)}),
                            ({"moment": torch.ones(4)}, {"moment": torch.zeros(4)}),
                            ({"moment": torch.ones(4)}, {"moment": torch.full((4,), float("nan"))}),
                            ({"moment": torch.ones(4)}, {})):
            with self.assertRaises(AssertionError):
                _assert_tree_equal(self, left, right, float_roundoff=True)


@unittest.skipUnless(os.environ.get("GIFT_BASELINE_TEST_ROOT"), "real CPU model tests require an explicit new output directory")
class RealModelContinuationTests(unittest.TestCase):
    """Opt-in native CPU continuation checks; outputs and fixtures are retained in the chosen new directory."""
    @classmethod
    def setUpClass(cls):
        """Prepare synthetic inputs and a CPU child environment for native-model subprocesses."""
        cls.root = Path(os.environ["GIFT_BASELINE_TEST_ROOT"]).resolve()
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.data = cls.root / "inputs"
        cls.data.mkdir()
        values = np.random.default_rng(23).normal(size=(2, 80, 64, 64)).astype(np.float32)
        with h5py.File(cls.data / "synthetic.h5", "x") as handle:
            handle.create_dataset("training/vorticity", data=values)
            handle.create_dataset("training/trajectory_index", data=np.arange(len(values), dtype=np.int64))
            handle.create_dataset("training/time", data=np.arange(values.shape[1], dtype=np.float64) * .02)
            handle.attrs["purpose"] = "NONFORMAL synthetic CPU adapter/continuation test"
        cls.env = dict(os.environ, GIFT_DATA_ROOT=str(cls.data), CUDA_VISIBLE_DEVICES="-1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
        cls.env.pop("NVIDIA_TF32_OVERRIDE", None)

    def _check_model(self, model):
        """Compare continuous versus paused/resumed model, optimizer, normalizers, RNG and numeric history."""
        root = Path(__file__).resolve().parents[1]
        commands = [("continuous", []), ("continued", ["--stop-after-epoch", "1"]),
                    ("continued", ["--resume"])]
        for index, (name, extra) in enumerate(commands):
            argv = [sys.executable, "-B", "-m", f"training.train_{model}", "--run-training",
                    "--tiny", "--tiny-trajectories", "1", "--tiny-batch", "1", "--tiny-epochs", "2",
                    "--device", "cpu", "--data-file", "synthetic.h5", "--output",
                    str(self.root / f"{model}_{name}"), *extra]
            result = subprocess.run(argv, env=self.env, cwd=root, capture_output=True, text=True, timeout=180)
            (self.root / f"{model}_{index}_stdout.txt").write_text(result.stdout, encoding="utf-8")
            (self.root / f"{model}_{index}_stderr.txt").write_text(result.stderr, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
        def latest(name):
            """Load the selected run's last journal boundary through its JSON pointer."""
            folder = self.root / f"{model}_{name}"
            pointer = json.loads((folder / "LATEST.json").read_text())
            return torch.load(folder / pointer["file"], map_location="cpu", weights_only=True)
        left, right = latest("continuous"), latest("continued")
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "normalization"):
            _assert_tree_equal(self, left["payload"][key], right["payload"][key],
                               float_roundoff=model == "unet" and key in ("model_state_dict", "optimizer_state_dict"), path=key)
        _assert_tree_equal(self, left["rng"], right["rng"])
        self.assertEqual(left["payload"]["optimizer_updates"], 2)
        self.assertFalse(left["payload"]["formal"])
        for name, resumed in (("continuous", False), ("continued", True)):
            exported = torch.load(self.root / f"{model}_{name}" / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(exported["schema"], "gift.independent-baseline-weights.v1")
            self.assertEqual(exported["artifact_role"], "test_only")
            self.assertEqual(exported["same_run_resume_used"], resumed)
            self.assertNotIn("optimizer_state_dict", exported)
            own = right if resumed else left
            _assert_tree_equal(self, exported["model_state_dict"], own["payload"]["model_state_dict"])
        # Compare the sampled windows, updates, learning rates and losses exactly;
        # only measured wall-clock duration is excluded.
        self.assertEqual(len(left["payload"]["history"]), len(right["payload"]["history"]))
        for first, second in zip(left["payload"]["history"], right["payload"]["history"]):
            _assert_tree_equal(self, {k: v for k, v in first.items() if k != "epoch_seconds"},
                               {k: v for k, v in second.items() if k != "epoch_seconds"})
        tolerance = "bounded dtype-scaled model/Adam roundoff; other state exact" if model == "unet" else "all compared state exact"
        print(f"{model}: CPU continuous vs resumed: {tolerance}", flush=True)

    def test_uno_native_cpu_continuation(self):
        """Exercise the native U-NO CPU controller with a two-epoch interrupted run."""
        self._check_model("uno")

    def test_unet_cpu_continuation(self):
        """Exercise U-Net CPU continuation and compare its full saved state."""
        self._check_model("unet")

    def test_fno2d_cpu_continuation(self):
        """Exercise native FNO-2D CPU continuation through the common baseline controller."""
        self._check_model("fno2d")

    def test_fno3d_cpu_continuation(self):
        """Exercise native FNO-3D continuation, including saved position/time normalization."""
        self._check_model("fno3d")


if __name__ == "__main__":
    unittest.main()
