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

from training.baseline_control import PROTOCOLS, _arguments, configuration, epoch_schedule, objective, save_boundary


class ProtocolTests(unittest.TestCase):
    def test_formal_budgets(self):
        self.assertEqual([(PROTOCOLS[name]["epochs"], PROTOCOLS[name]["micro_batch"],
                           PROTOCOLS[name]["accumulation"], PROTOCOLS[name]["rollout"])
                          for name in ("fno2d", "fno3d", "uno", "unet")],
                         [(500, 10, 2, 150), (500, 5, 2, 150), (150, 16, 1, 20), (500, 20, 1, 4)])

    def test_tiny_is_distinct_budget(self):
        config = configuration("uno", _arguments("uno", ["--tiny"]))
        self.assertFalse(config["formal"])
        self.assertEqual((config["epochs"], config["micro_batch"], config["rollout"]), (2, 1, 1))

    def test_uno_schedule_is_epoch_local(self):
        config = dict(PROTOCOLS["uno"], model="uno")
        first = epoch_schedule(config, 0, 501)
        np.random.seed(891)
        second = epoch_schedule(config, 0, 501)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(len(first[0]), 1000)
        self.assertLessEqual(first[1].max(), 435)

    def test_unet_closed_loop_keeps_gradient(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.4))
            def forward(self, history):
                return self.scale * history[:, -1:]
        network = Toy()
        loss, _ = objective(network, torch.ones(1, 46, 64, 64), torch.ones(1, 2, 64, 64), "unet")
        loss.backward()
        self.assertAlmostEqual(float(network.scale.grad), -1.8, places=5)

    def test_checkpoint_interval_and_forced_boundaries(self):
        config = configuration("unet", _arguments("unet", []))
        self.assertEqual(config["checkpoint_interval"], 10)
        self.assertFalse(save_boundary(9, config, 500))
        self.assertTrue(save_boundary(10, config, 500))
        self.assertTrue(save_boundary(4, config, 4))
        self.assertTrue(save_boundary(500, config, 500))
        self.assertTrue(save_boundary(1, dict(config, formal=False), 2))


def _assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


@unittest.skipUnless(os.environ.get("GIFT_BASELINE_TEST_ROOT"), "real CPU model tests require an explicit new output directory")
class RealModelContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["GIFT_BASELINE_TEST_ROOT"]).resolve()
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.data = cls.root / "inputs"
        cls.data.mkdir()
        values = np.random.default_rng(23).normal(size=(2, 80, 64, 64)).astype(np.float32)
        with h5py.File(cls.data / "synthetic.h5", "x") as handle:
            handle.create_dataset("training/vorticity", data=values)
            handle.attrs["purpose"] = "NONFORMAL synthetic CPU adapter/continuation test"
        cls.env = dict(os.environ, GIFT_DATA_ROOT=str(cls.data), CUDA_VISIBLE_DEVICES="-1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
        cls.env.pop("NVIDIA_TF32_OVERRIDE", None)

    def _check_model(self, model):
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
            folder = self.root / f"{model}_{name}"
            pointer = json.loads((folder / "LATEST.json").read_text())
            return torch.load(folder / pointer["file"], map_location="cpu", weights_only=True)
        left, right = latest("continuous"), latest("continued")
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "normalization"):
            _assert_tree_equal(self, left["payload"][key], right["payload"][key])
        _assert_tree_equal(self, left["rng"], right["rng"])
        self.assertEqual(left["payload"]["optimizer_updates"], 2)
        self.assertFalse(left["payload"]["formal"])
        for name, resumed in (("continuous", False), ("continued", True)):
            exported = torch.load(self.root / f"{model}_{name}" / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(exported["schema"], "gift.independent-baseline-weights.v1")
            self.assertEqual(exported["artifact_role"], "test_only")
            self.assertEqual(exported["same_run_resume_used"], resumed)
            self.assertNotIn("optimizer_state_dict", exported)
            _assert_tree_equal(self, exported["model_state_dict"], left["payload"]["model_state_dict"])
        for first, second in zip(left["payload"]["history"], right["payload"]["history"]):
            _assert_tree_equal(self, {k: v for k, v in first.items() if k != "epoch_seconds"},
                               {k: v for k, v in second.items() if k != "epoch_seconds"})
        print(f"{model}: CPU continuous vs resumed model/optimizer/scheduler/RNG/history EXACT", flush=True)

    def test_uno_native_cpu_continuation(self):
        self._check_model("uno")

    def test_unet_cpu_continuation(self):
        self._check_model("unet")

    def test_fno2d_cpu_continuation(self):
        self._check_model("fno2d")

    def test_fno3d_cpu_continuation(self):
        self._check_model("fno3d")


if __name__ == "__main__":
    unittest.main()
