"""Train and evaluate the native GIFT generator on a cross-domain fixture dataset.

Two stages, both using the project's own code without modification:

``train``     calls ``training.gift_prediction_control.run_generator`` with the
              project's ``PredictionTrainingConfig(fixture_only=True)``
              configuration. Fixture mode is the project's declared path for
              non-canonical diagnostic data: it relaxes the formal population and
              identity checks and marks every artifact as ``fixture_only``, so a
              diagnostic run can never be mistaken for a formal result. The
              generator, budget, optimizer, scheduler and sampling code are the
              unchanged full-data implementation (500 trajectory epochs).

``evaluate``  loads the exported ``model.pt`` with ``gift.identified.load_generator``
              and performs recursive RK4 rollouts with
              ``gift.identified.rollout_rk4``: anchor frame 250 (t = 5.0), 150
              steps of dt = 0.02 through t = 8.0, one reported frame every 10
              steps. Metrics are the mean relative L2 over the fixture test
              trajectories, reported next to a persistence baseline (the anchor
              frame held constant).

Reference values measured on the verification machine (RTX 5060 Laptop, torch
2.10.0+cu128) at lead 3.0 s, mean relative L2 over 20 held-out trajectories:

  heat       0.0239   (persistence 0.4211)
  advect     0.0245   (persistence 1.4020)
  reactdiff  0.0051   (persistence 0.0064)

Cross-platform floating-point differences can move these numbers slightly; the
reported figures are measurements, not guarantees. See EXPERIMENTS.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scripts.generate_data import safe_output, sha256, write_json_new  # noqa: E402

# Rollout settings shared by every fixture kind: anchor at frame 250 (t=5.0),
# 150 steps of dt 0.02 (to t=8.0), one reported frame every 10 steps. REFERENCE
# holds the measured final mean relative L2 on the verification machine.
DT = 0.02
ANCHOR = 250
STEPS = 150
SAVE_EVERY = 10
REFERENCE = {"heat": 0.0239, "advect": 0.0245, "reactdiff": 0.0051}


def resolve_device(name):
    available = torch.cuda.is_available()
    if name == "cuda" and not available:
        raise ValueError("cuda was requested but is not available")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if available else "cpu")


def train_stage(dataset, output, device):
    from training.gift_prediction_control import PredictionTrainingConfig, run_generator

    # fixture_only relaxes the formal population/identity checks and marks every
    # artifact fixture_only, so a diagnostic run can never be read as formal output.
    config = PredictionTrainingConfig(fixture_only=True)
    report = run_generator(dataset, output / "generator", device=str(device),
                           config=config, execution="auto")
    return report


def relative_l2(prediction, truth):
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    denominator = np.sqrt(np.square(truth.astype(np.float64)).sum())
    return float(np.sqrt(np.square(difference).sum()) / max(denominator, 1e-30))


def evaluate_stage(dataset, output, device, grid):
    from gift.identified import FixedBandwidthGridGenerator, load_generator, rollout_rk4

    # Load the exported generator, roll it out from the anchor frame with the
    # project's own RK4 integrator, and score against a held-constant persistence
    # baseline so the fixture transferability is measured relatively.

    model_path = output / "generator" / "model.pt"
    model = load_generator(model_path, device=device)
    runner = FixedBandwidthGridGenerator(model, grid, reference_grid=grid)
    with h5py.File(dataset, "r") as handle:
        truth_all = handle["test/vorticity"][:]
        ids = np.asarray(handle["test/trajectory_index"][:], dtype=np.int64)
    initial = truth_all[:, ANCHOR]
    truth = truth_all[:, ANCHOR:ANCHOR + STEPS + 1:SAVE_EVERY]
    with torch.inference_mode():
        states = torch.from_numpy(initial).to(device)
        rolled = rollout_rk4(runner, states, dt=DT, steps=STEPS, save_every=SAVE_EVERY)
    rolled = rolled.cpu().numpy()

    leads = [step * DT for step in range(0, STEPS + 1, SAVE_EVERY)]
    rows, summary = [], []
    for index, lead in enumerate(leads):
        model_error = float(np.mean([relative_l2(rolled[index, row], truth[row, index])
                                     for row in range(len(ids))]))
        persistence_error = float(np.mean([relative_l2(initial[row], truth[row, index])
                                           for row in range(len(ids))]))
        summary.append({"lead_time": lead, "model_mean_relative_l2": model_error,
                        "persistence_mean_relative_l2": persistence_error,
                        "population": int(len(ids))})
        rows.append((lead, model_error, persistence_error))

    metrics = output / "summary" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("x", encoding="utf-8", newline="") as stream:
        stream.write("method,metric,lead_time,lead_steps,population,mean_relative_l2\n")
        for lead, model_error, persistence_error in rows:
            steps = int(round(lead / DT))
            stream.write(f"GIFT,mean_relative_l2,{lead:.6f},{steps},{len(ids)},{model_error:.17g}\n")
            stream.write(f"persistence,mean_relative_l2,{lead:.6f},{steps},{len(ids)},{persistence_error:.17g}\n")
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "numpy": np.__version__, "h5py": h5py.__version__, "torch": torch.__version__,
               "device": str(device), "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    report = {"dataset": dataset.name, "dataset_sha256": sha256(dataset),
              "model_sha256": sha256(model_path), "anchor_frame": ANCHOR,
              "time_step": DT, "steps": STEPS, "save_every": SAVE_EVERY,
              "final_mean_relative_l2": summary[-1]["model_mean_relative_l2"],
              "final_persistence_mean_relative_l2": summary[-1]["persistence_mean_relative_l2"],
              "per_lead": summary, "runtime": runtime,
              "reference_final": REFERENCE.get(dataset.stem, None),
              "scope": ("diagnostic fixture evaluation of the native GIFT generator; "
                        "not part of the formal published experiments")}
    write_json_new(output / "summary" / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("heat", "advect", "reactdiff"))
    parser.add_argument("--data", type=Path, required=True, help="fixture dataset from generate_crossdomain_data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--execute", action="store_true",
                        help="run the stage; the default is a read-only plan")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error("--data must point to an existing fixture dataset")
    plan = {"kind": args.kind, "stage": args.stage, "data": str(args.data),
            "output": str(args.output), "device": args.device, "grid": args.grid,
            "budget": "generator 500 trajectory epochs (fixture configuration)",
            "reference_final_mean_relative_l2": REFERENCE[args.kind]}
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("plan only; add --execute to train and/or evaluate")
        return

    device = resolve_device(args.device)
    output = safe_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.stage in ("train", "all"):
        train_stage(args.data, output, device)
    if args.stage in ("evaluate", "all"):
        report = evaluate_stage(args.data, output, device, args.grid)
        print(json.dumps({"kind": args.kind,
                          "final_mean_relative_l2": report["final_mean_relative_l2"],
                          "reference_final": report["reference_final"],
                          "model_sha256": report["model_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
