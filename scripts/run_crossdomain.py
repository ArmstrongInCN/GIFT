"""Train a cross-domain generator or evaluate saved weights against persistence.

Evaluation starts at t=5 and advances to t=8. Independent evaluation uses
--model for the weights and --output for a new result directory.
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
from scripts.generate_data import safe_output, write_json_new  # noqa: E402

DT = 0.02
ANCHOR = 250
STEPS = 150
SAVE_EVERY = 10


def resolve_device(name):
    """Resolve auto to an available backend; reject an explicitly unavailable CUDA request."""
    available = torch.cuda.is_available()
    if name == "cuda" and not available:
        raise ValueError("cuda was requested but is not available")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if available else "cpu")


def train_stage(dataset, output, device):
    """Train the diagnostic generator into its own subdirectory using the declared budget."""
    from training.gift_prediction_control import PredictionTrainingConfig, run_generator

    config = PredictionTrainingConfig(fixture_only=True)
    report = run_generator(dataset, output / "generator", device=str(device),
                           config=config, execution="auto")
    return report


def relative_l2(prediction, truth):
    """Return the spatial L2 error in float64, with a small floor for a zero truth norm."""
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    denominator = np.sqrt(np.square(truth.astype(np.float64)).sum())
    return float(np.sqrt(np.square(difference).sum()) / max(denominator, 1e-30))


def validate_evaluation_data(dataset, grid):
    """Require matching test axes and the observed t=5 through t=8 report frames.

    The dt=0.02 schedule needs indices 250 through 400, hence at least 401
    frames. Validation happens before creating outputs or starting an all-stage
    training request that would later be unevaluable.
    """
    with h5py.File(dataset, "r") as handle:
        field = handle["test/vorticity"]
        ids = handle["test/trajectory_index"]
        times = handle["test/time"]
        if (field.ndim != 4 or field.shape[0] != len(ids)
                or field.shape[0] == 0 or field.shape[2:] != (grid, grid)):
            raise ValueError("Test fields must match the population and requested grid")
        if field.shape[1] < ANCHOR + STEPS + 1:
            raise ValueError("Evaluation requires at least 401 frames (t=0 through t=8)")
        positions = np.arange(ANCHOR, ANCHOR + STEPS + 1, SAVE_EVERY)
        if len(times) != field.shape[1] or not np.allclose(
                times[positions], positions * DT, rtol=0, atol=2e-12):
            raise ValueError("Evaluation times must use the dt=0.02 observation grid")


def evaluate_stage(dataset, output, device, grid, model_path=None):
    """Roll out the saved generator from t=5 and compare with observed truth and persistence.

    Only every tenth frame is read for reporting. The model receives the
    initial field; future observations are used for scoring. Model output is
    [report_time,trajectory,N,N], while HDF truth is trajectory-major.
    """
    from gift.identified import FixedBandwidthGridGenerator, load_generator, rollout_rk4

    validate_evaluation_data(dataset, grid)
    model_path = Path(model_path) if model_path is not None else output / "generator" / "model.pt"
    model = load_generator(model_path, device=device)
    runner = FixedBandwidthGridGenerator(model, grid, reference_grid=grid)
    with h5py.File(dataset, "r") as handle:
        truth = handle["test/vorticity"][:, ANCHOR:ANCHOR + STEPS + 1:SAVE_EVERY]
        ids = np.asarray(handle["test/trajectory_index"][:], dtype=np.int64)
    if not np.isfinite(truth).all():
        raise ValueError("Nonfinite test observations")
    initial = truth[:, 0]
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
    report = {"dataset": str(dataset.resolve()), "model": str(model_path.resolve()),
              "anchor_frame": ANCHOR,
              "time_step": DT, "steps": STEPS, "save_every": SAVE_EVERY,
              "final_mean_relative_l2": summary[-1]["model_mean_relative_l2"],
              "final_persistence_mean_relative_l2": summary[-1]["persistence_mean_relative_l2"],
              "per_lead": summary, "runtime": runtime,
              "scope": ("diagnostic fixture evaluation of the native GIFT generator; "
                        "not part of the formal published experiments")}
    write_json_new(output / "summary" / "summary.json", report)
    return report


def main(argv=None):
    """Plan or execute independent training/evaluation with an explicit model for evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("heat", "advect", "reactdiff"))
    parser.add_argument("--data", type=Path, required=True, help="fixture dataset from generate_crossdomain_data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="saved generator for --stage evaluate")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--execute", action="store_true",
                        help="run the stage; the default is a read-only plan")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error("--data must point to an existing fixture dataset")
    if args.stage == "evaluate" and (args.model is None or not args.model.is_file()):
        parser.error("--stage evaluate requires --model pointing to saved generator weights")
    if args.stage != "evaluate" and args.model is not None:
        parser.error("--model is only used by --stage evaluate")
    plan = {"kind": args.kind, "stage": args.stage, "data": str(args.data),
            "model": str(args.model) if args.model else None,
            "output": str(args.output), "device": args.device, "grid": args.grid,
            "budget": "generator 500 trajectory epochs (fixture configuration)"}
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("plan only; add --execute to train and/or evaluate")
        return

    device = resolve_device(args.device)
    if args.stage in ("evaluate", "all"):
        validate_evaluation_data(args.data, args.grid)
    output = safe_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.stage in ("train", "all"):
        train_stage(args.data, output, device)
    if args.stage in ("evaluate", "all"):
        report = evaluate_stage(args.data, output, device, args.grid, args.model)
        print(json.dumps({"kind": args.kind,
                          "final_mean_relative_l2": report["final_mean_relative_l2"],
                          "model": report["model"]}, indent=2))


if __name__ == "__main__":
    main()
