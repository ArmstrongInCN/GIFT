"""Run M2: five-method N64 autonomous prediction from t=5.0 to t=8.0."""

from __future__ import annotations

from dataclasses import replace

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from adapters.prediction import load_unet, unet_rollout, load_uno, uno_rollout  # noqa: E402
from experiments.formal._shared.common import (  # noqa: E402
    SEEDS,
    ProjectPaths,
    default_project_root,
    file_record,
    finish_output,
    load_fno_test_dt0p02,
    load_n64_long,
    metric_arrays,
    parse_seed_model_specs,
    seed_summary,
    summarize,
    time_metric_rows,
    write_csv_new,
    write_json_new,
)
from experiments.formal._shared.fno_runtime import (  # noqa: E402
    M2_FUTURE_INDICES,
    load_fno_models,
    predict_fno2d,
    predict_fno3d,
)
from experiments.formal._shared.gift_runtime import (  # noqa: E402
    load_gift_models,
    rollout_gift,
)
from experiments.formal._shared.resume import start_experiment  # noqa: E402


COHORT = "trajectories_1000_1199"
REPORT_TIMES = np.asarray([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
METHOD_CONFIGS = (
    ("GIFT", "20260820", "GIFT_seed_20260820"),
    ("GIFT", "20260821", "GIFT_seed_20260821"),
    ("GIFT", "20260822", "GIFT_seed_20260822"),
    ("FNO-2D", "fixed", "FNO_2D"),
    ("FNO-3D", "fixed", "FNO_3D"),
    ("U-NO", "0", "U_NO"),
    ("U-Net", "0", "U_Net"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="reuse completed calls from this same --output run")
    parser.add_argument("--gift-model", action="append", default=[])
    parser.add_argument("--low-model", type=Path, help="explicit frozen P21 prerequisite; default retains published model path")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gift-batch-size", type=int, default=16)
    parser.add_argument("--fno2d-batch-size", type=int, default=10)
    parser.add_argument("--fno3d-batch-size", type=int, default=5)
    parser.add_argument("--uno-batch-size", type=int, default=4)
    parser.add_argument("--unet-batch-size", type=int, default=4)
    parser.add_argument("--uno-model", type=Path)
    parser.add_argument("--unet-model", type=Path)
    parser.add_argument("--skip-plots", action="store_true", help="save numerical evidence; render figures independently later")
    return parser.parse_args()


def resolve_input(root: Path, value: Path | None, default: str) -> Path:
    from gift.paths import checkpoint_root
    path = checkpoint_root(root) / Path(default).relative_to("artifacts") if value is None else value
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def baseline_prediction(
    *,
    context: np.ndarray,
    device: torch.device,
    batch_size: int,
    predictor: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if batch_size < 1:
        raise ValueError("baseline batch size must be positive")
    report = np.empty((len(context), 7, 64, 64), dtype=np.float32)
    finite = np.zeros((len(context), 150), dtype=bool)
    report[:, 0] = context[:, -1]
    with torch.no_grad():
        for first in range(0, len(context), batch_size):
            last = min(first + batch_size, len(context))
            history = torch.from_numpy(context[first:last]).to(device)
            future = predictor(history)
            if future.shape != (last - first, 150, 64, 64):
                raise RuntimeError(
                    f"baseline prediction shape differs: {tuple(future.shape)}"
                )
            finite[first:last] = (
                torch.isfinite(future).flatten(2).all(dim=2).cpu().numpy()
            )
            report[first:last, 1:] = future[:, M2_FUTURE_INDICES.tolist()].cpu().numpy()
    first_nonfinite = np.full(len(context), -1, dtype=np.int32)
    for index in range(len(context)):
        failures = np.flatnonzero(~finite[index])
        if failures.size:
            first_nonfinite[index] = int(failures[0])
    return report, finite, first_nonfinite


def adaptation_record() -> dict[str, list[str]]:
    return {
        "FNO-2D": [
            "history length changed from 10 to 46 frames",
            "the one-step architecture remained four Fourier blocks with 12 by 12 modes and width 20",
            "training and evaluation use a 150-step closed-loop rollout",
        ],
        "FNO-3D": [
            "history length changed from 10 to 46 frames and output length to 150 frames",
            "the original non-recursive space-time prediction design was retained",
            "the formal model uses 8 by 8 by 8 Fourier modes",
        ],
        "U-NO": [
            "history length changed from 10 to 46 frames while retaining the four periodic coordinate channels",
            "training rollout shortened from 40 to 20 steps for device memory; evaluation remains 150 steps",
            "training duration fixed to 150 epochs and activation checkpointing changes memory use only",
        ],
        "U-Net": [
            "input and output channels changed to 46 and 1; the published encoder-decoder layers remain unchanged",
            "training uses a four-step closed-loop objective and evaluation uses 150 recursive steps",
            "training duration fixed to 500 epochs",
        ],
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root)
    if args.low_model is not None:
        paths = replace(paths, low_model=args.low_model.expanduser().resolve(strict=True))
    gift_models = parse_seed_model_specs(paths.root, args.gift_model)
    output = (
        args.output or paths.root / "reproduced_results" / "M2_recursive_prediction"
    )
    device = torch.device(args.device)

    uno_path = resolve_input(
        paths.root, args.uno_model, "artifacts/formal/uno/uno_terminal_epoch150.pt"
    )
    unet_path = resolve_input(
        paths.root,
        args.unet_model,
        "artifacts/formal/unet/unet_terminal_epoch500.pt",
    )
    session = start_experiment(args, paths, "M2", output, gift_models,
                               extra_files={"uno_model": uno_path, "unet_model": unet_path})
    output = session.output
    (output / "raw").mkdir()
    (output / "summary").mkdir()
    ids, times, truth = load_n64_long(paths)
    fine_ids, _, context, fine_times, fine_truth = load_fno_test_dt0p02(
        paths, 64, times
    )
    if not np.array_equal(ids, fine_ids) or not np.array_equal(times, fine_times):
        raise RuntimeError("M2 sparse and fine-time trajectory axes differ")
    if not np.array_equal(fine_truth, truth):
        raise RuntimeError("M2 fine-time truth differs from stored report truth")
    if not np.array_equal(context[:, -1], truth[:, 0]):
        raise RuntimeError("M2 history and report truth do not share the t=5 anchor")
    if not np.allclose(times, REPORT_TIMES, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("M2 report times differ")

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, np.ndarray]] = {}
    full_finite: dict[str, np.ndarray] = {}
    first_nonfinite: dict[str, np.ndarray] = {}
    gift_runtime: dict[str, Any] = {}
    for seed in SEEDS:
        low, branch, _ = load_gift_models(paths, gift_models[seed], 64, device)
        result = session.call(f"gift_{seed}", rollout_gift,
            low=low,
            branch=branch,
            initial_state=truth[:, 0],
            trajectory_ids=ids,
            duration=3.0,
            save_interval=0.5,
            device=device,
            batch_size=args.gift_batch_size,
            correction=True,
        )
        group_name = f"GIFT_seed_{seed}"
        predictions[group_name] = result.prediction
        metrics[group_name] = metric_arrays(result.prediction, truth)
        failures = np.asarray(result.failure_step, dtype=np.int32)
        if not np.all(failures == -1):
            raise RuntimeError(f"GIFT seed {seed} produced a failed M2 rollout")
        full_finite[group_name] = np.ones((len(ids), 150), dtype=bool)
        first_nonfinite[group_name] = failures
        gift_runtime[str(seed)] = {
            "finite_count_by_time": result.finite_by_time.sum(axis=0)
            .astype(int)
            .tolist(),
            "failure_step": failures.tolist(),
            "failure_reason": list(result.failure_reason),
            "correction": result.correction,
            "runtime": result.runtime,
        }

    fno2d, fno3d, _, payload3 = load_fno_models(paths, device)
    predictions["FNO_2D"], fno2d_audit = session.call("fno2d", predict_fno2d,
        fno2d, context, M2_FUTURE_INDICES, device, args.fno2d_batch_size
    )
    predictions["FNO_3D"], fno3d_audit, _ = session.call("fno3d", predict_fno3d,
        paths,
        fno3d,
        payload3,
        context,
        M2_FUTURE_INDICES,
        device,
        args.fno3d_batch_size,
    )
    for method, group_name, audit in (
        ("FNO-2D", "FNO_2D", fno2d_audit),
        ("FNO-3D", "FNO_3D", fno3d_audit),
    ):
        counts = audit["finite_trajectory_count_by_future_step"]
        if counts != [len(ids)] * 150:
            raise RuntimeError(f"{method} contains a non-finite future field")
        metrics[group_name] = metric_arrays(predictions[group_name], truth)
        full_finite[group_name] = np.ones((len(ids), 150), dtype=bool)
        first_nonfinite[group_name] = np.full(len(ids), -1, dtype=np.int32)
    del fno2d, fno3d
    if device.type == "cuda":
        torch.cuda.empty_cache()

    uno_model, _ = load_uno(uno_path, device)
    predictions["U_NO"], full_finite["U_NO"], first_nonfinite["U_NO"] = (
        session.call("uno", baseline_prediction,
            context=context,
            device=device,
            batch_size=args.uno_batch_size,
            predictor=lambda history, model=uno_model: uno_rollout(model, history, 150),
        )
    )
    metrics["U_NO"] = metric_arrays(predictions["U_NO"], truth)
    del uno_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unet_model, _ = load_unet(unet_path, device)
    predictions["U_Net"], full_finite["U_Net"], first_nonfinite["U_Net"] = (
        session.call("unet", baseline_prediction,
            context=context,
            device=device,
            batch_size=args.unet_batch_size,
            predictor=lambda history, model=unet_model: unet_rollout(
                model, history, 150
            ),
        )
    )
    metrics["U_Net"] = metric_arrays(predictions["U_Net"], truth)
    del unet_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for method, _, group_name in METHOD_CONFIGS:
        if not full_finite[group_name].all() or not np.all(
            first_nonfinite[group_name] == -1
        ):
            raise RuntimeError(f"{method} contains a non-finite future field")

    with h5py.File(output / "raw" / "predictions.h5", "x") as handle:
        handle.attrs["schema"] = "gift.formal.M2.raw.v2"
        handle.create_dataset("trajectory_ids", data=ids)
        handle.create_dataset("absolute_times", data=times)
        handle.create_dataset(
            "truth", data=truth, compression="gzip", compression_opts=4, shuffle=True
        )
        for _, _, group_name in METHOD_CONFIGS:
            group = handle.create_group(group_name)
            group.create_dataset(
                "prediction",
                data=predictions[group_name],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            for metric, value in metrics[group_name].items():
                group.create_dataset(metric, data=value)
            group.create_dataset(
                "finite_by_trajectory_future_step",
                data=full_finite[group_name],
                compression="gzip",
                compression_opts=4,
            )
            group.create_dataset(
                "first_nonfinite_future_step_zero_based",
                data=first_nonfinite[group_name],
            )

    metric_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for method, seed, group_name in METHOD_CONFIGS:
        metric_rows.extend(
            {"cohort": COHORT, **row}
            for row in time_metric_rows(
                experiment="M2",
                method=method,
                seed=seed,
                times=times,
                values=metrics[group_name]["full_relative_l2"],
            )
        )
        for trajectory_index, trajectory_id in enumerate(ids):
            for time_index, absolute_time in enumerate(times):
                raw_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "trajectory_id": int(trajectory_id),
                        "absolute_time": float(absolute_time),
                        "full_relative_l2": float(
                            metrics[group_name]["full_relative_l2"][
                                trajectory_index, time_index
                            ]
                        ),
                        "field_finite": True,
                        "all_150_future_fields_finite": True,
                        "first_nonfinite_future_step_zero_based": -1,
                    }
                )
    write_csv_new(output / "summary" / "metrics.csv", metric_rows)
    write_csv_new(output / "raw" / "per_trajectory_metrics.csv", raw_rows)

    key_metrics: dict[str, Any] = {
        "GIFT_t8": seed_summary(
            {
                seed: float(
                    metrics[f"GIFT_seed_{seed}"]["full_relative_l2"][:, -1].mean()
                )
                for seed in SEEDS
            }
        )
    }
    for method, _, group_name in METHOD_CONFIGS:
        if method != "GIFT":
            key_metrics[f"{method}_t8"] = summarize(
                metrics[group_name]["full_relative_l2"][:, -1]
            )
    finite_summary = {
        method: 200
        for method in ("GIFT", "FNO-2D", "FNO-3D", "U-NO", "U-Net")
    }
    write_json_new(
        output / "summary" / "summary.json",
        {
            "experiment_id": "M2",
            "populations": {COHORT: 200},
            "key_metrics": {COHORT: key_metrics},
            "complete_prediction_finite_count": finite_summary,
        },
    )

    report = {
        "schema": "gift.formal.M2.v5",
        "status": "complete",
        "experiment": "M2",
        "title": "N64 long-horizon autonomous prediction",
        "populations": {COHORT: {"trajectory_ids": [1000, 1199], "count": 200}},
        "absolute_times": times.tolist(),
        "inputs": {
            "long_truth": file_record(paths.standard_n64, paths.root),
            "fno_test_data": file_record(paths.fno_test_dt0p02, paths.root),
            "low_frequency_model": file_record(paths.low_model, paths.root),
            "gift_models": {
                str(seed): file_record(gift_models[seed], paths.root) for seed in SEEDS
            },
            "fno2d_model": file_record(paths.fno2d_model, paths.root),
            "fno3d_model": file_record(paths.fno3d_model, paths.root),
            "uno_model": file_record(uno_path, paths.root),
            "unet_model": file_record(unet_path, paths.root),
        },
        "summary": {COHORT: key_metrics},
        "gift_runtime": gift_runtime,
        "fno_inference": {"FNO-2D": fno2d_audit, "FNO-3D": fno3d_audit},
        "baseline_inference": {
            method: {
                "finite_trajectory_count_by_future_step": [200] * 150,
                "complete_prediction_finite_count": 200,
            }
            for method in ("U-NO", "U-Net")
        },
        "protocol": {
            "dt": 0.02,
            "test_trajectory_ids": [1000, 1199],
            "test_trajectory_count": 200,
            "training_data": {
                "GIFT": "50 N64 training trajectories and 20 validation trajectories, t=0.0 to 10.0",
                "baselines": "1,000 N64 training trajectories, t=0.0 to 10.0, sampled every 0.02",
            },
            "context": {
                "steps": 46,
                "absolute_times": [4.1, 5.0],
                "physical_span": 0.9,
            },
            "future": {
                "steps": 150,
                "absolute_times": [5.02, 8.0],
                "physical_span": 3.0,
            },
            "report_future_indices": M2_FUTURE_INDICES.tolist(),
            "report_future_indices_zero_based": M2_FUTURE_INDICES.tolist(),
            "GIFT": "single-anchor recursive prediction with recursive local correction; internal RK4 time step 0.02",
            "FNO-2D": "46-frame context; one-step model recursively called 150 times",
            "FNO-3D": "46-frame context; one non-recursive call producing all 150 future frames",
            "methods": {
                "GIFT": "single state at t=5.0; RK4 recursive prediction with recursive local correction",
                "FNO-2D": "46-frame context; one-step operator called recursively for 150 steps",
                "FNO-3D": "46-frame context; one non-recursive operator call producing 150 frames",
                "U-NO": "46-frame context; one-step operator called recursively for 150 steps",
                "U-Net": "46-frame context; one-step network called recursively for 150 steps",
            },
            "teacher_forcing": False,
            "teacher_forcing_during_test": False,
            "truth_restart": False,
            "test_statistics_used": False,
            "time_averaged_metric_reported": False,
        },
        "necessary_adaptations": adaptation_record(),
        "metric": "per-trajectory full-field relative L2, summarized separately at each report time",
        "raw_data": {
            "predictions": "raw/predictions.h5",
            "per_trajectory_metrics": "raw/per_trajectory_metrics.csv",
        },
        "figure": {
            "source_data": "figures/source_data.csv",
            "curve": "shape-preserving piecewise cubic Hermite interpolation through the evaluated points",
            "interpolation_used_for_metrics": False,
            "keyframes": {
                "trajectory_id": 1005,
                "absolute_times": [5.0, 6.0, 7.0, 8.0],
                "source": "raw/predictions.h5",
                "output_directory": "figures/keyframes",
            },
        },
    }

    from experiments.formal._shared.plotting import publish_numeric_then_plot

    plot_script = Path(__file__).with_name("plot_mean_error.py")
    keyframe_script = Path(__file__).with_name("plot_keyframes.py")
    commands = [
        [
            sys.executable,
            str(plot_script),
            "--metrics",
            str(output / "summary" / "metrics.csv"),
            "--output-dir",
            str(output / "figures"),
        ],
        [
            sys.executable,
            str(keyframe_script),
            "--input",
            str(output / "raw" / "predictions.h5"),
            "--output-dir",
            str(output / "figures" / "keyframes"),
            "--trajectory-id",
            "1005",
        ],
    ]
    session.finish()
    publish_numeric_then_plot(output, report, [] if args.skip_plots else commands)


if __name__ == "__main__":
    main()
