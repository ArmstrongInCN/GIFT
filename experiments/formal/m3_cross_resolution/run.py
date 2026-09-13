"""Run M3: N64-trained GIFT/FNO prediction on N64, N96, and N128 grids."""

from __future__ import annotations

from dataclasses import replace

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.formal._shared.common import (  # noqa: E402
    SEEDS,
    ProjectPaths,
    default_project_root,
    file_record,
    finish_output,
    load_cross_resolution,
    load_fno_test_dt0p02,
    metric_arrays,
    parse_seed_model_specs,
    seed_summary,
    summarize,
    time_metric_rows,
    write_csv_new,
    write_json_new,
)
from experiments.formal._shared.fno_runtime import (  # noqa: E402
    FNO2D_SAFE_BATCH,
    FNO3D_SAFE_BATCH,
    M3_FUTURE_INDICES,
    load_fno_models,
    predict_fno2d,
    predict_fno3d,
)
from experiments.formal._shared.gift_runtime import load_gift_models, rollout_gift  # noqa: E402
from experiments.formal._shared.resume import start_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="reuse completed calls from this same --output run")
    parser.add_argument("--gift-model", action="append", default=[])
    parser.add_argument("--low-model", type=Path, help="explicit frozen P21 prerequisite; default retains published model path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gift-batch-size", type=int, default=8)
    parser.add_argument("--fno2d-batch-size", type=int, default=10)
    parser.add_argument("--fno3d-batch-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root)
    if args.low_model is not None:
        paths = replace(paths, low_model=args.low_model.expanduser().resolve(strict=True))
    models = parse_seed_model_specs(paths.root, args.gift_model)
    output = args.output or paths.root / "reproduced_results" / "M3_cross_resolution"
    session = start_experiment(args, paths, "M3", output, models)
    output = session.output
    (output / "raw").mkdir()
    (output / "summary").mkdir()
    device = torch.device(args.device)
    raw = load_cross_resolution(paths)
    fno2d, fno3d, _, payload3 = load_fno_models(paths, device)
    all_metrics: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    runtime: dict[str, dict[str, object]] = {}
    fno_inference: dict[str, dict[str, object]] = {}
    fno3d_normalizer_audit: dict[str, dict[str, object]] = {}

    with h5py.File(output / "raw" / "predictions.h5", "x") as handle:
        handle.attrs["schema"] = "gift.formal.M3.raw.v2"
        for grid in (64, 96, 128):
            ids, times20, field = raw[grid]
            truth = field[:, 9:20]
            times = times20[9:20]
            fine_ids, _, context, fine_times, fine_truth = load_fno_test_dt0p02(
                paths, grid, times
            )
            if not np.array_equal(ids, fine_ids) or not np.allclose(
                times, fine_times, rtol=0.0, atol=2.0e-12
            ):
                raise RuntimeError(f"N{grid} coarse and fine-time axes differ")
            if not np.array_equal(truth, fine_truth):
                raise RuntimeError(
                    f"N{grid} fine-time truth differs from the stored report truth"
                )
            if not np.array_equal(context[:, -1], truth[:, 0]):
                raise RuntimeError(
                    f"N{grid} FNO context and GIFT truth do not share the t=5 anchor"
                )
            grid_group = handle.create_group(f"N{grid}")
            grid_group.create_dataset("trajectory_ids", data=ids)
            grid_group.create_dataset("absolute_times", data=times)
            grid_group.create_dataset("truth", data=truth, compression="gzip", compression_opts=4, shuffle=True)
            grid_metrics: dict[str, dict[str, np.ndarray]] = {}
            for seed in SEEDS:
                low, branch, _ = load_gift_models(paths, models[seed], grid, device)
                result = session.call(f"N{grid}_gift_{seed}", rollout_gift,
                    low=low,
                    branch=branch,
                    initial_state=truth[:, 0],
                    trajectory_ids=ids,
                    duration=1.0,
                    save_interval=0.1,
                    device=device,
                    batch_size=args.gift_batch_size,
                    correction=True,
                )
                method = f"GIFT_seed_{seed}"
                values = metric_arrays(result.prediction, truth)
                grid_metrics[method] = values
                group = grid_group.create_group(method)
                group.create_dataset("prediction", data=result.prediction, compression="gzip", compression_opts=4, shuffle=True)
                for name, value in values.items():
                    group.create_dataset(name, data=value)
                runtime[f"N{grid}_seed_{seed}"] = {
                    "finite_count_by_time": result.finite_by_time.sum(axis=0).astype(int).tolist(),
                    "failure_step": result.failure_step.tolist(),
                    "failure_reason": list(result.failure_reason),
                    "correction": result.correction,
                }
            fno2d_batch = min(args.fno2d_batch_size, FNO2D_SAFE_BATCH[grid])
            fno3d_batch = min(args.fno3d_batch_size, FNO3D_SAFE_BATCH[grid])
            prediction2, audit2 = session.call(f"N{grid}_fno2d", predict_fno2d,
                fno2d,
                context,
                M3_FUTURE_INDICES,
                device,
                fno2d_batch,
            )
            prediction3, audit3, normalizer_audit = session.call(f"N{grid}_fno3d", predict_fno3d,
                paths,
                fno3d,
                payload3,
                context,
                M3_FUTURE_INDICES,
                device,
                fno3d_batch,
            )
            fno_inference[f"N{grid}"] = {
                "FNO-2D": audit2,
                "FNO-3D": audit3,
            }
            fno3d_normalizer_audit[f"N{grid}"] = normalizer_audit
            for method, prediction in (("FNO-2D", prediction2), ("FNO-3D", prediction3)):
                values = metric_arrays(prediction, truth)
                grid_metrics[method] = values
                group = grid_group.create_group(method.replace("-", "_"))
                group.create_dataset("prediction", data=prediction, compression="gzip", compression_opts=4, shuffle=True)
                for name, value in values.items():
                    group.create_dataset(name, data=value)
            all_metrics[grid] = grid_metrics
            del context, fine_truth, prediction2, prediction3
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = []
    report_summary = {}
    for grid in (64, 96, 128):
        grid_ids, times20, _ = raw[grid]
        times = times20[9:20]
        cohorts = {
            "trajectories_1000_1199": np.ones(len(grid_ids), dtype=bool),
            "trajectories_1100_1199": grid_ids >= 1100,
        }
        for method, values in all_metrics[grid].items():
            seed: int | str = method.removeprefix("GIFT_seed_") if method.startswith("GIFT_seed_") else "fixed"
            for metric_name, metric_value in values.items():
                for cohort, selected in cohorts.items():
                    for row in time_metric_rows(
                        experiment="M3",
                        method=(
                            "GIFT" if method.startswith("GIFT_seed_") else method
                        ),
                        seed=seed,
                        times=times,
                        values=metric_value[selected],
                    ):
                        rows.append(
                            {
                                "resolution": f"N{grid}",
                                "cohort": cohort,
                                "metric": metric_name,
                                **row,
                            }
                        )
        report_summary[f"N{grid}"] = {}
        for cohort, selected in cohorts.items():
            report_summary[f"N{grid}"][cohort] = {
                "GIFT_t6": seed_summary(
                    {
                        seed: float(
                            all_metrics[grid][f"GIFT_seed_{seed}"][
                                "full_relative_l2"
                            ][selected, -1].mean()
                        )
                        for seed in SEEDS
                    }
                ),
                "FNO-2D_t6": summarize(
                    all_metrics[grid]["FNO-2D"]["full_relative_l2"][
                        selected, -1
                    ]
                ),
                "FNO-3D_t6": summarize(
                    all_metrics[grid]["FNO-3D"]["full_relative_l2"][
                        selected, -1
                    ]
                ),
            }
            if grid > 64:
                report_summary[f"N{grid}"][cohort][
                    "reliable_outer_P31_t6"
                ] = {
                    "GIFT": seed_summary(
                        {
                            seed: float(
                                all_metrics[grid][f"GIFT_seed_{seed}"][
                                    "reliable_outer_P31_relative_l2"
                                ][selected, -1].mean()
                            )
                            for seed in SEEDS
                        }
                    ),
                    "FNO-2D": summarize(
                        all_metrics[grid]["FNO-2D"][
                            "reliable_outer_P31_relative_l2"
                        ][selected, -1]
                    ),
                    "FNO-3D": summarize(
                        all_metrics[grid]["FNO-3D"][
                            "reliable_outer_P31_relative_l2"
                        ][selected, -1]
                    ),
                }
    write_csv_new(output / "summary" / "metrics.csv", rows)
    write_json_new(
        output / "summary" / "summary.json",
        {
            "experiment_id": "M3",
            "populations": {
                "trajectories_1000_1199": 200,
                "trajectories_1100_1199": 100,
            },
            "key_metrics": report_summary,
        },
    )
    report = {
        "schema": "gift.formal.M3.v4",
        "status": "complete",
        "experiment": "M3",
        "title": "cross-resolution prediction",
        "populations": {
            "trajectories_1000_1199": {
                "trajectory_ids": [1000, 1199],
                "count_per_resolution": 200,
            },
            "trajectories_1100_1199": {
                "trajectory_ids": [1100, 1199],
                "count_per_resolution": 100,
            },
        },
        "inputs": {
            "N64_raw_data": file_record(paths.dense_n64, paths.root),
            "N96_N128_raw_data": file_record(paths.cross_resolution, paths.root),
            "fno_test_data": file_record(paths.fno_test_dt0p02, paths.root),
            "low_frequency_model": file_record(paths.low_model, paths.root),
            "gift_models": {str(seed): file_record(models[seed], paths.root) for seed in SEEDS},
            "fno2d_model": file_record(paths.fno2d_model, paths.root),
            "fno3d_model": file_record(paths.fno3d_model, paths.root),
        },
        "summary": report_summary,
        "gift_runtime": runtime,
        "fno_inference": fno_inference,
        "fno3d_normalizer_audit": fno3d_normalizer_audit,
        "protocol": {
            "source_training_grid": 64,
            "target_grids": [64, 96, 128],
            "model_retraining_on_target_grids": False,
            "dt": 0.02,
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
            "report_future_indices": M3_FUTURE_INDICES.tolist(),
            "GIFT_constant_spectrum_factor": {"N64": 1.0, "N96": 2.25, "N128": 4.0},
            "GIFT_recursive_local_correction": "independent post-step P21/Q21 detection, gating, and component projection",
            "FNO_weights_and_modes_unchanged": True,
            "FNO-2D": "46-frame context; one-step model recursively called 150 times",
            "FNO-3D": "46-frame context; one non-recursive call producing all 150 future frames",
            "FNO3D_training_statistics_lift": "deterministic periodic Fourier interpolation on spatial axes only",
            "teacher_forcing": False,
            "truth_restart": False,
            "test_statistics_used": False,
        },
        "evaluation_scope": {
            "all_reported_trajectory_ids": [1000, 1199],
            "reported_subset_trajectory_ids": [1100, 1199],
            "architecture_selection_trajectory_ids": [1000, 1019],
            "high_resolution_values_used_for_weight_updates": False,
        },
    }
    session.finish()
    finish_output(output, report)


if __name__ == "__main__":
    main()
