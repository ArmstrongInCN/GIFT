"""Run S1: isolate the contribution of GIFT's high-frequency branch."""

from __future__ import annotations

from dataclasses import replace

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch

from experiments.formal._shared.common import (
    ProjectPaths,
    SEEDS,
    default_project_root,
    file_record,
    finish_output,
    metric_arrays,
    parse_seed_model_specs,
    relative_l2,
    seed_summary,
    spectral_masks,
    summarize,
    write_csv_new,
    write_json_new,
    write_npz_new,
)
from experiments.formal._shared.gift_runtime import load_gift_models, rollout_gift
from experiments.formal._shared.resume import start_experiment
from experiments.formal._shared.prediction_data import load_cross_resolution, load_n64_long
from experiments.formal._shared.gift_regimes import add_lite_arguments, resolve_regimes
from gift.data_splits import canonical_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="reuse completed calls from this same --output run")
    parser.add_argument("--gift-model", action="append", default=[])
    add_lite_arguments(parser)
    parser.add_argument("--gift-execution", choices=("eager", "cuda-graph"), default="eager")
    parser.add_argument("--gift-regime", choices=("GIFT", "GIFT-Lite"), default="GIFT")
    parser.add_argument("--low-model", type=Path, help="explicit frozen P21 prerequisite; default retains published model path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--derivative-batch-size", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def derivative_validation(
    paths: ProjectPaths,
    models: dict[int, Path],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Recompute the fourth-order validation derivative comparison from raw HDF5."""

    paths.require("standard_n64", "low_model")
    loaded = {
        seed: load_gift_models(paths, models[seed], 64, device)
        for seed in SEEDS
    }
    low = loaded[SEEDS[0]][0]
    masks = spectral_masks(64)
    derivative_bands: dict[str, np.ndarray | None] = {
        "full": None,
        "P21": masks["P21"],
        "Q21": masks["Q21"],
        "P22_31": masks["P22_31"],
        "outer_P31": masks["outer_P31"],
    }
    result: dict[str, list[np.ndarray]] = {
        "trajectory_id": [],
        "centre_index": [],
        "absolute_time": [],
    }
    for band in derivative_bands:
        result[f"disabled_{band}_relative_l2"] = []
        for seed in SEEDS:
            result[f"seed_{seed}_{band}_relative_l2"] = []
    with h5py.File(paths.standard_n64, "r") as handle:
        ids = np.asarray(handle["validation/trajectory_index"][:], dtype=np.int64)
        times = np.asarray(handle["validation/time"][:], dtype=np.float64)
        field = handle["validation/vorticity"]
        if handle.attrs.get("trajectory_id_scheme") != "canonical":
            ids = np.asarray(canonical_ids(ids), dtype=np.int64)
        if not np.array_equal(ids, np.arange(1000, 1020, dtype=np.int64)):
            raise ValueError("S1 validation IDs differ from 1000..1019")
        if field.shape != (20, 501, 64, 64):
            raise ValueError("S1 validation field shape differs")
        dt = float(np.mean(np.diff(times)))
        if not np.isclose(dt, 0.02, atol=1e-12, rtol=0.0):
            raise ValueError("S1 derivative time step differs from 0.02")
        for row, trajectory_id in enumerate(ids):
            raw = np.asarray(field[row], dtype=np.float32)
            state = raw[2:-2]
            truth = (
                raw[:-4] - 8.0 * raw[1:-3] + 8.0 * raw[3:-1] - raw[4:]
            ) / (12.0 * dt)
            result["trajectory_id"].append(
                np.full(len(state), trajectory_id, dtype=np.int64)
            )
            result["centre_index"].append(np.arange(2, 499, dtype=np.int16))
            result["absolute_time"].append(times[2:-2])
            for start in range(0, len(state), batch_size):
                stop = min(start + batch_size, len(state))
                state_tensor = torch.from_numpy(state[start:stop]).to(device)
                frozen = low(state_tensor)
                disabled = frozen.detach().cpu().numpy()
                target = truth[start:stop]
                for band, mask in derivative_bands.items():
                    result[f"disabled_{band}_relative_l2"].append(
                        relative_l2(disabled, target, mask)
                    )
                for seed in SEEDS:
                    branch = loaded[seed][1]
                    learned = branch.forward_with_generator_output(
                        state_tensor, frozen
                    )
                    enabled = disabled + learned.detach().cpu().numpy()
                    for band, mask in derivative_bands.items():
                        result[f"seed_{seed}_{band}_relative_l2"].append(
                            relative_l2(enabled, target, mask)
                        )
    arrays = {name: np.concatenate(parts) for name, parts in result.items()}
    expected = 20 * 497
    if any(len(value) != expected for value in arrays.values()):
        raise RuntimeError("S1 derivative raw array length differs")
    return arrays


def _endpoint_arrays(
    prediction: np.ndarray, truth: np.ndarray, index: int
) -> dict[str, np.ndarray]:
    values = metric_arrays(prediction[:, index : index + 1], truth[:, index : index + 1])
    result = {name: array[:, 0] for name, array in values.items()}
    masks = spectral_masks(truth.shape[-1])
    if bool(masks["reliable_outer_P31"].any()):
        result["reliable_outer_P31_dynamics_relative_l2"] = relative_l2(
            prediction[:, index] - prediction[:, 0],
            truth[:, index] - truth[:, 0],
            masks["reliable_outer_P31"],
        )
    return result


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root)
    if args.low_model is not None:
        paths = replace(paths, low_model=args.low_model.expanduser().resolve(strict=True))
    paths, models = resolve_regimes(args, paths)[args.gift_regime]
    output = args.output or paths.root / "reproduced_results" / "S1_high_frequency_branch"
    session = start_experiment(args, paths, "S1", output, models)
    output = session.output
    (output / "raw").mkdir()
    (output / "summary").mkdir()
    device = torch.device(args.device)

    derivative = session.call("validation_derivative", derivative_validation,
        paths, models, device, args.derivative_batch_size
    )
    raw: dict[str, np.ndarray] = {
        f"derivative_{name}": value for name, value in derivative.items()
    }
    ids, long_times, long_truth = load_n64_long(paths)
    cross = load_cross_resolution(paths)
    if not np.array_equal(long_truth[:, :3], cross[64][2][:, (9, 14, 19)]):
        raise RuntimeError("S1 N64 raw sources disagree at t=5.0,5.5,6.0")
    raw["test_trajectory_ids"] = ids
    raw["long_absolute_times"] = long_times

    reports: dict[str, dict[str, object]] = {}
    endpoint: dict[str, dict[str, np.ndarray]] = {}
    first_seed = SEEDS[0]
    for seed in SEEDS:
        low64, branch64, _ = load_gift_models(paths, models[seed], 64, device)
        enabled = session.call(f"N64_gift_{seed}", rollout_gift,
            execution=args.gift_execution,
            low=low64,
            branch=branch64,
            initial_state=long_truth[:, 0],
            trajectory_ids=ids,
            duration=3.0,
            save_interval=0.5,
            device=device,
            batch_size=args.rollout_batch_size,
            correction=True,
        )
        arm = f"seed_{seed}_enabled"
        endpoint[f"{arm}_N64_t6"] = _endpoint_arrays(enabled.prediction, long_truth, 2)
        endpoint[f"{arm}_N64_t8"] = _endpoint_arrays(enabled.prediction, long_truth, 6)
        raw[f"{arm}_N64_finite_by_time"] = enabled.finite_by_time
        reports[arm] = {
            "failure_step": enabled.failure_step.tolist(),
            "failure_reason": list(enabled.failure_reason),
            "correction": enabled.correction,
            "branch_audit": enabled.branch_audit,
        }
        if seed == first_seed:
            disabled = session.call("N64_disabled", rollout_gift,
                execution=args.gift_execution,
                low=low64,
                branch=None,
                initial_state=long_truth[:, 0],
                trajectory_ids=ids,
                duration=3.0,
                save_interval=0.5,
                device=device,
                batch_size=args.rollout_batch_size,
                correction=True,
            )
            endpoint["disabled_N64_t6"] = _endpoint_arrays(
                disabled.prediction, long_truth, 2
            )
            endpoint["disabled_N64_t8"] = _endpoint_arrays(
                disabled.prediction, long_truth, 6
            )
            raw["disabled_N64_finite_by_time"] = disabled.finite_by_time
            reports["disabled"] = {
                "failure_step": disabled.failure_step.tolist(),
                "failure_reason": list(disabled.failure_reason),
                "correction": disabled.correction,
                "branch_audit": disabled.branch_audit,
            }

        for grid in (96, 128):
            grid_ids, times20, field = cross[grid]
            truth = field[:, 9:20]
            low, branch, _ = load_gift_models(paths, models[seed], grid, device)
            enabled_grid = session.call(f"N{grid}_gift_{seed}", rollout_gift,
                execution=args.gift_execution,
                low=low,
                branch=branch,
                initial_state=truth[:, 0],
                trajectory_ids=grid_ids,
                duration=1.0,
                save_interval=0.1,
                device=device,
                batch_size=args.rollout_batch_size,
                correction=True,
            )
            grid_arm = f"seed_{seed}_enabled_N{grid}_t6"
            endpoint[grid_arm] = _endpoint_arrays(enabled_grid.prediction, truth, 10)
            raw[f"seed_{seed}_enabled_N{grid}_finite_by_time"] = enabled_grid.finite_by_time
            reports[f"seed_{seed}_enabled_N{grid}"] = {
                "failure_step": enabled_grid.failure_step.tolist(),
                "failure_reason": list(enabled_grid.failure_reason),
                "correction": enabled_grid.correction,
                "branch_audit": enabled_grid.branch_audit,
            }
            if seed == first_seed:
                disabled_grid = session.call(f"N{grid}_disabled", rollout_gift,
                    execution=args.gift_execution,
                    low=low,
                    branch=None,
                    initial_state=truth[:, 0],
                    trajectory_ids=grid_ids,
                    duration=1.0,
                    save_interval=0.1,
                    device=device,
                    batch_size=args.rollout_batch_size,
                    correction=True,
                )
                endpoint[f"disabled_N{grid}_t6"] = _endpoint_arrays(
                    disabled_grid.prediction, truth, 10
                )
                raw[f"disabled_N{grid}_finite_by_time"] = disabled_grid.finite_by_time
                reports[f"disabled_N{grid}"] = {
                    "failure_step": disabled_grid.failure_step.tolist(),
                    "failure_reason": list(disabled_grid.failure_reason),
                    "correction": disabled_grid.correction,
                    "branch_audit": disabled_grid.branch_audit,
                }

    for arm, values in endpoint.items():
        for name, value in values.items():
            raw[f"{arm}_{name}"] = value
    write_npz_new(output / "raw" / "metric_arrays.npz", raw)

    rows = []
    summary_report: dict[str, object] = {}
    tasks = {
        "validation_derivative": "derivative_{}full_relative_l2",
        "N64_t6": "{}_N64_t6_full_relative_l2",
        "N96_t6": "{}_N96_t6_full_relative_l2",
        "N128_t6": "{}_N128_t6_full_relative_l2",
        "N64_t8": "{}_N64_t8_full_relative_l2",
    }
    for task, template in tasks.items():
        task_cohorts = (
            {"validation_1000_1019": slice(None)}
            if task == "validation_derivative"
            else {
                "test_1040_1219": slice(None),
            }
        )
        summary_report[task] = {}
        for cohort, selected in task_cohorts.items():
            enabled_means: dict[int, float] = {}
            for seed in SEEDS:
                prefix = (
                    f"seed_{seed}_"
                    if task == "validation_derivative"
                    else f"seed_{seed}_enabled"
                )
                key = template.format(prefix)
                values = raw[key][selected]
                enabled_means[seed] = float(np.mean(values))
                rows.append(
                    {
                        "experiment": "S1",
                        "cohort": cohort,
                        "task": task,
                        "band": "full",
                        "method": args.gift_regime,
                        "seed": seed,
                        "mean_relative_l2": enabled_means[seed],
                        "finite_count": int(np.isfinite(values).sum()),
                        "population": int(values.size),
                    }
                )
            disabled_key = template.format(
                "disabled_" if task == "validation_derivative" else "disabled"
            )
            disabled_values = raw[disabled_key][selected]
            disabled_mean = float(np.mean(disabled_values))
            rows.append(
                {
                    "experiment": "S1",
                    "cohort": cohort,
                    "task": task,
                    "band": "full",
                    "method": f"{args.gift_regime} (high-frequency branch disabled)",
                    "seed": "fixed",
                    "mean_relative_l2": disabled_mean,
                    "finite_count": int(np.isfinite(disabled_values).sum()),
                    "population": int(disabled_values.size),
                }
            )
            summary_report[task][cohort] = {
                args.gift_regime: seed_summary(enabled_means),
                "disabled": summarize(disabled_values),
                "relative_reduction_percent": float(
                    100.0
                    * (
                        1.0
                        - np.mean(list(enabled_means.values())) / disabled_mean
                    )
                ),
            }

    comparison_keys: list[tuple[str, str, str, str]] = []
    for metric_name, band in (
        ("P21", "P21"),
        ("Q21", "Q21"),
        ("P22_31", "P22-31"),
        ("outer_P31", "outside P31"),
    ):
        comparison_keys.append(
            (
                "validation_derivative_frequency_band",
                band,
                f"derivative_seed_{{seed}}_{metric_name}_relative_l2",
                f"derivative_disabled_{metric_name}_relative_l2",
            )
        )
    for resolution in (64, 96, 128):
        metrics_for_grid = [
            ("P21", "P21_relative_l2"),
            ("Q21", "Q21_relative_l2"),
            ("P22-31", "P22_31_relative_l2"),
            ("outside P31", "outer_P31_relative_l2"),
        ]
        if resolution > 64:
            metrics_for_grid.extend(
                [
                    (
                        "reliable outside P31 state",
                        "reliable_outer_P31_relative_l2",
                    ),
                    (
                        "reliable outside P31 dynamics",
                        "reliable_outer_P31_dynamics_relative_l2",
                    ),
                ]
            )
        for band, metric_name in metrics_for_grid:
            comparison_keys.append(
                (
                    f"N{resolution}_t6_frequency_band",
                    band,
                    f"seed_{{seed}}_enabled_N{resolution}_t6_{metric_name}",
                    f"disabled_N{resolution}_t6_{metric_name}",
                )
            )

    for task, band, enabled_template, disabled_key in comparison_keys:
        task_cohorts = (
            {"validation_1000_1019": slice(None)}
            if task.startswith("validation_derivative")
            else {
                "test_1040_1219": slice(None),
            }
        )
        summary_key = f"{task}_{band}"
        summary_report[summary_key] = {}
        for cohort, selected in task_cohorts.items():
            enabled_means: dict[int, float] = {}
            for seed in SEEDS:
                values = raw[enabled_template.format(seed=seed)][selected]
                enabled_means[seed] = float(np.mean(values))
                rows.append(
                    {
                        "experiment": "S1",
                        "cohort": cohort,
                        "task": task,
                        "band": band,
                        "method": args.gift_regime,
                        "seed": seed,
                        "mean_relative_l2": enabled_means[seed],
                        "finite_count": int(np.isfinite(values).sum()),
                        "population": int(values.size),
                    }
                )
            disabled_values = raw[disabled_key][selected]
            disabled_mean = float(np.mean(disabled_values))
            rows.append(
                {
                    "experiment": "S1",
                    "cohort": cohort,
                    "task": task,
                    "band": band,
                    "method": f"{args.gift_regime} (high-frequency branch disabled)",
                    "seed": "fixed",
                    "mean_relative_l2": disabled_mean,
                    "finite_count": int(np.isfinite(disabled_values).sum()),
                    "population": int(disabled_values.size),
                }
            )
            summary_report[summary_key][cohort] = {
                args.gift_regime: seed_summary(enabled_means),
                "disabled": summarize(disabled_values),
                "relative_reduction_percent": (
                    float(
                        100.0
                        * (
                            1.0
                            - np.mean(list(enabled_means.values())) / disabled_mean
                        )
                    )
                    if disabled_mean != 0.0
                    else None
                ),
            }
    write_csv_new(output / "summary" / "metrics.csv", rows)
    write_json_new(
        output / "summary" / "summary.json",
        {
            "experiment_id": "S1",
            "populations": {
                "test_1040_1219": len(ids),
                "validation_1000_1019_derivative_states": 9940,
            },
            "key_metrics": summary_report,
        },
    )

    report = {
        "schema": "gift.formal.S1.v3",
        "status": "complete",
        "experiment": "S1",
        "training_regime": args.gift_regime,
        "title": "high-frequency branch effectiveness",
        "inputs": {
            "raw_N64": file_record(paths.standard_n64, paths.root),
            "raw_N64_dense": file_record(paths.dense_n64, paths.root),
            "raw_cross_resolution": file_record(paths.cross_resolution, paths.root),
            "low_frequency_model": file_record(paths.low_model, paths.root),
            "gift_models": {str(seed): file_record(models[seed], paths.root) for seed in SEEDS},
        },
        "summary": summary_report,
        "runtime_audit": reports,
        "protocol": {
            "only_changed_factor": "high-frequency branch output enabled versus zero",
            "recursive_local_correction": "independent post-step P21/Q21 correction enabled in both arms",
            "validation_derivative": "fourth-order centred observed difference on validation IDs 1000-1019",
            "prediction_population": "independent test IDs 1040-1219",
            "disabled_high_state": "Q21 right-hand side is zero; the same post-step Q21 local correction remains enabled",
        },
        "scientific_boundary": {
            "test_disjoint_from_training_and_validation": True,
            "N64_outside_P31_is_Nyquist_diagnostic": True,
            "cross_resolution_primary_outer_band_excludes_Nyquist_lines": True,
        },
    }
    session.finish()
    finish_output(output, report)


if __name__ == "__main__":
    main()
