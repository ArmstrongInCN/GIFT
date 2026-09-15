"""Run S3: recompute the principal GIFT results for three training seeds."""

from __future__ import annotations

from dataclasses import replace

import argparse
import copy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.formal._shared.common import (
    ProjectPaths,
    SEEDS,
    default_project_root,
    file_record,
    finish_output,
    metric_arrays,
    parse_seed_model_specs,
    seed_summary,
    sha256_file,
    write_csv_new,
    write_json_new,
    write_npz_new,
)
from experiments.formal._shared.prediction_data import load_cross_resolution, load_n64_long, evaluate_prediction_rhs
from experiments.formal._shared.gift_regimes import add_lite_arguments, resolve_regimes
from gift.data_splits import canonical_ids
from experiments.formal._shared.gift_runtime import load_gift_models, rollout_gift
from experiments.formal._shared.resume import start_experiment
from experiments.formal.train_gift_branches import (
    VALIDATION_ROLLOUT_BATCH_SIZE,
    _validate_rollout,
)


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
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--equation-batch-size", type=int, default=16)
    return parser.parse_args()


def _validation_anchor_target(
    paths: ProjectPaths,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths.require("standard_n64")
    with h5py.File(paths.standard_n64, "r") as handle:
        ids = np.asarray(handle["validation/trajectory_index"][:], dtype=np.int64)
        if handle.attrs.get("trajectory_id_scheme") != "canonical":
            ids = np.asarray(canonical_ids(ids), dtype=np.int64)
        times = np.asarray(handle["validation/time"][:], dtype=np.float64)
        anchor_columns = np.asarray((0, 100, 200, 300, 400), dtype=np.int64)
        target_columns = anchor_columns + 50
        columns = np.column_stack((anchor_columns, target_columns)).reshape(-1)
        selected = np.asarray(
            handle["validation/vorticity"][:, columns], dtype=np.float32
        )
    if not np.array_equal(ids, np.arange(1000, 1020, dtype=np.int64)):
        raise ValueError("S3 validation IDs differ from 1000..1019")
    if not np.allclose(times, np.arange(501) * 0.02, rtol=0.0, atol=2e-12):
        raise ValueError("S3 validation time support differs")
    if selected.shape != (20, 10, 64, 64) or not np.isfinite(selected).all():
        raise ValueError("S3 validation anchor/target fields differ")
    truth = selected.reshape(20, 5, 2, 64, 64).reshape(100, 2, 64, 64)
    source_ids = np.repeat(ids, 5)
    anchor_times = np.tile(times[anchor_columns], len(ids))
    rollout_ids = source_ids * 10 + np.tile(np.arange(5, dtype=np.int64), len(ids))
    if len(np.unique(rollout_ids)) != 100:
        raise AssertionError("S3 validation rollout identifiers are not unique")
    return rollout_ids, source_ids, anchor_times, truth


def _compact_rollout_audit(
    result: object, trajectory_ids: np.ndarray
) -> dict[str, object]:
    """Retain scalar and trajectory-level recursion evidence without step matrices."""

    correction = copy.deepcopy(result.correction)
    sources = [correction, *correction.get("components", {}).values()]
    for source in sources:
        source.pop("trigger_count_by_step_trajectory", None)
        source.pop("gate_failed_by_step_trajectory", None)
    failed = np.asarray(result.failure_step) >= 0
    return {
        "finite_count_by_time": result.finite_by_time.sum(axis=0)
        .astype(int)
        .tolist(),
        "failure_step": result.failure_step.tolist(),
        "failure_reason": list(result.failure_reason),
        "failure_trajectory_ids": np.asarray(trajectory_ids)[failed]
        .astype(int)
        .tolist(),
        "correction": correction,
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root)
    if args.low_model is not None:
        paths = replace(paths, low_model=args.low_model.expanduser().resolve(strict=True))
    paths, models = resolve_regimes(args, paths)[args.gift_regime]
    output = args.output or paths.root / "reproduced_results" / "S3_seed_stability"
    session = start_experiment(args, paths, "S3", output, models)
    output = session.output
    (output / "raw").mkdir()
    (output / "summary").mkdir()
    device = torch.device(args.device)

    _, validation_ids, validation_anchor_times, validation_truth = (
        _validation_anchor_target(paths)
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_truth)),
        batch_size=VALIDATION_ROLLOUT_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    test_ids, long_times, long_truth = load_n64_long(paths)
    cross = load_cross_resolution(paths)
    raw: dict[str, np.ndarray] = {
        "validation_trajectory_ids": validation_ids,
        "validation_anchor_times": validation_anchor_times,
        "test_trajectory_ids": test_ids,
        "long_absolute_times": long_times,
    }
    runtime: dict[str, object] = {}
    validation_rows = []
    training_rows = []

    for seed in SEEDS:
        low64, branch64, payload = load_gift_models(paths, models[seed], 64, device)
        if int(payload.get("seed", -1)) != seed or int(
            payload.get("training_seed", -1)
        ) != seed:
            raise ValueError(f"S3 model seed metadata differs for {seed}")
        training_rows.append(
            {
                "seed": seed,
                "training_regime": args.gift_regime,
                "checkpoint_sha256": sha256_file(models[seed]),
                "checkpoint_rule": "terminal_epoch" if args.gift_regime == "GIFT" else "validation_selected",
                "selection_phase": payload["phase"],
                "selection_epoch": payload["epoch"],
                "selection_metric": payload["selection_metric"],
                "selection_data_resolution": payload["selection_data_resolution"],
                "state_scale": payload["state_scale"],
                "high_scale": payload["high_scale"],
                "low_rhs_scale": payload["low_rhs_scale"],
                "high_rhs_scale": payload["high_rhs_scale"],
                "fixed_upper_output_cutoff": payload["fixed_upper_output_cutoff"],
            }
        )
        validation_metric = session.call(f"validation_seed_{seed}", _validate_rollout,
            branch64,
            low64,
            validation_loader,
            device,
            return_values=True,
        )
        validation_full = np.asarray(
            validation_metric["full_relative_values"], dtype=np.float64
        )
        validation_q21 = np.asarray(
            validation_metric["q21_relative_values"], dtype=np.float64
        )
        raw[f"seed_{seed}_validation_t6_full_relative_l2"] = validation_full
        raw[f"seed_{seed}_validation_t6_q21_relative_l2"] = validation_q21
        validation_rows.append(
            {
                "seed": seed,
                "full_relative_l2": float(validation_metric["lead_1_full_relative_l2"]),
                "q21_relative_l2": float(
                    validation_metric["lead_1_q21_relative_l2"]
                ),
                "population": int(validation_metric["population"]),
                "finite_count": int(validation_metric["finite_count"]),
            }
        )

        long_result = session.call(f"N64_gift_{seed}", rollout_gift,
            execution=args.gift_execution,
            low=low64,
            branch=branch64,
            initial_state=long_truth[:, 0],
            trajectory_ids=test_ids,
            duration=3.0,
            save_interval=0.5,
            device=device,
            batch_size=args.rollout_batch_size,
            correction=True,
        )
        long_metric = metric_arrays(long_result.prediction, long_truth)["full_relative_l2"]
        raw[f"seed_{seed}_N64_t6_full_relative_l2"] = long_metric[:, 2]
        raw[f"seed_{seed}_N64_t8_full_relative_l2"] = long_metric[:, -1]
        raw[f"seed_{seed}_N64_finite_by_time"] = long_result.finite_by_time
        recursive_prediction_audit = {
            "N64": _compact_rollout_audit(long_result, test_ids),
            "cross_resolution": {},
        }

        for grid in (96, 128):
            grid_ids, _, field = cross[grid]
            truth = field[:, (9, 19)]
            low, branch, _ = load_gift_models(paths, models[seed], grid, device)
            result = session.call(f"N{grid}_gift_{seed}", rollout_gift,
                execution=args.gift_execution,
                low=low,
                branch=branch,
                initial_state=truth[:, 0],
                trajectory_ids=grid_ids,
                duration=1.0,
                save_interval=1.0,
                device=device,
                batch_size=args.rollout_batch_size,
                correction=True,
            )
            grid_metric = metric_arrays(result.prediction, truth)[
                "full_relative_l2"
            ][:, -1]
            raw[f"seed_{seed}_N{grid}_t6_full_relative_l2"] = grid_metric
            raw[f"seed_{seed}_N{grid}_finite_by_time"] = result.finite_by_time
            recursive_prediction_audit["cross_resolution"][f"N{grid}"] = (
                _compact_rollout_audit(result, grid_ids)
            )

        equation_arrays, equation_report = session.call(f"equation_seed_{seed}", evaluate_prediction_rhs,
            paths,
            models[seed],
            device=device,
            batch_size=args.equation_batch_size,
            include_actions=False,
        )
        raw[f"seed_{seed}_equation_full_relative_l2"] = equation_arrays[
            "enabled_vs_reference_full"
        ]
        selected_metadata = {
            key: payload[key]
            for key in ("phase", "epoch", "selection_metric")
            if key in payload
        }
        runtime[str(seed)] = {
            "validation_finite_count": int(validation_metric["finite_count"]),
            "N64_finite_count_by_time": long_result.finite_by_time.sum(axis=0).astype(int).tolist(),
            "selected_model_metadata": selected_metadata or {"status": "not_embedded"},
            "equation_population": equation_report["population"],
            "recursive_prediction": recursive_prediction_audit,
        }

    write_npz_new(output / "raw" / "metric_arrays.npz", raw)
    write_csv_new(
        output / "raw" / "validation_metrics.csv", validation_rows
    )
    write_csv_new(output / "raw" / "training_metadata.csv", training_rows)
    quantities = {
        "validation_tau1": "validation_t6_full_relative_l2",
        "N64_t6": "N64_t6_full_relative_l2",
        "N96_t6": "N96_t6_full_relative_l2",
        "N128_t6": "N128_t6_full_relative_l2",
        "N64_t8": "N64_t8_full_relative_l2",
        "equation_full_action": "equation_full_relative_l2",
    }
    rows = []
    report_summary = {}
    for quantity, suffix in quantities.items():
        means = {
            seed: float(np.mean(raw[f"seed_{seed}_{suffix}"])) for seed in SEEDS
        }
        aggregate = seed_summary(means)
        report_summary[quantity] = aggregate
        for seed in SEEDS:
            values = np.asarray(raw[f"seed_{seed}_{suffix}"], dtype=np.float64)
            rows.append(
                {
                    "experiment": "S3",
                    "training_regime": args.gift_regime,
                    "quantity": quantity,
                    "seed": seed,
                    "within_seed_population": int(values.size),
                    "within_seed_finite_count": int(np.isfinite(values).sum()),
                    "within_seed_mean": means[seed],
                    "between_seed_mean": aggregate["mean"],
                    "between_seed_sample_sd": aggregate["sample_sd"],
                    "coefficient_of_variation_percent": aggregate[
                        "coefficient_of_variation_percent"
                    ],
                }
            )
    write_csv_new(output / "summary" / "metrics.csv", rows)
    write_json_new(
        output / "summary" / "summary.json",
        {
            "experiment_id": "S3",
            "seeds": list(SEEDS),
            "test_trajectory_ids": [int(test_ids[0]), int(test_ids[-1])],
            "training_regime": args.gift_regime,
            "key_metrics": report_summary,
        },
    )

    report = {
        "schema": "gift.formal.S3.v3",
        "status": "complete",
        "experiment": "S3",
        "training_regime": args.gift_regime,
        "title": "random-seed stability",
        "inputs": {
            "raw_N64": file_record(paths.standard_n64, paths.root),
            "dense_N64_equation_data": file_record(paths.dense_n64, paths.root),
            "raw_cross_resolution": file_record(paths.cross_resolution, paths.root),
            "low_frequency_model": file_record(paths.low_model, paths.root),
            "independently_trained_gift_models": {
                str(seed): file_record(models[seed], paths.root) for seed in SEEDS
            },
        },
        "summary": report_summary,
        "runtime_audit": runtime,
        "validation_rollout_protocol": {
            "weights_selected_by_this_evaluation": False,
            "full_data_training_selection": "terminal_epoch_not_validation",
            "trajectory_ids": "1000..1019",
            "anchors": [0.0, 2.0, 4.0, 6.0, 8.0],
            "lead_time": 1.0,
            "batch_size": VALIDATION_ROLLOUT_BATCH_SIZE,
            "metric_implementation": "train_gift_branches._validate_rollout",
        },
        "scope": {
            "changed_randomness": "high-frequency branch initialization, sample order, and optimisation path",
            "shared_low_frequency_model": True,
            "training_repetitions": 3,
            "population_uncertainty_interval_claimed": False,
        },
        "recursive_prediction_protocol": {
            "local_correction": "independent post-step P21/Q21 detection, gating, and component projection",
            "enabled_by_default": True,
        },
        "scientific_boundary": {
            "test_disjoint_from_training_and_validation": True,
        },
    }
    session.finish()
    finish_output(output, report)


if __name__ == "__main__":
    main()
