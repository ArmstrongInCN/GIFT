"""Run S2: compare GIFT recursion with and without local correction."""

from __future__ import annotations

from dataclasses import replace

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

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
    load_correction_holdout,
    load_n64_long,
    metric_arrays,
    parse_seed_model_specs,
    seed_summary,
    summarize,
    write_csv_new,
    write_json_new,
)
from experiments.formal._shared.gift_runtime import (
    DEFAULT_P21_CORRECTION_POLICY,
    DEFAULT_Q21_CORRECTION_POLICY,
    load_gift_models,
    rollout_gift,
)
from experiments.formal._shared.resume import start_experiment


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
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def _report_without_step_matrices(
    value: dict[str, Any],
    *,
    steps: int,
    trajectory_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Move large per-step matrices from the JSON audit into the raw HDF5 file."""

    report = copy.deepcopy(value)
    population = len(trajectory_ids)
    component_reports = report.get("components")
    if not isinstance(component_reports, dict):
        raise TypeError("correction report lacks component audits")
    sources = {
        "aggregate": report,
        "P21": component_reports.get("P21"),
        "Q21": component_reports.get("Q21"),
    }
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for scope, source in sources.items():
        if not isinstance(source, dict):
            raise TypeError(f"correction report lacks the {scope} audit")
        raw_trigger = source.pop("trigger_count_by_step_trajectory", None)
        raw_gate = source.pop("gate_failed_by_step_trajectory", None)
        if raw_trigger is None or raw_gate is None:
            raise KeyError(f"correction report lacks the {scope} per-step audit")
        trigger = np.asarray(raw_trigger, dtype=np.int64)
        gate = np.asarray(raw_gate, dtype=np.bool_)
        expected_shape = (steps, population)
        if trigger.shape != expected_shape or gate.shape != expected_shape:
            raise ValueError(
                f"{scope} correction matrix shape differs: expected "
                f"{expected_shape}, found trigger={trigger.shape}, gate={gate.shape}"
            )
        if np.any(trigger < 0):
            raise ValueError(f"{scope} correction trigger counts must be nonnegative")

        triggered = np.any(trigger > 0, axis=0)
        gated = np.any(gate, axis=0)
        expected_summary = {
            "trigger_pixel_events": int(trigger.sum()),
            "triggered_step_count": int(np.any(trigger > 0, axis=1).sum()),
            "triggered_trajectory_count": int(triggered.sum()),
            "triggered_trajectory_ids": trajectory_ids[triggered].astype(int).tolist(),
            "gate_failed_trajectory_count": int(gated.sum()),
            "gate_failed_trajectory_ids": trajectory_ids[gated].astype(int).tolist(),
        }
        for key, expected in expected_summary.items():
            if source.get(key) != expected:
                raise RuntimeError(f"{scope} correction report field differs: {key}")
        matrices[scope] = {"trigger": trigger, "gate": gate}

    if not np.array_equal(
        matrices["aggregate"]["trigger"],
        matrices["P21"]["trigger"] + matrices["Q21"]["trigger"],
    ):
        raise RuntimeError("aggregate correction triggers do not equal P21 plus Q21")
    if not np.array_equal(
        matrices["aggregate"]["gate"],
        matrices["P21"]["gate"] | matrices["Q21"]["gate"],
    ):
        raise RuntimeError("aggregate correction gates do not equal P21 OR Q21")
    return report, matrices


def _raw_dataset_name(scope: str, quantity: str) -> str:
    prefix = "correction" if scope == "aggregate" else f"correction_{scope}"
    return f"{prefix}_{quantity}"


def _correction_summary(value: dict[str, Any]) -> dict[str, Any]:
    components = value["components"]
    fields = (
        "trigger_pixel_events",
        "triggered_step_count",
        "triggered_trajectory_count",
        "triggered_trajectory_ids",
        "gate_failed_trajectory_count",
        "gate_failed_trajectory_ids",
    )
    sources = {
        "aggregate": value,
        "P21": components["P21"],
        "Q21": components["Q21"],
    }
    return {
        scope: {key: source[key] for key in fields} for scope, source in sources.items()
    }


def _summary_row(
    *,
    cohort: str,
    subject: str,
    arm: str,
    seed: int,
    absolute_time: float,
    values: np.ndarray,
) -> dict[str, Any]:
    stats = summarize(values)
    return {
        "experiment": "S2",
        "cohort": cohort,
        "subject": subject,
        "method": "GIFT"
        if arm == "corrected"
        else "GIFT (recursive local correction disabled)",
        "seed": seed,
        "absolute_time": float(absolute_time),
        "lead_time": float(absolute_time - 5.0),
        **stats,
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root)
    if args.low_model is not None:
        paths = replace(paths, low_model=args.low_model.expanduser().resolve(strict=True))
    models = parse_seed_model_specs(paths.root, args.gift_model)
    output = (
        args.output
        or paths.root / "reproduced_results" / "S2_recursive_local_correction"
    )
    session = start_experiment(args, paths, "S2", output, models)
    output = session.output
    (output / "raw").mkdir()
    (output / "summary").mkdir()
    device = torch.device(args.device)

    stress_ids, times, stress_truth = load_n64_long(paths)
    holdout_ids, holdout_times, holdout_truth = load_correction_holdout(paths)
    if not np.array_equal(times, holdout_times):
        raise RuntimeError("S2 cohorts use different report times")
    cohorts = {
        "benchmark": (stress_ids, stress_truth),
        "independent_holdout": (holdout_ids, holdout_truth),
    }
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    all_metrics: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}

    with h5py.File(output / "raw" / "predictions.h5", "x") as handle:
        handle.attrs["schema"] = "gift.formal.S2.raw.v3"
        handle.create_dataset("absolute_times", data=times)
        for cohort_name, (ids, truth) in cohorts.items():
            cohort_group = handle.create_group(cohort_name)
            cohort_group.create_dataset("trajectory_ids", data=ids)
            cohort_group.create_dataset(
                "truth",
                data=truth,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            for seed in SEEDS:
                low, branch, _ = load_gift_models(paths, models[seed], 64, device)
                seed_group = cohort_group.create_group(f"seed_{seed}")
                audit_key = f"{cohort_name}_seed_{seed}"
                audit[audit_key] = {}
                for arm, correction in (("corrected", True), ("uncorrected", False)):
                    result = session.call(f"{cohort_name}_gift_{seed}_{arm}", rollout_gift,
                        low=low,
                        branch=branch,
                        initial_state=truth[:, 0],
                        trajectory_ids=ids,
                        duration=3.0,
                        save_interval=0.5,
                        device=device,
                        batch_size=args.batch_size,
                        correction=correction,
                    )
                    metrics = metric_arrays(result.prediction, truth)
                    all_metrics[(cohort_name, seed, arm)] = metrics
                    arm_group = seed_group.create_group(arm)
                    arm_group.create_dataset(
                        "prediction",
                        data=result.prediction,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
                    for name, value in metrics.items():
                        arm_group.create_dataset(name, data=value)
                    arm_group.create_dataset(
                        "finite_by_time", data=result.finite_by_time
                    )
                    arm_group.create_dataset("failure_step", data=result.failure_step)
                    string_type = h5py.string_dtype(encoding="utf-8")
                    arm_group.create_dataset(
                        "failure_reason",
                        data=np.asarray(
                            [value or "" for value in result.failure_reason],
                            dtype=object,
                        ),
                        dtype=string_type,
                    )
                    correction_report, correction_matrices = (
                        _report_without_step_matrices(
                            result.correction,
                            steps=150,
                            trajectory_ids=ids,
                        )
                    )
                    for scope, matrices in correction_matrices.items():
                        trigger_matrix = matrices["trigger"]
                        gate_matrix = matrices["gate"]
                        trigger_count = trigger_matrix.sum(axis=0, dtype=np.int64)
                        trigger_step_count = np.count_nonzero(
                            trigger_matrix > 0, axis=0
                        ).astype(np.int64)
                        gate_failed = np.any(gate_matrix, axis=0)
                        arm_group.create_dataset(
                            _raw_dataset_name(
                                scope, "trigger_count_by_step_trajectory"
                            ),
                            data=trigger_matrix,
                            compression="gzip",
                            compression_opts=4,
                        )
                        arm_group.create_dataset(
                            _raw_dataset_name(scope, "trigger_count_by_trajectory"),
                            data=trigger_count,
                        )
                        arm_group.create_dataset(
                            _raw_dataset_name(scope, "step_count_by_trajectory"),
                            data=trigger_step_count,
                        )
                        arm_group.create_dataset(
                            _raw_dataset_name(scope, "gate_failed_by_step_trajectory"),
                            data=gate_matrix,
                            compression="gzip",
                            compression_opts=4,
                        )
                        arm_group.create_dataset(
                            _raw_dataset_name(scope, "gate_failed_by_trajectory"),
                            data=gate_failed,
                        )
                    aggregate_trigger = correction_matrices["aggregate"]["trigger"]
                    if not np.array_equal(
                        aggregate_trigger.sum(axis=0),
                        result.correction_trigger_count,
                    ):
                        raise RuntimeError(
                            "rollout aggregate correction trigger totals differ"
                        )
                    if not np.array_equal(
                        np.count_nonzero(aggregate_trigger > 0, axis=0),
                        result.correction_step_count,
                    ):
                        raise RuntimeError(
                            "rollout aggregate correction step totals differ"
                        )
                    failed = np.flatnonzero(result.failure_step >= 0)
                    first_failed = (
                        int(failed[np.argmin(result.failure_step[failed])])
                        if len(failed)
                        else None
                    )
                    audit[audit_key][arm] = {
                        "finite_count_by_time": result.finite_by_time.sum(axis=0)
                        .astype(int)
                        .tolist(),
                        "failure_trajectory_ids": ids[failed].astype(int).tolist(),
                        "failure_step": result.failure_step.tolist(),
                        "failure_reason": list(result.failure_reason),
                        "first_failure": (
                            {
                                "trajectory_id": int(ids[first_failed]),
                                "step": int(result.failure_step[first_failed]),
                                "absolute_time": float(
                                    5.0 + 0.02 * result.failure_step[first_failed]
                                ),
                                "reason": result.failure_reason[first_failed],
                            }
                            if first_failed is not None
                            else None
                        ),
                        "correction": correction_report,
                        "runtime": result.runtime,
                        "branch_audit": result.branch_audit,
                    }
                    for time_index, absolute_time in enumerate(times):
                        values = metrics["full_relative_l2"][:, time_index]
                        rows.append(
                            _summary_row(
                                cohort=cohort_name,
                                subject="all trajectories",
                                arm=arm,
                                seed=seed,
                                absolute_time=float(absolute_time),
                                values=values,
                            )
                        )
    write_csv_new(output / "summary" / "metrics.csv", rows)

    summary_report: dict[str, Any] = {}
    for cohort_name in cohorts:
        corrected_terminal = {
            seed: float(
                summarize(
                    all_metrics[(cohort_name, seed, "corrected")]["full_relative_l2"][
                        :, -1
                    ]
                )["mean"]
            )
            for seed in SEEDS
        }
        uncorrected_terminal = {
            seed: float(
                summarize(
                    all_metrics[(cohort_name, seed, "uncorrected")]["full_relative_l2"][
                        :, -1
                    ]
                )["mean"]
            )
            for seed in SEEDS
        }
        summary_report[cohort_name] = {
            "t8": {
                "corrected": seed_summary(corrected_terminal),
                "uncorrected_finite_values": seed_summary(uncorrected_terminal),
            },
            "finite_population_required_for_direct_error_ranking": True,
            "correction_audit_by_seed_and_arm": {
                str(seed): {
                    arm: _correction_summary(
                        audit[f"{cohort_name}_seed_{seed}"][arm]["correction"]
                    )
                    for arm in ("corrected", "uncorrected")
                }
                for seed in SEEDS
            },
            "finite_count_by_seed_and_arm": {
                str(seed): {
                    arm: audit[f"{cohort_name}_seed_{seed}"][arm][
                        "finite_count_by_time"
                    ]
                    for arm in ("corrected", "uncorrected")
                }
                for seed in SEEDS
            },
            "failure_trajectory_ids_by_seed_and_arm": {
                str(seed): {
                    arm: audit[f"{cohort_name}_seed_{seed}"][arm][
                        "failure_trajectory_ids"
                    ]
                    for arm in ("corrected", "uncorrected")
                }
                for seed in SEEDS
            },
        }

    write_json_new(
        output / "summary" / "summary.json",
        {
            "schema": "gift.formal.S2.summary.v3",
            "experiment_id": "S2",
            "cohorts": {
                "benchmark": {"trajectory_ids": [1000, 1199], "count": 200},
                "independent_holdout": {
                    "trajectory_ids": [1200, 1399],
                    "count": 200,
                },
            },
            "key_metrics": summary_report,
        },
    )

    report = {
        "schema": "gift.formal.S2.v3",
        "status": "complete",
        "experiment": "S2",
        "title": "recursive local correction effectiveness",
        "inputs": {
            "benchmark_raw_data": file_record(paths.standard_n64, paths.root),
            "independent_holdout_raw_data": file_record(paths.fno_training, paths.root),
            "low_frequency_model": file_record(paths.low_model, paths.root),
            "gift_models": {
                str(seed): file_record(models[seed], paths.root) for seed in SEEDS
            },
        },
        "summary": summary_report,
        "failure_and_trigger_audit": audit,
        "protocol": {
            "only_changed_factor": (
                "post-RK4 independent P21 and Q21 recursive local correction: "
                "both components enabled versus both components disabled"
            ),
            "high_frequency_branch": "enabled in both arms",
            "rk4_step": 0.02,
            "report_times": times.tolist(),
            "correction_thresholds_fixed_before_test": True,
            "correction_components": {
                "P21": {
                    "score_threshold": DEFAULT_P21_CORRECTION_POLICY.score_threshold,
                    "amplitude_threshold": (
                        DEFAULT_P21_CORRECTION_POLICY.amplitude_threshold
                    ),
                    "periodic_window": (
                        2 * DEFAULT_P21_CORRECTION_POLICY.window_radius + 1
                    ),
                    "maximum_flagged_fraction": (
                        DEFAULT_P21_CORRECTION_POLICY.maximum_flagged_fraction
                    ),
                },
                "Q21": {
                    "score_threshold": DEFAULT_Q21_CORRECTION_POLICY.score_threshold,
                    "amplitude_threshold": (
                        DEFAULT_Q21_CORRECTION_POLICY.amplitude_threshold
                    ),
                    "periodic_window": (
                        2 * DEFAULT_Q21_CORRECTION_POLICY.window_radius + 1
                    ),
                    "maximum_flagged_fraction": (
                        DEFAULT_Q21_CORRECTION_POLICY.maximum_flagged_fraction
                    ),
                },
            },
            "component_gates_are_independent": True,
        },
        "scientific_boundary": {
            "failure_avoidance_conclusion_is_data_dependent": True,
            "no_failure_trajectory_or_step_is_preselected": True,
            "benchmark_architecture_development_trajectory_ids": [1000, 1019],
            "independent_holdout_trajectory_ids": [1200, 1399],
        },
    }
    session.finish()
    finish_output(output, report)


if __name__ == "__main__":
    main()
