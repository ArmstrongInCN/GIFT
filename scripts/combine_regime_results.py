"""Combine completed GIFT and GIFT-Lite S1/S3 measurements, without inference.

Each regime remains an independently resumable experiment. This command only
assembles their measured tables and binds both source reports by SHA-256.
"""

import argparse
import csv
import json
from pathlib import Path

from experiments.formal._shared.common import finish_output, write_csv_new, write_json_new
from scripts.verify_results import verify_result


def combine(experiment, full, lite, output):
    if experiment not in ("S1", "S3"):
        # Only S1 (high-frequency branch) and S3 (seed stability) run both the full
        # GIFT and GIFT-Lite regimes as independently resumable experiments.
        raise ValueError("only S1 and S3 use separate regime measurements")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("choose a new combined-results output directory")
    sources, summaries, all_rows, column_names = {}, {}, [], None
    inputs = {}
    for regime, directory in (("GIFT", full), ("GIFT-Lite", lite)):
        directory = Path(directory).resolve(strict=True)
        # Each regime is re-verified as a completed run that used the independent
        # prediction test partition before its measured table is merged.
        if directory == output or directory in output.parents or output in directory.parents:
            raise ValueError("combined output must remain separate from source runs")
        checked = verify_result(directory, experiment)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("training_regime") != regime or report.get("status") != "complete":
            raise ValueError("source run has the wrong regime or is incomplete")
        if report.get("scientific_boundary", {}).get("test_disjoint_from_training_and_validation") is not True:
            raise ValueError("source run does not use the independent prediction test partition")
        # The dense observation input differs by experiment (S1 uses the raw N64
        # dense frames, S3 the dense N64 equation data); both regimes must bind the
        # same test observations so the combined table is comparable.
        dense_key = "raw_N64_dense" if experiment == "S1" else "dense_N64_equation_data"
        common = {name: report["inputs"][name]["sha256"]
                  for name in ("raw_N64", "raw_cross_resolution", dense_key)}
        if inputs and common != inputs:
            raise ValueError("the two regimes use different test observations")
        inputs = common
        with (directory / "summary/metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if not rows or (column_names is not None and reader.fieldnames != column_names):
                raise ValueError("regime summary columns differ or are empty")
            column_names = reader.fieldnames
        for row in rows:
            if row.get("experiment") != experiment:
                raise ValueError("summary row belongs to another experiment")
            if experiment == "S3" and row.get("training_regime") != regime:
                raise ValueError("summary row belongs to another training regime")
            if experiment == "S1" and row.get("method") not in (regime, f"{regime} (high-frequency branch disabled)"):
                raise ValueError("S1 method label differs from its source regime")
            row["training_regime"] = regime
        all_rows.extend(rows)
        summaries[regime] = json.loads((directory / "summary/summary.json").read_text(encoding="utf-8"))
        sources[regime] = {"report_sha256": checked["report_sha256"], "report": report}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary").mkdir()
    write_csv_new(output / "summary/metrics.csv", all_rows)
    write_json_new(output / "summary/summary.json", {
        "experiment_id": experiment, "training_regimes": ["GIFT", "GIFT-Lite"],
        # The four-vortex test split (IDs 1040-1219) is the common held-out set
        # both regimes were measured against.
        "test_trajectory_ids": [1040, 1219], "key_metrics": summaries})
    finish_output(output, {"schema": "gift.regime-summary.v1", "experiment": experiment,
        "status": "complete", "scope": "aggregation_of_completed_regime_runs",
        "inference_performed_by_this_command": False, "raw_outputs_included": False,
        "source_measurements": sources, "common_observation_hashes": inputs,
        "training_regimes": ["GIFT", "GIFT-Lite"]})
    return {"status": "complete", "experiment": experiment, "rows": len(all_rows), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment", choices=("S1", "S3"), required=True)
    parser.add_argument("--gift", type=Path, required=True)
    parser.add_argument("--gift-lite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(combine(args.experiment, args.gift, args.gift_lite, args.output), indent=2))


if __name__ == "__main__":
    main()
