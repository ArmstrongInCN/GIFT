"""Assemble M1's 45 measured parameter rows from 15 independent completed jobs.

No training, checkpoint loading, fitting or model selection takes place here.
The numeric commit is written before optional, separately runnable SVG plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from experiments.formal.m1_equation_identification import run

NOISE = {"noise_000": 0., "noise_001": 1., "noise_010": 10.}
TRUTH = {"nu": .01, "beta": 1., "gamma": 1.}
FIELDS = ["experiment", "condition", "noise_percent", "method", "parameter",
          "estimate", "reference", "absolute_error", "relative_error_percent"]


def collect(jobs):
    """Verify completed job hashes, full coverage and common input per condition."""
    jobs = Path(jobs).resolve(strict=True)
    rows, evidence, data_by_condition = [], [], {}
    for condition, noise in NOISE.items():
        for method, directory in run.METHODS.items():
            job = jobs / condition / directory
            if jobs not in job.resolve(strict=True).parents:
                raise ValueError("M1 job escapes input root")
            run.read_completed(job, method, condition)
            record = json.loads((job / "result.json").read_text(encoding="utf-8"))
            commit = json.loads((job / "COMPLETE.json").read_text(encoding="utf-8"))
            identity = record["identity"]
            if identity.get("method") != method or identity.get("condition") != condition:
                raise ValueError("M1 source method/condition mismatch")
            data = identity["data"]
            if condition in data_by_condition and data != data_by_condition[condition]:
                raise ValueError("M1 methods used different observation data")
            data_by_condition[condition] = data
            parameters = record["parameters"]
            if set(parameters) != set(TRUTH) or any(not math.isfinite(value) for value in parameters.values()):
                raise ValueError("M1 needs exactly three finite parameters")
            with (job / "summary.csv").open(encoding="utf-8", newline="") as stream:
                summary = list(csv.DictReader(stream))
            if (len(summary) != 3 or {row["parameter"] for row in summary} != set(TRUTH)
                    or any(row["condition"] != condition or row["method"] != method
                           or float(row["estimate"]) != parameters[row["parameter"]] for row in summary)):
                raise ValueError("M1 job summary and measured parameters differ")
            for parameter, truth in TRUTH.items():
                estimate = parameters[parameter]
                error = abs(estimate-truth)
                rows.append(dict(experiment="M1", condition=condition, noise_percent=noise,
                    method=method, parameter=parameter, estimate=estimate, reference=truth,
                    absolute_error=error, relative_error_percent=100*error/abs(truth)))
            item = {"method": method, "condition": condition, "identity": identity,
                    "completion_sha256": run.digest(job / "COMPLETE.json"), "files": commit["files"],
                    "readout_or_regression_seconds": record["elapsed_seconds"]}
            if method.startswith("PINN"):
                item["training_cost"] = record["reference_binding"]["training_cost"]
            evidence.append(item)
    return rows, evidence


def aggregate(jobs, output, *, skip_plots=False):
    jobs, output = Path(jobs).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("M1 aggregation requires a new output directory")
    if output == jobs or jobs in output.parents or output in jobs.parents:
        raise ValueError("M1 aggregate and job roots must be separate")
    rows, evidence = collect(jobs)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary").mkdir()
    with (output / "summary/metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # Reverify all source files before making the numeric-completion record.
    if collect(jobs) != (rows, evidence):
        raise ValueError("M1 source jobs changed during aggregation")
    run.write_json(output / "job_sources.json", {"jobs": evidence})
    files = []
    for name in ("summary/metrics.csv", "job_sources.json"):
        path = output / name
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": run.digest(path)})
    report = {"experiment": "M1", "status": "numerical_complete", "parameter_rows": len(rows),
              "completed_jobs": len(evidence), "numeric_files": files, "inputs": {"jobs": evidence},
              "aggregation_source_sha256": run.digest(Path(__file__)),
              "training_executed": False, "inference_executed": False,
              "reference": TRUTH, "statistic": "one estimate per method/condition; absolute percentage error",
              "method_ranking_required": False}
    run.write_json(output / "numeric_report.json", report)
    if not skip_plots:
        try:
            subprocess.run([sys.executable, "-B", "-m",
                "experiments.formal.m1_equation_identification.plot_parameters",
                "--result-dir", str(output), "--output-dir", str(output / "figures")],
                cwd=run.ROOT, check=True)
        except Exception as error:
            run.write_json(output / "plot_failure.json", {"status": "plot_failed",
                "error": str(error), "numeric_results_preserved": True})
            raise
    return {"status": "numerical_complete" if skip_plots else "complete", "completed_jobs": 15,
            "parameter_rows": 45, "output": str(output), "training_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.jobs, args.output, skip_plots=args.skip_plots), indent=2))


if __name__ == "__main__":
    main()
