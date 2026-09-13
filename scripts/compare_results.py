"""Read-only numeric agreement check; not proof of training or provenance."""
import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = dict(M1="M1_equation_identification", M2="M2_recursive_prediction",
                   M3="M3_cross_resolution", S1="S1_high_frequency_branch",
                   S2="S2_recursive_local_correction", S3="S3_seed_stability")
SCALAR_COLUMNS = {
    "M1": ["estimate", "reference", "absolute_error", "relative_error_percent"],
    "M2": ["mean", "sample_sd", "median", "p95", "maximum"],
    "M3": ["mean", "sample_sd", "median", "p95", "maximum"],
    "S1": ["mean_relative_l2"],
    "S2": ["mean", "sample_sd", "median", "p95", "maximum"],
    "S3": ["within_seed_mean", "between_seed_mean", "between_seed_sample_sd", "coefficient_of_variation_percent"],
}
COUNTS = {"population", "finite_count", "within_seed_population", "within_seed_finite_count"}


def _read(path, experiment):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        keys = [name for name in columns if name not in SCALAR_COLUMNS[experiment] and name not in COUNTS]
        rows = {}
        for row in reader:
            key = tuple(str(Decimal(row[name]).normalize()) if name in {"absolute_time", "lead_time", "noise_percent"}
                        else row[name] for name in keys)
            if key in rows:
                raise ValueError(f"Duplicate row key: {key}")
            rows[key] = row
    if not rows:
        raise ValueError("Empty result table")
    return columns, rows


def compare(experiment, candidate, reference):
    columns, new = _read(candidate, experiment)
    old_columns, old = _read(reference, experiment)
    if columns != old_columns:
        raise ValueError("CSV columns/order differ")
    failures = []
    if new.keys() != old.keys():
        failures.append(dict(kind="row_keys", missing=sorted(old.keys() - new.keys()), extra=sorted(new.keys() - old.keys())))
    numeric_count = count_count = 0
    for key in sorted(new.keys() & old.keys()):
        for column in SCALAR_COLUMNS[experiment]:
            value, target = float(new[key][column]), float(old[key][column])
            allowed = 0.0005 + 0.05 * abs(target)
            numeric_count += 1
            if not (math.isfinite(value) and math.isfinite(target)) or abs(value - target) > allowed:
                failures.append(dict(kind="numeric", key=key, column=column, candidate=str(value),
                                     reference=str(target), allowed_absolute_difference=str(allowed)))
        for column in COUNTS & set(columns):
            count_count += 1
            if Decimal(new[key][column]) != Decimal(old[key][column]):
                failures.append(dict(kind="count", key=key, column=column))
    return dict(schema="gift.numeric-comparison.v1", experiment=experiment,
                status="PASS" if not failures else "FAIL",
                scope="numeric_csv_only_not_provenance_or_training_verification",
                self_comparison=Path(candidate).resolve() == Path(reference).resolve(),
                tolerance=dict(absolute=0.0005, relative_to_reference=0.05),
                candidate_rows=len(new), reference_rows=len(old), numeric_comparisons=numeric_count,
                count_comparisons=count_count, failures=failures,
                inputs={name: dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                        for name, path in (("candidate", candidate), ("reference", reference))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=DIRECTORIES)
    parser.add_argument("candidate", type=Path, help="new summary/metrics.csv")
    parser.add_argument("--output", type=Path, help="optional new JSON report, existing file refused")
    args = parser.parse_args()
    reference = ROOT / "results/formal" / DIRECTORIES[args.experiment] / "summary/metrics.csv"
    report = compare(args.experiment, args.candidate, reference)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"failures", "inputs"}}, indent=2))
    print(f"Failures: {len(report['failures'])}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
