"""Create a read-only-derived, create-only union of the six summary CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.formal._shared.common import (
    create_output,
    default_project_root,
    file_record,
    finish_output,
    write_csv_new,
)


RESULT_NAMES = (
    # The six experiments whose summary CSV files live under results/formal and
    # share the same column family. S4 is absent on purpose: it is evaluated on
    # a second initial-condition population and is summarised in its own
    # directory, so folding it in here would mix two populations in one table.
    ("M1", "M1_equation_identification"),
    ("M2", "M2_recursive_prediction"),
    ("M3", "M3_cross_resolution"),
    ("S1", "S1_high_frequency_branch"),
    ("S2", "S2_recursive_local_correction"),
    ("S3", "S3_seed_stability"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve(strict=True)
    results = (args.results_root or root / "results" / "formal").resolve(strict=True)
    output = args.output or root / "reproduced_results" / "formal_aggregate"
    output = create_output(output, "formal_aggregate")
    (output / "summary").mkdir()
    source_records = {}
    source_rows = []
    # Column set is the union over experiments rather than a fixed schema, so a
    # column that only one experiment reports survives; the order below is the
    # order in which the experiments contribute their columns.
    fields = ["experiment_id"]
    for experiment_id, directory in RESULT_NAMES:
        path = results / directory / "summary" / "metrics.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"empty formal summary: {path}")
        source_records[experiment_id] = file_record(path, root)
        for row in rows:
            normalized = {"experiment_id": experiment_id, **row}
            source_rows.append(normalized)
            for field in normalized:
                if field not in fields:
                    fields.append(field)
    # Rows that do not carry a column get an empty string; no value is imputed.
    union = [{field: row.get(field, "") for field in fields} for row in source_rows]
    write_csv_new(output / "summary" / "metrics.csv", union)
    report = {
        "schema": "gift.formal.aggregate.v1",
        "status": "complete",
        "experiment": "formal_aggregate",
        "source_results_root": str(results),
        "experiment_order": [item[0] for item in RESULT_NAMES],
        "row_count": len(union),
        "source_summary_files": source_records,
        "scientific_values_recomputed": False,
        "role": "lossless tabular union; each experiment remains authoritative",
    }
    finish_output(output, report)


if __name__ == "__main__":
    main()
