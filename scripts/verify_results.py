"""Read-only result/figure integrity checks, not a substitute for metric recomputation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from experiments.formal._shared.figure_evidence import sha256, verify_svg


def verify_files(root, records):
    root = Path(root).resolve(strict=True)
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty result file inventory is required")
    seen, verified = set(), []
    for record in records:
        relative = record["path"]
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                or any(part in ("", ".", "..") for part in relative.split("/"))
                or relative.casefold() in seen):
            raise ValueError("Unsafe/duplicate result file path")
        seen.add(relative.casefold())
        path = root / relative
        if (path.is_symlink() or not path.is_file() or root not in path.resolve(strict=True).parents
                or type(record["bytes"]) is not int or record["bytes"] < 0
                or not isinstance(record["sha256"], str)
                or not re.fullmatch(r"[A-Fa-f0-9]{64}", record["sha256"])):
            raise ValueError("Invalid result file or hash declaration")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"].upper():
            raise ValueError("Result file size/SHA256 differs: " + relative)
        verified.append(relative)
    return verified


def verify_result(root, experiment):
    """Verify the numerical commit, then the final manifest when present.

    S2 may contain recorded nonfinite predictions; integrity checking neither
    removes those values nor treats their presence as successful prediction.
    """
    root = Path(root).resolve(strict=True)
    numeric = root / "numeric_report.json"
    path = numeric if numeric.is_file() else root / "report.json"
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    statuses = {"complete", "numerical_complete", "numerical_complete_plot_failed"}
    if report.get("experiment") != experiment or report.get("status") not in statuses:
        raise ValueError("Result is not a completed numerical experiment")
    verified = []
    if numeric.is_file():
        verified += verify_files(root, report["numeric_files"])
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("experiment") != experiment or manifest.get("status") not in statuses:
            raise ValueError("Result manifest identity/completion differs")
        verified += verify_files(root, manifest["files"])
    elif not numeric.is_file():
        raise ValueError("Missing result completion manifest")
    if "summary/metrics.csv" not in verified:
        raise ValueError("Result does not bind summary metrics")
    return dict(experiment=experiment, files_verified=len(set(verified)),
                report_sha256=sha256(path), status="PASS_INTEGRITY_ONLY",
                numerical_metrics_recomputed=False, training_performed=False)


def verify_figures(root, result):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "figure_manifest.json").read_text(encoding="utf-8"))
    evidence = manifest.get("evidence", {})
    if (manifest.get("status") != "complete" or evidence.get("experiment") != result["experiment"]
            or evidence.get("report_sha256", "").upper() != result["report_sha256"]):
        raise ValueError("Figure package belongs to another numerical report")
    files = verify_files(root, manifest["files"])
    images = [name for name in files if name.lower().endswith(".svg")]
    if not images or set(images) != set(manifest["svg_checks"]):
        raise ValueError("Figure SVG inventory differs")
    for name in images:
        actual = verify_svg(root / name, editable_text="panels" not in Path(name).parts)
        if actual != manifest["svg_checks"][name]:
            raise ValueError("Figure SVG structure differs")
    return dict(status="PASS_INTEGRITY_ONLY", svg_files_verified=len(images),
                source_report_bound=True, visual_inspection_performed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--experiment", choices=("M1", "M2", "M3", "S1", "S2", "S3"), required=True)
    parser.add_argument("--figures", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    result = verify_result(args.result_dir, args.experiment)
    result["figure_checks"] = [verify_figures(path, result) for path in args.figures]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
