"""Publish original-template figures against the frozen single-model measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from adapters import public_metadata

from experiments.formal._shared.figure_evidence import file_record, sha256, verify_svg
from scripts.verify_published_results import verify_package

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("confirmation", "old_test")


def copy_once(source: Path, destination: Path) -> None:
    """Copy exact numeric/figure bytes; publish JSON provenance without source code."""
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        cleaned = public_metadata(payload)
        if cleaned != payload:
            write_once(destination, cleaned)
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise FileExistsError(f"different existing publication: {destination}")
        return
    shutil.copyfile(source, destination)


def write_once(path: Path, payload: dict) -> None:
    """Commit deterministic metadata without replacing an existing publication."""
    text = json.dumps(public_metadata(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"different existing metadata: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def numeric_source(source: Path, destination: Path) -> dict:
    """Keep field arrays numeric; carry small display metadata in the manifest."""
    with np.load(source, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files if archive[k].dtype.kind in "biuf"}
        metadata = {k: archive[k].tolist() for k in archive.files if archive[k].dtype.kind not in "biuf"}
        if set(metadata) - {"methods", "training_seeds_json", "checkpoint_sha256_json"}:
            raise ValueError("Unexpected nonnumeric source metadata")
    if not metadata:
        copy_once(source, destination)
    elif not destination.exists():
        np.savez_compressed(destination, **arrays)
    else:
        with np.load(destination, allow_pickle=False) as current:
            if set(current.files) != set(arrays):
                raise ValueError("Published numeric source inventory differs")
            for key, array in arrays.items():
                np.testing.assert_array_equal(current[key], array)
    return metadata


def publish(experiment: str, figures_root: Path, source_run: Path) -> dict:
    """Bind both cohort plots to the unchanged summary and their exact raw sources."""
    original = ROOT / "results/formal" / f"{experiment}_native_single76_n300_20261005"
    package = ROOT / "results/formal" / f"{experiment}_native_single76_n300_original_style_20261005"
    if (package / "published.json").is_file():
        return {"package": str(package), "integrity": verify_package(package, experiment)}
    for filename in ("metrics.csv", "summary.json"):
        copy_once(original / "summary" / filename, package / "summary" / filename)
    report_hash = sha256(package / "summary/summary.json")
    figure_directories = []
    for cohort in COHORTS:
        source = figures_root / experiment / cohort
        source_manifest = json.loads((source / "figure_manifest.json").read_text(encoding="utf-8"))
        report_path = source_run / f"{experiment.lower()}_evaluation" / cohort / "numeric_report.json"
        actual_report_hash = sha256(report_path)
        if (source_manifest.get("status") != "complete"
                or source_manifest.get("evidence", {}).get("experiment") != experiment
                or source_manifest.get("evidence", {}).get("report_sha256", "").upper() != actual_report_hash):
            raise ValueError(f"figure evidence is not bound to the completed report: {source}")
        for record in source_manifest["files"]:
            item = source / record["path"]
            if item.stat().st_size != record["bytes"] or sha256(item) != record["sha256"].upper():
                raise ValueError(f"figure output changed after the manifest: {item}")
        relative = f"figures/{cohort}"
        destination = package / relative
        destination.mkdir(parents=True, exist_ok=True)
        previews = []
        array_metadata = {}
        for item in sorted(source.rglob("*")):
            if not item.is_file() or item.name == "figure_manifest.json":
                continue
            if item.suffix == ".png":
                name = f"{experiment.lower()}_single76_n300_{cohort}_{item.stem}_20261005.png"
                preview = ROOT / "assets/figures" / name
                copy_once(item, preview)
                previews.append(file_record(preview, ROOT))
            elif item.suffix == ".npz":
                array_metadata[item.name] = numeric_source(item, destination / item.relative_to(source))
            elif item.suffix in {".svg", ".csv", ".json"}:
                copy_once(item, destination / item.relative_to(source))
            else:
                raise ValueError(f"unexpected plot output: {item}")
        files = sorted(p for p in destination.rglob("*") if p.is_file() and p.name != "figure_manifest.json")
        svgs = {p.relative_to(destination).as_posix(): verify_svg(p, editable_text="panels" not in p.parts)
                for p in files if p.suffix == ".svg"}
        if len(svgs) < 2 or not previews:
            raise ValueError("both original-template charts and their PNG previews are required")
        evidence = dict(source_manifest.get("evidence", {}))
        evidence.update(experiment=experiment, report_sha256=report_hash,
                        source_numeric_report_sha256=actual_report_hash,
                        source_figure_manifest_sha256=sha256(source / "figure_manifest.json"),
                        cohort=cohort, population=180,
                        sources=[file_record(package / "summary/metrics.csv", package)])
        payload = dict(source_manifest)
        payload.update(status="complete", evidence=evidence,
                       source_array_metadata=array_metadata,
                       figure_contract=dict(backend="Python matplotlib",
                           archetype="original manuscript comparison templates",
                           editable_vector_curves=True, editable_vector_fields=True,
                           field_previews=previews,
                           residual_definition="prediction minus truth",
                           single_training_instance=True, ANIE_in_W2=False),
                       files=[file_record(p, destination) for p in files], svg_checks=svgs)
        write_once(destination / "figure_manifest.json", payload)
        figure_directories.append(relative)
    original_publication = json.loads((original / "published.json").read_text(encoding="utf-8"))
    inventory = sorted(p for p in package.rglob("*") if p.is_file() and p.name != "published.json")
    publication = dict(original_publication)
    publication.update(figure_directories=figure_directories,
                       measurement=dict(original_publication["measurement"], report_sha256=report_hash),
                       files=[file_record(p, package) for p in inventory])
    write_once(package / "published.json", publication)
    return {"package": str(package), "integrity": verify_package(package, experiment)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    outputs = [publish(experiment, args.figure_root.resolve(strict=True), args.source_run.resolve(strict=True))
               for experiment in ("W1", "W2")]
    write_once(args.figure_root / "PUBLICATION.json", {"status": "complete", "outputs": outputs})
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
