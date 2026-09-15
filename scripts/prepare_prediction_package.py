"""Export canonical prediction IDs into a new data package, preserving M1 bytes.

No input is edited, removed or overwritten. Floating-point observations are
copied losslessly. The identity-defined validation/test partition is applied
before any model result is read; this program does not read model results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from gift.data_splits import canonical_ids, prediction_split_manifest, split_name
from training.checkpoints import digest_file

PREDICTION_FILES = (
    "fno/fno1000_n64_t0_t10_dt0p02.h5",
    "fair_short_horizon/fno1000_n64_t0_t10_dt0p1.h5",
    "fair_short_horizon/standard_ns_n64_full_spectrum_test_t4p1_t6p0_dt0p1.h5",
    "cross_resolution/cross_resolution_n96_n128_t4p1_t6p0_dt0p1.h5",
    "fno/fno_test_n64_n96_n128_dt0p02.h5",
)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def safe_member(root, relative):
    parts = relative.split("/")
    if not relative or "\\" in relative or ":" in relative or any(x in ("", ".", "..") for x in parts):
        raise ValueError("unsafe data-package member")
    result = (root / relative).resolve(strict=True)
    if root not in result.parents:
        raise ValueError("data-package member escapes the input directory")
    return result


def _attributes(source, destination):
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _dataset(source, parent, name, rows=None):
    """Copy along the trajectory axis and check decoded bytes after writing."""
    if rows is None:
        source.file.copy(source, parent, name=name)
        return None
    shape = (len(rows), *source.shape[1:])
    kwargs = {}
    if source.ndim > 1:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    if source.ndim == 4:
        # Training reads one trajectory at a time. Keep chunks within a single
        # trajectory instead of repeatedly decompressing neighbouring rows.
        kwargs["chunks"] = (1, min(16, shape[1]), *shape[2:])
    target = parent.create_dataset(name, shape=shape, dtype=source.dtype, **kwargs)
    _attributes(source, target)
    digest = hashlib.sha256()
    for first in range(0, len(rows), 4):
        selected = rows[first:first + 4]
        value = np.ascontiguousarray(source[selected])
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError("source scientific array contains nonfinite observations")
        target[first:first + len(selected)] = value
        actual = np.ascontiguousarray(target[first:first + len(selected)])
        if not np.array_equal(actual, value):
            raise RuntimeError("lossless observation-copy check failed")
        digest.update(value.tobytes())
    return {"shape": list(shape), "dtype": str(source.dtype), "decoded_sha256": digest.hexdigest()}


def _observation_group(source, parent, name, ids, rows, records):
    target = parent.create_group(name)
    _attributes(source, target)
    target.attrs["trajectory_id_scheme"] = "canonical"
    for key, dataset in source.items():
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("unexpected nested object in observation group")
        if key == "trajectory_index":
            target.create_dataset(key, data=ids[rows].astype(np.int64))
        elif key != "time" and dataset.ndim > 0 and dataset.shape[0] == len(ids):
            record = _dataset(dataset, target, key, rows)
            if record:
                records[f"{target.name}/{key}"] = record
        else:
            _dataset(dataset, target, key)
    records[target.name] = {"trajectory_ids": ids[rows].astype(int).tolist(),
                            "split": "validation" if name == "validation_aux" else name}


def _walk(source, destination, records):
    for name, value in source.items():
        if not isinstance(value, h5py.Group):
            _dataset(value, destination, name)
            continue
        if {"trajectory_index", "time", "vorticity"}.issubset(value):
            ids = np.asarray(canonical_ids(value["trajectory_index"][:]), dtype=np.int64)
            if name == "test":
                for split, group_name in (("validation", "validation_aux"), ("test", "test")):
                    rows = np.asarray([i for i, identifier in enumerate(ids) if split_name(identifier) == split], dtype=np.int64)
                    if len(rows):
                        _observation_group(value, destination, group_name, ids, rows, records)
                if any(split_name(identifier) == "training" for identifier in ids):
                    raise ValueError("training observations cannot be exported as prediction test data")
            else:
                if name not in ("training", "validation") or any(split_name(i) != name for i in ids):
                    raise ValueError("observation split disagrees with canonical trajectory identity")
                _observation_group(value, destination, name, ids, np.arange(len(ids)), records)
        else:
            child = destination.create_group(name)
            _attributes(value, child)
            _walk(value, child, records)


def canonical_hdf5(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with h5py.File(source, "r") as original, h5py.File(destination, "x") as output:
        _attributes(original, output)
        output.attrs["trajectory_id_scheme"] = "canonical"
        output.attrs["prediction_split_schema"] = "gift.prediction-splits.v1"
        metadata = json.loads(original.attrs.get("metadata_json", "{}"))
        _walk(original, output, records)
        # Older metadata use both ID lists and inclusive ranges. The datasets
        # are authoritative: derive explicit IDs from the copied groups.
        for key in ("training_trajectory_ids", "validation_trajectory_ids", "trajectory_ids",
                    "pair_commit_path", "pair_commit_required", "artifact_status",
                    "pretest_binding_sha256", "reserved_test_count", "first_50_bitwise_copied",
                    "validation_bitwise_copied"):
            metadata.pop(key, None)
        populations = {split: sorted({i for record in records.values()
                       if record.get("split") == split for i in record["trajectory_ids"]})
                       for split in ("training", "validation", "test")}
        for split, ids in populations.items():
            if ids:
                metadata[f"{split}_trajectory_ids"] = ids
                metadata[f"{split}_count"] = len(ids)
        if "trajectory_count_per_grid" in metadata:
            metadata["trajectory_count_per_grid"] = len(populations["test"])
        metadata.update(trajectory_id_scheme="canonical", status="complete",
                        schema_version="gift.canonical-prediction-observations.v1")
        output.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
    return records


def prepare(source_root, output_root):
    source_root = Path(source_root).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if output_root.exists() or source_root == output_root or source_root in output_root.parents:
        raise ValueError("choose a new output directory outside the read-only input package")
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    generated = manifest.get("schema") == "gift.generated-data-collection.v1"
    if manifest.get("schema") not in ("gift.data-package-manifest.v1", "gift.generated-data-collection.v1"):
        raise ValueError("expected a complete source package or generated collection manifest")
    if manifest.get("data_profile") == "canonical":
        raise ValueError("input is already canonical; do not remap its IDs again")
    if generated:
        from scripts import assemble_generated_data as assembly
        if (manifest.get("missing_jobs") != []
                or set(manifest.get("available_jobs", [])) != set(assembly.CLEAN_PATHS) | set(assembly.SAMPLING) | {"noise"}
                or manifest.get("status") != "ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED"
                or manifest.get("scientific_arrays_modified") is not False
                or manifest.get("released_truth_used_to_fill_gaps") is not False
                or manifest.get("full_hash_and_finite_checks_performed") is not True
                or manifest.get("assembler_sha256") != digest_file(Path(assembly.__file__))):
            raise ValueError("generated collection is incomplete or its assembly provenance differs")
    declared = {}
    for record in manifest["files"]:
        relative = record["path"]
        if relative.casefold() in declared:
            raise ValueError("duplicate data-package member")
        path = safe_member(source_root, relative)
        if path.stat().st_size != record["bytes"] or digest_file(path).lower() != record["sha256"].lower():
            raise ValueError(f"input package hash/size differs: {relative}")
        declared[relative.casefold()] = record
    required = (*PREDICTION_FILES, "standard_ns_n64_full_spectrum.h5",
                "m1_parameter_identification/noise_001.h5", "m1_parameter_identification/noise_010.h5",
                *(f"auxiliary/pinn_sampling/noise_{level}_seed1234.npz" for level in ("000", "001", "010")))
    if any(name.casefold() not in declared for name in required):
        raise ValueError("source package is missing an experiment input")
    if generated:
        from training.baseline_control import _regenerated_input
        _regenerated_input(source_root, source_root / PREDICTION_FILES[0], full=False)
        for job in manifest["available_jobs"]:
            assembly.load_attempt(job, source_root / "provenance" / job)
    disk_parent = output_root.parent
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    decoded_bytes = 0
    for record in manifest["files"]:
        if record["path"].endswith(".h5"):
            sizes = []
            with h5py.File(source_root / record["path"], "r") as handle:
                handle.visititems(lambda _, obj: sizes.append(obj.size * obj.dtype.itemsize)
                                  if isinstance(obj, h5py.Dataset) else None)
            decoded_bytes += sum(sizes) * (2 if record["path"] == "standard_ns_n64_full_spectrum.h5" else 1)
    required_bytes = max(2 * sum(record["bytes"] for record in manifest["files"]), int(decoded_bytes * 1.1)) + 2 * 1024**3
    if shutil.disk_usage(disk_parent).free < required_bytes:
        raise OSError(f"insufficient output disk space; reserve at least {required_bytes} bytes")
    output_root.mkdir(parents=True, exist_ok=False)
    decoded = {}
    # Keep identification observations, noise and sensor designs byte-for-byte.
    for record in manifest["files"]:
        name = record["path"]
        if name in PREDICTION_FILES or name in ("splits.json", "schema.json", "README.md", "DATA_DICTIONARY.md",
                                               "initial_conditions/baseline_extra950.json", "LICENSE.txt"):
            continue
        target = output_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / name, target)
        if digest_file(target).lower() != record["sha256"].lower():
            raise RuntimeError("byte-preserving M1/support copy failed")
    for name in (*PREDICTION_FILES, "standard_ns_n64_full_spectrum.h5"):
        target_name = "prediction/" + name if name == "standard_ns_n64_full_spectrum.h5" else name
        decoded[target_name] = canonical_hdf5(source_root / name, output_root / target_name)
        print(json.dumps({"completed_prediction_file": target_name}), flush=True)
    from scripts.generate_data import array_hash, EXTRA_PARAMETERS_HASH
    training_path = output_root / PREDICTION_FILES[0]
    with h5py.File(training_path, "r") as handle:
        parameters = np.asarray(handle["training/initial_condition_parameters"][50:], dtype=np.float64)
    if parameters.shape != (950, 4, 4) or array_hash(parameters) != EXTRA_PARAMETERS_HASH:
        raise ValueError("extra training initial-condition parameters differ")
    (output_root / "initial_conditions").mkdir(exist_ok=True)
    write_json(output_root / "initial_conditions/baseline_extra950.json", {
        "schema": "gift.initial-conditions.v1", "data_license": "CC-BY-4.0",
        "source_path": PREDICTION_FILES[0], "source_file_sha256": digest_file(training_path),
        "source_dataset": "training/initial_condition_parameters", "source_rows": [50, 1000],
        "parameter_order": ["circulation", "sigma", "centre_x", "centre_y"], "dtype": "float64",
        "trajectory_id_scheme": "canonical", "trajectory_ids": list(range(50, 1000)),
        "parameters": parameters.tolist(), "parameters_sha256": array_hash(parameters),
        "origin": "specified initial conditions; original random seed unknown"})
    splits = prediction_split_manifest()
    splits["files"] = {name: {key: record for key, record in records.items() if "trajectory_ids" in record}
                       for name, records in decoded.items()}
    identification_files = ("standard_ns_n64_full_spectrum.h5",
                            "m1_parameter_identification/noise_001.h5",
                            "m1_parameter_identification/noise_010.h5")
    splits["identification"] = {"identity_scope": "M1 local trajectory IDs",
                                "training": list(range(50)), "validation": list(range(50, 70)),
                                "observation_files": list(identification_files),
                                "pinn_observation_training_rows": 24000,
                                "pinn_observation_validation_rows": 6000,
                                "pinn_local_trajectory_id": 0}
    write_json(output_root / "splits.json", splits)
    write_json(output_root / "schema.json", {"schema": "gift.observation-schema.v1", "files": decoded})
    metadata_root = Path(__file__).resolve().parents[1] / "docs" / "data_package"
    for name in ("README.md", "DATA_DICTIONARY.md", "LICENSE.txt"):
        shutil.copy2(metadata_root / name, output_root / name)
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(output_root).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest_file(path), "data_license": "CC-BY-4.0",
                          "category": "scientific_input" if path.suffix in (".h5", ".npz") else "metadata"})
    result = {"schema": "gift.data-package-manifest.v1", "data_profile": "canonical",
              "title": "GIFT Navier–Stokes input data", "hash_algorithm": "SHA-256",
              "paths_relative_to": "manifest.json directory", "file_count": len(files),
              "total_bytes": sum(record["bytes"] for record in files), "files": files,
              "data_license": "CC-BY-4.0", "doi": None, "contains_trained_models": False,
              "excluded_self": ["manifest.json"], "m1_observations_unchanged": True,
              "prediction_observations_lossless": True, "source_package_manifest_sha256": digest_file(source_root / "manifest.json"),
              "exporter_sha256": digest_file(Path(__file__))}
    write_json(output_root / "manifest.json", result)
    return {"status": "complete", "output": str(output_root), "files": len(files), "bytes": result["total_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
