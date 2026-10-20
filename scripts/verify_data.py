"""Read-only verification of a separately downloaded GIFT input-data package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

FORMAL_ROLES = {
    "four_vortex.standard_n64", "four_vortex.dense_n64", "four_vortex.cross_resolution",
    "four_vortex.fno_training", "four_vortex.fno_training_dt0p02", "four_vortex.fno_test_dt0p02",
    "w1_w2.train300", "w1_w2.train50", "w1_w2.confirmation_n64", "w1_w2.confirmation_n96_n128",
    "s3.trajectories", "s3.lite", "w3.cases", "w3.parameters",
}


def formal_catalog(root, paths):
    """Require the complete formal role map and its manifest-listed metadata."""
    from gift.paths import dataset_catalog, package_member
    catalog = dataset_catalog(root)
    if catalog is None or not FORMAL_ROLES <= set(catalog["roles"]):
        raise ValueError("Missing formal dataset input roles")
    required = {"README.md", "DATA_DICTIONARY.md", "LICENSE.txt", "dataset.json",
                "metadata/schema.json", "metadata/splits.json", "metadata/provenance.json",
                "metadata/experiment_coverage.json"}
    if (catalog.get("schema_file") != "metadata/schema.json"
            or catalog.get("splits_file") != "metadata/splits.json" or not required <= set(paths)):
        raise ValueError("Missing required formal package metadata")
    for role, relative in catalog["roles"].items():
        member = package_member(root, relative)
        if relative.startswith("source_data/"):
            raise ValueError("Formal input roles cannot select historical source_data")
        if role == "w3.cases":
            if not member.is_dir():
                raise ValueError("W3 cases role must select a directory")
        elif relative not in paths or not member.is_file():
            raise ValueError("Formal role is not a manifest-listed file: " + role)
    return catalog


def formal_observations(root, catalog, entries):
    """Check protocol identities, time axes and the fifty case/sampling receipts.

    Only small identity/time arrays are read here. File hashes were already
    checked against the manifest; receipts are compared to those same records.
    """
    import h5py
    import numpy as np
    from gift.paths import data_path
    from gift.gaussian_package import validate_input

    def group(role, name, ids, times, grid):
        with h5py.File(data_path(root, role), "r") as handle:
            value = handle[name]
            saved_times = value["absolute_time"] if "absolute_time" in value else value["time"]
            if (value["trajectory_index"].dtype != np.dtype("int64")
                    or value["trajectory_index"][:].tolist() != list(ids)
                    or value["vorticity"].shape != (len(ids), len(times), grid, grid)
                    or value["vorticity"].dtype != np.dtype("float32")
                    or saved_times.dtype != np.dtype("float64")
                    or saved_times.shape != (len(times),)
                    or not np.allclose(saved_times[:], times, rtol=0, atol=2e-12)):
                raise ValueError("Formal observation identities/shape/time differ: " + role + ":" + name)

    train_ids = range(1000)
    validation = range(1000, 1020)
    aux = range(1020, 1040)
    test_ids = range(1040, 1220)
    for role, dt in (("four_vortex.fno_training", .1), ("four_vortex.fno_training_dt0p02", .02)):
        for name, ids in (("training", train_ids), ("validation", validation)):
            group(role, name, ids, np.arange(round(10 / dt) + 1) * dt, 64)
    for name, ids in (("training", range(50)), ("validation", validation)):
        group("four_vortex.standard_n64", name, ids, np.arange(501) * .02, 64)
    for name, ids in (("test", test_ids), ("validation_aux", aux)):
        group("four_vortex.standard_n64", name, ids, 5 + np.arange(11) * .5, 64)
        group("four_vortex.dense_n64", name, ids, 4.1 + np.arange(20) * .1, 64)
        for grid in (96, 128):
            group("four_vortex.cross_resolution", f"N{grid}/{name}", ids, 4.1 + np.arange(20) * .1, grid)
        for grid, frames in ((64, 196), (96, 96), (128, 96)):
            group("four_vortex.fno_test_dt0p02", f"N{grid}/{name}", ids, 4.1 + np.arange(frames) * .02, grid)
    chosen = np.random.default_rng(2026100304).permutation(1000)[:300].tolist()
    for role, ids in (("w1_w2.train300", chosen), ("w1_w2.train50", chosen[:50])):
        group(role, "training", ids, 5 + np.arange(76) * .04, 64)
        group(role, "validation", validation, 5 + np.arange(76) * .04, 64)
    group("w1_w2.confirmation_n64", "test", range(100000, 100180), 5 + np.arange(151) * .02, 64)
    for grid in (96, 128):
        group("w1_w2.confirmation_n96_n128", f"N{grid}/test", range(100000, 100180), 5 + np.arange(51) * .02, grid)
    for role in ("s3.trajectories", "s3.lite"):
        validate_input(root, data_path(root, role), full=False)

    records = {entry["path"]: entry for entry in entries}
    parameter_relative = catalog["roles"]["w3.parameters"]
    with data_path(root, "w3.parameters").open(encoding="utf-8-sig", newline="") as stream:
        parameters = list(csv.DictReader(stream))
    cases = [f"parameter_{index:03d}" for index in range(50)]
    if [row.get("case_id") for row in parameters] != cases:
        raise ValueError("W3 parameter design must contain the ordered fifty cases")
    cases_root = data_path(root, "w3.cases")
    initial_relative = catalog["roles"]["w3.cases"] + "/INITIAL_CONDITIONS.json"
    initial = json.loads(package_path(root, initial_relative).read_text(encoding="utf-8"))
    initial_parameters = np.asarray(initial.get("splits", {}).get("training", {}).get("parameters"), dtype=np.float64)
    if (initial.get("schema") != "gift.w3-single-trajectory-initial-conditions.v1"
            or initial.get("splits", {}).get("training", {}).get("ids") != [0]
            or initial_parameters.shape != (1, 4, 4) or not np.isfinite(initial_parameters).all()):
        raise ValueError("W3 initial-condition identity/parameters differ")
    initial_digest = records.get(initial_relative, {}).get("sha256")
    if initial_digest is None:
        raise ValueError("W3 initial conditions are not manifest-bound")
    if sorted(path.name for path in cases_root.iterdir() if path.is_dir()) != cases:
        raise ValueError("W3 case directory inventory differs")
    for row in parameters:
        case_id = row["case_id"]
        prefix = catalog["roles"]["w3.cases"] + "/" + case_id + "/"
        def read(relative):
            return json.loads(package_path(root, prefix + relative).read_text(encoding="utf-8"))
        receipt = read("DATA_RECEIPT.json")
        lock, qa = read("DATA_LOCK.json"), read("DATA_QA.json")
        truth = {key: float(row[key]) for key in ("nu", "beta", "gamma")}
        if (receipt.get("schema") != "gift.w3-single-trajectory-data-receipt.v1"
                or receipt.get("status") != "passed" or receipt.get("case_id") != case_id
                or receipt.get("truth") != truth or lock.get("truth") != truth or qa.get("truth") != truth
                or any(not np.isfinite(value) or value <= 0 for value in truth.values())
                or lock.get("case_id") != case_id or qa.get("case_id") != case_id
                or qa.get("status") != "passed"
                or receipt.get("profile") != "single_trajectory_terminal_v1"
                or lock.get("profile") != "single_trajectory_terminal_v1"
                or receipt.get("data_file", {}).get("relative_path") != "standard_ns_n64_full_spectrum.h5"
                or receipt.get("parameter_csv_sha256") != records[parameter_relative]["sha256"]
                or lock.get("parameter_csv_sha256") != records[parameter_relative]["sha256"]
                or receipt.get("initial_conditions_sha256") != initial_digest
                or lock.get("initial_conditions_sha256") != initial_digest
                or receipt.get("training_trajectories_per_case") != 1
                or receipt.get("validation_trajectories_per_case") != 0):
            raise ValueError("W3 case truth/profile receipt differs: " + case_id)
        bindings = {"standard_ns_n64_full_spectrum.h5": receipt["data_file"]["sha256"],
                    "DATA_LOCK.json": receipt["data_lock_sha256"],
                    "DATA_QA.json": receipt["data_qa_sha256"],
                    "OBSERVATION_PRESERVATION.json": receipt["observation_preservation_sha256"]}
        for relative, expected in bindings.items():
            if records.get(prefix + relative, {}).get("sha256") != expected:
                raise ValueError("W3 data receipt checksum differs: " + case_id)
        data_name = prefix + "standard_ns_n64_full_spectrum.h5"
        if records[data_name]["bytes"] != receipt["data_file"]["bytes"]:
            raise ValueError("W3 observation byte count differs")
        with h5py.File(package_path(root, data_name), "r") as handle:
            if (set(handle) != {"training"} or handle["training/vorticity"].shape != (1, 501, 64, 64)
                    or handle["training/trajectory_index"][:].tolist() != [0]
                    or not np.array_equal(handle["training/initial_condition_parameters"][:], initial_parameters)
                    or not np.allclose(handle["training/time"][:], np.arange(501) * .02, rtol=0, atol=2e-12)):
                raise ValueError("W3 must contain only the single ID0 trajectory")
        sample = read("auxiliary/pinn_sampling/noise_000_seed1234.receipt.json")
        bound = sample.get("case_binding", {})
        expected_binding = {"case_id": case_id, "observation_sha256": records[data_name]["sha256"],
            "data_receipt_sha256": records[prefix + "DATA_RECEIPT.json"]["sha256"],
            "data_lock_sha256": records[prefix + "DATA_LOCK.json"]["sha256"],
            "data_qa_sha256": records[prefix + "DATA_QA.json"]["sha256"],
            "parameter_csv_sha256": records[parameter_relative]["sha256"],
            "initial_conditions_sha256": initial_digest, "data_profile": "single_trajectory_terminal_v1"}
        if (sample.get("schema") != "gift.w3-parameter-sampling-receipt.v1"
                or sample.get("status") != "passed" or bound != expected_binding
                or any(sample.get(key) != value for key, value in {
                    "seed": 1234, "sensors": 500, "time_samples": 60, "train_rows": 24000,
                    "holdout_rows": 6000, "physics_rows": 84000,
                    "truth_parameters_passed_to_sampling": False}.items())
                or sample.get("sampling_sha256") != records[prefix + "auxiliary/pinn_sampling/noise_000_seed1234.npz"]["sha256"]):
            raise ValueError("W3 sampling receipt binding differs: " + case_id)


def verify_formal_splits(root, catalog, numeric):
    """Compare every declared observation group with its actual stored ID/time axes."""
    import h5py
    import numpy as np
    document = json.loads(package_path(root, catalog["splits_file"]).read_text(encoding="utf-8"))
    expected_files = {name for name in numeric if name.endswith(".h5")}
    if (document.get("schema") != "gift.formal-dataset-splits.v1"
            or set(document.get("files", {})) != expected_files):
        raise ValueError("Formal split file inventory differs")
    for relative in sorted(expected_files):
        declared = document["files"][relative]
        with h5py.File(package_path(root, relative), "r") as handle:
            observed = {}
            def visit(name, value):
                if isinstance(value, h5py.Dataset) and name.rsplit("/", 1)[-1] == "vorticity":
                    group = value.parent
                    times = np.asarray(group["absolute_time"][:] if "absolute_time" in group else group["time"][:])
                    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
                        raise ValueError("Invalid formal split time axis")
                    observed[group.name.lstrip("/")] = (group["trajectory_index"][:].tolist(), list(value.shape), times)
            handle.visititems(visit)
            if set(declared) != set(observed):
                raise ValueError("Formal split group inventory differs: " + relative)
            for name, (ids, shape, times) in observed.items():
                record = declared[name]
                spacing = float(times[1] - times[0]) if len(times) > 1 else 0.0
                if (record.get("trajectory_ids") != ids or record.get("vorticity_shape") != shape
                        or record.get("time_count") != len(times)
                        or not np.allclose([record.get("time_start"), record.get("time_stop"), record.get("time_spacing")],
                                           [times[0], times[-1], spacing], rtol=0, atol=2e-12)
                        or (len(times) > 1 and not np.allclose(np.diff(times), spacing, rtol=0, atol=2e-12))):
                    raise ValueError("Formal split identities/shape/time differ: " + relative + ":" + name)


def verify_formal_inputs(root, catalog, entries, schema, *, full):
    """Inspect every experiment input while keeping source_data an integrity-only archive."""
    numeric = {entry["path"] for entry in entries
               if PurePosixPath(entry["path"]).suffix in (".h5", ".npz")
               and not entry["path"].startswith("source_data/")}
    if schema.get("schema") != "gift.observation-schema.v1" or set(schema.get("files", {})) != numeric:
        raise ValueError("Formal scientific array schema inventory differs")
    for relative in sorted(numeric):
        measured = inspect_arrays(package_path(root, relative), full=full)
        declared = schema["files"][relative]
        expected = {name.lstrip("/"): {key: record[key] for key in ("shape", "dtype")}
                    for name, record in declared.items()}
        if measured != expected:
            raise ValueError("Formal array shape/dtype inventory mismatch: " + relative)
    verify_formal_splits(root, catalog, numeric)
    formal_observations(root, catalog, entries)


def sha256(path):
    """Hash a package file in bounded blocks during explicit integrity verification."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_path(root, relative):
    """Accept ordinary package files only, without links or path escapes."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Invalid package-relative path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or str(parts) != relative or any(p in (".", "..") for p in parts.parts):
        raise ValueError("Noncanonical package-relative path")
    path = root / relative
    if any(item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction())
           for item in (path, *path.parents) if item != root):
        raise ValueError("Linked package paths are not accepted")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("Package entry is not an ordinary local file")
    return resolved


def inspect_arrays(path, *, full=False):
    """Describe numeric array shapes/dtypes and optionally scan all HDF values.

    NPZ arrays are decoded and checked for finiteness on every call. HDF files
    support schema-only inspection; full=True reads every value in bounded
    trajectory/frame blocks for large fields. No file is modified.
    """
    import h5py
    import numpy as np

    arrays = {}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                value = archive[name]
                if value.dtype.kind not in "fiub" or not np.isfinite(value).all():
                    raise ValueError(f"Invalid numeric array: {path.name}:{name}")
                arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        return arrays
    with h5py.File(path, "r") as handle:
        visited = set()

        def visit(group):
            """Inspect local HDF datasets once, rejecting external links and optional nonfinite values."""
            address = h5py.h5o.get_info(group.id).addr
            if address in visited:
                return
            visited.add(address)
            for name in group:
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise ValueError("HDF5 external/soft links are not accepted")
                item = group[name]
                if isinstance(item, h5py.Group):
                    visit(item)
                    continue
                if item.is_virtual or item.external or item.dtype.kind not in "fiub":
                    raise ValueError("Expected numeric arrays stored inside this HDF5")
                arrays[item.name.lstrip("/")] = {"shape": list(item.shape), "dtype": str(item.dtype)}
                if full:
                    # Keep memory bounded even for the largest trajectory collections.
                    if item.ndim >= 4:
                        for row in range(item.shape[0]):
                            for frame in range(0, item.shape[1], 16):
                                if not np.isfinite(item[row, frame:frame + 16]).all():
                                    raise ValueError(f"Nonfinite field: {item.name}")
                    elif not np.isfinite(item[...]).all():
                        raise ValueError(f"Nonfinite array: {item.name}")

        visit(handle)
    return arrays


def verify(root, *, full_array_scan=False):
    # Read-only check of a separately downloaded input package: every file's byte
    # hash, the numeric array inventory, and (optionally) every stored value's
    # finiteness. It never reruns experiments or accepts a summary as a full run.
    """Check a downloadable package inventory and optionally every experiment input value.

    File sizes/hashes, declared schemas and split axes are checked read-only.
    HDF full_array_scan enables bounded trajectory/frame scans; input NPZ arrays
    are always decoded and checked. Formal source_data archives receive byte
    integrity checks only. Passing integrity does not rerun a scientific
    experiment or establish equality with an unpublished reference solution.
    """
    root = Path(root).resolve(strict=True)
    manifest = json.loads(package_path(root, "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "gift.data-package-manifest.v1":
        raise ValueError("Expected the downloadable input package, not a generated-run collection")
    entries = manifest["files"]
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)) or "manifest.json" in paths:
        raise ValueError("Duplicate/self-referential manifest entries")
    formal = manifest.get("data_profile") == "formal"
    catalog = formal_catalog(root, paths) if formal else None
    if not formal:
        required = {"README.md", "DATA_DICTIONARY.md", "LICENSE.txt", "splits.json", "schema.json"}
        if not required <= set(paths):
            raise ValueError("Missing required metadata")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(paths) | {"manifest.json"}:
        raise ValueError("Unlisted or missing package files")
    for entry in entries:
        path = package_path(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"].lower():
            raise ValueError(f"Size/SHA-256 mismatch: {entry['path']}")
    schema_relative = catalog["schema_file"] if formal else "schema.json"
    schema = json.loads(package_path(root, schema_relative).read_text(encoding="utf-8"))
    if formal:
        verify_formal_inputs(root, catalog, entries, schema, full=full_array_scan)
    numeric = set() if formal else {name for name in paths if PurePosixPath(name).suffix in (".h5", ".npz")}
    # Canonical packages carry remapped prediction IDs and a richer observation
    # schema, so their group IDs and array inventory are checked beyond hashes.
    canonical = manifest.get("data_profile") == "canonical"
    if canonical:
        from scripts.prepare_prediction_package import PREDICTION_FILES
        expected = set(PREDICTION_FILES) | {"prediction/standard_ns_n64_full_spectrum.h5"}
        gaussian_files = {name for name in paths if name.startswith('s3_gaussian/')}
        if gaussian_files:
            # A Gaussian extension must ship exactly its five S3 files and bind the
            # same cohort partitions and observation schema the append step recorded.
            required_gaussian = {'s3_gaussian/' + name for name in
                                ('trajectories.h5', 'lite.h5', 'run.json', 'COMPLETE.json', 'manifest.json')}
            if gaussian_files != required_gaussian:
                raise ValueError('Gaussian extension inventory differs')
            from gift.gaussian_package import validate_input
            from gift.prediction_cohorts import GAUSSIAN, partitions
            nested = json.loads((root/'s3_gaussian/manifest.json').read_text(encoding='utf-8'))
            split = json.loads((root/'splits.json').read_text(encoding='utf-8'))
            if split.get('additional_populations', {}).get(GAUSSIAN) != partitions(GAUSSIAN):
                raise ValueError('Gaussian extension splits differ')
            for name in ('trajectories.h5', 'lite.h5'):
                relative = 's3_gaussian/' + name
                validate_input(root, root/relative, full=False)
                if schema['files'].get(relative) != nested['observation_schema'][name]:
                    raise ValueError('Gaussian extension schema binding differs')
                expected.add(relative)
        if schema.get("schema") != "gift.observation-schema.v1" or set(schema["files"]) != expected or not expected <= numeric:
            raise ValueError("Canonical observation schema inventory differs")
    elif not formal and set(schema["files"]) != numeric:
        raise ValueError("Numeric schema file inventory differs")
    for relative in sorted(numeric):
        measured = inspect_arrays(root / relative, full=full_array_scan)
        if not canonical:
            if measured != schema["files"][relative]:
                raise ValueError(f"Array shape/dtype inventory mismatch: {relative}")
        elif relative in schema["files"]:
            import h5py
            declared = schema["files"][relative]
            with h5py.File(root / relative, "r") as handle:
                for name, record in declared.items():
                    if "trajectory_ids" in record:
                        if handle[name + "/trajectory_index"][:].tolist() != record["trajectory_ids"]:
                            raise ValueError("Canonical group trajectory IDs differ")
                    elif measured.get(name.lstrip("/")) != {key: record[key] for key in ("shape", "dtype")}:
                        raise ValueError("Canonical scientific array shape/dtype differs")
            fields = {name for name in measured if name.rsplit("/", 1)[-1] not in ("time", "trajectory_index")}
            if fields != {name.lstrip("/") for name, record in declared.items() if "shape" in record}:
                raise ValueError("Canonical scientific array inventory differs")
    total = sum(entry["bytes"] for entry in entries)
    if manifest.get("file_count") != len(entries) or manifest.get("total_bytes") != total:
        raise ValueError("Manifest totals differ")
    return {"status": "verified", "files": len(entries), "bytes": total,
            "manifest_sha256": sha256(root / "manifest.json"),
            "all_file_hashes_checked": True, "array_shapes_dtypes_checked": True,
            "all_numeric_values_finite_checked": full_array_scan and not formal,
            "all_experiment_input_values_finite_checked": full_array_scan,
            "numeric_value_scan_scope": "experiment_inputs_excluding_source_data" if formal else "all_numeric_files",
            "dataset_roles_checked": formal, "w3_cases_checked": 50 if formal else 0,
            "experiment_results_verified": False}


def main(argv=None):
    """Print package integrity results, optionally scanning all HDF values as well."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-array-scan", action="store_true",
                        help="Scan every experiment input value (formal source_data is integrity-only)")
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root, full_array_scan=args.full_array_scan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
