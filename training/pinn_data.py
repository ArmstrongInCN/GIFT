"""Read-only provenance checks for PINN inputs regenerated from initial conditions.

NumPy/HDF5 only: usable in the separate TensorFlow environment without PyTorch.
This is an input gate, not a change to the network, loss, optimizer or sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from scripts import assemble_generated_data as assembly
from scripts import generate_data as gen


def _inventory(base):
    path = assembly.checked_file(base, "manifest.json")
    manifest = assembly.read_json(path)
    if (manifest.get("schema") != "gift.generated-data-collection.v1"
            or manifest.get("data_profile") != "regenerated"
            or manifest.get("status") != "ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED"
            or manifest.get("scientific_arrays_modified") is not False
            or manifest.get("released_truth_used_to_fill_gaps") is not False
            or manifest.get("full_hash_and_finite_checks_performed") is not True
            or manifest.get("assembler_sha256") != gen.sha256(assembly.__file__)):
        raise ValueError("Generated PINN collection status/provenance/assembler differs")
    available, missing = manifest.get("available_jobs", []), manifest.get("missing_jobs", [])
    allowed = set(assembly.CLEAN_PATHS) | set(assembly.SAMPLING) | {"noise"}
    if (len(available) != len(set(available)) or len(missing) != len(set(missing))
            or set(available) & set(missing) or set(available) | set(missing) != allowed):
        raise ValueError("Generated PINN collection job inventory differs")
    records, names = {}, set()
    for record in manifest["files"]:
        relative = record["path"]
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                or any(part in ("", ".", "..") for part in relative.split("/"))
                or relative.casefold() in names):
            raise ValueError("Unsafe/duplicate generated collection path")
        names.add(relative.casefold())
        file = assembly.checked_file(base, relative)
        if (type(record["bytes"]) is not int or file.stat().st_size != record["bytes"]
                or not isinstance(record["sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])):
            raise ValueError("Generated collection file size/SHA declaration differs")
        records[relative] = record
    return manifest, records, gen.sha256(path)


def verify_regenerated_inputs(sampling, data_path, data_root, condition):
    """Verify the complete clean→noise→sampling chain; never accept a bare new H5.

    Unrelated FNO inputs are not rehashed. Consumed scientific inputs are fully
    hashed and scanned; targets are recalculated from the declared observation H5.
    Success certifies inputs/provenance, not agreement with experimental results.
    """
    if condition not in assembly.SAMPLING.values():
        raise ValueError("Unknown PINN noise condition")
    base = Path(data_root).resolve(strict=True)
    if base == gen.ROOT or gen.ROOT in base.parents:
        raise ValueError("Generated data must remain outside the repository")
    standard = assembly.CLEAN_PATHS["standard"]
    data_relative = standard if condition == "noise_000" else "m1_parameter_identification/" + condition + ".h5"
    sampling_relative = "auxiliary/pinn_sampling/" + condition + "_seed1234.npz"
    if (Path(sampling).resolve(strict=True) != assembly.checked_file(base, sampling_relative)
            or Path(data_path).resolve(strict=True) != assembly.checked_file(base, data_relative)):
        raise ValueError("PINN inputs must match their canonical condition paths in the collection")
    manifest, records, manifest_digest = _inventory(base)
    sampling_job = next(name for name, value in assembly.SAMPLING.items() if value == condition)
    jobs = ["standard"] + ([] if condition == "noise_000" else ["noise"]) + [sampling_job]
    if not set(jobs) <= set(manifest["available_jobs"]):
        raise ValueError("Missing clean/noise/sampling generation dependency")
    consumed, attempts, validated, available = {}, {}, {}, {}

    def consume(relative, category):
        record = records.get(relative)
        if record is None or record.get("category") != category:
            raise ValueError("Missing/wrong generated input or provenance category")
        if gen.sha256(assembly.checked_file(base, relative)) != record["sha256"]:
            raise ValueError("Generated PINN input/provenance SHA256 differs")
        consumed[relative] = record["sha256"]
        return record

    consume("splits.json", "split_metadata")
    for job in jobs:
        for filename in ("run.json", "COMPLETE.json"):
            consume("provenance/" + job + "/" + filename, "generation_provenance")
        attempt = assembly.load_attempt(job, base / "provenance" / job)
        attempts[job] = attempt
        if job == "standard":
            canonical = gen.make_plan(argparse.Namespace(dataset="standard", initial_conditions=None,
                subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))
            if attempt["scientific"].get("sources") != canonical["sources"]:
                raise ValueError("Missing/different PINN data generator or solver source")
            outputs = {"data.h5": standard}
        elif job == "noise":
            outputs = {value + ".h5": "m1_parameter_identification/" + value + ".h5"
                       for value in ("noise_001", "noise_010")}
        else:
            outputs = {"sampling.npz": sampling_relative}
        for relative in outputs.values():
            record = consume(relative, "generated_scientific_input")
            if (record.get("generation_job") != job
                    or record.get("attempt_id") != attempt["run"]["attempt_id"]):
                raise ValueError("Generated PINN input and generation attempt differ")
            if relative.endswith(".h5"):
                with h5py.File(base / relative, "r") as handle:
                    if (handle.attrs.get("fixture_only", False) or any(
                            token in str(handle.attrs.get(key, "")).upper()
                            for key in ("purpose", "attempt_id") for token in ("MOCK", "TEST", "PILOT"))):
                        raise ValueError("Test/mock/pilot data cannot qualify as PINN training input")
        paths = {name: base / relative for name, relative in outputs.items()}
        result = (assembly.validate_clean(attempt, full=True, data_path=paths["data.h5"])
                  if job == "standard" else
                  assembly.validate_derivative(attempt, available, full=True, data_paths=paths))
        for _, relative, digest, splits in result:
            if relative not in outputs.values() or records[relative]["sha256"] != digest:
                raise ValueError("Generated PINN file and completion receipt differ")
            available[relative], validated[relative] = digest, splits
    splits = assembly.read_json(base / "splits.json")
    if (splits.get("schema") != "gift.generated-splits.v1"
            or any(splits["files"].get(relative) != value for relative, value in validated.items())):
        raise ValueError("Generated PINN split metadata differs")
    from scripts.generate_sampling import measurement_targets
    with np.load(base / sampling_relative, allow_pickle=False) as arrays, h5py.File(base / data_relative, "r") as handle:
        if not np.array_equal(arrays["targets"], measurement_targets(handle, arrays)):
            raise ValueError("Generated PINN targets differ from their declared observation H5")
    # Inputs retain the same public fields as the released route; the extra
    # provenance explicitly prevents treating regenerated bytes as released data.
    bound = {key: {"sha256": consumed[relative].upper(), "bytes": records[relative]["bytes"]}
             for key, relative in (("sampling", sampling_relative), ("data", data_relative))}
    bound.update(condition=condition, provenance={"profile": "regenerated", "manifest_sha256": manifest_digest,
        "consumed_files": consumed, "generation_attempts": {
            name: {"attempt_id": item["run"]["attempt_id"], "binding_sha256": item["run"]["binding_sha256"],
                   "sources": item["scientific"]["sources"]} for name, item in attempts.items()},
        "full_input_checks_performed": True, "targets_recomputed_and_exact": True,
        "published_dataset_identity": False, "numerical_reproduction_verified": False})
    return bound


def read_regenerated_inputs(sampling, data_path, data_root, condition, train_rows=None, physics_rows=None):
    """Select the same sensor-major rows and full collocation order as released inputs."""
    bound = verify_regenerated_inputs(sampling, data_path, data_root, condition)
    with np.load(sampling, allow_pickle=False) as archive:
        coordinates = archive["measurement_coordinates"][archive["train_indices"]].copy()
        targets = archive["targets"][archive["train_indices"]].copy()
        physics = np.concatenate([archive["lhs_coordinates"], coordinates]).astype(np.float32)
        lower, upper = archive["lower"].copy(), archive["upper"].copy()
    for count, rows in ((train_rows, len(coordinates)), (physics_rows, len(physics))):
        if count is not None and (type(count) is not int or not 0 < count <= rows):
            raise ValueError("Invalid diagnostic row limit")
    coordinates, targets, physics = coordinates[:train_rows], targets[:train_rows], physics[:physics_rows]
    bound["selection"] = {"train_rows": len(coordinates), "physics_rows": len(physics),
                          "diagnostic_subset": train_rows is not None or physics_rows is not None}
    return coordinates, targets, physics, lower, upper, bound
