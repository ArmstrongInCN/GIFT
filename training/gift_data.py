"""Read-only input and prerequisite gates for independent GIFT training.

Collection orchestration only: scientific schemas/protocols remain owned by
the generation assembler. No arrays, models or numerical thresholds are edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from training.checkpoints import digest_file

DATASETS = {
    "noise_000": "standard_ns_n64_full_spectrum.h5",
    "noise_001": "m1_parameter_identification/noise_001.h5",
    "noise_010": "m1_parameter_identification/noise_010.h5",
}
RELEASED_SHA = {
    "noise_000": "c3f21ef13d0c81713f2770fc67547c91bd5a793fa34a9eecc5c46537237fa8a5",
    "noise_001": "97c5773290b5c50393ff64b7b42c021fd54f2bb429fc6d57cd4599f09f9818bc",
    "noise_010": "87661d4ec2844fd2eec4256d4dcf2c2fea1b948acb39adde454e95e855ddd3bc",
}
PROFILE_SOURCES = (
    "training/gift_data.py", "scripts/assemble_generated_data.py",
    "scripts/generate_data.py", "scripts/generate_noise.py", "src/even_full_spectrum_ns.py",
    "src/gift/prediction_cohorts.py", "src/gift/gaussian_package.py",
    "src/gift/gaussian_initial.py", "src/gift/canonical_package.py",
)


def input_file(base: Path, condition: str, profile: str) -> Path:
    """The S4 Lite input has its own IDs and is never an M1 noise condition."""
    if profile == 'gaussian':
        if condition != 'noise_000':
            raise ValueError('S4 contains only clean Gaussian observations, not an M1 noise experiment')
        return Path(base)/'s4_gaussian/lite.h5'
    return Path(base)/DATASETS[condition]


def validate_input(base: Path, condition: str, profile: str, *, full: bool) -> dict:
    """Before training, require released bytes or a complete generated collection.

    full=False is only a read-only plan: no published byte identity is claimed.
    """
    from scripts import assemble_generated_data as assembly
    from scripts import generate_data as gen

    base = Path(base).resolve(strict=True)
    if profile == 'gaussian':
        from gift.gaussian_package import validate_input as validate_gaussian
        result = validate_gaussian(base, input_file(base, condition, profile), full=full)
        return dict(result, condition=condition)
    relative = DATASETS[condition]
    path = assembly.checked_file(base, relative)
    if profile == "released":
        actual = digest_file(path) if full else None
        if full and actual != RELEASED_SHA[condition]:
            raise ValueError("released GIFT input SHA256 differs; no training started")
        return dict(profile=profile, condition=condition, file=relative, sha256=actual,
                    bytes=path.stat().st_size, full_input_checks_performed=full,
                    published_dataset_identity=full, numerical_reproduction_verified=False)
    if profile != "regenerated":
        raise ValueError("unknown GIFT data profile")
    raw = assembly.checked_file(base, "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema") != "gift.generated-data-collection.v1"
            or manifest.get("data_profile") != profile
            or manifest.get("status") != "ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED"
            or manifest.get("scientific_arrays_modified") is not False
            or manifest.get("released_truth_used_to_fill_gaps") is not False
            or manifest.get("full_hash_and_finite_checks_performed") is not True
            or manifest.get("assembler_sha256") != digest_file(Path(assembly.__file__))):
        raise ValueError("regenerated collection status/provenance/assembler differs")
    jobs = ["standard"] + ([] if condition == "noise_000" else ["noise"])
    allowed = set(assembly.CLEAN_PATHS) | set(assembly.SAMPLING) | {"noise"}
    available, missing = manifest.get("available_jobs", []), manifest.get("missing_jobs", [])
    if (len(available) != len(set(available)) or len(missing) != len(set(missing))
            or set(available) & set(missing) or set(available) | set(missing) != allowed
            or not set(jobs).issubset(available)):
        raise ValueError("regenerated collection job inventory differs")
    records, seen = {}, set()
    for record in manifest["files"]:
        name, expected = record["path"], record["sha256"]
        if (not name or "\\" in name or ":" in name or any(p in ("", ".", "..") for p in name.split("/"))
                or name.casefold() in seen):
            raise ValueError("unsafe/duplicate generated collection path")
        seen.add(name.casefold())
        item = assembly.checked_file(base, name)
        if (item.stat().st_size != record["bytes"] or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("generated collection file size/SHA declaration differs")
        records[name] = record
    consumed = {}

    def consume(name, category, *, metadata=False):
        record = records.get(name)
        if record is None or record.get("category") != category:
            raise ValueError("missing/wrong generated input or provenance category")
        item = assembly.checked_file(base, name)
        if (full or metadata) and digest_file(item) != record["sha256"]:
            raise ValueError("generated input/provenance SHA256 differs")
        consumed[name] = record["sha256"]
        return item

    splits = assembly.read_json(consume("splits.json", "split_metadata", metadata=True))
    if splits.get("schema") != "gift.generated-splits.v1":
        raise ValueError("generated split schema differs")
    parent_hashes, attempts = {}, {}
    canonical = gen.make_plan(argparse.Namespace(dataset="standard", initial_conditions=None,
        subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))
    for job in jobs:
        for filename in ("run.json", "COMPLETE.json"):
            consume(f"provenance/{job}/{filename}", "generation_provenance", metadata=True)
        attempt = assembly.load_attempt(job, base / "provenance" / job)
        targets = [DATASETS["noise_000"]] if job == "standard" else [DATASETS[x] for x in ("noise_001", "noise_010")]
        paths = {Path(name).name: consume(name, "generated_scientific_input") for name in targets}
        if full:
            for item in paths.values():
                with h5py.File(item, "r") as handle:
                    if (handle.attrs.get("fixture_only", False)
                            or any(token in str(handle.attrs.get(key, "")).upper()
                                   for key in ("purpose", "attempt_id") for token in ("MOCK", "TEST", "PILOT"))):
                        raise ValueError("test/mock/pilot data cannot qualify as regenerated training input")
        if job == "standard":
            plan = attempt["scientific"]
            if plan.get("schema") != canonical["schema"] or plan.get("sources") != canonical["sources"] or plan.get("dt") != gen.DT:
                raise ValueError("generation source/PDE protocol differs")
            validated = assembly.validate_clean(attempt, full=full, data_path=paths[Path(targets[0]).name])
        else:
            validated = assembly.validate_derivative(attempt, parent_hashes, full=full, data_paths=paths)
        for _, name, digest, groups in validated:
            record = records[name]
            if (record.get("generation_job") != job or record.get("attempt_id") != attempt["run"]["attempt_id"]
                    or record["sha256"] != digest or splits["files"].get(name) != groups):
                raise ValueError("generated file/receipt/split binding differs")
            parent_hashes[name] = digest
        attempts[job] = dict(attempt_id=attempt["run"]["attempt_id"],
                             binding_sha256=attempt["run"]["binding_sha256"],
                             sources=attempt["scientific"]["sources"])
    return dict(profile=profile, condition=condition, file=relative,
                sha256=consumed[relative] if full else None, bytes=path.stat().st_size,
                manifest_sha256=hashlib.sha256(raw).hexdigest(), consumed_files=consumed,
                generation_attempts=attempts, full_input_checks_performed=full,
                published_dataset_identity=False, numerical_reproduction_verified=False)


def validate_low_prerequisite(payload: dict, training_data: dict, formal_configuration: dict) -> None:
    """Do not let a copied published/tiny/wrong-data low model seed fresh high training."""
    data, config = payload.get("training_data", {}), payload.get("training_configuration", {})
    expected_ids = list(range(1220, 1270)) if training_data.get('profile') == 'gaussian' else list(range(50))
    def normalize(value):
        return json.loads(json.dumps(value))
    if (payload.get("format_version") != 3 or payload.get("artifact_role") != "trained_generator"
            or payload.get("fresh_training", {}).get("pretrained_model_loaded") is not False
            or payload.get("condition", "noise_000") != "noise_000"
            or str(data.get("sha256", "")).lower() != str(training_data.get("sha256", "")).lower()
            or not training_data.get("sha256") or not training_data.get("full_input_checks_performed")
            or data.get("group") != "training" or data.get("trajectory_ids") != expected_ids
            or any(normalize(config.get(key)) != normalize(value) for key, value in formal_configuration.items())):
        raise ValueError("low prerequisite is not fresh formal clean training on this exact standard input")
    provenance = data.get("provenance")
    if provenance is None:
        # Historical independent v3 payloads predate profile metadata. Accept
        # only their explicitly fresh/formal record and exact released SHA.
        if training_data["profile"] != "released" or data["sha256"].lower() != RELEASED_SHA["noise_000"]:
            raise ValueError("regenerated low prerequisite requires input-profile provenance")
    elif provenance != training_data:
        raise ValueError("low prerequisite input-profile/collection binding differs")
