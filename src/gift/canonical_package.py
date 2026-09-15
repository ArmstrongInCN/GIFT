"""Integrity checks for canonical prediction inputs, without running a model."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .data_splits import prediction_split_manifest


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _member(root, relative):
    parts = relative.split("/")
    if not relative or "\\" in relative or ":" in relative or any(p in ("", ".", "..") for p in parts):
        raise ValueError("unsafe canonical data-package path")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError("canonical data-package member escapes its root")
    return path


def training_input(base, path, *, full):
    """Check declared bytes, splits and (for training) all finite decoded fields.

    A manifest establishes input integrity, not experimental reproducibility.
    The completed run records its manifest and observation hashes separately.
    """
    base, path = Path(base).resolve(strict=True), Path(path).resolve(strict=True)
    relative = "fno/fno1000_n64_t0_t10_dt0p02.h5"
    if path != _member(base, relative):
        raise ValueError("canonical training requires its package's dense training input")
    manifest_path = _member(base, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "gift.data-package-manifest.v1"
            or manifest.get("data_profile") != "canonical"
            or manifest.get("prediction_observations_lossless") is not True):
        raise ValueError("canonical package profile/integrity declaration differs")
    records = {}
    for record in manifest["files"]:
        name = record["path"]
        file = _member(base, name)
        if name.casefold() in records:
            raise ValueError("duplicate canonical data-package path")
        sha = record["sha256"]
        if (file.stat().st_size != record["bytes"] or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("canonical package size/hash declaration differs")
        records[name.casefold()] = record
    consumed = {}
    for name in (relative, "splits.json", "schema.json"):
        record = records.get(name)
        if record is None:
            raise ValueError("canonical package lacks training input or split/schema metadata")
        if (name != relative or full) and _digest(_member(base, name)) != record["sha256"]:
            raise ValueError("canonical package consumed-file SHA256 differs")
        consumed[name] = record["sha256"]
    splits = json.loads((base / "splits.json").read_text(encoding="utf-8"))
    for key, value in prediction_split_manifest().items():
        if splits.get(key) != value:
            raise ValueError("canonical prediction partitions differ")
    schema = json.loads((base / "schema.json").read_text(encoding="utf-8"))
    scientific = schema["files"][relative]
    with h5py.File(path, "r") as handle:
        if (handle.attrs.get("trajectory_id_scheme") != "canonical"
                or handle.attrs.get("fixture_only", False)):
            raise ValueError("canonical input identity scheme or fixture flag differs")
        for group, expected_ids in (("training", list(range(1000))), ("validation", list(range(1000, 1020)))):
            ids = handle[f"{group}/trajectory_index"][:]
            field = handle[f"{group}/vorticity"]
            record = scientific[f"/{group}/vorticity"]
            if (not np.array_equal(ids, expected_ids)
                    or field.shape != (len(expected_ids), 501, 64, 64)
                    or field.dtype != np.dtype("float32")
                    or record["shape"] != list(field.shape) or record["dtype"] != str(field.dtype)
                    or not np.allclose(handle[f"{group}/time"][:], np.arange(501) * .02, rtol=0, atol=2e-12)):
                raise ValueError("canonical training/validation population, shape or time differs")
            if full:
                digest = hashlib.sha256()
                for first in range(0, len(ids), 4):
                    value = np.ascontiguousarray(field[first:first + 4])
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite canonical training/validation observations")
                    digest.update(value.tobytes())
                if digest.hexdigest() != record["decoded_sha256"]:
                    raise ValueError("canonical decoded observations SHA256 differs")
    return dict(profile="canonical", manifest_sha256=_digest(manifest_path), consumed_files=consumed,
                full_input_checks_performed=full, published_dataset_identity=False,
                numerical_reproduction_verified=False)
