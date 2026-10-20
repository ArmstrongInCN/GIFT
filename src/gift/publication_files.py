"""Inspect public model containers without executing pickle or extracting archives.

Hashes establish identity, not the absence of source text. This module reads the
actual metadata in JSON, safe PyTorch checkpoints, sharded checkpoints and NumPy
archives, including members nested in ZIP releases. It never writes input files.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from gift.publication_metadata import source_metadata_findings

MAX_CONTAINER_BYTES = 2 * 1024 ** 3
MAX_ARCHIVE_MEMBERS = 10000
MAX_ARCHIVE_DEPTH = 4
SHARD_SCHEMA = "gift.sharded-checkpoint.v1"
_SOURCE_SUFFIXES = {".py", ".pyc", ".pyo", ".ipynb", ".patch", ".diff",
                    ".sh", ".ps1", ".bat", ".cmd", ".c", ".cpp", ".cu",
                    ".h", ".hpp", ".js", ".ts"}
_UNSUPPORTED_CONTAINERS = {".pkl", ".pickle", ".tar", ".gz", ".bz2", ".xz",
                           ".7z", ".rar"}


def _metadata_findings(value, label):
    """Attach the containing file identity to every prohibited metadata field."""
    return [f"Source metadata inside publication: {label}:{item['path']} ({item['reason']})"
            for item in source_metadata_findings(value)]


def _safe_weights(stream, label, depth=0, budget=None):
    """Use PyTorch's restricted loader; unsupported pickle globals are failures."""
    import torch

    try:
        if depth >= MAX_ARCHIVE_DEPTH:
            return [f"Nested checkpoint depth exceeds publication audit limit: {label}"]
        budget = [0] if budget is None else budget
        failures = []
        if zipfile.is_zipfile(stream):
            with zipfile.ZipFile(stream) as archive:
                records = archive.infolist()
                names = [row.filename for row in records]
                if (len(records) > MAX_ARCHIVE_MEMBERS or len({name.casefold() for name in names}) != len(names)
                        or sum(row.file_size for row in records) > MAX_CONTAINER_BYTES
                        or any(not _safe_member(row.filename)
                               or stat.S_ISLNK(row.external_attr >> 16) or row.flag_bits & 1
                               for row in records)):
                    raise ValueError("Unsafe PyTorch archive member")
                # torch.load may ignore extra members. Audit those source/JSON
                # members too instead of allowing an ignored payload to hide.
                for row in records:
                    suffix = PurePosixPath(row.filename).suffix.lower()
                    nested_label = f"{label}!{row.filename}"
                    if suffix in _SOURCE_SUFFIXES:
                        failures.append(f"Executable/source member inside public model container: {nested_label}")
                    elif suffix == ".json":
                        failures.extend(_metadata_findings(_decode_json(archive.read(row)), nested_label))
                    elif suffix in _UNSUPPORTED_CONTAINERS and PurePosixPath(row.filename).name != "data.pkl":
                        failures.append(f"Unsupported publication container rejected: {nested_label}")
                    elif (PurePosixPath(row.filename).name not in {
                            "data.pkl", "version", "byteorder", ".format_version",
                            ".storage_alignment", "serialization_id"}
                          and not re.fullmatch(r".*/data/[0-9]+", row.filename)):
                        # Unexpected extra records may themselves be archives.
                        # Known tensor storage and pickle records are handled by
                        # the restricted loader rather than interpreted here.
                        failures.extend(_bytes_findings(archive.read(row), row.filename,
                                                       nested_label, depth + 1, budget))
        if hasattr(stream, "seek"):
            stream.seek(0)
        value = torch.load(stream, map_location="cpu", weights_only=True)
    except Exception as error:
        # No unsafe fallback: loading with weights_only=False would execute the
        # object being audited. Keep diagnostics concise and avoid printing data.
        return [f"Unsafe or unreadable checkpoint rejected: {label} ({type(error).__name__})"]
    return failures + _metadata_findings(value, label)


def _json_object(pairs):
    """Reject duplicate keys so hidden fields cannot disappear during decoding."""
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = child
    return value


def _decode_json(data):
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=_json_object)


def _numpy_findings(stream, label):
    """Inspect text metadata and reject object arrays without unpickling them."""
    import numpy as np

    def inspect(array, name):
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot be safely audited")
        if array.dtype.kind not in "US":
            return []
        value = array.tolist()
        def decode(item):
            if isinstance(item, bytes):
                return item.decode("utf-8")
            if isinstance(item, list):
                return [decode(child) for child in item]
            return item
        value = decode(value)
        failures = _metadata_findings({name: value}, label)
        for item in array.reshape(-1).tolist():
            text = decode(item)
            if isinstance(text, str) and text.lstrip().startswith(("{", "[")):
                try:
                    nested = json.loads(text, object_pairs_hook=_json_object)
                except ValueError:
                    continue
                failures.extend(_metadata_findings(nested, f"{label}:{name}"))
        return failures

    try:
        if zipfile.is_zipfile(stream):
            with zipfile.ZipFile(stream) as archive:
                records = archive.infolist()
                names = [row.filename for row in records]
                if (len(records) > MAX_ARCHIVE_MEMBERS or len({name.casefold() for name in names}) != len(names)
                        or sum(row.file_size for row in records) > MAX_CONTAINER_BYTES
                        or any(not _safe_member(row.filename) or not row.filename.endswith(".npy")
                               or stat.S_ISLNK(row.external_attr >> 16) or row.flag_bits & 1
                               for row in records)):
                    raise ValueError("Unsafe or unexpected NumPy archive member")
        if hasattr(stream, "seek"):
            stream.seek(0)
        loaded = np.load(stream, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                failures = []
                for name in loaded.files:
                    failures.extend(inspect(loaded[name], name))
                return failures
        return inspect(loaded, "array")
    except Exception as error:
        return [f"Unsafe or unreadable NumPy container rejected: {label} ({type(error).__name__})"]


def _safe_member(name):
    """Accept portable relative ZIP names; never resolve or extract a member."""
    path = PurePosixPath(name)
    return (bool(name) and "\\" not in name and not path.is_absolute()
            and all(part not in {"", ".", ".."} for part in name.split("/"))
            and all(part == part.rstrip(" .") and ":" not in part for part in name.split("/")))


def _archive_shards(value, name, members, archive, label, depth, budget):
    """Validate and reconstruct a ZIP-contained shard group in memory only."""
    if (set(value) != {"schema", "bytes", "sha256", "parts"}
            or type(value["bytes"]) is not int
            or not 0 < value["bytes"] <= MAX_CONTAINER_BYTES
            or not isinstance(value["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", value["sha256"])
            or not isinstance(value["parts"], list) or not 1 <= len(value["parts"]) <= 2048):
        raise ValueError("Invalid checkpoint index")
    parent = PurePosixPath(name).parent
    parts, names, total = [], set(), hashlib.sha256()
    for row in value["parts"]:
        if (not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row["path"], str)
                or not re.fullmatch(r"weights-[0-9]{4}\.part", row["path"])
                or row["path"] in names or type(row["bytes"]) is not int
                or not 0 < row["bytes"] <= 90 * 1024 ** 2
                or not isinstance(row["sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
            raise ValueError("Invalid checkpoint part")
        names.add(row["path"])
        member = (parent / row["path"]).as_posix()
        if member not in members or members[member].file_size != row["bytes"]:
            raise ValueError("Missing checkpoint part or size mismatch")
        data = archive.read(member)
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Checkpoint part SHA-256 differs")
        total.update(data)
        parts.append(data)
    if sum(map(len, parts)) != value["bytes"] or total.hexdigest() != value["sha256"]:
        raise ValueError("Reconstructed checkpoint identity differs")
    return _safe_weights(io.BytesIO(b"".join(parts)), label, depth + 1, budget), {
        (parent / row["path"]).as_posix() for row in value["parts"]}


def _zip_findings(stream, label, depth, budget):
    """Traverse bounded ZIP members and nested containers without extraction."""
    failures = []
    try:
        with zipfile.ZipFile(stream) as archive:
            records = archive.infolist()
            if len(records) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("Too many archive members")
            members = {}
            portable_names = set()
            for row in records:
                name = row.filename.rstrip("/") if row.is_dir() else row.filename
                if (not _safe_member(name) or name.casefold() in portable_names
                        or stat.S_ISLNK(row.external_attr >> 16) or row.flag_bits & 1):
                    raise ValueError("Unsafe, duplicate, symbolic-link or encrypted ZIP member")
                portable_names.add(name.casefold())
                budget[0] += row.file_size
                if budget[0] > MAX_CONTAINER_BYTES:
                    raise ValueError("Archive expands beyond publication audit limit")
                if not row.is_dir():
                    members[row.filename] = row
            bound_parts = set()
            decoded = {}
            for name in members:
                if PurePosixPath(name).suffix.lower() != ".json":
                    continue
                value = _decode_json(archive.read(name))
                decoded[name] = value
                if isinstance(value, dict) and value.get("schema") == SHARD_SCHEMA:
                    result, parts = _archive_shards(value, name, members, archive,
                                                   f"{label}!{name}", depth, budget)
                    failures.extend(result)
                    bound_parts.update(parts)
            for name in members:
                nested_label = f"{label}!{name}"
                if name in bound_parts:
                    continue
                if name in decoded:
                    failures.extend(_metadata_findings(decoded[name], nested_label))
                else:
                    failures.extend(_bytes_findings(archive.read(name), name, nested_label, depth + 1, budget))
    except Exception as error:
        failures.append(f"Unsafe or unreadable ZIP rejected: {label} ({type(error).__name__}: {error})")
    return failures


def _bytes_findings(data, name, label, depth, budget):
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in _SOURCE_SUFFIXES:
        return [f"Executable/source member inside public model container: {label}"]
    if suffix in {".pt", ".pth"}:
        return _safe_weights(io.BytesIO(data), label, depth, budget)
    if suffix == ".json":
        return _metadata_findings(_decode_json(data), label)
    if suffix in {".npz", ".npy"}:
        return _numpy_findings(io.BytesIO(data), label)
    if suffix == ".part":
        return [f"Checkpoint part requires its validated checkpoint index: {label}"]
    if suffix in _UNSUPPORTED_CONTAINERS:
        return [f"Unsupported publication container rejected: {label}"]
    if suffix == ".zip" or zipfile.is_zipfile(io.BytesIO(data)):
        if depth >= MAX_ARCHIVE_DEPTH:
            return [f"Nested archive depth exceeds publication audit limit: {label}"]
        return _zip_findings(io.BytesIO(data), label, depth, budget)
    return []


def audit_public_file(path):
    """Return failures for actual public-file content; an empty list passes.

    A standalone ``.part`` is deliberately rejected. Audit its sibling index
    JSON instead, or use ``audit_public_files`` to validate and bind all parts.
    Unrecognized pickle objects are rejected rather than loaded unsafely.
    """
    path = Path(path)
    label = str(path)
    try:
        if not path.is_file() or path.is_symlink():
            return [f"Missing or symbolic-link publication file: {label}"]
        if path.stat().st_size > MAX_CONTAINER_BYTES:
            return [f"Publication container exceeds audit size limit: {label}"]
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            return _safe_weights(path, label)
        if suffix in {".npz", ".npy"}:
            return _numpy_findings(path, label)
        if suffix == ".json":
            value = _decode_json(path.read_bytes())
            failures = _metadata_findings(value, label)
            if isinstance(value, dict) and value.get("schema") == SHARD_SCHEMA:
                from training.weight_files import reconstruct

                with io.BytesIO() as buffer:
                    reconstruct(path, buffer)
                    buffer.seek(0)
                    failures.extend(_safe_weights(buffer, label))
            return failures
        return _bytes_findings(path.read_bytes(), path.name, label, 0, [0])
    except Exception as error:
        return [f"Unreadable publication container rejected: {label} ({type(error).__name__}: {error})"]


def audit_public_files(paths):
    """Audit explicit files or folders, verifying each sharded checkpoint once.

    No Git selection is inferred here. Callers can supply the exact proposed
    publication set, including an external release ZIP or unpacked model bundle.
    """
    selected = set()
    for item in paths:
        path = Path(item)
        if path.is_dir() and not path.is_symlink():
            selected.update(child for child in path.rglob("*") if child.is_file())
        else:
            selected.add(path)
    selected = {path.absolute() for path in selected}
    bound_parts = set()
    for path in selected:
        if path.suffix.lower() != ".json" or not path.is_file():
            continue
        try:
            value = _decode_json(path.read_bytes())
            if isinstance(value, dict) and value.get("schema") == SHARD_SCHEMA:
                from training.weight_files import read_index

                index = read_index(path)
                bound_parts.update((path.parent / row["path"]).absolute() for row in index["parts"])
        except Exception:
            continue  # audit_public_file reports the malformed index itself.
    failures = []
    for path in sorted(selected):
        if path in bound_parts:
            continue
        failures.extend(audit_public_file(path))
    return failures
