"""Lossless, SHA-bound file parts for distributing a large trained checkpoint.

This changes file storage only. Training/resume journals use their own format;
the reconstructed bytes are passed to torch.load(weights_only=True) unchanged.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re

SCHEMA = "gift.sharded-checkpoint.v1"
MAX_BYTES = 2 * 1024 ** 3


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate checkpoint-index key")
        result[key] = value
    return result


def read_index(path):
    """Validate metadata and flat, local part paths before reading large files."""
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 1024 ** 2:
        raise ValueError("Checkpoint index is unexpectedly large")
    index = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if (not isinstance(index, dict) or set(index) != {"schema", "bytes", "sha256", "parts"} or index["schema"] != SCHEMA
            or type(index["bytes"]) is not int or not 0 < index["bytes"] <= MAX_BYTES
            or not isinstance(index["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", index["sha256"])
            or not isinstance(index["parts"], list) or not 1 <= len(index["parts"]) <= 2048):
        raise ValueError("Invalid checkpoint index")
    names = set()
    for part in index["parts"]:
        if (not isinstance(part, dict) or set(part) != {"path", "bytes", "sha256"}
                or not isinstance(part["path"], str)
                or not re.fullmatch(r"weights-[0-9]{4}\.part", part["path"])
                or part["path"] in names or type(part["bytes"]) is not int
                or not 0 < part["bytes"] <= 90 * 1024 ** 2
                or not isinstance(part["sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", part["sha256"])):
            raise ValueError("Invalid/duplicate checkpoint part")
        names.add(part["path"])
        file = path.parent / part["path"]
        if file.is_symlink() or file.resolve(strict=True).parent != path.parent or not file.is_file():
            raise ValueError("Checkpoint parts must be ordinary sibling files")
        if file.stat().st_size != part["bytes"]:
            raise ValueError("Checkpoint part size differs")
    if sum(part["bytes"] for part in index["parts"]) != index["bytes"]:
        raise ValueError("Checkpoint total size differs")
    return index


def reconstruct(path, output):
    """Stream exact bytes to a caller-owned file or memory buffer, checking hashes."""
    path = Path(path).resolve(strict=True)
    index = read_index(path)
    total = hashlib.sha256()
    for part in index["parts"]:
        digest = hashlib.sha256()
        count = 0
        with (path.parent / part["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
                count += len(block)
                if count > part["bytes"]:
                    raise ValueError("Checkpoint part grew during reading")
                digest.update(block)
                total.update(block)
                output.write(block)
        if count != part["bytes"] or digest.hexdigest() != part["sha256"]:
            raise ValueError("Checkpoint part SHA-256 differs")
    if total.hexdigest() != index["sha256"]:
        raise ValueError("Reconstructed checkpoint SHA-256 differs")
    return index


def load_weights(path):
    """Load a normal .pt/.pth file or an explicitly named sharded JSON index."""
    import torch

    path = Path(path)
    if path.suffix.lower() != ".json":
        return torch.load(path, map_location="cpu", weights_only=True)
    with io.BytesIO() as buffer:
        reconstruct(path, buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=True)


def split_weights(source, output, *, part_bytes=40 * 1024 ** 2):
    """Create a new distribution folder, never overwrite an input or prior export."""
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if type(part_bytes) is not int or not 0 < part_bytes <= 90 * 1024 ** 2:
        raise ValueError("Part size must be positive and at most 90 MiB")
    if source.suffix.lower() not in (".pt", ".pth") or not 0 < source.stat().st_size <= MAX_BYTES:
        raise ValueError("Expected a bounded .pt/.pth checkpoint")
    if (source.stat().st_size + part_bytes - 1) // part_bytes > 2048:
        raise ValueError("Part size would create more than 2048 parts")
    for name in ("GIFT_DATA_ROOT", "GIFT_EXTERNAL_ROOT"):
        if os.environ.get(name):
            protected = Path(os.environ[name]).resolve()
            if output == protected or protected in output.parents:
                raise ValueError("Weight export must not write into data or upstream source roots")
    # Refuse to disguise arbitrary input data as a model distribution.
    payload = load_weights(source)
    if (not isinstance(payload, dict) or payload.get("status") != "complete"
            or not isinstance(payload.get("model_state_dict"), dict) or not payload["model_state_dict"]):
        raise ValueError("Expected complete terminal model weights")
    del payload
    output.mkdir(parents=True, exist_ok=False)
    total, parts = hashlib.sha256(), []
    with source.open("rb") as stream:
        while block := stream.read(part_bytes):
            name = "weights-%04d.part" % (len(parts) + 1)
            with (output / name).open("xb") as destination:
                destination.write(block)
            total.update(block)
            parts.append({"path": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    index = {"schema": SCHEMA, "bytes": sum(part["bytes"] for part in parts),
             "sha256": total.hexdigest(), "parts": parts}
    target = output / "weights.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")
    # Validate source identity again after the copy. Never silently publish a torn file.
    actual = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            actual.update(block)
    if actual.hexdigest() != index["sha256"]:
        raise ValueError("Source checkpoint changed during splitting; export retained for inspection")
    read_index(target)
    return target
