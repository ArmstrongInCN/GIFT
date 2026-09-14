"""Read-only verification of a separately downloaded GIFT input-data package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
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
    if any(item.is_symlink() for item in (path, *path.parents) if item != root):
        raise ValueError("Linked package paths are not accepted")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("Package entry is not an ordinary local file")
    return resolved


def inspect_arrays(path, *, full=False):
    """Inspect stored numeric arrays; optionally read every element for finiteness."""
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
    root = Path(root).resolve(strict=True)
    manifest = json.loads(package_path(root, "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "gift.data-package-manifest.v1":
        raise ValueError("Expected the downloadable input package, not a generated-run collection")
    entries = manifest["files"]
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)) or "manifest.json" in paths:
        raise ValueError("Duplicate/self-referential manifest entries")
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
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    numeric = {name for name in paths if PurePosixPath(name).suffix in (".h5", ".npz")}
    if set(schema["files"]) != numeric:
        raise ValueError("Numeric schema file inventory differs")
    for relative in sorted(numeric):
        measured = inspect_arrays(root / relative, full=full_array_scan)
        if measured != schema["files"][relative]:
            raise ValueError(f"Array shape/dtype inventory mismatch: {relative}")
    total = sum(entry["bytes"] for entry in entries)
    if manifest.get("file_count") != len(entries) or manifest.get("total_bytes") != total:
        raise ValueError("Manifest totals differ")
    return {"status": "verified", "files": len(entries), "bytes": total,
            "manifest_sha256": sha256(root / "manifest.json"),
            "all_file_hashes_checked": True, "array_shapes_dtypes_checked": True,
            "all_numeric_values_finite_checked": full_array_scan,
            "experiment_results_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-array-scan", action="store_true",
                        help="Also read every numeric value; this takes longer than hash/schema checks")
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root, full_array_scan=args.full_array_scan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
