"""Verify reference artifact bytes without loading pickle or starting inference.

A missing separately distributed weight is a failure, never a successful quick
verification. This checks the catalog, not numerical or training reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path, catalog: dict) -> dict:
    root = root.resolve(strict=True)
    rows = []
    seen = set()
    for record in catalog["files"]:
        relative = record["path"]
        path = (root / relative).resolve()
        if path == root or root not in path.parents or relative in seen:
            raise ValueError("Artifact catalog contains an unsafe/duplicate path")
        seen.add(relative)
        row = {"path": relative, "distribution": record["distribution"]}
        if not path.is_file():
            row["status"] = "MISSING"
        elif path.stat().st_size != record["bytes"]:
            row["status"] = "SIZE_MISMATCH"
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            row["sha256"] = digest.hexdigest()
            row["status"] = "PASS" if row["sha256"] == record["sha256"].lower() else "HASH_MISMATCH"
        rows.append(row)
    return {"status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
            "scope": "catalog_file_integrity_only", "training_or_inference_performed": False,
            "not_yet_in_catalog": catalog.get("missing", []), "files": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    catalog = json.loads((args.root / "artifacts" / "CHECKPOINTS.json").read_text(encoding="utf-8"))
    result = verify(args.root, catalog)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
