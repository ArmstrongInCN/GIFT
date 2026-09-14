"""Obtain or verify pinned upstream checkouts outside the GIFT/data folders.

Existing directories are only verified, never fetched into, checked out or edited.
No upstream implementation is copied into the GIFT repository.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()


def verify(directory, record):
    if git(directory, "rev-parse", "HEAD") != record["commit"]:
        raise ValueError("upstream commit differs: " + directory.name)
    for name, expected in record["files"].items():
        path = (directory / name).resolve(strict=True)
        if directory.resolve() not in path.parents:
            raise ValueError("upstream file escapes checkout")
        if hashlib.sha256(path.read_bytes()).hexdigest().upper() != expected["sha256"]:
            raise ValueError("upstream source bytes differ: " + name)


def prepare(root, sources=None, verify_only=False):
    root = Path(root).expanduser().resolve()
    protected = [ROOT]
    if os.environ.get("GIFT_DATA_ROOT"):
        protected.append(Path(os.environ["GIFT_DATA_ROOT"]).resolve())
    if any(root == path or path in root.parents for path in protected):
        raise ValueError("external source root must be outside project and data")
    registry = json.loads((ROOT / "external_sources.json").read_text(encoding="utf-8"))["sources"]
    selected = list(registry) if sources is None else sources
    if any(name not in registry for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("unknown or duplicate source")
    results = []
    for name in selected:
        record = registry[name]
        directory = (root / record["directory"]).resolve()
        if root not in directory.parents:
            raise ValueError("external checkout escapes selected root")
        existed = directory.exists()
        if not existed:
            if verify_only:
                raise FileNotFoundError(directory)
            directory.mkdir(parents=True, exist_ok=False)
            # Disable newline rewriting when creating this new upstream checkout.
            git(directory, "init", "--quiet")
            git(directory, "config", "core.autocrlf", "false")
            git(directory, "remote", "add", "origin", record["repository"])
            git(directory, "fetch", "--depth", "1", "origin", record["commit"])
            git(directory, "checkout", "--detach", record["commit"])
        verify(directory, record)
        results.append({"source": name, "commit": record["commit"], "verified": True,
                        "existing_directory_unchanged": existed})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.source, args.verify_only), indent=2))


if __name__ == "__main__":
    main()
