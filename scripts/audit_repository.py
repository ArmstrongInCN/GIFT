"""Read-only pre-publication checks for the exact proposed Git file list.

Does not assert scientific reproduction or grant licences. In --tracked mode,
Git's tracked files and their actual staged bytes are audited before pushing.
The ordinary file-selection mode respects .gitignore after git init.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
# Directories that must never appear in the published Git file list; the audit
# rejects any tracked file whose top-level path falls in this set.
FORBIDDEN = {"data", "external", "vendor", "third_party", "benchmarks", "backup", "论文写作"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def staged_byte_failures(root, files):
    """Ensure Git will publish exactly the working bytes that were audited.

    Git newline conversion can alter CSV/JSON hashes without changing any
    working file. Compare raw Git blob IDs (header plus unfiltered bytes) with
    stage zero, supporting either repository object format. Never check out,
    renormalize, rewrite the index, or print file contents in this checker.
    """
    # Recompute each staged file's raw Git blob hash (header plus unfiltered bytes)
    # and compare it to the index, so a renormalized CSV/JSON cannot diverge from
    # what the audit already checked in the working tree.
    root = Path(root).resolve(strict=True)
    algorithm = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-object-format"], text=True).strip()
    if algorithm not in {"sha1", "sha256"}:
        raise ValueError("Unsupported Git object format")
    raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"])
    staged = {}
    failures = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, encoded_path = item.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        relative = encoded_path.decode("utf-8")
        if relative in staged or stage != "0" or mode not in {"100644", "100755"}:
            failures.append("Unsupported/conflicted staged entry: " + relative)
        staged[relative] = object_id
    if set(staged) != set(files):
        failures.append("Staged file inventory differs from audited selection")
    for relative in files:
        path = root / relative
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
            failures.append("Cannot verify staged bytes for unsafe/missing file: " + relative)
            continue
        value = hashlib.new(algorithm, usedforsecurity=False)
        value.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(block)
        if value.hexdigest() != staged.get(relative):
            failures.append("Staged bytes differ from audited working file: " + relative)
    return failures


def function_fingerprints(path):
    """Find exact nontrivial AST copies, not merely names or shared formulas."""
    try:
        module = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, SyntaxError):
        return set()
    return {hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and len(list(ast.walk(node))) >= 100}


def numeric_checkpoint_allowlist():
    """Explicit published checkpoint inventory, not arbitrary data archives.

    Catalogued files must remain under artifacts, with exact byte hashes.
    Native PINN terminal validation and experiment validation are separate gates.
    """
    manifest_path = ROOT / "artifacts/CHECKPOINTS.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {}
    role_suffixes = {
        "trained_weights": {".npz", ".pt", ".pth"},
        "resumable_training_state": {".npz", ".pt", ".pth"},
        "trained_weights_index": {".json"},
        "trained_weights_part": {".part"},
    }
    for row in manifest["files"]:
        relative = row["path"]
        path = (ROOT / relative).resolve(strict=True)
        if (not relative.startswith("artifacts/") or ROOT not in path.parents
                or "\\" in relative or ":" in relative or any(part in ("", ".", "..") for part in relative.split("/"))
                or path.suffix.lower() not in role_suffixes.get(row.get("artifact_role"), set())
                or relative in allowed
                or not re.fullmatch(r"[a-fA-F0-9]{64}", row.get("sha256", ""))
                or type(row.get("bytes")) is not int or row["bytes"] <= 0):
            raise ValueError("Invalid checkpoint catalog entry")
        allowed[relative] = row
    from training.weight_files import read_index, reconstruct
    linked_parts = set()
    for relative, row in allowed.items():
        if row["artifact_role"] != "trained_weights_index":
            continue
        index = read_index(ROOT / relative)
        for part in index["parts"]:
            name = (Path(relative).parent / part["path"]).as_posix()
            record = allowed.get(name, {})
            if (record.get("artifact_role") != "trained_weights_part"
                    or record.get("checkpoint_index") != relative
                    or record.get("sha256", "").lower() != part["sha256"]
                    or record.get("bytes") != part["bytes"]):
                raise ValueError("Checkpoint part/catalog binding differs")
            linked_parts.add(name)
        class DiscardBytes:
            def write(self, block):
                return len(block)
        reconstruct(ROOT / relative, DiscardBytes())
    if linked_parts != {name for name, row in allowed.items() if row["artifact_role"] == "trained_weights_part"}:
        raise ValueError("Orphan checkpoint part in catalog")
    return allowed


def figure_source_allowlist():
    """Permit only verified selected-panel arrays in explicit report packages."""
    from scripts.verify_published_results import EXPERIMENTS, verify_package
    allowed = {}
    for experiment, directory in EXPERIMENTS.items():
        relative = "results/formal/" + directory
        root = ROOT / relative
        if not root.exists():
            continue
        checked = verify_package(root, experiment)
        manifest = json.loads((root / "published.json").read_text(encoding="utf-8"))
        records = {r["path"]: r for r in manifest["files"]}
        for name in checked["figure_array_files"]:
            allowed[relative + "/" + name] = records[name]
    return allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    if not (ROOT / ".git").is_dir():
        raise SystemExit("Run git init in the project first so the exact ignore policy is respected")
    command = ["git", "-C", str(ROOT), "ls-files", "-z"]
    if not args.tracked:
        command += ["--cached", "--others", "--exclude-standard"]
    files = sorted(set(item.decode("utf-8") for item in subprocess.check_output(command).split(b"\0") if item))
    failures, warnings = [], []
    try:
        numeric_weights = numeric_checkpoint_allowlist()
    except (ValueError, KeyError, OSError) as error:
        numeric_weights = {}
        failures.append("Numeric checkpoint allowlist invalid: " + str(error))
    try:
        figure_sources = figure_source_allowlist()
    except (ValueError, KeyError, TypeError, OSError) as error:
        figure_sources = {}
        failures.append("Published figure source allowlist invalid: " + str(error))
    external_fingerprints = set()
    external = os.environ.get("GIFT_EXTERNAL_ROOT")
    if external:
        source_root = Path(external).resolve(strict=True)
        if source_root == ROOT or ROOT in source_root.parents:
            failures.append("External checkout is inside the project repository")
        else:
            for source in source_root.rglob("*.py"):
                external_fingerprints.update(function_fingerprints(source))
    else:
        warnings.append("No GIFT_EXTERNAL_ROOT: exact copied-function check was not performed")
    for relative in files:
        path = ROOT / relative
        if path.is_symlink() or path.resolve().parent == path.resolve():
            failures.append(f"Unexpected symbolic link: {relative}")
            continue
        if ROOT not in path.resolve().parents:
            failures.append(f"File resolves outside repository: {relative}")
            continue
        if not path.is_file():
            failures.append(f"Tracked file is missing: {relative}")
            continue
        record = numeric_weights.get(relative) or figure_sources.get(relative)
        allowed_npz = record is not None
        if allowed_npz and (path.stat().st_size != record["bytes"]
                            or digest(path) != record["sha256"].lower()):
            failures.append(f"Numeric artifact integrity differs: {relative}")
        # Raw data, external sources and large binary artifacts may never enter the
        # repository; only the catalogued checkpoint/figure artifacts are permitted.
        if (Path(relative).parts[0] in FORBIDDEN or path.suffix.lower() in {".h5", ".hdf5"}
                or (path.suffix.lower() in {".npz", ".pt", ".pth", ".part"} and not allowed_npz)):
            failures.append(f"Data or external source in Git file list: {relative}")
        if path.stat().st_size > 100 * 1024 * 1024:
            # GitHub refuses ordinary Git objects above 100 MiB, so such files must
            # be split (see split_checkpoint.py) or distributed another way.
            failures.append(f"File exceeds GitHub's 100 MiB ordinary Git limit: {relative}")
        if path.suffix == ".py":
            if function_fingerprints(path) & external_fingerprints:
                failures.append(f"Exact nontrivial upstream function/class copy: {relative}")
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            except SyntaxError as error:
                failures.append(f"Python syntax error: {relative}:{error.lineno}")
        if path.suffix in {".py", ".md", ".json", ".toml", ".txt", ".yml"}:
            text = path.read_text(encoding="utf-8-sig")
            # Deliberately do not print a detected secret or its matching line.
            if re.search(r"(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", text):
                failures.append(f"Possible credential content (not displayed): {relative}")
    for name in ("README.md", "AGENTS.md", "LICENSE", "external_sources.json", "EXPERIMENTS.md"):
        if name not in files:
            failures.append(f"Required publication file missing from Git selection: {name}")
    for match in re.finditer(r"!?\[[^\]]*\]\(([^\s)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8")):
        target = match[1].split("#", 1)[0]
        if target and not re.match(r"[a-z]+://", target) and not (ROOT / target).exists():
            failures.append(f"Broken README link: {target}")
    if args.tracked:
        try:
            failures.extend(staged_byte_failures(ROOT, files))
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            failures.append("Staged-byte verification failed: " + str(error))
    result = {"status": "PASS" if not failures else "FAIL", "tracked_only": args.tracked,
              "staged_bytes_checked": bool(args.tracked),
              "files_checked": len(files), "bytes": sum((ROOT / name).stat().st_size for name in files if (ROOT / name).is_file()),
              "data_in_repository": False if not any("Data or external" in item for item in failures) else True,
              "scientific_reproduction_claimed": False,
              "failures": failures, "warnings": warnings}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
