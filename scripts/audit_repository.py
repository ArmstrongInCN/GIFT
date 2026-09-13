"""Read-only pre-publication checks for the exact proposed Git file list.

Does not assert scientific reproduction or grant licences. In --tracked mode,
Git's tracked files and their actual staged bytes are audited before pushing.
The ordinary candidate mode respects .gitignore after git init.
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
FORBIDDEN = {"data", "external", "vendor", "third_party", "benchmarks", "backup", "论文写作"}
REFERENCE_SHA = "874859f3b40565c6ee252ced471b507f475421213a89e682863db410275cfa5c"


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
    """Allow only six sealed numeric weight archives, never arbitrary NPZ data.

    The adapter pins the reference manifest, which pins every archive. The main
    checkpoint catalog must independently agree. No model code or pickle runs.
    """
    adapter = ROOT / "adapters/pinn_reference.py"
    if not adapter.exists():
        return {}
    parsed = ast.parse(adapter.read_text(encoding="utf-8"))
    pins = [node.value.value for node in parsed.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            and any(isinstance(target, ast.Name) and target.id == "MANIFEST_SHA256" for target in node.targets)]
    manifest_path = ROOT / "artifacts/pinn_reference/manifest.json"
    if len(pins) != 1 or digest(manifest_path) != pins[0]:
        raise ValueError("Numeric reference manifest does not match its adapter pin")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_role") != "reference_not_resume" or manifest.get("fresh_training") is not False:
        raise ValueError("Numeric archive role is not reference weights")
    catalog = json.loads((ROOT / "artifacts/CHECKPOINTS.json").read_text(encoding="utf-8"))["files"]
    expected = {f"{condition}/{mode}/terminal_state.npz" for condition in
                ("noise_000", "noise_001", "noise_010") for mode in ("open", "known")}
    runs = manifest["runs"]
    if len(runs) != 6 or {row["path"] for row in runs} != expected:
        raise ValueError("Numeric reference file inventory differs")
    allowed = {}
    for row in runs:
        relative = "artifacts/pinn_reference/" + row["path"]
        entries = [entry for entry in catalog if entry["path"] == relative]
        if (len(entries) != 1 or entries[0].get("format") != "numeric_npz_reference_state"
                or entries[0]["sha256"] != row["sha256"] or entries[0]["bytes"] != row["bytes"]):
            raise ValueError("Numeric reference and checkpoint catalogs disagree")
        allowed[relative] = row
    return allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    if not (ROOT / ".git").is_dir():
        raise SystemExit("Run git init in the candidate first so the exact ignore policy is respected")
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
    external_fingerprints = set()
    external = os.environ.get("GIFT_EXTERNAL_ROOT")
    if external:
        source_root = Path(external).resolve(strict=True)
        if source_root == ROOT or ROOT in source_root.parents:
            failures.append("External checkout is inside the candidate repository")
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
        allowed_npz = relative in numeric_weights
        if allowed_npz and (path.stat().st_size != numeric_weights[relative]["bytes"]
                            or digest(path) != numeric_weights[relative]["sha256"]):
            failures.append(f"Numeric checkpoint integrity differs: {relative}")
        if (Path(relative).parts[0] in FORBIDDEN or path.suffix.lower() in {".h5", ".hdf5"}
                or (path.suffix.lower() == ".npz" and not allowed_npz)):
            failures.append(f"Data or external source in Git file list: {relative}")
        if path.stat().st_size > 100 * 1024 * 1024:
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
    if digest(ROOT / "EXPERIMENTS.md") != REFERENCE_SHA:
        failures.append("The original EXPERIMENTS.md bytes changed")
    for name in ("README.md", "AGENTS.md", "LICENSE", "external_sources.json"):
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
