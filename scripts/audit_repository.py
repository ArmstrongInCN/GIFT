"""Check the files selected for a local Git publication.

Checks syntax, links and excluded data files. Full result and checkpoint
verification remains available through the dedicated verification commands.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Compute a streaming SHA-256 for an explicitly requested catalog check."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def selected_files(root, *, tracked=False):
    """List tracked files, optionally including unignored untracked files.

    Git supplies the selection, so ignore rules and worktrees whose .git is a
    file behave like ordinary checkouts. No index or working file is changed.
    """
    command = ["git", "-C", str(root), "ls-files", "-z"]
    if not tracked:
        command += ["--cached", "--others", "--exclude-standard"]
    selected = {item.decode("utf-8") for item in subprocess.check_output(command).split(b"\0") if item}
    if not tracked:
        # Default publication inspects the proposed working tree. The explicit
        # index mode still reports absent tracked files through audit_files.
        deleted = subprocess.check_output(["git", "-C", str(root), "ls-files", "--deleted", "-z"])
        selected -= {item.decode("utf-8") for item in deleted.split(b"\0") if item}
    return sorted(selected)


def external_boundary_failures(root, files):
    """Reject bundled implementations and executable patch records in a release.

    The project keeps its own GIFT models and experiment controllers. External
    model classes, regression kernels and source editing belong to user-prepared
    modules outside this directory; the local adapters namespace is an interface.
    """
    problems = []
    external_classes = {"FNO2d", "FNO3d", "SpectralConv2d_fast", "SpectralConv3d",
                        "UNO", "U_net", "PhysicsInformedNN", "model_blocks",
                        "Decoder_NN_2D", "SingleConvNeuralNet", "SimpleAttention",
                        "Integral_spatial_attention_solver_multbatch"}
    external_functions = {"STRidge", "TrainSTRidge", "linear_attention", "causal_linear_attn"}
    source_names = {"fourier_2d_time.py", "fourier_3d.py", "navier_stokes_uno2d.py",
                    "integral_operators.py", "U-net.py", "PDE_FIND.py",
                    "nadam_optimizer.py", "external_optimizer.py",
                    "Attentional_IE_solver.py", "Galerkin_transformer.py"}
    for relative in files:
        path = Path(root) / relative
        if not path.is_file():
            continue
        parts = Path(relative).parts
        if (parts[0] in {"external", "vendor", "third_party", "external-adapters",
                         "external_runtime", "gift-external-runtime"}
                or (parts[0] == "adapters" and relative.replace("\\", "/") != "adapters/__init__.py")
                or path.name in source_names):
            problems.append("External implementation inside repository: " + relative)
        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except (SyntaxError, UnicodeError):
                continue  # Syntax diagnostics are reported by audit_files.
            for node in ast.walk(tree):
                if ((isinstance(node, ast.ClassDef) and node.name in external_classes)
                        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name in external_functions)):
                    problems.append("Bundled external algorithm definition: " + relative + ":" + node.name)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in {"exec", "eval"}):
                    problems.append("Executable source adaptation inside repository: " + relative)
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, UnicodeError):
                continue
            def contains_source(item):
                if isinstance(item, dict):
                    patches = item.get("patches", [])
                    if isinstance(patches, list) and any(
                            isinstance(patch, dict) and {"before", "after"} & patch.keys()
                            for patch in patches):
                        return True
                    return any(contains_source(child) for child in item.values())
                return isinstance(item, list) and any(contains_source(child) for child in item)
            if contains_source(value):
                problems.append("Executable patch source text inside repository: " + relative)
    return problems


def numeric_checkpoint_allowlist():
    """Read catalog entries and check the declared index-to-part relationships.

    This checks metadata consistency without reconstructing weight files or
    loading models; the dedicated weight verification tool checks actual bytes.
    """
    catalog = ROOT / "artifacts/CHECKPOINTS.json"
    if not catalog.is_file():
        return {}
    allowed = {}
    for row in json.loads(catalog.read_text(encoding="utf-8"))["files"]:
        relative = row["path"]
        path = (ROOT / relative).resolve()
        if not relative.startswith("artifacts/") or ROOT not in path.parents or relative in allowed:
            raise ValueError("Invalid checkpoint catalog path")
        allowed[relative] = row
    from training.weight_files import read_index
    linked = set()
    for relative, row in allowed.items():
        if row["artifact_role"] != "trained_weights_index":
            continue
        for part in read_index(ROOT / relative)["parts"]:
            name = (Path(relative).parent / part["path"]).as_posix()
            record = allowed.get(name, {})
            if (record.get("artifact_role") != "trained_weights_part"
                    or record.get("checkpoint_index") != relative
                    or record.get("bytes") != part["bytes"]
                    or record.get("sha256", "").lower() != part["sha256"].lower()):
                raise ValueError("Checkpoint part/catalog binding differs")
            linked.add(name)
    if linked != {name for name, row in allowed.items() if row["artifact_role"] == "trained_weights_part"}:
        raise ValueError("Orphan checkpoint part in catalog")
    return allowed


def figure_source_allowlist():
    """Collect all file records declared by formal published packages."""
    records = {}
    for path in (ROOT / "results").glob("formal/*/published.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        prefix = path.parent.relative_to(ROOT)
        for row in manifest["files"]:
            records[(prefix / row["path"]).as_posix()] = row
    return records


def audit_files(root, files, numeric_artifacts):
    """Return publication problems for the selected files, without modifying them.

    Checks cover local file paths, excluded data/weight files, ordinary Git size
    limits, Python syntax and recognizable credential text. This is a repository
    publication check, not numerical experiment verification.
    """
    failures = []
    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink() or root not in path.resolve().parents:
            failures.append("Missing or external file: " + relative)
            continue
        suffix = path.suffix.lower()
        if (suffix in {".h5", ".hdf5"}
                or (suffix in {".npz", ".pt", ".pth", ".part"} and relative not in numeric_artifacts)
                or Path(relative).parts[0] in {"data", "external"}):
            failures.append("Uncatalogued data or weights: " + relative)
        if path.stat().st_size > 100 * 1024 * 1024:
            failures.append("File exceeds ordinary Git size limit: " + relative)
        if suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            except (SyntaxError, UnicodeError) as error:
                failures.append("Python syntax/encoding error: %s: %s" % (relative, error))
        if suffix in {".py", ".md", ".json", ".toml", ".txt", ".yml"}:
            content = path.read_text(encoding="utf-8-sig")
            if re.search(r"(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", content):
                failures.append("Possible credential content: " + relative)
    return failures


def main(argv=None):
    """Audit the Git-selected files and README links; return a nonzero status on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="inspect tracked files only")
    args = parser.parse_args(argv)
    files = selected_files(ROOT, tracked=args.tracked)
    artifacts = numeric_checkpoint_allowlist() | figure_source_allowlist()
    failures = audit_files(ROOT, files, artifacts)
    failures.extend(external_boundary_failures(ROOT, files))
    for name in ("README.md", "LICENSE"):
        if name not in files:
            failures.append("Required publication file missing: " + name)
    for match in re.finditer(r"!?\[[^\]]*\]\(([^\s)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8")):
        target = match[1].split("#", 1)[0]
        if target and not re.match(r"[a-z]+://", target) and not (ROOT / target).exists():
            failures.append("Broken README link: " + target)
    print(json.dumps({"status": "PASS" if not failures else "FAIL",
                      "tracked_only": args.tracked, "files_checked": len(files),
                      "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
