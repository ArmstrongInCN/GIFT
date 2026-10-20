"""Export source-free current W1/W2 models and activate a verified bundle.

No training observations, external algorithm source or optimizer journals are
distributed. Original checkpoints stay read-only. Public copies omit source-text
metadata when necessary and retain every tensor byte; runtime paths are derived
only in a separate activation directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from gift.paths import package_member
from gift.publication_metadata import public_metadata, source_metadata_findings
from scripts.run_w1_single import fixed_json, file_hash
from scripts.reproduce_prediction import METHODS, ROOT, output_root

SCHEMA = "gift.current-model-bundle.v2"
METADATA_POLICY = "source-free-provenance-v1"


def read(path):
    """Read JSON metadata without modifying original provenance."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path, root):
    """Bind one ordinary bundle member's exact bytes."""
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=file_hash(path))


def tensor_identity(payload):
    """Hash all nested tensor names, dtypes, shapes and exact logical bytes.

    Inspect CPU views without changing parameters, gradients or the input
    containers. Byte comparison includes signed zero and nonfinite bit patterns;
    it does not replace the model/recipe checks used before export.
    """
    import torch
    digest = hashlib.sha256()
    count = 0

    def visit(value, path):
        nonlocal count
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            header = json.dumps([path, str(tensor.dtype), list(tensor.shape)],
                                ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
            count += 1
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], path + "/" + str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, path + "/" + str(index))

    visit(payload, "")
    return dict(tensor_count=count, tensor_sha256=digest.hexdigest())


def export_public_checkpoint(source, destination):
    """Write a new source-free inference copy, proving tensors are unchanged.

    A clean checkpoint retains its original file bytes. A checkpoint containing
    source metadata is safely loaded, filtered, and saved only to the new
    destination. Unknown pickle globals are never allowed for export.
    """
    import torch
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Public checkpoint destination already exists")
    before_hash = file_hash(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    before_tensors = tensor_identity(payload)
    removed = source_metadata_findings(payload)
    if removed:
        cleaned = public_metadata(payload)
        if source_metadata_findings(cleaned):
            raise ValueError("Source text remains after public checkpoint filtering")
        torch.save(cleaned, destination)
        restored = torch.load(destination, map_location="cpu", weights_only=True)
        if tensor_identity(restored) != before_tensors:
            raise ValueError("Public checkpoint tensors differ; retain incomplete export for inspection")
    else:
        shutil.copy2(source, destination)
        if file_hash(destination) != before_hash:
            raise ValueError("Copied clean checkpoint bytes differ")
    if file_hash(source) != before_hash:
        raise ValueError("Original checkpoint changed during read-only export")
    return dict(source=str(source), source_bytes=source.stat().st_size,
                source_sha256=before_hash, public_bytes=destination.stat().st_size,
                public_sha256=file_hash(destination), removed_source_fields=removed,
                tensor_values_unchanged=True, **before_tensors)


def export_public_json(source, destination):
    """Copy metadata without source text and record original/public identities."""
    source, destination = Path(source), Path(destination)
    payload = read(source)
    removed = source_metadata_findings(payload)
    if removed:
        fixed_json(destination, public_metadata(payload))
    else:
        shutil.copy2(source, destination)
    return dict(source=str(source), source_sha256=file_hash(source),
                public_sha256=file_hash(destination), removed_source_fields=removed)


def validate_gift_pairs(models):
    """Verify both branch architectures and their actual frozen generator tensors."""
    import torch
    from training.w1_v2_gift import load_gift_checkpoint
    with torch.random.fork_rng(devices=[]):
        for method in ("GIFT", "GIFT-N50"):
            load_gift_checkpoint(models[method], device="cpu")


def export_bundle(output, *, native_root=None, extension_root=None, aligned_root=None):
    """Export nine tensor-identical public models and source-free lock evidence."""
    from experiments.provenance import external_source
    from experiments.additional.w1_current_evaluate import fixed_sources
    from experiments.additional.w1_extended_evaluate import checkpoint_metadata
    roots = [Path(value).expanduser().resolve(strict=True) for value in (
        native_root or external_source("prediction_native"),
        extension_root or external_source("prediction_extension"),
        aligned_root or external_source("prediction_aligned"))]
    paths, base, extension = fixed_sources(*roots)
    # Validate the completed slots before creating a distribution directory.
    for method, path in paths.items():
        checkpoint_metadata(method, path, base, extension)
    validate_gift_pairs(paths)
    output = output_root(output, *roots)
    if output.exists():
        raise FileExistsError("Use a new bundle directory; existing evidence is never replaced")
    selected = {method: (path, f"models/{method}/" + ("branch/model.pt" if method.startswith("GIFT") else path.name))
                for method, path in paths.items()}
    selected["GIFT-generator"] = (roots[0] / "training/GIFT_seed20260820/generator/model.pt", "models/GIFT/generator/model.pt")
    selected["GIFT-N50-generator"] = (roots[1] / "training/GIFT-N50_seed20260820/generator/model.pt", "models/GIFT-N50/generator/model.pt")
    for path, _ in selected.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    models = {}
    originals = {}
    for method, (path, relative) in selected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        originals[relative] = export_public_checkpoint(path, destination)
        if method in METHODS:
            models[method] = relative
    metadata = output / "metadata"
    metadata.mkdir()
    for name, source in (("base_lock.json", roots[2] / "PROTOCOL_LOCK.json"),
                         ("extension_lock.json", roots[1] / "EXTENSION_LOCK.json"),
                         ("anie_STATUS.source.json", paths["anie_ns"].parent / "STATUS.json")):
        originals["metadata/" + name] = export_public_json(source, metadata / name)
    # A relative pointer is a derived operational record. The original receipt
    # remains read-only at its source; the public snapshot is separately bound.
    status = read(metadata / "anie_STATUS.source.json")
    status.update(selected_checkpoint=Path(models["anie_ns"]).name,
                  source_status_snapshot="../../metadata/anie_STATUS.source.json")
    fixed_json((output / models["anie_ns"]).parent / "STATUS.json", status)
    fixed_json(output / "ORIGINS.json", dict(schema="gift.model-bundle-origins.v2", files=originals,
                tensor_values_changed=False,
                source_metadata_removed=any(row["removed_source_fields"] for row in originals.values()),
                original_sources_read_only=True, observations_included=False,
                complete_training_resume_state_included=False))
    files = [record(path, output) for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = dict(schema=SCHEMA, scope="current_W1_W2_inference_models_and_lock_metadata",
                    models=models, files=files, observations_included=False,
                    external_algorithm_source_included=False, complete_training_resume_state_included=False,
                    source_metadata_policy=METADATA_POLICY)
    fixed_json(output / "manifest.json", manifest)
    verify_bundle(output)
    return dict(status="exported_not_uploaded", root=str(output), models=len(models),
                model_files=len(selected), bytes=sum(row["bytes"] for row in files),
                manifest_sha256=file_hash(output / "manifest.json"))


def verify_bundle(bundle):
    """Check inventory and actual nested contents before trusting a public bundle."""
    from gift.publication_files import audit_public_files
    bundle = Path(bundle).expanduser().resolve(strict=True)
    manifest = read(package_member(bundle, "manifest.json"))
    if (manifest.get("schema") != SCHEMA or set(manifest.get("models", {})) != set(METHODS)
            or manifest.get("observations_included") is not False
            or manifest.get("external_algorithm_source_included") is not False
            or manifest.get("complete_training_resume_state_included") is not False
            or manifest.get("source_metadata_policy") != METADATA_POLICY):
        raise ValueError("Expected an explicitly scoped current inference-model bundle")
    names = set()
    for row in manifest["files"]:
        relative = row["path"]
        path = package_member(bundle, relative)
        if (relative.casefold() in names or not path.is_file() or path.is_symlink()
                or path.suffix.lower() not in {".pt", ".json"}):
            raise ValueError("Duplicate/non-file model bundle member")
        names.add(relative.casefold())
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError("Model bundle member hash differs: " + relative)
    actual_paths = [path for path in bundle.rglob("*") if path.is_file()]
    actual = {path.relative_to(bundle).as_posix().casefold() for path in actual_paths}
    if actual != names | {"manifest.json"}:
        raise ValueError("Model bundle inventory differs from its directory")
    if any(relative.casefold() not in names for relative in manifest["models"].values()):
        raise ValueError("A model role is absent from the bound inventory")
    problems = audit_public_files(actual_paths)
    if problems:
        raise ValueError("Public bundle contains forbidden or unsafe contents: " + "; ".join(problems))
    return manifest


def activate_bundle(bundle, output):
    """Write current-machine evaluation pointers separately from immutable weights."""
    from experiments.additional.w1_extended_evaluate import checkpoint_metadata
    bundle = Path(bundle).expanduser().resolve(strict=True)
    manifest = verify_bundle(bundle)
    output = output_root(output, bundle)
    if output.exists():
        raise FileExistsError("Use a new activation directory")
    models = {method: str(package_member(bundle, relative)) for method, relative in manifest["models"].items()}
    base = read(bundle / "metadata/base_lock.json")
    extension = read(bundle / "metadata/extension_lock.json")
    base["protocol_file"] = str(ROOT / "docs/W1_W2_PROTOCOL.md")
    extension["base_lock"] = str(output / "PROTOCOL_LOCK.json")
    for method in ("GIFT", "fno2d", "fno3d", "anie_ns"):
        extension["reused_models"][method] = dict(checkpoint=models[method], sha256=file_hash(models[method]), source_root=str(bundle))
    for method, path in models.items():
        checkpoint_metadata(method, path, base, extension)
    validate_gift_pairs(models)
    output.mkdir(parents=True, exist_ok=False)
    fixed_json(output / "PROTOCOL_LOCK.json", base)
    fixed_json(output / "EXTENSION_LOCK.json", extension)
    receipt = dict(schema="gift.current-prediction.models.v1", checkpoints=models, failed_runs={},
                   base_lock=str(output / "PROTOCOL_LOCK.json"), extension_lock=str(output / "EXTENSION_LOCK.json"),
                   bundle_root=str(bundle), bundle_manifest_sha256=file_hash(bundle / "manifest.json"))
    fixed_json(output / "MODELS.json", receipt)
    return dict(status="activated_verified_models", root=str(output), models=len(models), bundle=str(bundle))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    for key in ("native-root", "extension-root", "aligned-root"):
        export.add_argument("--" + key, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    activate = sub.add_parser("activate")
    activate.add_argument("--bundle", type=Path, required=True)
    activate.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    if action == "verify":
        manifest = verify_bundle(**args)
        result = dict(status="PASS_ALL_BUNDLE_HASHES", files=len(manifest["files"]), models=len(manifest["models"]))
    else:
        result = {"export": export_bundle, "activate": activate_bundle}[action](**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
