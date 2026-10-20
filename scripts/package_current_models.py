"""Export immutable current W1/W2 models and activate a verified portable bundle.

No training observations, external algorithm source or optimizer journals are
distributed. Weight files are copied byte-for-byte; derived runtime lock paths
are written only into a separate new activation directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from gift.paths import package_member
from scripts.run_w1_single import fixed_json, file_hash
from scripts.reproduce_prediction import METHODS, ROOT, output_root

SCHEMA = "gift.current-model-bundle.v1"


def read(path):
    """Read JSON metadata without modifying original provenance."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path, root):
    """Bind one ordinary bundle member's exact bytes."""
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=file_hash(path))


def validate_gift_pairs(models):
    """Verify both branch architectures and their actual frozen generator tensors."""
    import torch
    from training.w1_v2_gift import load_gift_checkpoint
    with torch.random.fork_rng(devices=[]):
        for method in ("GIFT", "GIFT-N50"):
            load_gift_checkpoint(models[method], device="cpu")


def export_bundle(output, *, native_root=None, extension_root=None, aligned_root=None):
    """Copy the nine current model files and necessary lock/status evidence once."""
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
        original_hash = file_hash(path)
        shutil.copy2(path, destination)
        if file_hash(destination) != original_hash:
            raise ValueError("Copied model bytes differ; preserve this incomplete export")
        originals[relative] = dict(source=str(path), bytes=path.stat().st_size, sha256=original_hash)
        if method in METHODS:
            models[method] = relative
    metadata = output / "metadata"
    metadata.mkdir()
    for name, source in (("base_lock.json", roots[2] / "PROTOCOL_LOCK.json"),
                         ("extension_lock.json", roots[1] / "EXTENSION_LOCK.json"),
                         ("anie_STATUS.original.json", paths["anie_ns"].parent / "STATUS.json")):
        shutil.copy2(source, metadata / name)
        originals["metadata/" + name] = dict(source=str(source), sha256=file_hash(source))
    # A relative pointer is a derived operational record. The original receipt
    # remains a separate byte-identical member, never rewritten in its source.
    status = read(metadata / "anie_STATUS.original.json")
    status.update(selected_checkpoint=Path(models["anie_ns"]).name,
                  original_status_file="../../metadata/anie_STATUS.original.json")
    fixed_json((output / models["anie_ns"]).parent / "STATUS.json", status)
    fixed_json(output / "ORIGINS.json", dict(schema="gift.model-bundle-origins.v1", files=originals,
                tensor_or_weight_metadata_changes=False, observations_included=False,
                complete_training_resume_state_included=False))
    files = [record(path, output) for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = dict(schema=SCHEMA, scope="current_W1_W2_inference_models_and_lock_metadata",
                    models=models, files=files, observations_included=False,
                    external_algorithm_source_included=False, complete_training_resume_state_included=False)
    fixed_json(output / "manifest.json", manifest)
    return dict(status="exported_not_uploaded", root=str(output), models=len(models),
                model_files=len(selected), bytes=sum(row["bytes"] for row in files),
                manifest_sha256=file_hash(output / "manifest.json"))


def verify_bundle(bundle):
    """Check the entire flat inventory before using any model or mutable output."""
    bundle = Path(bundle).expanduser().resolve(strict=True)
    manifest = read(package_member(bundle, "manifest.json"))
    if (manifest.get("schema") != SCHEMA or set(manifest.get("models", {})) != set(METHODS)
            or manifest.get("observations_included") is not False
            or manifest.get("external_algorithm_source_included") is not False
            or manifest.get("complete_training_resume_state_included") is not False):
        raise ValueError("Expected an explicitly scoped current inference-model bundle")
    names = set()
    for row in manifest["files"]:
        relative = row["path"]
        path = package_member(bundle, relative)
        if relative.casefold() in names or not path.is_file():
            raise ValueError("Duplicate/non-file model bundle member")
        names.add(relative.casefold())
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError("Model bundle member hash differs: " + relative)
    actual = {path.relative_to(bundle).as_posix().casefold() for path in bundle.rglob("*") if path.is_file()}
    if actual != names | {"manifest.json"}:
        raise ValueError("Model bundle inventory differs from its directory")
    if any(relative.casefold() not in names for relative in manifest["models"].values()):
        raise ValueError("A model role is absent from the bound inventory")
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
