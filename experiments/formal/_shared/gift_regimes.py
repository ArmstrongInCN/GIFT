"""Resolve full-data and reduced-data GIFT without mislabelling checkpoints."""

from dataclasses import replace
from pathlib import Path

import torch

from .common import SEEDS, ProjectPaths, parse_seed_model_specs
from training.checkpoints import digest_file, state_digest
from gift.prediction_cohorts import FOUR_VORTEX, GAUSSIAN, partitions


def add_lite_arguments(parser, *, allow_subset=False):
    """Add explicit generator/branch selection for the 50-trajectory GIFT-Lite comparison."""
    parser.add_argument("--lite-low-model", type=Path, help="50-trajectory GIFT-Lite generator")
    parser.add_argument("--lite-gift-model", action="append", default=[],
                        help="GIFT-Lite branch as SEED=PATH; repeat for all three seeds")
    if allow_subset:
        parser.add_argument("--gift-regimes", nargs="+", choices=("GIFT", "GIFT-Lite"),
                            default=["GIFT", "GIFT-Lite"],
                            help="evaluate the explicitly selected completed training regimes")


def resolve_regimes(args, paths):
    """Default published layout separates both regimes while W3 keeps its model."""
    from gift.paths import checkpoint_root
    artifacts = checkpoint_root(paths.root)
    cohort = paths.prediction_cohort
    split = partitions(cohort)
    selected = getattr(args, "gift_regime", None)
    requested = [selected] if selected is not None else getattr(args, "gift_regimes", ["GIFT", "GIFT-Lite"])
    if not requested or len(set(requested)) != len(requested) or set(requested) - {"GIFT", "GIFT-Lite"}:
        raise ValueError("select each supported training regime at most once")
    if cohort == GAUSSIAN:
        artifacts = artifacts / 's3_gaussian'
        from gift.paths import data_root, data_path
        from gift.gaussian_package import DENSE_FILE, LITE_FILE
        gaussian_files = {
            "GIFT": data_path(data_root(paths.root), "s3.trajectories", DENSE_FILE),
            "GIFT-Lite": data_path(data_root(paths.root), "s3.lite", LITE_FILE),
        }
        gaussian_hashes = {}
    # W3 uses the unchanged root observation file; prediction has its own IDs.
    canonical_standard = paths.standard_n64.parent / "prediction" / paths.standard_n64.name
    if cohort == FOUR_VORTEX and canonical_standard.is_file():
        paths = replace(paths, standard_n64=canonical_standard)
    def data_binding_matches(payload, name):
        """Compare the declared cohort and, when recorded, the Gaussian observation content."""
        data = payload.get("training_data") or {}
        if data.get("cohort") not in (None, cohort):
            return False
        recorded = data.get("sha256")
        if cohort != GAUSSIAN or recorded is None:
            return True
        if name not in gaussian_hashes:
            gaussian_hashes[name] = digest_file(gaussian_files[name])
        return recorded.lower() == gaussian_hashes[name].lower()

    generator_hashes = {}

    def generator_binding_matches(payload, low_payload, low_path):
        """Match a branch to its actual frozen generator weights, supporting legacy file digests."""
        recorded = payload.get("generator_weights_sha256")
        if recorded is not None:
            key = (low_path, "state")
            if key not in generator_hashes:
                generator_hashes[key] = state_digest(low_payload["model_state_dict"])
            return recorded.lower() == generator_hashes[key]
        legacy = payload.get("frozen_generator_sha256")
        if legacy is None:
            return True
        key = (low_path, "file")
        if key not in generator_hashes:
            generator_hashes[key] = digest_file(low_path)
        return legacy.lower() == generator_hashes[key].lower()

    result = {}
    if "GIFT" in requested:
        full_low = (args.low_model if args.low_model is not None
                    else artifacts / "gift_full" / "generator.pt").resolve(strict=True)
        full_specs = args.gift_model or [
            f"{seed}={artifacts / 'gift_full' / f'gift_seed_{seed}.pt'}" for seed in SEEDS]
        full_models = parse_seed_model_specs(paths.root, full_specs)
        full_payload = torch.load(full_low, map_location="cpu", weights_only=True)
        if (full_payload.get("training_regime") != "full_data_prediction"
                or full_payload.get("training_data", {}).get("training_ids") != split['training']):
            raise ValueError("GIFT requires a generator from its declared training population")
        if not data_binding_matches(full_payload, "GIFT"):
            raise ValueError('S3 generator training population differs from the Gaussian input')
        for seed, path in full_models.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("seed") != seed or payload.get("phase") != "terminal_epoch"
                    or payload.get("training_data", {}).get("training_ids") != split['training']
                    or not generator_binding_matches(payload, full_payload, full_low)):
                raise ValueError("GIFT requires completed full-data branches with matching seed and generator")
            if not data_binding_matches(payload, "GIFT"):
                raise ValueError('S3 branch input population differs')
        result["GIFT"] = (replace(paths, low_model=full_low), full_models)
    if "GIFT-Lite" in requested:
        lite_default = ProjectPaths.from_root(paths.root)
        lite_low = (args.lite_low_model if args.lite_low_model is not None
                    else artifacts/'gift_lite/generator.pt' if cohort == GAUSSIAN
                    else lite_default.low_model).resolve(strict=True)
        lite_specs = args.lite_gift_model or ([f"{seed}={artifacts/'gift_lite'/f'gift_seed_{seed}.pt'}"
                                              for seed in SEEDS] if cohort == GAUSSIAN else [])
        lite_models = parse_seed_model_specs(paths.root, lite_specs)
        lite_payload = torch.load(lite_low, map_location="cpu", weights_only=True)
        if lite_payload.get("training_data", {}).get("trajectory_ids") != split['gift_lite_training']:
            raise ValueError("GIFT-Lite requires the preserved 50-trajectory generator")
        if not data_binding_matches(lite_payload, "GIFT-Lite"):
            raise ValueError('S3 Lite generator training population differs from the Gaussian input')
        for seed, path in lite_models.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("seed") != seed or not generator_binding_matches(payload, lite_payload, lite_low):
                raise ValueError("GIFT-Lite branch seed or generator binding differs")
            if not data_binding_matches(payload, "GIFT-Lite"):
                raise ValueError('S3 Lite branch input population differs')
        result["GIFT-Lite"] = (replace(paths, low_model=lite_low), lite_models)
    return result


def extra_regime_files(regimes):
    """Bind the second regime to the same experiment's resume identity."""
    if len(regimes) == 1:
        return {}
    paths, models = regimes["GIFT-Lite"]
    return {"lite_low_model": paths.low_model,
            **{f"lite_gift_{seed}": path for seed, path in models.items()}}


def regime_group(name, seed):
    """Build a stable HDF5/metric group name for a training regime and seed."""
    return f"{name.replace('-', '_')}_seed_{seed}"
