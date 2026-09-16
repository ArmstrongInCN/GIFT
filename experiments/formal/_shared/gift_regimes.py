"""Resolve full-data and reduced-data GIFT without mislabelling checkpoints."""

from dataclasses import replace
from pathlib import Path

import torch

from .common import SEEDS, ProjectPaths, parse_seed_model_specs
from training.checkpoints import digest_file
from gift.prediction_cohorts import FOUR_VORTEX, GAUSSIAN, partitions


def add_lite_arguments(parser, *, allow_subset=False):
    parser.add_argument("--lite-low-model", type=Path, help="50-trajectory GIFT-Lite generator")
    parser.add_argument("--lite-gift-model", action="append", default=[],
                        help="GIFT-Lite branch as SEED=PATH; repeat for all three seeds")
    if allow_subset:
        parser.add_argument("--gift-regimes", nargs="+", choices=("GIFT", "GIFT-Lite"),
                            default=["GIFT", "GIFT-Lite"],
                            help="evaluate the explicitly selected completed training regimes")


def resolve_regimes(args, paths):
    """Default published layout separates both regimes while M1 keeps its model."""
    from gift.paths import checkpoint_root
    artifacts = checkpoint_root(paths.root)
    cohort = getattr(paths, 'prediction_cohort', FOUR_VORTEX)
    split = partitions(cohort)
    if cohort == GAUSSIAN:
        artifacts = artifacts / 's4_gaussian'
        from gift.paths import data_root
        from gift.gaussian_package import DENSE_FILE, LITE_FILE
        gaussian_hashes = {"GIFT": digest_file(data_root(paths.root) / DENSE_FILE),
                           "GIFT-Lite": digest_file(data_root(paths.root) / LITE_FILE)}
    # M1 uses the unchanged root observation file; prediction has its own IDs.
    canonical_standard = paths.standard_n64.parent / "prediction" / paths.standard_n64.name
    if cohort == FOUR_VORTEX and canonical_standard.is_file():
        paths = replace(paths, standard_n64=canonical_standard)
    selected = getattr(args, "gift_regime", None)
    requested = [selected] if selected is not None else getattr(args, "gift_regimes", ["GIFT", "GIFT-Lite"])
    if not requested or len(set(requested)) != len(requested) or set(requested) - {"GIFT", "GIFT-Lite"}:
        raise ValueError("select each supported training regime at most once")
    result = {}
    if "GIFT" in requested:
        full_low = (args.low_model if args.low_model is not None
                    else artifacts / "gift_full" / "generator.pt").resolve(strict=True)
        full_specs = args.gift_model or [
            f"{seed}={artifacts / 'gift_full' / f'gift_seed_{seed}.pt'}" for seed in SEEDS]
        full_models = parse_seed_model_specs(paths.root, full_specs)
        full_payload = torch.load(full_low, map_location="cpu", weights_only=True)
        if (full_payload.get("training_regime") != "full_data_prediction"
                or full_payload.get("training_cost", {}).get("epochs") != 500
                or full_payload.get("training_cost", {}).get("optimizer_updates") != 31500
                or full_payload.get("training_data", {}).get("training_ids") != split['training']):
            raise ValueError("GIFT requires the completed 1,000-trajectory, 500-epoch generator")
        expected_low = digest_file(full_low).lower()
        if cohort == GAUSSIAN and (
                full_payload.get('training_data', {}).get('sha256') != gaussian_hashes['GIFT']
                or full_payload.get('fresh_training', {}).get('pretrained_model_loaded') is not False):
            raise ValueError('S4 generator must be trained fresh on this Gaussian population')
        for seed, path in full_models.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("seed") != seed or payload.get("phase") != "terminal_epoch"
                    or payload.get("training_cost", {}).get("epochs") != 500
                    or payload.get("training_cost", {}).get("optimizer_updates") != 45200
                    or payload.get("training_data", {}).get("training_ids") != split['training']
                    or payload.get("frozen_generator_sha256", "").lower() != expected_low):
                raise ValueError("GIFT requires completed full-data branches with matching seed and generator")
            if cohort == GAUSSIAN and payload.get('training_data', {}).get('sha256') != gaussian_hashes['GIFT']:
                raise ValueError('S4 branch input population differs')
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
        expected_low = digest_file(lite_low).lower()
        if cohort == GAUSSIAN and (
                lite_payload.get('training_data', {}).get('sha256', '').lower() != gaussian_hashes['GIFT-Lite']
                or lite_payload.get('fresh_training', {}).get('pretrained_model_loaded') is not False):
            raise ValueError('S4 Lite generator must be trained fresh on this Gaussian population')
        for seed, path in lite_models.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("seed") != seed or payload.get("frozen_generator_sha256", "").lower() != expected_low:
                raise ValueError("GIFT-Lite branch seed or generator binding differs")
            if cohort == GAUSSIAN and payload.get('training_data', {}).get('sha256', '').lower() != gaussian_hashes['GIFT-Lite']:
                raise ValueError('S4 Lite branch input population differs')
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
    return f"{name.replace('-', '_')}_seed_{seed}"
