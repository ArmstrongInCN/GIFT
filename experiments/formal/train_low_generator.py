"""Train one clean/noisy GIFT generator or resume only its own saved attempt.

Set GIFT_DATA_ROOT to the separately downloaded input package. --dry-run writes
nothing; --run-training requires a new output directory; --resume requires the
same output directory and unchanged source/data/config/numerical environment.
Published terminal weights are never accepted as resume inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gift.paths import data_root
from training.gift_data import validate_input
from experiments.formal._shared.gift_generator_training import (
    GeneratorTrainingConfig, file_record, inspect_observation_dataset,
    train_fresh_generator, write_json_new,
)

DATASETS = {
    "noise_000": "standard_ns_n64_full_spectrum.h5",
    "noise_001": "m1_parameter_identification/noise_001.h5",
    "noise_010": "m1_parameter_identification/noise_010.h5",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    action.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--condition", choices=tuple(DATASETS), default="noise_000")
    parser.add_argument("--data-profile", choices=("released", "regenerated"), default="released")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026072301)
    parser.add_argument("--phase-steps", type=int, nargs="+", default=(6000, 6000))
    parser.add_argument("--phase-learning-rates", type=float, nargs="+", default=(1e-2, 3e-3))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--affine-refit-interval", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--rank-rcond", type=float)
    parser.add_argument("--allow-nondeterministic", action="store_true",
                        help="diagnostic only; not the formal deterministic protocol")
    args = parser.parse_args(argv)
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    if args.resume and args.output is None:
        parser.error("--resume requires the same explicit --output directory")
    return args


def _configuration(args: argparse.Namespace) -> GeneratorTrainingConfig:
    config = GeneratorTrainingConfig(
        phase_steps=tuple(args.phase_steps),
        phase_learning_rates=tuple(args.phase_learning_rates),
        batch_size=args.batch_size, affine_refit_interval=args.affine_refit_interval,
        validation_interval=args.validation_interval, rank_rcond=args.rank_rcond,
        seed=args.seed, strict_determinism=not args.allow_nondeterministic,
    )
    config.validate()
    return config


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    root = Path(args.project_root).resolve(strict=True)
    if root != PROJECT_ROOT:
        raise ValueError("--project-root must identify the code being executed")
    inputs = data_root(root)
    dataset = inputs / DATASETS[args.condition]
    output = (args.output if args.output is not None else
              root / "runs" / f"gift_low_{args.condition}_seed_{args.seed}").resolve()
    published = root / "artifacts"
    if (output == inputs or inputs in output.parents or output == root or
            output == published or published in output.parents):
        raise ValueError("training output must not overwrite input or published artifacts")
    name = "gift_main.pt" if args.condition == "noise_000" else f"gift_{args.condition}.pt"
    return root, dataset, output, name


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    root, dataset, output, name = _paths(args)
    config = _configuration(args)
    return {
        "schema": "gift.generator.independent-training-plan.v1",
        "status": "dry_run", "condition": args.condition,
        "project_root": str(root), "dataset": str(dataset),
        "dataset_exists": dataset.is_file(), "output": str(output),
        "output_exists": output.exists(), "endpoint": str(output / name),
        "device": args.device, "configuration": config.__dict__,
        "data_profile": args.data_profile,
        "input_provenance": (validate_input(data_root(root), args.condition, args.data_profile, full=False)
                             if dataset.is_file() else None),
        "input_verification": "metadata_only" if dataset.is_file() else "not_performed_missing_input",
        "checkpoint_interval": args.checkpoint_interval,
        "freshness_contract": {"pretrained_model_loaded": False,
                               "fresh_action_loads_checkpoints": False,
                               "resume_requires_same_attempt": True},
        "outputs_written": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root, dataset, output, name = _paths(args)
    config = _configuration(args)
    if args.resume:
        if not (output / "checkpoints" / "LATEST.json").is_file():
            raise FileNotFoundError("no own-attempt saved boundary in --output")
        if any((output / item).exists() for item in
               (name, "training_report.json", "training_history.csv", "manifest.json")):
            raise FileExistsError("completed/partially published outputs cannot be overwritten")
    elif output.exists():
        raise FileExistsError(f"fresh training requires a new directory: {output}")
    dataset_layout = inspect_observation_dataset(dataset.resolve(strict=True), config)
    input_binding = validate_input(data_root(root), args.condition, args.data_profile, full=True)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    report = train_fresh_generator(
        dataset=dataset, checkpoint=output / name,
        report_path=output / "training_report.json", history_path=output / "training_history.csv",
        condition=args.condition, device_name=args.device, config=config,
        project_root=root, entry_script=Path(__file__), resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        input_binding=input_binding,
    )
    manifest = {
        "schema": "gift.generator.independent-attempt.v1", "status": "complete",
        "mode": report["mode"], "condition": args.condition,
        "formal_protocol_defaults_used": config == GeneratorTrainingConfig(),
        "freshness_contract": report["freshness_contract"], "dataset_layout": dataset_layout,
        "files": [file_record(output / item, output)
                  for item in (name, "training_history.csv", "training_report.json")],
    }
    write_json_new(output / "manifest.json", manifest)
    return {"status": "complete", "mode": report["mode"], "output": str(output),
            "checkpoint": file_record(output / name),
            "manifest": file_record(output / "manifest.json"),
            "pretrained_model_loaded": False, "own_resume_state_loaded": args.resume}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_plan(args) if args.dry_run else run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
