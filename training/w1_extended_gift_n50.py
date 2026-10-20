"""Train the N50 sample-budget control with the frozen current GIFT recipe.

The shared generator and branch loops retain all phase epochs, batches and
optimizer semantics. Only the predeclared training population is reduced.
This new comparison is GIFT-N50, not the historical validation-selected Lite.
"""
import argparse
from pathlib import Path
import json
import torch

from training import w1_single_gift as native


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    job = json.loads(args.run_config.read_text(encoding="utf-8"))
    if (job.get("method") != "GIFT-N50" or job.get("expected_training_count") != 50
            or job.get("seed") != 20260820 or job.get("generator_seed") != 2026072301
            or job.get("stage") != "formal" or job.get("profile") != native.PROFILE):
        raise ValueError("GIFT-N50 comparison job differs from its declared sole slot")
    if json.dumps(job["config"], sort_keys=True) != json.dumps(native.native_training_config(), sort_keys=True):
        raise ValueError("GIFT-N50 must retain the current fixed GIFT phase/optimizer recipe")
    output = Path(job["output"]).resolve()
    config = native.NativeGIFTConfig()
    try:
        output.mkdir(parents=True, exist_ok=True)
        native.common._atomic_json(output / "STATUS.json", {"status": "running", "method": "GIFT-N50",
                                  "seed": config.branch_seed, "stage": "formal", "profile": native.PROFILE})
        result = native.run_pipeline(job["window"], output, config=config, device=job.get("device", "cuda"),
                                     resume=args.resume, expected_training_count=50)
        native.common._atomic_json(output / "COMPARISON_LABEL.json", {
            "method": "GIFT-N50", "architecture_method": "GIFT", "training_count": 50,
            "seed": config.branch_seed, "generator_seed": config.generator_seed,
            "status": result["status"], "checkpoint": result["paths"]["branch_model"],
            "interpretation": "same stage epochs and optimizer recipe; reduced data and update counts; not original GIFT-Lite"})
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        kind = ("resource" if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                else "numerical" if isinstance(error, (FloatingPointError, ArithmeticError)) else "infrastructure")
        native.common._atomic_json(output / "FAILURE.json", {"method": "GIFT-N50", "seed": config.branch_seed,
            "stage": "formal", "profile": native.PROFILE, "status": "failed", "reason": repr(error),
            "failure_kind": kind})
        raise


if __name__ == "__main__":
    main()
