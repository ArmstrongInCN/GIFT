"""Reevaluate the seven fixed W1 models using the current single-state protocol.

Existing checkpoints are resolved from the external source registry. The aligned
ANIE lock is combined with the unchanged six-method recipes in memory, without
editing their completed training locks or selecting another checkpoint. This
entry performs prediction and evaluation only, from explicitly supplied truth.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from experiments.provenance import external_source
from experiments.additional import w1_extended_evaluate as evaluation


def fixed_sources(native_root, extension_root, aligned_root):
    """Resolve one completed checkpoint per method; use the corrected ANIE pair."""
    native, extra, aligned = map(Path, (native_root, extension_root, aligned_root))
    checkpoints = {
        "GIFT": native / "training/GIFT_seed20260820/branch/model.pt",
        "fno2d": native / "training/fno2d_seed0/terminal_weights.pt",
        "fno3d": native / "training/fno3d_seed0/terminal_weights.pt",
        "anie_ns": aligned / "training/anie_ns_seed7/best.pt",
        "uno": extra / "training/uno_seed0/terminal_weights.pt",
        "unet": extra / "training/unet_seed0/terminal_weights.pt",
        "GIFT-N50": extra / "training/GIFT-N50_seed20260820/branch/model.pt",
    }
    base_lock = evaluation._lock(aligned / "PROTOCOL_LOCK.json")
    extension_lock = copy.deepcopy(evaluation._lock(extra / "EXTENSION_LOCK.json"))
    # The extension's other six slots remain fixed; the final W1 measurement uses
    # the same prescribed ANIE seed/recipe with the documented coordinate fix.
    extension_lock["reused_models"]["anie_ns"] = {
        "checkpoint": str(checkpoints["anie_ns"].resolve(strict=True)),
        "sha256": evaluation.shared.file_sha256(checkpoints["anie_ns"]),
        "source_root": str(aligned.resolve(strict=True)),
    }
    return checkpoints, base_lock, extension_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=external_source("prediction_native"))
    parser.add_argument("--extension-root", type=Path, default=external_source("prediction_extension"))
    parser.add_argument("--aligned-root", type=Path, default=external_source("prediction_aligned"))
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--cohort", choices=("confirmation", "old_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--job-events", type=Path)
    args = parser.parse_args(argv)
    checkpoints, base, extension = fixed_sources(args.native_root, args.extension_root, args.aligned_root)
    return evaluation.evaluate(checkpoints, {}, base_lock=base, extension_lock=extension,
                               data_file=args.data_file, group=args.group, cohort=args.cohort,
                               output=args.output, device=args.device, batch_size=args.batch_size,
                               resume=args.resume, plots=not args.skip_plots, job_events=args.job_events)


if __name__ == "__main__":
    main()
