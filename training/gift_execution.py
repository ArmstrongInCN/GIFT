"""Shared execution policy for every independent GIFT training entry point."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

EXECUTION_CHOICES = ("auto", "eager", "cuda-graph")
EXECUTION_HELP = (
    "auto uses CUDA graphs for fresh GPU training and eager on CPU; resume keeps "
    "the journal's recorded backend. Explicit eager is the numerical reference."
)
# Capture depends on these tensor operations even when a caller uses only the
# generator phase. Binding the complete shared engine avoids unrecorded edits.
EXECUTION_SOURCES = (
    "training/gift_execution.py", "training/gift_acceleration.py",
    "training/gift_branch_kernels.py", "src/gift/execution.py",
    "experiments/formal/train_gift_branches.py",
    "experiments/formal/_shared/high_frequency.py",
    "experiments/formal/_shared/gift_runtime.py",
    "experiments/formal/_shared/common.py",
)


def configure_training_runtime(*, strict):
    """Fix kernel selection before CUDA work without changing precision or RNG.

    A seed alone does not make cuDNN backward deterministic. Use the same
    settings for eager and graph training, including the reduced-data branch.
    The journal records these flags; incompatible old runtime states are not
    silently resumed. Strictness remains specific to each scientific trainer.
    """
    # Force a deterministic cuBLAS workspace and disable benchmark/cudnn
    # nondeterminism; a seed alone does not make backward passes reproducible.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(strict, warn_only=not strict)


def resolve_execution(requested, device, *, resume=False, checkpoint_directory=None):
    """Select an execution path, never weaken the journal's identity checks.

    Automatic resume inherits only the backend; source/data/runtime are still
    independently checked by CheckpointStore. An explicit backend mismatch is
    rejected there. Capture failures are errors, not silent numerical fallbacks.
    """
    if requested not in EXECUTION_CHOICES:
        raise ValueError("unknown GIFT training execution backend")
    device = torch.device(device)
    # On auto-resume, inherit the backend recorded in the journal rather than
    # re-deriving it; the source/data/runtime checks still run independently.
    if requested == "auto" and resume:
        if checkpoint_directory is None:
            raise ValueError("automatic resume needs its own checkpoint directory")
        attempt = json.loads((Path(checkpoint_directory)/"ATTEMPT.json").read_text(encoding="utf-8"))
        requested = attempt["identity"].get("execution", "eager")
        if requested not in ("eager", "cuda-graph"):
            raise ValueError("saved training backend is invalid")
    if device.type != "cuda":
        return "eager"
    return "cuda-graph" if requested == "auto" else requested
