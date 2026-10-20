"""Locate immutable models after a checkout or model bundle changes machines.

Recorded model specifications remain evidence. Only source-directory locations
are overridden for construction; architecture, coefficients and tensors retain
their recorded values. GIFT's consumer verifies the selected generator's tensor
digest before combining it with the branch.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path


def runtime_model_spec(spec):
    """Copy a model specification and apply the user's current upstream root."""
    result = copy.deepcopy(spec)
    method = result.get("method")
    variable = "GIFT_ANIE_SOURCE_ROOT" if method in {"anie_ns", "anie_f"} else "GIFT_EXTERNAL_ROOT"
    if method in {"anie_ns", "anie_f", "fno2d", "fno3d", "uno", "unet"} and os.environ.get(variable):
        root = Path(os.environ[variable]).expanduser().resolve(strict=True)
        project = Path(__file__).resolve().parents[1]
        if not root.is_dir() or root == project or project in root.parents:
            raise ValueError(f"{variable} must name an upstream directory outside GIFT")
        result["source_root"] = str(root)
    return result


def generator_checkpoint(branch_path, payload):
    """Find a sibling generator first, then the recorded absolute/relative path.

    A relocated bundle retains branch/model.pt and generator/model.pt under the
    same training slot. An existing sibling is authoritative and must pass the
    caller's tensor-binding check; a wrong sibling never falls back silently.
    """
    branch_path = Path(branch_path).resolve(strict=True)
    sibling = branch_path.parent.parent / "generator" / "model.pt"
    if sibling.is_file():
        return sibling
    recorded = Path(payload["model_spec"]["generator_checkpoint"]).expanduser()
    if not recorded.is_absolute():
        recorded = branch_path.parent / recorded
    if not recorded.is_file():
        raise FileNotFoundError(
            "Missing paired GIFT generator; retain training/<slot>/generator/model.pt "
            "beside branch/model.pt when transferring a model bundle")
    return recorded.resolve(strict=True)
