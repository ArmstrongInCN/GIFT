"""Read the canonical prediction input schema; package audits are separate."""
import hashlib
from pathlib import Path

import h5py
import numpy as np


def _digest(path):
    """Hash a file in bounded blocks for callers performing a package audit."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _member(root, relative):
    """Resolve a slash-separated package member without escaping ``root``.

    ``root`` is the resolved package directory; absolute paths, parent steps
    and backslash/drive notation are rejected before accessing the member.
    """
    parts = relative.split("/")
    if not relative or "\\" in relative or ":" in relative or any(p in ("", ".", "..") for p in parts):
        raise ValueError("unsafe data-package path")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError("data-package member escapes its root")
    return path


def training_input(base, path, *, full):
    """Validate canonical training/validation observations and return their IDs.

    Groups contain float32 ``[trajectories, 501, 64, 64]`` vorticity, canonical
    ID vectors, and times ``0..10`` at spacing ``0.02``. ``full`` streams each
    field row to check finite values; schema-only checks avoid reading
    the entire field. ``base`` is retained in the reader interface but does
    not constrain ``path``. The report does not certify a published archive.
    """
    path = Path(path).resolve(strict=True)
    groups = {"training": list(range(1000)), "validation": list(range(1000, 1020))}
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("trajectory_id_scheme") != "canonical":
            raise ValueError("canonical trajectory identity scheme required")
        fixture_only = bool(handle.attrs.get("fixture_only", False))
        for group, ids in groups.items():
            field = handle[f"{group}/vorticity"]
            if (handle[f"{group}/trajectory_index"][:].tolist() != ids
                    or field.shape != (len(ids), 501, 64, 64)
                    or field.dtype != np.dtype("float32")
                    or not np.allclose(handle[f"{group}/time"][:], np.arange(501) * .02, rtol=0, atol=2e-12)):
                raise ValueError("canonical population, shape, dtype or time differs")
            if full:
                for row in range(len(ids)):
                    if not np.isfinite(field[row]).all():
                        raise ValueError("nonfinite canonical observations")
    return dict(profile="canonical", file=str(path), fixture_only=fixture_only, trajectory_ids=groups["training"],
                validation_ids=groups["validation"], full_input_checks_performed=full,
                published_dataset_identity=False, numerical_reproduction_verified=False)
