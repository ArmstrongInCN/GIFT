"""Explicit paths for independently distributed inputs and generated outputs.

Datasets and upstream algorithm checkouts must stay outside the Git repository.
No environment variable is silently rewritten, and this module does no I/O
other than resolving paths. Consumers open datasets read-only.
"""

from __future__ import annotations

import os
from pathlib import Path


def _external(name: str, project_root: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        raise FileNotFoundError(
            f"Set {name} to the separately downloaded directory; "
            "see docs/SETUP.md. Inputs are not bundled with the code."
        )
    path = Path(value).expanduser().resolve()
    root = Path(project_root).resolve()
    if path == root or root in path.parents:
        raise ValueError(f"{name} must point outside the Git repository: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return path


def data_root(project_root: Path) -> Path:
    """The Zenodo package root, containing standard_ns_n64_full_spectrum.h5."""
    return _external("GIFT_DATA_ROOT", project_root)


def external_root(project_root: Path) -> Path:
    """User-managed, pinned third-party checkouts; never distributed by GIFT."""
    return _external("GIFT_EXTERNAL_ROOT", project_root)


def checkpoint_root(project_root: Path) -> Path:
    """Published artifacts by default; an explicit fresh model bundle is allowed."""
    value = os.environ.get("GIFT_CHECKPOINT_ROOT")
    return (Path(value).expanduser() if value else Path(project_root) / "artifacts").resolve()
