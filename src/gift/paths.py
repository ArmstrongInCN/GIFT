"""Resolve explicit input paths; consumers open files and writers protect outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _configured_root(name: str) -> Path:
    """Resolve a required input-directory environment variable for a reader."""
    value = os.environ.get(name)
    if not value:
        raise FileNotFoundError(
            f"Set {name} to the input directory; see docs/SETUP.md."
        )
    return Path(value).expanduser().resolve()


def data_root(project_root: Path) -> Path:
    """Return the explicitly selected observation package root."""
    return _configured_root("GIFT_DATA_ROOT")


def package_member(root: Path, relative: str) -> Path:
    """Resolve an existing local package member, rejecting escapes and links.

    Catalogue paths use forward slashes and remain valid after a package is
    copied to another machine. Both files and directories may be selected.
    """
    root = Path(root).resolve(strict=True)
    if (not isinstance(relative, str) or "\\" in relative or ":" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("Invalid package-relative path")
    member = root / relative
    for item in (member, *member.parents):
        if item == root:
            break
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError("Linked package paths are not accepted")
    resolved = member.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError("Package member escapes its root")
    return resolved


def dataset_catalog(root: Path) -> dict | None:
    """Read an optional formal catalogue; malformed catalogues never select legacy data."""
    root = Path(root).resolve()
    if not (root / "dataset.json").exists() and not (root / "dataset.json").is_symlink():
        return None
    document = json.loads(package_member(root, "dataset.json").read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or document.get("schema") != "gift.dataset.v1"
            or not isinstance(document.get("roles"), dict)
            or not document["roles"]):
        raise ValueError("Expected a gift.dataset.v1 catalogue with input roles")
    return document


def data_path(root: Path, role: str, legacy: str | None = None) -> Path:
    """Resolve a named formal input, or a historical path when no catalogue exists.

    Formal role paths must exist inside the package. Legacy paths are constructed
    without opening unused inputs, preserving independently generated S3 roots.
    """
    root = Path(root).resolve()
    document = dataset_catalog(root)
    if document is None:
        if legacy is None:
            raise ValueError("This input role requires dataset.json: " + role)
        return root / legacy
    if role not in document["roles"]:
        raise ValueError("Missing dataset input role: " + role)
    return package_member(root, document["roles"][role])


def default_data_path(role: str, legacy: str) -> Path:
    """Select a future local queue's input without changing any existing run lock.

    An explicit GIFT_DATA_ROOT takes precedence over the experiment machine's
    historical default. A catalogue-free plan may name files not yet prepared.
    """
    root = Path(os.environ.get("GIFT_DATA_ROOT", r"D:\GIFT\GIFT-dataset")).expanduser()
    return data_path(root, role, legacy)


def external_root(project_root: Path) -> Path:
    """User-selected parent directory for upstream checkouts."""
    return _configured_root("GIFT_EXTERNAL_ROOT")


def checkpoint_root(project_root: Path) -> Path:
    """Published artifacts by default; an explicit fresh model bundle is allowed."""
    value = os.environ.get("GIFT_CHECKPOINT_ROOT")
    return (Path(value).expanduser() if value else Path(project_root) / "artifacts").resolve()
