"""Interface to user-prepared implementations outside the GIFT checkout.

This namespace contains no model, optimizer, source compiler or algorithm patch.
Readers prepare the modules described in docs/EXTERNAL_MANUAL_ADAPTATION.md.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def adapter_root(project_root=PROJECT_ROOT):
    """Resolve the external interface directory, refusing paths inside GIFT."""
    project = Path(project_root).resolve()
    configured = os.environ.get("GIFT_EXTERNAL_ADAPTER_ROOT")
    root = (Path(configured).expanduser() if configured else
            project.parent / "gift-external-runtime").resolve()
    if root == project or project in root.parents:
        raise ValueError("GIFT_EXTERNAL_ADAPTER_ROOT must be outside the GIFT repository")
    package = (root / "adapters").resolve()
    if package == project or project in package.parents:
        raise ValueError("External adapter modules must be outside the GIFT repository")
    return root


EXTERNAL_ADAPTER_ROOT = adapter_root()
__path__ = [str(EXTERNAL_ADAPTER_ROOT / "adapters")]


def source_file(project_root, relative):
    """Locate source provenance after an interface implementation moves.

    Logical provenance keys stay unchanged; their actual file can reside in the
    external directory. This does not execute or modify that file.
    """
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source records require a relative file path")
    project = Path(project_root).resolve()
    external = adapter_root(project) / relative
    if external.is_file():
        if project == external.resolve() or project in external.resolve().parents:
            raise ValueError("External source record resolves inside GIFT")
        return external
    return project / relative


def public_metadata(value):
    """Filter public provenance only when requested by an export operation.

    Configuration-only callers can import the external interface namespace
    without installing or importing the numerical GIFT package first.
    """
    from gift.publication_metadata import public_metadata as clean_metadata
    return clean_metadata(value)


def __getattr__(name):
    """Lazily expose the established external model interface."""
    if name in ("build_model", "load_checkpoint", "source_record"):
        try:
            module = importlib.import_module(".models", __name__)
        except ModuleNotFoundError as error:
            if error.name != "adapters.models":
                raise
            raise ModuleNotFoundError(
                "Prepare external interface modules outside GIFT and set "
                "GIFT_EXTERNAL_ADAPTER_ROOT; see docs/EXTERNAL_MANUAL_ADAPTATION.md"
            ) from error
        return getattr(module, name)
    raise AttributeError(name)


__all__ = ["adapter_root", "source_file", "public_metadata",
           "build_model", "load_checkpoint", "source_record"]
