"""Bind SVG exports to completed numerical evidence, without loading models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree

import numpy as np


def error_axis(values, default_limits: tuple[float, float]) -> dict:
    """Retain the design range when possible; never floor or hide zero errors."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("plotted errors must be finite and nonnegative")
    low, high = default_limits
    observed_min, observed_max = float(values.min()), float(values.max())
    if observed_min == 0:
        return {"scale": "symlog", "limits": (0.0, max(high, 1.2 * observed_max)),
                "linthresh": low, "expanded": True}
    limits = (observed_min / 1.2 if observed_min < low else low, max(high, observed_max * 1.2)
              if observed_max > high else high)
    return {"scale": "log", "limits": limits, "expanded": limits != default_limits}


def symmetric_limit(observed: float, preferred: float) -> float:
    """One shared colour scale per quantity, rounded outward only when needed."""
    if not np.isfinite(observed) or observed < 0 or preferred <= 0:
        raise ValueError("invalid field range")
    return max(preferred, float(np.ceil(observed)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_record(path: Path, root: Path) -> dict:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def bind_result(root: Path, experiment: str, required: tuple[str, ...]) -> dict:
    """Verify files against the numerical commit or the finished run manifest.

    This accepts both pre-plot numerical completion and fully completed results.
    Neither a plain CSV nor a folder name is proof of a completed experiment.
    """
    root = root.resolve(strict=True)
    report_path = root / "numeric_report.json"
    if not report_path.is_file():
        report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("experiment") != experiment or report.get("status") not in {
        "complete", "numerical_complete", "numerical_complete_plot_failed"
    }:
        raise ValueError("figure input is not a completed numerical experiment")
    records = report.get("numeric_files")
    manifest_path = root / "manifest.json"
    if records is None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        records = manifest["files"]
    verified = []
    for name in required:
        path = (root / name).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("figure source escapes the result directory")
        matches = [row for row in records if row["path"] == name]
        actual = file_record(path, root)
        if len(matches) != 1 or actual["bytes"] != matches[0]["bytes"] or (
            actual["sha256"] != matches[0]["sha256"].upper()
        ):
            raise ValueError(f"figure input differs from completed experiment: {name}")
        verified.append(actual)
    # Preserve the model/data bindings without publishing local working paths.
    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()
                    if key not in {"path", "record", "root", "output_directory"}}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value
    return {"experiment": experiment, "report_sha256": sha256(report_path),
            "sources": verified, "experiment_inputs": portable(report.get("inputs", {}))}


def verify_svg(path: Path, *, editable_text: bool = True) -> dict:
    root = ElementTree.parse(path).getroot()
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("not an SVG document")
    tags = [element.tag.rsplit("}", 1)[-1] for element in root.iter()]
    if "image" in tags or "script" in tags or "foreignObject" in tags:
        raise ValueError("SVG contains a raster image or active/non-vector content")
    if editable_text and "text" not in tags:
        raise ValueError("SVG has no editable text")
    return {"editable_text": "text" in tags, "embedded_raster_images": 0,
            "width": root.get("width"), "height": root.get("height")}


def _crisp_vector_cells(path: Path) -> int:
    """Avoid viewer-dependent white seams between adjacent vector grid cells.

    Only the rendering hint of Matplotlib QuadMesh groups changes. Cell paths,
    fill colours, coordinates, text and scientific arrays remain untouched.
    """
    original = path.read_bytes()
    modified, count = re.subn(rb'<g id="QuadMesh_[0-9]+">',
        lambda match: match[0][:-1] + b' shape-rendering="crispEdges">', original)
    if count:
        path.write_bytes(modified)
    return count


def finish_figures(output: Path, evidence: dict, renderer: Path, details: dict) -> None:
    """Commit exact plotted sources and SVGs. Existing outputs are never replaced."""
    if (output / "figure_manifest.json").exists():
        raise FileExistsError("Figure package is already committed; choose a new output")
    images = sorted(output.rglob("*.svg"))
    if not images:
        raise ValueError("no SVGs were exported")
    cell_groups = {path.relative_to(output).as_posix(): _crisp_vector_cells(path) for path in images}
    qa = {path.relative_to(output).as_posix(): verify_svg(
        path, editable_text="panels" not in path.relative_to(output).parts
    ) for path in images}
    files = [file_record(path, output) for path in sorted(output.rglob("*")) if path.is_file()]
    payload = {"status": "complete", "evidence": evidence, "details": details,
               "renderer": {"path": renderer.name, "sha256": sha256(renderer)},
               "shared_renderer_sha256": sha256(Path(__file__)),
               "vector_cell_rendering": {"policy": "crispEdges on QuadMesh groups only", "groups_per_svg": cell_groups},
               "svg_checks": qa, "files": files}
    with (output / "figure_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
