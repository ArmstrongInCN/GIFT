"""Load hash-pinned external models without vendoring their implementations.

No training, data access, source rewriting on disk, or precision-policy changes.
FNO's upstream scripts execute training at module scope: only their two model
class AST nodes are compiled, with one input-width constant changed beforehand.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = {"fno2d": 466437, "fno3d": 3282457, "uno": 15290641, "unet": 24971551}


def source_record(model: str, external_root: str | Path | None = None) -> dict[str, Any]:
    """Verify every locked file before executing any external model code."""
    if model not in EXPECTED_PARAMETERS:
        raise ValueError(f"unknown model: {model}")
    family = "fno" if model.startswith("fno") else model
    configured = external_root or os.environ.get("GIFT_EXTERNAL_ROOT")
    if not configured:
        raise ValueError("set GIFT_EXTERNAL_ROOT to a directory outside this repository")
    base = Path(configured).expanduser().resolve(strict=True)
    root = (base / family).resolve(strict=True)
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents:
        raise ValueError("third-party sources must be outside the release repository")
    lock = json.loads((PROJECT_ROOT / "external_sources.json").read_text(encoding="utf-8"))
    entry = lock["sources"][family]
    actual = {}
    for relative, expected in entry["files"].items():
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError(f"external source escapes its checkout: {relative}")
        value = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if value != expected["sha256"]:
            raise ValueError(f"external source SHA256 differs: {path}")
        actual[relative] = value
    return {"family": family, "repository": entry["repository"], "commit": entry["commit"],
            "directory": str(root), "source_sha256": actual}


def _fno_class(model: str, root: Path):
    import numpy as np
    import torch

    names = ("SpectralConv2d_fast", "FNO2d") if model == "fno2d" else ("SpectralConv3d", "FNO3d")
    filename = "fourier_2d_time.py" if model == "fno2d" else "fourier_3d.py"
    path = root / filename
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
    if [node.name for node in selected] != list(names):
        raise ValueError("locked FNO class selection differs")
    constructor = next(node for node in selected[1].body
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    matches = [node for node in constructor.body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
               and isinstance(node.targets[0].value, ast.Name)
               and node.targets[0].value.id == "self" and node.targets[0].attr == "fc0"]
    if len(matches) != 1:
        raise ValueError("expected exactly one FNO fc0 constructor assignment")
    call = matches[0].value
    old_width, new_width = (12, 48) if model == "fno2d" else (13, 49)
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "nn"
            and call.func.attr == "Linear" and len(call.args) == 2
            and isinstance(call.args[0], ast.Constant) and call.args[0].value == old_width):
        raise ValueError("locked FNO input projection differs")
    call.args[0] = ast.copy_location(ast.Constant(new_width), call.args[0])
    module = ModuleType(f"gift_external_{model}")
    module.__dict__.update(torch=torch, np=np, nn=torch.nn, F=torch.nn.functional)
    code = compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec")
    exec(code, module.__dict__)
    return getattr(module, names[1])


def _import_file(name: str, path: Path, *, omit_import_seeds: bool = False,
                 only_classes: tuple[str, ...] | None = None) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path.resolve():
            raise RuntimeError(f"module-name collision for {name}; use a separate model process")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = module
    try:
        if only_classes:
            import torch
            parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            selected = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in only_classes]
            if tuple(node.name for node in selected) != only_classes:
                raise ValueError("locked external utility class selection differs")
            module.__dict__["torch"] = torch
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
        elif omit_import_seeds:
            parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            seeds = {ast.dump(ast.parse(text).body[0])
                     for text in ("torch.manual_seed(0)", "np.random.seed(0)")}
            removed = [node for node in parsed.body if ast.dump(node) in seeds]
            if len(removed) != 2:
                raise ValueError("expected exactly two upstream module-level seed calls")
            parsed.body = [node for node in parsed.body if ast.dump(node) not in seeds]
            exec(compile(parsed, str(path), "exec"), module.__dict__)
        else:
            spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    return module


def uno_components(external_root: str | Path | None = None):
    """Return native upstream UNO, Adam and LpLoss; no gradient replacements."""
    record = source_record("uno", external_root)
    root = Path(record["directory"])
    for name in ("integral_operators", "utilities3", "Adam", "navier_stokes_uno2d"):
        _import_file(name, root / f"{name}.py", omit_import_seeds=name == "navier_stokes_uno2d",
                     only_classes=("LpLoss",) if name == "utilities3" else None)
    return (sys.modules["navier_stokes_uno2d"].UNO,
            sys.modules["Adam"].Adam, sys.modules["utilities3"].LpLoss)


def fno_utilities(external_root: str | Path | None = None):
    """Load the two native utility classes from the external pinned source."""
    import numpy as np
    import torch

    record = source_record("fno2d", external_root)
    path = Path(record["directory"]) / "utilities3.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = ("UnitGaussianNormalizer", "LpLoss")
    selected = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
    if [node.name for node in selected] != list(names):
        raise ValueError("locked FNO utility class selection differs")
    module = ModuleType("gift_external_fno_utilities")
    module.__dict__.update(torch=torch, np=np)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.UnitGaussianNormalizer, module.LpLoss


def build_model(model: str, device: str = "cpu", external_root: str | Path | None = None):
    """Construct the formal 46-history model under the caller's numerical profile."""
    record = source_record(model, external_root)
    root = Path(record["directory"])
    if model == "fno2d":
        instance = _fno_class(model, root)(12, 12, 20)
    elif model == "fno3d":
        instance = _fno_class(model, root)(8, 8, 8, 20)
    elif model == "uno":
        upstream, _, _ = uno_components(external_root)
        instance = upstream(in_width=50, width=32, pad=0, factor=0.75)
    else:
        upstream = _import_file("gift_external_unet_229da3e", root / "Baselines" / "U-net.py")
        instance = upstream.U_net(input_channels=46, output_channels=1, kernel_size=3, dropout_rate=0)
    count = sum(parameter.numel() for parameter in instance.parameters())
    if count != EXPECTED_PARAMETERS[model]:
        raise ValueError(f"parameter count differs for {model}: {count}")
    instance.gift_external_source = record
    return instance.to(device)


def load_checkpoint(model: str, checkpoint: str | Path, device: str = "cpu",
                    external_root: str | Path | None = None):
    """Strict state-key/shape loading only; this is not a provenance or resume audit."""
    import torch

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("expected a weights-only-readable dictionary with model_state_dict")
    instance = build_model(model, "cpu", external_root)
    instance.load_state_dict(payload["model_state_dict"], strict=True)
    instance.to(device).eval()
    return instance, payload
