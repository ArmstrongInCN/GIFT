"""Same-experiment inference receipts; never a replacement for model training.

Completed numerical calls are stored as checksummed JSON + NumPy arrays (no
pickle). A receipt is committed only after both payloads have been flushed.
An interrupted call is recomputed; earlier completed calls are read back. Each
resume writes a new aggregation directory, preserving incomplete/old reports.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import os
import platform
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from training.checkpoints import digest_file, runtime_identity


def _json_file(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Encode only explicit scientific value types, without executable objects."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be experiment receipts")
        name = f"array_{len(arrays)}"
        arrays[name] = value
        return {"type": "array", "name": name}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if dataclasses.is_dataclass(value) and type(value).__name__ == "RolloutResult":
        return {"type": "rollout", "fields": _encode(vars(value), arrays)}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode(v, arrays) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"unsupported experiment receipt value: {type(value).__name__}")


def _decode(tree: dict[str, Any], arrays: Any) -> Any:
    kind = tree["type"]
    if kind == "array":
        return np.asarray(arrays[tree["name"]])
    if kind == "scalar":
        return tree["value"]
    if kind == "dict":
        return {_decode(k, arrays): _decode(v, arrays) for k, v in tree["items"]}
    if kind in ("list", "tuple"):
        values = [_decode(v, arrays) for v in tree["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "rollout":
        from .gift_runtime import RolloutResult
        return RolloutResult(**_decode(tree["fields"], arrays))
    raise ValueError("unknown experiment receipt value type")


class ExperimentSession:
    """One source/data/config-bound run, locked against concurrent resumptions.

    Call keys are explicit method/seed/grid/cohort identities in the experiment
    entry points. Only deterministic, evaluation-mode numerical work belongs in
    `call`; model initialization and training must not be cached here.
    """

    def __init__(self, directory: Path, identity: dict[str, Any], *, resume: bool = False):
        self.directory = Path(directory).resolve()
        self.identity = json.loads(json.dumps(identity, sort_keys=True, default=str))
        self._lock = None
        self.events: list[dict[str, str]] = []
        self._seen: set[str] = set()
        if resume:
            original = json.loads((self.directory / "EXPERIMENT.json").read_text(encoding="utf-8"))
            if original["identity"] != self.identity:
                raise ValueError("Experiment resume identity differs: source, input, checkpoint or runtime changed")
            self.run_id = original["run_id"]
        else:
            self.run_id = uuid.uuid4().hex
            self.directory.mkdir(parents=True, exist_ok=False)
            _json_file(self.directory / "EXPERIMENT.json", {
                "schema": "gift.experiment-attempt.v1", "identity": self.identity, "run_id": self.run_id,
                "boundary": "completed method/seed/grid/cohort numerical call",
                "training_performed_by_cache": False,
            })
        self.cache = self.directory / ".resume"
        self.cache.mkdir(exist_ok=True)
        self._acquire_lock()
        atexit.register(self.close)
        if resume:
            self.output = self.directory / "resumed_attempts" / uuid.uuid4().hex
            self.output.mkdir(parents=True, exist_ok=False)
        else:
            self.output = self.directory

    def _acquire_lock(self) -> None:
        stream = (self.cache / "writer.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if stream.seek(0, 2) == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError("Another process owns this experiment; stop it before resuming") from error
        self._lock = stream

    def close(self) -> None:
        if self._lock is not None:
            if os.name == "nt":
                import msvcrt
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
            self._lock.close()
            self._lock = None

    def call(self, key: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Read a completed call, or compute and durably commit a new one."""
        if self._lock is None:
            raise RuntimeError("Experiment session is closed")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or key in self._seen:
            raise ValueError("Invalid or repeated numerical call key")
        self._seen.add(key)
        receipt = self.cache / f"{key}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved["key"] != key or saved["identity"] != self.identity or saved["run_id"] != self.run_id:
                raise ValueError("Numerical receipt identity mismatch")
            payloads = {}
            for name in ("tree", "arrays"):
                record = saved[name]
                path = (self.cache / record["file"]).resolve(strict=True)
                if path.parent != self.cache or digest_file(path) != record["sha256"]:
                    raise ValueError("Numerical receipt path/hash mismatch; no silent replacement")
                payloads[name] = path
            tree = json.loads(payloads["tree"].read_text(encoding="utf-8"))
            with np.load(payloads["arrays"], allow_pickle=False) as arrays:
                result = _decode(tree, arrays)
            action = "reused_completed_same_run_call"
        else:
            result = function(*args, **kwargs)
            arrays = {}
            tree = _encode(result, arrays)
            token = uuid.uuid4().hex
            tree_path = self.cache / f"{key}.{token}.json"
            array_path = self.cache / f"{key}.{token}.npz"
            _json_file(tree_path, tree)
            with array_path.open("xb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            pending_receipt = self.cache / f"{key}.{token}.receipt.json"
            _json_file(pending_receipt, {
                "key": key, "identity": self.identity, "run_id": self.run_id,
                "tree": {"file": tree_path.name, "sha256": digest_file(tree_path)},
                "arrays": {"file": array_path.name, "sha256": digest_file(array_path)},
            })
            # Rename is the commit point. A partial temporary receipt is ignored
            # after a crash; it is retained on disk, not automatically deleted.
            pending_receipt.rename(receipt)
            action = "computed_and_saved"
        self.events.append({"key": key, "action": action})
        print(f"[{key}] {action}", flush=True)
        return result

    def finish(self) -> None:
        _json_file(self.output / "continuation.json", {
            "schema": "gift.experiment-continuation.v1", "run_root": str(self.directory), "run_id": self.run_id,
            "aggregation_output": str(self.output), "calls": self.events,
            "not_a_training_checkpoint": True,
        })
        self.close()


def start_experiment(args: Any, paths: Any, experiment: str, output: Path,
                     models: dict[int, Path], *, extra_files: dict[str, Path] | None = None) -> ExperimentSession:
    """Bind exactly the scientific inputs used by this experiment before reuse."""
    names = {
        "M2": ("standard_n64", "fno_test_dt0p02", "low_model", "fno2d_model", "fno3d_model"),
        "M3": ("dense_n64", "cross_resolution", "fno_test_dt0p02", "low_model", "fno2d_model", "fno3d_model"),
        "S1": ("standard_n64", "dense_n64", "cross_resolution", "low_model"),
        "S2": ("standard_n64", "fno_training", "low_model"),
        "S3": ("standard_n64", "dense_n64", "cross_resolution", "low_model"),
    }[experiment]
    files = {name: Path(getattr(paths, name)) for name in names}
    files.update({f"gift_{seed}": path for seed, path in models.items()})
    files.update(extra_files or {})
    inputs = {name: {"path": str(path.resolve(strict=True)), "sha256": digest_file(path)}
              for name, path in files.items()}
    # Bind the implemented runtime dependencies, not unrelated trainers. This
    # allows work on M1/PINN without invalidating a completed M2 inference unit.
    entry = {"M2": "m2_recursive_prediction", "M3": "m3_cross_resolution",
             "S1": "s1_high_frequency_branch", "S2": "s2_recursive_local_correction",
             "S3": "s3_seed_stability"}[experiment]
    relative_sources = [f"experiments/formal/{entry}/run.py", "training/checkpoints.py"]
    relative_sources += [f"experiments/formal/_shared/{name}.py"
                         for name in ("common", "gift_runtime", "high_frequency", "resume")]
    if experiment == "S3":
        relative_sources += ["experiments/formal/_shared/equation.py",
                             "experiments/formal/train_gift_branches.py"]
    if experiment in ("M2", "M3"):
        relative_sources += ["experiments/formal/_shared/fno_runtime.py",
                             "training/baseline_control.py", "adapters/models.py",
                             "adapters/__init__.py", "src/even_full_spectrum_ns.py"]
    if experiment == "M2":
        relative_sources += ["adapters/prediction.py"]
    source_paths = [paths.root / name for name in relative_sources]
    # Package initializers are executable too, even when currently empty.
    source_paths.extend(path for path in (
        paths.root / "experiments/__init__.py", paths.root / "experiments/formal/__init__.py",
        paths.root / "experiments/formal/_shared/__init__.py",
        paths.root / f"experiments/formal/{entry}/__init__.py",
        paths.root / "training/__init__.py") if path.is_file())
    source_paths.extend(sorted((paths.root / "src" / "gift").rglob("*.py")))
    source = {path.relative_to(paths.root).as_posix(): digest_file(path) for path in source_paths}
    external = {}
    if experiment in ("M2", "M3"):
        from adapters.models import source_record
        for method in (("fno2d", "fno3d", "uno", "unet") if experiment == "M2" else ("fno2d", "fno3d")):
            external[method] = source_record(method)
    config = {key: value for key, value in vars(args).items()
              if key not in ("output", "resume", "skip_plots", "project_root")}
    runtime = runtime_identity()
    runtime.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    runtime.update(os=platform.platform(), machine=platform.machine(),
                   processor=platform.processor(), torch_build=torch.__config__.show())
    if torch.device(args.device).type == "cuda":
        properties = torch.cuda.get_device_properties(torch.device(args.device))
        runtime["selected_gpu"] = {"name": properties.name, "major": properties.major,
            "minor": properties.minor, "total_memory": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
            "uuid": str(getattr(properties, "uuid", "not_available"))}
    session = ExperimentSession(output, {"experiment": experiment, "inputs": inputs,
        "source": source, "external": external, "config": config, "runtime": runtime}, resume=args.resume)
    print(f"Experiment aggregation output: {session.output}", flush=True)
    return session
