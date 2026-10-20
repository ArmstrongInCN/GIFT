"""Same-experiment inference receipts; never a replacement for model training.

Completed numerical calls are stored as checksummed JSON + HDF5 arrays (legacy NumPy receipts remain readable; no
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
import h5py
import torch

from training.checkpoints import digest_file, runtime_identity
from adapters import source_file


def _resume_signature(identity):
    """Content and scientific settings identify reusable numerical work."""
    if "inputs" not in identity:
        return {key: value for key, value in identity.items() if key not in ("source", "runtime")}
    operational = {"device", "gift_execution", "gift_model", "lite_gift_model"}
    config = {key: value for key, value in identity.get("config", {}).items()
              if key not in operational and not key.endswith(("_model", "_batch_size"))
              and key != "batch_size"}
    return {"experiment": identity["experiment"], "config": config,
            "inputs": {name: {key: value for key, value in record.items() if key != "path"}
                       for name, record in identity["inputs"].items()}}


def _json_file(path: Path, value: Any) -> None:
    """Create and fsync one JSON payload before it can become part of a committed receipt."""
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
    """Reconstruct supported numerical containers from JSON tags and HDF5/legacy NPZ arrays."""
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
    """One input/config-bound run, locked against concurrent resumptions.

    Call keys are explicit method/seed/grid/cohort identities in the experiment
    entry points. Only deterministic, evaluation-mode numerical work belongs in
    `call`; model initialization and training must not be cached here.
    """

    def __init__(self, directory: Path, identity: dict[str, Any], *, resume: bool = False):
        """Open one run and its single-writer cache, comparing scientific inputs on resume.

        Completed calls stay under the original run; a resume creates a new aggregation
        output so earlier incomplete reports remain available for inspection.
        """
        self.directory = Path(directory).resolve()
        self.identity = json.loads(json.dumps(identity, sort_keys=True, default=str))
        self.signature = _resume_signature(self.identity)
        self._calls = {}
        self._lock = None
        self.events: list[dict[str, str]] = []
        self._seen: set[str] = set()
        if resume:
            original = json.loads((self.directory / "EXPERIMENT.json").read_text(encoding="utf-8"))
            if _resume_signature(original["identity"]) != self.signature:
                raise ValueError("Experiment resume identity differs: input or scientific configuration changed")
            self.run_id = original["run_id"]
        else:
            self.run_id = uuid.uuid4().hex
            self.directory.mkdir(parents=True, exist_ok=False)
            _json_file(self.directory / "EXPERIMENT.json", {
                "schema": "gift.experiment-attempt.v2", "identity": self.identity, "run_id": self.run_id,
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
        """Acquire a nonblocking process lock so two resumptions cannot commit the same call."""
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
        """Release the writer lock; safe to call again from the exit handler."""
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
            saved_signature = saved.get("signature")
            if saved_signature is None:
                saved_signature = _resume_signature(saved["identity"])
            if saved["key"] != key or saved_signature != self.signature or saved["run_id"] != self.run_id:
                raise ValueError("Numerical receipt identity mismatch")
            payloads = {}
            for name in ("tree", "arrays"):
                record = saved[name]
                base = self.directory if saved.get("schema") == "gift.experiment-call.v2" else self.cache
                path = (base / record["file"]).resolve(strict=True)
                if base not in path.parents or digest_file(path) != record["sha256"]:
                    raise ValueError("Numerical receipt path/hash mismatch; no silent replacement")
                payloads[name] = path
            tree = json.loads(payloads["tree"].read_text(encoding="utf-8"))
            if payloads["arrays"].suffix == ".h5":
                with h5py.File(payloads["arrays"], "r") as arrays:
                    result = _decode(tree, arrays)
            else:
                with np.load(payloads["arrays"], allow_pickle=False) as arrays:
                    result = _decode(tree, arrays)
            action = "reused_completed_same_run_call"
        else:
            result = function(*args, **kwargs)
            arrays = {}
            tree = _encode(result, arrays)
            token = uuid.uuid4().hex
            tree_path = self.cache / f"{key}.{token}.json"
            units = self.directory / "raw" / "units"
            units.mkdir(parents=True, exist_ok=True)
            array_path = units / f"{key}.{token}.h5"
            _json_file(tree_path, tree)
            with h5py.File(array_path, "x") as handle:
                for name, value in arrays.items():
                    handle.create_dataset(name, data=value,
                                          **({"compression": "gzip"} if value.ndim and value.size else {}))
                handle.flush()
            with array_path.open("r+b") as stream:
                os.fsync(stream.fileno())
            pending_receipt = self.cache / f"{key}.{token}.receipt.json"
            saved = {"schema": "gift.experiment-call.v2", "key": key,
                "signature": self.signature, "run_id": self.run_id,
                "provenance": {"source": self.identity.get("source", {}),
                               "runtime": self.identity.get("runtime", {})},
                "tree": {"file": tree_path.relative_to(self.directory).as_posix(), "sha256": digest_file(tree_path)},
                "arrays": {"file": array_path.relative_to(self.directory).as_posix(), "sha256": digest_file(array_path)}}
            if tree["type"] == "rollout":
                saved["prediction_array"] = next(value["name"] for name, value in tree["fields"]["items"]
                                                 if name.get("value") == "prediction")
            elif tree["type"] == "tuple" and tree["items"] and tree["items"][0]["type"] == "array":
                saved["prediction_array"] = tree["items"][0]["name"]
            _json_file(pending_receipt, saved)
            # Rename is the commit point. A partial temporary receipt is ignored
            # after a crash; it is retained on disk, not automatically deleted.
            pending_receipt.rename(receipt)
            action = "computed_and_saved"
        self._calls[key] = saved
        self.events.append({"key": key, "action": action})
        print(f"[{key}] {action}", flush=True)
        return result

    def save_prediction(self, group, key, prediction):
        """Reference the committed prediction array instead of saving it twice."""
        saved = self._calls[key]
        name = saved.get("prediction_array")
        if name is None:
            # Legacy NumPy receipts remain readable without altering their files.
            group.create_dataset("prediction", data=prediction, compression="gzip", compression_opts=4)
            return
        source = self.directory / saved["arrays"]["file"]
        destination = Path(group.file.filename).parent / "units" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination and not destination.exists():
            os.link(source, destination)
        relative = os.path.relpath(destination, Path(group.file.filename).parent).replace("\\", "/")
        group["prediction"] = h5py.ExternalLink(relative, name)

    def finish(self) -> None:
        """Write the aggregation's continuation record and release its numerical-call lock."""
        _json_file(self.output / "continuation.json", {
            "schema": "gift.experiment-continuation.v1", "run_root": str(self.directory), "run_id": self.run_id,
            "aggregation_output": str(self.output), "calls": self.events,
            "current_identity": self.identity,
            "not_a_training_checkpoint": True,
        })
        self.close()


def start_experiment(args: Any, paths: Any, experiment: str, output: Path,
                     models: dict[int, Path], *, extra_files: dict[str, Path] | None = None) -> ExperimentSession:
    """Bind exactly the scientific inputs used by this experiment before reuse."""
    names = {
        "W1": ("standard_n64", "fno_test_dt0p02", "low_model", "fno2d_model", "fno3d_model"),
        "S3": ("standard_n64", "fno_test_dt0p02", "low_model", "fno2d_model", "fno3d_model"),
        "W2": ("dense_n64", "cross_resolution", "fno_test_dt0p02", "low_model", "fno2d_model", "fno3d_model"),
        "S1": ("standard_n64", "dense_n64", "cross_resolution", "low_model"),
        "S2": ("standard_n64", "low_model"),
    }[experiment]
    files = {name: Path(getattr(paths, name)) for name in names}
    files.update({f"gift_{seed}": path for seed, path in models.items()})
    files.update(extra_files or {})
    hashes = {}
    inputs = {}
    for name, path in files.items():
        path = path.resolve(strict=True)
        if path not in hashes:
            hashes[path] = digest_file(path)
        inputs[name] = {"path": str(path), "sha256": hashes[path]}
    # Record the sources used by this numerical route for provenance. Reuse is
    # decided by _resume_signature from scientific inputs/settings; comments,
    # source revisions and runtime descriptions do not invalidate receipts.
    entry = {"W1": "w1_recursive_prediction", "W2": "w2_cross_resolution",
             "S1": "s1_high_frequency_branch", "S2": "s2_recursive_local_correction",
             "S3": "s3_initial_distribution"}[experiment]
    relative_sources = [f"experiments/formal/{entry}/run.py", "training/checkpoints.py"]
    relative_sources += [f"experiments/formal/_shared/{name}.py"
                         for name in ("common", "gift_runtime", "high_frequency", "resume")]
    relative_sources += ["experiments/formal/_shared/prediction_data.py",
                         "experiments/formal/_shared/gift_regimes.py"]
    if experiment in ("W1", "W2", "S3"):
        relative_sources += ["experiments/formal/_shared/fno_runtime.py",
                             "training/baseline_control.py", "adapters/models.py",
                             "training/budgets.py", "training/weight_files.py",
                             "adapters/__init__.py", "src/even_full_spectrum_ns.py"]
    if experiment in ("W1", "S3"):
        relative_sources += ["adapters/prediction.py"]
    if experiment == "S3":
        relative_sources += ["experiments/formal/w1_recursive_prediction/run.py"]
    source = {name: digest_file(source_file(paths.root, name)) for name in relative_sources}
    # Package initializers are executable too, even when currently empty.
    source_paths = [path for path in (
        paths.root / "experiments/__init__.py", paths.root / "experiments/formal/__init__.py",
        paths.root / "experiments/formal/_shared/__init__.py",
        paths.root / f"experiments/formal/{entry}/__init__.py",
        paths.root / "training/__init__.py") if path.is_file()]
    source_paths.extend(sorted((paths.root / "src" / "gift").rglob("*.py")))
    source.update({path.relative_to(paths.root).as_posix(): digest_file(path) for path in source_paths})
    external = {}
    if experiment in ("W1", "W2", "S3"):
        from adapters.models import source_record
        for method in (("fno2d", "fno3d", "uno", "unet") if experiment in ("W1", "S3") else ("fno2d", "fno3d")):
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
