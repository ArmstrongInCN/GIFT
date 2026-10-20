"""Small, portable, same-run checkpoint journal with complete random state.

Each boundary is immutable. Only the tiny LATEST.json pointer is atomically
replaced after the new checkpoint has been flushed. A crash can lose progress
since the last saved boundary, never silently turn a terminal weight file into
an optimizer resume state. No checkpoint or previous output is deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.configuration import validate_resume, validate_training_state


def digest_file(path: Path) -> str:
    """Stream a file in 8 MiB blocks and return its hexadecimal SHA-256.

    Callers use this for explicit integrity records and small journal pointers;
    ordinary model compatibility is checked from loaded state, not this digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity() -> dict[str, Any]:
    """Record the numerical environment as provenance, independently of resume."""
    return {
        "python": platform.python_version(), "numpy": str(np.__version__),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        # Reduction scheduling can change training even with identical seeds.
        # Record actual Torch counts as well as the launch environment: changing
        # an environment variable after import need not change an existing pool.
        "threading": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "OMP_DYNAMIC", "MKL_DYNAMIC",
            )},
        },
    }


def capture_rng() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU and the active CUDA device generator.

    Convert NumPy's uint32 key array to primitive values for weights-only
    serialization. An uninitialized CUDA runtime contributes no device state,
    so CPU checkpointing does not initialize a GPU just to record RNG.
    """
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": ([torch.cuda.get_rng_state()] if torch.cuda.is_initialized() else []),
        "torch_cuda_device": torch.cuda.current_device() if torch.cuda.is_initialized() else None,
        "numpy": (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])),
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Restore random streams after constructing models, optimizers and data loaders.

    Construction may itself draw random numbers, so restoring earlier would
    shift the resumed trajectory. New journals save the active CUDA device
    state as one entry; legacy journals may contain a list indexed by device.
    A saved CUDA RNG state requires an available training device.
    """
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, gaussian = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, gaussian))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU checkpoint requires CUDA to restore its random state")
        device = torch.cuda.current_device()
        saved_device = state.get("torch_cuda_device", device)
        states = state["torch_cuda"]
        index = 0 if "torch_cuda_device" in state else saved_device
        if index >= len(states):
            raise RuntimeError("Saved CUDA random state does not contain the training device")
        torch.cuda.set_rng_state(states[index].cpu(), device)


def state_digest(state):
    """Fingerprint loaded model tensors without another pass over the weight file."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def restore_optimizer(optimizer, state):
    """Load native optimizer slots and validate parameter-shaped numerical state.

    Momentum/variance slots must match their associated Parameter shape and
    contain finite values. The scalar step counter is exempt from the shape
    rule because native optimizers store it independently of parameter layout.
    """
    optimizer.load_state_dict(state)
    for parameter, slots in optimizer.state.items():
        for name, value in slots.items():
            if isinstance(value, torch.Tensor):
                if name != "step" and value.shape != parameter.shape:
                    raise ValueError(f"Optimizer {name} shape differs from its parameter")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Optimizer {name} contains nonfinite values")


def write_model_export(path, payload):
    """Atomically publish final model state or accept an identical existing export.

    If model.pt already exists, compare tensor keys, shapes, dtypes and
    values plus normalization/model configuration. Otherwise flush a unique
    temporary file and create the destination by hard link, which cannot
    replace prior output. Optimizer/RNG state remains in the journal.
    """
    path = Path(path)
    state_key = next((key for key in ("model_state_dict", "model_state", "state") if key in payload), None)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        expected = payload[state_key] if state_key else payload
        actual = existing.get(state_key, {}) if state_key else existing
        if set(actual) != set(expected) or any(
                (not isinstance(actual[key], torch.Tensor) or actual[key].shape != value.shape
                 or actual[key].dtype != value.dtype or not torch.equal(actual[key], value))
                if isinstance(value, torch.Tensor) else actual[key] != value
                for key, value in expected.items()):
            raise ValueError("Existing terminal model differs from the completed checkpoint")
        def equal(left, right):
            """Compare nested export configuration, including tensor shape, dtype and values."""
            if isinstance(right, torch.Tensor):
                return (isinstance(left, torch.Tensor) and left.shape == right.shape
                        and left.dtype == right.dtype and torch.equal(left, right))
            if isinstance(right, dict):
                return isinstance(left, dict) and left.keys() == right.keys() and all(equal(left[k], v) for k, v in right.items())
            if isinstance(right, (tuple, list)):
                return isinstance(left, (tuple, list)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
            return left == right
        for key in ("normalization", "model_configuration", "model_config"):
            if key in payload and not equal(existing.get(key), payload[key]):
                raise ValueError("Existing terminal model configuration differs from the completed checkpoint")
        return path
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    return path


def write_json_export(path, value):
    """Create export metadata atomically if it is missing.

    This helper treats an existing record as already published and leaves
    it untouched; callers obtain its contents or validate run completion.
    Flush a sibling temporary file before create-only publication.
    """
    path = Path(path)
    if path.exists():
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


class CheckpointStore:
    """A checkpoint directory belonging to exactly one configured training run.

    `identity` contains the model configuration, seed and training budget, plus
    descriptive provenance. Caller-owned `payload`
    contains model/optimizer/scheduler/loader state and phase/selection/history.
    NumPy arrays must be converted to tensors or lists before saving.
    """

    def __init__(self, directory: Path, identity: dict[str, Any], *, resume: bool = False):
        """Create a new attempt or read its latest committed boundary.

        identity is JSON-compatible configuration/provenance. Loading exposes
        payload and rng without applying them: the caller must first construct
        and restore its model, optimizer and scheduler, then restore RNG state.
        """
        self.directory = Path(directory).resolve()
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        self.runtime = runtime_identity()
        self.payload: dict[str, Any] | None = None
        self.rng: dict[str, Any] | None = None
        self.boundary: str | None = None
        self._latest_digest: str | None = None
        if resume:
            self._load()
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            record = {"schema": "gift.training-attempt.v1", "identity": self.identity,
                      "runtime": self.runtime, "run_id": uuid.uuid4().hex}
            with (self.directory / "ATTEMPT.json").open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")

    def _load(self) -> None:
        """Load the latest weights-only boundary and validate its progress metadata.

        The pointer must name a sibling .pt file. New pointers include file size;
        legacy pointers may omit it. Original identity remains available as
        provenance, while compatible present-day configuration governs resume.
        """
        attempt = json.loads((self.directory / "ATTEMPT.json").read_text(encoding="utf-8"))
        validate_resume(self.identity, attempt["identity"])
        self.original_identity = attempt["identity"]
        pointer_bytes = (self.directory / "LATEST.json").read_bytes()
        latest = json.loads(pointer_bytes)
        self._latest_digest = hashlib.sha256(pointer_bytes).hexdigest()
        path = (self.directory / latest["file"]).resolve(strict=True)
        if path.parent != self.directory or path.suffix != ".pt":
            raise ValueError("Checkpoint pointer escapes its own run directory")
        if "bytes" in latest and path.stat().st_size != latest["bytes"]:
            raise ValueError("Checkpoint file size differs")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_resume(self.identity, saved["identity"])
        validate_training_state(saved["payload"])
        self.payload, self.rng, self.boundary = saved["payload"], saved["rng"], saved["boundary"]

    def restore_random_state(self) -> None:
        """Restore resumed RNG after all model, optimizer and loader setup is complete."""
        if self.rng is None:
            raise RuntimeError("No resumed random state to restore")
        restore_rng(self.rng)

    def save(self, boundary: str, payload: dict[str, Any]) -> Path:
        """Refuse stale/concurrent writers instead of overwriting another branch."""
        with self._writer_lock():
            pointer = self.directory / "LATEST.json"
            current = digest_file(pointer) if pointer.exists() else None
            if current != self._latest_digest:
                raise RuntimeError("Another writer advanced this run; refusing a stale checkpoint")
            path = self._save_locked(boundary, payload)
            self._latest_digest = digest_file(pointer)
            return path

    @contextmanager
    def _writer_lock(self):
        # The persistent one-byte file is harmless; the OS releases the lock on
        # process exit, including a crash. Never remove another process's lock.
        """Hold an OS lock for one checkpoint commit and release it on context exit.

        The persistent one-byte lock file is not a ownership certificate. OS
        locking prevents simultaneous writers and releases automatically on exit.
        """
        with (self.directory / ".writer.lock").open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another process is saving this run") from error
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _save_locked(self, boundary: str, payload: dict[str, Any]) -> Path:
        """Flush an immutable payload and atomically advance the latest pointer.

        The caller holds the writer lock. Capture RNG alongside payload, write
        the .pt file first, then replace LATEST.json; interruption before replacement
        leaves the preceding boundary current.
        """
        if not boundary or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in boundary):
            raise ValueError("Use a simple checkpoint boundary name")
        token = uuid.uuid4().hex
        path = self.directory / f"{boundary}_{token}.pt"
        saved = {"schema": "gift.training-boundary.v1", "identity": self.identity,
                 "runtime": self.runtime, "boundary": boundary,
                 "rng": capture_rng(), "payload": payload}
        with path.open("xb") as stream:
            torch.save(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        pointer = self.directory / f"LATEST_{token}.tmp"
        with pointer.open("x", encoding="utf-8") as stream:
            json.dump({"file": path.name, "bytes": path.stat().st_size}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer, self.directory / "LATEST.json")
        return path
