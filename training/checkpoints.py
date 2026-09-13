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


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity() -> dict[str, Any]:
    """Reject silent numerical-profile changes when continuing a saved run."""
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
        # Missing and empty values deliberately remain different. Old journals
        # without this binding require their original code, not a guessed upgrade.
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
    """Represent NumPy state with primitive values safe for weights_only loading."""
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "numpy": (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])),
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Restore after constructing/loading models, optimizers, and data loaders."""
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, gaussian = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, gaussian))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG device count differs from the saved run")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


class CheckpointStore:
    """A checkpoint directory belonging to exactly one configured training run.

    `identity` should include source/data hashes, model configuration, seed and
    training budget, but not the absolute output path. Caller-owned `payload`
    contains model/optimizer/scheduler/loader state and phase/selection/history.
    NumPy arrays must be converted to tensors or lists before saving.
    """

    def __init__(self, directory: Path, identity: dict[str, Any], *, resume: bool = False):
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
        attempt = json.loads((self.directory / "ATTEMPT.json").read_text(encoding="utf-8"))
        if attempt["identity"] != self.identity:
            raise ValueError("Resume identity differs: data, source, model, seed or budget changed")
        if attempt["runtime"] != self.runtime:
            raise ValueError("Resume numerical runtime differs; do not silently continue this run")
        pointer_bytes = (self.directory / "LATEST.json").read_bytes()
        latest = json.loads(pointer_bytes)
        self._latest_digest = hashlib.sha256(pointer_bytes).hexdigest()
        path = (self.directory / latest["file"]).resolve(strict=True)
        if path.parent != self.directory or path.suffix != ".pt":
            raise ValueError("Checkpoint pointer escapes its own run directory")
        if digest_file(path) != latest["sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != self.identity or saved["runtime"] != self.runtime:
            raise ValueError("Checkpoint identity/runtime mismatch")
        self.payload, self.rng, self.boundary = saved["payload"], saved["rng"], saved["boundary"]

    def restore_random_state(self) -> None:
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
            json.dump({"file": path.name, "sha256": digest_file(path)}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer, self.directory / "LATEST.json")
        return path
