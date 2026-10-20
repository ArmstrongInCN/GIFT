"""Portable input-source and child-environment boundaries for training queues."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def queue_lease(directory):
    """Hold a nonblocking OS lock for this queue on Windows or POSIX.

    The handle stays open for the full queue lifetime; process exit releases
    the lock without treating a stale lock-file byte as a live owner.
    """
    with (Path(directory) / "queue.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def configured_source(explicit, variable, *, required=False):
    """Resolve explicit > environment without supplying an author's machine path.

    Planning may retain ``None`` before dependencies are installed. Execution
    requires an existing external directory and reports the setting to supply.
    """
    value = explicit if explicit is not None else os.environ.get(variable)
    if not value:
        if required:
            raise FileNotFoundError(f"Set {variable} or supply the queue's explicit source option; see docs/SETUP.md")
        return None
    source = Path(value).expanduser().resolve()
    if source == ROOT or ROOT in source.parents:
        raise ValueError("External algorithm sources must stay outside GIFT: " + str(source))
    if required and not source.is_dir():
        raise FileNotFoundError(f"{variable} source directory does not exist: {source}")
    return str(source)


def validate_job_source(job):
    """Reject unavailable native-model sources before starting a training child."""
    spec = job.get("model_spec", {})
    if "source_root" in spec:
        variable = "GIFT_ANIE_SOURCE_ROOT" if str(spec.get("method", "")).startswith("anie") else "GIFT_EXTERNAL_ROOT"
        # A stored path is provenance after a run moves machines. Current
        # source settings locate the same architecture without editing locks.
        configured_source(os.environ.get(variable) or spec["source_root"], variable, required=True)


def child_environment():
    """Retain user source/device/thread choices and make local src importable."""
    environment = dict(os.environ)
    search = [str(ROOT / "src")]
    search.extend(part for part in environment.get("PYTHONPATH", "").split(os.pathsep)
                  if part and part not in search)
    environment["PYTHONPATH"] = os.pathsep.join(search)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(variable, "4")
    return environment
