"""First-party checkpoint protocol for SciPy 1.7.3's external setulb kernel.

No optimizer algorithm is implemented here. Only unbounded, analytic-gradient
L-BFGS-B is supported. Every advance returns at a reverse-communication boundary;
the caller may persist that complete state before the next objective evaluation.
This is not a PINN objective, a TensorFlow checkpoint, or a full training stage.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import uuid

import numpy as np
import scipy
from scipy.optimize import _lbfgsb


LOCKED_OPTIONS = {"maxcor": 50, "maxls": 50, "maxiter": 10000, "maxfun": 10000,
                  "gtol": 1e-5, "ftol": 0.1 * np.finfo(np.float64).eps}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest().upper()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def runtime_profile():
    if scipy.__version__ != "1.7.3" or np.__version__ != "1.21.6":
        raise RuntimeError("This private-API protocol requires SciPy1.7.3/NumPy1.21.6")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError("Set %s=1 before starting Python" % key)
    return {"python": platform.python_version(), "scipy": scipy.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "kernel_sha256": digest(_lbfgsb.__file__),
            "scipy_interface_sha256": digest(Path(_lbfgsb.__file__).parent / "lbfgsb.py"),
            "fortran_integer_dtype": str(_lbfgsb.types.intvar.dtype),
            "threads": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


class LBFGSB:
    """One kernel invocation at a time; no solver restart on FG/NEW_X resume.

    `identity` must identify the objective/data/source and parameter packing.
    Options default to the original PINN limits. Overrides are explicit and
    bound, chiefly for small budget-stop tests; no bounded constraints exist.
    """
    def __init__(self, x0, identity, options=None):
        x = np.asarray(x0, dtype=np.float64)
        if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
            raise ValueError("Expected a finite nonempty flat parameter vector")
        self.options = dict(LOCKED_OPTIONS)
        if options:
            if set(options) - set(self.options):
                raise ValueError("Unknown L-BFGS-B option")
            self.options.update(options)
        for key in ("maxcor", "maxls", "maxiter", "maxfun"):
            if type(self.options[key]) is not int or self.options[key] <= 0:
                raise ValueError("Positive integer required: " + key)
        for key in ("gtol", "ftol"):
            if not np.isfinite(self.options[key]) or self.options[key] < 0:
                raise ValueError("Finite nonnegative tolerance required: " + key)
        self.identity = {"objective": identity, "options": self.options, "dimension": int(x.size),
                         "initial_x_sha256": hashlib.sha256(x.tobytes()).hexdigest().upper(),
                         "source_sha256": digest(Path(__file__)), "runtime": runtime_profile(),
                         "constraints": "none", "gradient": "analytic float64"}
        self.identity = json.loads(json.dumps(self.identity, allow_nan=False))
        n, m = x.size, self.options["maxcor"]
        integer = _lbfgsb.types.intvar.dtype
        # Sizes/dtypes are the external f2py ABI, not a local solver algorithm.
        layout = {"x": ((n,), np.float64), "g": ((n,), np.float64), "f": ((), np.float64),
                  "l": ((n,), np.float64), "u": ((n,), np.float64), "nbd": ((n,), integer),
                  "wa": ((2*m*n+5*n+11*m*m+8*m,), np.float64), "iwa": ((3*n,), integer),
                  "task": ((1,), "S60"), "csave": ((1,), "S60"), "lsave": ((4,), integer),
                  "isave": ((44,), integer), "dsave": ((29,), np.float64),
                  "cached_x": ((n,), np.float64), "cached_g": ((n,), np.float64), "cached_f": ((), np.float64)}
        self.arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in layout.items()}
        self.arrays["x"][:] = x
        self.arrays["task"][:] = b"START"
        self.counts = {"iterations": 0, "evaluations": 0, "fg_requests": 0, "kernel_calls": 0}
        self.cached = False
        self.pending_stop = None
        self.done = False

    @property
    def task(self):
        return self.arrays["task"].tobytes().rstrip(b"\0 ").decode("ascii")

    @property
    def x(self):
        return self.arrays["x"].copy()

    def advance(self, objective):
        if self.done:
            raise RuntimeError("Terminal solver state; refusing another kernel call")
        a = self.arrays
        if self.pending_stop is not None:
            a["task"][:] = self.pending_stop.encode("ascii")
            self.pending_stop = None
        elif self.task.startswith("FG"):
            if not self.cached or not np.array_equal(a["x"], a["cached_x"]):
                f, g = objective(a["x"].copy())
                g = np.asarray(g, dtype=np.float64)
                if np.ndim(f) != 0 or g.shape != a["g"].shape or not np.isfinite(f) or not np.all(np.isfinite(g)):
                    raise ValueError("Objective must return a finite scalar and full analytic gradient")
                a["cached_x"][:] = a["x"]
                a["cached_g"][:] = g
                a["cached_f"][...] = f
                self.cached = True
                self.counts["evaluations"] += 1
            a["f"][...] = a["cached_f"]
            a["g"][:] = a["cached_g"]
        _lbfgsb.setulb(self.options["maxcor"], a["x"], a["l"], a["u"], a["nbd"], a["f"], a["g"],
                       self.options["ftol"] / np.finfo(np.float64).eps, self.options["gtol"],
                       a["wa"], a["iwa"], a["task"], -1, a["csave"], a["lsave"], a["isave"], a["dsave"], self.options["maxls"])
        self.counts["kernel_calls"] += 1
        if self.task.startswith("FG"):
            self.counts["fg_requests"] += 1
        elif self.task.startswith("NEW_X"):
            self.counts["iterations"] += 1
            # Match public SciPy's accepted-iteration budget semantics: a line
            # search may exceed maxfun before returning an accepted point.
            if self.counts["iterations"] >= self.options["maxiter"]:
                self.pending_stop = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
            elif self.counts["evaluations"] > self.options["maxfun"]:
                self.pending_stop = "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
        else:
            self.done = True
        return {"task": self.task, "x": self.x, "f": float(a["f"]), **self.counts, "done": self.done}

    def metadata(self):
        return {"identity": self.identity, "counts": self.counts, "cached": self.cached,
                "pending_stop": self.pending_stop, "done": self.done,
                "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in self.arrays.items()}}

    def restore(self, arrays, metadata):
        if metadata["identity"] != self.identity or metadata["arrays"] != self.metadata()["arrays"] or set(arrays) != set(self.arrays):
            raise ValueError("Saved solver identity or ABI differs")
        for key, expected in self.arrays.items():
            value = arrays[key]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("Saved array schema differs: " + key)
            if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                raise ValueError("Nonfinite saved array: " + key)
        if set(metadata["counts"]) != set(self.counts) or any(type(v) is not int or v < 0 for v in metadata["counts"].values()):
            raise ValueError("Invalid saved counters")
        if type(metadata["cached"]) is not bool or type(metadata["done"]) is not bool:
            raise ValueError("Invalid saved flags")
        if metadata["pending_stop"] not in (None, "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT", "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"):
            raise ValueError("Invalid pending stop")
        self.arrays = {k: v.copy() for k, v in arrays.items()}
        self.counts, self.cached = dict(metadata["counts"]), metadata["cached"]
        self.pending_stop, self.done = metadata["pending_stop"], metadata["done"]


@contextmanager
def _lock(directory):
    with (directory / ".writer.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Journal:
    """Same-attempt immutable state pairs; pointer commits only after fsync."""
    def __init__(self, directory, solver, resume=False):
        self.directory, self.solver = Path(directory).resolve(), solver
        self.latest = None
        if not resume:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.run_id = uuid.uuid4().hex
            _json(self.directory / "RUN.json", {"run_id": self.run_id, "identity": solver.identity})
        else:
            header = json.loads((self.directory / "RUN.json").read_text(encoding="utf-8"))
            if header["identity"] != solver.identity:
                raise ValueError("Resume is not the same objective/source/config/runtime")
            self.run_id = header["run_id"]
            self._load()

    def _pointer(self):
        path = self.directory / "LATEST.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(self):
        with _lock(self.directory):
            if self._pointer() != self.latest:
                raise ValueError("A different writer advanced the journal")
            name = "boundary_%08d" % self.solver.counts["kernel_calls"]
            folder = self.directory / name
            folder.mkdir(exist_ok=False)
            with (folder / "state.npz").open("xb") as stream:
                np.savez_compressed(stream, **self.solver.arrays)
                stream.flush()
                os.fsync(stream.fileno())
            record = {"run_id": self.run_id, "state": self.solver.metadata(), "npz_sha256": digest(folder / "state.npz")}
            _json(folder / "state.json", record)
            latest = {"run_id": self.run_id, "directory": name, "metadata_sha256": digest(folder / "state.json")}
            temporary = self.directory / ("LATEST_%s.tmp" % uuid.uuid4().hex)
            _json(temporary, latest)
            os.replace(str(temporary), str(self.directory / "LATEST.json"))
            self.latest = latest
            return folder

    def _load(self):
        latest = self._pointer()
        if not latest or set(latest) != {"run_id", "directory", "metadata_sha256"} or latest["run_id"] != self.run_id:
            raise ValueError("Invalid same-run pointer")
        name = latest["directory"]
        if not isinstance(name, str) or not name.startswith("boundary_") or not name[9:].isdigit():
            raise ValueError("Invalid boundary name")
        folder = (self.directory / name).resolve(strict=True)
        if folder.parent != self.directory or digest(folder / "state.json") != latest["metadata_sha256"]:
            raise ValueError("Checkpoint metadata integrity failed")
        record = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if record["run_id"] != self.run_id or digest(folder / "state.npz") != record["npz_sha256"]:
            raise ValueError("Checkpoint data integrity failed")
        with np.load(folder / "state.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        self.solver.restore(arrays, record["state"])
        self.latest = latest
