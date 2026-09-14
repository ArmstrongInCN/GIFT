"""Data, geometry and resumable state around native external PINN operations.

Only the coefficient dimension and problem library are adapted. The network,
coordinate derivatives, polynomial powers, loss and NAdam update are upstream.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import SimpleNamespace
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLING_DATA = {
    "5D3D0EF4BE9E7DFD03BB23A444BAC837AF3D349D54EF42E20C707D5EF9D0F4C4": ("noise_000", "C3F21EF13D0C81713F2770FC67547C91BD5A793FA34A9EECC5C46537237FA8A5"),
    "091BF97BA7BAB923158B5E5F74DD1469381213EBC169A4BCD2633A52D92D09C3": ("noise_001", "97C5773290B5C50393FF64B7B42C021FD54F2BB429FC6D57CD4599F09F9818BC"),
    "86E993B6F70F11401F1BB40DEC11F6AC1BCC76CB62613F6F382FB4C8DB8B0D76": ("noise_010", "87661D4EC2844FD2EEC4256D4DCF2C2FEA1B948ACB39ADDE454E95E855DDD3BC"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def sources(external_root=None):
    configured = external_root or os.environ.get("GIFT_EXTERNAL_ROOT")
    if not configured:
        raise ValueError("Set GIFT_EXTERNAL_ROOT to the external clone parent")
    registry = json.loads((ROOT / "external_sources.json").read_text(encoding="utf-8"))["sources"]
    records, files = {}, {}
    for key in ("pinn_sr", "tensorflow115"):
        record = registry[key]
        directory = (Path(configured) / record["directory"]).resolve(strict=True)
        forbidden = [ROOT]
        if os.environ.get("GIFT_DATA_ROOT"):
            forbidden.append(Path(os.environ["GIFT_DATA_ROOT"]).resolve())
        if any(directory == p or p in directory.parents for p in forbidden):
            raise ValueError("Third-party source must stay outside code and data packages")
        commit = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        if commit != record["commit"]:
            raise ValueError("External commit mismatch: " + key)
        records[key] = {"commit": commit, "repository": record["repository"], "files": {}}
        for relative, expected in record["files"].items():
            path = (directory / relative).resolve(strict=True)
            if directory not in path.parents or sha256(path) != expected["sha256"]:
                raise ValueError("External byte mismatch: " + relative)
            records[key]["files"][relative] = expected["sha256"]
            files[path.name] = path
    return records, files


def _class(mode, tf, files, device):
    path = next(path for name, path in files.items() if name.startswith("NS_Vorticity_"))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef) and node.name == "PhysicsInformedNN")
    edits = 0
    for node in ast.walk(selected):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "lambda_w" for t in node.targets):
            for value in ast.walk(node.value):
                if isinstance(value, ast.List) and [getattr(item, "value", None) for item in value.elts] == [60, 1]:
                    value.elts[0].value = 90 if mode == "open" else 4
                    edits += 1
    if edits != 1:
        raise ValueError("exactly one coefficient dimension adjustment required")
    nadam = _module(files["nadam_optimizer.py"], "external_native_nadam")
    scipy = _module(files["external_optimizer.py"], "external_native_scipy")

    class Compatibility:
        contrib = SimpleNamespace(opt=SimpleNamespace(NadamOptimizer=nadam.NadamOptimizer,
            ScipyOptimizerInterface=scipy.ScipyOptimizerInterface))
        def __getattr__(self, name):
            return getattr(tf.compat.v1, name)
        def Session(self, config):
            config.log_device_placement = False
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1
            config.device_count["GPU"] = int(device == "gpu")
            config.gpu_options.allow_growth = True
            return tf.compat.v1.Session(config=config)

    namespace = {"tf": Compatibility(), "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])), str(path), "exec"), namespace)
    upstream = namespace["PhysicsInformedNN"]

    class Geometry(upstream):
        def net_f(self, x, y, t):
            self.forcing_y = y
            return super().net_f(x, y, t)
        def build_library(self, data, derivatives, derivatives_description, PolyOrder=2, data_description=None):
            u, v, w = data
            q = -4 * tf.cos(4 * self.forcing_y)
            if mode == "known":
                return [w, u*derivatives[1] + v*derivatives[2], derivatives[3] + derivatives[5], q], ["w", "advection", "laplacian", "q"]
            terms, names = super().build_library(data, derivatives, derivatives_description, PolyOrder, data_description)
            for name, term in zip(["q", "q^2", "u*q", "v*q", "w*q"], [q, q**2, u*q, v*q, w*q]):
                for suffix, derivative in zip(derivatives_description, derivatives):
                    terms.append(term * derivative)
                    names.append(name + suffix)
            return terms, names
    return Geometry


class PINNRuntime:
    """Fresh native initialization; all global variables form a resume boundary."""
    def __init__(self, mode, lower, upper, external_root=None, device="cpu"):
        if mode not in ("known", "open") or device not in ("cpu", "gpu"):
            raise ValueError("invalid PINN mode or device")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if os.environ.get(key) != "1":
                raise ValueError("Set %s=1 before Python starts" % key)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != ("0" if device == "gpu" else "-1"):
            raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly for a separate PINN process")
        import tensorflow as tf
        import scipy
        if (tf.__version__, np.__version__, scipy.__version__) != ("2.10.0", "1.21.6", "1.7.3"):
            raise ValueError("Use Python 3.8 with TensorFlow2.10/NumPy1.21.6/SciPy1.7.3")
        tf.compat.v1.disable_eager_execution()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        if device == "gpu" and len(tf.config.list_physical_devices("GPU")) != 1:
            raise ValueError("Exactly one visible GPU required; no CPU fallback")
        self.source_records, files = sources(external_root)
        self.tf, self.mode, self.iteration = tf, mode, 0
        self.profile = {"python": platform.python_version(), "tensorflow": tf.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform(), "dtype": "float32",
                        "device": device, "threads": 1, "tensorflow_tf32": False,
                        "initialization": "native_Xavier_seed1234",
                        "environment": {key: os.environ.get(key) for key in (
                            "TF_ENABLE_ONEDNN_OPTS", "TF_DETERMINISTIC_OPS",
                            "TF_CUDNN_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")}}
        lower, upper = np.asarray(lower, np.float32), np.asarray(upper, np.float32)
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(upper > lower):
            raise ValueError("invalid normalization bounds")
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device("/GPU:0" if device == "gpu" else "/CPU:0"):
            tf.compat.v1.set_random_seed(1234)
            geometry = _class(mode, tf, files, device)
            dummy = np.zeros((1, 3), np.float32)
            self.model = geometry(dummy, dummy[:, :1], dummy[:, :1], dummy[:, :1], dummy,
                                  dummy, dummy[:, :1], dummy[:, :1], dummy[:, :1],
                                  [3]+[60]*8+[3], lower, upper, 1)
            model = self.model
            self.session = model.sess
            self.parameters = model.biases + model.weights + [model.lambda_w]
            self.variables = tf.compat.v1.global_variables()
            self.mask = np.ones((90 if mode == "open" else 4, 1), np.float32)
            self.restore_placeholders = [tf.compat.v1.placeholder(v.dtype, v.shape) for v in self.variables]
            self.restore_op = tf.group(*[v.assign(p) for v, p in zip(self.variables, self.restore_placeholders)])
            self.coefficient_input = tf.compat.v1.placeholder(tf.float32, self.mask.shape)
            self.coefficient_assign = model.lambda_w.assign(self.coefficient_input)
            self.native_gradients = tf.compat.v1.gradients(model.loss, self.parameters)
            self.library_names = list(model.lib_descr)
            if len(self.library_names) != len(self.mask):
                raise ValueError("native library and coefficient dimensions disagree")
            self.profile["session_devices"] = [item.name for item in self.session.list_devices()]
            if device == "gpu":
                self.profile["gpu_details"] = tf.config.experimental.get_device_details(
                    tf.config.list_physical_devices("GPU")[0])
                if not any("GPU:0" in name for name in self.profile["session_devices"]):
                    self.close()
                    raise ValueError("GPU session unavailable")

    def close(self):
        if getattr(self, "session", None) is not None:
            self.session.close()
            self.session = None

    def state(self):
        return {"variables": self.session.run(self.variables), "mask": self.mask.copy(),
                "iteration": self.iteration}

    def restore(self, state):
        values = state["variables"]
        if len(values) != len(self.variables) or any(
            value.dtype != variable.dtype.as_numpy_dtype or list(value.shape) != variable.shape.as_list()
            or not np.isfinite(value).all() for value, variable in zip(values, self.variables)
        ):
            raise ValueError("invalid native TensorFlow state layout")
        mask = np.asarray(state["mask"])
        if mask.shape != self.mask.shape or not np.isin(mask, [0., 1.]).all():
            raise ValueError("invalid sparse mask")
        if type(state["iteration"]) is not int or state["iteration"] < 0:
            raise ValueError("invalid NAdam update count")
        self.session.run(self.restore_op, dict(zip(self.restore_placeholders, values)))
        self.mask, self.iteration = mask.astype(np.float32).copy(), state["iteration"]

    def set_coefficients(self, values, mask=None):
        values = np.asarray(values, np.float32).reshape(-1, 1)
        if values.shape != self.mask.shape or not np.isfinite(values).all():
            raise ValueError("invalid coefficient vector")
        if mask is not None:
            mask = np.asarray(mask, np.float32).reshape(-1, 1)
            if mask.shape != self.mask.shape or not np.isin(mask, [0., 1.]).all():
                raise ValueError("invalid coefficient mask")
            self.mask = mask.copy()
        self.session.run(self.coefficient_assign, {self.coefficient_input: values})

    def feed(self, coordinates, targets, physics, physics_weight=1., l1_weight=1e-7, learning_rate=1e-3):
        if (coordinates.ndim != 2 or coordinates.shape[1] != 3 or targets.shape != coordinates.shape
                or physics.ndim != 2 or physics.shape[1] != 3 or not len(coordinates) or not len(physics)
                or any(not np.isfinite(value).all() for value in (coordinates, targets, physics))):
            raise ValueError("finite [rows,3] observations and collocation points required")
        m = self.model
        return {m.x_tf: coordinates[:, :1], m.y_tf: coordinates[:, 1:2], m.t_tf: coordinates[:, 2:],
                m.u_tf: targets[:, :1], m.v_tf: targets[:, 1:2], m.w_tf: targets[:, 2:],
                m.x_f_tf: physics[:, :1], m.y_f_tf: physics[:, 1:2], m.t_f_tf: physics[:, 2:],
                m.NonZeroMask_w_tf: self.mask, m.loss_f_coeff_tf: physics_weight,
                m.L1_coeff_tf: l1_weight, m.lr_tf: learning_rate}

    def metrics(self, feed):
        m = self.model
        values = self.session.run([m.loss, m.loss_u, m.loss_v, m.loss_w, m.loss_f_w, m.loss_lambda_w], feed)
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite native PINN loss")
        return dict(zip(("loss", "u", "v", "w", "physics", "l1"), map(float, values)))

    def step(self, feed):
        self.session.run(self.model.train_op_Adam, feed)
        self.iteration += 1

    def library(self, physics):
        m = self.model
        return self.session.run([m.Phi, m.w_t_pred], {
            m.x_f_tf: physics[:, :1], m.y_f_tf: physics[:, 1:2], m.t_f_tf: physics[:, 2:],
            m.NonZeroMask_w_tf: self.mask})


def read_inputs(sampling, data_path, train_rows=None, physics_rows=None):
    paths = {"sampling": Path(sampling).resolve(strict=True), "data": Path(data_path).resolve(strict=True)}
    bound = {key: {"sha256": sha256(path), "bytes": path.stat().st_size} for key, path in paths.items()}
    condition = SAMPLING_DATA.get(bound["sampling"]["sha256"])
    if condition is None or bound["data"]["sha256"] != condition[1]:
        raise ValueError("sampling and dataset must be a matching declared noise condition")
    bound["condition"] = condition[0]
    with np.load(paths["sampling"], allow_pickle=False) as archive:
        coordinates = archive["measurement_coordinates"][archive["train_indices"]].copy()
        targets = archive["targets"][archive["train_indices"]].copy()
        physics = np.concatenate([archive["lhs_coordinates"], coordinates]).astype(np.float32)
        lower, upper = archive["lower"].copy(), archive["upper"].copy()
    for count, rows in ((train_rows, len(coordinates)), (physics_rows, len(physics))):
        if count is not None and (type(count) is not int or not 0 < count <= rows):
            raise ValueError("invalid diagnostic row limit")
    coordinates, targets, physics = coordinates[:train_rows], targets[:train_rows], physics[:physics_rows]
    bound["selection"] = {"train_rows": len(coordinates), "physics_rows": len(physics),
                          "diagnostic_subset": train_rows is not None or physics_rows is not None}
    return coordinates, targets, physics, lower, upper, bound

@contextmanager
def _writer(directory):
    """Short nonblocking checkpoint commit lock; lock file is never deleted."""
    with (directory / ".writer.lock").open("a+b") as stream:
        stream.seek(0, 2)
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
