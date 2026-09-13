"""First-party data/geometry/state glue around externally owned PINN/TF code.

NAdam only: not the full PINN-SR ADO, STRidge or L-BFGS experiment. This module
contains no upstream network, sparse regression or optimizer implementation.
TensorFlow is imported only when a runtime is explicitly constructed.
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
INITIAL_SHA = "AB0834A1494D2D46E0CBC8A4E24D29DE91B5F10EAD97F7DB6990A0C302D990CB"
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


def _class(mode, tf, files, device="cpu"):
    upstream_path = next(path for name, path in files.items() if name.startswith("NS_Vorticity_"))
    parsed = ast.parse(upstream_path.read_text(encoding="utf-8"))
    classes = [n for n in ast.walk(parsed) if isinstance(n, ast.ClassDef) and n.name == "PhysicsInformedNN"]
    if len(classes) != 1:
        raise ValueError("Expected one pinned class")
    selected, edits = classes[0], 0
    for node in ast.walk(selected):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "lambda_w" for t in node.targets):
            for value in ast.walk(node.value):
                if isinstance(value, ast.List) and len(value.elts) == 2 and all(isinstance(x, ast.Constant) for x in value.elts) and [x.value for x in value.elts] == [60, 1]:
                    value.elts[0].value = 90 if mode == "open" else 4
                    edits += 1
    if edits != 1:
        raise ValueError("Expected one coefficient dimension edit")
    # GPU02_KNOWN_MASK_SHAPE_BEGIN: the historical KC mask is exactly [4, 1].
    if mode == "known":
        mask_assignments = [n for n in ast.walk(selected) if isinstance(n, ast.Assign)
                            and len(n.targets) == 1
                            and ast.dump(n.targets[0]) == ast.dump(ast.parse("self.NonZeroMask_w_tf = None").body[0].targets[0])]
        expected_mask = ast.parse("tf.placeholder(tf.float32)", mode="eval").body
        if len(mask_assignments) != 1 or ast.dump(mask_assignments[0].value) != ast.dump(expected_mask):
            raise ValueError("Expected exactly the pinned shapeless KC mask placeholder")
        mask_assignments[0].value.keywords.append(ast.keyword(
            arg="shape", value=ast.List(elts=[ast.Constant(value=4, kind=None), ast.Constant(value=1, kind=None)], ctx=ast.Load())))
    # GPU02_KNOWN_MASK_SHAPE_END
    # Retain the upstream network body but identify its joint coordinate tensor.
    # The historical experiment differentiates that tensor once, then slices;
    # separate x/y/t backprop paths differ for nonzero equation coefficients.
    net_u = next(n for n in selected.body if isinstance(n, ast.FunctionDef) and n.name == "net_U")
    concat_calls = [n for n in ast.walk(net_u) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "tf" and n.func.attr == "concat"]
    if len(concat_calls) != 1 or ast.dump(concat_calls[0].args[0]) != ast.dump(ast.List(elts=[ast.Name(id=k, ctx=ast.Load()) for k in ("x", "y", "t")], ctx=ast.Load())):
        raise ValueError("Expected the pinned net_U coordinate concatenation")
    concat_calls[0].func = ast.Name(id="gift_coordinates", ctx=ast.Load())
    library = next(n for n in selected.body if isinstance(n, ast.FunctionDef) and n.name == "build_library")
    powers = [n for n in ast.walk(library) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)]
    if len(powers) != 1 or not isinstance(powers[0].right, ast.Name) or powers[0].right.id != "j":
        raise ValueError("Expected the pinned univariate library power")

    class PolynomialOperation(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if node is powers[0]:
                return ast.Call(func=ast.Name(id="gift_polynomial", ctx=ast.Load()), args=[node.left, node.right], keywords=[])
            return self.generic_visit(node)

    PolynomialOperation().visit(library)

    def gift_polynomial(value, power):
        # Pow's reverse pass g*(2*x) can differ from Mul's (g*x)+(g*x).
        # Preserve the current experiment's actual graph, not just its algebra.
        if power == 1:
            return value
        if power == 2:
            return value * value
        raise ValueError("Only the fixed second-order library is supported")
    coordinate_views, derivative_cache = {}, {}

    def gift_coordinates(values, axis):
        joint = tf.concat(values, axis)
        for index, value in enumerate(values):
            coordinate_views[value] = (joint, index)
        return joint
    nadam = _module(files["nadam_optimizer.py"], "gift_external_nadam115")
    scipy_interface = _module(files["external_optimizer.py"], "gift_external_scipy115")

    class Facade:
        contrib = SimpleNamespace(opt=SimpleNamespace(NadamOptimizer=nadam.NadamOptimizer,
                    ScipyOptimizerInterface=scipy_interface.ScipyOptimizerInterface))

        def __getattr__(self, name):
            return getattr(tf.compat.v1, name)

        def Session(self, config=None):
            config.log_device_placement = False
            config.device_count["GPU"] = 1 if device == "gpu" else 0
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1
            if device == "gpu":
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.90
            return tf.compat.v1.Session(config=config)

        def gradients(self, ys, xs):
            if xs not in coordinate_views:
                raise ValueError("Unexpected upstream derivative coordinate")
            joint, index = coordinate_views[xs]
            key = (ys, joint)
            if key not in derivative_cache:
                derivative_cache[key] = tf.gradients(ys, joint)[0]
            return [derivative_cache[key][:, index:index+1]]

    namespace = {"tf": Facade(), "np": np, "gift_coordinates": gift_coordinates, "gift_polynomial": gift_polynomial}
    tree = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(tree, str(upstream_path), "exec"), namespace)
    upstream = namespace["PhysicsInformedNN"]

    class Geometry(upstream):
        def net_f(self, x, y, t):
            self._gift_y = y
            return super().net_f(x, y, t)

        def build_library(self, data, derivatives, derivatives_description, PolyOrder=2, data_description=None):
            u, v, w = data
            q = -4.0 * tf.cos(4.0 * self._gift_y)
            if mode == "known":
                return [w, u*derivatives[1] + v*derivatives[2], derivatives[3] + derivatives[5], q], ["w", "advection", "laplacian", "q"]
            terms, names = super().build_library(data, derivatives, derivatives_description, PolyOrder, data_description)
            for name, term in zip(["q", "q^2", "u*q", "v*q", "w*q"], [q, q*q, u*q, v*q, w*q]):
                for suffix, derivative in zip(derivatives_description, derivatives):
                    terms.append(term * derivative)
                    names.append(name + suffix)
            return terms, names

    return Geometry


class PINNRuntime:
    """Isolated device-explicit float32 boundary; GPU remains unvalidated."""
    def __init__(self, mode, lower, upper, initial, external_root=None, device="cpu"):
        if mode not in ("open", "known"):
            raise ValueError(mode)
        if device not in ("cpu", "gpu") or (device == "gpu" and mode != "known"):
            raise ValueError("GPU validation is KC/known only")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if os.environ.get(key) != "1":
                raise ValueError("Set %s=1 before Python starts" % key)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != ("0" if device == "gpu" else "-1"):
            raise ValueError("Explicit device requires CUDA_VISIBLE_DEVICES=0 (GPU) or -1 (CPU)")
        import tensorflow as tf
        import scipy
        if (tf.__version__, np.__version__, scipy.__version__) != ("2.10.0", "1.21.6", "1.7.3"):
            raise ValueError("Use the documented PINN numeric environment; other versions are unvalidated")
        tf.compat.v1.disable_eager_execution()
        self.tf, self.mode, self.iteration = tf, mode, 0
        self.source_records, files = sources(external_root)
        self.profile = {"python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(), "machine": platform.machine(),
                        "processor": platform.processor(), "dtype": "float32", "device": "CPU", "threads": 1,
                        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_ENABLE_ONEDNN_OPTS", "TF_DETERMINISTIC_OPS")},
                        "training_random_ops": False}
        if device == "gpu":
            physical = tf.config.list_physical_devices("GPU")
            if len(physical) != 1:
                raise ValueError("Exactly one visible GPU is required; CPU fallback is forbidden")
            details = tf.config.experimental.get_device_details(physical[0])
            build = tf.sysconfig.get_build_info()
            hardware = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=uuid,name,driver_version",
                                       "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
            self.profile.update(device="GPU:0", gpu_details=details, gpu_uuid_name_driver=hardware,
                cuda_build=build.get("cuda_version"), cudnn_build=build.get("cudnn_version"),
                tensorflow_tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
                session=dict(gpu_count=1, allow_growth=True, memory_fraction=0.9,
                             allow_soft_placement=True, intra_threads=1, inter_threads=1))
            self.profile["environment"].update({key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")})
        lower, upper = np.asarray(lower, np.float32), np.asarray(upper, np.float32)
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(upper > lower):
            raise ValueError("Invalid normalization bounds")
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device("/GPU:0" if device == "gpu" else "/CPU:0"):
            tf.compat.v1.set_random_seed(1234)
            geometry = _class(mode, tf, files, device=device)
            dummy = np.zeros((1, 3), np.float32)
            self.model = geometry(dummy, dummy[:, :1], dummy[:, :1], dummy[:, :1], dummy,
                                  dummy, dummy[:, :1], dummy[:, :1], dummy[:, :1],
                                  [3] + [60]*8 + [3], lower, upper, 1)
            m = self.model
            self.session = m.sess
            if device == "gpu":
                devices = [value.name for value in self.session.list_devices()]
                if not any("GPU:0" in value for value in devices):
                    self.close()
                    raise ValueError("Session has no GPU:0; CPU fallback is forbidden")
                self.profile["session_devices"] = devices
            network = [v for pair in zip(m.weights, m.biases) for v in pair]
            self.parameters = network + [m.lambda_w]
            if len(initial) != len(network) or any(np.asarray(a).dtype != np.float32 or list(a.shape) != v.shape.as_list() for a, v in zip(initial, network)):
                self.close()
                raise ValueError("Expected the 18 canonical float32 initialization arrays")
            self.session.run([v.assign(a) for a, v in zip(initial, network)])
            self.mask = np.ones((90 if mode == "open" else 4, 1), np.float32)
            self.prediction = m.u_pred.op.inputs[0]
            if self.prediction.shape.as_list() != [None, 3]:
                raise ValueError("Upstream prediction producer changed")
            self.targets = tf.compat.v1.placeholder(tf.float32, [None, 3])
            self.scales = [tf.compat.v1.placeholder(tf.float32, []) for _ in range(3)]
            residual = m.f_w_pred
            losses = [self.scales[0] * tf.reduce_sum(tf.square(self.prediction-self.targets)),
                      self.scales[1] * tf.reduce_sum(tf.square(residual)),
                      self.scales[2] * tf.norm(m.lambda_w, ord=1)]
            self.totals = [tf.Variable(0.0, trainable=False, name="gift_total_%d" % i) for i in range(3)]
            self.accumulators = [tf.Variable(tf.zeros_like(v), trainable=False, name="gift_gradient_%d" % i) for i, v in enumerate(self.parameters)]
            self.accumulate = []
            for loss, total in zip(losses, self.totals):
                gradients = tf.gradients(loss, self.parameters)
                self.accumulate.append(tf.group(total.assign_add(loss), *[
                    accumulator.assign_add(tf.zeros_like(v) if g is None else g)
                    for accumulator, g, v in zip(self.accumulators, gradients, self.parameters)]))
            self.zero = tf.group(*[v.assign(tf.zeros_like(v)) for v in self.totals+self.accumulators])
            self.divisor = tf.compat.v1.placeholder(tf.float32, [])
            self.divide = tf.group(*[a.assign(a/self.divisor) for a in self.accumulators])
            self.update = m.optimizer_Adam.apply_gradients(list(zip(self.accumulators, self.parameters)))
            self.session.run(tf.compat.v1.variables_initializer(self.totals+self.accumulators))
            self.variables = tf.compat.v1.global_variables()
            self.restore_placeholders = [tf.compat.v1.placeholder(v.dtype, v.shape) for v in self.variables]
            self.restore_op = tf.group(*[v.assign(p) for v, p in zip(self.variables, self.restore_placeholders)])

    def close(self):
        if getattr(self, "session", None) is not None:
            self.session.close()
            self.session = None

    def _coordinates(self, values, physics=False):
        values = np.asarray(values)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("Coordinates must be finite float32 N x 3")
        m = self.model
        placeholders = (m.x_f_tf, m.y_f_tf, m.t_f_tf) if physics else (m.x_tf, m.y_tf, m.t_tf)
        return {p: values[:, i:i+1] for i, p in enumerate(placeholders)}

    def predict(self, coordinates):
        return self.session.run(self.prediction, self._coordinates(coordinates))

    def library(self, coordinates):
        return self.session.run([self.model.Phi, self.model.w_t_pred], self._coordinates(coordinates, True))

    def set_coefficients(self, coefficients, mask=None):
        coefficients = np.asarray(coefficients, np.float32)
        mask = self.mask if mask is None else np.asarray(mask, np.float32)
        if coefficients.shape != self.mask.shape or mask.shape != self.mask.shape or not np.all(np.isfinite(coefficients)) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Invalid coefficients or binary mask")
        with self.graph.as_default():
            self.session.run(self.model.lambda_w.assign(coefficients))
        self.mask = mask.copy()

    def objective(self, coordinates, targets, physics, chunk=32768, physics_weight=1.0, l1_weight=1e-7):
        if chunk <= 0 or len(coordinates) == 0 or len(physics) == 0:
            raise ValueError("Positive chunk and nonempty rows required")
        targets = np.asarray(targets)
        if targets.dtype != np.float32 or targets.shape != coordinates.shape or not np.all(np.isfinite(targets)):
            raise ValueError("Targets must match finite float32 coordinates")
        if not np.isfinite(physics_weight) or not np.isfinite(l1_weight) or physics_weight < 0 or l1_weight < 0:
            raise ValueError("Loss weights must be finite and nonnegative")
        self.session.run(self.zero)
        for start in range(0, len(coordinates), chunk):
            feed = self._coordinates(coordinates[start:start+chunk])
            feed.update({self.targets: targets[start:start+chunk], self.scales[0]: 1.0/len(coordinates)})
            self.session.run(self.accumulate[0], feed)
        for start in range(0, len(physics), chunk):
            feed = self._coordinates(physics[start:start+chunk], True)
            feed.update({self.scales[1]: physics_weight/len(physics), self.model.NonZeroMask_w_tf: self.mask})
            self.session.run(self.accumulate[1], feed)
        if l1_weight:
            self.session.run(self.accumulate[2], {self.scales[2]: l1_weight})
        values = self.session.run(self.totals)
        total = max(float(values[0]+values[1]+values[2]), 1e-30)
        self.session.run(self.divide, {self.divisor: total})
        gradients = self.session.run(self.accumulators)
        if not np.isfinite(total) or any(not np.all(np.isfinite(g)) for g in gradients):
            raise ValueError("Nonfinite loss or gradient")
        return {"loss": math.log(total), "data": float(values[0]), "physics": float(values[1]),
                "l1": float(values[2]), "total": total}, gradients

    def step(self, coordinates, targets, physics, learning_rate=1e-3, **kwargs):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Positive finite learning rate required")
        metrics, _ = self.objective(coordinates, targets, physics, **kwargs)
        self.session.run(self.update, {self.model.lr_tf: learning_rate})
        self.iteration += 1
        return {"iteration": self.iteration, **metrics}

    def state(self):
        return {"variables": self.session.run(self.variables), "mask": self.mask.copy(), "iteration": self.iteration}

    def restore(self, state):
        arrays = state["variables"]
        if len(arrays) != len(self.variables) or any(a.dtype != v.dtype.as_numpy_dtype or list(a.shape) != v.shape.as_list() or not np.all(np.isfinite(a)) for a, v in zip(arrays, self.variables)):
            raise ValueError("Checkpoint variable schema mismatch")
        if state["mask"].dtype != np.float32 or state["mask"].shape != self.mask.shape or not np.all((state["mask"] == 0) | (state["mask"] == 1)):
            raise ValueError("Checkpoint mask mismatch")
        if type(state["iteration"]) is not int or state["iteration"] < 0:
            raise ValueError("Invalid iteration")
        self.session.run(self.restore_op, dict(zip(self.restore_placeholders, arrays)))
        self.mask, self.iteration = state["mask"].copy(), state["iteration"]


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


class NAdamCheckpoints:
    """Immutable NPZ/JSON states, same-run identity, atomic LATEST, no pickle."""
    def __init__(self, directory, runtime, identity, resume=False):
        self.directory, self.runtime = Path(directory).resolve(), runtime
        self.identity = {**identity, "protocol": "pinn-nadam-only-v1", "own_source_sha256": sha256(Path(__file__)),
                         "external_sources": runtime.source_records, "numeric_profile": runtime.profile,
                         "mode": runtime.mode}
        self.identity = json.loads(json.dumps(self.identity, allow_nan=False))
        self.latest = None
        if not resume:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.run_id = str(uuid.uuid4())
            _json(self.directory / "RUN.json", {"run_id": self.run_id, "identity": self.identity,
                                                "scope": "NAdam only; not complete PINN-SR"})
        else:
            header = json.loads((self.directory / "RUN.json").read_text(encoding="utf-8"))
            if header["identity"] != self.identity:
                raise ValueError("Resume source/data/config/profile identity differs")
            self.run_id = header["run_id"]
            self._restore()

    def _latest(self):
        path = self.directory / "LATEST.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _restore(self):
        latest = self._latest()
        if not latest or set(latest) != {"checkpoint", "metadata_sha256", "run_id"} or latest["run_id"] != self.run_id:
            raise ValueError("Missing or foreign same-run LATEST")
        name = latest["checkpoint"]
        if not isinstance(name, str) or not name.startswith("step_") or not name[5:].isdigit():
            raise ValueError("Invalid checkpoint name")
        folder = (self.directory / name).resolve(strict=True)
        if folder.parent != self.directory or sha256(folder / "state.json") != latest["metadata_sha256"]:
            raise ValueError("Checkpoint metadata integrity failed")
        metadata = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if metadata["run_id"] != self.run_id or metadata["identity"] != self.identity or sha256(folder / "state.npz") != metadata["npz_sha256"]:
            raise ValueError("Checkpoint data integrity failed")
        descriptors = [{"name": v.name, "shape": v.shape.as_list(), "dtype": v.dtype.name} for v in self.runtime.variables]
        if descriptors != metadata["variables"]:
            raise ValueError("Graph state schema differs")
        with np.load(folder / "state.npz", allow_pickle=False) as archive:
            expected = ["variable_%d" % i for i in range(len(descriptors))] + ["mask", "numpy_rng"]
            if sorted(archive.files) != sorted(expected):
                raise ValueError("Unexpected checkpoint array keys")
            state = {"variables": [archive[k].copy() for k in expected[:-2]], "mask": archive["mask"].copy(), "iteration": metadata["iteration"]}
            numpy_rng = archive["numpy_rng"].copy()
        self.runtime.restore(state)
        rng = metadata["numpy_rng"]
        np.random.set_state((rng[0], numpy_rng, rng[1], rng[2], rng[3]))
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        random.setstate(tuples(metadata["python_rng"]))
        self.latest = latest

    def save(self):
        with _writer(self.directory):
            if self._latest() != self.latest:
                raise ValueError("Another writer advanced this run; reopen with resume")
            state = self.runtime.state()
            folder = self.directory / ("step_%08d" % state["iteration"])
            folder.mkdir(exist_ok=False)
            arrays = {"variable_%d" % i: a for i, a in enumerate(state["variables"])}
            numpy_rng = np.random.get_state()
            arrays.update(mask=state["mask"], numpy_rng=numpy_rng[1])
            with (folder / "state.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            metadata = {"identity": self.identity, "run_id": self.run_id, "iteration": state["iteration"],
                        "variables": [{"name": v.name, "shape": v.shape.as_list(), "dtype": v.dtype.name} for v in self.runtime.variables],
                        "npz_sha256": sha256(folder / "state.npz"), "python_rng": random.getstate(),
                        "numpy_rng": [numpy_rng[0], numpy_rng[2], numpy_rng[3], numpy_rng[4]],
                        "tensorflow_random_state": "No stochastic graph operation after canonical initialization; all training state tensors included"}
            _json(folder / "state.json", metadata)
            latest = {"checkpoint": folder.name, "metadata_sha256": sha256(folder / "state.json"), "run_id": self.run_id}
            temporary = self.directory / ("LATEST.%s.tmp" % uuid.uuid4().hex)
            _json(temporary, latest)
            os.replace(str(temporary), str(self.directory / "LATEST.json"))
            self.latest = latest
            return latest


def read_inputs(initialization, sampling, data_path, train_rows=None, physics_rows=None):
    paths = {"initialization": Path(initialization).resolve(strict=True), "sampling": Path(sampling).resolve(strict=True), "data": Path(data_path).resolve(strict=True)}
    bound = {key: {"sha256": sha256(path), "bytes": path.stat().st_size} for key, path in paths.items()}
    if bound["initialization"]["sha256"] != INITIAL_SHA:
        raise ValueError("Only the documented untrained initialization is accepted")
    condition = SAMPLING_DATA.get(bound["sampling"]["sha256"])
    if condition is None or bound["data"]["sha256"] != condition[1]:
        raise ValueError("Sampling cache and data must be a canonical matching noise condition")
    bound["condition"] = condition[0]
    with np.load(paths["initialization"], allow_pickle=False) as archive:
        initial = [archive["variable_%d" % i].copy() for i in range(18)]
    with np.load(paths["sampling"], allow_pickle=False) as archive:
        coordinates = archive["measurement_coordinates"][archive["train_indices"]].copy()
        targets = archive["targets"][archive["train_indices"]].copy()
        physics = np.concatenate([archive["lhs_coordinates"], coordinates], axis=0).astype(np.float32)
        lower, upper = archive["lower"].copy(), archive["upper"].copy()
    for count, rows in ((train_rows, len(coordinates)), (physics_rows, len(physics))):
        if count is not None and (type(count) is not int or count <= 0 or count > rows):
            raise ValueError("Invalid diagnostic row limit")
    coordinates, targets = coordinates[:train_rows], targets[:train_rows]
    physics = physics[:physics_rows]
    bound["selection"] = {"train_rows": len(coordinates), "physics_rows": len(physics), "diagnostic_subset": train_rows is not None or physics_rows is not None}
    return initial, coordinates, targets, physics, lower, upper, bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("open", "known"), required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1, help="Additional fixed-lr NAdam updates in this invocation")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--train-rows", type=int)
    parser.add_argument("--physics-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.chunk <= 0:
        parser.error("Positive steps/chunk required")
    if not args.execute:
        print(json.dumps({"plan_only": True, "scope": "NAdam only, not full PINN", "mode": args.mode,
                          "additional_steps": args.steps, "resume": args.resume, "output": str(args.output)}))
        return
    if args.mode != "known":
        parser.error("Open NAdam stage execution is not validated: nonzero/two-chunk gradient differs. Only known/KC is enabled; open graph APIs remain diagnostic only.")
    values = read_inputs(args.initialization, args.sampling, args.data, args.train_rows, args.physics_rows)
    initial, coordinates, targets, physics, lower, upper, bound = values
    runtime = PINNRuntime(args.mode, lower, upper, initial)
    config = {"seed": 1234, "chunk": args.chunk, "learning_rate": 1e-3, "physics_weight": 1.0, "l1_weight": 1e-7}
    try:
        store = NAdamCheckpoints(args.output, runtime, {"inputs": bound, "config": config}, args.resume)
        if not args.resume:
            store.save()
        for _ in range(args.steps):
            metrics = runtime.step(coordinates, targets, physics, chunk=args.chunk)
            store.save()
            print(json.dumps({"scope": "nadam_only_not_full_pinn", **metrics, "checkpoint": store.latest["checkpoint"]}), flush=True)
        for key, path in (("initialization", args.initialization), ("sampling", args.sampling), ("data", args.data)):
            if sha256(path) != bound[key]["sha256"]:
                raise ValueError("Bound input changed during execution: " + key)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
