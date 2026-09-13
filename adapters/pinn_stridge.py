"""First-party boundary around two externally supplied EQDiscovery methods.

Python 3.8 / NumPy 1.21.6 only. No TensorFlow/Torch imports, graph construction,
algorithm vendoring, full PINN training, or complete ADO continuation here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("TrainSTRidge", "STRidge")
EXPECTED_COMMIT = "9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496"
EXPECTED_SOURCE = "C5AE04D1C9E220B912C5D0ABD89D03216B85649C08236F5B9A861F98198BA261"
SOURCE_FILE = "Examples/Discovery with Single Dataset/Vorticity/NS_Vorticity_UniformSetting_OneAdam_PreADOPt.py"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_record(value):
    value = np.asarray(value)
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def _source(external_root):
    configured = external_root or os.environ.get("GIFT_EXTERNAL_ROOT")
    if not configured:
        raise ValueError("Set GIFT_EXTERNAL_ROOT to the external clone parent")
    registry_path = ROOT / "external_sources.json"
    record = json.loads(registry_path.read_text(encoding="utf-8"))["sources"]["pinn_sr"]
    if (record["commit"] != EXPECTED_COMMIT or record["files"][SOURCE_FILE]["sha256"] != EXPECTED_SOURCE
            or record["repository"] != "https://github.com/isds-neu/EQDiscovery"):
        raise ValueError("PINN source registry differs from the audited mapping")
    base = Path(configured).resolve(strict=True)
    directory = (base / record["directory"]).resolve(strict=True)
    protected = [ROOT]
    if os.environ.get("GIFT_DATA_ROOT"):
        protected.append(Path(os.environ["GIFT_DATA_ROOT"]).resolve())
    if base not in directory.parents or any(directory == p or p in directory.parents for p in protected):
        raise ValueError("External PINN source must remain outside code/data packages")
    commit = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    path = (directory / SOURCE_FILE).resolve(strict=True)
    if directory not in path.parents or commit != EXPECTED_COMMIT or _sha(path).upper() != EXPECTED_SOURCE:
        raise ValueError("Pinned PINN commit/source SHA256 differs")
    return path, dict(repository=record["repository"], commit=commit, file=SOURCE_FILE,
                     sha256=EXPECTED_SOURCE.lower(), registry_sha256=_sha(registry_path),
                     adapter_sha256=_sha(__file__))


class _EmptyIndexCompatibility(ast.NodeTransformer):
    """One checked NumPy empty-array compatibility node; no search/solver rewrite."""
    def __init__(self):
        self.edits = 0

    def visit_Compare(self, node):
        if (isinstance(node.left, ast.Name) and node.left.id == "biginds"
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq)
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.List)
                and not node.comparators[0].elts):
            self.edits += 1
            return ast.copy_location(ast.Compare(
                left=ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[node.left], keywords=[]),
                ops=[ast.NotEq()], comparators=[ast.Constant(value=0)]), node)
        return self.generic_visit(node)


class _RandomBoundary:
    def seed(self, seed):
        if seed != 0:
            raise ValueError("Unexpected STRidge split seed")
        self.rng = np.random.RandomState(seed)

    def choice(self, *args, **kwargs):
        # Same selected set as upstream, formal adapter's ascending row order.
        return np.sort(self.rng.choice(*args, **kwargs))


class _LinalgBoundary:
    def __getattr__(self, name):
        return getattr(np.linalg, name)

    def lstsq(self, matrix, target):
        return np.linalg.lstsq(matrix, target, rcond=-1)


class _NumpyBoundary:
    def __init__(self):
        self.random, self.linalg = _RandomBoundary(), _LinalgBoundary()

    def __getattr__(self, name):
        return getattr(np, name)


def _native_methods(path, columns):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest().upper() != EXPECTED_SOURCE:
        raise ValueError("External STRidge source changed before compilation")
    parsed = ast.parse(source.decode("utf-8"))
    classes = [node for node in ast.walk(parsed) if isinstance(node, ast.ClassDef)
               and node.name == "PhysicsInformedNN"]
    if len(classes) != 1:
        raise ValueError("Expected one pinned PINN class")
    selected = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    if [node.name for node in selected] != list(METHODS):
        raise ValueError("Expected exactly two ordered native STRidge methods")
    tree = ast.Module(body=[ast.ClassDef(name="ExternalSTRidge", bases=[], keywords=[],
                                        body=selected, decorator_list=[])], type_ignores=[])
    compatibility = _EmptyIndexCompatibility()
    tree = compatibility.visit(tree)
    if compatibility.edits != 1:
        raise ValueError("Expected exactly one native biginds empty comparison")
    ast.fix_missing_locations(tree)
    namespace = dict(np=_NumpyBoundary(), lambda_history_STRidge=np.zeros((columns, 1)),
                     ridge_append_counter_STRidge=np.empty(0))
    # External function bodies live only in this private in-memory namespace.
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["ExternalSTRidge"]()


def fit_stridge(matrix, target, inherited, l0_penalty=None, *, iteration=0, external_root=None):
    """Return native regression plus explicit bindings; never update a TF graph.

    iteration is the zero-based ADO round (0..5). Later rounds require the first
    round's unchanged l0 value. Caller owns its immutable stage journal and must
    bind the complete physical-row population: this function permits tiny tests.
    """
    if sys.version_info[:2] != (3, 8) or np.__version__ != "1.21.6":
        raise ValueError("Use the documented Python 3.8 / NumPy 1.21.6 PINN environment")
    if type(iteration) is not int or not 0 <= iteration < 6:
        raise ValueError("Expected zero-based ADO iteration 0..5")
    if l0_penalty is None and iteration != 0:
        raise ValueError("Later ADO rounds require the persisted first-round l0 penalty")
    if l0_penalty is not None and (not math.isfinite(float(l0_penalty)) or float(l0_penalty) < 0):
        raise ValueError("Invalid persisted l0 penalty")
    matrix, target, inherited = map(np.asarray, (matrix, target, inherited))
    if (matrix.ndim != 2 or matrix.shape[1] not in (4, 90) or matrix.shape[0] < 5
            or matrix.dtype != np.float32 or target.dtype != np.float32
            or target.shape not in ((matrix.shape[0],), (matrix.shape[0], 1))
            or inherited.shape not in ((matrix.shape[1],), (matrix.shape[1], 1))
            or inherited.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or any(not np.isfinite(value).all() for value in (matrix, target, inherited))):
        raise ValueError("Expected finite float32 library/target and real 4/90-coordinate coefficients")
    inputs = dict(matrix=_array_record(matrix), target=_array_record(target), inherited=_array_record(inherited))
    rows = np.sort(np.random.RandomState(0).choice(len(matrix), int(len(matrix) * .8), replace=False))
    norms = np.linalg.norm(matrix[rows], ord=2, axis=0)
    if not np.isfinite(norms).all() or not (norms > 0).all():
        raise ValueError("Zero/nonfinite selected training-column norm; no epsilon or algorithm substitution")
    path, source_binding = _source(external_root)
    native = _native_methods(path, matrix.shape[1])
    # ADO invokes a single frozen inherited vector throughout this tolerance search.
    snapshot = inherited.astype(np.float64).reshape(-1, 1).copy()
    native.lambda_w = "first_party_inherited_snapshot"
    native.sess = SimpleNamespace(run=lambda token: snapshot.copy())
    native.it = iteration
    values = native.TrainSTRidge(matrix, target.reshape(-1, 1), 1e-5, 1., 100, 10,
                                l0_penalty, 2, .8, False)
    coefficients = np.asarray(values[0], dtype=np.float64).reshape(-1).copy()
    fixed_l0 = float(native.l0_penalty_0 if l0_penalty is None else l0_penalty)
    histories = values[1:5]
    if any(len(value) != len(histories[0]) for value in histories):
        raise ValueError("Native accepted-history arrays disagree")
    accepted = [dict(kind="initial" if index == 0 else "accepted", score=float(score),
                     residual_mse=float(mse), l0_cost=float(cost),
                     reported_tolerance_after_accept_increment=float(tolerance))
                for index, (score, mse, cost, tolerance) in enumerate(zip(*histories))]
    best_tolerance, best_error = float(values[5][0]), float(histories[0][-1])
    if not np.isfinite(coefficients).all() or not all(math.isfinite(x) for x in (fixed_l0, best_tolerance, best_error)):
        raise ValueError("Native STRidge produced nonfinite results")
    after = dict(matrix=_array_record(matrix), target=_array_record(target), inherited=_array_record(inherited))
    if after != inputs:
        raise RuntimeError("Native STRidge unexpectedly mutated caller inputs")
    return dict(coefficients=coefficients, l0_penalty=fixed_l0, best_tolerance=best_tolerance,
                best_error=best_error, accepted_history=accepted, source_binding=source_binding,
                input_binding=dict(inputs, training_rows=_array_record(rows), iteration=iteration,
                                   supplied_l0_penalty=None if l0_penalty is None else float(l0_penalty)),
                runtime=dict(python=platform.python_version(), numpy=np.__version__,
                             omp_threads=os.environ.get("OMP_NUM_THREADS"),
                             mkl_threads=os.environ.get("MKL_NUM_THREADS"),
                             openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS")),
                audit=dict(scope="native_STRidge_only_not_complete_PINN", inputs_unchanged=True,
                           history_scope="initial_plus_accepted_only_not_all_100_trials",
                           tolerance_history_semantics="accepted entries report tolerance after increment; use best_tolerance for selected tolerance",
                           inherited_float64=True, training_rows_sorted=True, lstsq_rcond=-1,
                           empty_comparison_ast_edits=1, full_population_acceptance_performed=False))
