"""Original periodic-data glue; PDE-FIND algorithms stay in an external clone.

Only hash-checked upstream function ASTs are executed. No sparse-regression or
polynomial-library implementation is copied here. The data transformations
retain the previous project protocol, including separate noisy-velocity SVD.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import math
import operator
import os
from pathlib import Path
import subprocess
from types import ModuleType

import h5py
import numpy as np
from scipy.signal import savgol_coeffs

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "86911349b9fac82996e35465a13d06205730cc05"
SOURCE_SHA256 = "79D05AEE89900F57A1E2EE91040C09C55B953DAE0C8BF4AAF1CED61DF6B66E2A"
CONDITIONS = ("noise_000", "noise_001", "noise_010")
DERIVATIVES = ["", "w_{x}", "w_{y}", "w_{xx}", "w_{xy}", "w_{yy}"]
FIELDS = ["w", "u", "v", "q"]
RIDGE = {"lam": 1e-5, "d_tol": 5.0, "maxit": 25, "STR_iters": 10,
         "split": 0.8, "normalize": 2, "l0_penalty": None}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest().upper()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest().upper()


def source_record(external_root: Path | None = None) -> dict:
    configured = external_root or os.environ.get("GIFT_EXTERNAL_ROOT")
    if not configured:
        raise ValueError("set GIFT_EXTERNAL_ROOT to the parent of an external pde_find clone")
    directory = (Path(configured).expanduser().resolve(strict=True) / "pde_find").resolve(strict=True)
    if directory == ROOT or ROOT in directory.parents:
        raise ValueError("PDE-FIND source must stay outside the code repository")
    data = os.environ.get("GIFT_DATA_ROOT")
    if data and (Path(data).resolve() == directory or Path(data).resolve() in directory.parents):
        raise ValueError("PDE-FIND source must not be bundled inside the data package")
    path = (directory / "PDE_FIND.py").resolve(strict=True)
    if directory not in path.parents or digest(path) != SOURCE_SHA256:
        raise ValueError("PDE_FIND.py canonical source SHA256 differs")
    head = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise ValueError("PDE-FIND checkout is not at the pinned commit")
    return {"repository": "https://github.com/snagcliffs/PDE-FIND", "commit": head,
            "path": str(path), "sha256": SOURCE_SHA256,
            "license": None, "redistribution_permission": "not_identified"}


class _MembershipArray(np.ndarray):
    """Keep sampled values/order; accelerate only membership queries."""
    def __new__(cls, values, population):
        result = np.asarray(values).view(cls)
        result.membership = np.zeros(population, dtype=bool)
        result.membership[np.asarray(values, dtype=np.int64)] = True
        return result

    def __array_finalize__(self, source):
        self.membership = getattr(source, "membership", None)

    def __contains__(self, item):
        return bool(0 <= int(item) < len(self.membership) and self.membership[int(item)])


class _RandomScope:
    """Same MT19937 seed/choice as upstream, without mutating global NumPy RNG."""
    def __init__(self):
        self.generator = np.random.RandomState(0)
        self.last_indices = None

    def seed(self, seed):
        self.generator.seed(seed)

    def choice(self, population, size, replace=True):
        values = self.generator.choice(population, size, replace=replace)
        self.last_indices = values.copy()
        return _MembershipArray(values, population) if not replace else values


class _NumpyScope:
    def __init__(self):
        self.random = _RandomScope()

    def __getattr__(self, name):
        return getattr(np, name)


def load_upstream(external_root: Path | None = None):
    record = source_record(external_root)
    path = Path(record["path"])
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = ("PolyDiffPoint", "build_Theta", "TrainSTRidge", "STRidge")
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if tuple(node.name for node in nodes) != names:
        raise ValueError("pinned PDE-FIND function selection differs")
    changes = []
    if int(np.__version__.split(".")[0]) >= 2:
        function = next(node for node in nodes if node.name == "STRidge")
        comparisons = [node for node in ast.walk(function) if isinstance(node, ast.Compare)
                       and isinstance(node.left, ast.Name) and node.left.id == "biginds"
                       and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq)
                       and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.List)
                       and not node.comparators[0].elts]
        if len(comparisons) != 1:
            raise ValueError("expected exactly one old empty-index predicate")
        comparison = comparisons[0]
        comparison.left = ast.Call(func=ast.Name(id="len", ctx=ast.Load()),
                                   args=[comparison.left], keywords=[])
        comparison.comparators = [ast.Constant(0)]
        changes.append("NumPy2: one STRidge empty-index comparison uses len(biginds) != 0")
    namespace = _NumpyScope()
    module = ModuleType("gift_external_pde_find")
    module.__dict__.update(np=namespace, LA=np.linalg, itertools=itertools, operator=operator)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"),
         module.__dict__)
    module.source = {**record, "in_memory_compatibility": changes,
                     "random_scope": "isolated RandomState, original seed0 and draw order",
                     "membership_acceleration_only": True}
    return module


def read_observations(path: Path):
    """Read only one original training trajectory; never metadata or truth."""
    with h5py.File(path, "r") as handle:
        group = handle["training"]
        if set(group) != {"time", "trajectory_index", "vorticity"}:
            raise ValueError("training observation fields differ")
        ids = np.asarray(group["trajectory_index"][:], dtype=np.int64)
        times = np.asarray(group["time"][:], dtype=np.float64)
        field = group["vorticity"]
        if not np.array_equal(ids, np.arange(50)) or field.shape != (50, 501, 64, 64):
            raise ValueError("expected the original 50-trajectory N64 training observation set")
        if field.dtype != np.dtype("float32") or times.shape != (501,):
            raise ValueError("observation dtype/time shape differs")
        if not np.allclose(times, np.arange(501) * 0.02, rtol=0, atol=2e-12):
            raise ValueError("observation times differ")
        raw = np.asarray(field[0], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite observations")
    return raw, times


def velocity(raw):
    n = raw.shape[-1]
    frequency = np.fft.fftfreq(n, d=1.0 / n)
    ky, kx = np.meshgrid(frequency, frequency, indexing="ij")
    dx = np.where(kx == -n // 2, 0.0, kx)
    dy = np.where(ky == -n // 2, 0.0, ky)
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0
    psi = np.fft.fft2(raw, axes=(-2, -1)) / k2
    psi[:, 0, 0] = 0.0
    return (np.fft.ifft2(1j * dy * psi, axes=(-2, -1)).real,
            np.fft.ifft2(-1j * dx * psi, axes=(-2, -1)).real)


def truncated_svd(field, rank):
    n = field.shape[-1]
    matrix = field.transpose(1, 2, 0).reshape(n * n, field.shape[0])
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    rebuilt = (left[:, :rank] * singular[:rank]) @ right[:rank]
    result = rebuilt.reshape(n, n, field.shape[0]).transpose(2, 0, 1)
    return result, {"rank": rank, "singular_values_sha256": array_digest(singular),
                    "retained_energy_fraction": float(np.square(singular[:rank]).sum() /
                                                       np.square(singular).sum())}


def derivative_weights(window, order, spacing):
    return savgol_coeffs(window, 5, deriv=order, delta=spacing,
                         pos=window // 2, use="dot").astype(np.float64)


def _periodic(values, weights, axis):
    result = np.zeros_like(values, dtype=np.float64)
    for weight, offset in zip(weights, range(-(len(weights) // 2), len(weights) // 2 + 1)):
        result += float(weight) * np.roll(values, shift=-offset, axis=axis)
    return result


def prepare_rows(raw, times, condition):
    """Project-specific geometry adapter; formal reader fixes N64 and 501 frames."""
    if condition not in CONDITIONS or raw.ndim != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError("invalid condition or observation shape")
    raw = np.asarray(raw, dtype=np.float64)
    n = raw.shape[-1]
    if n < 20 or n % 2 or len(times) != len(raw) or not np.isfinite(raw).all():
        raise ValueError("periodic observations do not support fixed polynomial windows")
    dt, dx = float(times[1] - times[0]), 2.0 * math.pi / n
    if not np.allclose(np.diff(times), dt, rtol=0, atol=2e-12):
        raise ValueError("time steps are not uniform")
    u, v = velocity(raw)
    preprocessing = {"mode": "clean_no_svd"}
    state = raw
    if condition != "noise_000":
        state, w_info = truncated_svd(raw, 26)
        u, u_info = truncated_svd(u, 20)
        v, v_info = truncated_svd(v, 20)
        preprocessing = {"mode": "separate_observed_w_u_v_svd", "w": w_info,
                         "u": u_info, "v": v_info, "ranks_not_retuned": True}
    first, last = (int(round(fraction * (len(raw) - 1))) for fraction in (12 / 150, 130 / 150))
    centres = np.rint(np.linspace(first, last, 60)).astype(np.int64)
    if len(np.unique(centres)) != 60 or centres[0] < 4 or centres[-1] >= len(raw) - 4:
        raise ValueError("time support cannot provide 60 original polynomial centres")
    w = state[centres]
    wt = np.tensordot(state[centres[:, None] + np.arange(-4, 5)[None, :]],
                      derivative_weights(9, 1, dt), axes=([1], [0]))
    wx = _periodic(w, derivative_weights(19, 1, dx), 2)
    wxx = _periodic(w, derivative_weights(19, 2, dx), 2)
    wy = _periodic(w, derivative_weights(9, 1, dx), 1)
    wyy = _periodic(w, derivative_weights(9, 2, dx), 1)
    wxy = (np.roll(wx, -1, axis=1) - np.roll(wx, 1, axis=1)) / (2 * dx)
    q = np.broadcast_to((-4.0 * np.cos(4.0 * (2.0 * math.pi * np.arange(n) / n)))[None, :, None], w.shape)
    fields = dict(w=w, u=u[centres], v=v[centres], q=q, wt=wt, wx=wx, wy=wy,
                  wxx=wxx, wxy=wxy, wyy=wyy)
    rows = {name: value.transpose(1, 2, 0).reshape(n * n * 60, 1) for name, value in fields.items()}
    if not all(np.isfinite(value).all() for value in rows.values()):
        raise ValueError("nonfinite derivative rows")
    return rows, {"preprocessing": preprocessing, "time_indices": centres.tolist(),
                  "row_order": "y,x,time", "rows": n * n * 60,
                  "poly_degree": 5, "time_y_window": 9, "x_window": 19,
                  "dt": dt, "dx": dx, "raw_vorticity_sha256": array_digest(raw)}


def library(upstream, rows, *, known=False):
    if known:
        # Combine in float64 before complex64 rounding, not by summing rounded open columns.
        theta = np.column_stack((rows["w"][:, 0],
                                 rows["u"][:, 0] * rows["wx"][:, 0] + rows["v"][:, 0] * rows["wy"][:, 0],
                                 rows["wxx"][:, 0] + rows["wyy"][:, 0], rows["q"][:, 0]))
        names = ["w", "u*w_x+v*w_y", "w_xx+w_yy", "q"]
    else:
        data = np.hstack([rows[name] for name in FIELDS])
        derivatives = np.hstack([np.ones_like(rows["w"]), *(rows[name] for name in
                                                         ("wx", "wy", "wxx", "wxy", "wyy"))])
        theta, names = upstream.build_Theta(data, derivatives, DERIVATIVES, 2, data_description=FIELDS)
    theta = np.asarray(theta, dtype=np.complex64)
    if theta.shape != (len(rows["w"]), 4 if known else 90) or len(set(names)) != len(names):
        raise ValueError("upstream candidate library shape or descriptions differ")
    if not np.isfinite(theta).all():
        raise ValueError("nonfinite candidate library")
    return theta, names


def regress(upstream, theta, target):
    if theta.dtype != np.complex64 or target.shape != (len(theta), 1):
        raise ValueError("regression dtype or target shape differs")
    coefficients = np.asarray(upstream.TrainSTRidge(theta, target, **RIDGE, print_best_tol=True)).reshape(-1)
    if len(coefficients) != theta.shape[1] or not np.isfinite(coefficients).all():
        raise ValueError("upstream regression returned invalid coefficients")
    return coefficients


def parameter_readout(names, coefficients, *, known=False):
    def value(name):
        scalar = complex(coefficients[names.index(name)])
        if abs(scalar.imag) > 1e-6 * max(1.0, abs(scalar.real)):
            raise ValueError("unexpected imaginary regression coefficient")
        return float(scalar.real)
    return {"nu": value("w_xx+w_yy") if known else 0.5 * (value("w_{xx}") + value("w_{yy}")),
            "beta": -value("u*w_x+v*w_y") if known else -0.5 * (value("uw_{x}") + value("vw_{y}")),
            "gamma": value("q")}
