"""Shared paths, raw-data readers, metrics, and create-only output helpers.

The formal experiment code deliberately treats ``results`` as output only.
Scientific inputs are restricted to the external raw HDF5 package and explicitly
selected model parameters. Published reference tables are never model inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SEEDS = (20260820, 20260821, 20260822)
EXPERIMENTS = {
    "M1": "equation_parameter_identification",
    "M2": "recursive_prediction",
    "M3": "cross_resolution_prediction",
    "S1": "high_frequency_branch_ablation",
    "S2": "recursive_local_correction_ablation",
    "S3": "random_seed_stability",
}
EXPECTED_TEST_IDS = np.arange(1000, 1200, dtype=np.int64)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    low_model: Path
    fno2d_model: Path
    fno3d_model: Path
    standard_n64: Path
    dense_n64: Path
    cross_resolution: Path
    fno_training: Path
    fno_training_dt0p02: Path
    fno_test_dt0p02: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        root = Path(root).resolve(strict=True)
        activate_project(root)
        from gift.paths import checkpoint_root, data_root

        data = data_root(root)
        artifacts = checkpoint_root(root)
        return cls(
            root=root,
            low_model=artifacts
            / "fixed_k21_n64_unmasked"
            / "gift_main.pt",
            fno2d_model=artifacts
            / "fno"
            / "fno2d_ntrain1000_terminal_epoch500.pt",
            fno3d_model=artifacts
            / "fno"
            / "fno3d_ntrain1000_terminal_epoch500.pt",
            standard_n64=data / "standard_ns_n64_full_spectrum.h5",
            dense_n64=data
            / "fair_short_horizon"
            / "standard_ns_n64_full_spectrum_test_t4p1_t6p0_dt0p1.h5",
            cross_resolution=data
            / "cross_resolution"
            / "cross_resolution_n96_n128_t4p1_t6p0_dt0p1.h5",
            fno_training=data
            / "fair_short_horizon"
            / "fno1000_n64_t0_t10_dt0p1.h5",
            fno_training_dt0p02=data
            / "fno"
            / "fno1000_n64_t0_t10_dt0p02.h5",
            fno_test_dt0p02=data
            / "fno"
            / "fno_test_n64_n96_n128_dt0p02.h5",
        )

    def require(self, *names: str) -> None:
        for name in names:
            path = Path(getattr(self, name))
            if not path.is_file():
                raise FileNotFoundError(f"required project input is missing: {path}")


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def activate_project(project_root: Path) -> None:
    """Make project packages importable without depending on the working directory."""

    root = Path(project_root).resolve(strict=True)
    for path in (root, root / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _resolve_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def parse_seed_model_specs(
    project_root: Path, specifications: Sequence[str] | None
) -> dict[int, Path]:
    """Resolve ``SEED=PATH`` arguments or the formal default model layout."""

    root = Path(project_root).resolve(strict=True)
    result: dict[int, Path] = {}
    for specification in specifications or ():
        seed_text, separator, path_text = specification.partition("=")
        if not separator:
            raise ValueError("--gift-model must use SEED=PATH")
        seed = int(seed_text)
        if seed not in SEEDS or seed in result:
            raise ValueError(f"invalid or duplicate GIFT training seed: {seed}")
        result[seed] = _resolve_from_root(root, path_text)
    if result and set(result) != set(SEEDS):
        raise ValueError(f"exactly these GIFT seeds are required: {SEEDS}")
    if result:
        return result

    activate_project(root)
    from gift.paths import checkpoint_root
    artifacts = checkpoint_root(root)
    layouts = ("gift/gift_seed_{seed}.pt", "gift/seed_{seed}.pt",
               "high_frequency_branch/gift_seed_{seed}.pt",
               "high_frequency_branch/seed_{seed}.pt")
    for template in layouts:
        candidate = {seed: artifacts / template.format(seed=seed) for seed in SEEDS}
        if all(path.is_file() for path in candidate.values()):
            return {seed: path.resolve(strict=True) for seed, path in candidate.items()}
    expected = ", ".join(template.format(seed=SEEDS[0]) for template in layouts)
    raise FileNotFoundError(
        "the three trained GIFT model files were not found; use repeated "
        f"--gift-model SEED=PATH or one formal layout such as: {expected}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_record(path: Path, project_root: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    display: str
    if project_root is not None:
        try:
            display = resolved.relative_to(Path(project_root).resolve(strict=True)).as_posix()
        except ValueError:
            display = str(resolved)
    else:
        display = str(resolved)
    return {
        "path": display,
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def create_output(path: Path, experiment: str) -> Path:
    output = Path(path).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    output.mkdir(parents=True)
    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(nested) for nested in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_new(path: Path, payload: Any) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_ready(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV output cannot be empty")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one ordered schema")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def write_npz_new(path: Path, arrays: Mapping[str, Any]) -> None:
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **dict(arrays))
        stream.flush()
        os.fsync(stream.fileno())


def finish_output(output: Path, report: Mapping[str, Any]) -> None:
    write_json_new(output / "report.json", report)
    publication = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema": "gift.experiment-manifest.v1",
        "status": report.get("status", "complete"),
        "experiment": report["experiment"],
        "files": [file_record(path, output) for path in sorted(publication)],
    }
    write_json_new(output / "manifest.json", manifest)


def _validate_ids(ids: np.ndarray, expected: np.ndarray, label: str) -> None:
    if ids.dtype != np.int64 or not np.array_equal(ids, expected):
        raise ValueError(f"{label}: trajectory IDs differ from the formal population")


def load_n64_short(paths: ProjectPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return trajectory IDs, context, and t=5..6 truth."""

    paths.require("dense_n64")
    expected_times = np.round(np.arange(41, 61, dtype=np.float64) / 10.0, 12)
    with h5py.File(paths.dense_n64, "r") as handle:
        ids = np.asarray(handle["test/trajectory_index"][:], dtype=np.int64)
        times = np.asarray(handle["test/time"][:], dtype=np.float64)
        field = np.asarray(handle["test/vorticity"][:], dtype=np.float32)
    _validate_ids(ids, EXPECTED_TEST_IDS, "N64 short test")
    if not np.array_equal(times, expected_times) or field.shape != (200, 20, 64, 64):
        raise ValueError("N64 short test time axis or shape differs")
    if not np.isfinite(field).all():
        raise ValueError("N64 short test contains nonfinite values")
    return ids, times[:10], field[:, :10], field[:, 9:20]


def load_n64_long(paths: ProjectPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IDs, t=5:0.5:8 times, and truth from the raw full-spectrum HDF5."""

    paths.require("standard_n64")
    expected_times = np.arange(5.0, 8.0 + 0.25, 0.5, dtype=np.float64)
    with h5py.File(paths.standard_n64, "r") as handle:
        ids = np.asarray(handle["test/trajectory_index"][:], dtype=np.int64)
        all_times = np.asarray(handle["test/time"][:], dtype=np.float64)
        index = [int(np.flatnonzero(np.isclose(all_times, value, atol=1e-12))[0]) for value in expected_times]
        truth = np.asarray(handle["test/vorticity"][:, index], dtype=np.float32)
    _validate_ids(ids, EXPECTED_TEST_IDS, "N64 long test")
    if truth.shape != (200, 7, 64, 64) or not np.isfinite(truth).all():
        raise ValueError("N64 long truth shape or finiteness differs")
    return ids, expected_times, truth


def load_fno_test_dt0p02(
    paths: ProjectPaths,
    grid: int,
    report_times: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the native ``dt=0.02`` FNO context and selected truth frames.

    The stored time axis is checked in full and report frames are addressed by
    integer indices.  No temporal interpolation or test-set normalization is
    performed by this reader.
    """

    if grid not in (64, 96, 128):
        raise ValueError("formal FNO test grids are N64/N96/N128")
    paths.require("fno_test_dt0p02")
    frame_count = 196 if grid == 64 else 96
    stop_time = 8.0 if grid == 64 else 6.0
    expected_times = 4.1 + np.arange(frame_count, dtype=np.float64) * 0.02
    requested_times = np.asarray(report_times, dtype=np.float64)
    if (
        requested_times.ndim != 1
        or requested_times.size == 0
        or not np.isfinite(requested_times).all()
        or np.any(np.diff(requested_times) <= 0.0)
        or requested_times[0] < 4.1 - 2.0e-12
        or requested_times[-1] > stop_time + 2.0e-12
    ):
        raise ValueError(f"N{grid} requested report times are invalid")

    with h5py.File(paths.fno_test_dt0p02, "r") as handle:
        raw_metadata = handle.attrs.get("metadata_json")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        try:
            metadata = json.loads(str(raw_metadata))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("formal FNO test metadata is missing or invalid") from error
        if (
            metadata.get("schema_version") != "gift.fno.test-data.v1"
            or metadata.get("status") != "complete"
            or float(metadata.get("stored_dt", float("nan"))) != 0.02
            or metadata.get("interpolation_used") is not False
        ):
            raise ValueError("formal FNO test metadata differs")

        group = handle[f"N{grid}/test"]
        id_data = group["trajectory_index"]
        time_data = group["time"]
        field_data = group["vorticity"]
        if id_data.shape != (200,) or id_data.dtype != np.dtype("int64"):
            raise ValueError(f"N{grid} FNO test trajectory-index contract differs")
        if time_data.shape != (frame_count,) or time_data.dtype != np.dtype("float64"):
            raise ValueError(f"N{grid} FNO test time-axis contract differs")
        if field_data.shape != (200, frame_count, grid, grid) or field_data.dtype != np.dtype(
            "float32"
        ):
            raise ValueError(f"N{grid} FNO test field contract differs")

        ids = np.asarray(id_data[:], dtype=np.int64)
        times = np.asarray(time_data[:], dtype=np.float64)
        _validate_ids(ids, EXPECTED_TEST_IDS, f"N{grid} FNO test")
        if not np.allclose(times, expected_times, rtol=0.0, atol=2.0e-12):
            raise ValueError(
                f"N{grid} FNO test times differ from t=4.10:0.02:{stop_time:.2f}"
            )

        positions = np.rint((requested_times - 4.1) / 0.02).astype(np.int64)
        if (
            np.any(positions < 0)
            or np.any(positions >= frame_count)
            or not np.allclose(times[positions], requested_times, rtol=0.0, atol=2.0e-12)
        ):
            raise ValueError(f"N{grid} report times are not stored native frames")
        context_times = times[:46]
        if not np.isclose(context_times[-1], 5.0, rtol=0.0, atol=2.0e-12):
            raise ValueError(f"N{grid} FNO context does not end at t=5.0")
        context = np.asarray(field_data[:, :46], dtype=np.float32)
        truth = np.asarray(field_data[:, positions], dtype=np.float32)

    if context.shape != (200, 46, grid, grid) or truth.shape != (
        200,
        len(requested_times),
        grid,
        grid,
    ):
        raise ValueError(f"N{grid} FNO context or selected truth shape differs")
    if not np.isfinite(context).all() or not np.isfinite(truth).all():
        raise ValueError(f"N{grid} FNO context or selected truth contains nonfinite values")
    return ids, context_times, context, times[positions], truth


def load_cross_resolution(paths: ProjectPaths) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return raw 20-frame fields for N64, N96, and N128."""

    ids64, _, context64, truth64 = load_n64_short(paths)
    field64 = np.concatenate((context64, truth64[:, 1:]), axis=1)
    result = {64: (ids64, np.round(np.arange(41, 61) / 10.0, 12), field64)}
    paths.require("cross_resolution")
    expected_times = np.round(np.arange(41, 61, dtype=np.float64) / 10.0, 12)
    with h5py.File(paths.cross_resolution, "r") as handle:
        for grid in (96, 128):
            group = f"N{grid}/test"
            ids = np.asarray(handle[f"{group}/trajectory_index"][:], dtype=np.int64)
            times = np.asarray(handle[f"{group}/time"][:], dtype=np.float64)
            field = np.asarray(handle[f"{group}/vorticity"][:], dtype=np.float32)
            _validate_ids(ids, EXPECTED_TEST_IDS, f"N{grid} cross-resolution test")
            if not np.array_equal(times, expected_times) or field.shape != (200, 20, grid, grid):
                raise ValueError(f"N{grid} cross-resolution time axis or shape differs")
            if not np.isfinite(field).all():
                raise ValueError(f"N{grid} cross-resolution test contains nonfinite values")
            result[grid] = (ids, times, field)
    return result


def load_correction_holdout(paths: ProjectPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IDs 1200..1399 and t=5:0.5:8 truth from the raw FNO-training HDF5."""

    paths.require("fno_training")
    expected_ids = np.arange(1200, 1400, dtype=np.int64)
    expected_times = np.arange(5.0, 8.0 + 0.25, 0.5, dtype=np.float64)
    with h5py.File(paths.fno_training, "r") as handle:
        ids_all = np.asarray(handle["training/trajectory_index"][:], dtype=np.int64)
        times_all = np.asarray(handle["training/time"][:], dtype=np.float64)
        row_lookup = {int(value): index for index, value in enumerate(ids_all)}
        rows = [row_lookup[int(value)] for value in expected_ids]
        columns = [int(np.flatnonzero(np.isclose(times_all, value, atol=1e-12))[0]) for value in expected_times]
        truth = np.asarray(handle["training/vorticity"][rows][:, columns], dtype=np.float32)
    if truth.shape != (200, 7, 64, 64) or not np.isfinite(truth).all():
        raise ValueError("correction holdout shape or finiteness differs")
    return expected_ids, expected_times, truth


def spectral_masks(grid: int) -> dict[str, np.ndarray]:
    modes = np.fft.fftfreq(grid, d=1.0 / grid)
    ky, kx = np.meshgrid(modes, modes, indexing="ij")
    p21 = (np.abs(kx) <= 21) & (np.abs(ky) <= 21)
    p31 = (np.abs(kx) <= 31) & (np.abs(ky) <= 31)
    non_nyquist = (np.abs(kx) < grid // 2) & (np.abs(ky) < grid // 2)
    return {
        "P21": p21,
        "Q21": ~p21,
        "P22_31": p31 & ~p21,
        "outer_P31": ~p31,
        "reliable_outer_P31": (~p31) & non_nyquist,
    }


def project_numpy(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.fft2(value, axes=(-2, -1)) * mask, axes=(-2, -1)
    ).real.astype(np.float32)


def relative_l2(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    left = np.asarray(prediction, dtype=np.float32)
    right = np.asarray(truth, dtype=np.float32)
    if left.shape != right.shape or left.ndim < 3:
        raise ValueError("prediction and truth shapes differ")
    if mask is not None:
        left = project_numpy(left, mask)
        right = project_numpy(right, mask)
    leading = left.shape[:-2]
    numerator = np.linalg.norm((left - right).astype(np.float64).reshape(*leading, -1), axis=-1)
    denominator = np.linalg.norm(right.astype(np.float64).reshape(*leading, -1), axis=-1)
    return numerator / np.maximum(denominator, np.finfo(np.float64).tiny)


def metric_arrays(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    masks = spectral_masks(truth.shape[-1])
    result = {
        "full_relative_l2": relative_l2(prediction, truth),
        "P21_relative_l2": relative_l2(prediction, truth, masks["P21"]),
        "Q21_relative_l2": relative_l2(prediction, truth, masks["Q21"]),
        "P22_31_relative_l2": relative_l2(prediction, truth, masks["P22_31"]),
        "outer_P31_relative_l2": relative_l2(
            prediction, truth, masks["outer_P31"]
        ),
    }
    if bool(masks["reliable_outer_P31"].any()):
        result["reliable_outer_P31_relative_l2"] = relative_l2(
            prediction, truth, masks["reliable_outer_P31"]
        )
    return result


def summarize(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    selected = array[finite]
    return {
        "population": int(array.size),
        "finite_count": int(finite.sum()),
        "mean": float(selected.mean()) if selected.size else None,
        "sample_sd": float(selected.std(ddof=1)) if selected.size > 1 else 0.0 if selected.size else None,
        "median": float(np.median(selected)) if selected.size else None,
        "p95": float(np.quantile(selected, 0.95)) if selected.size else None,
        "maximum": float(selected.max()) if selected.size else None,
    }


def time_metric_rows(
    *, experiment: str, method: str, seed: int | str, times: np.ndarray, values: np.ndarray
) -> list[dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(times):
        raise ValueError("time metric must have shape [trajectory,time]")
    rows = []
    for index, absolute_time in enumerate(times):
        summary = summarize(matrix[:, index])
        rows.append(
            {
                "experiment": experiment,
                "method": method,
                "seed": seed,
                "absolute_time": float(absolute_time),
                "lead_time": float(absolute_time - 5.0),
                **summary,
            }
        )
    return rows


def seed_summary(values: Mapping[int, float]) -> dict[str, Any]:
    ordered = np.asarray([values[seed] for seed in SEEDS], dtype=np.float64)
    if not np.isfinite(ordered).all():
        raise ValueError("seed summary requires three finite values")
    mean = float(ordered.mean())
    sd = float(ordered.std(ddof=1))
    return {
        "values": {str(seed): float(values[seed]) for seed in SEEDS},
        "mean": mean,
        "sample_sd": sd,
        "coefficient_of_variation_percent": float(100.0 * sd / mean) if mean != 0.0 else None,
    }
