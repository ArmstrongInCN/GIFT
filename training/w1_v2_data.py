"""The shared W1-v2 observation window, with no test access during training.

Fields use [trajectory, local_time, y, x], physical vorticity units. The source
frames 250..400, optionally strided, are copied without interpolation; local
time is 0..3 seconds and absolute_time remains 5..8. Only training and checkpoint validation exist
in the derived container.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

SCHEMA = "gift.w1-fair151.window.v2"
SPARSE_SCHEMA = "gift.w1-sparse.window.v3"
SUPPORTED_SCHEMAS = (SCHEMA, SPARSE_SCHEMA)
SAMPLE_SEED = 2026100301
SUBSET_SEED = 2026100304
LOCAL_TIMES = np.arange(151, dtype=np.float64) * .02


def window_manifest(training_count=1000, time_stride=1, subset_seed=SUBSET_SEED):
    """Declare canonical train IDs and observed frames before accessing any field.

    The subset draw is independent of epoch ordering and model initialization.
    Full-data defaults retain canonical row order. Reduced data use the first
    ``training_count`` entries of one fixed permutation of the original 1000 IDs.
    A stride must retain both endpoints of the three-second observation window.
    """
    for name, value in (("training_count", training_count), ("time_stride", time_stride),
                        ("subset_seed", subset_seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    if not 2 <= training_count <= 1000:
        raise ValueError("training_count must be between 2 and 1000")
    if time_stride < 1 or time_stride > 150 or 150 % time_stride:
        raise ValueError("time_stride must divide the 150 source intervals")
    if subset_seed < 0:
        raise ValueError("subset_seed must be nonnegative")
    ids = (np.arange(1000) if training_count == 1000 else
           np.random.default_rng(subset_seed).permutation(1000)[:training_count])
    frames = np.arange(250, 401, time_stride, dtype=np.int64)
    times = LOCAL_TIMES[::time_stride]
    return dict(training_count=int(training_count), time_stride=int(time_stride),
                subset_seed=int(subset_seed), training_ids=ids.tolist(),
                validation_ids=np.arange(1000, 1020).tolist(),
                source_frame_indices=frames.tolist(), query_times=times.tolist(),
                points=len(times), future_steps=len(times) - 1,
                observation_dt_seconds=float(.02 * time_stride),
                observation_interval=[5., 8.], units="raw_vorticity",
                axes=["trajectory", "local_time", "y", "x"])


def _manifest_text(manifest):
    """Canonical JSON binds the actual IDs, physical times and source frame indices."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_manifest(source):
    """Verify a sparse window's independently reproducible selection declaration."""
    try:
        text = source.attrs["manifest_json"]
        manifest = json.loads(text)
        expected = window_manifest(manifest["training_count"], manifest["time_stride"], manifest["subset_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid sparse window manifest") from error
    if manifest != expected or source.attrs.get("manifest_sha256") != hashlib.sha256(
            _manifest_text(expected).encode("utf-8")).hexdigest():
        raise ValueError("sparse window manifest IDs/grid/receipt differ")
    return expected


def epoch_order(population: int, epoch: int, seed: int = SAMPLE_SEED):
    """One complete permutation, independent of model/dropout/init RNG streams."""
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(population)


def prepare_window(source, destination, *, fixture=False, training_count=1000, time_stride=1,
                   subset_seed=SUBSET_SEED):
    """Create a new train/validation-only HDF5, or verify its completed receipt.

    Each decoded split is fingerprinted while copying. A partial destination
    is never treated as complete or silently replaced. External data stay read
    only; no test group or auxiliary validation trajectory is accessed.
    """
    manifest = window_manifest(training_count, time_stride, subset_seed)
    schema = SCHEMA if training_count == 1000 and time_stride == 1 else SPARSE_SCHEMA
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination.exists():
        with h5py.File(destination, "r") as existing:
            if existing.attrs.get("schema") != schema or existing.attrs.get("status") != "complete":
                raise ValueError("existing W1 window is incomplete or incompatible")
            if existing.attrs["source_file"] != str(source) or bool(existing.attrs["fixture_only"]) != fixture:
                raise ValueError("W1 window source/fixture configuration differs")
            if schema == SPARSE_SCHEMA and _read_manifest(existing) != manifest:
                raise ValueError("W1 window selection/grid configuration differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as original, h5py.File(destination, "x") as output:
        output.attrs.update(schema=schema, status="building", source_file=str(source),
                            source_bytes=source.stat().st_size, fixture_only=fixture,
                            trajectory_id_scheme="canonical", source_frame_start=250,
                            source_frame_end=400, prediction_origin_seconds=5.)
        if schema == SPARSE_SCHEMA:
            text = _manifest_text(manifest)
            output.attrs.update(manifest_json=text, manifest_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        frames = np.asarray(manifest["source_frame_indices"], dtype=np.int64)
        local_times = np.asarray(manifest["query_times"], dtype=np.float64)
        for name, ids_expected in (("training", np.arange(1000)),
                                   ("validation", np.arange(1000, 1020))):
            group = original[name]
            field, ids = group["vorticity"], np.asarray(group["trajectory_index"][:])
            times = np.asarray(group["time"][:])
            if field.dtype != np.float32 or field.shape != (len(ids), 501, 64, 64):
                raise ValueError("expected original float32 [B,501,64,64]")
            if not np.allclose(times, np.arange(501) * .02, rtol=0, atol=2e-12):
                raise ValueError("source time grid differs")
            if not fixture and not np.array_equal(ids, ids_expected):
                raise ValueError("source canonical training/validation IDs differ")
            if not len(ids) or len(set(map(int, ids))) != len(ids):
                raise ValueError("invalid split IDs")
            if name == "training" and schema == SPARSE_SCHEMA:
                selected = np.asarray(manifest["training_ids"], dtype=np.int64)
                id_rows = {int(ident): row for row, ident in enumerate(ids)}
                if not all(int(ident) in id_rows for ident in selected):
                    raise ValueError("source is missing selected training IDs")
                rows = [id_rows[int(ident)] for ident in selected]
                ids = selected
            else:
                rows = range(len(ids))
            target = output.create_group(name)
            target["trajectory_index"] = ids
            target["time"] = local_times
            target["absolute_time"] = times[frames]
            copied = target.create_dataset("vorticity", shape=(len(ids), len(frames), 64, 64),
                                           dtype="f4", chunks=(1, len(frames), 64, 64), compression="lzf")
            fingerprint = hashlib.sha256()
            for row, source_row in enumerate(rows):
                values = np.ascontiguousarray(field[source_row, 250:401:time_stride], dtype=np.float32)
                if not np.isfinite(values).all() or np.any(np.linalg.norm(values.reshape(len(frames), -1), axis=1) == 0):
                    raise ValueError("source has nonfinite/zero-norm observation")
                copied[row] = values
                fingerprint.update(values.tobytes())
            target.attrs["observations_sha256"] = fingerprint.hexdigest()
        output.attrs["status"] = "complete"
        output.flush()
    return destination


class WindowBank:
    """Resident declared-grid observations; training-only FNO3D statistics.

    Legacy full151 receipts keep their original binding for compatible resume.
    Sparse windows additionally bind the seeded subset, frame indices and grid.
    """

    def __init__(self, path, *, fixture=False):
        self.path = Path(path).resolve(strict=True)
        with h5py.File(self.path, "r") as source:
            schema = source.attrs.get("schema")
            if schema not in SUPPORTED_SCHEMAS or source.attrs.get("status") != "complete":
                raise ValueError("not a completed shared W1-v2 window")
            if bool(source.attrs["fixture_only"]) != fixture:
                raise ValueError("fixture/formal window differs")
            if set(source.keys()) != {"training", "validation"}:
                raise ValueError("shared training window must contain only training and validation")
            manifest = _read_manifest(source) if schema == SPARSE_SCHEMA else window_manifest()
            self.times = self.query_times = np.asarray(manifest["query_times"], dtype=np.float64)
            self.future_steps = manifest["future_steps"]
            self.time_stride = manifest["time_stride"]
            self.binding = dict(schema=schema, file=str(self.path), source_file=source.attrs["source_file"],
                                fixture_only=fixture, observation_interval=[5., 8.], points=151)
            if schema == SPARSE_SCHEMA:
                self.binding.update({key: value for key, value in manifest.items()
                                     if key not in ("training_ids", "validation_ids")})
                self.binding["manifest_sha256"] = source.attrs["manifest_sha256"]
            for name, expected in (("training", np.asarray(manifest["training_ids"])),
                                   ("validation", np.asarray(manifest["validation_ids"]))):
                group = source[name]
                value = np.asarray(group["vorticity"], dtype=np.float32)
                ids = np.asarray(group["trajectory_index"], dtype=np.int64)
                if group["vorticity"].dtype != np.float32 or value.shape != (len(ids), len(self.times), 64, 64) or not np.isfinite(value).all():
                    raise ValueError("invalid fixed-window fields")
                if not len(ids) or len(np.unique(ids)) != len(ids) or np.any(np.linalg.norm(
                        value.reshape(len(ids), len(self.times), -1), axis=2) == 0):
                    raise ValueError("invalid fixed-window identities/zero-norm targets")
                if not fixture and not np.array_equal(ids, expected):
                    raise ValueError("fixed-window identities differ")
                if not np.allclose(group["time"][:], self.times, atol=2e-12, rtol=0):
                    raise ValueError("local time grid differs")
                if not np.allclose(group["absolute_time"][:], 5 + self.times, atol=2e-12, rtol=0):
                    raise ValueError("absolute observation interval differs")
                fingerprint = hashlib.sha256(memoryview(value).cast("B")).hexdigest()
                if fingerprint != group.attrs["observations_sha256"]:
                    raise ValueError("decoded shared observations differ from their copy receipt")
                setattr(self, name, value)
                setattr(self, name + "_ids", ids)
                self.binding[name] = dict(ids=ids.tolist(), observations_sha256=fingerprint)

    def batch(self, rows, device, *, validation=False):
        """Return initial [B,64,64] and truth [B,T,64,64]; no history tensor."""
        values = self.validation if validation else self.training
        truth = torch.from_numpy(np.ascontiguousarray(values[rows])).to(device)
        return truth[:, 0], truth

    def normalization(self, method):
        """Fit native position/time mean and unbiased SD using training only.

        FNO3D uses [y,x,1] input and [y,x,T-1] output statistics; every other
        method consumes raw fields. No validation/test statistic is included.
        """
        if method != "fno3d":
            return {"kind": "none"}
        series = torch.from_numpy(self.training).permute(0, 2, 3, 1)
        mean, std = series.mean(0), series.std(0, correction=1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid training FNO3D normalizer")
        return dict(kind="paper_UnitGaussianNormalizer", eps=1e-5, validation_or_test_used=False,
                    input_mean=mean[..., :1].contiguous(), input_std=std[..., :1].contiguous(),
                    output_mean=mean[..., 1:].contiguous(), output_std=std[..., 1:].contiguous())


def main():
    """CLI creates only the common train/validation window and its provenance."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-count", type=int, default=1000)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--subset-seed", type=int, default=SUBSET_SEED)
    args = parser.parse_args()
    path = prepare_window(args.source, args.output, training_count=args.training_count,
                          time_stride=args.time_stride, subset_seed=args.subset_seed)
    with h5py.File(path) as source:
        print(json.dumps({"path": str(path), "schema": source.attrs["schema"], "status": source.attrs["status"]}))


if __name__ == "__main__":
    main()
