"""Create the PINN measurement/LHS input cache from one specified H5; NumPy only.

This is data preparation, not a PINN network/optimizer/training implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import uuid

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_data import ROOT, canonical, safe_output, sha256, write_json_new
from scripts.generate_noise import rng_arrays, restore_rng


def sampling_design(times, grid, rng, sensors=500, time_samples=60, lhs_points=60000):
    """Declared random sampling order, with sensor-major C-flattened coordinates."""
    sensor_flat = rng.choice(grid*grid, sensors, replace=False)
    sy, sx = np.divmod(sensor_flat, grid)
    selected = rng.choice(len(times), time_samples, replace=False)
    x = (2.0*math.pi) * sx.astype(np.float32) / float(grid)
    y = (2.0*math.pi) * sy.astype(np.float32) / float(grid)
    coordinates = np.column_stack([np.repeat(x, time_samples), np.repeat(y, time_samples),
                                    np.tile(times[selected], sensors)]).astype(np.float32)
    rows = sensors*time_samples
    train = rng.choice(rows, int(rows*(1.0-.2)), replace=False)
    mask = np.zeros(rows, dtype=bool)
    mask[train] = True
    lower = np.asarray([0., 0., float(times[0])], dtype=np.float64)
    edge = 2.0*math.pi * float(grid-1) / float(grid)
    upper = np.asarray([edge, edge, float(times[-1])], dtype=np.float64)
    # Three independently permuted stratified axes; preserve arithmetic/draw order.
    cuts = np.linspace(0., 1., lhs_points+1)
    uniform = rng.rand(lhs_points, 3)
    points = np.zeros_like(uniform)
    for axis in range(3):
        points[:, axis] = uniform[:, axis] * (cuts[1:]-cuts[:-1]) + cuts[:-1]
    design = np.zeros_like(points)
    for axis in range(3):
        order = rng.permutation(range(lhs_points))
        design[:, axis] = points[order, axis]
    lhs = (lower + (upper-lower)*design).astype(np.float32)
    return {"measurement_coordinates": coordinates, "train_indices": train.astype(np.int64),
            "validation_indices": np.flatnonzero(~mask).astype(np.int64), "lhs_coordinates": lhs,
            "sensor_flat": sensor_flat.astype(np.int64), "time_indices": selected.astype(np.int64),
            "lower": lower.astype(np.float32), "upper": upper.astype(np.float32),
            "trajectory_id": np.asarray([0], dtype=np.int64)}


def measurement_targets(source, arrays):
    selected = arrays["time_indices"]
    order = np.argsort(selected)
    frames = np.asarray(source["training/vorticity"][0, selected[order]], dtype=np.float32)[np.argsort(order)]
    if not np.isfinite(frames).all():
        raise ValueError("Input frames contain nonfinite/unwritten values")
    n = frames.shape[-1]
    frequency = np.fft.fftfreq(n, d=1.0/n).astype(np.float32)
    ky, kx = np.meshgrid(frequency, frequency, indexing="ij")
    dx, dy = np.where(kx == -n//2, 0., kx), np.where(ky == -n//2, 0., ky)
    denominator = kx*kx + ky*ky
    denominator[0, 0] = 1.
    # NumPy 1.x promoted float32 FFT input to double; NumPy 2.x does not.
    # Lock the original cache's double-FFT semantics explicitly (not a tolerance change).
    potential = np.fft.fft2(frames.astype(np.float64), axes=(-2, -1)) / denominator
    potential[..., 0, 0] = 0.
    u = np.fft.ifft2(1j*dy*potential, axes=(-2, -1)).real.astype(np.float32)
    v = np.fft.ifft2(-1j*dx*potential, axes=(-2, -1)).real.astype(np.float32)
    sy, sx = np.divmod(arrays["sensor_flat"], n)
    return np.stack([u[:, sy, sx].T, v[:, sy, sx].T, frames[:, sy, sx].T], axis=-1).reshape(-1, 3).astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Specified clean OR noisy H5; opened read-only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-design", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    output, source_path = safe_output(args.output), args.input.resolve()
    if output == source_path.parent or output in source_path.parents:
        raise ValueError("Output must not contain the input")
    with h5py.File(source_path, "r") as source:
        if source.attrs.get("status") == "INCOMPLETE":
            raise ValueError("Incomplete input rejected")
        field = source["training/vorticity"]
        times = np.asarray(source["training/time"], dtype=np.float32)
        if (field.shape[1:] != (501, 64, 64) or field.dtype != np.dtype("float32") or
                int(source["training/trajectory_index"][0]) != 0 or
                not np.allclose(times, np.arange(501, dtype=np.float32)*np.float32(.02), atol=1e-6, rtol=0)):
            raise ValueError("Expected trajectory ID 0, N64 and 501 times from 0 to 10")
    binding = {"schema": "gift.sampling-generation.v1", "input_sha256": sha256(source_path),
               "input_name": source_path.name, "python": platform.python_version(),
               "numpy": np.__version__, "h5py": h5py.__version__, "seed": 1234,
               "sensors": 500, "time_samples": 60, "lhs_points": 60000,
               "velocity_fft": "explicit_float64_complex128_original_cache_semantics",
               "sources": {"scripts/generate_sampling.py": sha256(__file__),
                           "scripts/generate_noise.py": sha256(ROOT / "scripts/generate_noise.py"),
                           "scripts/generate_data.py": sha256(ROOT / "scripts/generate_data.py")}}
    if not args.execute:
        print(json.dumps(binding, indent=2))
        return 0
    binding_hash = hashlib.sha256(canonical(binding).encode()).hexdigest()
    if args.resume:
        if (output / "COMPLETE.json").exists():
            raise ValueError("Attempt already complete")
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if run["binding_sha256"] != binding_hash or run["binding"] != binding:
            raise ValueError("Resume input/source/configuration/runtime mismatch")
        receipt = json.loads((output / "design.json").read_text(encoding="utf-8"))
        checkpoint = output / "design.npz"
        if sha256(checkpoint) != receipt["sha256"]:
            raise ValueError("Design checkpoint hash mismatch")
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["attempt_id"]) != run["attempt_id"] or str(saved["binding_sha256"]) != binding_hash:
                raise ValueError("Foreign design checkpoint")
            rng = restore_rng(saved)
            if str(saved["phase"]) != "design_complete_targets_pending":
                raise ValueError("Unexpected checkpoint phase")
            arrays = {key: saved[key].copy() for key in saved.files if key not in
                      {"attempt_id", "binding_sha256", "phase"} and not key.startswith("rng_")}
    else:
        output.mkdir(parents=True, exist_ok=False)
        rng = np.random.RandomState(1234)
        arrays = sampling_design(times, 64, rng)
        run = {"attempt_id": str(uuid.uuid4()), "binding_sha256": binding_hash, "binding": binding}
        write_json_new(output / "run.json", run)
        with (output / "design.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays, **rng_arrays(rng), attempt_id=run["attempt_id"],
                                binding_sha256=binding_hash, phase="design_complete_targets_pending")
        write_json_new(output / "design.json", {"sha256": sha256(output / "design.npz")})
    if args.stop_after_design:
        print(json.dumps({"status": "PAUSED", "phase": "design_complete_targets_pending"}))
        return 0
    with h5py.File(source_path, "r") as source:
        arrays["targets"] = measurement_targets(source, arrays)
    # Targets have no additional random draws. A restart uses the same saved design.
    target = output / "sampling.npz"
    if target.exists():
        # Recover a crash between complete atomic data writing and receipt writing only.
        with np.load(target, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[k], v) for k, v in arrays.items()):
                raise ValueError("Existing sampling output differs; refuse overwrite")
    else:
        temporary = output / ("sampling_" + uuid.uuid4().hex + ".tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.rename(target)
    write_json_new(output / "COMPLETE.json", {"attempt_id": run["attempt_id"],
                   "status": "COMPLETE_GENERATION", "binding_sha256": binding_hash,
                   "sampling_sha256": sha256(target), "arrays": len(arrays),
                   "network_initialization_generated": False, "model_training_performed": False,
                   "input_is_specified_observations": True, "reference_comparison_performed": False})
    print(json.dumps({"status": "COMPLETE_GENERATION", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
