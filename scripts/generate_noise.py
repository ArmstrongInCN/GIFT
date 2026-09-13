"""Derive paired 1%/10% observations from a specified clean H5, without training."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from scripts.generate_data import ROOT, atomic_json, canonical, safe_output, sha256, write_json_new

GROUPS = ("training", "validation")
FRACTIONS = {"noise_001": .01, "noise_010": .10}


def rng_arrays(rng):
    algorithm, keys, position, cached, gaussian = rng.get_state()
    return {"rng_algorithm": np.asarray(algorithm), "rng_keys": keys,
            "rng_position": np.int64(position), "rng_has_gauss": np.int64(cached),
            "rng_cached_gaussian": np.float64(gaussian)}


def restore_rng(saved):
    if str(saved["rng_algorithm"]) != "MT19937":
        raise ValueError("Only this attempt's MT19937 state is supported")
    rng = np.random.RandomState()
    rng.set_state(("MT19937", saved["rng_keys"], int(saved["rng_position"]),
                   int(saved["rng_has_gauss"]), float(saved["rng_cached_gaussian"])))
    return rng


def group_statistics(field):
    """Original order: whole trajectories, float64 sums, then population variance."""
    count, total, squares = 0, 0., 0.
    for row in range(field.shape[0]):
        block = np.asarray(field[row], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Clean data contains nonfinite/unwritten values")
        count += block.size
        total += float(np.sum(block, dtype=np.float64))
        squares += float(np.sum(block * block, dtype=np.float64))
    mean = total / count
    sigma = math.sqrt(max(squares / count - mean * mean, 0.))
    if sigma <= 0:
        raise ValueError("Clean group standard deviation must be positive")
    return {"count": count, "mean": mean, "std_ddof0": sigma, "sum_square": squares}


def inspect_clean(source, pilot):
    groups = {}
    if "noise_fraction" in source.attrs or source.attrs.get("status") == "INCOMPLETE":
        raise ValueError("Input must be a complete clean dataset, not a noise output")
    for name in GROUPS:
        group = source[name]
        ids = np.asarray(group["trajectory_index"], dtype=np.int64)
        times = np.asarray(group["time"], dtype=np.float64)
        shape = tuple(group["vorticity"].shape)
        if len(shape) != 4 or shape[0] != len(ids) or shape[1] != len(times) or shape[-2:] != (64, 64):
            raise ValueError("Expected [trajectory,time,64,64] and matching axes")
        if not len(ids) or not len(times) or group["vorticity"].dtype != np.dtype("float32"):
            raise ValueError("Expected nonempty float32 clean fields")
        if not pilot:
            expected = np.arange(50) if name == "training" else np.arange(50, 70)
            if not np.array_equal(ids, expected) or len(times) != 501 or not np.allclose(times, np.arange(501)*.02, atol=2e-12, rtol=0):
                raise ValueError("Formal clean IDs/times mismatch; use --pilot only for non-formal testing")
        groups[name] = {"shape": list(shape), "ids": ids.tolist(), "times": times.tolist()}
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Allow reduced clean input; never formal output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=32, help="Number of 16-frame blocks")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.checkpoint_every < 1 or (args.stop_after_chunks is not None and args.stop_after_chunks < 0):
        parser.error("Invalid checkpoint/pause interval")
    output, clean = safe_output(args.output), args.clean.resolve()
    if output == clean.parent or output in clean.parents:
        raise ValueError("Output must not contain the input")
    with h5py.File(clean, "r") as source:
        groups = inspect_clean(source, args.pilot)
    binding = {"schema": "gift.noise-generation.v1", "input_sha256": sha256(clean),
               "input_name": clean.name, "pilot": args.pilot, "groups": groups,
               "seed": 0, "rng": "MT19937", "scaling": "std(clean_group,ddof=0)",
               "group_order": list(GROUPS), "frame_chunk": 16, "fractions": FRACTIONS,
               "python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__,
               "sources": {"scripts/generate_noise.py": sha256(__file__),
                           "scripts/generate_data.py": sha256(ROOT / "scripts/generate_data.py")}}
    if not args.execute:
        print(json.dumps(binding, indent=2))
        return 0
    binding_hash = hashlib.sha256(canonical(binding).encode()).hexdigest()
    tasks = [(name, row, start, min(start+16, shape[1])) for name in GROUPS
             for shape in [groups[name]["shape"]] for row in range(shape[0])
             for start in range(0, shape[1], 16)]
    if args.resume:
        if (output / "COMPLETE.json").exists():
            raise ValueError("Attempt already complete")
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if run["binding"] != binding or run["binding_sha256"] != binding_hash:
            raise ValueError("Resume input/source/configuration/runtime mismatch")
        latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
        checkpoint = output / latest["file"]
        if checkpoint.parent != output or not checkpoint.name.startswith("checkpoint_") or sha256(checkpoint) != latest["sha256"]:
            raise ValueError("Checkpoint path/hash mismatch")
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["attempt_id"]) != run["attempt_id"] or str(saved["binding_sha256"]) != binding_hash:
                raise ValueError("Foreign checkpoint rejected")
            position, rng = int(saved["position"]), restore_rng(saved)
        stats, mode = run["statistics"], "r+"
    else:
        output.mkdir(parents=True, exist_ok=False)
        with h5py.File(clean, "r") as source:
            stats = {name: group_statistics(source[name + "/vorticity"]) for name in GROUPS}
        run = {"attempt_id": str(uuid.uuid4()), "binding_sha256": binding_hash,
               "binding": binding, "statistics": stats}
        write_json_new(output / "run.json", run)
        position, rng, mode = 0, np.random.RandomState(0), "x"
    with ExitStack() as stack:
        source = stack.enter_context(h5py.File(clean, "r"))
        outputs = {name: stack.enter_context(h5py.File(output / (name + ".h5"), mode)) for name in FRACTIONS}
        for name, handle in outputs.items():
            if mode == "x":
                handle.attrs.update(attempt_id=run["attempt_id"], binding_sha256=binding_hash,
                                    status="INCOMPLETE", noise_fraction=FRACTIONS[name], noise_seed=0,
                                    noise_formula="w+eta*std(clean_group,ddof=0)*Z", pilot=args.pilot)
                for group_name in GROUPS:
                    info = groups[group_name]
                    group = handle.create_group(group_name)
                    group.create_dataset("trajectory_index", data=np.asarray(info["ids"], dtype=np.int64))
                    group.create_dataset("time", data=np.asarray(info["times"], dtype=np.float64))
                    group.create_dataset("vorticity", shape=info["shape"], dtype="f4",
                                         chunks=(1, min(16, info["shape"][1]), 64, 64), compression="lzf", fillvalue=np.nan)
            elif handle.attrs.get("attempt_id") != run["attempt_id"] or handle.attrs.get("binding_sha256") != binding_hash:
                raise ValueError("Output H5 identity mismatch")

        def save(next_position):
            for handle in outputs.values():
                handle.flush()
            path = output / ("checkpoint_" + uuid.uuid4().hex + ".npz")
            with path.open("xb") as stream:
                np.savez(stream, attempt_id=run["attempt_id"], binding_sha256=binding_hash,
                         position=np.int64(next_position), **rng_arrays(rng))
            atomic_json(output / "latest.json", {"file": path.name, "sha256": sha256(path)})

        if mode == "x":
            save(0)
        if not 0 <= position <= len(tasks):
            raise ValueError("Invalid checkpoint task position")
        for index in range(position, len(tasks)):
            if args.stop_after_chunks is not None and index-position >= args.stop_after_chunks:
                save(index)
                print(json.dumps({"status": "PAUSED", "next_chunk": index, "total_chunks": len(tasks)}))
                return 0
            name, row, start, stop = tasks[index]
            clean_values = np.asarray(source[name + "/vorticity"][row, start:stop], dtype=np.float32)
            if not np.isfinite(clean_values).all():
                raise ValueError("Nonfinite clean chunk")
            z = rng.standard_normal(clean_values.shape)
            sigma = stats[name]["std_ddof0"]
            for condition, handle in outputs.items():
                noisy = (clean_values.astype(np.float64) + FRACTIONS[condition] * sigma * z).astype(np.float32)
                handle[name + "/vorticity"][row, start:stop] = noisy
            if (index + 1) % args.checkpoint_every == 0:
                save(index + 1)
        save(len(tasks))
        for handle in outputs.values():
            handle.attrs["status"] = "COMPLETE_PILOT" if args.pilot else "COMPLETE_GENERATION"
    write_json_new(output / "COMPLETE.json", {"attempt_id": run["attempt_id"],
                   "status": "COMPLETE_PILOT" if args.pilot else "COMPLETE_GENERATION",
                   "binding_sha256": binding_hash,
                   "outputs": {name + ".h5": sha256(output / (name + ".h5")) for name in FRACTIONS},
                   "clean_reintegrated_here": False, "source_is_specified_clean_input": True,
                   "reference_comparison_performed": False})
    print(json.dumps({"status": "COMPLETE_PILOT" if args.pilot else "COMPLETE_GENERATION", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
