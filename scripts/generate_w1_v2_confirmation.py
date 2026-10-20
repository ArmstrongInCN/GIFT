"""Generate the prespecified independent W1-v2 confirmation cohort.

The default command only prints a plan. Full generation requires an explicit
protocol/development lock and --run-generation; no trained model or reference
solution is read. Existing outputs are preserved, with own-attempt batch resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.generate_data import draw_parameters
from experiments.additional.w1_v2_evaluate import evaluation_grid, sparse_training_contract

SEED = 2026100302
TRAJECTORY_IDS = np.arange(100000, 100180, dtype=np.int64)
INTERNAL_DT = .005
SAVED_STEPS = np.arange(1000, 1601, 4, dtype=np.int64)
ABSOLUTE_TIMES = SAVED_STEPS.astype(np.float64) * INTERNAL_DT
LOCAL_TIMES = ABSOLUTE_TIMES - 5.


def _sha256(path):
    """Record code/lock provenance without reading any solution data."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path, value):
    """Create a new artifact; never replace prior generation evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def confirmation_plan(query_times=None, *, profile=None):
    """Return the fixed cohort, draw rule and numerical contract before generation."""
    grid = evaluation_grid(query_times)
    return {
        "schema": "gift.w1-v2-confirmation.v1", "seed": SEED,
        "rng": "numpy.default_rng / PCG64", "trajectory_id_scheme": "canonical",
        "trajectory_ids": TRAJECTORY_IDS.tolist(), "population": 180,
        "initial_condition_distribution": "same independent four-vortex draw as scripts.generate_data.draw_parameters",
        "parameter_axis": ["circulation", "sigma", "center_x", "center_y"],
        "parameter_shape": [180, 4, 4], "grid": 64, "padding_grid": 99,
        "physical_domain": "[0,2pi)^2, array axes [y,x]", "viscosity": .01,
        "forcing_amplitude": 1., "forcing_wavenumber": 4,
        "vorticity_forcing": "-4*cos(4*y)", "internal_dt": INTERNAL_DT,
        "integrate_from": 0., "integrate_until": 8., "internal_steps": 1600,
        "save_interval": .02, "absolute_time": ABSOLUTE_TIMES.tolist(),
        "time": LOCAL_TIMES.tolist(), "vorticity_shape": [180, 151, 64, 64],
        "evaluation_query_times": grid.tolist(), "evaluation_absolute_time": (5+grid).tolist(),
        "evaluation_dt": float(grid[1]), "evaluation_points": len(grid),
        "evaluation_future_steps": len(grid)-1,
        "evaluation_reference_selection": "exact saved nodes only; no interpolation",
        "vorticity_dtype": "float32", "uncompressed_field_bytes": 180 * 151 * 64 * 64 * 4,
        "solver": "existing even-full-spectrum Fourier pseudospectral ETDRK4",
        "selection": "all prescribed draws; no selection by prediction accuracy or trajectory appearance",
        "purpose": ("independent in-distribution confirmation after protocol and fixed training-configuration lock; no development search"
                    if profile == "single_native76_n300" else
                    "independent in-distribution confirmation after protocol and development lock"),
    }


def locked_configuration(path):
    """Require the legacy development lock or the explicit single-run configuration lock.

    The single-run profile has no development search. Its four native recipes are
    fixed before generation instead; the legacy default and lock remain unchanged.
    """
    path = Path(path).expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    single = value.get("profile") == "single_native76_n300"
    if single:
        if (value.get("schema") != "gift.w1-single-native.protocol-lock.v1"
                or value.get("protocol_locked") is not True or value.get("configurations_locked") is not True):
            raise ValueError("single-run confirmation requires the explicit protocol/configurations lock")
        configurations = value.get("training_configurations")
        if not isinstance(configurations, dict) or set(configurations) != {"anie_ns", "fno2d", "fno3d", "GIFT"}:
            raise ValueError("single-run lock requires all four prespecified method configurations")
    else:
        if value.get("protocol_locked") is not True or value.get("development_locked") is not True:
            raise ValueError("confirmation requires both protocol_locked=true and development_locked=true")
        configurations = value.get("selected_configurations")
    if (not isinstance(configurations, dict) or not configurations
            or any(not isinstance(item, dict) or not item for item in configurations.values())):
        raise ValueError("lock must contain nonempty selected_configurations for the declared methods")
    protocol = Path(value.get("protocol_file", ""))
    if not protocol.is_absolute() or not protocol.is_file():
        raise ValueError("protocol_file must name an existing absolute file")
    grid = evaluation_grid(value.get("evaluation_query_times"), dt=value.get("evaluation_dt"))
    if single and len(grid) != 76:
        raise ValueError("single_native76_n300 requires the76-point .04-second grid")
    for key, expected in (("observation_points", len(grid)), ("future_steps", len(grid)-1)):
        if key in value and value[key] != expected:
            raise ValueError("confirmation lock " + key + " differs from evaluation_query_times")
    contract = sparse_training_contract(grid)
    if contract:
        required = ("evaluation_query_times", "evaluation_dt", "observation_points", "future_steps",
                    "training_count", "training_subset_seed")
        if not all(key in value for key in required):
            raise ValueError("sparse confirmation requires an explicit locked grid and training subset")
        if value["training_count"] != contract["training_count"] or value["training_subset_seed"] != contract["subset_seed"]:
            raise ValueError("sparse confirmation requires the prespecified training subset/count/seed")
    return {"lock_file": str(path), "lock_sha256": _sha256(path),
            "protocol_file": str(protocol), "protocol_sha256": _sha256(protocol),
            "evaluation_query_times": grid.tolist(), "evaluation_dt": float(grid[1]),
            "evaluation_points": len(grid), "evaluation_future_steps": len(grid)-1,
            "lock_snapshot": value}


def integrate_batch(parameters, *, saved_steps=SAVED_STEPS, device="cpu"):
    """Integrate from t=0; return float32 [B,T,64,64] on the requested real nodes.

    ``saved_steps`` uses integer .005-step indices. The formal caller always
    uses 1000:4:1600; small indices permit a short CPU numerical contract test.
    No interpolation, trained model, future observation or rejection sampling
    enters this function. A nonfinite field aborts generation rather than being
    dropped from the prescribed population.
    """
    import torch
    from even_full_spectrum_ns import (
        EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver, build_initial_hat,
    )
    parameters = np.asarray(parameters, dtype=np.float64)
    steps = np.asarray(saved_steps)
    if parameters.ndim != 3 or parameters.shape[1:] != (4, 4) or not len(parameters) or not np.isfinite(parameters).all():
        raise ValueError("parameters must be finite, nonempty [B,4,4]")
    if (steps.ndim != 1 or not len(steps) or steps.dtype.kind not in "iu"
            or steps[0] < 0 or np.any(np.diff(steps) <= 0)):
        raise ValueError("saved_steps must be strictly increasing nonnegative integers")
    configuration = EvenFullSpectrumConfiguration(grid=64, padding_grid=99, viscosity=.01,
                                                   forcing_amplitude=1., forcing_wavenumber=4, dt=INTERNAL_DT)
    solver = EvenFullSpectrumNSSolver(configuration, device=device)
    output = np.empty((len(parameters), len(steps), 64, 64), dtype=np.float32)
    with torch.inference_mode():
        state = build_initial_hat(parameters, grid=64, device=device)
        save_index = 0
        for step in range(int(steps[-1]) + 1):
            if step == int(steps[save_index]):
                field = torch.fft.ifft2(state).real.cpu().numpy()
                if not np.isfinite(field).all():
                    raise FloatingPointError(f"nonfinite prescribed trajectory at solver step {step}")
                output[:, save_index] = field
                save_index += 1
                if save_index == len(steps):
                    break
            state = solver.advance(state)
    return output, solver.metadata


def generate_confirmation(lock_file, output, *, device="cpu", batch_size=8, resume=False):
    """Commit all 180 fixed trajectories to a new container; resume only this attempt."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    lock = locked_configuration(lock_file)
    output = Path(output).expanduser().resolve()
    plan = confirmation_plan(lock["evaluation_query_times"], profile=lock["lock_snapshot"].get("profile"))
    parameters = draw_parameters(180, SEED)
    dataset = output / "confirmation_n64_dt0p02.h5"
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "plan.json", plan)
        _write_json(output / "lock.json", lock)
        _write_json(output / "initial_conditions.json", {
            "trajectory_id_scheme": "canonical", "trajectory_ids": TRAJECTORY_IDS.tolist(),
            "seed": SEED, "parameter_axis": plan["parameter_axis"], "parameters": parameters.tolist(),
        })
        with h5py.File(dataset, "x") as handle:
            single = lock["lock_snapshot"].get("profile") == "single_native76_n300"
            handle.attrs.update(complete=False, status="incomplete", trajectory_id_scheme="canonical",
                                cohort="w1_v2_independent_confirmation", seed=SEED, completed_trajectories=0,
                                protocol_locked=True, development_locked=not single, reference_save_interval=.02,
                                evaluation_dt=plan["evaluation_dt"], evaluation_points=plan["evaluation_points"],
                                evaluation_future_steps=plan["evaluation_future_steps"],
                                evaluation_query_times_json=json.dumps(plan["evaluation_query_times"]))
            if single:
                handle.attrs.update(configurations_locked=True, development_used=False,
                                    protocol_profile="single_native76_n300")
            node = handle.create_group("test")
            node.create_dataset("trajectory_index", data=TRAJECTORY_IDS)
            node.create_dataset("time", data=LOCAL_TIMES)
            node.create_dataset("absolute_time", data=ABSOLUTE_TIMES)
            node.create_dataset("initial_condition_parameters", data=parameters)
            node.create_dataset("vorticity", shape=(180, 151, 64, 64), dtype="f4",
                                chunks=(1, 1, 64, 64), compression="lzf", fillvalue=np.nan)
    else:
        original_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        original_lock = json.loads((output / "lock.json").read_text(encoding="utf-8"))
        # An existing dense151 attempt predates explicit evaluation labels. Its
        # original numerical contract remains the same default151-point task.
        if "evaluation_query_times" not in original_plan:
            default_plan = confirmation_plan()
            for key in default_plan:
                if key.startswith("evaluation_"):
                    original_plan[key] = default_plan[key]
        current = lock["lock_snapshot"]
        original = original_lock["lock_snapshot"]
        configuration_key = "training_configurations" if current.get("profile") == "single_native76_n300" else "selected_configurations"
        if (original_plan != plan or original.get("profile") != current.get("profile")
                or original.get(configuration_key) != current.get(configuration_key)):
            raise ValueError("resume must preserve the declared cohort and locked configurations")
    sources = {str(path.relative_to(ROOT)): _sha256(path) for path in (
        Path(__file__), ROOT / "scripts" / "generate_data.py", ROOT / "src" / "even_full_spectrum_ns.py")}
    segments = list(output.glob("segment_*.json"))
    segment_path = output / f"segment_{len(segments):04d}.json"
    started = time.perf_counter()
    with h5py.File(dataset, "r+") as handle:
        node = handle["test"]
        if (not np.array_equal(node["trajectory_index"][:], TRAJECTORY_IDS)
                or not np.array_equal(node["initial_condition_parameters"][:], parameters)
                or not np.array_equal(node["absolute_time"][:], ABSOLUTE_TIMES)):
            raise ValueError("own-attempt resume identities, initial conditions or save nodes differ")
        first = int(handle.attrs["completed_trajectories"])
        if not 0 <= first <= 180 or (first < 180 and bool(handle.attrs["complete"])):
            raise ValueError("generation checkpoint progress is inconsistent")
        if first == 180:
            handle.attrs.update(complete=True, status="complete")
            handle.flush()
            if not (output / "complete.json").exists():
                _write_json(output / "complete.json", {
                    "status": "complete", "data_file": str(dataset), "population": 180,
                    "recovered_final_marker": True, "solver_metadata": json.loads(handle.attrs["solver_metadata"]),
                    "seed": SEED, "trajectory_ids": TRAJECTORY_IDS.tolist(), "all_prescribed_draws_retained": True,
                })
            return {"status": "already_complete", "data_file": str(dataset), "population": 180}
        _write_json(segment_path, {"started_utc": datetime.now(timezone.utc).isoformat(),
                                  "first_trajectory": first, "device": str(device), "batch_size": batch_size,
                                  "sources": sources, "current_protocol_sha256": lock["protocol_sha256"],
                                  "current_lock_sha256": lock["lock_sha256"],
                                  "numpy_version": np.__version__, "h5py_version": h5py.__version__})
        for last in range(first, 180, batch_size):
            stop = min(last + batch_size, 180)
            fields, solver_metadata = integrate_batch(parameters[last:stop], device=device)
            node["vorticity"][last:stop] = fields
            # Flush the whole batch before committing its progress. On resume an
            # uncommitted partial batch is regenerated from its saved parameters.
            handle.flush()
            handle.attrs["completed_trajectories"] = stop
            handle.attrs["solver_metadata"] = json.dumps(solver_metadata, allow_nan=False)
            if stop == 180:
                handle.attrs.update(complete=True, status="complete")
            handle.flush()
            print(json.dumps({"status": "generation_committed", "completed": stop,
                              "population": 180, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    report = {"status": "complete", "data_file": str(dataset), "population": 180,
              "elapsed_segment_seconds": time.perf_counter() - started, "solver_metadata": solver_metadata,
              "seed": SEED, "trajectory_ids": TRAJECTORY_IDS.tolist(), "all_prescribed_draws_retained": True}
    _write_json(output / "complete.json", report)
    return report


def main(argv=None):
    """Print the fixed plan unless full generation is explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_generation:
        lock = locked_configuration(args.lock) if args.lock else None
        plan = confirmation_plan(lock["evaluation_query_times"] if lock else None,
                                 profile=lock["lock_snapshot"].get("profile") if lock else None)
        if lock:
            plan["lock"] = lock
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return plan
    if args.lock is None or args.output is None:
        parser.error("--run-generation requires --lock and --output")
    return generate_confirmation(args.lock, args.output, device=args.device,
                                 batch_size=args.batch_size, resume=args.resume)


if __name__ == "__main__":
    main()
