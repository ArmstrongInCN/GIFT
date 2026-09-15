"""Fresh NS integration from declared initial conditions; own-attempt resume only.

No reference solution or trained weights are opened by this program. See
docs/DATA_GENERATION.md for provenance and the limits of numerical equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import uuid

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DT = 0.005
FIRST_FIVE = [
    [[4.2, .36, 1.35, 3.55], [-4.2, .36, 1.35, 2.55], [3.6, .52, 4.9, 2.35], [-3.6, .52, 4.9, 3.75]],
    [[4., .40, .95, 3.35], [-4., .40, .95, 2.30], [3.4, .55, 5.15, 2.15], [-3.4, .55, 5.15, 3.70]],
    [[4.4, .34, 1.60, 3.80], [-4.4, .34, 1.60, 2.75], [3.2, .58, 4.65, 2.10], [-3.2, .58, 4.65, 3.60]],
    [[3.8, .44, 1.20, 3.25], [-3.8, .44, 1.20, 2.10], [3.7, .48, 5., 2.55], [-3.7, .48, 5., 3.85]],
    [[4.1, .37, 1.50, 3.50], [-4.1, .37, 1.50, 2.40], [3.5, .53, 4.80, 2.25], [-3.5, .53, 4.80, 3.95]],
]
EXTRA_PARAMETERS_HASH = "77a90ccd984690b4c5c3ca45c281af69e59199f138fd6aa05e3902ccac817126"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def draw_parameters(count, seed):
    """PCG64 draw order recovered and checked against stored parameter arrays."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(count):
        lg, rg = rng.uniform(3.6, 4.6), rng.uniform(3., 4.)
        ls, rs = rng.uniform(.33, .47), rng.uniform(.45, .61)
        lx, rx = rng.uniform(.8, 1.8), rng.uniform(4.3, 5.5)
        ly, ry = rng.uniform(2.8, 3.4), rng.uniform(2.8, 3.3)
        ld, rd = rng.uniform(.95, 1.25), rng.uniform(1.15, 1.55)
        values.append([[1.5*lg, ls, lx, ly+.5*ld], [-1.5*lg, ls, lx, ly-.5*ld],
                       [1.5*rg, rs, rx, ry-.5*rd], [-1.5*rg, rs, rx, ry+.5*rd]])
    return np.asarray(values, dtype=np.float64)


def seed_parameters():
    first = np.asarray(FIRST_FIVE, dtype=np.float64)
    first[:, :, 0] *= 1.5
    pool = draw_parameters(1150, 2026080801)
    return {"training": (np.arange(50), np.concatenate([first, draw_parameters(45, 2026080604)])),
            "validation": (np.arange(50, 70), pool[:20]),
            "test": (np.arange(1000, 1200), pool[950:1150])}


def extra_parameters(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters = np.asarray(value["parameters"], dtype=np.float64)
    ids = np.asarray(value["trajectory_ids"], dtype=np.int64)
    if value.get("trajectory_id_scheme") == "canonical":
        if not np.array_equal(ids, np.arange(50, 1000)):
            raise ValueError("canonical Extra-950 identities must be prediction training IDs 50-999")
        from gift.data_splits import storage_ids
        ids = np.asarray(storage_ids(ids), dtype=np.int64)
    if (value.get("schema") != "gift.initial-conditions.v1" or
            parameters.shape != (950, 4, 4) or not np.array_equal(ids, np.arange(1200, 2150)) or
            not np.isfinite(parameters).all() or array_hash(parameters) != EXTRA_PARAMETERS_HASH):
        raise ValueError("Extra-950 initial-condition schema/IDs/content do not match the declared protocol")
    return ids, parameters


def parse_ids(value):
    if value is None:
        return None
    result = []
    for token in value.split(","):
        parts = token.strip().split(":")
        if len(parts) == 1:
            result.append(int(parts[0]))
        elif len(parts) == 2:
            lo, hi = map(int, parts)
            if hi < lo:
                raise ValueError("ID ranges are inclusive and must ascend")
            result.extend(range(lo, hi + 1))
        else:
            raise ValueError("Use IDs such as 0,2,50:52 (inclusive ranges)")
    if not result or len(result) != len(set(result)):
        raise ValueError("Subset IDs must be nonempty and unique")
    return set(result)


def make_plan(args):
    pools = seed_parameters()
    input_hash = None
    if args.dataset in ("fno-training", "fno-training-coarse"):
        if not args.initial_conditions:
            raise ValueError("FNO training requires --initial-conditions baseline_extra950.json")
        ids, params = extra_parameters(args.initial_conditions)
        old_ids, old_params = pools["training"]
        pools["training"] = (np.concatenate([old_ids, ids]), np.concatenate([old_params, params]))
        input_hash = sha256(args.initial_conditions)
    if args.dataset == "standard":
        specs = [(x, x, 64, 50 if x == "training" else 20 if x == "validation" else 200,
                  list(range(0, 2001, 4)) if x != "test" else list(range(1000, 2001, 100)))
                 for x in ("training", "validation", "test")]
    elif args.dataset in ("fno-training", "fno-training-coarse"):
        stride = 4 if args.dataset == "fno-training" else 20
        specs = [(x, x, 64, 50 if x == "training" else 20, list(range(0, 2001, stride)))
                 for x in ("training", "validation")]
    elif args.dataset == "fno-test":
        specs = [(f"N{n}/test", "test", n, 200 if n == 64 else 8,
                  list(range(820, (1600 if n == 64 else 1200) + 1, 4))) for n in (64, 96, 128)]
    elif args.dataset == "cross-resolution":
        specs = [(f"N{n}/test", "test", n, 8, list(range(820, 1201, 20))) for n in (96, 128)]
    else:
        specs = [("test", "test", 64, 200, list(range(820, 1201, 20)))]
    chosen = parse_ids(args.subset)
    found, groups = set(), []
    for name, split, n, batch, steps in specs:
        if args.split != "all" and args.split != split:
            continue
        ids, params = pools[split]
        keep = np.ones(len(ids), dtype=bool) if chosen is None else np.isin(ids, list(chosen))
        ids, params = ids[keep], params[keep]
        if not len(ids):
            continue
        found.update(map(int, ids))
        if args.pilot_steps is not None:
            steps = sorted(set([0, args.pilot_steps] + list(range(0, args.pilot_steps + 1, 4))))
        groups.append({"name": name, "split": split, "grid": n,
                       "batch_size": args.batch_size or batch, "steps": steps,
                       "ids": ids.tolist(), "parameters": params.tolist(),
                       "parameters_sha256": array_hash(params)})
    if not groups or (chosen is not None and found != chosen):
        raise ValueError("Empty selection or IDs outside the selected dataset/split")
    return {"schema": "gift.data-generation.v1", "dataset": args.dataset,
            "pilot": args.pilot_steps is not None, "subset": args.subset,
            "dt": DT, "device": args.device, "groups": groups,
            "initial_conditions_file_sha256": input_hash,
            "extra950_origin": "specified_initial_conditions_seed_unknown" if input_hash else None,
            "physical_protocol": {"equation": "omega_t+u.grad(omega)=nu*laplacian(omega)-4*cos(4*y)",
                                  "domain": [0., 2.*math.pi], "periodic": True, "viscosity": .01,
                                  "forcing_amplitude": 1., "forcing_wavenumber": 4,
                                  "internal_dt": DT, "real_dtype": "float32", "complex_dtype": "complex64",
                                  "solver": "full-spectrum Fourier pseudospectral ETDRK4",
                                  "state_mask_applied": False, "padding": {"64":99,"96":147,"128":195}},
            "sources": {"scripts/generate_data.py": sha256(__file__),
                        "src/even_full_spectrum_ns.py": sha256(ROOT / "src/even_full_spectrum_ns.py")}}


def stored_times(plan, group):
    steps = np.asarray(group["steps"], dtype=np.int64)
    if not plan["pilot"] and plan["dataset"] in ("short-test", "cross-resolution"):
        if np.any(steps % 20):
            raise ValueError("The formal 0.1 report schedule requires integer multiples of 20 solver steps")
        return np.round((steps // 20).astype(np.float64) / 10., 12)
    return steps.astype(np.float64) * DT


def write_generation_metadata(handle, plan, *, complete):
    """Reader-compatible metadata, with regenerated provenance kept explicit."""
    if plan["dataset"] != "fno-test":
        return
    status = "incomplete"
    if complete:
        status = "complete" if not plan["pilot"] and plan["subset"] is None else "complete_nonformal_selection"
    handle.attrs["metadata_json"] = canonical({
        "schema_version": "gift.fno.test-data.v1", "status": status,
        "stored_dt": .02, "interpolation_used": False,
        "generation_profile": "regenerated", "source_hashes": plan["sources"],
        "physical_protocol": plan["physical_protocol"], "pilot": plan["pilot"],
        "subset": plan["subset"], "copied_reference_truth": False,
    })


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json_new(temporary, value)
    os.replace(temporary, path)


def safe_output(path):
    output = Path(path).resolve()
    protected = [ROOT]
    if os.environ.get("GIFT_DATA_ROOT"):
        protected.append(Path(os.environ["GIFT_DATA_ROOT"]).resolve())
    for base in protected:
        if output == base or base in output.parents or output in base.parents:
            raise ValueError("Output must be a separate directory outside code and GIFT_DATA_ROOT")
    return output


def runtime_record(torch, device):
    record = {"python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "h5py": h5py.__version__, "torch": torch.__version__,
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "cuda_runtime": torch.version.cuda, "device": str(device),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if device.type == "cuda":
        record.update(device_name=torch.cuda.get_device_name(device),
                      device_capability=list(torch.cuda.get_device_capability(device)))
    return record


def execute(args, plan):
    import torch
    from src.even_full_spectrum_ns import (
        EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver, build_initial_hat,
        hermitian_imaginary_defect,
    )
    output = safe_output(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    binding = {"plan": plan, "runtime": runtime_record(torch, device)}
    binding_hash = hashlib.sha256(canonical(binding).encode()).hexdigest()
    jobs = [(i, start, min(start + group["batch_size"], len(group["ids"])))
            for i, group in enumerate(plan["groups"])
            for start in range(0, len(group["ids"]), group["batch_size"])]
    if args.resume:
        if (output / "COMPLETE.json").exists():
            raise ValueError("This attempt is already complete; it is not a new generation")
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if run["binding_sha256"] != binding_hash or run["binding"] != binding:
            raise ValueError("Resume source/configuration/parameters/runtime mismatch")
        latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
        checkpoint = output / latest["file"]
        if checkpoint.parent != output or not checkpoint.name.startswith("checkpoint_"):
            raise ValueError("Checkpoint must belong to this attempt directory")
        if latest["attempt_id"] != run["attempt_id"] or sha256(checkpoint) != latest["sha256"]:
            raise ValueError("Checkpoint identity/hash mismatch")
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["attempt_id"]) != run["attempt_id"] or str(saved["binding_sha256"]) != binding_hash:
                raise ValueError("Foreign checkpoint rejected")
            job_index, step = int(saved["job_index"]), int(saved["step"])
            saved_hat = saved["state_hat"].copy()
            if saved_hat.dtype != np.complex64 or not np.isfinite(saved_hat).all():
                raise ValueError("Invalid saved spectral state")
        mode = "r+"
    else:
        output.mkdir(parents=True, exist_ok=False)
        run = {"attempt_id": str(uuid.uuid4()), "binding_sha256": binding_hash, "binding": binding}
        write_json_new(output / "run.json", run)
        job_index, step, saved_hat, mode = 0, 0, None, "x"
    h5_path = output / "data.h5"
    with h5py.File(h5_path, mode) as handle, torch.inference_mode():
        if mode == "x":
            handle.attrs.update(attempt_id=run["attempt_id"], status="INCOMPLETE",
                                binding_sha256=binding_hash, generated_from="initial_conditions_not_saved_truth",
                                pilot=plan["pilot"], internal_dt=DT)
            write_generation_metadata(handle, plan, complete=False)
            for group in plan["groups"]:
                g = handle.create_group(group["name"])
                g.create_dataset("trajectory_index", data=np.asarray(group["ids"], dtype=np.int64))
                g.create_dataset("time", data=stored_times(plan, group))
                parameter_name = "initial_parameters" if group["name"].startswith("N") else "initial_condition_parameters"
                g.create_dataset(parameter_name, data=np.asarray(group["parameters"], dtype=np.float64))
                n = group["grid"]
                g.create_dataset("vorticity", shape=(len(group["ids"]), len(group["steps"]), n, n),
                                 dtype="f4", chunks=(1, 1, n, n), compression="lzf", shuffle=True,
                                 fillvalue=np.nan)
        elif handle.attrs.get("attempt_id") != run["attempt_id"] or handle.attrs.get("binding_sha256") != binding_hash:
            raise ValueError("H5 does not belong to this attempt")

        def save(next_job, current_step, state):
            handle.flush()
            name = "checkpoint_" + uuid.uuid4().hex + ".npz"
            with (output / name).open("xb") as stream:
                np.savez(stream, attempt_id=run["attempt_id"], binding_sha256=binding_hash,
                         job_index=np.int64(next_job), step=np.int64(current_step),
                         state_hat=np.empty((0,), np.complex64) if state is None else state.detach().cpu().numpy())
            atomic_json(output / "latest.json", {"attempt_id": run["attempt_id"], "file": name,
                                                  "sha256": sha256(output / name)})

        advanced = 0
        if mode == "x":
            save(0, 0, None)
        for j in range(job_index, len(jobs)):
            i, start, end = jobs[j]
            group = plan["groups"][i]
            n = group["grid"]
            solver = EvenFullSpectrumNSSolver(EvenFullSpectrumConfiguration(grid=n, padding_grid={64: 99, 96: 147, 128: 195}[n]), device=device)
            handle[group["name"]].attrs["solver_metadata"] = canonical(solver.metadata)
            if j == job_index and saved_hat is not None and saved_hat.size:
                if saved_hat.shape != (end-start, n, n):
                    raise ValueError("Checkpoint state/batch shape mismatch")
                state = torch.from_numpy(saved_hat).to(device)
                current = step
            else:
                state = build_initial_hat(group["parameters"][start:end], grid=n, device=device)
                current = 0
            index = {value: k for k, value in enumerate(group["steps"])}
            while True:
                if not torch.isfinite(state).all():
                    raise ValueError(f"Nonfinite state at job {j}, step {current}")
                if current in index:
                    defect = hermitian_imaginary_defect(state)
                    if defect > 1e-5 or float(state[..., 0, 0].abs().max()) > 1e-6:
                        raise ValueError("Native-spectrum zero-mode/Hermitian integrity check failed")
                    handle[group["name"] + "/vorticity"][start:end, index[current]] = torch.fft.ifft2(state).real.cpu().numpy()
                if current == group["steps"][-1]:
                    save(j + 1, 0, None)
                    break
                if args.stop_after_steps is not None and advanced >= args.stop_after_steps:
                    save(j, current, state)
                    print(json.dumps({"status": "PAUSED", "attempt_id": run["attempt_id"], "job": j, "step": current}), flush=True)
                    return 0
                state = solver.advance(state)
                current += 1
                advanced += 1
                if current % args.checkpoint_every == 0:
                    save(j, current, state)
            print(json.dumps({"job_completed": j + 1, "jobs": len(jobs), "group": group["name"], "ids": [group["ids"][start], group["ids"][end-1]]}), flush=True)
        handle.attrs["status"] = "COMPLETE_PILOT" if plan["pilot"] else "COMPLETE_GENERATION"
        write_generation_metadata(handle, plan, complete=True)
        handle.flush()
    write_json_new(output / "COMPLETE.json", {"attempt_id": run["attempt_id"],
                   "status": "COMPLETE_PILOT" if plan["pilot"] else "COMPLETE_GENERATION",
                   "binding_sha256": binding_hash, "data_sha256": sha256(h5_path),
                   "reference_comparison_performed": False, "full_budget_reproduction_claim": False})
    print(json.dumps({"status": "COMPLETE_PILOT" if plan["pilot"] else "COMPLETE_GENERATION", "output": str(output)}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["standard", "fno-training", "fno-training-coarse", "fno-test", "cross-resolution", "short-test"])
    parser.add_argument("--split", choices=["all", "training", "validation", "test"], default="all")
    parser.add_argument("--subset", help="Global IDs, e.g. 0,2,50:52; inclusive ranges")
    parser.add_argument("--initial-conditions", type=Path, help="Extra-950 parameter JSON (not vorticity)")
    parser.add_argument("--output", type=Path, required=True, help="New external attempt directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pilot-steps", type=int, help="Explicit non-formal short integration from t=0")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--stop-after-steps", type=int, help="Graceful pause after this invocation's number of advances")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Without this flag, print a plan and write nothing")
    args = parser.parse_args(argv)
    if args.checkpoint_every < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("Checkpoint interval and batch size must be positive")
    if args.pilot_steps is not None and not 0 <= args.pilot_steps <= 2000:
        parser.error("Pilot steps must be between 0 and 2000")
    if args.stop_after_steps is not None and args.stop_after_steps < 0:
        parser.error("Stop-after steps cannot be negative")
    plan = make_plan(args)
    safe_output(args.output)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
