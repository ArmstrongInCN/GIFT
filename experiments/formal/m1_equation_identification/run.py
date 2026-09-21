"""One M1 method/condition per job; no fabricated complete five-method table.

PDE-FIND uses the user's pinned external source. GIFT reads one bound trained
generator. PINN methods evaluate an explicitly bound completed native training checkpoint.
A completed job can be
read without inference, or verified/reused by repeating its execution command.

GIFT is identified from the same single training trajectory as the other four
configurations: its generator is trained on local trajectory 0 (see
artifacts/m1_parameter_identification/manifest.json) and the coefficient readout
projects the frozen generator onto that same trajectory.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
from datetime import datetime, timezone
import json
import io
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

METHODS = {"GIFT": "gift", "PDE-FIND": "pde_find", "PDE-FIND-KC": "pde_find_kc",
           "PINN-SR": "pinn_sr", "PINN-SR-KC": "pinn_sr_kc"}
DATASETS = {"noise_000": "standard_ns_n64_full_spectrum.h5",
            "noise_001": "m1_parameter_identification/noise_001.h5",
            "noise_010": "m1_parameter_identification/noise_010.h5"}
# The five configurations share one training trajectory: PDE-FIND and PINN-SR
# read local trajectory 0, and GIFT is both trained on and read out from it.
GIFT_TRAJECTORY_COUNT = 1


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest().upper()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _input_root():
    value = os.environ.get("GIFT_DATA_ROOT")
    if not value:
        raise ValueError("set GIFT_DATA_ROOT to the external input data package")
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir() or root == ROOT or ROOT in root.parents:
        raise ValueError("input data package must be an external directory")
    return root


def _job_path(args):
    output = Path(args.output).expanduser().resolve()
    published = ROOT / "artifacts"
    if output == ROOT or output == published or published in output.parents:
        raise ValueError("job output must not replace the code root or published artifacts")
    if os.environ.get("GIFT_DATA_ROOT"):
        inputs = Path(os.environ["GIFT_DATA_ROOT"]).expanduser().resolve()
        if output == inputs or inputs in output.parents:
            raise ValueError("job outputs must stay outside the read-only input package")
    return output / args.condition / METHODS[args.method]


def _dataset_binding(condition):
    root = _input_root()
    relative = DATASETS[condition]
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("dataset path escapes input package")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    matches = [item for item in manifest["files"] if item["path"] == relative]
    if len(matches) != 1:
        raise ValueError("dataset is not uniquely bound in the external manifest")
    expected = matches[0]
    actual = digest(path)
    if actual != expected["sha256"].upper() or path.stat().st_size != expected["bytes"]:
        raise ValueError("input dataset differs from the package manifest")
    return path, {"relative_path": relative, "bytes": path.stat().st_size, "sha256": actual}


def _checkpoint(args):
    if args.checkpoint is not None:
        if not args.checkpoint_sha256:
            raise ValueError("an explicit checkpoint requires --checkpoint-sha256")
        path, expected = args.checkpoint.resolve(strict=True), args.checkpoint_sha256.upper()
    else:
        manifest = json.loads((ROOT / "artifacts/m1_parameter_identification/manifest.json").read_text())
        matches = [item for item in manifest["gift_checkpoints"] if item["condition"] == args.condition]
        if len(matches) != 1:
            raise ValueError("condition lacks one bound GIFT checkpoint")
        item = matches[0]
        path, expected = (ROOT / item["path"]).resolve(strict=True), item["sha256"].upper()
        if ROOT not in path.parents:
            raise ValueError("published checkpoint path escapes the repository")
    if digest(path) != expected:
        raise ValueError("GIFT checkpoint SHA256 differs")
    return path, {"sha256": expected, "bytes": path.stat().st_size}


def calculate(method, condition, dataset, checkpoint, device):
    if method == "GIFT":
        return {"parameters": gift_readout(ROOT, dataset, checkpoint, device),
                "training_executed": False, "forward_executed": True,
                "native_coefficients": [], "readout": "original P21 component projection"}
    if method.startswith("PINN"):
        from adapters.pinn_reference import read_checkpoint
        reference = read_checkpoint(checkpoint, condition, "known" if method == "PINN-SR-KC" else "open")
        if (reference["method"] != method or reference["condition"] != condition
                or reference["source_data_protocol"]["data_sha256"].upper() != digest(dataset)):
            raise ValueError("PINN checkpoint method/condition/data binding differs")
        return {"parameters": reference["parameters"], "scope": "trained_coefficients_readout",
                "readout_scope": "trained_coefficients_readout", "training": False, "forward": False,
                "training_executed": False, "forward_executed": False, "sparse_regression_executed": False,
                "fresh_training": False, "full_budget_retraining_verified": False,
                "reference_binding": reference,
                "native_coefficients": [{"term": name, "real": value, "imaginary": 0.0}
                    for name, value in zip(reference["coefficient_names"], reference["coefficients"])]}
    from adapters import pde_find as pde
    upstream = pde.load_upstream()
    raw, times = pde.read_observations(dataset)
    rows, protocol = pde.prepare_rows(raw, times, condition)
    known = method == "PDE-FIND-KC"
    theta, names = pde.library(upstream, rows, known=known)
    upstream_stdout = io.StringIO()
    with contextlib.redirect_stdout(upstream_stdout):
        coefficients = pde.regress(upstream, theta, rows["wt"])
    return {"parameters": pde.parameter_readout(names, coefficients, known=known),
            "training_executed": False, "forward_executed": False,
            "sparse_regression_executed": True, "protocol": protocol,
            "upstream_stdout": upstream_stdout.getvalue(),
            "upstream": upstream.source, "ridge": pde.RIDGE,
            "library_dtype": str(theta.dtype), "library_shape": list(theta.shape),
            "library_sha256": pde.array_digest(theta),
            "training_indices_sha256": pde.array_digest(upstream.np.random.last_indices),
            "native_coefficients": [{"term": name, "real": float(complex(value).real),
                                     "imaginary": float(complex(value).imag)}
                                    for name, value in zip(names, coefficients)]}


def read_completed(job, method, condition, identity=None):
    completion = json.loads((job / "COMPLETE.json").read_text(encoding="utf-8"))
    if completion.get("schema") != "gift.independent-m1-job-commit.v1":
        raise ValueError("not an independent M1 job commit")
    files = completion.get("files", {})
    if set(files) != {"result.json", "summary.csv", "native_coefficients.csv"}:
        raise ValueError("job commit file set differs")
    for name, checksum in files.items():
        if digest(job / name) != checksum:
            raise ValueError("completed job file SHA256 differs")
    record = json.loads((job / "result.json").read_text(encoding="utf-8"))
    if (record.get("status") != "complete" or record.get("method") != method or
            record.get("condition") != condition or record.get("identity") != completion.get("identity")):
        raise ValueError("completed job identity/status differs")
    if identity is not None and record["identity"] != identity:
        raise ValueError("completed job uses different source, data, checkpoint or environment")
    return {"status": "cached_result_read", "method": method, "condition": condition,
            "job": str(job), "parameters": record["parameters"],
            "scientific_acceptance": record["scientific_acceptance"],
            "training_executed": False, "forward_executed": False,
            "sparse_regression_executed": False, "record_sha256": files["result.json"],
            "cached_result_only": True, "trained_archive_read": False,
            "source_readout_scope": record.get("readout_scope")}


def _pinn_checkpoint_binding(args, data_record):
    from adapters import pinn_reference
    mode = "known" if args.method == "PINN-SR-KC" else "open"
    path = (args.checkpoint if args.checkpoint is not None else
            pinn_reference.REFERENCE_ROOT / args.condition / mode / "result.json").resolve(strict=True)
    checksum = digest(path)
    if args.checkpoint is not None and checksum != args.checkpoint_sha256.upper():
        raise ValueError("explicit PINN result checksum differs")
    result = pinn_reference.read_checkpoint(path, args.condition, mode)
    if result["source_data_protocol"]["data_sha256"].upper() != data_record["sha256"]:
        raise ValueError("PINN checkpoint is bound to another dataset")
    return path, {"sha256": checksum, "bytes": path.stat().st_size,
                  "archive_sha256": result["archive_sha256"], "mode": mode,
                  "artifact_role": "completed_native_training",
                  "source_data_protocol": result["source_data_protocol"]}


def execute(args):
    is_pinn = args.method.startswith("PINN")
    job = _job_path(args)
    dataset, data_record = _dataset_binding(args.condition)
    if is_pinn:
        checkpoint, checkpoint_record = _pinn_checkpoint_binding(args, data_record)
    else:
        from adapters import pde_find as pde
        checkpoint, checkpoint_record = _checkpoint(args) if args.method == "GIFT" else (None, None)
    adapter = "adapters/pinn_reference.py" if is_pinn else "adapters/pde_find.py"
    sources = {adapter: digest(ROOT / adapter),
               "experiments/formal/m1_equation_identification/run.py": digest(Path(__file__))}
    upstream = None if args.method == "GIFT" or is_pinn else pde.source_record()
    if args.method == "GIFT":
        for relative in ("src/gift/model.py", "src/gift/identified.py", "src/even_full_spectrum_ns.py"):
            sources[relative] = digest(ROOT / relative)
    import numpy as np
    runtime = {"python": platform.python_version(), "numpy": str(np.__version__),
               "device": args.device if args.method == "GIFT" else "cpu",
               "thread_environment": {name: os.environ.get(name) for name in
                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    if not is_pinn:
        import scipy
        runtime["scipy"] = str(scipy.__version__)
    if args.method == "GIFT":
        import torch
        runtime.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                       matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_tf32=torch.backends.cudnn.allow_tf32,
                       tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"),
                       cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       torch_threads=torch.get_num_threads(),
                       gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)
    identity = {"method": args.method, "condition": args.condition, "data": data_record,
                "checkpoint": checkpoint_record, "sources": sources, "runtime": runtime,
                "upstream_sha256": None if upstream is None else upstream["sha256"]}
    if job.exists():
        return read_completed(job, args.method, args.condition, identity)
    pending = job.with_name(job.name + ".partial")
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(pending / "START.json", {"identity": identity, "pid": os.getpid(),
                                       "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        result = calculate(args.method, args.condition, dataset, checkpoint, args.device)
        parameters = result["parameters"]
        if set(parameters) != {"nu", "beta", "gamma"} or not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError("M1 requires exactly three finite reported parameters")
        if digest(dataset) != data_record["sha256"] or (checkpoint and digest(checkpoint) != checkpoint_record["sha256"]):
            raise ValueError("input changed during M1 job")
        if is_pinn and _pinn_checkpoint_binding(args, data_record)[1] != checkpoint_record:
            raise ValueError("PINN completed checkpoint changed during M1 job")
        if upstream is not None and pde.source_record()["sha256"] != upstream["sha256"]:
            raise ValueError("external source changed during M1 job")
        if any(digest(ROOT / relative) != checksum for relative, checksum in sources.items()):
            raise ValueError("project source changed during M1 job")
        record = {**result, "schema": "gift.independent-m1-job.v1", "status": "complete",
                  "method": args.method, "condition": args.condition, "identity": identity,
                  "scientific_acceptance": "not_evaluated_against_published_reference",
                  "complete_five_method_experiment": False,
                  "elapsed_seconds": time.perf_counter() - started}
        write_json(pending / "result.json", record)
        for name, fields, rows in (
            ("summary.csv", ["condition", "method", "parameter", "estimate"],
             [dict(condition=args.condition, method=args.method, parameter=key, estimate=parameters[key])
              for key in ("nu", "beta", "gamma")]),
            ("native_coefficients.csv", ["term", "real", "imaginary"], result["native_coefficients"]),
        ):
            with (pending / name).open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
        write_json(pending / "COMPLETE.json", {"schema": "gift.independent-m1-job-commit.v1",
                   "identity": identity, "files": {name: digest(pending / name)
                    for name in ("result.json", "summary.csv", "native_coefficients.csv")}})
        pending.rename(job)
        return {"status": "complete", "method": args.method, "condition": args.condition,
                "job": str(job), "parameters": parameters,
                "scientific_acceptance": record["scientific_acceptance"]}
    except BaseException as error:
        if pending.exists() and not (pending / "FAILURE.json").exists():
            write_json(pending / "FAILURE.json", {"status": "failed", "error": str(error),
                                                 "traceback": traceback.format_exc()})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--use-results", action="store_true")
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--condition", choices=tuple(DATASETS), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "m1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args(argv)
    if args.method.startswith("PDE") and (args.checkpoint or args.checkpoint_sha256):
        parser.error("PDE-FIND does not use a trained checkpoint")
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        parser.error("--checkpoint and --checkpoint-sha256 must be supplied together")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.use_results:
        result = read_completed(_job_path(args), args.method, args.condition)
    elif args.execute:
        result = execute(args)
    else:
        is_pinn = args.method.startswith("PINN")
        result = {"status": "dry_run", "method": args.method, "condition": args.condition,
                  "implemented": True, "writes": 0,
                  "job": str(_job_path(args)), "dataset": str(_input_root() / DATASETS[args.condition]),
                  "scope": "trained_coefficients_readout" if is_pinn else "one independent job, not the complete original M1 chain",
                  "training": False, "forward": args.method == "GIFT", "fresh_training": False,
                  "reference_archive_required": is_pinn, "archive_availability_checked": False,
                  "full_budget_retraining_verified": False, "next_action": "--execute"}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def gift_readout(
    project: Path, dataset: Path, checkpoint: Path, device_name: str
) -> dict[str, float]:
    import h5py
    import numpy as np
    import torch

    source_root = project / "src"
    sys.path.insert(0, str(source_root))
    import gift
    from even_full_spectrum_ns import (
        EvenFullSpectrumConfiguration,
        EvenFullSpectrumNSSolver,
    )

    with h5py.File(dataset, "r") as handle:
        ids = np.asarray(handle["training/trajectory_index"][:GIFT_TRAJECTORY_COUNT], dtype=np.int64)
        times = np.asarray(handle["training/time"][:], dtype=np.float64)
        target_times = np.round(5.0 + 0.1 * np.arange(11), 12)
        selected = np.asarray(
            [
                int(np.flatnonzero(np.isclose(times, value, rtol=0.0, atol=2e-12))[0])
                for value in target_times
            ],
            dtype=np.int64,
        )
        if not np.array_equal(ids, np.arange(GIFT_TRAJECTORY_COUNT, dtype=np.int64)):
            raise ValueError("GIFT M1 training trajectory IDs differ")
        states = np.asarray(handle["training/vorticity"][:GIFT_TRAJECTORY_COUNT, selected], dtype=np.float32)
    if states.shape != (GIFT_TRAJECTORY_COUNT, 11, 64, 64) or not np.isfinite(states).all():
        raise ValueError("GIFT M1 readout states differ")

    device = torch.device(device_name)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("format_version") != 3
        or payload.get("artifact_role") != "trained_generator"
    ):
        raise ValueError("GIFT M1 checkpoint format differs")
    if str(payload.get("training_data", {}).get("sha256", "")).upper() != digest(
        dataset
    ):
        raise ValueError("GIFT M1 checkpoint/data binding differs")
    configuration = payload["model_configuration"]
    if configuration.get("cutoff") != 21 or configuration.get("rank") != 8:
        raise ValueError("GIFT M1 checkpoint configuration differs")
    model = gift.GIFTGenerator(**configuration)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model = model.to(device).eval()
    solver = EvenFullSpectrumNSSolver(
        EvenFullSpectrumConfiguration(
            grid=64,
            padding_grid=99,
            viscosity=0.01,
            forcing_amplitude=1.0,
            forcing_wavenumber=4,
            dt=0.005,
        ),
        device=device,
    )
    frequency = torch.fft.fftfreq(64, d=1.0 / 64.0, device=device)
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    k2 = kx.square() + ky.square()
    p21 = (kx.abs() <= 21) & (ky.abs() <= 21)
    nonzero = p21 & (k2 > 0)
    forcing_numerator = 0.0
    forcing_denominator = 0.0
    quadratic_numerator = 0.0
    quadratic_denominator = 0.0
    linear_gram = np.zeros((2, 2), dtype=np.float64)
    linear_rhs = np.zeros(2, dtype=np.float64)
    flat = states.reshape(-1, 64, 64)
    with torch.inference_mode():
        for start in range(0, len(flat), 16):
            state = torch.from_numpy(flat[start : start + 16]).to(device)
            state_hat = torch.fft.fft2(state)
            learned = model.components_hat(state)
            constant = torch.fft.ifft2(learned["constant"] * p21).real.double()
            unit_forcing = torch.fft.ifft2(
                solver.forcing_hat.expand_as(state_hat) * p21
            ).real.double()
            forcing_numerator += float((unit_forcing * constant).sum())
            forcing_denominator += float(unit_forcing.square().sum())
            quadratic = torch.fft.ifft2(learned["quadratic"] * p21).real.double()
            unit_transport = torch.fft.ifft2(
                -solver.transport_hat(state_hat) * p21
            ).real.double()
            quadratic_numerator += float((unit_transport * quadratic).sum())
            quadratic_denominator += float(unit_transport.square().sum())
            state_coordinate = torch.fft.ifft2(state_hat * nonzero).real.double()
            laplacian = torch.fft.ifft2(-k2 * state_hat * nonzero).real.double()
            linear = torch.fft.ifft2(learned["linear"] * nonzero).real.double()
            design = np.stack(
                (
                    state_coordinate.reshape(-1).cpu().numpy(),
                    laplacian.reshape(-1).cpu().numpy(),
                ),
                axis=1,
            )
            target = linear.reshape(-1).cpu().numpy()
            linear_gram += design.T @ design
            linear_rhs += design.T @ target
    coefficients = np.linalg.solve(linear_gram, linear_rhs)
    return {
        "nu": float(coefficients[1]),
        "beta": float(quadratic_numerator / quadratic_denominator),
        "gamma": float(forcing_numerator / forcing_denominator),
    }


if __name__ == "__main__":
    main()
