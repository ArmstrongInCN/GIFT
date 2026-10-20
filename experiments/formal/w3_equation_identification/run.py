"""One W3 method/condition per job; no fabricated complete five-method table.

PDE-FIND calls the selected external candidate-library implementation. GIFT
reads a trained generator; PINN methods read coefficients from an explicitly
selected completed native result. A completed job can be read without inference
or verified/reused by repeating its execution command.

GIFT is identified from the same single training trajectory as the other four
configurations: its generator is trained on local trajectory 0 (see
artifacts/w3_parameter_identification/manifest.json) and the coefficient readout
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
from adapters import source_file

METHODS = {"GIFT": "gift", "PDE-FIND": "pde_find", "PDE-FIND-KC": "pde_find_kc",
           "PINN-SR": "pinn_sr", "PINN-SR-KC": "pinn_sr_kc"}
DATASETS = {"noise_000": "standard_ns_n64_full_spectrum.h5",
            "noise_001": "w3_parameter_identification/noise_001.h5",
            "noise_005": "w3_parameter_identification/noise_005.h5"}
# The five configurations share one training trajectory: PDE-FIND and PINN-SR
# read local trajectory 0, and GIFT is both trained on and read out from it.
GIFT_TRAJECTORY_COUNT = 1


def digest(path):
    """Stream an input, model or completed-job digest for the selected W3 measurement."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest().upper()


def write_json(path, value):
    """Create and fsync a finite JSON job record without overwriting existing evidence."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _input_root():
    """Resolve the configured observation package used by the selected noise condition."""
    value = os.environ.get("GIFT_DATA_ROOT")
    if not value:
        raise ValueError("set GIFT_DATA_ROOT to the external input data package")
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("input data package must be a directory")
    return root


def _job_path(args):
    """Choose the method/condition job directory while protecting code, artifacts and observations."""
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
    """Fingerprint the selected input file; no distribution manifest is required."""
    root = _input_root()
    relative = DATASETS[condition]
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("dataset path escapes input package")
    actual = digest(path)
    return path, {"relative_path": relative, "bytes": path.stat().st_size, "sha256": actual,
                  "integrity_scope": "selected_input_file; package_audit_not_performed"}


def _checkpoint(args):
    """Resolve an explicit GIFT export or the recorded default for the selected noise condition.

    An explicit checksum is optional; recorded published defaults retain their
    archival checksum. The actual content digest is recorded for completed-job reuse.
    """
    if args.checkpoint is not None:
        path = args.checkpoint.resolve(strict=True)
        expected = args.checkpoint_sha256.upper() if args.checkpoint_sha256 else None
    else:
        manifest = json.loads((ROOT / "artifacts/w3_parameter_identification/manifest.json").read_text())
        matches = [item for item in manifest["gift_checkpoints"] if item["condition"] == args.condition]
        if len(matches) != 1:
            raise ValueError("condition lacks one bound GIFT checkpoint")
        item = matches[0]
        path, expected = (ROOT / item["path"]).resolve(strict=True), item["sha256"].upper()
        if ROOT not in path.parents:
            raise ValueError("published checkpoint path escapes the repository")
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError("GIFT checkpoint SHA256 differs")
    return path, {"sha256": actual, "bytes": path.stat().st_size}


def calculate(method, condition, dataset, checkpoint, device, *, data_record=None):
    """Measure parameters using the selected method's actual readout or regression route.

    GIFT projects frozen generator components; PINN reads completed learned
    coefficients; PDE-FIND regresses its upstream candidate library on observations.
    This entry does not retrain GIFT/PINN or manufacture missing method results.
    """
    if method == "GIFT":
        return {"parameters": gift_readout(ROOT, dataset, checkpoint, device,
                                          dataset_sha256=None if data_record is None else data_record["sha256"]),
                "training_executed": False, "forward_executed": True,
                "native_coefficients": [], "readout": "original P21 component projection"}
    if method.startswith("PINN"):
        from adapters.pinn_reference import read_checkpoint
        reference = read_checkpoint(checkpoint, condition, "known" if method == "PINN-SR-KC" else "open")
        data_hash = reference["source_data_protocol"].get("data_sha256")
        actual_hash = None if data_hash is None else (digest(dataset) if data_record is None else data_record["sha256"])
        if (reference["method"] != method or reference["condition"] != condition
                or (data_hash is not None and data_hash.upper() != actual_hash)):
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
    """Verify and read one committed W3 job without rerunning its numerical calculation.

    When current inputs are supplied, method, condition, observation content and
    checkpoint content must match. Source comments and runtime records are provenance.
    """
    completion = json.loads((job / "COMPLETE.json").read_text(encoding="utf-8"))
    if completion.get("schema") != "gift.independent-w3-job-commit.v1":
        raise ValueError("not an independent W3 job commit")
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
    if identity is not None:
        def inputs(value):
            """Extract the scientific inputs that must agree before reusing a completed job."""
            return (value["method"], value["condition"], value["data"]["sha256"],
                    None if value["checkpoint"] is None else value["checkpoint"]["sha256"])
        if inputs(record["identity"]) != inputs(identity):
            raise ValueError("completed job uses different method, data or checkpoint")
    return {"status": "cached_result_read", "method": method, "condition": condition,
            "job": str(job), "parameters": record["parameters"],
            "scientific_acceptance": record["scientific_acceptance"],
            "training_executed": False, "forward_executed": False,
            "sparse_regression_executed": False, "record_sha256": files["result.json"],
            "cached_result_only": True, "trained_archive_read": False,
            "source_readout_scope": record.get("readout_scope")}


def _pinn_checkpoint_binding(args, data_record):
    """Read the selected completed PINN coefficient result and optional observation/checksum binding."""
    from adapters import pinn_reference
    mode = "known" if args.method == "PINN-SR-KC" else "open"
    path = (args.checkpoint if args.checkpoint is not None else
            pinn_reference.REFERENCE_ROOT / args.condition / mode / "result.json").resolve(strict=True)
    checksum = digest(path)
    if args.checkpoint_sha256 and checksum != args.checkpoint_sha256.upper():
        raise ValueError("explicit PINN result checksum differs")
    result = pinn_reference.read_checkpoint(path, args.condition, mode)
    data_hash = result["source_data_protocol"].get("data_sha256")
    if data_hash is not None and data_hash.upper() != data_record["sha256"]:
        raise ValueError("PINN checkpoint is bound to another dataset")
    return path, {"sha256": checksum, "bytes": path.stat().st_size,
                  "archive_sha256": result["archive_sha256"], "mode": mode,
                  "artifact_role": "completed_native_training",
                  "source_data_protocol": result["source_data_protocol"]}


def execute(args):
    """Commit one independent W3 measurement through a preserved partial-job directory.

    An existing completed job is reused only for the same scientific inputs. A
    failed calculation writes FAILURE.json and leaves its partial evidence visible;
    a completion rename occurs after finite parameters and all tables are durable.
    """
    is_pinn = args.method.startswith("PINN")
    job = _job_path(args)
    dataset, data_record = _dataset_binding(args.condition)
    if is_pinn:
        checkpoint, checkpoint_record = _pinn_checkpoint_binding(args, data_record)
    else:
        from adapters import pde_find as pde
        checkpoint, checkpoint_record = _checkpoint(args) if args.method == "GIFT" else (None, None)
    adapter = "adapters/pinn_reference.py" if is_pinn else "adapters/pde_find.py"
    sources = {adapter: digest(source_file(ROOT, adapter)),
               "experiments/formal/w3_equation_identification/run.py": digest(Path(__file__))}
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
        result = calculate(args.method, args.condition, dataset, checkpoint, args.device, data_record=data_record)
        parameters = result["parameters"]
        if set(parameters) != {"nu", "beta", "gamma"} or not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError("W3 requires exactly three finite reported parameters")
        if checkpoint and digest(checkpoint) != checkpoint_record["sha256"]:
            raise ValueError("checkpoint changed during W3 job")
        if is_pinn and _pinn_checkpoint_binding(args, data_record)[1] != checkpoint_record:
            raise ValueError("PINN completed checkpoint changed during W3 job")
        record = {**result, "schema": "gift.independent-w3-job.v1", "status": "complete",
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
        write_json(pending / "COMPLETE.json", {"schema": "gift.independent-w3-job-commit.v1",
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
    """Parse one method/noise condition and reject checkpoint options for PDE-FIND regression."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--use-results", action="store_true")
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--condition", choices=tuple(DATASETS), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "w3")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args(argv)
    if args.method.startswith("PDE") and (args.checkpoint or args.checkpoint_sha256):
        parser.error("PDE-FIND does not use a trained checkpoint")
    if args.checkpoint_sha256 is not None and args.checkpoint is None:
        parser.error("--checkpoint-sha256 requires --checkpoint")
    return args


def main(argv=None):
    """Print a plan, execute a measurement, or read an already completed W3 job."""
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
                  "scope": "trained_coefficients_readout" if is_pinn else "one independent job, not the complete original W3 chain",
                  "training": False, "forward": args.method == "GIFT", "fresh_training": False,
                  "reference_archive_required": is_pinn, "archive_availability_checked": False,
                  "full_budget_retraining_verified": False, "next_action": "--execute"}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def gift_readout(
    project: Path, dataset: Path, checkpoint: Path, device_name: str, *, dataset_sha256=None
) -> dict[str, float]:
    """Estimate nu, beta and gamma by projecting frozen generator components on one training trajectory.

    Observed states have [1,11,64,64] shape at t=5.0..6.0 in steps of 0.1. P21
    constant and quadratic actions are compared with unit forcing/transport; a
    two-column state/Laplacian least-squares projection separates the linear action.
    The reference operators are used for coefficient readout, never weight updates.
    """
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
            raise ValueError("GIFT W3 training trajectory IDs differ")
        states = np.asarray(handle["training/vorticity"][:GIFT_TRAJECTORY_COUNT, selected], dtype=np.float32)
    if states.shape != (GIFT_TRAJECTORY_COUNT, 11, 64, 64) or not np.isfinite(states).all():
        raise ValueError("GIFT W3 readout states differ")

    device = torch.device(device_name)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("format_version") != 3
        or payload.get("artifact_role") != "trained_generator"
    ):
        raise ValueError("GIFT W3 checkpoint format differs")
    data_hash = payload.get("training_data", {}).get("sha256")
    if data_hash is not None and data_hash.upper() != (dataset_sha256 or digest(dataset)):
        raise ValueError("GIFT W3 checkpoint/data binding differs")
    configuration = payload["model_configuration"]
    if configuration.get("cutoff") != 21 or configuration.get("rank") != 8:
        raise ValueError("GIFT W3 checkpoint configuration differs")
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
