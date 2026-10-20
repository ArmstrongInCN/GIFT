"""Generate clean W3 trajectories for the fixed fifty equation instances.

The current single_trajectory_terminal_v1 profile contains only training ID0.
The historical validation_best_v1 profile also retains validation IDs50--69.
Fields are
fresh ETDRK4 integrations of the recorded four-vortex initial conditions;
case coefficients never enter a model-training target. This entry point only
generates and audits observations. Outputs must be outside the repository.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.generate_data import seed_parameters, runtime_record
from src.even_full_spectrum_ns import (
    EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver, build_initial_hat,
)

PARAMETER_CSV_SHA256 = "43b8b7d0edb5d70f4b7c7519946c7e358e88011db582f09c5d7e6940f1893a14"
DATA_FILENAME = "standard_ns_n64_full_spectrum.h5"
SAVED_STEPS = np.arange(0, 2001, 4, dtype=np.int64)
TIMES = SAVED_STEPS.astype(np.float64) * .005
SINGLE_PROFILE = "single_trajectory_terminal_v1"
LEGACY_PROFILE = "validation_best_v1"


def profile_schema(profile, record):
    """Keep historical schemas stable and identify newly integrated single inputs."""
    if profile not in (LEGACY_PROFILE, SINGLE_PROFILE):
        raise ValueError("unknown W3 data profile")
    kind = "single-trajectory" if profile == SINGLE_PROFILE else "parameter"
    return f"gift.w3-{kind}-{record}.v1"


def sha256(path: Path) -> str:
    """Stream a lowercase content digest without loading a full HDF container."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    """Create and sync finite evidence without overwriting an earlier record."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def utc_now() -> str:
    """Return a timezone-explicit timestamp for generation and CPU audit events."""
    return datetime.now(timezone.utc).isoformat()


def event(path: Path, value: dict) -> None:
    """Append one attempt's real cost; scientific training ledgers are separate."""
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def case_lease(path: Path):
    """Hold a nonblocking OS lease, released even after an interrupted process.

    The one-byte file remains as evidence. It is not a stale PID-based lock
    requiring deletion; the operating system owns and releases the lease.
    """
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def parameter_cases(path: Path) -> list[dict]:
    """Read the frozen CSV exactly; never redraw or replace failed instances."""
    path = Path(path).resolve(strict=True)
    if sha256(path) != PARAMETER_CSV_SHA256:
        raise ValueError("parameter CSV differs from the locked fifty-case design")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["case_id", "nu", "beta", "gamma"]:
            raise ValueError("parameter CSV columns differ")
        rows = [{"case_id": row["case_id"], "truth": {
            name: float(row[name]) for name in ("nu", "beta", "gamma")}} for row in reader]
    if [row["case_id"] for row in rows] != [f"parameter_{i:03d}" for i in range(50)]:
        raise ValueError("parameter CSV must contain all fifty ordered case identities")
    for row in rows:
        for name, bounds in {"nu": (.0075, .015), "beta": (.75, 1.25), "gamma": (.75, 1.25)}.items():
            if not bounds[0] <= row["truth"][name] <= bounds[1]:
                raise ValueError("coefficient outside the predetermined parameter domain")
    return rows


def initial_conditions(profile=LEGACY_PROFILE) -> dict:
    """Freeze ID0, adding the original validation population only for the old profile."""
    pools = seed_parameters()
    selected = {"training": (pools["training"][0][:1], pools["training"][1][:1])}
    if profile == LEGACY_PROFILE:
        selected["validation"] = pools["validation"]
    return {"schema": profile_schema(profile, "initial-conditions"),
            "parameter_axis": ["circulation", "sigma", "center_x", "center_y"],
            "domain": "[0,2*pi)^2; array axes [y,x]",
            "source": "scripts.generate_data.seed_parameters; no saved field used as an input",
            "splits": {split: {"ids": ids.tolist(), "parameters": parameters.tolist()}
                for split, (ids, parameters) in selected.items()}}


def data_lock(case: dict, initial_hash: str, profile=LEGACY_PROFILE) -> dict:
    """Bind each case's true equation and exact observation population."""
    initial = initial_conditions(profile)
    result = {"schema": profile_schema(profile, "data-lock"), "case_id": case["case_id"],
            "truth": case["truth"], "parameter_csv_sha256": PARAMETER_CSV_SHA256,
            "initial_conditions_sha256": initial_hash,
            "equation": "omega_t=-beta*(u*omega_x+v*omega_y)+nu*laplacian(omega)-4*gamma*cos(4*y)",
            "velocity": "unit Biot-Savart; beta does not scale velocity or initial fields",
            "solver": {"grid": 64, "padding_grid": 99, "internal_dt": .005,
                       "name": "full-spectrum Fourier pseudospectral ETDRK4",
                       "real_dtype": "float32", "complex_dtype": "complex64",
                       "forcing_wavenumber": 4, "state_mask_applied": False},
            "saved_steps": SAVED_STEPS.tolist(), "times": TIMES.tolist(),
            "data_quality_gates": {
                "fd_oracle_max_ape_percent": 1.,
                "initial_cpu_gpu_float32_difference_relative_bound": 5e-6,
                "zero_mean_relative_bound": 2e-6,
                "on_failure": "pause complete design and report; retain case, no redraw or protocol change"},
            "splits": {name: {"ids": item["ids"], "population": len(item["ids"]),
                              "vorticity_shape": [len(item["ids"]), 501, 64, 64]}
                       for name, item in initial["splits"].items()},
            "noise_fraction": 0., "all_prescribed_trajectories_retained": True}
    if profile == SINGLE_PROFILE:
        result.update(profile=profile, trajectory_count=1, monitor_is_training_data=True,
                      independent_validation=False)
        result["splits"]["validation"] = dict(ids=[], population=0,
            vorticity_shape=[0, 501, 64, 64], hdf_group_present=False)
    return result


def make_solver(truth: dict, *, device: str = "cpu") -> EvenFullSpectrumNSSolver:
    """Apply actual nu, beta and gamma to all four ETDRK4 stages."""
    return EvenFullSpectrumNSSolver(EvenFullSpectrumConfiguration(
        grid=64, padding_grid=99, viscosity=truth["nu"],
        advection_coefficient=truth["beta"], forcing_amplitude=truth["gamma"],
        forcing_wavenumber=4, dt=.005), device=device)


def integrate_batch(parameters, truth: dict, *, device="cpu", saved_steps=SAVED_STEPS):
    """Return fresh [B,T,64,64] float32 fields at exact internal step indices.

    Small integer schedules are allowed only for bounded CPU contract tests;
    the formal caller always supplies 0:4:2000. No interpolation is performed.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    steps = np.asarray(saved_steps)
    if (parameters.ndim != 3 or parameters.shape[1:] != (4, 4) or not len(parameters)
            or not np.isfinite(parameters).all()):
        raise ValueError("initial parameters must be finite nonempty [B,4,4]")
    if (steps.ndim != 1 or len(steps) == 0 or steps.dtype.kind not in "iu"
            or steps[0] != 0 or np.any(np.diff(steps) <= 0)):
        raise ValueError("save indices must be increasing integers starting at zero")
    solver = make_solver(truth, device=device)
    output = np.empty((len(parameters), len(steps), 64, 64), dtype=np.float32)
    with torch.inference_mode():
        state = build_initial_hat(parameters, grid=64, device=device)
        save_index = 0
        for step in range(int(steps[-1]) + 1):
            if step == int(steps[save_index]):
                field = torch.fft.ifft2(state).real.cpu().numpy()
                if not np.isfinite(field).all():
                    raise FloatingPointError(f"nonfinite generated trajectory at solver step {step}")
                output[:, save_index] = field
                save_index += 1
                if save_index == len(steps):
                    break
            state = solver.advance(state)
    return output, solver.metadata


def least_squares(design: np.ndarray, targets: np.ndarray, truth: dict) -> dict:
    """Audit unit-basis identifiability using normalized float64 Gram matrices."""
    x = np.asarray(design, dtype=np.float64).reshape(-1, 3)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    scales = np.linalg.norm(x, axis=0)
    if np.any(scales == 0.) or not np.isfinite(scales).all():
        raise ValueError("unit coefficient design has a zero or nonfinite column")
    normalized = x / scales
    gram = normalized.T @ normalized
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] <= np.finfo(np.float64).eps * 3 * eigenvalues[-1]:
        raise ValueError("unit coefficient design is rank deficient")
    coefficients = np.linalg.solve(gram, normalized.T @ y) / scales
    actual = np.asarray([truth[name] for name in ("nu", "beta", "gamma")])
    norm = np.linalg.norm(y)
    return {"unit_basis": ["laplacian", "negative_unit_transport", "q=-4*cos(4*y)"],
            "coefficients": coefficients.tolist(),
            "ape_percent": (100 * np.abs(coefficients - actual) / np.abs(actual)).tolist(),
            "rank": 3, "normalized_design_condition": float(np.sqrt(eigenvalues[-1] / eigenvalues[0])),
            "rhs_relative_residual": float(np.linalg.norm(y - x @ coefficients) / norm)}


def audit_case(dataset: Path, lock: dict) -> dict:
    """Check all fields and the original t5--6 eleven-node unit-basis FD oracle.

    This is observation quality evidence, not a fitted algorithm result. The
    transport uses full states before P21 output projection, matching the
    earlier data precheck. It does not alter GIFT's P21 model or readout.
    """
    started = time.perf_counter()
    profile = lock.get("profile", LEGACY_PROFILE)
    initial = initial_conditions(profile)
    results = {}
    with h5py.File(dataset, "r") as handle, torch.inference_mode():
        if profile == SINGLE_PROFILE and set(handle) != {"training"}:
            raise ValueError("single-trajectory observations must contain only training ID0")
        if (handle.attrs.get("case_id") != lock["case_id"]
                or json.loads(handle.attrs["truth_json"]) != lock["truth"]):
            raise ValueError("HDF case equation binding differs")
        for split, item in initial["splits"].items():
            node = handle[split]
            if (tuple(node["vorticity"].shape) != tuple(lock["splits"][split]["vorticity_shape"])
                    or node["vorticity"].dtype != np.dtype("float32")
                    or not np.array_equal(node["trajectory_index"][:], item["ids"])
                    or not np.array_equal(node["initial_condition_parameters"][:], item["parameters"])
                    or not np.array_equal(node["time"][:], TIMES)
                    or int(node.attrs["completed_trajectories"]) != len(item["ids"])):
                raise ValueError("HDF population, progress, IDs, initial parameters or exact times differ")
            maximum, mean = 0., 0.
            for i in range(len(item["ids"])):
                for begin in range(0, 501, 32):
                    fields = node["vorticity"][i, begin:begin+32]
                    if not np.isfinite(fields).all():
                        raise FloatingPointError("nonfinite or unwritten formal observation")
                    maximum = max(maximum, float(np.abs(fields).max()))
                    mean = max(mean, float(np.abs(fields.mean(axis=(-2, -1), dtype=np.float64)).max()))
            expected = torch.fft.ifft2(build_initial_hat(item["parameters"], device="cpu")).real.numpy()
            observed = node["vorticity"][:, 0]
            difference = float(np.max(np.abs(observed.astype(np.float64)-expected.astype(np.float64))))
            # Parameters are exact. FFT fields can vary across CPU/GPU libraries
            # at float32 rounding, so this bound is explicit rather than a claim
            # of cross-device bitwise field equality.
            initial_bound = 5e-6 * max(1., float(np.abs(expected).max()))
            if difference > initial_bound or mean > 2e-6 * max(1., maximum):
                raise ValueError("initial field or zero-mean quality bound exceeded")
            results[split] = {"population": len(item["ids"]), "frames": 501,
                             "finite_all_fields": True, "max_absolute_vorticity": maximum,
                             "max_absolute_spatial_mean": mean,
                             "initial_max_absolute_difference_from_cpu": difference,
                             "initial_difference_bound": initial_bound,
                             "exact_initial_parameters": True, "exact_ids_and_time": True}
        # The original W3 readout uses these eleven time nodes. Its derivative
        # stencil includes the four adjacent .02 observations at each node.
        centers = np.arange(250, 301, 5, dtype=np.int64)
        selected = np.unique(np.concatenate([centers + offset for offset in (-2,-1,0,1,2)]))
        gathered = handle["training/vorticity"][0, selected]
        lookup = {int(node): gathered[i] for i, node in enumerate(selected)}
    fields = np.stack([lookup[int(i)] for i in centers])
    fd = np.stack([(lookup[int(i-2)]-8*lookup[int(i-1)]+8*lookup[int(i+1)]-lookup[int(i+2)])
                   / (12*.02) for i in centers])
    solver = make_solver({"nu": .01, "beta": 1., "gamma": 1.}, device="cpu")
    mask21 = (solver.kx.abs() <= 21) & (solver.ky.abs() <= 21)
    with torch.inference_mode():
        wh = torch.fft.fft2(torch.from_numpy(fields))
        lap = torch.fft.ifft2(-solver.k2 * wh * mask21).real.numpy()
        transport = torch.fft.ifft2(-solver.transport_hat(wh) * mask21).real.numpy()
        q = torch.fft.ifft2(solver.forcing_hat).real.numpy()
        design = np.stack([lap, transport, np.broadcast_to(q, fields.shape)], axis=-1)
        fd_projected = torch.fft.ifft2(torch.fft.fft2(torch.from_numpy(fd))*mask21).real.numpy()
    actual = np.asarray([lock["truth"][name] for name in ("nu", "beta", "gamma")])
    exact_rhs = np.einsum("txyc,c->txy", design.astype(np.float64), actual)
    oracle = {"time_nodes": TIMES[centers].tolist(), "training_id": 0,
              "purpose": "data quality only; not an algorithm score or training target",
              "analytic_unit_fit": least_squares(design, exact_rhs, lock["truth"]),
              "original_fp32_fourth_order_fd_fit": least_squares(design, fd_projected, lock["truth"])}
    # Nine predetermined precheck points were below .1%. The fixed 1% gate is
    # a data-integrity safeguard, not a model-performance selection criterion;
    # failure pauses the complete design without replacing the case.
    if max(oracle["original_fp32_fourth_order_fd_fit"]["ape_percent"]) > 1.:
        raise ValueError("original finite-difference oracle coefficient error exceeds fixed 1% data gate")
    result = {"schema": profile_schema(profile, "data-qa"), "status": "passed",
            "case_id": lock["case_id"], "truth": lock["truth"], "splits": results,
            "unit_operator_oracle": oracle, "fd_oracle_gate_max_ape_percent": 1.,
            "cpu_audit_seconds": time.perf_counter()-started}
    if profile == SINGLE_PROFILE:
        result.update(profile=profile, physical_trajectory_count=1, validation_group_present=False)
    return result


def generate_case(data_root: Path, case: dict, initial_hash: str, *, device="cpu", batch_size=7,
                  resume=False, profile=LEGACY_PROFILE):
    """Commit one case, then audit and bind it; resume own committed batches only.

    An interrupted uncommitted trajectory batch is reintegrated from its own
    identical initial conditions. Committed batches are not regenerated; all
    attempted generation wall time is retained in the independent event log.
    """
    output = data_root / case["case_id"]
    output.mkdir(parents=False, exist_ok=resume)
    lock = data_lock(case, initial_hash, profile)
    lock_path, dataset = output / "DATA_LOCK.json", output / DATA_FILENAME
    initial = initial_conditions(profile)
    with case_lease(output / "GENERATION_LEASE.lock"):
        receipt_path = output / "DATA_RECEIPT.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("schema") != profile_schema(profile, "data-receipt")
                    or json.loads(lock_path.read_text(encoding="utf-8")) != lock
                    or receipt.get("status") != "passed" or receipt.get("truth") != case["truth"]
                    or receipt.get("data_lock_sha256") != sha256(lock_path)
                    or receipt.get("data_qa_sha256") != sha256(output / "DATA_QA.json")
                    or receipt.get("data_file", {}).get("sha256") != sha256(dataset)):
                raise ValueError("existing completed data receipt no longer matches its own artifacts")
            return receipt
        if lock_path.exists():
            if not resume or json.loads(lock_path.read_text(encoding="utf-8")) != lock:
                raise ValueError("resume case equation/population configuration differs")
        else:
            write_new(lock_path, lock)
        mode = "r+" if dataset.exists() else "x"
        started, timestamp = time.perf_counter(), utc_now()
        sources = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in
                   (Path(__file__), ROOT/"scripts/generate_data.py", ROOT/"src/even_full_spectrum_ns.py")}
        event_path = output / "DATA_GENERATION_EVENTS.jsonl"
        identity = {"case_id": case["case_id"], "pid": os.getpid(), "started_utc": timestamp,
                    "device": str(device), "batch_size": batch_size, "sources": sources}
        event(event_path, {**identity, "event": "started", "runtime": runtime_record(torch, torch.device(device))})
        try:
            with h5py.File(dataset, mode) as handle:
                if mode == "x":
                    handle.attrs.update(case_id=case["case_id"], truth_json=json.dumps(case["truth"]),
                                        complete=False, status="incomplete", internal_dt=.005,
                                        noise_fraction=0., generated_from="explicit_initial_conditions",
                                        trajectory_id_scheme="original_W3_storage_ids", grid=64,
                                        saved_interval=.02, padding_grid=99)
                    for split, item in initial["splits"].items():
                        node = handle.create_group(split)
                        node.attrs["completed_trajectories"] = 0
                        node.create_dataset("trajectory_index", data=np.asarray(item["ids"], dtype=np.int64))
                        node.create_dataset("time", data=TIMES)
                        node.create_dataset("initial_condition_parameters", data=np.asarray(item["parameters"], dtype=np.float64))
                        node.create_dataset("vorticity", shape=(len(item["ids"]),501,64,64), dtype="f4",
                                            chunks=(1,1,64,64), compression="lzf", shuffle=True, fillvalue=np.nan)
                    handle.flush()
                if (handle.attrs.get("case_id") != case["case_id"]
                        or json.loads(handle.attrs["truth_json"]) != case["truth"]):
                    raise ValueError("resume HDF belongs to a different case or equation")
                for split, item in initial["splits"].items():
                    node = handle[split]
                    if (not np.array_equal(node["trajectory_index"][:], item["ids"])
                            or not np.array_equal(node["initial_condition_parameters"][:], item["parameters"])
                            or not np.array_equal(node["time"][:], TIMES)):
                        raise ValueError("resume HDF observed population or initial states differ")
                    first = int(node.attrs["completed_trajectories"])
                    if not 0 <= first <= len(item["ids"]):
                        raise ValueError("invalid own-case committed progress")
                    for begin in range(first, len(item["ids"]), batch_size):
                        end = min(begin+batch_size, len(item["ids"]))
                        batch_started = time.perf_counter()
                        fields, metadata = integrate_batch(item["parameters"][begin:end], case["truth"], device=device)
                        node["vorticity"][begin:end] = fields
                        handle.flush()
                        node.attrs["completed_trajectories"] = end
                        handle.attrs["solver_metadata_json"] = json.dumps(metadata, allow_nan=False)
                        handle.flush()
                        record = {**identity, "event": "batch_committed", "split": split,
                                  "begin": begin, "end": end, "elapsed_seconds": time.perf_counter()-batch_started,
                                  "committed_utc": utc_now()}
                        event(event_path, record)
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                handle.attrs.update(complete=True, status="complete")
                handle.flush()
            qa_path = output / "DATA_QA.json"
            qa = audit_case(dataset, lock)
            if qa_path.exists():
                original = json.loads(qa_path.read_text(encoding="utf-8"))
                if original.get("status") != "passed" or original.get("truth") != case["truth"]:
                    raise ValueError("existing data audit is incompatible")
            else:
                write_new(qa_path, qa)
            receipt = {"schema": profile_schema(profile, "data-receipt"), "status": "passed",
                       "case_id": case["case_id"], "truth": case["truth"],
                       "parameter_csv_sha256": PARAMETER_CSV_SHA256,
                       "initial_conditions_sha256": initial_hash,
                       "data_lock_sha256": sha256(lock_path), "data_qa_sha256": sha256(qa_path),
                       "data_file": {"path": str(dataset.resolve()), "relative_path": DATA_FILENAME,
                                     "sha256": sha256(dataset), "bytes": dataset.stat().st_size},
                       "completed_utc": utc_now(), "generation_sources": sources}
            if profile == SINGLE_PROFILE:
                receipt.update(profile=profile, training_trajectories_per_case=1,
                    validation_trajectories_per_case=0, new_integration_performed=True)
            write_new(receipt_path, receipt)
            event(event_path, {**identity, "event": "completed", "elapsed_seconds": time.perf_counter()-started,
                               "data_receipt_sha256": sha256(receipt_path), "completed_utc": utc_now()})
            return receipt
        except BaseException as exc:
            event(event_path, {**identity, "event": "interrupted_or_failed", "elapsed_seconds": time.perf_counter()-started,
                               "exception": type(exc).__name__, "message": str(exc), "ended_utc": utc_now()})
            raise


def audit_complete_cohort(data_root: Path, cases: list[dict], profile=LEGACY_PROFILE) -> dict:
    """Require all fifty passed observation receipts with matching real file hashes.

    Per-case numerical QA is already bound to its exact HDF bytes. This closure
    checks the complete population rather than accepting a subset of successful
    cases; transferred absolute source paths are descriptive, relative paths
    are checked against the current data root.
    """
    if [case["case_id"] for case in cases] != [f"parameter_{i:03d}" for i in range(50)]:
        raise ValueError("complete cohort audit requires the fifty locked cases")
    initial_hash = sha256(data_root / "INITIAL_CONDITIONS.json")
    records = []
    for case in cases:
        folder = data_root / case["case_id"]
        receipt_path, qa_path, lock_path = (folder / name for name in
                                           ("DATA_RECEIPT.json", "DATA_QA.json", "DATA_LOCK.json"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        dataset = folder / DATA_FILENAME
        if (receipt.get("schema") != profile_schema(profile, "data-receipt")
                or receipt.get("status") != "passed" or qa.get("status") != "passed"
                or receipt.get("case_id") != case["case_id"] or qa.get("case_id") != case["case_id"]
                or receipt.get("truth") != case["truth"] or qa.get("truth") != case["truth"]
                or lock != data_lock(case, initial_hash, profile)
                or receipt.get("parameter_csv_sha256") != PARAMETER_CSV_SHA256
                or receipt.get("initial_conditions_sha256") != initial_hash
                or receipt.get("data_lock_sha256") != sha256(lock_path)
                or receipt.get("data_qa_sha256") != sha256(qa_path)
                or receipt.get("data_file", {}).get("relative_path") != DATA_FILENAME
                or receipt.get("data_file", {}).get("sha256") != sha256(dataset)
                or receipt.get("data_file", {}).get("bytes") != dataset.stat().st_size):
            raise ValueError(f"complete cohort has incompatible or damaged case {case['case_id']}")
        records.append({"case_id": case["case_id"], "truth": case["truth"],
                        "data_sha256": receipt["data_file"]["sha256"],
                        "data_receipt_sha256": sha256(receipt_path),
                        "data_lock_sha256": sha256(lock_path), "data_qa_sha256": sha256(qa_path),
                        "fd_oracle_max_ape_percent": max(qa["unit_operator_oracle"]
                                                        ["original_fp32_fourth_order_fd_fit"]["ape_percent"])})
    result = {"schema": profile_schema(profile, "data-cohort-qa"), "status": "passed",
            "cases": records, "case_count": 50,
            "trajectory_count": 50 if profile == SINGLE_PROFILE else 1050,
            "parameter_csv_sha256": PARAMETER_CSV_SHA256,
            "initial_conditions_sha256": initial_hash,
            "all_case_file_hashes_and_numerical_audits_bound": True,
            "all_prescribed_cases_retained": True}
    if profile == SINGLE_PROFILE:
        result.update(profile=profile, training_trajectories_per_case=1,
            validation_trajectories_per_case=0, training_trajectory_count=50,
            validation_trajectory_count=0, new_integration_performed=True)
    return result


def write_status(path: Path, value: dict) -> None:
    """Replace only the operational status pointer; evidence journals remain append-only."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None) -> int:
    """Plan by default; explicit execution creates only the selected fixed cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="external run root; data live under root/data")
    parser.add_argument("--parameter-csv", type=Path, required=True)
    parser.add_argument("--profile", choices=(LEGACY_PROFILE, SINGLE_PROFILE), default=LEGACY_PROFILE,
                        help="explicit current one-trajectory or historical validation protocol")
    parser.add_argument("--case-id", action="append", help="repeat to assign fixed cases to a compute device")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-all", action="store_true", help="CPU closure audit of all fifty existing case receipts")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    run = args.root.expanduser().resolve()
    if run == ROOT or ROOT in run.parents:
        raise ValueError("new observations must stay outside the project repository")
    if os.environ.get("GIFT_DATA_ROOT"):
        original = Path(os.environ["GIFT_DATA_ROOT"]).expanduser().resolve()
        if run == original or original in run.parents:
            raise ValueError("new observations must stay outside original GIFT_DATA_ROOT")
    all_cases = parameter_cases(args.parameter_csv)
    cases = all_cases
    if args.case_id:
        if len(set(args.case_id)) != len(args.case_id) or not set(args.case_id) <= {row["case_id"] for row in cases}:
            raise ValueError("selected case identities must be unique members of the fixed design")
        cases = [row for row in cases if row["case_id"] in args.case_id]
    print(json.dumps({"schema": "gift.w3-parameter-data-plan.v1", "run_root": str(run),
                      "parameter_csv_sha256": PARAMETER_CSV_SHA256, "cases": cases,
                      "profile": args.profile,
                      "observations_per_case": {"training": 1,
                          "validation": 0 if args.profile == SINGLE_PROFILE else 20},
                      "frames": 501, "device": args.device, "execute": args.execute}, ensure_ascii=False), flush=True)
    if args.audit_all:
        if args.case_id:
            raise ValueError("all-cohort audit cannot use a case subset")
        report = audit_complete_cohort(run / "data", all_cases, args.profile)
        audit_path = run / "data" / "DATA_COHORT_QA.json"
        if audit_path.exists():
            if json.loads(audit_path.read_text(encoding="utf-8")) != report:
                raise ValueError("existing cohort audit differs from actual complete data")
        else:
            write_new(audit_path, report)
        print(json.dumps({"status": "passed", "data_cohort_qa": str(audit_path)}), flush=True)
        return 0
    if not args.execute:
        return 0
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = run / "data"
    data.mkdir(parents=True, exist_ok=True)
    parameter_copy = data / "parameters.csv"
    if parameter_copy.exists():
        if sha256(parameter_copy) != PARAMETER_CSV_SHA256:
            raise ValueError("run-local parameter CSV differs from the fixed design")
    else:
        with parameter_copy.open("xb") as stream:
            stream.write(args.parameter_csv.read_bytes())
    initial_path = data / "INITIAL_CONDITIONS.json"
    initial = initial_conditions(args.profile)
    if initial_path.exists():
        if json.loads(initial_path.read_text(encoding="utf-8")) != initial:
            raise ValueError("shared explicit initial-condition list differs")
    else:
        write_new(initial_path, initial)
    initial_hash = sha256(initial_path)
    status_path = data / f"DATA_GENERATION_STATUS_{os.getpid()}.json"
    identity = {"schema": "gift.w3-parameter-data-generation-status.v1", "pid": os.getpid(),
                "parent_pid": os.getppid(), "started_utc": utc_now(), "device": args.device,
                "assigned_cases": [case["case_id"] for case in cases], "completed_cases": []}
    try:
        for case in cases:
            write_status(status_path, {**identity, "status": "running", "current_case": case["case_id"]})
            receipt = generate_case(data, case, initial_hash, device=args.device,
                                    batch_size=args.batch_size, resume=args.resume, profile=args.profile)
            identity["completed_cases"].append(case["case_id"])
            print(json.dumps({"event": "case_data_passed", "case_id": case["case_id"],
                              "data_sha256": receipt["data_file"]["sha256"]}), flush=True)
        if len(cases) == 50:
            report = audit_complete_cohort(data, all_cases, args.profile)
            audit_path = data / "DATA_COHORT_QA.json"
            if audit_path.exists():
                if json.loads(audit_path.read_text(encoding="utf-8")) != report:
                    raise ValueError("existing complete cohort audit differs")
            else:
                write_new(audit_path, report)
        write_status(status_path, {**identity, "status": "complete", "completed_utc": utc_now(),
                                  "complete_fifty_case_cohort": len(cases) == 50})
    except BaseException as exc:
        write_status(status_path, {**identity, "status": "paused_or_failed", "current_case": case["case_id"],
                                  "ended_utc": utc_now(), "exception": type(exc).__name__, "message": str(exc)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
