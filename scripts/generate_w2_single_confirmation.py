"""Add paired N96/N128 references for the existing fixed N64 confirmation cohort.

Parameters and IDs come from the completed N64 artifact. No new draws, model,
target-grid fitting, rejection sampling, or interpolation enters generation.
The default command prints a plan; --run-generation commits a new own-run HDF.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from experiments.additional.w1_v2_evaluate import file_sha256, write_json
from scripts.generate_w1_v2_confirmation import locked_configuration

IDS = np.arange(100000, 100180, dtype=np.int64)
SAVED_STEPS = np.arange(1000, 1201, 4, dtype=np.int64)
ABSOLUTE_TIMES = SAVED_STEPS.astype(np.float64) * .005


def generation_inputs(base_lock, extension_lock, n64_confirmation):
    """Verify locked configurations and preserve the already evaluated population.

This is an extension of a previously evaluated confirmation cohort, not a new
blind test. That distinction is recorded before any high-resolution simulation.
"""
    base = locked_configuration(base_lock)
    extension_path = Path(extension_lock).resolve(strict=True)
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    if (extension.get("schema") != "gift.w1w2-native.extension-lock.v1"
            or extension.get("profile") != "single_native76_n300"
            or extension.get("configurations_locked") is not True):
        raise ValueError("W2 generation requires the fixed extension configurations lock")
    n64_path = Path(n64_confirmation).resolve(strict=True)
    with h5py.File(n64_path, "r") as handle:
        node = handle["test"]
        if (not bool(handle.attrs.get("complete", False))
                or handle.attrs.get("trajectory_id_scheme") != "canonical"
                or not np.array_equal(node["trajectory_index"][:], IDS)):
            raise ValueError("N64 source must retain the complete original confirmation180")
        parameters = node["initial_condition_parameters"][:]
        if parameters.shape != (180, 4, 4) or not np.isfinite(parameters).all():
            raise ValueError("N64 source must preserve finite exact initial parameters")
        source_times = node["absolute_time"][:]
        columns = np.searchsorted(source_times, ABSOLUTE_TIMES)
        if (np.any(columns >= len(source_times))
                or not np.allclose(source_times[columns], ABSOLUTE_TIMES, rtol=0, atol=1e-12)):
            # Float64 storage from the old generator can differ by one ulp.
            columns = np.asarray([np.argmin(np.abs(source_times-t)) for t in ABSOLUTE_TIMES])
            if not np.allclose(source_times[columns], ABSOLUTE_TIMES, rtol=0, atol=1e-12):
                raise ValueError("N64 confirmation lacks the actual t5..6 .02 reference nodes")
    return dict(base_lock_file=str(Path(base_lock).resolve()), base_lock_sha256=base["lock_sha256"],
                extension_lock_file=str(extension_path), extension_lock_sha256=file_sha256(extension_path),
                n64_source_file=str(n64_path), n64_source_sha256=file_sha256(n64_path),
                trajectory_ids=IDS.tolist(), parameters=parameters.tolist(), n64_columns=columns.tolist())


def generation_plan(inputs):
    """Return the paired numerical contract without simulating or selecting fields."""
    return dict(schema="gift.w2-single-native.confirmation-plan.v1", grids=[64, 96, 128],
                generated_grids=[96, 128], n64_reference="reuse unchanged completed N64 source",
                population_per_grid=180, trajectory_ids=IDS.tolist(), parameters_source=inputs,
                interpretation="paired extension of the already evaluated confirmation180; not a fresh blind cohort",
                solver="existing even-full-spectrum Fourier pseudospectral ETDRK4",
                physical_domain="[0,2pi)^2 periodic, array axes [y,x]", viscosity=.01,
                forcing_amplitude=1., forcing_wavenumber=4, vorticity_forcing="-4*cos(4*y)",
                internal_dt=.005, integrate_from=0., integrate_until=6., internal_steps=1200,
                padding_grids={"96": 147, "128": 195}, saved_interval=.02,
                absolute_times=ABSOLUTE_TIMES.tolist(), evaluation_absolute_times=(5+np.arange(26)*.04).tolist(),
                evaluation_reference_selection="exact dense51 nodes at stride2; no interpolation",
                stored_shape_by_grid={str(n): [180, 51, n, n] for n in (96, 128)},
                dtype="float32", all_prescribed_draws_retained=True, models_or_test_statistics_used=False)


def integrate_batch(parameters, grid, *, saved_steps=SAVED_STEPS, device="cpu"):
    """Integrate independent same-parameter fields and return [B,T,N,N] float32.

The original solver retains every native Fourier coefficient. Its padding is
alias-control quadrature, not an imposed state cutoff. A nonfinite reference
aborts the fixed population rather than dropping or replacing any draw.
"""
    import torch
    from even_full_spectrum_ns import EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver, build_initial_hat
    parameters, steps = np.asarray(parameters, np.float64), np.asarray(saved_steps)
    if grid not in (96, 128):
        raise ValueError("only missing N96/N128 references may be generated")
    if parameters.ndim != 3 or parameters.shape[1:] != (4, 4) or not len(parameters) or not np.isfinite(parameters).all():
        raise ValueError("finite nonempty [B,4,4] initial parameters required")
    if steps.ndim != 1 or not len(steps) or steps.dtype.kind not in "iu" or steps[0] < 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("save steps must be increasing nonnegative integer .005-step indices")
    solver = EvenFullSpectrumNSSolver(EvenFullSpectrumConfiguration(
        grid=grid, padding_grid={96: 147, 128: 195}[grid], viscosity=.01,
        forcing_amplitude=1., forcing_wavenumber=4, dt=.005), device=device)
    output = np.empty((len(parameters), len(steps), grid, grid), np.float32)
    with torch.inference_mode():
        state, save = build_initial_hat(parameters, grid=grid, device=device), 0
        for step in range(int(steps[-1])+1):
            if step == int(steps[save]):
                field = torch.fft.ifft2(state).real.cpu().numpy()
                if not np.isfinite(field).all():
                    raise FloatingPointError(f"nonfinite N{grid} reference at step {step}")
                output[:, save] = field
                save += 1
                if save == len(steps):
                    break
            state = solver.advance(state)
    return output, solver.metadata


def generate_confirmation(base_lock, extension_lock, n64_confirmation, output, *, device="cpu", batch_size=4, resume=False):
    """Persist paired references in a new directory and resume its committed batches."""
    if batch_size < 1:
        raise ValueError("positive generation batch size required")
    inputs = generation_inputs(base_lock, extension_lock, n64_confirmation)
    plan, parameters = generation_plan(inputs), np.asarray(inputs["parameters"], np.float64)
    output = Path(output).resolve()
    dataset = output / "confirmation_n96_n128_dt0p02.h5"
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        with h5py.File(dataset, "x") as handle:
            handle.attrs.update(complete=False, status="incomplete", trajectory_id_scheme="canonical",
                                n64_source_file=inputs["n64_source_file"], n64_source_sha256=inputs["n64_source_sha256"],
                                continuous_initial_parameters_shared_across_grids=True,
                                confirmation_previously_evaluated=True, reference_save_interval=.02)
            for grid in (96, 128):
                node = handle.create_group(f"N{grid}/test")
                node.attrs["completed_trajectories"] = 0
                node.create_dataset("trajectory_index", data=IDS)
                node.create_dataset("absolute_time", data=ABSOLUTE_TIMES)
                node.create_dataset("time", data=ABSOLUTE_TIMES-5.)
                node.create_dataset("initial_parameters", data=parameters)
                node.create_dataset("vorticity", shape=(180, 51, grid, grid), dtype="f4",
                                    chunks=(1, 1, grid, grid), compression="lzf", fillvalue=np.nan)
    elif json.loads((output / "plan.json").read_text(encoding="utf-8")) != plan:
        raise ValueError("resume must preserve the exact prior population, source and numerical plan")
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text(encoding="utf-8"))
    segment = output / f"segment_{len(list(output.glob('segment_*.json'))):04d}.json"
    write_json(segment, dict(started_utc=datetime.now(timezone.utc).isoformat(), device=str(device), batch_size=batch_size,
                            source_sha256=file_sha256(Path(__file__)), inputs=inputs))
    started = time.perf_counter()
    with h5py.File(dataset, "r+") as handle:
        for grid in (96, 128):
            node = handle[f"N{grid}/test"]
            if (not np.array_equal(node["trajectory_index"][:], IDS)
                    or not np.array_equal(node["initial_parameters"][:], parameters)
                    or not np.array_equal(node["absolute_time"][:], ABSOLUTE_TIMES)):
                raise ValueError("own-attempt saved identities, parameters or reference nodes changed")
            first = int(node.attrs["completed_trajectories"])
            if not 0 <= first <= 180:
                raise ValueError("invalid generation progress")
            for begin in range(first, 180, batch_size):
                end = min(begin+batch_size, 180)
                fields, metadata = integrate_batch(parameters[begin:end], grid, device=device)
                node["vorticity"][begin:end] = fields
                handle.flush()
                node.attrs["completed_trajectories"] = end
                node.attrs["solver_metadata_json"] = json.dumps(metadata, allow_nan=False)
                handle.flush()
                print(json.dumps(dict(status="w2_reference_committed", grid=grid, completed=end,
                                      population=180, elapsed_segment_seconds=time.perf_counter()-started)), flush=True)
        handle.attrs.update(complete=True, status="complete")
        handle.flush()
    report = dict(status="complete", data_file=str(dataset), n64_data_file=inputs["n64_source_file"],
                  grids=[64, 96, 128], population_per_grid=180, trajectory_ids=IDS.tolist(),
                  all_prescribed_draws_retained=True, elapsed_segment_seconds=time.perf_counter()-started,
                  interpretation=plan["interpretation"])
    write_json(output / "complete.json", report)
    return report


def main(argv=None):
    """Plan by default; generation is explicit and never opens model checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--extension-lock", type=Path, required=True)
    parser.add_argument("--n64-confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_generation:
        plan = generation_plan(generation_inputs(args.base_lock, args.extension_lock, args.n64_confirmation))
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return plan
    if args.output is None:
        parser.error("--run-generation requires --output")
    return generate_confirmation(args.base_lock, args.extension_lock, args.n64_confirmation, args.output,
                                 device=args.device, batch_size=args.batch_size, resume=args.resume)


if __name__ == "__main__":
    main()
