"""Generate diagnostic cross-domain fixture observations for GIFT transferability tests.

Three first-order scalar systems are written on the same observation schema the
prediction trainers already consume (square grid, 501 frames, dt = 0.02, float32,
groups ``training`` / ``validation`` / ``test`` with ``trajectory_index`` and
``time``):

  heat       u_t = kappa * Lap(u)                        kappa = 0.03
  advect     u_t = -c1 u_x - c2 u_y                      (c1, c2) = (0.7, 0.4)
  reactdiff  u_t = 0.02 * Lap(u) + r u (1 - u)           r = 1.0

Heat and advection use exact spectral solutions; reaction-diffusion is integrated
with plain RK4 on half-frame internal steps, using the spectral Laplacian.

One fixed physical parameter set per file keeps a single autonomous system with
varying initial conditions only. This is the setting the generator's
state-to-derivative map assumes; mixing parameters across trajectories makes that
map ill-posed (measured during the transferability study, see EXPERIMENTS.md).

Every file carries ``fixture_only = True``: these observations are diagnostic
inputs and cannot qualify as formal project experiments. Run commands from the
project root; the output must be a new directory outside the repository.

See ``scripts/run_crossdomain.py`` for training and evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_data import ROOT, safe_output, sha256, write_json_new  # noqa: E402

# Observation dt of 0.02 matches the project's stored frame spacing, so these
# fixtures reuse the training/evaluation readers unchanged.
DT = 0.02
TRAIN_IDS = tuple(range(250))
VALIDATION_IDS = tuple(range(1000, 1020))
TEST_IDS = tuple(range(2000, 2020))
FIXED = {"heat": {"kappa": 0.03}, "advect": {"c": (0.7, 0.4)}, "reactdiff": {"r": 1.0}}
EQUATIONS = {"heat": "u_t = kappa * Lap(u)",
             "advect": "u_t = -c1 u_x - c2 u_y",
             "reactdiff": "u_t = 0.02 * Lap(u) + r u (1 - u)"}


def wave_numbers(grid):
    axis = np.fft.fftfreq(grid) * grid
    kx, ky = np.meshgrid(axis, axis)
    return kx, ky, kx * kx + ky * ky


def smooth_field(rng, grid, k0=8.0):
    """Zero-mean smooth random field with spectral decay exp(-(k/k0)^2), k0 = 8."""
    kx, ky, k2 = wave_numbers(grid)
    spectrum = np.fft.fft2(rng.standard_normal((grid, grid))) * np.exp(-0.5 * (np.sqrt(k2) / k0) ** 2)
    field = np.fft.ifft2(spectrum).real
    field -= field.mean()
    return field.astype(np.float64)


def frames_from_factor(initial_spectrum, frames, factor):
    """Stack inverse transforms of ``initial_spectrum * factor(t)`` for t = i * DT."""
    out = np.empty((frames,) + initial_spectrum.shape, dtype=np.float32)
    for index in range(frames):
        out[index] = np.fft.ifft2(initial_spectrum * factor(index * DT)).real.astype(np.float32)
    return out


def heat_trajectory(rng, grid, frames):
    """Exact solution: each mode decays as exp(-kappa |k|^2 t)."""
    kappa = FIXED["heat"]["kappa"]
    _, _, k2 = wave_numbers(grid)
    initial = np.fft.fft2(smooth_field(rng, grid))
    return frames_from_factor(initial, frames, lambda t: np.exp(-kappa * k2 * t))


def advect_trajectory(rng, grid, frames):
    """Exact solution: each mode rotates in phase by exp(-i (c1 kx + c2 ky) t)."""
    c1, c2 = FIXED["advect"]["c"]
    kx, ky, _ = wave_numbers(grid)
    initial = np.fft.fft2(smooth_field(rng, grid))
    return frames_from_factor(initial, frames, lambda t: np.exp(-1j * (c1 * kx + c2 * ky) * t))


def reactdiff_rhs(state, grid, r, kappa=0.02):
    _, _, k2 = wave_numbers(grid)
    laplacian = np.fft.ifft2(-k2 * np.fft.fft2(state)).real
    return kappa * laplacian + r * state * (1.0 - state)


def reactdiff_trajectory(rng, grid, frames):
    """Plain RK4 with half-frame internal steps; the source term is nonlinear."""
    r = FIXED["reactdiff"]["r"]
    state = np.clip(0.5 + 0.3 * smooth_field(rng, grid), 0.05, 0.95)
    out = np.empty((frames, grid, grid), dtype=np.float32)
    out[0] = state.astype(np.float32)
    half = DT / 2.0
    for index in range(1, frames):
        for _ in range(2):
            k1 = reactdiff_rhs(state, grid, r)
            k2 = reactdiff_rhs(state + 0.5 * half * k1, grid, r)
            k3 = reactdiff_rhs(state + 0.5 * half * k2, grid, r)
            k4 = reactdiff_rhs(state + half * k3, grid, r)
            state = state + (half / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[index] = state.astype(np.float32)
    return out


TRAJECTORY = {"heat": heat_trajectory, "advect": advect_trajectory,
              "reactdiff": reactdiff_trajectory}


def populations(kind, seed, grid, frames, counts):
    rng = np.random.default_rng(seed)
    return {group: np.stack([TRAJECTORY[kind](rng, grid, frames) for _ in range(count)])
            for group, count in counts.items()}


def write_dataset(path, kind, fields, ids, frames):
    with h5py.File(path, "x") as handle:
        # fixture_only marks these as diagnostic inputs that can never qualify as
        # a formal experiment; canonical IDs keep the reader's identity logic simple.
        handle.attrs["fixture_only"] = True
        handle.attrs["trajectory_id_scheme"] = "canonical"
        handle.attrs["equation"] = EQUATIONS[kind]
        handle.attrs["observation"] = "scalar field, periodic square grid"
        handle.attrs["description"] = (
            "GIFT cross-domain diagnostic fixture; not a formal project dataset")
        for name, value in FIXED[kind].items():
            handle.attrs["parameter_" + name] = np.asarray(value, dtype=np.float64)
        for group in ("training", "validation", "test"):
            group_handle = handle.create_group(group)
            group_handle.create_dataset("vorticity", data=fields[group])
            group_handle.create_dataset("trajectory_index", data=np.asarray(ids[group], dtype=np.int64))
            group_handle.create_dataset("time", data=np.arange(frames) * DT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=tuple(EQUATIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="write the dataset; the default is a read-only plan")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--frames", type=int, default=501)
    parser.add_argument("--train-count", type=int, default=len(TRAIN_IDS))
    parser.add_argument("--validation-count", type=int, default=len(VALIDATION_IDS))
    parser.add_argument("--test-count", type=int, default=len(TEST_IDS))
    args = parser.parse_args(argv)

    if args.grid < 43 or args.grid % 2:
        # The model cutoff is 21 modes, so the grid needs at least 2*21+1 = 43
        # points and must be even for the real FFT/spectral Laplacian.
        parser.error("grid must be an even size of at least 43 (model cutoff 21)")
    if args.frames < 51:
        parser.error("frames must be at least 51 (5-frame stencil plus 50-step rollout window)")
    counts = {"training": args.train_count, "validation": args.validation_count,
              "test": args.test_count}
    if min(counts.values()) < 1:
        parser.error("population counts must be positive")
    ids = {"training": TRAIN_IDS[:counts["training"]],
           "validation": VALIDATION_IDS[:counts["validation"]],
           "test": TEST_IDS[:counts["test"]]}

    plan = {"kind": args.kind, "equation": EQUATIONS[args.kind], "parameters": FIXED[args.kind],
            "seed": args.seed, "grid": args.grid, "frames": args.frames,
            "time_step": DT, "populations": counts,
            "fixture_only": True, "output": str(args.output)}
    if not args.execute:
        print(plan)
        print("plan only; add --execute to write the dataset")
        return

    output = safe_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dataset_path = output / f"{args.kind}.h5"
    fields = populations(args.kind, args.seed, args.grid, args.frames, counts)
    write_dataset(dataset_path, args.kind, fields, ids, args.frames)
    for group in ("training", "validation", "test"):
        values = fields[group]
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {group} observations")
    provenance = {"plan": plan, "dataset": dataset_path.name,
                  "sha256": sha256(dataset_path),
                  "shapes": {group: list(fields[group].shape) for group in counts},
                  "ids": {group: list(ids[group]) for group in counts},
                  "generator": "scripts/generate_crossdomain_data.py",
                  "project_root_name": ROOT.name,
                  "scope": "diagnostic fixture observations; cannot qualify as a formal project experiment"}
    write_json_new(output / "PROVENANCE.json", provenance)
    print(f"wrote {dataset_path} sha256={provenance['sha256']}")
    print(f"wrote {output / 'PROVENANCE.json'}")


if __name__ == "__main__":
    main()
