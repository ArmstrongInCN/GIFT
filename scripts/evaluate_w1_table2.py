"""Compute Table 2 from a frozen W1 GIFT pair and confirmation observations.

All 180 trajectories and 76 times t=5..8 receive equal spatial/state weight.
Channels use physical vorticity/time units: C,Q on P21; A on nonzero P21.
Only equation coefficients are fitted; model weights remain frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from even_full_spectrum_ns import (
    EvenFullSpectrumConfiguration,
    EvenFullSpectrumNSSolver,
)
from experiments.formal._shared.common import (
    file_record,
    write_csv_new,
    write_json_new,
)
from experiments.model_locations import generator_checkpoint
from scripts.reproduce_prediction import output_root
from training.w1_v2_gift import load_gift_checkpoint


def evaluate(model, dataset, *, device="cpu", batch_size=16):
    """Return pooled parameter/channel metrics for [trajectory,time,y,x] data.

    FFT/model inference retains float32/complex64; physical-field products and
    pooled sums use float64. Linear readout jointly fits omega and Delta omega,
    so an additional state-proportional coefficient does not contaminate nu.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model.cutoff != 21:
        raise ValueError("Table 2 requires the N64, K21 W1 generator")
    device = torch.device(device)
    model = model.to(device).eval().requires_grad_(False)
    solver = EvenFullSpectrumNSSolver(
        EvenFullSpectrumConfiguration(grid=64, padding_grid=99), device=device
    )
    frequency = torch.fft.fftfreq(64, d=1 / 64, device=device)
    ky, kx = torch.meshgrid(frequency, frequency, indexing="ij")
    k2 = kx.square() + ky.square()
    retained = (kx.abs() <= 21) & (ky.abs() <= 21)
    nonzero = retained & (k2 > 0)
    sums = np.zeros(9, dtype=np.float64)
    energies = np.zeros(3, dtype=np.float64)
    errors = np.zeros(3, dtype=np.float64)
    times = 5 + .04 * np.arange(76)

    def inverse(spectrum):
        return torch.fft.ifft2(spectrum).real.double()

    def inner(left, right):
        return (left * right).sum(dim=(-2, -1))

    with h5py.File(dataset, "r") as handle, torch.inference_mode():
        group = handle["test"]
        ids = np.asarray(group["trajectory_index"])
        saved = np.asarray(group["absolute_time"])
        frames = np.arange(0, 151, 2)
        raw = group["vorticity"]
        if (raw.shape != (180, 151, 64, 64)
                or raw.dtype != np.dtype("float32")
                or not np.array_equal(ids, np.arange(100000, 100180))
                or saved.shape != (151,)
                or not np.allclose(saved[frames], times, rtol=0, atol=2e-10)):
            raise ValueError("Expected the full W1 confirmation set on N64")
        for trajectory in range(len(ids)):
            states = np.asarray(raw[trajectory, frames], dtype=np.float32)
            if not np.isfinite(states).all():
                raise ValueError(f"Nonfinite observation in trajectory {ids[trajectory]}")
            for start in range(0, len(times), batch_size):
                state = torch.from_numpy(states[start:start + batch_size]).to(device)
                state_hat = torch.fft.fft2(state)
                learned = model.components_hat(state)
                c = inverse(learned["constant"] * retained)
                a = inverse(learned["linear"] * nonzero)
                q = inverse(learned["quadratic"] * retained)
                f = inverse(solver.forcing_hat.expand_as(state_hat) * retained)
                w = inverse(state_hat * nonzero)
                d = inverse(-k2 * state_hat * nonzero)
                t = inverse(-solver.transport_hat(state_hat) * retained)
                # f2,Cf,T2,Qt,W2,WD,D2,AW,AD are sufficient for pooled LS.
                values = [inner(f, f), inner(c, f), inner(t, t), inner(q, t),
                          inner(w, w), inner(w, d), inner(d, d),
                          inner(a, w), inner(a, d)]
                sums += torch.stack(values, dim=-1).sum(0).cpu().numpy()
                # Table order: linear, nonlinear, bias field. Compare with true
                # physical terms, not their fitted coefficient projections.
                for index, (prediction, truth) in enumerate(
                        [(a, .01 * d), (q, t), (c, f)]):
                    energies[index] += float(inner(truth, truth).sum())
                    errors[index] += float(inner(prediction - truth, prediction - truth).sum())

    if (not np.isfinite(np.r_[sums, energies, errors]).all()
            or np.any(energies <= 0)):
        raise ValueError("Channel metrics require finite outputs and nonzero reference energy")
    gram = np.array([[sums[4], sums[5]], [sums[5], sums[6]]])
    scale = np.sqrt(np.diag(gram))
    a0, nu = np.linalg.solve(
        gram / np.outer(scale, scale), sums[7:9] / scale
    ) / scale
    estimates = [nu, sums[3] / sums[2], sums[1] / sums[0]]
    norms = np.sqrt(energies)
    rows = []
    for i, (parameter, channel, reference) in enumerate(
            [("nu", "linear", .01), ("beta", "nonlinear", 1.),
             ("gamma", "bias_field", 1.)]):
        rows.append({
            "parameter": parameter, "channel": channel, "reference": reference,
            "estimate": float(estimates[i]),
            "ape_percent": float(100 * abs(estimates[i] - reference) / reference),
            "channel_relative_l2_error_percent": float(100 * np.sqrt(errors[i] / energies[i])),
            "norm_share_percent": float(100 * norms[i] / norms.sum()),
            "energy_share_percent": float(100 * energies[i] / energies.sum()),
        })
    return {
        "rows": rows, "a0": float(a0), "trajectory_ids": ids.tolist(),
        "absolute_times": times.tolist(), "state_count": len(ids) * len(times),
        "reference_channel_energies": energies.tolist(),
        "channel_error_energies": errors.tolist(),
        "linear_gram": gram.tolist(), "linear_rhs": sums[7:9].tolist(),
        "definitions": {
            "channel_relative_l2_error_percent": "100*sqrt(sum((G_c-F_c)^2)/sum(F_c^2))",
            "norm_share_percent": "100*sqrt(sum(F_c^2))/sum_d(sqrt(sum(F_d^2)))",
            "energy_share_percent": "100*sum(F_c^2)/sum_d(sum(F_d^2))",
            "pooling": "Sum over all states and grid points before taking ratios",
            "linear_readout": "A=a0*P21_nonzero(omega)+nu*Delta(P21_nonzero(omega))",
        },
    }


def main(argv=None):
    """Read explicit scientific inputs and create a new CSV/JSON result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="W1 branch/model.pt; sibling generator required")
    parser.add_argument("--data", type=Path, required=True, help="W1 confirmation_n64.h5")
    parser.add_argument("--output", type=Path, required=True, help="new directory outside checkout and inputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    dataset = args.data.expanduser().resolve(strict=True)
    output = output_root(args.output, dataset.parent, checkpoint.parent.parent)
    if output.exists():
        raise FileExistsError(f"Use a new result directory: {output}")
    low, _, payload = load_gift_checkpoint(checkpoint, device=args.device)
    result = evaluate(low.model, dataset, device=args.device, batch_size=args.batch_size)
    result["inputs"] = {
        "checkpoint": file_record(checkpoint), "observations": file_record(dataset),
        "generator": file_record(generator_checkpoint(checkpoint, payload)),
    }
    result["generator_weights_sha256"] = payload["model_spec"]["generator_weights_sha256"]
    result["device"] = str(args.device)
    output.mkdir(parents=True, exist_ok=False)
    write_csv_new(output / "table2.csv", result["rows"])
    write_json_new(output / "table2.json", result)
    print(output / "table2.csv")


if __name__ == "__main__":
    main()
