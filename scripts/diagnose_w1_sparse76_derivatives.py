"""Quantify coarse five-point labels against dense observation differences.

The comparison uses the predeclared common training subset only. Dense
finite differences are an approximate diagnostic reference, not PDE truth,
additional training labels, or evidence of prediction accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def diagnose(cache, output):
    """Read only training t5..8; compare common centre times .08:.04:2.92."""
    ids = np.random.default_rng(2026100304).permutation(1000)[:300]
    centres = np.arange(2, 74, dtype=np.int64) * 2
    records, coarse_norms, dense_norms = [], [], []
    with h5py.File(cache, "r") as source:
        if source.attrs.get("schema") != "gift.w1-fair151.window.v2" or source.attrs.get("status") != "complete":
            raise ValueError("expected completed dense common-window cache")
        group = source["training"]
        if (not np.allclose(group["time"][:], np.arange(151)*.02, rtol=0, atol=2e-12)
                or not np.allclose(group["absolute_time"][:], 5 + np.arange(151)*.02, rtol=0, atol=2e-12)):
            raise ValueError("diagnostic dense cache physical time grid differs")
        cache_identity = dict(file=str(Path(cache).resolve()), schema=source.attrs["schema"],
                              training_observations_sha256=group.attrs["observations_sha256"])
        if not np.array_equal(group["trajectory_index"][:], np.arange(1000)):
            raise ValueError("diagnostic cache must have original ordered training IDs0..999")
        if group["vorticity"].shape != (1000, 151, 64, 64):
            raise ValueError("expected dense t5..8 training observations")
        for trajectory in ids:
            fields = np.asarray(group["vorticity"][trajectory], dtype=np.float64)
            dense = (fields[centres - 2] - 8 * fields[centres - 1]
                     + 8 * fields[centres + 1] - fields[centres + 2]) / (12 * .02)
            coarse = (fields[centres - 4] - 8 * fields[centres - 2]
                      + 8 * fields[centres + 2] - fields[centres + 4]) / (12 * .04)
            denominator = np.linalg.norm(dense.reshape(len(centres), -1), axis=1)
            if not np.isfinite(coarse).all() or not np.isfinite(dense).all() or np.any(denominator <= 0):
                raise ValueError("nonfinite or zero-norm diagnostic derivative")
            relative = np.linalg.norm((coarse - dense).reshape(len(centres), -1), axis=1) / denominator
            records.append(dict(trajectory_id=int(trajectory), relative_l2_by_time=relative.tolist(),
                                mean_relative_l2=float(relative.mean())))
            coarse_norms.append(float(np.square(coarse).sum()))
            dense_norms.append(float(np.square(dense).sum()))
    all_errors = np.array([row["relative_l2_by_time"] for row in records])
    payload = dict(schema="gift.w1-sparse76.derivative-diagnostic.v1", subset_seed=2026100304,
                   dense_cache=cache_identity, diagnostic_computation_dtype="float64 from float32 observations",
                   formal_target_dtype="native float32; diagnostic is not additional training labels",
                   training_ids=ids.tolist(), observations_only=True, validation_read=False,
                   test_read=False, used_as_training_labels=False, reference="dense .02 five-point observation difference, not exact PDE RHS",
                   observation_dt=.04, reference_observation_dt=.02, local_centre_times=(centres * .02).tolist(),
                   comparisons=int(all_errors.size), mean_relative_l2=float(all_errors.mean()),
                   median_relative_l2=float(np.median(all_errors)), p95_relative_l2=float(np.quantile(all_errors, .95)),
                   maximum_relative_l2=float(all_errors.max()),
                   coarse_to_dense_global_rms_ratio=float(np.sqrt(sum(coarse_norms) / sum(dense_norms))),
                   per_trajectory=records,
                   caveat="Difference of two finite-difference approximations; does not bound PDE derivative error or model prediction error.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in payload.items() if key not in ("per_trajectory", "training_ids")}))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.dense_window, args.output)
