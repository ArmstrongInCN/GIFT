"""Observed-state sampling for full-data GIFT prediction training.

One epoch is one sampled window for each trajectory, visited once. No analytic
PDE derivatives, physical coefficients, test arrays or published weights are
read. State derivatives use the existing fourth-order observation stencil.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

from gift.data_splits import canonical_ids, TRAINING_IDS, CHECKPOINT_VALIDATION_IDS
from training.checkpoints import digest_file

DT = 0.02
ROLLOUT_OFFSETS = np.asarray((0, 5, 10, 20, 25, 30, 40, 50), dtype=np.int64)


def epoch_indices(population: int, frames: int, seed: int, epoch: int, *, rollout=False):
    """Index arrays address trajectory rows; each row appears exactly once."""
    if population < 1 or frames < (51 if rollout else 5) or epoch < 0:
        raise ValueError("invalid epoch sampling dimensions")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, int(rollout)]))
    order = rng.permutation(population)
    centres = (rng.integers(0, frames - 50, population, dtype=np.int64) if rollout
               else rng.integers(2, frames - 2, population, dtype=np.int64))
    return order, centres


def observed_pairs(raw: np.ndarray, order: np.ndarray, centres: np.ndarray):
    """Apply the original five-frame derivative stencil to sampled observations."""
    rows, times = order, centres[order]
    state = np.ascontiguousarray(raw[rows, times], dtype=np.float32)
    target = (raw[rows, times - 2] - 8.0 * raw[rows, times - 1]
              + 8.0 * raw[rows, times + 1] - raw[rows, times + 2]) / (12.0 * DT)
    return torch.from_numpy(state), torch.from_numpy(np.ascontiguousarray(target))


def observed_sequences(raw: np.ndarray, order: np.ndarray, anchors: np.ndarray):
    times = anchors[order, None] + ROLLOUT_OFFSETS[None]
    return np.ascontiguousarray(raw[order[:, None], times], dtype=np.float32)


class ObservationBank:
    """Load only train/validation observed vorticity, with scientific identity."""

    def __init__(self, path: Path, *, fixture: bool = False):
        path = Path(path).resolve(strict=True)
        self.binding = {"file": path.name, "sha256": digest_file(path),
                        "bytes": path.stat().st_size, "fixture_only": fixture}
        scientific = hashlib.sha256()
        with h5py.File(path, "r") as handle:
            if not fixture and bool(handle.attrs.get("fixture_only", False)):
                raise ValueError("fixture observations cannot be used for formal training")
            canonical = handle.attrs.get("trajectory_id_scheme") == "canonical"
            for group, expected in (("training", TRAINING_IDS),
                                    ("validation", CHECKPOINT_VALIDATION_IDS)):
                ids = tuple(map(int, handle[f"{group}/trajectory_index"][:]))
                public = ids if canonical else canonical_ids(ids)
                raw = handle[f"{group}/vorticity"]
                times = np.asarray(handle[f"{group}/time"][:], dtype=np.float64)
                if (raw.ndim != 4 or raw.shape[-1] != raw.shape[-2]
                        or raw.dtype != np.dtype("float32") or len(public) != raw.shape[0]
                        or len(set(public)) != len(public) or len(times) != raw.shape[1]
                        or not np.allclose(times, np.arange(len(times)) * DT, rtol=0, atol=2e-12)):
                    raise ValueError(f"invalid observed {group} schema")
                if not fixture and (public != expected or raw.shape != (len(expected), 501, 64, 64)):
                    raise ValueError(f"formal {group} identities/shape differ")
                if fixture and handle.attrs.get("fixture_only") != True:
                    raise ValueError("fixture mode requires an explicitly marked fixture")
                value = np.empty(raw.shape, dtype=np.float32)
                scientific.update(group.encode("ascii"))
                scientific.update(np.asarray(public, dtype="<i8").tobytes())
                scientific.update(times.astype("<f8").tobytes())
                square_sum = 0.0
                for row in range(len(public)):
                    value[row] = raw[row]
                    if not np.isfinite(value[row]).all():
                        raise ValueError(f"nonfinite observed {group} trajectory {public[row]}")
                    scientific.update(value[row].tobytes())
                    square_sum += float(np.square(value[row], dtype=np.float64).sum())
                setattr(self, group, value)
                self.binding[group + "_ids"] = list(public)
                if group == "training":
                    self.state_scale = float(np.sqrt(square_sum / value.size))
        if set(self.binding["training_ids"]) & set(self.binding["validation_ids"]):
            raise ValueError("training/validation identity overlap")
        if not np.isfinite(self.state_scale) or self.state_scale <= 0:
            raise ValueError("invalid training-only state scale")
        self.binding["observations_sha256"] = scientific.hexdigest()
        self.binding["time_step"] = DT

    def pairs(self, seed: int, epoch: int, *, validation=False):
        raw = self.validation if validation else self.training
        order, centres = epoch_indices(len(raw), raw.shape[1], seed, epoch)
        return observed_pairs(raw, order, centres)

    def sequences(self, seed: int, epoch: int, *, validation=False):
        raw = self.validation if validation else self.training
        order, anchors = epoch_indices(len(raw), raw.shape[1], seed, epoch, rollout=True)
        return observed_sequences(raw, order, anchors)
