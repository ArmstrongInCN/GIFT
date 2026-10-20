"""Train/validation-only observed windows for the ANIE prediction baseline."""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

TRAIN_FILE = "fno/fno1000_n64_t0_t10_dt0p02.h5"
DT = 0.02
WINDOW_FRAMES = 151
VALIDATION_ANCHOR = 250
TRAIN_INDICES = np.linspace(0, WINDOW_FRAMES - 1, 10, dtype=np.int64)


def epoch_indices(population: int, frames: int, seed: int, epoch: int):
    """Return a permutation and one row-indexed 151-frame anchor per trajectory.

    Use an independent SeedSequence([seed,epoch]), with zero-based epoch.
    Every selected row appears once; anchors are inclusive from 0 through
    frames-151, so each observed window fits without dropping a tail.
    """
    if population < 1 or frames < WINDOW_FRAMES or epoch < 0:
        raise ValueError("invalid ANIE epoch sampling dimensions")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    return (rng.permutation(population),
            rng.integers(0, frames - WINDOW_FRAMES + 1, population, dtype=np.int64))


class ObservationBank:
    """Expose train/validation windows, checking loaded observations for finiteness."""

    def __init__(self, root: Path, path: Path, *, formal: bool, trajectories: int,
                 cache: bool = False, verify_full: bool = True):
        """Open float32 train/validation observations with axes (trajectory, frame, y, x).

        ANIE requires 64 by 64 fields sampled every DT=0.02 seconds and at least
        151 frames. cache=True copies selected data to RAM and closes HDF5;
        otherwise slices stay lazy until close(). A requested full scan or cache
        fill checks finite values and fingerprints the decoded scientific arrays.
        """
        root, path = Path(root).resolve(strict=True), Path(path).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("ANIE input must stay within its explicit data root")
        self.handle = None
        self.cache = bool(cache)
        provenance = {"profile": "observations", "full_input_checks_performed": bool(verify_full)}
        self.binding = dict(file=path.relative_to(root).as_posix(),
                            bytes=path.stat().st_size,
                            file_sha256=None,
                            sha256_verified=False,
                            cohort="four_vortex", fixture_only=not formal,
                            time_step=DT, provenance=provenance)
        scientific = hashlib.sha256()
        handle = h5py.File(path, "r")
        try:
            fixture = bool(handle.attrs.get("fixture_only", False))
            self.binding["fixture_only"] = fixture
            for group in ("training", "validation"):
                field = handle[f"{group}/vorticity"]
                ids = np.asarray(handle[f"{group}/trajectory_index"][:], dtype=np.int64)
                times = np.asarray(handle[f"{group}/time"][:], dtype=np.float64)
                if (field.ndim != 4 or field.shape[-2:] != (64, 64)
                        or field.dtype != np.dtype("float32") or len(ids) != field.shape[0]
                        or len(set(map(int, ids))) != len(ids)
                        or len(times) != field.shape[1] or len(times) < WINDOW_FRAMES
                        or not np.allclose(times, np.arange(len(times)) * DT, rtol=0, atol=2e-12)):
                    raise ValueError(f"invalid ANIE {group} observations")
                if group == "training" and not 1 <= trajectories <= len(ids):
                    raise ValueError("selected ANIE trajectory count exceeds training population")
                scientific.update(group.encode("ascii"))
                scientific.update(ids.astype("<i8").tobytes())
                scientific.update(times.astype("<f8").tobytes())
                selected = len(ids) if formal or group == "validation" else trajectories
                if cache:
                    value = np.empty((selected, *field.shape[1:]), dtype=np.float32)
                else:
                    value = field
                # Hash arrays during the requested finite scan or cache fill.
                if verify_full or cache:
                    for first in range(0, len(ids), 4):
                        block = np.ascontiguousarray(field[first:first + 4])
                        if not np.isfinite(block).all():
                            raise ValueError(f"nonfinite ANIE {group} observations")
                        scientific.update(block.tobytes())
                        if cache and first < selected:
                            end = min(first + len(block), selected)
                            value[first:end] = block[:end-first]
                self.binding[group + "_ids"] = ids.tolist()
                self.binding[group + "_shape"] = list(field.shape)
                setattr(self, group, value)
            if set(self.binding["training_ids"]) & set(self.binding["validation_ids"]):
                raise ValueError("ANIE training and validation identities overlap")
            self.binding["source_fixture_flag"] = fixture
            self.binding["observations_sha256"] = scientific.hexdigest() if verify_full or cache else None
            if cache:
                handle.close()
            else:
                self.handle = handle
        except BaseException:
            handle.close()
            raise

    def close(self):
        """Close the lazy HDF5 handle, if present; repeated calls are harmless."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        """Return this bank for a with block whose exit releases any HDF5 handle."""
        return self

    def __exit__(self, *args):
        """Release the HDF5 handle when a with block ends, including on an exception."""
        self.close()

    def batch(self, rows, anchors, device, *, validation=False, indices=None):
        """Return initial fields [B,64,64] and observed targets [B,T,64,64].

        rows selects trajectory rows; anchors is indexed by the original row,
        not by batch position. indices gives increasing offsets in a 151-frame
        window, including offsets 0 and 150. Read only the selected observations
        and place contiguous float32 tensors on device.
        """
        field = self.validation if validation else self.training
        rows = np.asarray(rows, dtype=np.int64)
        anchors = np.asarray(anchors, dtype=np.int64)
        indices = np.arange(WINDOW_FRAMES) if indices is None else np.asarray(indices, dtype=np.int64)
        if (indices.ndim != 1 or len(indices) < 2 or indices[0] != 0
                or indices[-1] != WINDOW_FRAMES - 1 or np.any(np.diff(indices) <= 0)):
            raise ValueError("ANIE observed indices must increase from the initial to the final window frame")
        values = []
        for row in rows:
            anchor = int(anchors[int(row)])
            if not 0 <= int(row) < len(field) or not 0 <= anchor <= field.shape[1] - WINDOW_FRAMES:
                raise ValueError("ANIE observed window lies outside its trajectory")
            values.append(np.asarray(field[int(row), anchor + indices], dtype=np.float32))
        value = np.ascontiguousarray(np.stack(values))
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite observed ANIE window")
        target = torch.from_numpy(value).to(device)
        return target[:, 0], target

    def validation_indices(self):
        """Return all validation rows and one fixed 3-second-window anchor per row.

        Use frame 250 when the trajectory is long enough, otherwise frame 0.
        Keeping this window fixed makes epoch measurements directly comparable.
        """
        count, frames = self.validation.shape[:2]
        anchor = VALIDATION_ANCHOR if frames >= VALIDATION_ANCHOR + WINDOW_FRAMES else 0
        return np.arange(count, dtype=np.int64), np.full(count, anchor, dtype=np.int64)
