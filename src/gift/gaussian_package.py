"""Read Gaussian prediction inputs independently of generation journals."""
from pathlib import Path

import h5py
import numpy as np

from .prediction_cohorts import GAUSSIAN, partitions

DIRECTORY = "s3_gaussian"
DENSE_FILE = DIRECTORY + "/trajectories.h5"
LITE_FILE = DIRECTORY + "/lite.h5"


def group_ids(*, lite=False):
    """Return canonical IDs for Gaussian HDF groups, including the lite subset.

    Lite files contain 50 training and 20 checkpoint-validation trajectories.
    Full files also carry the remaining validation population and fixed test
    IDs; those extra trajectories are not used by checkpoint validation.
    """
    split = partitions(GAUSSIAN)
    groups = dict(training=split["gift_lite_training"] if lite else split["training"],
                  validation=split["checkpoint_validation"])
    if not lite:
        groups.update(validation_aux=split["validation"][20:], test=split["test"])
    return groups


def validate_input(base, path, *, full=False):
    """Check a full/lite Gaussian observation file and describe its selected IDs.

    Vorticity is float32 ``[trajectories, 501, 64, 64]`` with ``0.02`` time
    spacing over ``0..10``. Group names and canonical row identities must
    match this cohort. ``full=True`` additionally streams training and
    checkpoint-validation rows for finite values. ``base`` is an interface
    argument, not a restriction on the explicit path or a package approval.
    """
    path = Path(path).resolve(strict=True)
    with h5py.File(path, "r") as handle:
        if (handle.attrs.get("prediction_cohort") != GAUSSIAN
                or handle.attrs.get("trajectory_id_scheme") != "canonical"):
            raise ValueError("Gaussian cohort and canonical trajectory identities required")
        fixture_only = bool(handle.attrs.get("fixture_only", False))
        lite = "test" not in handle
        expected = group_ids(lite=lite)
        if set(handle) != set(expected):
            raise ValueError("Gaussian observation groups differ")
        for name, ids in expected.items():
            field = handle[name + "/vorticity"]
            if (handle[name + "/trajectory_index"][:].tolist() != ids
                    or field.shape != (len(ids), 501, 64, 64)
                    or field.dtype != np.dtype("float32")
                    or not np.allclose(handle[name + "/time"][:], np.arange(501) * .02, rtol=0, atol=2e-12)):
                raise ValueError("Gaussian identities, shape, dtype or times differ")
            if full and name in ("training", "validation"):
                for row in range(len(ids)):
                    if not np.isfinite(field[row]).all():
                        raise ValueError("Nonfinite Gaussian observations")
    return dict(profile="gaussian", cohort=GAUSSIAN, file=str(path), fixture_only=fixture_only, sha256=None,
                bytes=path.stat().st_size, trajectory_ids=expected["training"],
                validation_ids=expected["validation"], full_input_checks_performed=full,
                published_dataset_identity=False, numerical_reproduction_verified=False)
