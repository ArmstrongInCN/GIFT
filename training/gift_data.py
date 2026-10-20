"""Observation schemas and generator structure for independent GIFT training."""
from __future__ import annotations

from pathlib import Path

import h5py
from gift.paths import data_path

# Historical input files keyed by noise condition. Current W3 uses an explicit
# single-case root; the formal package's clean default is the prediction cohort.
DATASETS = {
    "noise_000": "standard_ns_n64_full_spectrum.h5",
    "noise_001": "w3_parameter_identification/noise_001.h5",
    "noise_005": "w3_parameter_identification/noise_005.h5",
}
# Published byte identities retained for historical metadata consumers.
RELEASED_SHA = {
    "noise_000": "c3f21ef13d0c81713f2770fc67547c91bd5a793fa34a9eecc5c46537237fa8a5",
    "noise_001": "97c5773290b5c50393ff64b7b42c021fd54f2bb429fc6d57cd4599f09f9818bc",
}
PROFILE_SOURCES = (
    "training/gift_data.py", "scripts/assemble_generated_data.py",
    "scripts/generate_data.py", "scripts/generate_noise.py", "src/even_full_spectrum_ns.py",
    "src/gift/prediction_cohorts.py", "src/gift/gaussian_package.py",
    "src/gift/gaussian_initial.py", "src/gift/canonical_package.py",
)


def input_file(base: Path, condition: str, profile: str) -> Path:
    """Select the observation file for an W3 noise condition or the separate S3 Lite profile.

    Gaussian Lite is a clean prediction cohort and accepts only noise_000.
    A formal catalogue selects its clean prediction role. Historical noisy
    profiles require their independently generated, catalogue-free input root.
    """
    # S3 Gaussian lite input is keyed separately and must not be confused with
    # an W3 noise condition.
    if profile == 'gaussian':
        if condition != 'noise_000':
            raise ValueError('S3 contains only clean Gaussian observations, not an W3 noise experiment')
        return data_path(base, 's3.lite', 's3_gaussian/lite.h5')
    role = ('four_vortex.standard_n64' if condition == 'noise_000'
            else 'historical_identification.' + condition)
    return data_path(base, role, DATASETS[condition])


def validate_input(base: Path, condition: str, profile: str, *, full: bool,
                   include_validation: bool = True) -> dict:
    """Inspect train/validation observation schemas and optionally scan finite values.

    Fields are float32 [trajectory,frame,y,x] on a square grid with DT=0.02
    seconds. IDs must be unique and disjoint across splits; at least five
    frames support the observation derivative stencil. Return descriptive
    input metadata without certifying released file bytes. Explicit
    include_validation=False checks only training observations and never opens
    an unused validation group.
    """
    import numpy as np

    base = Path(base).resolve(strict=True)
    path = input_file(base, condition, profile).resolve(strict=True)
    if base not in path.parents:
        raise ValueError("Training input must stay within its selected data root")
    with h5py.File(path, "r") as handle:
        identities = []
        for name in (("training", "validation") if include_validation else ("training",)):
            field = handle[f"{name}/vorticity"]
            ids = np.asarray(handle[f"{name}/trajectory_index"][:])
            times = np.asarray(handle[f"{name}/time"][:])
            if (field.ndim != 4 or field.shape[-2] != field.shape[-1]
                    or field.dtype != np.dtype("float32") or len(ids) != len(field)
                    or len(set(map(int, ids))) != len(ids) or len(times) != field.shape[1]
                    or len(times) < 5 or not np.allclose(times, np.arange(len(times)) * .02, rtol=0, atol=2e-12)):
                raise ValueError(f"Invalid {name} observations")
            identities.append(set(map(int, ids)))
            if full:
                for first in range(0, len(ids), 4):
                    if not np.isfinite(field[first:first + 4]).all():
                        raise ValueError(f"Nonfinite {name} observations")
        if include_validation and identities[0] & identities[1]:
            raise ValueError("Training and validation identities overlap")
    return dict(profile=profile, condition=condition, file=path.relative_to(base).as_posix(),
                sha256=None, bytes=path.stat().st_size, full_input_checks_performed=full,
                published_dataset_identity=False, numerical_reproduction_verified=False)


def validate_low_prerequisite(payload: dict, training_data: dict, formal_configuration: dict) -> None:
    """Check generator format, stored weights and configured cutoff/rank before branch use.

    training_data remains a compatible API argument for older callers.
    Its historical file identity or training budget is not an inference
    prerequisite; the branch checks actual loaded inputs when resuming.
    """
    if payload.get("format_version") not in (2, 3) or not payload.get("model_state_dict"):
        raise ValueError("Incomplete generator weights")
    configuration = payload.get("model_configuration", {})
    for key in ("cutoff", "rank"):
        if key in formal_configuration and configuration.get(key) != formal_configuration[key]:
            raise ValueError("Generator model configuration differs")
