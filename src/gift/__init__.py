"""GIFT: field-tomographic identification with rank-aware affine gauges."""

from .identified import (
    FixedBandwidthGridGenerator,
    load_generator,
    load_release_payload,
    rollout_rk4,
)
from .identifiability import (
    AffineMinimumNormFit,
    default_rank_rcond,
    fit_affine_minimum_norm,
    identifiability_grid,
)
from .model import FreeHermitianTable, GIFTGenerator, WaveGrid, make_wave_grid

__all__ = [
    "AffineMinimumNormFit",
    "FreeHermitianTable",
    "FixedBandwidthGridGenerator",
    "GIFTGenerator",
    "WaveGrid",
    "default_rank_rcond",
    "fit_affine_minimum_norm",
    "identifiability_grid",
    "load_generator",
    "load_release_payload",
    "make_wave_grid",
    "rollout_rk4",
]

__version__ = "1.0.0"
