"""GIFT public numerical API, loaded only when a numerical symbol is requested.

Metadata validation is also used in standalone audit and external PINN
environments. Importing that functionality must not initialize Torch.
"""

from importlib import import_module

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

_MODULES = {
    "identified": ("FixedBandwidthGridGenerator", "load_generator", "load_release_payload", "rollout_rk4"),
    "identifiability": ("AffineMinimumNormFit", "default_rank_rcond", "fit_affine_minimum_norm", "identifiability_grid"),
    "model": ("FreeHermitianTable", "GIFTGenerator", "WaveGrid", "make_wave_grid"),
}


def __getattr__(name):
    """Resolve the established public symbols from their numerical modules."""
    for module_name, names in _MODULES.items():
        if name in names:
            value = getattr(import_module("." + module_name, __name__), name)
            globals()[name] = value
            return value
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
