"""Observed PINN inputs share one reader, regardless of package provenance."""
from pathlib import Path


def verify_regenerated_inputs(sampling, data_path, data_root, condition):
    """Read PINN observations and return their binding after checking data-root containment.

    The shared adapter validates coordinate, target and physics arrays;
    condition selects the noise cohort. This helper returns metadata only.
    """
    from adapters.pinn_runtime import read_inputs
    base = Path(data_root).resolve(strict=True)
    for path in (sampling, data_path):
        if base not in Path(path).resolve(strict=True).parents:
            raise ValueError("PINN input must stay within its selected data root")
    return read_inputs(sampling, data_path, condition=condition)[-1]


def read_regenerated_inputs(sampling, data_path, data_root, condition, train_rows=None, physics_rows=None):
    """Return selected PINN arrays with regenerated-profile provenance.

    train_rows and physics_rows optionally restrict diagnostic populations.
    Array layout and bounds come from the shared PINN reader; the last tuple
    item is its binding augmented with the regenerated profile.
    """
    from adapters.pinn_runtime import read_inputs
    base = Path(data_root).resolve(strict=True)
    for path in (sampling, data_path):
        if base not in Path(path).resolve(strict=True).parents:
            raise ValueError("PINN input must stay within its selected data root")
    selected = read_inputs(sampling, data_path, train_rows, physics_rows, condition=condition)
    bound = dict(selected[-1], provenance={"profile": "regenerated"})
    return (*selected[:-1], bound)
