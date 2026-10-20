"""Relative equation-action error used by independent test evaluation."""

import numpy as np
import torch


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> np.ndarray:
    """Return one float64 spatial error ratio per [batch,N,N] reference action."""
    top = numerator.flatten(1).double().norm(dim=1)
    bottom = denominator.flatten(1).double().norm(dim=1).clamp_min(1.0e-30)
    return (top / bottom).detach().cpu().numpy()
