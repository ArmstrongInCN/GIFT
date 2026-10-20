"""Recompute native FNO calls without applying BatchNorm updates twice.

Training-mode BatchNorm uses the current batch's statistics for its output,
but mutates running buffers used by evaluation. Checkpoint backward replays
that forward. Temporary buffer clones isolate replay mutations while keeping
the original normalization, gradients and one update per actual forward.
"""
from contextlib import contextmanager, nullcontext

import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def _recompute_batchnorm_buffers(model):
    """Give replay private running buffers, restoring the original references.

    Swapping references avoids an in-place restoration on tensors saved for
    autograd. Clone on context entry, after all original forwards have run;
    native training BatchNorm outputs do not depend on these running values.
    No module's training flag, momentum, affine weights or forward is changed.
    """
    originals = []
    try:
        for module in model.modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, name)
                if value is not None:
                    originals.append((module, name, value))
                    setattr(module, name, value.detach().clone())
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def checkpoint_native_model(model, inputs):
    """Checkpoint one native model call with RNG replay and isolated BN buffers.

    Use this for training-mode recursive FNO2D calls. Each original forward
    retains its ordinary running-statistics update; backward recomputation
    preserves those buffers for subsequent evaluation and checkpoints.
    """
    return checkpoint(model, inputs, use_reentrant=False, preserve_rng_state=True,
        context_fn=lambda: (nullcontext(), _recompute_batchnorm_buffers(model)))
