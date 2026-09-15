"""Training-only layouts; preserve ordinary parameters and native autograd.

One derivative batch, or one unchanged truncated-backpropagation segment, owns
its packed spectral-weight views. Rebuild them before the next backward segment
and after every optimizer update. Never detach, serialize or cache learned
weights across segments. Prediction calls do not use this helper.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class BranchTrainingView:
    """Delegate model state and projections, specializing only training work."""

    def __init__(self, model):
        self.model = model
        self.packed = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def begin_segment(self):
        if self.packed is not None:
            raise RuntimeError("previous training segment was not released")
        # Make channel contractions contiguous once, not at every RK4 stage.
        # These remain differentiable views/copies of the original Parameters.
        self.packed = {
            id(block.spectral): tuple(
                weight.permute(2, 3, 0, 1).contiguous().permute(2, 3, 0, 1)
                for weight in (block.spectral.weight_positive, block.spectral.weight_negative)
            )
            for block in self.model.blocks
        }

    def end_segment(self):
        self.packed = None

    def block(self, block, field):
        if self.packed is None:
            raise RuntimeError("begin a training segment before spectral execution")
        spectral = block.spectral
        ny, nx = field.shape[-2:]
        modes = spectral.modes
        if 2 * modes > ny or modes > nx // 2 + 1:
            raise ValueError("configured spectral modes exceed the incoming grid")
        spectrum = torch.fft.rfft2(field)
        positive_weight, negative_weight = self.packed[id(spectral)]
        positive = spectral.multiply(spectrum[:, :, :modes, :modes], positive_weight)
        negative = spectral.multiply(spectrum[:, :, -modes:, :modes], negative_weight)
        # Disjoint positive/middle/negative supports have identical values to
        # two slice assignments, without full-sized CopySlices gradient copies.
        middle = positive.new_zeros(field.shape[0], spectral.channels, ny - 2 * modes, modes)
        output_hat = F.pad(torch.cat((positive, middle, negative), dim=-2),
                           (0, nx // 2 + 1 - modes))
        value = torch.fft.irfft2(output_hat, s=(ny, nx)) + block.pointwise(field)
        return F.gelu(value) if block.activate else value

    def forward_with_generator_output(self, state, low_rhs):
        if self.packed is None:
            raise RuntimeError("begin a training segment before branch execution")
        return self.model.forward_with_generator_output(state, low_rhs, _training_kernels=self)
