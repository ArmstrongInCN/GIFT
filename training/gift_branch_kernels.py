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
        """Wrap a branch without copying its Parameters or serializable state.

        Packed spectral views start empty and live only for one backward segment.
        """
        self.model = model
        self.packed = None

    def __getattr__(self, name):
        """Forward ordinary branch attributes and methods to the wrapped model."""
        return getattr(self.model, name)

    def begin_segment(self):
        """Build differentiable spectral layouts for the next backward segment.

        Permuting channel axes makes the contraction layout contiguous while
        keeping autograd connected to the original Parameters. An active segment
        must be released before constructing another one.
        """
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
        """Drop packed views after backward so the next segment uses fresh weights."""
        self.packed = None

    def block(self, block, field):
        """Apply a spectral-plus-pointwise block to [B,C,ny,nx] real fields.

        Assemble positive and negative rfft2 mode supports with a zero middle
        band, then invert at the original grid size. The assembly preserves
        gradients without creating full-grid slice-assignment copies.
        """
        if self.packed is None:
            raise RuntimeError("begin a training segment before spectral execution")
        spectral = block.spectral
        ny, nx = field.shape[-2:]
        modes = spectral.modes
        # Guard the spectral block against a grid smaller than twice the modes;
        # rfft2 leaves only nx//2+1 positive frequency columns per row.
        if 2 * modes > ny or modes > nx // 2 + 1:
            raise ValueError("configured spectral modes exceed the incoming grid")
        # Real-input FFT yields a Hermitian spectrum; the branch reads the lower
        # and upper mode blocks separately and stitches them back below.
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
        """Evaluate the wrapped branch with this segment's packed spectral kernels.

        state is the observed field and low_rhs is the frozen generator output;
        the wrapped branch retains projection, scaling and output semantics.
        """
        if self.packed is None:
            raise RuntimeError("begin a training segment before branch execution")
        return self.model.forward_with_generator_output(state, low_rhs, _training_kernels=self)
