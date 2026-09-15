"""Opt-in fixed-shape CUDA execution; model mathematics and optimizers stay eager.

Graph objects are ephemeral. Checkpoints contain ordinary model/optimizer/RNG
state, and graphs are rebuilt after loading. This helper is for deterministic
GIFT tensor functions, not modules with random draws or mutable forward buffers.
"""
from __future__ import annotations

import time
import torch


class StaticTensorCall:
    """Capture one shape and retain its input, output and gradient storage.

    Returned tensors are borrowed: the next call overwrites them. Optimizer
    steps, clipping, scheduler steps and safety decisions remain outside capture.
    Parameter storage/device/dtype must not be replaced while a call is alive.
    """

    def __init__(self, function, inputs, *, parameters=(), backward=False, frozen=False):
        if not inputs or any(x.device.type != "cuda" for x in inputs):
            raise ValueError("CUDA graph inputs must be nonempty CUDA tensors")
        if any(x.device != inputs[0].device for x in inputs):
            raise ValueError("CUDA graph inputs must share one device")
        self.inputs = tuple(x.detach().clone() for x in inputs)
        self.parameters = tuple(parameters)
        self.backward = backward
        self.storage = tuple((p.data_ptr(), p.shape, p.dtype, p.device) for p in self.parameters)
        self.versions = tuple(p._version for p in self.parameters) if frozen else None
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(inputs[0].device)
        started = time.perf_counter()
        try:
            stream = torch.cuda.Stream(device=inputs[0].device)
            stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    if backward:
                        for parameter in self.parameters:
                            parameter.grad = None
                    function(*self.inputs)
            torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
            torch.cuda.synchronize(inputs[0].device)
            if backward:
                for parameter in self.parameters:
                    parameter.grad = None
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = function(*self.inputs)
            torch.cuda.synchronize(inputs[0].device)
            self.gradients = tuple(p.grad for p in self.parameters) if backward else ()
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, inputs[0].device)
        self.setup_seconds = time.perf_counter() - started

    def __call__(self, *inputs):
        if len(inputs) != len(self.inputs):
            raise ValueError("graph input count differs")
        for actual, static in zip(inputs, self.inputs):
            if actual.shape != static.shape or actual.dtype != static.dtype or actual.device != static.device:
                raise ValueError("graph input shape, dtype or device differs")
        if any((p.data_ptr(), p.shape, p.dtype, p.device) != identity
               for p, identity in zip(self.parameters, self.storage)):
            raise RuntimeError("parameter storage changed; rebuild the CUDA graph")
        if self.versions is not None and tuple(p._version for p in self.parameters) != self.versions:
            raise RuntimeError("frozen parameters changed; rebuild the inference adapter and CUDA graph")
        # Another batch shape may own a different graph/gradient buffer set.
        # Rebind those buffers; replay fills them, rather than accumulating the
        # previous batch's gradients. Never zero_grad(set_to_none) after replay.
        if self.backward:
            for parameter, gradient in zip(self.parameters, self.gradients):
                parameter.grad = gradient
        for actual, static in zip(inputs, self.inputs):
            static.copy_(actual)
        self.graph.replay()
        return self.outputs
