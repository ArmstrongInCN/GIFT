"""Benchmark GIFT execution on observed training windows, without formal outputs.

Writes a new JSON receipt only. Disposable optimizer updates never replace model
files or count toward a scientific training budget. CUDA measurements include
input copies, forward/backward, clipping and AdamW; graph setup is separate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch

from gift import load_generator, FixedBandwidthGridGenerator
from training.gift_prediction_data import ObservationBank
from training.train_gift_predictor import derivative_loader, derivative_epoch
from training.gift_acceleration import TrainingEngine
from training.checkpoints import digest_file
from experiments.formal.train_gift_branches import _short_epoch, _long_epoch
from experiments.formal._shared.high_frequency import HighFrequencyBranch, HighFrequencyConfig, load_high_frequency_model
from experiments.formal._shared.gift_generator_training import configure_determinism, configure_variable_projection_parameters


def main(argv=None):
    # Paired eager-vs-graph timing on observed training windows. Every optimizer
    # update is disposable (AdamW with a throwaway schedule); nothing is saved or
    # counted toward a scientific training budget, and it requires CUDA.
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--generator', type=Path, required=True)
    p.add_argument('--branch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--kinds', nargs='+', choices=('generator', 'derivative', 'short', 'long'),
                   default=('generator', 'derivative', 'short', 'long'))
    args = p.parse_args(argv)
    if args.output.exists(): raise FileExistsError(args.output)
    if not 1 <= args.rows <= 1000 or args.repeats < 1 or args.threads < 1:
        raise ValueError('invalid benchmark size')
    if not torch.cuda.is_available(): raise RuntimeError('this paired performance benchmark requires CUDA')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    device = torch.device('cuda')
    configure_determinism(20260820, strict=False)
    bank = ObservationBank(args.dataset)
    generator = load_generator(args.generator, device=device).requires_grad_(False)
    frozen = FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    branch_payload = torch.load(args.branch, map_location='cpu', weights_only=True)
    if 'payload' in branch_payload and 'identity' in branch_payload:
        # An explicit training checkpoint is accepted for disposable timing only.
        scales = branch_payload['identity']['scales']
        base = HighFrequencyBranch(HighFrequencyConfig(), state_scale=scales[0], high_scale=scales[1],
                                   low_rhs_scale=scales[2], high_rhs_scale=scales[3], frozen_generator=frozen).to(device)
        base.load_state_dict(branch_payload['payload']['model'])
    else:
        base, _ = load_high_frequency_model(args.branch, device, frozen)
    base.requires_grad_(True)
    _, derivative_data = derivative_loader(bank, 20260820, 461, frozen, device)
    state, target = bank.pairs(2026072301, 1)
    sequence = torch.from_numpy(bank.sequences(20260820, 461))
    receipt = {'scope': 'disposable timing, not formal training or accuracy evidence',
               'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'threads': args.threads,
               'data': bank.binding, 'generator_sha256': digest_file(args.generator),
               'branch_sha256': digest_file(args.branch), 'rows': args.rows, 'repeats': args.repeats,
               'timing_scope': 'H2D input copies, forward/backward, clipping, AdamW and scalar loss reporting; excludes setup/data preparation/validation/affine fit',
               'stages': {}}
    for kind in args.kinds:
        configure_determinism(2026072301 if kind == 'generator' else 20260820, strict=kind == 'generator')
        if kind == 'generator':
            models = [load_generator(args.generator, device=device) for _ in range(2)]
            params = [configure_variable_projection_parameters(m) for m in models]
            inputs, size, scale, lr, wd = (state, target), 16, float(target.square().mean()), .003, 1e-8
        else:
            models = [copy.deepcopy(base) for _ in range(2)]
            params = [list(m.parameters()) for m in models]
            inputs, size = (derivative_data, 16) if kind == 'derivative' else ((sequence,), 5)
            scale = float(base.high_rhs_scale)
            lr, wd = (.0015 if kind == 'derivative' else .0002 if kind == 'short' else .00008), 1e-6
        initial = copy.deepcopy(models[0].state_dict())
        batches = [tuple(x[i:min(i+size, args.rows)] for x in inputs) for i in range(0, args.rows, size)]
        engine = TrainingEngine(models[1], kind=kind, frozen=frozen, scale=scale)
        for batch in batches:
            key = tuple((x.shape, x.dtype, device) for x in batch)
            if key not in engine.graphs: engine.gradients(*(x.to(device) for x in batch))
        def run(index, optimizer):
            model = models[index]
            if index == 1: return engine.epoch(batches, optimizer, device)
            if kind == 'derivative': return derivative_epoch(model, batches, optimizer, device, scale)
            if kind in ('short', 'long'):
                return (_short_epoch if kind == 'short' else _long_epoch)(model, frozen, batches, optimizer, device)
            total = 0.
            for s, t in batches:
                s, t = s.to(device), t.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(s) - t).square().mean() / scale
                if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite generator loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params[index], 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(s)
            return total / args.rows
        # Warm both paths without counting the discarded updates.
        for i in (0, 1): run(i, torch.optim.AdamW(params[i], lr=lr, weight_decay=wd))
        timings = [[], []]
        for repeat in range(args.repeats):
            losses = [None, None]
            for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                models[i].load_state_dict(initial)
                optimizer = torch.optim.AdamW(params[i], lr=lr, weight_decay=wd)
                torch.cuda.synchronize(); started = time.perf_counter()
                losses[i] = run(i, optimizer)
                torch.cuda.synchronize(); timings[i].append(time.perf_counter() - started)
            # The eager and graph paths must land on bitwise-identical state and
            # loss, otherwise the reported speedup would compare two different runs.
            if losses[0] != losses[1] or any(not torch.equal(v, models[1].state_dict()[k]) for k, v in models[0].state_dict().items()):
                raise AssertionError(f'{kind}: final state/loss differs; do not claim equivalent speed')
        result = {'eager_seconds': timings[0], 'graph_seconds': timings[1],
                  'median_speedup': statistics.median(timings[0]) / statistics.median(timings[1]),
                  'setup_seconds': sum(g.setup_seconds for g in engine.graphs.values()),
                  'updates_per_repeat': len(batches), 'final_model_and_loss_bitwise_equal': True}
        receipt['stages'][kind] = result
        print(json.dumps({kind: result}), flush=True)
        del engine, models, params
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__': main()
