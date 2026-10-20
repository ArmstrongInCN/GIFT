"""Measure full151 native forward/backward cost before development selection.

Measurements use training trajectories only, without performance selection.
The chosen physical microbatch is fixed across each method's LR candidates
and seeds. It also sets FNO2D's native BatchNorm batch, which accumulation
does not turn into the effective update batch.
All models are fresh random initialization, and no resulting weight is used
as a development/formal starting point.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from adapters.models import fno_utilities
from training.w1_v2_data import WindowBank
from training.w1_v2_train import atomic_json, build_network, forward_model, native_loss, seed_initialization


def measure(bank, spec, micro):
    """Two synchronized native gradient calls; the first includes kernel warmup."""
    seed_initialization(7)
    model = build_network(spec, "cuda")
    norm = bank.normalization(spec["method"])
    criterion = fno_utilities(spec["source_root"])[1](size_average=False) if spec["method"].startswith("fno") else None
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=.0001)
    initial, truth = bank.batch(list(range(micro)), "cuda")
    torch.cuda.reset_peak_memory_stats()
    costs = []
    for _ in range(2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_model(model, initial, spec, norm, training=True)
        loss = native_loss(prediction, truth, spec["method"], criterion)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("fresh preflight objective is nonfinite")
        loss.backward()
        gradients_finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if gradients_finite:
            optimizer.step()
        torch.cuda.synchronize()
        costs.append(time.perf_counter() - start)
    return dict(micro_batch=micro, seconds=costs, gradients_finite=gradients_finite, seconds_per_trajectory=costs[-1] / micro,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), parameters=sum(p.numel() for p in model.parameters()),
                forecast_epoch_seconds=1000 * costs[-1] / micro, full_time_points=151)


def main():
    """Save all feasibility attempts, retaining failures rather than hiding them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anie-source", type=Path, required=True)
    parser.add_argument("--fno-source", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("anie_ns", "anie_f", "fno2d", "fno3d"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = WindowBank(args.window)
    report = dict(device=torch.cuda.get_device_name(), attempts=[], selected_micro_batches={},
                  selection_reason="memory/cost only; no validation/test used", full151=True)
    for method, candidates in (("anie_ns", [6]), ("anie_f", [6]), ("fno2d", [16]), ("fno3d", [8])):
        if args.methods and method not in args.methods:
            continue
        spec = dict(method=method, history=1, source_root=str(args.anie_source if method.startswith("anie") else args.fno_source))
        if method.startswith("anie"):
            spec["condition_mode"] = "native" if method == "anie_ns" else "fixed_f"
        if method == "fno2d":
            spec["activation_checkpoint"] = False
        completed = False
        for micro in candidates + [1]:
            try:
                result = measure(bank, spec, micro)
                report["attempts"].append(dict(method=method, model_spec=spec,
                    status="successful" if result["gradients_finite"] else "gradient_instability",
                    memory_feasible=True, **result))
                report["selected_micro_batches"][method] = micro
                completed = True
            except torch.cuda.OutOfMemoryError as error:
                report["attempts"].append(dict(method=method, micro_batch=micro, status="out_of_memory", reason=str(error)))
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(args.output, report)
            print(json.dumps(report["attempts"][-1]), flush=True)
            if completed:
                break
        if not completed:
            raise RuntimeError(f"{method}: even microbatch1 is not feasible")
    report["upper_bound_gpu_hours_4_models"] = sum(
        row["forecast_epoch_seconds"] * (400 + 3 * 1500) / 3600
        for row in report["attempts"] if row.get("memory_feasible"))
    report["minimum_no_early_stop_gpu_hours_4_models"] = sum(
        row["forecast_epoch_seconds"] * (400 + 3 * 500) / 3600
        for row in report["attempts"] if row.get("memory_feasible"))
    report["timing_caveat"] = "extrapolation excludes full validation, setup, checkpoint IO and GIFT; actual cost can differ"
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
