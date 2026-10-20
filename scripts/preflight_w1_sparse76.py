"""Disposable, training-only full76/tail12 costs for the N300 comparison."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from adapters.models import fno_utilities
from scripts.run_w1_v2 import job_configuration
from training.w1_v2_data import WindowBank, epoch_order
from training.w1_v2_train import atomic_json, build_network, forward_model, native_loss, seed_initialization


def measure(bank, plan, method, rows, *, horizon=75):
    """Time native gradient groups; lr0 prevents using probes as training starts."""
    seed_initialization(plan["development_seed"])
    job = job_configuration(plan, method, "development", plan["development_seed"], plan["candidates"][method][0])
    spec = job["model_spec"]
    model = build_network(spec, "cuda").train()
    normalization = bank.normalization(method)
    criterion = fno_utilities(spec["source_root"])[1](size_average=False) if method.startswith("fno") else None
    optimizer = torch.optim.Adam(model.parameters(), lr=0., weight_decay=.0001)
    micro = plan["physical_micro_batches"][method]
    order = epoch_order(len(bank.training), 0, plan["sample_seed"])
    group = order[:rows] if rows == plan["effective_batch"] else order[-rows:]
    costs, flags = [], []
    for _ in range(4):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_finite = True
        for offset in range(0, rows, micro):
            positions = group[offset:min(offset + micro, rows)]
            initial, truth = bank.batch(positions, "cuda")
            prediction = forward_model(model, initial, spec, normalization, training=True, horizon=horizon)
            expected = horizon + 1 if method == "fno2d" else plan["observation_points"]
            if prediction.shape != (len(positions), expected, 64, 64):
                raise ValueError("probe output does not match the declared actual computational grid")
            loss = native_loss(prediction, truth, method, criterion)
            loss_finite = loss_finite and bool(torch.isfinite(loss))
            (loss * (len(positions) / rows)).backward()
        gradients_finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters() if parameter.grad is not None)
        if loss_finite and gradients_finite:
            optimizer.step()
        state_finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()) and all(
            bool(torch.isfinite(value).all()) for slots in optimizer.state.values()
            for value in slots.values() if isinstance(value, torch.Tensor))
        torch.cuda.synchronize()
        costs.append(time.perf_counter() - begin)
        flags.append(dict(loss_finite=loss_finite, gradients_finite=gradients_finite, state_finite=state_finite))
        if not all(flags[-1].values()):
            break
    result = dict(method=method, rows=rows, physical_micro_batch=micro, future_steps=horizon,
                  training_ids=bank.training_ids[group].tolist(),
                  warmup_seconds=costs[0], measured_seconds=costs[1:],
                  median_seconds=statistics.median(costs[1:]) if len(costs) > 1 else None,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  finite_checks=flags, all_finite=all(all(flag.values()) for flag in flags),
                  source_root=spec["source_root"], query_times=spec["query_times"],
                  last_solver_iterations=model.last_solver_iterations if method.startswith("anie") else None)
    del optimizer, criterion, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    """Save all probes; random full-horizon failure cannot be called successful."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "RUN_PLAN.json").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    bank = WindowBank(plan["window"])
    if bank.training.shape != (300, 76, 64, 64) or bank.validation.shape != (20, 76, 64, 64):
        raise ValueError("preflight requires the legitimate N30076 task")
    output = args.root / "PREFLIGHT_NEURAL.json"
    if output.exists():
        raise FileExistsError("preserve the previous probe; use a new attempt filename rather than overwriting")
    report = dict(schema="gift.w1-sparse76.neural-cost-preflight.v1", device=torch.cuda.get_device_name(),
                  purpose="disposable_cost_and_gradient_feasibility", adam_lr=0.,
                  validation_loaded_for_receipt_integrity=True, validation_used_for_selection=False,
                  test_read=False, formal_initial_weights_saved=False, data_binding=bank.binding, attempts=[])
    for method in ("anie_ns", "anie_f", "fno2d", "fno3d"):
        for rows in (32, 12):
            result = measure(bank, plan, method, rows)
            report["attempts"].append(result)
            atomic_json(output, report)
            print(json.dumps(result), flush=True)
        method_rows = [row for row in report["attempts"] if row["method"] == method]
        if not all(row["all_finite"] for row in method_rows):
            if method != "fno2d":
                raise FloatingPointError(f"{method}: full76 gradient/Adam feasibility failed")
            first_stage = measure(bank, plan, method, 32, horizon=1)
            report["attempts"].append(first_stage)
            atomic_json(output, report)
            if not first_stage["all_finite"]:
                raise FloatingPointError("FNO2D declared initial curriculum stage failed")
            report["fno2d_requires_predeclared_curriculum"] = True
    report["status"] = "complete"
    report["epoch_seconds_estimate"] = {}
    for method in ("anie_ns", "anie_f", "fno2d", "fno3d"):
        rows = [row for row in report["attempts"] if row["method"] == method and row["future_steps"] == 75]
        if all(row["all_finite"] and row["median_seconds"] is not None for row in rows):
            report["epoch_seconds_estimate"][method] = 9 * rows[0]["median_seconds"] + rows[1]["median_seconds"]
    atomic_json(output, report)
    print(json.dumps({"status": "complete", "epoch_seconds_estimate": report["epoch_seconds_estimate"]}), flush=True)


if __name__ == "__main__":
    main()
