"""Fresh fixed-origin ANIE/FNO training for the declared W1-v2 comparison.

The common controller handles exposure, validation and continuation. Native
model classes, Adam and relative-L2 definitions remain upstream components.
Every update averages across its actual trajectory count; the native FNO2D
sum across the declared forecast losses remains a sum. FNO2D memory checkpointing
recomputes native calls during backward without truncating the forecast graph.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from adapters.anie import ANIEPredictor
from adapters.anie_config import ANIEConfig
from adapters.models import build_model, fno_utilities
from adapters import source_file
from gift.publication_metadata import public_metadata
from training.baseline_control import _fno3d_input
from training.checkpoints import capture_rng, restore_rng, restore_optimizer, runtime_identity, state_digest, digest_file
from training.w1_v2_data import WindowBank, epoch_order

METHODS = ("anie_ns", "anie_f", "fno2d", "fno3d")


def atomic_json(path, payload):
    """Replace this run's mutable JSON pointer after a flushed sibling write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def atomic_torch(path, payload):
    """Replace a mutable checkpoint owned by this new run, never old artifacts."""
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def seed_initialization(seed):
    """Seed model/dropout streams; trajectory permutations use a separate RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_network(model_spec, device="cpu"):
    """Construct the explicit single-frame native model, without resetting RNG."""
    method = model_spec["method"]
    if method not in METHODS or model_spec.get("history") != 1:
        raise ValueError("W1-v2 requires an explicit supported single-frame model")
    if method.startswith("anie"):
        condition = "native" if method == "anie_ns" else "fixed_f"
        if model_spec.get("condition_mode") != condition:
            raise ValueError("ANIE name and condition branch disagree")
        return ANIEPredictor(config=ANIEConfig(condition_mode=condition),
                             source_root=model_spec["source_root"]).to(device)
    return build_model(method, device, model_spec["source_root"], history=1)


def model_query_times(model_spec):
    """Read local seconds from the scientific model spec, retaining the old grid.

    Native FNO2D steps and FNO3D coordinates require a uniform grid. The shared
    observation task fixes both endpoints at 0 and 3 seconds; a query grid is
    part of checkpoint identity rather than an operational inference option.
    """
    times = np.asarray(model_spec.get("query_times", np.arange(151) * .02), dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            abs(times[0]) > 2e-12 or abs(times[-1] - 3.) > 2e-12 or
            not np.all(np.diff(times) > 0) or
            not np.allclose(np.diff(times), 3. / (len(times) - 1), rtol=0, atol=2e-12)):
        raise ValueError("W1 query_times must be a uniform local-seconds grid from 0 to 3")
    if "future_steps" in model_spec and (isinstance(model_spec["future_steps"], bool) or
            not isinstance(model_spec["future_steps"], int) or model_spec["future_steps"] != len(times) - 1):
        raise ValueError("model future_steps and query_times differ")
    if "observation_dt" in model_spec and (isinstance(model_spec["observation_dt"], bool) or
            not isinstance(model_spec["observation_dt"], (int, float)) or
            not math.isfinite(model_spec["observation_dt"]) or
            abs(model_spec["observation_dt"] - 3. / (len(times) - 1)) > 2e-12):
        raise ValueError("model observation_dt and physical query_times differ")
    return times


def _bind_window_grid(model_spec, config, binding):
    """Bind dynamic data times before model creation and scientific resume checks.

    Legacy receipts/specs keep their original dictionary identity. A sparse
    receipt must supply its exact times, even when callers omit the redundant
    model field; explicit caller values may never contradict the observations.
    """
    spec = dict(model_spec)
    if "query_times" in binding:
        if "query_times" in spec:
            provided = np.asarray(spec["query_times"], dtype=np.float64)
            observed = np.asarray(binding["query_times"], dtype=np.float64)
            if provided.shape != observed.shape or not np.allclose(provided, observed, rtol=0, atol=2e-12):
                raise ValueError("model query_times differ from the shared observation grid")
        # Bind the observed canonical float64 values, allowing harmless last-bit
        # differences between arange(points)*dt and strided dense_times.
        spec["query_times"] = binding["query_times"]
        if "future_steps" in spec and spec["future_steps"] != binding["future_steps"]:
            raise ValueError("model future_steps differ from the shared observation grid")
        spec["future_steps"] = binding["future_steps"]
    times = model_query_times(spec)
    if binding.get("points", len(times)) != len(times):
        raise ValueError("model and shared observation point counts differ")
    if config.get("future_steps", len(times) - 1) != len(times) - 1:
        raise ValueError("training future_steps differ from the model/data grid")
    if len(times) != 151 and config.get("future_steps") != len(times) - 1:
        raise ValueError("a changed grid requires explicit training future_steps")
    return spec


def forward_model(model, initial, model_spec, normalization, device=None, training=False, horizon=None):
    """Predict [B,T,64,64] from only [B,64,64] omega(5).

    ANIE's first output is its native reconstruction. FNO's first output is
    the supplied field; only the remaining T-1 frames count in forecast scores.
    No target/history appears in this interface. FNO2D keeps every recursive
    dependency, optionally recomputing calls to bound resident training memory.
    """
    method = model_spec["method"]
    times = model_query_times(model_spec)
    future_steps = len(times) - 1
    horizon = future_steps if horizon is None else horizon
    if (isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or
            not 1 <= horizon <= future_steps or (method != "fno2d" and horizon != future_steps)):
        raise ValueError("only FNO2D's declared training curriculum can shorten the loss horizon")
    if method.startswith("anie"):
        return (model(initial, times=torch.as_tensor(times, dtype=initial.dtype, device=initial.device))
                if "query_times" in model_spec else model(initial))
    context = initial.unsqueeze(-1)
    if method == "fno3d":
        if future_steps // 2 + 1 < getattr(model, "modes3", 8):
            raise ValueError("FNO3D future grid is shorter than its native temporal mode count")
        native, _ = fno_utilities(model_spec["source_root"])
        normalizers = []
        for prefix in ("input", "output"):
            instance = native.__new__(native)
            instance.mean = normalization[prefix + "_mean"].to(initial.device)
            instance.std = normalization[prefix + "_std"].to(initial.device)
            instance.eps = normalization["eps"]
            normalizers.append(instance)
        first, second = normalizers
        raw = model(_fno3d_input(first.encode(context), future_steps))
        future = second.decode(raw.reshape(len(initial), *initial.shape[1:], future_steps))
    else:
        frames = []
        if training and model_spec.get("activation_checkpoint", False):
            from training.w1_v2_checkpoint import checkpoint_native_model
        for _ in range(horizon):
            next_frame = (checkpoint_native_model(model, context)
                          if training and model_spec.get("activation_checkpoint", False) else model(context))
            frames.append(next_frame)
            context = next_frame
        future = torch.cat(frames, dim=-1)
    return torch.cat((initial[:, None], future.permute(0, 3, 1, 2)), dim=1)


def native_loss(prediction, truth, method, criterion=None):
    """Mean over trajectories; ANIE full-grid MSE or native FNO horizon objective.

    ANIE includes the native initial reconstruction with weight 1/T. FNO2D
    retains its upstream sum over future frames; coarsening changes its scale.
    """
    if method.startswith("anie"):
        return (prediction - truth).square().mean()
    if method == "fno3d":
        return criterion(prediction[:, 1:].reshape(len(truth), -1), truth[:, 1:].reshape(len(truth), -1)) / len(truth)
    return sum(criterion(prediction[:, step].reshape(len(truth), -1),
                         truth[:, step].reshape(len(truth), -1))
               for step in range(1, prediction.shape[1])) / len(truth)


def training_horizon(method, epoch, config):
    """Predeclared FNO2D closed-loop curriculum; eval uses the complete grid.

    Every curriculum stage uses model predictions as its next input. The first
    six20-epoch stages were introduced after the original150-step random-init
    gradient failed. Changed grids still need fresh stability preflight; no
    stage adds observations, teacher forcing or detached state.
    """
    full = config.get("future_steps", 150)
    if isinstance(full, bool) or not isinstance(full, int) or full < 1:
        raise ValueError("future_steps must be a positive integer")
    if method != "fno2d" or not config.get("closed_loop_curriculum", False):
        return full
    horizons = config.get("closed_loop_horizons", [1, 5, 10, 20, 50, 100, 150])
    if (not isinstance(horizons, (list, tuple)) or len(horizons) != 7 or
            any(isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= full for h in horizons) or
            any(b < a for a, b in zip(horizons, horizons[1:])) or horizons[-1] != full):
        raise ValueError("closed_loop_horizons must declare seven monotone valid stages ending at future_steps")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("curriculum epoch must be a positive integer")
    return horizons[min((epoch - 1) // 20, 6)]


def relative_curves(prediction, truth):
    """Per-trajectory per-frame physical relative L2, accumulated in float64."""
    error = (prediction.double() - truth.double()).flatten(2)
    denominator = truth.double().flatten(2).norm(dim=2)
    if not bool((denominator > 0).all()):
        raise ValueError("zero-norm scoring target")
    return error.norm(dim=2) / denominator


@torch.no_grad()
def validation_measurement(model, bank, spec, normalization, device, micro_batch):
    """Score all fixed validation trajectories; any failed future gives +inf."""
    model.eval()
    curves = []
    for start in range(0, len(bank.validation), micro_batch):
        initial, truth = bank.batch(np.arange(start, min(start + micro_batch, len(bank.validation))),
                                   device, validation=True)
        prediction = forward_model(model, initial, spec, normalization, device, training=False)
        curves.append(relative_curves(prediction, truth).cpu())
    curves = torch.cat(curves).numpy()
    finite = np.isfinite(curves[:, 1:]).all(1)
    metric = float(curves[:, 1:].mean()) if finite.all() else float("inf")
    return dict(selection_metric=metric, future_relative_l2=metric, endpoint_relative_l2=float(curves[:, -1].mean()),
                initial_relative_l2=float(curves[:, 0].mean()), failed_trajectories=int((~finite).sum()),
                per_trajectory_future=curves[:, 1:].mean(1).tolist(), relative_l2_by_time=curves.mean(0).tolist())


def finite_json(value):
    """Encode failed numerical values as null with an explicit failure record."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def update_selection(selection, metric, epoch):
    """Best checkpoint uses every improvement; plateau uses a 0.1% anchor.

    Cumulative improvements can eventually clear the significant-improvement
    anchor. Equal scores retain the earlier checkpoint. Patience counts only
    scheduled validations, not intermediate training epochs.
    """
    better = math.isfinite(metric) and metric < selection["best_metric"]
    if better:
        selection.update(best_metric=metric, best_epoch=epoch)
    significant = math.isfinite(metric) and metric < selection["plateau_anchor"] * (1 - .001)
    if significant:
        selection.update(plateau_anchor=metric, stale_checks=0)
    else:
        selection["stale_checks"] += 1
    return better


def state_cpu(model):
    """Detached CPU state for portable inference and resume journals."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def scientific_identity(identity):
    """Compare actual inputs/method/settings; paths and source text are provenance."""
    result = dict(identity)
    result["model_spec"] = {k: v for k, v in identity["model_spec"].items() if k != "source_root"}
    result["data"] = {k: v for k, v in identity["data"].items() if k not in ("file", "source_file")}
    return result


def run_training(window, output, *, model_spec, config, device="cuda", resume=False, stop_after_epoch=None):
    """Run/resume one predeclared candidate or formal seed at epoch boundaries.

    Mutable latest/best refer only to this newly created run. History and best
    selection events are retained in the journal. Stop-after is an operational
    pause and cannot change the saved scheduler's scientific epoch horizon.
    """
    output = Path(output).resolve()
    bank = WindowBank(window, fixture=config.get("fixture", False))
    model_spec = _bind_window_grid(model_spec, config, bank.binding)
    from experiments.model_locations import runtime_model_spec
    runtime_spec = runtime_model_spec(model_spec)
    # Validate the complete declared curriculum before creating any run files.
    training_horizon(model_spec["method"], 1, config)
    seed_initialization(config["seed"])
    model = build_network(runtime_spec, device)
    project_root = Path(__file__).resolve().parents[1]
    source_provenance = dict(upstream=model.source_identity() if model_spec["method"].startswith("anie") else model.gift_external_source,
        runtime_source_root=runtime_spec["source_root"],
        project_files_sha256={relative: digest_file(source_file(project_root, relative)) for relative in (
            "training/w1_v2_train.py", "training/w1_v2_data.py", "adapters/anie.py", "adapters/anie_config.py",
            "adapters/models.py", "anie_sources.json", "external_sources.json")})
    normalization = bank.normalization(model_spec["method"])
    criterion = fno_utilities(runtime_spec["source_root"])[1](size_average=False) if model_spec["method"].startswith("fno") else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"],
                                                          eta_min=.01 * config["learning_rate"])
    selection = dict(best_metric=float("inf"), best_epoch=None, plateau_anchor=float("inf"), stale_checks=0)
    history, validations, start_epoch, updates = [], [], 0, 0
    identity = dict(schema="gift.w1-v2.training.v2", model_spec=model_spec, config=config, data=bank.binding)
    if resume:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if scientific_identity(saved["identity"]) != scientific_identity(identity):
            raise ValueError("W1-v2 scientific resume configuration/data differ")
        model.load_state_dict(saved["model_state"], strict=True)
        restore_optimizer(optimizer, saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        history, validations, selection = saved["history"], saved["validations"], saved["selection"]
        start_epoch, updates = saved["epoch"], saved["updates"]
        if saved.get("status") == "complete":
            return saved
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "ATTEMPT.json", dict(identity=identity, environment=runtime_identity(),
                                                  parameter_count=sum(p.numel() for p in model.parameters()),
                                                  initial_state_sha256=state_digest(model.state_dict()),
                                                  source_provenance=source_provenance,
                                                  precision="float32, TF32 disabled", fresh_initialization=True))
        atomic_torch(output / "latest.pt", dict(identity=identity, model_state=state_cpu(model),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=capture_rng(),
                     epoch=0, updates=0, selection=selection, history=[], validations=[], status="initialized",
                     terminal_reason=None, peak_memory_bytes=0, normalization=normalization))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    end_epoch = min(config["max_epochs"], stop_after_epoch or config["max_epochs"])
    status, terminal_reason = "running", None
    try:
        for epoch in range(start_epoch + 1, end_epoch + 1):
            model.train()
            order = epoch_order(len(bank.training), epoch - 1, config["sample_seed"])
            begin = time.perf_counter()
            train_loss = 0.
            horizon = training_horizon(model_spec["method"], epoch, config)
            used_lr = optimizer.param_groups[0]["lr"]
            for batch_start in range(0, len(order), config["effective_batch"]):
                group = order[batch_start:batch_start + config["effective_batch"]]
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, len(group), config["micro_batch"]):
                    rows = group[micro_start:micro_start + config["micro_batch"]]
                    initial, truth = bank.batch(rows, device)
                    prediction = forward_model(model, initial, runtime_spec, normalization, device, training=True, horizon=horizon)
                    loss = native_loss(prediction, truth, model_spec["method"], criterion)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite training objective")
                    (loss * (len(rows) / len(group))).backward()
                    train_loss += float(loss.detach()) * len(rows)
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()) or any(
                    isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
                    for slots in optimizer.state.values() for value in slots.values()):
                    raise FloatingPointError("nonfinite parameter or Adam state after update")
                updates += 1
            scheduler.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            history.append(dict(epoch=epoch, loss=train_loss / len(order), seconds=seconds, learning_rate=used_lr,
                                updates=updates, trajectories_seen=epoch * len(order), training_future_steps=horizon,
                                cumulative_supervised_future_fields=sum(row["training_future_steps"] * len(order) for row in history) + horizon * len(order)))
            measurement = None
            if epoch % config["validation_interval"] == 0 or epoch == config["max_epochs"] or epoch == 500:
                val_begin = time.perf_counter()
                measurement = validation_measurement(model, bank, runtime_spec, normalization, device, config["micro_batch"])
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                measurement.update(epoch=epoch, updates=updates, seconds=time.perf_counter() - val_begin)
                validations.append(measurement)
                better = update_selection(selection, measurement["selection_metric"], epoch)
                if better:
                    atomic_torch(output / "best.pt", dict(schema="gift.w1-v2.weights.v2", status="selected",
                                 model_state=state_cpu(model), model_spec=model_spec, config=config,
                                 normalization=normalization, epoch=epoch, val_metric=selection["best_metric"],
                                 stage=config["stage"], seed=config["seed"], data_binding=bank.binding,
                                 source_provenance=public_metadata(source_provenance)))
                    with (output / "BEST_HISTORY.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(dict(epoch=epoch, val_metric=selection["best_metric"]), allow_nan=False) + "\n")
                if config["stage"] == "formal" and epoch >= config["min_epochs"] and selection["stale_checks"] >= config["patience"]:
                    status, terminal_reason = "complete", "validation_plateau"
            if epoch == config["max_epochs"]:
                status, terminal_reason = "complete", "budget_limit"
            peak = torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0
            if epoch % 10 == 0 or epoch == end_epoch or status == "complete":
                payload = dict(identity=identity, model_state=state_cpu(model), optimizer=optimizer.state_dict(),
                               scheduler=scheduler.state_dict(), rng=capture_rng(), epoch=epoch, updates=updates,
                               selection=selection, history=history, validations=validations, status=status,
                               terminal_reason=terminal_reason, peak_memory_bytes=peak, normalization=normalization)
                atomic_torch(output / "latest.pt", payload)
                if epoch == 500:
                    atomic_torch(output / "epoch500.pt", payload)
                if status == "complete":
                    atomic_torch(output / "terminal.pt", payload)
                atomic_json(output / "STATUS.json", finite_json(dict(status=status, epoch=epoch, updates=updates,
                            best_epoch=selection["best_epoch"], best_val=selection["best_metric"],
                            last_validation=measurement or (validations[-1] if validations else None),
                            training_seconds=sum(r["seconds"] for r in history),
                            validation_seconds=sum(r["seconds"] for r in validations), peak_memory_bytes=peak,
                            terminal_reason=terminal_reason, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))
                atomic_json(output / "HISTORY.json", finite_json(dict(training=history, validation=validations)))
                print(json.dumps(finite_json(dict(method=model_spec["method"], seed=config["seed"], epoch=epoch,
                      status=status, train_loss=history[-1]["loss"], best_val=selection["best_metric"])), allow_nan=False), flush=True)
            if status == "complete":
                break
    except Exception as error:
        atomic_json(output / "FAILURE.json", dict(method=model_spec["method"], seed=config["seed"],
                    stage=config["stage"], status="failed", reason=repr(error),
                    failure_kind="numerical" if isinstance(error, (FloatingPointError, ArithmeticError)) else "infrastructure",
                    completed_epochs=len(history), updates=updates))
        raise
    return payload


def main():
    """Execute a concrete JSON run; scientific configuration is immutable on resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    spec = json.loads(args.run_config.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    run_training(spec["window"], spec["output"], model_spec=spec["model_spec"], config=spec["config"],
                 device=spec.get("device", "cuda"), resume=args.resume, stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
