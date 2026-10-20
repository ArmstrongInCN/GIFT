"""One fixed U-NO/U-Net run for the N300, 76-point, single-state W1/W2 task.

The original project budgets, native optimizer and summed relative-L2 losses
are retained. Training starts at t=5 and supervises 20 future steps for U-NO or
four for U-Net. Full 75-step validation is diagnostic, never a selection gate.
Scientific failures and physical-batch OOM remain visible in the unique slot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from adapters.w1_extended import build_network, forward_model, METHODS
from adapters.models import uno_components
from adapters import source_file
from gift.publication_metadata import public_metadata
from training.baseline_control import _configure_profile
from training.checkpoints import capture_rng, restore_rng, restore_optimizer, runtime_identity, state_digest, digest_file
from training.w1_single_train import _retain_resume_evidence, _restore_complete_progress
from training.w1_v2_data import WindowBank, epoch_order, SAMPLE_SEED
from training.w1_v2_train import (
    _bind_window_grid, atomic_json, atomic_torch, finite_json, model_query_times,
    relative_curves, scientific_identity, state_cpu,
)

PROFILE = "single_native76_n300"
WEIGHTS_SCHEMA = "gift.w1-extended-native.weights.v1"
TRAINING_SCHEMA = "gift.w1-extended-native.training.v1"


def native_training_config(method):
    """Declare one fixed original-project recipe, without search or reseeding."""
    if method not in METHODS:
        raise ValueError("unsupported extended baseline")
    batch = 16 if method == "uno" else 20
    return dict(profile=PROFILE, stage="formal", fixture=False, seed=0,
                sample_seed=SAMPLE_SEED, future_steps=75, training_future_steps=20 if method == "uno" else 4,
                effective_batch=batch, micro_batch=batch, max_epochs=500,
                learning_rate=.001, weight_decay=1e-5 if method == "uno" else 1e-4,
                scheduler_type="StepLR", scheduler_step_size=100, scheduler_gamma=.5,
                selection="terminal", validation_interval=10, checkpoint_interval=10,
                teacher_forcing=False, detach_rollout=False, closed_loop_curriculum=False,
                activation_checkpoint=method == "uno", normalization="none",
                training_origin="t5_only", native_loss="batch_sum_time_sum_relative_l2")


def _validate_config(method, config):
    """Permit only operational checkpoint cadence and explicitly marked CPU fixtures."""
    expected = native_training_config(method)
    variable = {"checkpoint_interval"}
    if config.get("fixture", False):
        variable |= {"fixture", "max_epochs", "effective_batch", "micro_batch", "validation_interval"}
    if set(config) != set(expected):
        raise ValueError("extended configuration keys differ from the fixed recipe")
    for key, value in expected.items():
        if key not in variable and config.get(key) != value:
            raise ValueError("extended configuration differs: " + key)
    for key in ("max_epochs", "effective_batch", "micro_batch", "checkpoint_interval", "validation_interval"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if config["effective_batch"] != config["micro_batch"]:
        raise ValueError("native physical batch cannot be split or accumulated")


def make_optimizer(model, spec, config):
    """Use the U-NO public Adam implementation and the existing U-Net Adam."""
    if spec["method"] == "uno":
        _, optimizer_type, loss_type = uno_components(source=model.gift_external_source)
        criterion = loss_type(size_average=False)
    else:
        optimizer_type, criterion = torch.optim.Adam, None
    return (optimizer_type(model.parameters(), lr=config["learning_rate"],
                           weight_decay=config["weight_decay"]), criterion)


def native_loss(prediction, truth, method, criterion=None):
    """Sum future per-field relative L2 over time and the actual physical batch.

    This is the public U-NO ``LpLoss(size_average=False)`` loop. U-Net retains
    the project's explicit equivalent L2 sum. Known t=0 is excluded. The tail
    batch is not rescaled to the full batch and no time averaging enters Adam.
    """
    if method not in METHODS:
        raise ValueError("unsupported native loss")
    losses = []
    for step in range(1, prediction.shape[1]):
        observed = truth[:, step].reshape(len(truth), -1)
        predicted = prediction[:, step].reshape(len(truth), -1)
        if not bool((observed.norm(dim=1) > 0).all()):
            raise FloatingPointError("zero-norm supervised target")
        losses.append(criterion(predicted, observed) if method == "uno" else
                      ((predicted - observed).norm(dim=1) / observed.norm(dim=1)).sum())
    return sum(losses)


@torch.no_grad()
def validation_measurement(model, bank, spec, device, batch_size):
    """Measure the complete 20-validation population through all 75 future steps."""
    model.eval()
    curves, future_sse, full_sse, count, future_elements, full_elements = [], 0., 0., 0, 0, 0
    failed = 0
    for start in range(0, len(bank.validation), batch_size):
        rows = np.arange(start, min(start + batch_size, len(bank.validation)))
        initial, truth = bank.batch(rows, device, validation=True)
        prediction = forward_model(model, initial, spec, device)
        failed += int((~torch.isfinite(prediction).flatten(1).all(1)).sum())
        squared = (prediction.double() - truth.double()).square()
        future_sse += float(squared[:, 1:].sum())
        full_sse += float(squared.sum())
        future_elements += squared[:, 1:].numel()
        full_elements += squared.numel()
        count += len(rows)
        curves.append(relative_curves(prediction, truth).cpu())
    values = torch.cat(curves).numpy()
    return dict(validation_mse=full_sse / full_elements if failed == 0 else float("inf"),
                future_mse=future_sse / future_elements if failed == 0 else float("inf"),
                future_relative_l2=float(values[:, 1:].mean()) if failed == 0 else float("inf"),
                endpoint_relative_l2=float(values[:, -1].mean()) if failed == 0 else float("inf"),
                failed_trajectories=failed, trajectories=count, selection_metric=None,
                relative_l2_by_time=values.mean(0).tolist())


def _source_provenance(model, spec):
    """Record the public files and local interface changes without editing sources."""
    root = Path(__file__).resolve().parents[1]
    names = (["navier_stokes_uno2d.py", "integral_operators.py", "Adam.py", "utilities3.py", "ns_train_2d.py", "ns_uno2d_main.py"]
             if spec["method"] == "uno" else ["Baselines/U-net.py"])
    source = Path(model.gift_external_source["directory"])
    local = ["adapters/w1_extended.py", "adapters/models.py", "training/w1_extended_baselines.py",
             "training/w1_v2_data.py", "training/w1_single_train.py", "training/baseline_control.py", "external_sources.json"]
    return dict(upstream=model.gift_external_source,
                native_files_sha256={name: digest_file(source / name) for name in names},
                project_files_sha256={name: digest_file(source_file(root, name)) for name in local},
                optimizer="public UNO.Adam" if spec["method"] == "uno" else "torch.optim.Adam")


def _failure_kind(error):
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
        return "resource"
    if "total norm" in str(error) and "non-finite" in str(error):
        return "numerical"
    return "numerical" if isinstance(error, (FloatingPointError, ArithmeticError)) else "infrastructure"


def run_training(window, output, *, model_spec, config, device="cuda", resume=False, stop_after_epoch=None):
    """Train one fresh slot or continue only its own full-state epoch checkpoint."""
    method = model_spec.get("method")
    _validate_config(method, config)
    bank = WindowBank(window, fixture=config["fixture"])
    spec = _bind_window_grid(model_spec, config, bank.binding)
    if not config["fixture"] and (len(bank.training) != 300 or len(model_query_times(spec)) != 76):
        raise ValueError("formal extended input must contain N300 and the 76-point grid")
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or stop_after_epoch < 1):
        raise ValueError("stop_after_epoch must be a positive integer")
    output = Path(output).resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif (output / "FAILURE.json").exists():
        failure = json.loads((output / "FAILURE.json").read_text(encoding="utf-8"))
        if failure["failure_kind"] != "infrastructure":
            raise ValueError("a numerical/resource failure cannot retry this unique slot")
    history, validations, updates, start_epoch, peak = [], [], 0, 0, 0
    last_epoch, last_validation = 0, None
    try:
        from experiments.model_locations import runtime_model_spec
        runtime_spec = runtime_model_spec(spec)
        _configure_profile(method, config["seed"])
        torch.set_default_dtype(torch.float32)
        model = build_network(runtime_spec, device)
        provenance = _source_provenance(model, runtime_spec)
        optimizer, criterion = make_optimizer(model, runtime_spec, config)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"],
                                                     gamma=config["scheduler_gamma"])
        identity = dict(schema=TRAINING_SCHEMA, model_spec=spec, config=config, data=bank.binding)
        if resume:
            saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            if scientific_identity(saved["identity"]) != scientific_identity(identity):
                raise ValueError("extended resume configuration/data differ")
            model.load_state_dict(saved["model_state"], strict=True)
            restore_optimizer(optimizer, saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            restore_rng(saved["rng"])
            history, validations = saved["history"], saved["validations"]
            start_epoch, updates, peak = saved["epoch"], saved["updates"], saved["peak_memory_bytes"]
            last_epoch = start_epoch
            last_validation = validations[-1] if validations else None
            if saved["status"] == "complete":
                _restore_complete_progress(output, saved)
                return saved
            _retain_resume_evidence(output, start_epoch)
        else:
            atomic_json(output / "ATTEMPT.json", dict(identity=identity, environment=runtime_identity(),
                        source_provenance=provenance, fresh_initialization=True,
                        initial_state_sha256=state_digest(model.state_dict()),
                        parameter_count=sum(item.numel() for item in model.parameters()),
                        physical_batch=config["effective_batch"], window_origin="t5 only"))
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        def journal(epoch, status):
            return dict(identity=identity, model_state=state_cpu(model), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), rng=capture_rng(), epoch=epoch, updates=updates,
                        history=history, validations=validations, status=status,
                        peak_memory_bytes=peak, normalization={"kind": "none"},
                        selected_checkpoint=str(output / "terminal_weights.pt") if status == "complete" else None,
                        source_provenance=provenance)

        if not resume:
            atomic_torch(output / "latest.pt", journal(0, "initialized"))
            atomic_json(output / "STATUS.json", dict(status="initialized", epoch=0, updates=0,
                        method=method, seed=config["seed"], max_epochs=config["max_epochs"],
                        selected_checkpoint=None, training_seconds=0., validation_seconds=0.))
        end_epoch = min(config["max_epochs"], stop_after_epoch or config["max_epochs"])
        if end_epoch <= start_epoch:
            return saved
        for epoch in range(start_epoch + 1, end_epoch + 1):
            epoch_begin = time.perf_counter()
            model.train()
            order = epoch_order(len(bank.training), epoch - 1, config["sample_seed"])
            used_lr, epoch_loss, begin = optimizer.param_groups[0]["lr"], 0., time.perf_counter()
            for start in range(0, len(order), config["effective_batch"]):
                rows = order[start:start + config["effective_batch"]]
                initial, truth = bank.batch(rows, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward_model(model, initial, runtime_spec, device, steps=config["training_future_steps"],
                                           training=True, checkpoint_uno=config["activation_checkpoint"])
                loss = native_loss(prediction, truth, method, criterion)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training objective")
                loss.backward()
                if method == "uno":
                    # The existing UNO controller uses infinity as a diagnostic
                    # limit: finite gradients are unchanged, but an overflowed
                    # total norm is still an explicit numerical failure.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
                elif any(item.grad is not None and not bool(torch.isfinite(item.grad).all()) for item in model.parameters()):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                if any(not bool(torch.isfinite(item).all()) for item in model.parameters()) or any(
                        isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
                        for slots in optimizer.state.values() for value in slots.values()):
                    raise FloatingPointError("nonfinite parameter or Adam state after update")
                updates += 1
                epoch_loss += float(loss.detach())
            scheduler.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            history.append(dict(epoch=epoch, loss=epoch_loss / len(order),
                                loss_definition="batch/time sum divided by epoch population for logging only",
                                seconds=time.perf_counter() - begin, learning_rate=used_lr,
                                next_learning_rate=optimizer.param_groups[0]["lr"], updates=updates,
                                trajectories_seen=epoch * len(order), training_future_steps=config["training_future_steps"],
                                cumulative_supervised_future_fields=epoch * len(order) * config["training_future_steps"]))
            if epoch % config["validation_interval"] == 0 or epoch == config["max_epochs"]:
                val_begin = time.perf_counter()
                measured = validation_measurement(model, bank, runtime_spec, device, config["effective_batch"])
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                measured.update(epoch=epoch, updates=updates, seconds=time.perf_counter() - val_begin)
                validations.append(measured)
                last_validation = measured
            status = "complete" if epoch == config["max_epochs"] else "running"
            if str(device).startswith("cuda"):
                peak = max(peak, torch.cuda.max_memory_allocated())
            checkpoint_seconds = 0.
            last_epoch = epoch
            if status == "complete":
                save_begin = time.perf_counter()
                atomic_torch(output / "terminal_weights.pt", dict(schema=WEIGHTS_SCHEMA, status="terminal",
                            model_state=state_cpu(model), model_spec=spec, config=config, normalization={"kind": "none"},
                            epoch=epoch, selection="terminal", stage=config["stage"], seed=config["seed"],
                            val_metric=last_validation["validation_mse"], data_binding=bank.binding,
                            source_provenance=public_metadata(provenance)))
                checkpoint_seconds += time.perf_counter() - save_begin
            history[-1].update(checkpoint_seconds=checkpoint_seconds, epoch_total_seconds=time.perf_counter() - epoch_begin)
            if epoch % config["checkpoint_interval"] == 0 or epoch == end_epoch:
                save_begin = time.perf_counter()
                payload = journal(epoch, status)
                atomic_torch(output / "latest.pt", payload)
                if status == "complete":
                    atomic_torch(output / "terminal.pt", payload)
                checkpoint_seconds += time.perf_counter() - save_begin
            history[-1].update(checkpoint_seconds=checkpoint_seconds, epoch_total_seconds=time.perf_counter() - epoch_begin)
            atomic_json(output / "STATUS.json", finite_json(dict(status=status, epoch=epoch, updates=updates,
                        method=method, seed=config["seed"], max_epochs=config["max_epochs"],
                        training_future_steps=config["training_future_steps"], last_validation=last_validation,
                        selection="terminal", best_epoch=None, selected_checkpoint=(str(output / "terminal_weights.pt")
                        if status == "complete" else None), training_seconds=sum(row["seconds"] for row in history),
                        validation_seconds=sum(row["seconds"] for row in validations),
                        checkpoint_seconds=sum(row.get("checkpoint_seconds", 0.) for row in history),
                        epoch_total_seconds=sum(row["epoch_total_seconds"] for row in history),
                        peak_memory_bytes=peak, terminal_reason="native_epoch_limit" if status == "complete" else None,
                        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))
            atomic_json(output / "HISTORY.json", finite_json(dict(training=history, validation=validations)))
            print(json.dumps(dict(method=method, epoch=epoch, updates=updates, status=status,
                                  training_relative_l2_sum_per_trajectory=history[-1]["loss"])), flush=True)
        return payload
    except Exception as error:
        failure = dict(method=method, seed=config["seed"], stage=config["stage"], status="failed",
                       reason=repr(error), failure_kind=_failure_kind(error), completed_epochs=last_epoch,
                       updates=updates, failed_epoch=locals().get("epoch"), selected_checkpoint=None,
                       retry_parameters_changed=False, physical_batch=config["effective_batch"])
        atomic_json(output / "FAILURE.json", failure)
        atomic_json(output / "STATUS.json", failure)
        raise


def main():
    """Consume a declared unique run slot; no runtime parameter selection exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    run = json.loads(args.run_config.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    run_training(run["window"], run["output"], model_spec=run["model_spec"], config=run["config"],
                 device=run.get("device", "cuda"), resume=args.resume, stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
