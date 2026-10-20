"""One fresh ANIE/FNO run using the declared author training configuration.

The shared window contains only training and validation observations. ANIE
retains its native MSE, delayed cosine, original EarlyStopping and validation
best model. FNO retains the official batch-summed relative L2, StepLR and
terminal model. Only FNO2D's predeclared closed-loop horizon curriculum differs
from its ten-step example; every stage keeps the complete prediction graph.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from adapters.anie import ANIEPredictor, native_early_stopping
from adapters import source_file
from adapters.anie_config import ANIEConfig
from adapters.models import build_model, fno_utilities
from gift.publication_metadata import public_metadata
from training.checkpoints import capture_rng, restore_rng, restore_optimizer, runtime_identity, state_digest, digest_file
from training.w1_v2_data import WindowBank, epoch_order, SAMPLE_SEED
from training.w1_v2_train import (
    _bind_window_grid, atomic_json, atomic_torch, finite_json, forward_model,
    model_query_times, relative_curves, scientific_identity, seed_initialization,
    state_cpu,
)

PROFILE = "single_native76_n300"
SOURCE_PROFILE = "native_ns_single_v1"
NS_SOURCE_PROFILES = (SOURCE_PROFILE, "aligned_ns_single_v1")
METHODS = ("anie_ns", "fno2d", "fno3d")
WEIGHTS_SCHEMA = "gift.w1-single-native.weights.v1"
TRAINING_SCHEMA = "gift.w1-single-native.training.v1"
CURRICULUM = [1, 2, 5, 10, 25, 50, 75]


def native_training_config(method):
    """Return one fixed configuration, with no candidate or seed selection.

    The ANIE final twelve trajectories remain in an actual-size mean update.
    FNO uses physical batches, not microbatch accumulation across BatchNorm.
    ``fixture`` may only be enabled with a fixture-marked input for CPU tests.
    """
    if method not in METHODS:
        raise ValueError("unsupported single-native method")
    config = dict(profile=PROFILE, stage="formal", fixture=False,
                  sample_seed=SAMPLE_SEED, future_steps=75, checkpoint_interval=10,
                  validation_interval=1, weight_decay=1e-4)
    if method == "anie_ns":
        config.update(seed=7, learning_rate=1e-4, effective_batch=32, micro_batch=6,
                      max_epochs=5000, scheduler_type="CosineAnnealingLR",
                      scheduler_T_max=101, scheduler_eta_min=1e-9,
                      scheduler_first_completed_epoch=17,
                      early_stopping_patience=1000, early_stopping_min_delta=0.,
                      selection="validation_mse_best", closed_loop_curriculum=False)
    else:
        batch = 20 if method == "fno2d" else 10
        config.update(seed=0, learning_rate=.001, effective_batch=batch, micro_batch=batch,
                      max_epochs=500, scheduler_type="StepLR", scheduler_step_size=100,
                      scheduler_gamma=.5, selection="terminal",
                      closed_loop_curriculum=method == "fno2d")
        if method == "fno2d":
            config.update(closed_loop_horizons=list(CURRICULUM), curriculum_stage_epochs=20)
    return config


def _validate_config(method, config):
    """Reject silent changes to the formal profile; permit tiny marked fixtures."""
    expected = native_training_config(method)
    if config.get("fixture", False):
        # Fixtures may shorten budgets and batches, but still exercise the
        # same scheduler, loss, model-selection and EarlyStopping semantics.
        variable = {"fixture", "max_epochs", "effective_batch", "micro_batch",
                    "early_stopping_patience", "checkpoint_interval"}
    else:
        variable = {"sample_seed", "checkpoint_interval"}
    for key, value in expected.items():
        if key not in variable and config.get(key) != value:
            raise ValueError("single-native configuration differs: " + key)
    for key in ("max_epochs", "effective_batch", "micro_batch", "checkpoint_interval"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(key + " must be a positive integer")
    if config["micro_batch"] > config["effective_batch"]:
        raise ValueError("micro_batch exceeds effective_batch")
    if method.startswith("fno") and config["micro_batch"] != config["effective_batch"]:
        raise ValueError("native FNO requires its complete physical batch")


def build_network(model_spec, device="cpu"):
    """Construct the explicitly recorded ANIE coordinate profile or one-state FNO.

    Historical native-coordinate checkpoints retain their source semantics.
    Corrected runs declare the aligned profile in training and inference alike;
    neither profile changes network shapes or the NS initial-state condition.
    """
    method = model_spec.get("method")
    if method not in METHODS or model_spec.get("history") != 1:
        raise ValueError("single-native models require a supported method and history=1")
    if method == "anie_ns":
        if (model_spec.get("source_profile") not in NS_SOURCE_PROFILES or
                model_spec.get("condition_mode") != "native"):
            raise ValueError("single-native ANIE requires an explicit NS source profile")
        return ANIEPredictor(ANIEConfig(condition_mode="native", source_profile=model_spec["source_profile"]),
                             source_root=model_spec["source_root"]).to(device)
    return build_model(method, device, model_spec["source_root"], history=1)




def make_scheduler(optimizer, method, config):
    """Use native PyTorch scheduler classes and the declared author constants."""
    if method == "anie_ns":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["scheduler_T_max"], eta_min=config["scheduler_eta_min"])
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])


def advance_scheduler(scheduler, method, completed_epoch, config):
    """ANIE zero-based i>15 starts after completed epoch17; FNO steps every epoch."""
    if method != "anie_ns" or completed_epoch >= config["scheduler_first_completed_epoch"]:
        scheduler.step()


def training_horizon(method, epoch, config):
    """Read the fixed FNO2D course without validation-driven changes or detaching."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if method != "fno2d":
        return config["future_steps"]
    index = min((epoch - 1) // config["curriculum_stage_epochs"], len(CURRICULUM) - 1)
    return config["closed_loop_horizons"][index]


def native_loss(prediction, truth, method, criterion=None):
    """Preserve ANIE full-grid mean MSE and the official FNO batch sums.

    FNO2D sums future per-frame relative L2; FNO3D compares the flattened
    future spacetime field. Known t=0 is excluded from both FNO objectives.
    ANIE's raw initial reconstruction stays inside its native MSE.
    """
    if method == "anie_ns":
        return torch.nn.functional.mse_loss(prediction, truth.detach())
    if method == "fno3d":
        return criterion(prediction[:, 1:].reshape(len(truth), -1), truth[:, 1:].reshape(len(truth), -1))
    if method != "fno2d":
        raise ValueError("unsupported native loss method")
    return sum(criterion(prediction[:, step].reshape(len(truth), -1),
                         truth[:, step].reshape(len(truth), -1))
               for step in range(1, prediction.shape[1]))


@torch.no_grad()
def validation_measurement(model, bank, spec, normalization, device, batch_size):
    """Measure every validation trajectory; ANIE selection uses all-grid MSE.

    Future relative L2 remains a diagnostic. No test data or input history
    enters this function. A nonfinite complete prediction is a recorded
    diagnostic, never a score averaged over the finite subset. The ANIE
    caller treats it as a numerical failure; FNO keeps its terminal policy.
    """
    model.eval()
    curves, mse_sum, future_sse, initial_mse_sum = [], 0., 0., 0.
    failed = 0
    elements, count = 0, 0
    for start in range(0, len(bank.validation), batch_size):
        rows = np.arange(start, min(start + batch_size, len(bank.validation)))
        initial, truth = bank.batch(rows, device, validation=True)
        prediction = forward_model(model, initial, spec, normalization, device)
        failed += int((~torch.isfinite(prediction).flatten(1).all(1)).sum())
        squared = (prediction.double() - truth.double()).square()
        # Preserve native float32 F.mse_loss for ANIE checkpoint selection;
        # float64 physical diagnostics do not decide its best model.
        mse_sum += float(torch.nn.functional.mse_loss(prediction, truth)) * squared.numel()
        future_sse += float(squared[:, 1:].sum())
        initial_mse_sum += float(squared[:, 0].mean()) * len(rows)
        elements += squared.numel()
        count += len(rows)
        curves.append(relative_curves(prediction, truth).cpu())
    values = torch.cat(curves).numpy()
    mse = mse_sum / elements if failed == 0 else float("inf")
    future_mean = float(values[:, 1:].mean()) if failed == 0 else float("inf")
    return dict(validation_mse=mse, selection_metric=mse if spec["method"] == "anie_ns" else None,
                future_mse=float(future_sse / (elements - count * int(np.prod(bank.validation.shape[2:])))) if failed == 0 else float("inf"),
                future_relative_l2=future_mean, endpoint_relative_l2=float(values[:, -1].mean()) if failed == 0 else float("inf"),
                initial_relative_l2=float(values[:, 0].mean()) if failed == 0 else float("inf"),
                initial_mse=initial_mse_sum / count if failed == 0 else float("inf"),
                failed_trajectories=failed, trajectories=count, relative_l2_by_time=values.mean(0).tolist())


def _source_provenance(model, spec):
    """Record actual adapter/controller files and native training-control source."""
    root = Path(__file__).resolve().parents[1]
    paths = ["training/w1_single_train.py", "training/w1_v2_train.py", "training/w1_v2_data.py",
             "training/w1_v2_checkpoint.py", "adapters/anie.py", "adapters/anie_config.py",
             "adapters/models.py", "anie_sources.json", "external_sources.json"]
    upstream = model.source_identity() if spec["method"] == "anie_ns" else model.gift_external_source
    native_files = ([Path(spec["source_root"]) / "IE_source" / "utils.py"] if spec["method"] == "anie_ns" else
                    [Path(upstream["directory"]) / name for name in
                     ("utilities3.py", "fourier_2d_time.py" if spec["method"] == "fno2d" else "fourier_3d.py")])
    return dict(upstream=upstream, project_files_sha256={name: digest_file(source_file(root, name)) for name in paths},
                native_training_files_sha256={str(path.resolve()): digest_file(path) for path in native_files},
                optimizer="torch.optim.Adam", optimizer_module=torch.optim.Adam.__module__)


def _weights_payload(model, spec, config, normalization, bank, provenance, epoch, metric, status):
    """Save a portable selected/terminal model with its actual profile and data."""
    return dict(schema=WEIGHTS_SCHEMA, status=status, model_state=state_cpu(model), model_spec=spec,
                config=config, normalization=normalization, epoch=epoch, val_metric=metric,
                selection=config["selection"], stage=config["stage"], seed=config["seed"],
                data_binding=bank.binding, source_provenance=public_metadata(provenance))


def _retain_resume_evidence(output, checkpoint_epoch):
    """Keep pre-resume progress, including epochs beyond the last committed journal.

    Operational JSON may lead a ten-epoch checkpoint. It is copied into a
    fresh directory before any pointer is replaced, so replayed epochs never
    erase evidence of the first attempt. No history file is deleted.
    """
    directory = output / ("resume_evidence_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(exist_ok=False)
    for name in ("HISTORY.json", "STATUS.json", "FAILURE.json"):
        path = output / name
        if path.exists():
            (directory / name).write_bytes(path.read_bytes())
    atomic_json(directory / "RESUME.json", dict(checkpoint_epoch=checkpoint_epoch,
                reason="same-run infrastructure recovery; uncommitted work may replay"))


def _restore_complete_progress(output, saved):
    """Recover operational JSON after a committed terminal journal, keeping weights intact.

    Model weights precede the terminal journal, while STATUS/HISTORY follow it.
    A crash in that interval therefore needs only JSON recovery. Post-journal
    timing fields in an already complete status remain the original measurements.
    """
    config, spec = saved["identity"]["config"], saved["identity"]["model_spec"]
    method = spec["method"]
    selected = output / ("best.pt" if method == "anie_ns" else "terminal_weights.pt")
    if not saved.get("selected_checkpoint") or not selected.is_file():
        raise ValueError("completed journal requires its existing selected model weights")
    current = {}
    for name in ("STATUS.json", "HISTORY.json"):
        try:
            current[name] = json.loads((output / name).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current[name] = None
    status, history = current["STATUS.json"], current["HISTORY.json"]
    status_ready = isinstance(status, dict) and all(status.get(key) == value for key, value in
        dict(status="complete", epoch=saved["epoch"], updates=saved["updates"],
             selected_checkpoint=str(selected)).items())
    def committed_rows(rows):
        # The last JSON timing includes the checkpoint write, unlike its own
        # journal. Compare scientific progress while retaining measured timing.
        return [{key: value for key, value in row.items()
                 if key not in ("checkpoint_seconds", "epoch_total_seconds")} for row in rows]

    history_ready = (isinstance(history, dict) and isinstance(history.get("training"), list)
                     and isinstance(history.get("validation"), list)
                     and all(isinstance(row, dict) for row in history["training"])
                     and committed_rows(history["training"]) == finite_json(committed_rows(saved["history"]))
                     and history["validation"] == finite_json(saved["validations"]))
    if status_ready and history_ready:
        return
    _retain_resume_evidence(output, saved["epoch"])
    if not status_ready:
        training, validations = saved["history"], saved["validations"]
        selection = saved.get("selection", {})
        record = dict(status="complete", epoch=saved["epoch"], updates=saved["updates"],
                      method=method, seed=config["seed"], max_epochs=config["max_epochs"],
                      selected_checkpoint=str(selected), best_epoch=selection.get("best_epoch"),
                      last_validation=validations[-1] if validations else None,
                      training_seconds=sum(row["seconds"] for row in training),
                      validation_seconds=sum(row["seconds"] for row in validations),
                      checkpoint_seconds=sum(row.get("checkpoint_seconds", 0.) for row in training),
                      epoch_total_seconds=sum(row.get("epoch_total_seconds", 0.) for row in training),
                      peak_memory_bytes=saved["peak_memory_bytes"],
                      terminal_reason=saved.get("terminal_reason", "native_epoch_limit"),
                      time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      operational_progress_recovered_from="latest.pt")
        if method in ("uno", "unet"):
            record.update(selection="terminal", training_future_steps=config["training_future_steps"])
        else:
            record["best_val"] = selection.get("best_metric") if method == "anie_ns" else None
        atomic_json(output / "STATUS.json", finite_json(record))
    if not history_ready:
        atomic_json(output / "HISTORY.json", finite_json(dict(training=saved["history"], validation=saved["validations"])))


def run_training(window, output, *, model_spec, config, device="cuda", resume=False, stop_after_epoch=None):
    """Run one fixed native configuration, or recover that run's committed state.

    Checkpoints restore model, Adam, scheduler, native EarlyStopping and RNG.
    A numerical failure is terminal for this unique slot. Infrastructure
    interruption may recover the same configuration; no OOM batch fallback
    or hidden learning-rate change is performed.
    """
    method = model_spec.get("method")
    _validate_config(method, config)
    output = Path(output).resolve()
    bank = WindowBank(window, fixture=config["fixture"])
    spec = _bind_window_grid(model_spec, config, bank.binding)
    if not config["fixture"] and (len(bank.training) != 300 or len(model_query_times(spec)) != 76):
        raise ValueError("single-native formal input must contain N300 and the 76-point grid")
    if stop_after_epoch is not None and (isinstance(stop_after_epoch, bool) or
            not isinstance(stop_after_epoch, int) or stop_after_epoch < 1):
        raise ValueError("stop_after_epoch must be a positive integer")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    else:
        failure_path = output / "FAILURE.json"
        if failure_path.exists() and json.loads(failure_path.read_text(encoding="utf-8"))["failure_kind"] == "numerical":
            raise ValueError("a scientific numerical failure cannot retry this unique slot")
    try:
        from experiments.model_locations import runtime_model_spec
        runtime_spec = runtime_model_spec(spec)
        seed_initialization(config["seed"])
        model = build_network(runtime_spec, device)
        provenance = _source_provenance(model, runtime_spec)
        normalization = bank.normalization(method)
        criterion = fno_utilities(runtime_spec["source_root"])[1](size_average=False) if method.startswith("fno") else None
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = make_scheduler(optimizer, method, config)
        stopper = (native_early_stopping(runtime_spec["source_root"], config["early_stopping_patience"],
                                       config["early_stopping_min_delta"]) if method == "anie_ns" else None)
    except Exception as error:
        failure = dict(method=method, seed=config["seed"], stage=config["stage"], status="failed",
                       reason=repr(error), failure_kind="infrastructure", completed_epochs=0, updates=0,
                       failure_phase="initialization", selected_checkpoint=None)
        atomic_json(output / "FAILURE.json", failure)
        atomic_json(output / "STATUS.json", failure)
        raise
    identity = dict(schema=TRAINING_SCHEMA, model_spec=spec, config=config, data=bank.binding)
    history, validations, updates, start_epoch = [], [], 0, 0
    selection = dict(best_metric=float("inf"), best_epoch=None)
    peak = 0
    if resume:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if scientific_identity(saved["identity"]) != scientific_identity(identity):
            raise ValueError("single-native resume configuration/data differ")
        model.load_state_dict(saved["model_state"], strict=True)
        restore_optimizer(optimizer, saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        history, validations, selection = saved["history"], saved["validations"], saved["selection"]
        start_epoch, updates, peak = saved["epoch"], saved["updates"], saved["peak_memory_bytes"]
        if stopper is not None:
            stopper.__dict__.update(saved["early_stopping"])
        if saved["status"] == "complete":
            _restore_complete_progress(output, saved)
            return saved
        _retain_resume_evidence(output, start_epoch)
    else:
        atomic_json(output / "ATTEMPT.json", dict(identity=identity, environment=runtime_identity(),
                    source_provenance=provenance, fresh_initialization=True,
                    initial_state_sha256=state_digest(model.state_dict()),
                    parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                    precision="float32, TF32 disabled", physical_batch=config["micro_batch"]))
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def journal(epoch, status, reason):
        """Commit all continuation state together, including the selected-model pointer."""
        selected = (str(output / "best.pt") if method == "anie_ns" and selection["best_epoch"] is not None else
                    str(output / "terminal_weights.pt") if method.startswith("fno") and status == "complete" else None)
        return dict(identity=identity, model_state=state_cpu(model), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=capture_rng(), epoch=epoch, updates=updates,
                    early_stopping=copy.deepcopy(stopper.__dict__) if stopper is not None else None,
                    selection=copy.deepcopy(selection), history=history, validations=validations,
                    status=status, terminal_reason=reason, peak_memory_bytes=peak,
                    normalization=normalization, selected_checkpoint=selected, source_provenance=provenance)

    if not resume:
        atomic_torch(output / "latest.pt", journal(0, "initialized", None))
        atomic_json(output / "STATUS.json", dict(status="initialized", epoch=0, updates=0,
                    method=method, seed=config["seed"], max_epochs=config["max_epochs"],
                    selected_checkpoint=None, training_seconds=0., validation_seconds=0.,
                    checkpoint_seconds=0., epoch_total_seconds=0.))
    end_epoch = min(config["max_epochs"], stop_after_epoch or config["max_epochs"])
    if end_epoch <= start_epoch:
        return saved
    status, reason, last_epoch = "running", None, start_epoch
    try:
        for epoch in range(start_epoch + 1, end_epoch + 1):
            epoch_begin = time.perf_counter()
            checkpoint_seconds = 0.
            model.train()
            order = epoch_order(len(bank.training), epoch - 1, config["sample_seed"])
            horizon = training_horizon(method, epoch, config)
            used_lr = optimizer.param_groups[0]["lr"]
            begin, epoch_loss = time.perf_counter(), 0.
            for batch_start in range(0, len(order), config["effective_batch"]):
                group = order[batch_start:batch_start + config["effective_batch"]]
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, len(group), config["micro_batch"]):
                    rows = group[micro_start:micro_start + config["micro_batch"]]
                    initial, truth = bank.batch(rows, device)
                    prediction = forward_model(model, initial, runtime_spec, normalization, device, training=True, horizon=horizon)
                    loss = native_loss(prediction, truth, method, criterion)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite training objective")
                    # ANIE accumulates the effective group's actual mean.
                    # FNO gets one complete physical batch and its native sum.
                    (loss * (len(rows) / len(group)) if method == "anie_ns" else loss).backward()
                    epoch_loss += float(loss.detach()) * (len(rows) if method == "anie_ns" else 1)
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters()):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()) or any(
                        isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
                        for slots in optimizer.state.values() for value in slots.values()):
                    raise FloatingPointError("nonfinite parameter or Adam state after update")
                updates += 1
            advance_scheduler(scheduler, method, epoch, config)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            history.append(dict(epoch=epoch, loss=epoch_loss / len(order), seconds=time.perf_counter() - begin,
                           learning_rate=used_lr, next_learning_rate=optimizer.param_groups[0]["lr"],
                           updates=updates, trajectories_seen=epoch * len(order), training_future_steps=horizon,
                           checkpoint_seconds=0., epoch_total_seconds=0.,
                           cumulative_supervised_future_fields=sum(row["training_future_steps"] * len(order)
                                                                  for row in history) + horizon * len(order)))
            val_begin = time.perf_counter()
            measured = validation_measurement(model, bank, runtime_spec, normalization, device, config["micro_batch"])
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            measured.update(epoch=epoch, updates=updates, seconds=time.perf_counter() - val_begin)
            validations.append(measured)
            if stopper is not None:
                metric = measured["validation_mse"]
                if not math.isfinite(metric):
                    raise FloatingPointError("nonfinite complete ANIE validation MSE")
                if metric < selection["best_metric"]:
                    selection.update(best_metric=metric, best_epoch=epoch)
                    save_begin = time.perf_counter()
                    atomic_torch(output / "best.pt", _weights_payload(model, spec, config, normalization, bank,
                                 provenance, epoch, metric, "selected"))
                    with (output / "BEST_HISTORY.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(dict(epoch=epoch, validation_mse=metric), allow_nan=False) + "\n")
                    checkpoint_seconds += time.perf_counter() - save_begin
                stopper(metric)
                if stopper.early_stop:
                    status, reason = "complete", "native_early_stopping"
            if epoch == config["max_epochs"]:
                status, reason = "complete", "native_epoch_limit"
            if str(device).startswith("cuda"):
                peak = max(peak, torch.cuda.max_memory_allocated())
            last_epoch = epoch
            if status == "complete" and method.startswith("fno"):
                save_begin = time.perf_counter()
                atomic_torch(output / "terminal_weights.pt", _weights_payload(model, spec, config, normalization,
                             bank, provenance, epoch, measured["validation_mse"], "terminal"))
                checkpoint_seconds += time.perf_counter() - save_begin
            history[-1]["checkpoint_seconds"] = checkpoint_seconds
            history[-1]["epoch_total_seconds"] = time.perf_counter() - epoch_begin
            if epoch % config["checkpoint_interval"] == 0 or epoch == end_epoch or status == "complete":
                save_begin = time.perf_counter()
                payload = journal(epoch, status, reason)
                atomic_torch(output / "latest.pt", payload)
                if status == "complete":
                    atomic_torch(output / "terminal.pt", payload)
                checkpoint_seconds += time.perf_counter() - save_begin
            # These operational timing fields include this epoch's native
            # validation and actual weight/journal writes. JSON/log overhead
            # is outside the total; queue attempt wall time records that too.
            history[-1]["checkpoint_seconds"] = checkpoint_seconds
            history[-1]["epoch_total_seconds"] = time.perf_counter() - epoch_begin
            selected = str(output / "best.pt") if method == "anie_ns" and selection["best_epoch"] is not None else (
                       str(output / "terminal_weights.pt") if status == "complete" else None)
            atomic_json(output / "STATUS.json", finite_json(dict(status=status, epoch=epoch, updates=updates,
                        method=method, seed=config["seed"], best_epoch=selection["best_epoch"],
                        best_val=selection["best_metric"] if method == "anie_ns" else None,
                        last_validation=measured, selected_checkpoint=selected,
                        training_seconds=sum(row["seconds"] for row in history),
                        validation_seconds=sum(row["seconds"] for row in validations),
                        checkpoint_seconds=sum(row.get("checkpoint_seconds", 0.) for row in history),
                        epoch_total_seconds=sum(row.get("epoch_total_seconds", 0.) for row in history),
                        max_epochs=config["max_epochs"],
                        peak_memory_bytes=peak, terminal_reason=reason,
                        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))
            atomic_json(output / "HISTORY.json", finite_json(dict(training=history, validation=validations)))
            print(json.dumps(finite_json(dict(method=method, seed=config["seed"], epoch=epoch, status=status,
                                  validation_mse=measured["validation_mse"], updates=updates,
                                  validation_failed_trajectories=measured["failed_trajectories"])), allow_nan=False), flush=True)
            if status == "complete":
                break
    except Exception as error:
        kind = "numerical" if isinstance(error, (FloatingPointError, ArithmeticError)) else "infrastructure"
        failure = dict(method=method, seed=config["seed"], stage=config["stage"], status="failed",
                       reason=repr(error), failure_kind=kind, completed_epochs=last_epoch, updates=updates,
                       failed_epoch=epoch if "epoch" in locals() else None,
                       selected_checkpoint=None, retry_parameters_changed=False)
        atomic_json(output / "FAILURE.json", failure)
        atomic_json(output / "STATUS.json", failure)
        raise
    return payload


def main():
    """Execute the locked single-run JSON configuration without queue/search logic."""
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
