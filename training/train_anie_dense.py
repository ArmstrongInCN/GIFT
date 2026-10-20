"""Independent dense151 ANIE training and native same-run continuation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from adapters import source_file

from training.anie_data import ObservationBank, TRAIN_FILE, WINDOW_FRAMES, epoch_indices
from training.checkpoints import CheckpointStore, restore_optimizer, digest_file, runtime_identity
from training.anie_control import (checked_paths, configure_random, native_training_objects,
                                   step_scheduler, training_cost, export_terminal)


DENSE_INDICES = np.arange(WINDOW_FRAMES, dtype=np.int64)


def dense_time_grid_identity(device="cpu"):
    """Describe the exact float32 physical and normalized times used by native IE.

    Construct physical seconds from NumPy integer indices times float64 DT,
    then convert to Torch float32; normalized time divides by the 3-second
    window. This order differs in the last bits from float32 arange times DT.
    Return both value vectors and descriptive byte/backend metadata.
    """
    device = torch.device(device)
    physical = torch.as_tensor((DENSE_INDICES * .02).tolist(), dtype=torch.float32, device=device)
    normalized = physical / 3.0
    return dict(dtype="float32", byte_order=sys.byteorder, frames=WINDOW_FRAMES, window_seconds=3.0,
                declared_backend=device.type, device=str(device),
                physical_float32_values=physical.detach().cpu().tolist(),
                normalized_float32_values=normalized.detach().cpu().tolist(),
                physical_float32_sha256=hashlib.sha256(physical.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                normalized_float32_sha256=hashlib.sha256(normalized.detach().cpu().contiguous().numpy().tobytes()).hexdigest())


def record_solver_iterations(model, histogram):
    """Increment a histogram once per native forward microbatch.

    The adapter exposes last_solver_iterations when available; missing
    reporting is stored as unreported rather than an invented solver count.
    """
    iterations = getattr(model, "last_solver_iterations", None)
    key = str(int(iterations)) if iterations is not None else "unreported"
    histogram[key] = histogram.get(key, 0) + 1


def arguments(argv=None):
    """Parse ANIE training/resume options while keeping an ordinary invocation read-only.

    Positive counts determine the declared budget; --stop-after-epoch is a
    committed pause rather than a shorter scientific schedule. Model/data
    construction remains in main after argument validation.
    """
    parser = argparse.ArgumentParser(description="ANIE dense151 baseline; default is a read-only plan", allow_abbrev=False)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path,
                        help="fixed official ANIE checkout; alternatively GIFT_ANIE_SOURCE_ROOT")
    parser.add_argument("--data-file", default=TRAIN_FILE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution", choices=("auto", "eager"), default="auto",
                        help="auto resolves to eager: retain the native solver's dynamic Python convergence checks")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accumulation", type=int,
                        help="formal effective batch is32; default32/micro-batch")
    parser.add_argument("--cache-observations", action="store_true", help="retain decoded train/validation arrays in RAM")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--tiny", action="store_true", help="explicit NONFORMAL diagnostic budget")
    parser.add_argument("--tiny-epochs", type=int, default=2)
    parser.add_argument("--tiny-trajectories", type=int, default=2)
    args = parser.parse_args(argv)
    if args.resume and not args.run_training:
        parser.error("--resume requires --run-training")
    if args.run_training and args.output is None:
        parser.error("--run-training requires an explicit output directory")
    if any(value < 1 for value in (args.micro_batch, args.checkpoint_interval,
                                  args.tiny_epochs, args.tiny_trajectories)):
        parser.error("batch, accumulation, checkpoint interval and diagnostic counts must be positive")
    if args.accumulation is not None and args.accumulation < 1:
        parser.error("accumulation must be positive")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be positive")
    if args.epochs is not None and args.epochs < 1 or args.trajectories is not None and args.trajectories < 1:
        parser.error("epochs and trajectories must be positive")
    return args


def configuration(args):
    """Return the 151-observation ANIE protocol and explicit budget overrides.

    Physical times are frame indices multiplied by DT=0.02 seconds. A window
    contains 151 frames over 3 seconds. micro_batch controls resident memory;
    accumulation controls effective batch size and keeps the tail. Selection
    is always the terminal epoch, independent of validation loss.
    """
    if args.accumulation is None:
        accumulation = 1 if args.tiny else max(1, 32 // args.micro_batch)
    else:
        accumulation = args.accumulation
    return dict(method="anie_dense151", protocol="anie.dense151.v1", formal=not args.tiny, fixture_only=bool(args.tiny),
                epochs=args.epochs if args.epochs is not None else args.tiny_epochs if args.tiny else 500,
                trajectories=args.trajectories if args.trajectories is not None else args.tiny_trajectories if args.tiny else 1000,
                validation_trajectories=20, seed=args.seed, dt=0.02,
                window_seconds=3.0, frames=WINDOW_FRAMES, initial_mode="single_observed_field",
                training_time_points=151, prediction_time_points=151,
                training_frame_indices=DENSE_INDICES.tolist(),
                training_local_times=(DENSE_INDICES * .02).tolist(),
                training_time_grid_recipe="numpy.int64.arange151_times_float64_0.02_then_torch_float32;normalized=physical/3.0",
                window_sampling="SeedSequence([seed,zero_based_epoch]); each trajectory once",
                anchor_min=0, anchor_max=350, validation_anchor=250,
                micro_batch=args.micro_batch, accumulation=accumulation,
                effective_batch=args.micro_batch * accumulation, drop_last=False,
                cache_observations=bool(args.cache_observations),
                execution_requested=args.execution, execution_actual="eager",
                execution_reason="native iterative solver has dynamic convergence control; retain original Torch operations",
                precision="float32", amp=False, matmul_tf32=False, cudnn_tf32=False,
                optimizer="torch.optim.Adam", learning_rate=0.0001, weight_decay=0.0001,
                scheduler="torch.optim.lr_scheduler.CosineAnnealingLR", scheduler_t_max=101,
                scheduler_eta_min=1e-9, scheduler_first_step_after_completed_epoch=17,
                loss="torch.nn.functional.mse_loss_mean_151_observed_times_including_initial",
                normalization="none_raw_observed_vorticity", teacher_forcing=False,
                selection="terminal_epoch_not_validation", checkpoint_interval=args.checkpoint_interval,
                initialization="random_from_seed_no_published_checkpoint")


def objective(model, initial, target, times):
    """Compute native mean MSE over [B,151,64,64] observed targets.

    initial is [B,64,64] and times contains physical seconds local to the
    window. Include time zero, so initial reconstruction contributes to the
    loss; the solver output must have exactly the target shape.
    """
    prediction = model(initial, times=times)
    if prediction.shape != target.shape:
        raise ValueError("ANIE output must match the selected observed [B,151,64,64] times")
    return F.mse_loss(prediction, target)


def train_epoch(model, bank, optimizer, config, epoch, device):
    """Visit each selected trajectory once, accumulating a weighted ANIE MSE.

    The sampler returns one anchor per trajectory. Evaluate 151 observed
    times per window, weight each microbatch mean by its fraction of the
    effective group, and apply Adam only after finite-gradient checks. This
    preserves tail weighting. Return processed trajectories, actual updates
    and the trajectory-weighted mean loss plus native solver-iteration counts.
    """
    order, anchors = epoch_indices(config["trajectories"], bank.training.shape[1], config["seed"], epoch)
    model.train()
    effective = config["micro_batch"] * config["accumulation"]
    total, processed, updates = 0.0, 0, 0
    solver_iterations = {}
    started = time.perf_counter()
    times = torch.as_tensor(config["training_local_times"], dtype=torch.float32, device=device)
    for first in range(0, len(order), effective):
        group = order[first:first + effective]
        optimizer.zero_grad(set_to_none=True)
        for micro in range(0, len(group), config["micro_batch"]):
            rows = group[micro:micro + config["micro_batch"]]
            initial, target = bank.batch(rows, anchors, device, indices=config["training_frame_indices"])
            loss = objective(model, initial, target, times)
            record_solver_iterations(model, solver_iterations)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite ANIE training loss")
            # Each native MSE is a microbatch mean. Weight by its trajectory
            # count to preserve an effective-batch mean, including the tail.
            (loss * (len(rows) / len(group))).backward()
            total += float(loss.detach()) * len(rows)
            processed += len(rows)
        if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None):
            raise FloatingPointError("nonfinite ANIE gradient before native Adam update")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("nonfinite ANIE weights after native Adam update")
        updates += 1
        if updates % 8 == 0:
            print(json.dumps(dict(status="epoch_progress", epoch=epoch + 1,
                                  optimizer_updates_this_epoch=updates,
                                  processed_trajectories=processed,
                                  trajectories_total=len(order),
                                  elapsed_seconds=time.perf_counter() - started,
                                  training_solver_iterations=solver_iterations), allow_nan=False), flush=True)
    return dict(training_mse=total / processed, processed_trajectories=processed,
                optimizer_updates_this_epoch=updates, training_solver_iterations=solver_iterations,
                sampled_start_min=int(anchors.min()), sampled_start_max=int(anchors.max()))


@torch.no_grad()
def validation_measurement(model, bank, config, device):
    """Measure 151-time MSE on fixed validation windows without gradients.

    Average by trajectory count, including a partial last batch. A nonfinite
    measurement is returned as an explicit failure with a None MSE; it does
    not select different training weights or alter sampling.
    """
    model.eval()
    rows, anchors = bank.validation_indices()
    total, processed = 0.0, 0
    solver_iterations = {}
    times = torch.as_tensor(config["training_local_times"], dtype=torch.float32, device=device)
    for first in range(0, len(rows), config["micro_batch"]):
        selected = rows[first:first + config["micro_batch"]]
        initial, target = bank.batch(selected, anchors, device, validation=True,
                                     indices=config["training_frame_indices"])
        loss = objective(model, initial, target, times)
        record_solver_iterations(model, solver_iterations)
        if not torch.isfinite(loss):
            return dict(validation_mse=None, validation_finite=False,
                        validation_trajectories=len(rows), validation_anchor=int(anchors[0]),
                        validation_solver_iterations=solver_iterations)
        total += float(loss) * len(selected)
        processed += len(selected)
    return dict(validation_mse=total / processed, validation_finite=True,
                validation_trajectories=processed, validation_anchor=int(anchors[0]),
                validation_solver_iterations=solver_iterations)


def source_hashes():
    """Record the source files involved in this ANIE run as descriptive provenance.

    These digests document implementation changes; they do not establish
    model compatibility or require old training files for inference.
    """
    files = ("training/train_anie_dense.py", "training/anie_data.py", "adapters/anie.py",
             "training/checkpoints.py", "training/anie_control.py", "src/gift/canonical_package.py", "src/gift/data_splits.py")
    return {relative: digest_file(source_file(ROOT, relative)) for relative in files}


def main(argv=None):
    """Plan, train or resume the 151-time ANIE protocol using its native predictor.

    Construct the model and optimizer/scheduler, then load their saved state
    before restoring Python/NumPy/Torch RNG. Commit completed epochs with
    history and actual update counts. A complete checkpoint can recover a
    missing terminal export without rerunning optimization or validation.
    """
    args = arguments(argv)
    config = configuration(args)
    data_root, path, source_root, output = checked_paths(args)
    from adapters.anie import ANIEConfig, ANIEPredictor, source_identity
    upstream = source_identity(source_root=source_root)
    model_config = asdict(ANIEConfig())
    with ObservationBank(data_root, path, formal=config["formal"],
                         trajectories=config["trajectories"], cache=args.cache_observations if args.run_training else False,
                         verify_full=False) as bank:
        if not args.run_training:
            print(json.dumps(dict(status="dry_run", configuration=config,
                                  model_configuration=model_config, data=bank.binding, upstream=upstream,
                                  outputs_written=0, training_performed=False), indent=2))
            return
        device = torch.device(args.device)
        configure_random(config["seed"])
        time_grid = dense_time_grid_identity(device)
        expected_updates = (config["trajectories"] + config["effective_batch"] - 1) // config["effective_batch"]
        print(json.dumps(dict(status="preflight_ready", full_input_checks_performed=bank.binding["provenance"]["full_input_checks_performed"],
                              epochs=config["epochs"], trajectories_per_epoch=config["trajectories"],
                              optimizer_updates_per_epoch=expected_updates,
                              optimizer_updates_total=expected_updates * config["epochs"],
                              execution="eager", training_time_points=151, prediction_time_points=151,
                              micro_batch=config["micro_batch"], accumulation=config["accumulation"],
                              time_grid=time_grid,
                              estimated_training_seconds=None,
                              timing_note="No invented runtime estimate; use the measured pre-training GPU probe and epoch records")), flush=True)
        identity = dict(kind="gift.anie-dense-training.v1", configuration=config,
                        model_configuration=model_config, upstream=upstream, data=bank.binding,
                        time_grid=time_grid,
                        sources=source_hashes(), numerical_profile=runtime_identity(), device=str(device))
        model = ANIEPredictor(config=ANIEConfig(), source_root=source_root).to(device=device, dtype=torch.float32)
        optimizer, scheduler = native_training_objects(model, config)
        store = CheckpointStore(output, identity, resume=args.resume)
        history, updates, completed = [], 0, 0
        if args.resume:
            saved = store.payload
            model.load_state_dict(saved["model_state_dict"], strict=True)
            restore_optimizer(optimizer, saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])
            history, updates, completed = saved["history"], saved["optimizer_updates"], saved["completed_epochs"]
            if saved["selection"] != "terminal_epoch_not_validation" or saved["formal"] != config["formal"]:
                raise ValueError("ANIE checkpoint selection/formal status differs")
            if (not 0 <= completed <= config["epochs"] or len(history) != completed
                    or any(row["epoch"] != i + 1 or row["processed_trajectories"] != config["trajectories"]
                           for i, row in enumerate(history))
                    or sum(row["optimizer_updates_this_epoch"] for row in history) != updates
                    or (saved["status"] == "complete") != (completed == config["epochs"])):
                raise ValueError("ANIE checkpoint epoch, update count or terminal status is inconsistent")
            store.restore_random_state()

        def payload(epoch, status):
            """Package the epoch cursor, native model/Adam/cosine state and committed history."""
            return dict(status=status, completed_epochs=epoch, optimizer_updates=updates,
                        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(), history=history,
                        training_cost=training_cost(history, updates),
                        selection="terminal_epoch_not_validation", formal=config["formal"])

        if not args.resume:
            store.save("epoch_0000", payload(0, "running"))
        if completed == config["epochs"]:
            export_terminal(output, model, config, model_config, identity, history, updates, True)
            print(json.dumps(dict(status="terminal_export_recovered", completed_epochs=completed, output=str(output))))
            return
        stop = min(config["epochs"], args.stop_after_epoch or config["epochs"])
        if completed >= stop:
            raise ValueError("no remaining ANIE epochs before requested stopping boundary")
        for epoch in range(completed, stop):
            started = time.perf_counter()
            used_lr = optimizer.param_groups[0]["lr"]
            record = train_epoch(model, bank, optimizer, config, epoch, device)
            updates += record["optimizer_updates_this_epoch"]
            # Validation monitors the procedure; it never changes the selected
            # weights, learning rate, architecture or future training windows.
            record.update(validation_measurement(model, bank, config, device))
            step_scheduler(scheduler, epoch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record.update(epoch=epoch + 1, optimizer_updates=updates,
                          learning_rate_used=float(used_lr), learning_rate_next=float(optimizer.param_groups[0]["lr"]),
                          epoch_seconds=time.perf_counter() - started, formal=config["formal"])
            history.append(record)
            status = "complete" if epoch + 1 == config["epochs"] else "running"
            save = (not config["formal"] or (epoch+1) % config["checkpoint_interval"] == 0
                    or epoch+1 in (stop, config["epochs"]))
            checkpoint = store.save(f"epoch_{epoch+1:04d}", payload(epoch+1, status)) if save else None
            print(json.dumps(dict(record, checkpoint=checkpoint.name if checkpoint else None,
                                  status=status), allow_nan=False), flush=True)
        if stop == config["epochs"]:
            export_terminal(output, model, config, model_config, identity, history, updates, args.resume)
        print(json.dumps(dict(status=status, completed_epochs=stop, output=str(output),
                              formal=config["formal"], training_cost=training_cost(history, updates)), allow_nan=False))


if __name__ == "__main__":
    main()
