"""Project experiment scheduling around external baseline algorithms.

Independent same-run epoch-boundary continuation, never published-weight resume.
No upstream architecture, optimizer implementation or training script is copied.
The four CLIs share IO/checkpoint handling, not one compulsory multi-model run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from adapters.models import build_model, source_record, fno_utilities, uno_components
from adapters import source_file
from gift.publication_metadata import public_metadata
from training.checkpoints import CheckpointStore, restore_optimizer, digest_file, runtime_identity, write_model_export
from training.budgets import PREDICTION_EPOCHS
from training.configuration import output_path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = "fno/fno1000_n64_t0_t10_dt0p02.h5"
TRAIN_SHA = "1322ad0a5be67bccab2a93c55a5a5748f792f80f3a839032bce888959d54e3ee"
PROTOCOLS = {
    "fno2d": dict(epochs=PREDICTION_EPOCHS, trajectories=1000, history=46, rollout=150, micro_batch=10,
                  accumulation=2, learning_rate=0.001, weight_decay=0.0001, seed=0),
    "fno3d": dict(epochs=PREDICTION_EPOCHS, trajectories=1000, history=46, rollout=150, micro_batch=5,
                  accumulation=2, learning_rate=0.001, weight_decay=0.0001, seed=0),
    "uno": dict(epochs=PREDICTION_EPOCHS, trajectories=1000, history=46, rollout=20, micro_batch=16,
                accumulation=1, learning_rate=0.001, weight_decay=0.00001, seed=0),
    "unet": dict(epochs=PREDICTION_EPOCHS, trajectories=1000, history=46, rollout=4, micro_batch=20,
                 accumulation=1, learning_rate=0.001, weight_decay=0.0001, seed=0),
}


def _arguments(model: str, argv=None):
    """Parse one baseline's IO, training budget and pause/resume options.

    Without --run-training the controller only validates a plan. A resume
    requires an existing output; fresh training requires an explicit output.
    """
    parser = argparse.ArgumentParser(description=f"Independent {model} training; default is read-only dry run",
                                     allow_abbrev=False)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue this output's own checkpoint only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-file", default=TRAIN_FILE, help="path relative to GIFT_DATA_ROOT")
    parser.add_argument("--data-profile", choices=("released", "regenerated", "canonical", "gaussian"), default="released",
                        help="canonical: numbered prediction package; released/regenerated: storage-ID input")
    parser.add_argument("--stop-after-epoch", type=int, help="operational pause; does not change total run budget")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="save every N completed epochs; default 10")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tiny", action="store_true", help="explicit NONFORMAL test budget, never a formal artifact")
    parser.add_argument("--tiny-epochs", type=int, default=2)
    parser.add_argument("--tiny-trajectories", type=int, default=2)
    parser.add_argument("--tiny-batch", type=int, default=1)
    parser.add_argument("--tiny-rollout", type=int)
    args = parser.parse_args(argv)
    if args.data_profile == "gaussian" and args.data_file == TRAIN_FILE:
        args.data_file = "s3_gaussian/trajectories.h5"
    if args.resume and not args.run_training:
        parser.error("--resume requires --run-training")
    if args.run_training and args.output is None:
        parser.error("--run-training requires a new explicit --output, or the same --output with --resume")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be positive")
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    return args


def _data_path(relative: str) -> tuple[Path, Path]:
    """Resolve a dataset path relative to GIFT_DATA_ROOT and return (root, file)."""
    configured = os.environ.get("GIFT_DATA_ROOT")
    if not configured:
        raise ValueError("set GIFT_DATA_ROOT to the external dataset directory")
    base = Path(configured).expanduser().resolve(strict=True)
    from gift.paths import data_path
    roles = {TRAIN_FILE: "four_vortex.fno_training_dt0p02",
             "s3_gaussian/trajectories.h5": "s3.trajectories"}
    path = (data_path(base, roles[relative], relative) if relative in roles
            else base / relative).resolve(strict=True)
    if base not in path.parents:
        raise ValueError("--data-file must stay inside GIFT_DATA_ROOT")
    return base, path


def configuration(model: str, args) -> dict:
    """Build model-specific scientific counts from defaults and explicit overrides.

    A microbatch is the resident forward/backward size; accumulation gives
    its update group size. Tiny settings mark a diagnostic artifact. FNO-3D
    needs enough future points for its unchanged temporal spectral modes.
    """
    config = dict(PROTOCOLS[model])
    config.update(model=model, formal=not args.tiny, scheduler_step=100, scheduler_gamma=0.5,
                  selection="terminal_epoch_not_validation", teacher_forcing=False, detach_rollout=False,
                  checkpoint_interval=args.checkpoint_interval, data_profile=args.data_profile)
    if args.tiny:
        config.update(epochs=args.tiny_epochs, trajectories=args.tiny_trajectories,
                      micro_batch=args.tiny_batch, accumulation=1,
                      rollout=(16 if model == "fno3d" else 1) if args.tiny_rollout is None else args.tiny_rollout)
    for key in ("epochs", "trajectories", "seed"):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if any(config[key] < 1 for key in ("epochs", "trajectories", "micro_batch", "rollout")):
        raise ValueError("training counts must be positive")
    if model == "fno3d" and config["rollout"] < 14:
        raise ValueError("FNO-3D's unchanged modes3=8 needs at least 14 future points")
    effective = config["micro_batch"] * config["accumulation"]
    return config


def _validate_data(path: Path, config: dict, *, hash_bytes: bool, data_base: Path | None = None) -> dict:
    """Inspect training IDs, DT=0.02 time axis and float32 [trajectory,time,64,64] layout.

    Require enough selected trajectories and history-plus-rollout frames.
    Return shape/size provenance; sampled values are checked by _batch.
    hash_bytes and data_base are retained caller arguments, not instructions
    to rehash a large dataset or require a package receipt.
    """
    with h5py.File(path, "r") as handle:
        field = handle["training/vorticity"]
        shape = tuple(field.shape)
        if len(shape) != 4 or shape[0] < config["trajectories"] or shape[2:] != (64, 64):
            raise ValueError("expected at least the selected trajectories in [trajectory,time,64,64]")
        if field.dtype != np.dtype("float32") or shape[1] < config["history"] + config["rollout"]:
            raise ValueError("field dtype or available frames differ")
        ids = np.asarray(handle["training/trajectory_index"][:])
        if len(ids) != shape[0] or len(set(map(int, ids))) != len(ids):
            raise ValueError("training identities must be unique and match the population")
        times = np.asarray(handle["training/time"][:])
        if len(times) != shape[1] or not np.allclose(times, np.arange(shape[1]) * .02, rtol=0, atol=2e-12):
            raise ValueError("training time axis differs")
    return dict(shape=list(shape), dtype="float32", bytes=path.stat().st_size,
                sha256=None, sha256_verified=False,
                provenance=dict(profile=config.get("data_profile", "released"),
                                published_dataset_identity=False, full_input_checks_performed=False,
                                numerical_reproduction_verified=False))


def _configure_profile(model: str, seed: int) -> None:
    """Seed initialization and set the baseline's documented kernel/precision choices.

    U-Net and U-NO use deterministic cuDNN selection; the FNO families leave
    ambient TF32 settings in place and record them as provenance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if model in ("uno", "unet"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=model == "uno")
    if model == "unet":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    # FNO and UNO retain the original default/ambient TF32 flags. Record them;
    # the historical metadata do not establish a stronger universal profile.


def epoch_schedule(config: dict, epoch: int, frames: int):
    """Return trajectory order and one history-window start per selected row.

    Starts are indexed by trajectory row. U-NO samples starts independently
    by seed and epoch; the other protocols visit a deterministic stride-five
    anchor cycle and use their established per-epoch ordering.
    """
    count = config["trajectories"]
    possible = frames - config["history"] - config["rollout"] + 1
    if config["model"] == "uno":
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch]))
        return rng.permutation(count), rng.integers(0, possible, size=count, dtype=np.int64)
    anchors = np.arange(0, possible, 5, dtype=np.int64)
    index = np.random.default_rng(2026081401).permutation(len(anchors))
    starts = anchors[index[(epoch + 37 * np.arange(count)) % len(anchors)]]
    order = (np.random.default_rng(2026090101 + epoch).permutation(count)
             if config["model"] == "unet" else torch.randperm(count).cpu().numpy())
    return order, starts


def _batch(field, rows, starts, config, device):
    """Read selected history/future windows and return tensors in native layouts.

    Input HDF5 axes are [trajectory,frame,64,64]. U-Net returns channel-first
    [B,46,64,64] history; FNO/U-NO return [B,64,64,46]. Targets use the same
    layout with rollout frames, and all sampled values must be finite.
    """
    length = 46 + config["rollout"]
    values = np.stack([np.asarray(field[int(row), int(starts[row]):int(starts[row]) + length],
                                  dtype=np.float32) for row in rows])
    if not np.isfinite(values).all():
        raise FloatingPointError("sampled training window is nonfinite")
    tensor = torch.from_numpy(values)
    if config["model"] == "unet":
        context, target = tensor[:, :46], tensor[:, 46:]
    else:
        context = tensor[:, :46].permute(0, 2, 3, 1).contiguous()
        target = tensor[:, 46:].permute(0, 2, 3, 1).contiguous()
    return context.to(device), target.to(device)


def _fit_normalization(field, config: dict) -> dict:
    """Fit FNO-3D's position/time mean and sample deviation on training windows.

    Reduction includes only selected training trajectories and stride-five
    anchors, keeping the original row-major order. Store [64,64,time] input
    and output statistics separately; other model families return kind=none.
    """
    if config["model"] != "fno3d":
        return {"kind": "none"}
    # Preserve the original row-major all-training-window reduction order.
    series = np.asarray(field[:config["trajectories"]], dtype=np.float32)
    anchors = np.arange(0, series.shape[1] - 46 - config["rollout"] + 1, 5)
    means, stds = [], []
    for offset in range(46 + config["rollout"]):
        sample = torch.from_numpy(np.ascontiguousarray(series[:, anchors + offset]).reshape(-1, 64, 64))
        means.append(sample.mean(0))
        stds.append(sample.std(0, correction=1))
    mean, std = torch.stack(means, -1), torch.stack(stds, -1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise FloatingPointError("invalid training-only position normalizers")
    return dict(kind="paper_UnitGaussianNormalizer", eps=1e-5,
                input_mean=mean[..., :46].contiguous(), input_std=std[..., :46].contiguous(),
                output_mean=mean[..., 46:].contiguous(), output_std=std[..., 46:].contiguous())


def _normalizer_objects(saved: dict, device):
    """Rebuild native input/output normalizers from saved statistics on device.

    Bypass __init__ so resume uses the original training fit rather than
    refitting statistics from a potentially different window order.
    """
    native, _ = fno_utilities()
    result = []
    for prefix in ("input", "output"):
        instance = native.__new__(native)
        instance.mean = saved[f"{prefix}_mean"].to(device)
        instance.std = saved[f"{prefix}_std"].to(device)
        instance.eps = saved["eps"]
        result.append(instance)
    return tuple(result)


def _fno3d_input(context, future: int):
    # Project input-layout glue; the model's external spectral code is untouched.
    """Append normalized coordinates and repeat context over the future axis.

    context is [B,nx,ny,history]; return [B,nx,ny,future,history+3]. Spatial
    coordinates span [0,1], and future coordinates exclude the initial point.
    """
    batch, nx, ny, history = context.shape
    axes = [torch.tensor(np.linspace(0, 1, size), dtype=torch.float32) for size in (nx, ny)]
    axes.append(torch.tensor(np.linspace(0, 1, future + 1)[1:], dtype=torch.float32))
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).to(context.device)
    repeated = context.reshape(batch, nx, ny, 1, history).repeat(1, 1, 1, future, 1)
    return torch.cat((coordinates.unsqueeze(0).expand(batch, -1, -1, -1, -1), repeated), -1)


def objective(model, context, target, family: str, loss_function=None, normalizers=None):
    """Return differentiable training loss and detached full-window reporting loss.

    FNO-3D predicts its full future axis with saved normalizers. Other models
    roll forward using their own predictions, without detaching the context.
    Native losses retain their sum convention across trajectories/horizons;
    microbatch accumulation in main therefore sums, rather than averages, them.
    """
    if family == "fno3d":
        input_norm, output_norm = normalizers
        encoded_target = output_norm.encode(target)
        prediction = model(_fno3d_input(input_norm.encode(context), target.shape[-1])).reshape_as(target)
        prediction, decoded_target = output_norm.decode(prediction), output_norm.decode(encoded_target)
        loss = loss_function(prediction.reshape(len(target), -1), decoded_target.reshape(len(target), -1))
        return loss, loss.detach()
    window, predictions, losses = context, [], []
    steps = target.shape[1] if family == "unet" else target.shape[-1]
    full_num = torch.zeros(len(target), device=target.device)
    full_den = torch.zeros_like(full_num)
    for step in range(steps):
        prediction = (activation_checkpoint(model, window, use_reentrant=False)
                      if family == "uno" else model(window))
        predictions.append(prediction)
        if family == "unet":
            truth = target[:, step:step + 1]
            numerator = torch.linalg.vector_norm((prediction - truth).reshape(len(target), -1), dim=1)
            denominator = torch.linalg.vector_norm(truth.reshape(len(target), -1), dim=1)
            losses.append((numerator / denominator.clamp_min(torch.finfo(truth.dtype).tiny)).sum())
            full_num += (prediction.detach() - truth.detach()).double().square().flatten(1).sum(1).to(target.dtype)
            full_den += truth.detach().double().square().flatten(1).sum(1).to(target.dtype)
            window = torch.cat((window[:, 1:], prediction), 1)
        else:
            if family == "uno":
                truth = target[..., step:step + 1]
                if not (torch.norm(truth.reshape(len(target), -1), p=2, dim=1) > 0).all():
                    raise FloatingPointError("zero-norm U-NO target")
                losses.append(loss_function(prediction, truth))
            window = torch.cat((window[..., 1:], prediction), -1)
    if family == "fno2d":
        losses = [loss_function(prediction.reshape(len(target), -1),
                               target[..., index:index + 1].reshape(len(target), -1))
                  for index, prediction in enumerate(predictions)]
    loss = context.new_zeros(()) if family == "uno" else losses[0]
    for item in losses if family == "uno" else losses[1:]:
        loss = loss + item
    if family == "unet":
        full = (full_num.sqrt() / full_den.sqrt().clamp_min(torch.finfo(target.dtype).tiny)).sum()
    elif family == "fno2d":
        full = loss_function(torch.cat(predictions, -1).reshape(len(target), -1), target.reshape(len(target), -1)).detach()
    else:
        full = loss.detach()
    return loss, full


def training_cost(history, optimizer_updates):
    """Sum committed epoch timers; do not count paused time as GPU training."""
    return dict(completed_epochs=len(history), optimizer_updates=int(optimizer_updates),
                committed_epoch_seconds=sum(float(row["epoch_seconds"]) for row in history),
                timing_scope="epoch input loading, forward/backward, updates and CUDA synchronization",
                timing_excludes="setup, checkpoint IO, paused time and discarded uncommitted work")


def _export_terminal(output, network, normalization, config, identity, resumed, cost=None):
    """Publish final CPU weights with the fitted normalization needed for inference.

    The model/configuration and run pointer are descriptive export metadata.
    Adam, scheduler and RNG state stay in the separate resume journal.
    A completed checkpoint may call this again to finish interrupted export.
    """
    attempt = json.loads((output / "ATTEMPT.json").read_text(encoding="utf-8"))
    latest = json.loads((output / "LATEST.json").read_text(encoding="utf-8"))
    artifact = dict(schema="gift.independent-baseline-weights.v1", status="complete", method=config["model"],
                    model_state_dict={key: value.detach().cpu() for key, value in network.state_dict().items()},
                    normalization=normalization, formal_configuration=config,
                    run_provenance=dict(run_id=attempt["run_id"], identity=public_metadata(identity),
                                        terminal_boundary=public_metadata(latest)),
                    artifact_role="fresh_terminal" if config["formal"] else "test_only",
                    fresh_training=True, same_run_resume_used=resumed,
                    initialization="random_from_seed_no_published_checkpoint",
                    terminal_epoch=config["epochs"])
    if cost is not None:
        artifact["training_cost"] = cost
    write_model_export(output / "model.pt", artifact)


def save_boundary(epoch: int, config: dict, stop: int) -> bool:
    """Select interval, explicit pause and terminal epochs for durable checkpointing."""
    return (not config["formal"] or epoch % config["checkpoint_interval"] == 0
            or epoch == stop or epoch == config["epochs"])


def main(model_name: str, argv=None) -> None:
    """Validate, train or resume one native baseline and export its final weights.

    Restore model, optimizer, scheduler and normalization before restoring RNG.
    Each effective update group may contain several microbatches, including a
    smaller tail. A completed checkpoint can repair missing terminal export
    without executing another update; validation does not select a best epoch.
    """
    args = _arguments(model_name, argv)
    config = configuration(model_name, args)
    data_base, data_file = _data_path(args.data_file)
    source = source_record(model_name)
    data = _validate_data(data_file, config, hash_bytes=args.run_training, data_base=data_base)
    if not args.run_training:
        print(json.dumps(dict(status="dry_run", configuration=config, data=data,
                              upstream=source, outputs_written=0, training_performed=False), indent=2))
        return
    device = torch.device(args.device)
    output = args.output.expanduser().resolve()
    external = Path(os.environ["GIFT_EXTERNAL_ROOT"]).resolve()
    output = output_path(output, ROOT, data_base, external)
    _configure_profile(model_name, config["seed"])
    source.pop("directory")
    identity = dict(kind="gift.independent-baseline-training.v1", configuration=config,
                    data=data, data_file=args.data_file, upstream=source,
                    sources={relative: digest_file(source_file(ROOT, relative)) for relative in (
                        "training/baseline_control.py", "training/checkpoints.py", "training/budgets.py",
                        f"training/train_{model_name}.py", "adapters/models.py", "external_sources.json")},
                    numerical_profile=runtime_identity(), device=str(device))
    if model_name == "unet":
        identity["sources"]["training/unet_environment.py"] = digest_file(ROOT / "training/unet_environment.py")
    if args.data_profile == "regenerated":
        identity["sources"].update({relative: digest_file(ROOT / relative) for relative in (
            "scripts/assemble_generated_data.py", "scripts/generate_data.py", "src/even_full_spectrum_ns.py")})
    elif args.data_profile == "canonical":
        identity["sources"].update({relative: digest_file(ROOT / relative) for relative in (
            "src/gift/canonical_package.py", "src/gift/data_splits.py")})
    elif args.data_profile == "gaussian":
        identity["sources"].update({relative: digest_file(ROOT / relative) for relative in (
            "src/gift/gaussian_package.py", "src/gift/gaussian_initial.py",
            "src/gift/prediction_cohorts.py", "src/gift/canonical_package.py", "src/gift/data_splits.py")})
    network = build_model(model_name, str(device))
    if model_name == "uno":
        _, optimizer_type, criterion_type = uno_components()
    else:
        optimizer_type = torch.optim.Adam
        criterion_type = fno_utilities()[1] if model_name.startswith("fno") else None
    optimizer = optimizer_type(network.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = criterion_type(size_average=False) if criterion_type else None
    store = CheckpointStore(output, identity, resume=args.resume)
    history, updates, start_epoch = [], 0, 0
    resumed = bool(args.resume)
    with h5py.File(data_file, "r") as handle:
        field = handle["training/vorticity"]
        if args.resume:
            saved = store.payload
            network.load_state_dict(saved["model_state_dict"], strict=True)
            restore_optimizer(optimizer, saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])
            history, updates, start_epoch = saved["history"], saved["optimizer_updates"], saved["completed_epochs"]
            normalization = saved["normalization"]
            if not 0 <= start_epoch <= config["epochs"] or (saved["status"] == "complete") != (start_epoch == config["epochs"]):
                raise ValueError("Checkpoint epoch and completion status disagree")
            if saved["status"] == "complete":
                _export_terminal(output, network, normalization, config, identity, True,
                                 cost=training_cost(history, updates))
                print(json.dumps(dict(status="complete", completed_epochs=start_epoch, output=str(output))))
                return
        else:
            normalization = _fit_normalization(field, config)
        norms = _normalizer_objects(normalization, device) if model_name == "fno3d" else None

        def payload(epoch, status):
            """Capture the complete epoch boundary: weights, optimizer, schedule and progress."""
            return dict(status=status, model_state_dict=network.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                        normalization=normalization, completed_epochs=epoch, optimizer_updates=updates,
                        history=history, training_cost=training_cost(history, updates),
                        selection="terminal_epoch_not_validation", formal=not args.tiny)

        if args.resume:
            store.restore_random_state()
        else:
            store.save("epoch_0000", payload(0, "running"))
        stop = min(config["epochs"], args.stop_after_epoch or config["epochs"])
        if start_epoch >= stop:
            raise ValueError("no remaining epochs before requested stopping boundary")
        effective = config["micro_batch"] * config["accumulation"]
        for epoch in range(start_epoch, stop):
            started = time.perf_counter()
            order, starts = epoch_schedule(config, epoch, field.shape[1])
            network.train()
            backward_total = full_total = 0.0
            processed = 0
            used_lr = optimizer.param_groups[0]["lr"]
            for group_start in range(0, len(order), effective):
                group = order[group_start:group_start + effective]
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0, len(group), config["micro_batch"]):
                    rows = group[micro:micro + config["micro_batch"]]
                    context, target = _batch(field, rows, starts, config, device)
                    loss, full = objective(network, context, target, model_name, criterion, norms)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    backward_total += float(loss.detach())
                    full_total += float(full.detach())
                    processed += len(rows)
                if model_name == "uno":
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
                else:
                    if any(not torch.isfinite(parameter.grad).all() for parameter in network.parameters()
                           if parameter.grad is not None):
                        raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                updates += 1
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = dict(epoch=epoch + 1, optimizer_updates=updates, processed_trajectories=processed,
                          backward_loss_mean_per_trajectory=backward_total / processed,
                          full_loss_mean_per_trajectory=full_total / processed,
                          sampled_start_min=int(starts.min()), sampled_start_max=int(starts.max()),
                          learning_rate_used=float(used_lr), learning_rate_next=float(scheduler.get_last_lr()[0]),
                          epoch_seconds=time.perf_counter() - started, formal=not args.tiny)
            history.append(record)
            status = "complete" if epoch + 1 == config["epochs"] else "running"
            checkpoint = (store.save(f"epoch_{epoch + 1:04d}", payload(epoch + 1, status))
                          if save_boundary(epoch + 1, config, stop) else None)
            print(json.dumps(dict(record, checkpoint=checkpoint.name if checkpoint else None, status=status), allow_nan=False), flush=True)
    if status == "complete":
        _export_terminal(output, network, normalization, config, identity, resumed,
                         cost=training_cost(history, updates))
    print(json.dumps(dict(status=status, completed_epochs=stop, output=str(output),
                          formal=not args.tiny, training_cost=training_cost(history, updates),
                          published_numerical_reproduction_verified=False)))
