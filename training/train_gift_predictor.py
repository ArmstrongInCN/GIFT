"""Train one full-data GIFT branch against its freshly trained frozen generator.

The four stages contain 400, 50, 25 and 25 trajectory epochs. Each trajectory
contributes one observed window per epoch. The mathematical derivative and
rollout losses are reused unchanged from the project's GIFT training code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import uuid
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gift import FixedBandwidthGridGenerator, load_generator
from experiments.formal._shared.high_frequency import HighFrequencyBranch, HighFrequencyConfig
from experiments.formal.train_gift_branches import (
    _compute_low_rhs, _project, _square_mask, _training_scales, _validate_derivative,
    _validate_rollout, _short_epoch, _long_epoch, _snapshot, _write_checkpoint,
)
from training.checkpoints import digest_file, state_digest, restore_optimizer, write_json_export
from training.gift_prediction_control import (
    ROOT, PredictionTrainingConfig, configure_determinism, phase_at, source_identity,
    _open_run, synchronize, write_json_new, validate_phase_boundary,
)
from training.gift_prediction_data import ObservationBank
from training.gift_execution import EXECUTION_CHOICES, EXECUTION_HELP, resolve_execution

# Default seeds used by the published experiments; other seeds are allowed.
SEEDS = (20260820, 20260821, 20260822)


def branch_sources():
    """Extend generator source provenance with branch and continuation implementations."""
    value = source_identity()
    for name in ("training/train_gift_predictor.py", "experiments/formal/train_gift_branches.py",
                 "training/gift_continuation.py",
                 "experiments/formal/_shared/common.py", "experiments/formal/_shared/high_frequency.py",
                 "experiments/formal/_shared/gift_runtime.py", "training/gift_data.py",
                 "src/gift/paths.py"):
        if name not in value:
            value[name] = digest_file(ROOT / name)
    return value


def qualify_generator(path, bank, config):
    """Load usable generator weights; its branch-independent training metadata is descriptive."""
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if value.get("format_version") not in (2, 3) or not value.get("model_state_dict"):
        raise ValueError("Incomplete generator weights")
    if value.get("model_configuration", {}).get("cutoff") != config.cutoff:
        raise ValueError("Generator cutoff differs from the branch support")
    return value


def derivative_loader(bank, seed, epoch, frozen, device, *, validation=False, batch_size=16):
    """Prepare observed high-band derivatives and frozen low-band RHS for one epoch.

    Return an ordered DataLoader of (state,target,low_rhs) and its underlying
    tensors. Q21 complements the fixed cutoff-21 low support; no additional
    shuffle or random state is consumed after trajectory-window sampling.
    """
    states, targets = bank.pairs(seed, epoch, validation=validation)
    # Project the derivative target onto the complementary high band (modes
    # outside the Q21 low-frequency cutoff) and precompute the frozen low RHS.
    targets = torch.from_numpy(_project(targets.numpy(), ~_square_mask(21)))
    low = torch.from_numpy(_compute_low_rhs(frozen, states.numpy(), device))
    # The sampler already permuted trajectory rows; a second shuffle would
    # consume additional RNG state and make the epoch harder to audit.
    return DataLoader(TensorDataset(states, targets, low), batch_size=batch_size,
                      shuffle=False, num_workers=0), (states, targets, low)


def sequence_loader(bank, seed, epoch, *, validation=False, batch_size=5):
    """Build an ordered loader of sampled [B,8,y,x] observed rollout sequences."""
    value = bank.sequences(seed, epoch, validation=validation)
    return DataLoader(TensorDataset(torch.from_numpy(value)), batch_size=batch_size,
                      shuffle=False, num_workers=0)


def derivative_epoch(model, loader, optimizer, device, scale):
    """Fit Q21 derivative error, normalized by high-RHS scale, and return its weighted mean.

    Clip branch gradients at 1.0 before AdamW; a nonfinite loss or gradient
    raises before the update. Each loader row represents one trajectory window.
    """
    total, count = 0.0, 0
    model.train()
    for state, target, frozen in loader:
        state, target, frozen = (value.to(device) for value in (state, target, frozen))
        optimizer.zero_grad(set_to_none=True)
        loss = ((model.forward_with_generator_output(state, frozen) - target) / scale).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("branch derivative loss is nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(state)
        count += len(state)
    return total / count


def run_branch(dataset, low_model, output, *, seed, device="cuda", resume=False,
               config=PredictionTrainingConfig(), checkpoint_interval=10,
               stop_after_epoch=None, log_interval=10, execution="auto",
               continue_from=None, transition_record=None):
    """Train one high-frequency branch with a fixed generator through four phases.

    Prepare scales from training observations, then fit derivative, short
    and long rollouts in order. An optimizer belongs to a phase: restore it
    only when resuming inside that phase, and restore RNG after setup.
    Epoch checkpoints include the cumulative update count and history. Resume
    of a completed boundary can finish export without retraining; continuation
    uses a separate journal while retaining the parent boundary.
    """
    config.validate()
    if config.cutoff != 21:
        raise ValueError("The current high-frequency branch supports cutoff=21")
    if checkpoint_interval < 1 or log_interval < 1:
        raise ValueError("invalid checkpoint/log interval")
    total_epochs = sum(config.branch_phases)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= total_epochs:
        raise ValueError("stop boundary must be inside the configured budget")
    device = torch.device(device)
    execution = resolve_execution(execution, device, resume=resume,
                                  checkpoint_directory=Path(output)/"checkpoints")
    configure_determinism(seed, strict=False)
    bank = ObservationBank(dataset, fixture=config.fixture_only)
    payload = qualify_generator(low_model, bank, config)
    generator_weights = state_digest(payload["model_state_dict"])
    generator = load_generator(payload, device=device).eval().requires_grad_(False)
    frozen = FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    _, support = derivative_loader(bank, seed, 0, frozen, device, batch_size=config.batch_size)
    scales = _training_scales(*(value.numpy() for value in support))
    del support
    model = HighFrequencyBranch(HighFrequencyConfig(), state_scale=scales[0], high_scale=scales[1],
                                low_rhs_scale=scales[2], high_rhs_scale=scales[3],
                                frozen_generator=frozen).to(device)
    identity = {"role": "gift_prediction_branch", "seed": seed, "configuration": asdict(config),
                "data": bank.binding, "generator_weights_sha256": generator_weights, "scales": list(scales),
                "sources": branch_sources(), "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "selection": "terminal_epoch", "budget_unit": "trajectory_epoch"}
    identity["execution"] = execution if device.type == "cuda" else "eager"
    from training.gift_continuation import open_branch_run
    output, store = open_branch_run(output, identity, resume=resume,
                                   continue_from=continue_from, transition_record=transition_record)
    identity = store.identity
    saved = store.payload
    validate_phase_boundary(saved, config.branch_phases)
    if saved is not None and saved.get("completed") and (output / "COMPLETE.json").exists():
        return json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
    completed = 0 if saved is None else saved["epoch"]
    if not 0 <= completed <= total_epochs or (stop_after_epoch is not None and stop_after_epoch <= completed):
        raise ValueError("invalid saved/stop epoch boundary")
    history = [] if saved is None else saved["history"]
    updates = 0 if saved is None else saved["optimizer_updates"]
    elapsed = 0.0 if saved is None else saved["committed_seconds"]
    valid_derivative, _ = derivative_loader(bank, seed, 0, frozen, device, validation=True)
    valid_rollout = sequence_loader(bank, seed, 0, validation=True, batch_size=10)
    optimizer = scheduler = None
    engine = None
    previous_phase = None
    # Phases run derivative fit, short rollout, then two long-rollout stages with
    # descending learning rates; the derivative stage warms the band before the
    # costlier rollout horizons train it.
    names = ("derivative", "short_rollout", "long_rollout_1", "long_rollout_2")
    for epoch in range(completed + 1, total_epochs + 1):
        phase, phase_epoch = phase_at(epoch, config.branch_phases)
        if phase != previous_phase:
            lr = (0.0015, 0.0002, 0.00008, 0.00008)[phase]
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
            scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config.branch_phases[0], eta_min=0.0001) if phase == 0 else None)
            if saved is not None and epoch == completed + 1 and saved["phase"] == phase:
                restore_optimizer(optimizer, saved["optimizer"])
                if scheduler is not None:
                    scheduler.load_state_dict(saved["scheduler"])
            previous_phase = phase
            if execution == "cuda-graph":
                from training.gift_acceleration import TrainingEngine
                engine = TrainingEngine(model, kind=("derivative", "short", "long", "long")[phase],
                                        frozen=frozen, scale=scales[3], backend=execution)
        if saved is not None and epoch == completed + 1:
            store.restore_random_state()
        synchronize(device)
        started = time.perf_counter()
        if phase == 0:
            loader, _ = derivative_loader(bank, seed, epoch, frozen, device, batch_size=config.batch_size)
            loss = (engine.epoch(loader, optimizer, device) if engine is not None
                    else derivative_epoch(model, loader, optimizer, device, scales[3]))
            scheduler.step()
        else:
            loader = sequence_loader(bank, seed, epoch, batch_size=config.rollout_batch_size)
            loss = (engine.epoch(loader, optimizer, device) if engine is not None else
                    (_short_epoch if phase == 1 else _long_epoch)(model, frozen, loader, optimizer, device))
        if not math.isfinite(loss) or any(not bool(torch.isfinite(x).all()) for x in model.state_dict().values()):
            raise FloatingPointError("branch update produced nonfinite loss/parameters")
        updates += len(loader)
        validation = None
        if epoch == 1 or epoch % log_interval == 0 or phase_epoch == config.branch_phases[phase]:
            validation = (_validate_derivative(model, valid_derivative, device, scales[3]) if phase == 0
                          else _validate_rollout(model, frozen, valid_rollout, device))
            if any(isinstance(value, float) and not math.isfinite(value) for value in validation.values()):
                raise FloatingPointError("branch validation produced nonfinite values")
        synchronize(device)
        seconds = time.perf_counter() - started
        elapsed += seconds
        row = {"epoch": epoch, "phase": names[phase], "phase_epoch": phase_epoch,
               "trajectories_visited": len(bank.training), "optimizer_updates": len(loader),
               "cumulative_optimizer_updates": updates, "training_normalized_mse": loss,
               "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": seconds}
        history.append(row)
        if validation is not None:
            print(json.dumps({"seed": seed, **row}, allow_nan=False), flush=True)
        final = epoch == total_epochs
        if epoch % checkpoint_interval == 0 or phase_epoch == config.branch_phases[phase] or epoch == stop_after_epoch:
            store.save(f"epoch_{epoch:04d}", {
                "model": _snapshot(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch, "phase": phase, "optimizer_updates": updates,
                "history": history, "committed_seconds": elapsed, "completed": final,
            })
        if epoch == stop_after_epoch and not final:
            return {"status": "paused_at_committed_epoch", "epoch": epoch, "output": str(output)}
    # The derivative phase batches 16 trajectories; rollout phases batch 5, so
    # the per-epoch update count differs between stage groups.
    expected = (config.branch_phases[0] * math.ceil(len(bank.training) / config.batch_size)
                + sum(config.branch_phases[1:]) * math.ceil(len(bank.training) / config.rollout_batch_size))
    if updates != expected or len(history) != total_epochs:
        raise RuntimeError("completed branch budget differs")
    cost = {"epochs": total_epochs, "phase_epochs": list(config.branch_phases),
            "optimizer_updates": updates, "committed_epoch_seconds": elapsed,
            "shared_generator_epochs": sum(config.generator_phases),
            "shared_generator_sha256": generator_weights,
            "timing_scope": "sampling, frozen RHS preparation, training and scheduled validation; excludes initial setup, checkpoint writes and paused time"}
    binding = dict(bank.binding, profile="full_data_prediction", configuration=asdict(config),
                   selection="terminal_epoch", pretrained_branch_loaded=False, sources=identity["sources"],
                   execution=identity["execution"])
    if "continuation" in identity:
        binding["continuation"] = identity["continuation"]
    export_path = output / "model.pt"
    if export_path.exists():
        existing = torch.load(export_path, map_location="cpu", weights_only=True)
        state = existing.get("model_state", existing.get("state", {}))
        if set(state) != set(model.state_dict()) or any(state[k].dtype != v.dtype or not torch.equal(state[k], v.cpu()) for k, v in model.state_dict().items()):
            raise ValueError("Existing terminal branch differs from the completed checkpoint")
    else:
        temporary = export_path.with_name(export_path.name + "." + uuid.uuid4().hex + ".tmp")
        _write_checkpoint(temporary, model, seed=seed, phase="terminal_epoch", epoch=total_epochs,
                      selection_metric=history[-1]["validation"]["lead_1_full_relative_l2"],
                      generator_weights_sha256=generator_weights,
                      training_data=binding, training_cost=cost)
        os.link(temporary, export_path)
        temporary.unlink()
    write_json_export(output / "history.json", history)
    result = {"status": "complete", "role": identity["role"], "seed": seed,
              "configuration": asdict(config), "training_data": bank.binding, "training_cost": cost,
              "model_sha256": digest_file(output / "model.pt"), "runtime": store.runtime,
              "selection": "terminal_epoch", "sources": identity["sources"]}
    if "continuation" in identity:
        result["continuation"] = identity["continuation"]
    write_json_export(output / "COMPLETE.json", result)
    return result


def main(argv=None):
    """Parse one branch run, its generator input and optional resume/continuation path."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    action.add_argument("--resume", action="store_true")
    action.add_argument("--continue-from", type=Path,
                        help="Continue this unfinished run in a new journal.")
    parser.add_argument("--transition-record", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--low-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--execution", choices=EXECUTION_CHOICES, default="auto", help=EXECUTION_HELP)
    args = parser.parse_args(argv)
    if args.dry_run:
        result = {"configuration": asdict(PredictionTrainingConfig()), "seed": args.seed,
                  "dataset": str(args.dataset), "generator": str(args.low_model),
                  "output": str(args.output), "execution": args.execution,
                  "selection": "terminal_epoch", "writes": 0}
    else:
        result = run_branch(args.dataset, args.low_model, args.output, seed=args.seed,
                            device=args.device, resume=args.resume,
                            checkpoint_interval=args.checkpoint_interval, stop_after_epoch=args.stop_after_epoch,
                            execution=args.execution, continue_from=args.continue_from,
                            transition_record=args.transition_record)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
