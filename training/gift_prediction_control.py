"""Independent full-data GIFT training with real trajectory-epoch budgets.

The generator and branch each receive 500 epochs, reported separately. Native
GIFT mathematical components are reused. Epoch sampling and terminal checkpoint
selection implement the prediction comparison protocol; M1 is not affected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from gift import GIFTGenerator, fit_affine_minimum_norm, identifiability_grid
from training.checkpoints import CheckpointStore, digest_file
from training.gift_prediction_data import ObservationBank
from experiments.formal._shared.gift_generator_training import (
    clone_state, configure_determinism, configure_variable_projection_parameters,
    evaluate, write_json_new,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PredictionTrainingConfig:
    """Prespecified budgets; fixture configurations cannot qualify as formal runs."""

    generator_phases: tuple[int, int] = (250, 250)
    branch_phases: tuple[int, int, int, int] = (400, 50, 25, 25)
    generator_learning_rates: tuple[float, float] = (0.01, 0.003)
    generator_seed: int = 2026072301
    batch_size: int = 16
    rollout_batch_size: int = 5
    cutoff: int = 21
    rank: int = 8
    fixture_only: bool = False

    def validate(self):
        if (len(self.generator_phases) != 2 or len(self.branch_phases) != 4
                or min(*self.generator_phases, *self.branch_phases, self.batch_size,
                       self.rollout_batch_size, self.cutoff, self.rank) < 1
                or len(self.generator_learning_rates) != 2
                or min(self.generator_learning_rates) <= 0):
            raise ValueError("invalid GIFT epoch training configuration")
        if not self.fixture_only and self != PredictionTrainingConfig():
            raise ValueError("formal GIFT prediction settings must match the prespecified protocol")


def phase_at(epoch: int, counts: tuple[int, ...]) -> tuple[int, int]:
    """Return zero-based phase and one-based phase epoch for a global epoch."""
    if epoch < 1:
        raise ValueError("epoch must be positive")
    preceding = 0
    for phase, count in enumerate(counts):
        if epoch <= preceding + count:
            return phase, epoch - preceding
        preceding += count
    raise ValueError("epoch exceeds budget")


def source_identity():
    """Bind the numerical implementation, not unrelated reports or figures."""
    files = (
        "training/gift_prediction_control.py", "training/gift_prediction_data.py",
        "training/train_gift_generator.py",
        "training/checkpoints.py", "src/gift/data_splits.py",
        "src/gift/model.py", "src/gift/identifiability.py", "src/gift/identified.py",
        "src/gift/__init__.py",
        "experiments/formal/_shared/gift_generator_training.py",
        "training/gift_acceleration.py", "src/gift/execution.py",
        "experiments/formal/train_gift_branches.py",
        "experiments/formal/_shared/high_frequency.py",
        "experiments/formal/_shared/gift_runtime.py",
        "experiments/formal/_shared/common.py", "training/gift_data.py", "src/gift/paths.py",
    )
    return {name: digest_file(ROOT / name) for name in files}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _open_run(output, identity, resume):
    output = Path(output).resolve()
    if output == ROOT or output == ROOT / "artifacts" or ROOT / "artifacts" in output.parents:
        raise ValueError("training cannot overwrite the project or released weights")
    if resume:
        if any((output / name).exists() for name in ("model.pt", "COMPLETE.json")):
            raise FileExistsError("completed training output cannot be overwritten")
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output, CheckpointStore(output / "checkpoints", identity, resume=resume)


def run_generator(dataset, output, *, device="cuda", resume=False,
                  config=PredictionTrainingConfig(), checkpoint_interval=10,
                  stop_after_epoch=None, log_interval=10, execution="eager"):
    """Train native generator factors and affine tables without pretrained weights."""
    config.validate()
    if checkpoint_interval < 1 or log_interval < 1:
        raise ValueError("checkpoint/log intervals must be positive")
    total_epochs = sum(config.generator_phases)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= total_epochs:
        raise ValueError("stop boundary must be inside the declared budget")
    device = torch.device(device)
    if execution not in ("eager", "cuda-graph"):
        raise ValueError("unknown GIFT execution backend")
    if device.type != "cuda":
        execution = "eager"  # CPU fallback uses the reference path and identity.
    configure_determinism(config.generator_seed, strict=True)
    bank = ObservationBank(dataset, fixture=config.fixture_only)
    identity = {"role": "gift_prediction_generator", "configuration": asdict(config),
                "data": bank.binding, "sources": source_identity(), "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "selection": "terminal_epoch", "budget_unit": "trajectory_epoch"}
    identity["execution"] = execution if device.type == "cuda" else "eager"
    output, store = _open_run(output, identity, resume)
    saved = store.payload
    if saved is not None and not 0 <= saved["epoch"] <= total_epochs:
        raise ValueError("saved epoch is outside the configured budget")
    if saved is not None and stop_after_epoch is not None and stop_after_epoch <= saved["epoch"]:
        raise ValueError("stop boundary must follow the saved epoch")
    model = GIFTGenerator(config.cutoff, config.rank, bank.state_scale, False).to(device)
    parameters = configure_variable_projection_parameters(model)
    scale_state, scale_target = bank.pairs(config.generator_seed, 0)
    target_scale = float(scale_target.square().mean().clamp_min(1e-20))
    valid_state, valid_target = bank.pairs(config.generator_seed, 0, validation=True)
    if saved is None:
        fit_affine_minimum_norm(model, scale_state, scale_target, device=device,
                               subtract_quadratic=False)
    else:
        if saved["target_scale"] != target_scale or saved["state_scale"] != bank.state_scale:
            raise ValueError("resumed training scales differ")
        model.load_state_dict(saved["model"], strict=True)
    del scale_state, scale_target
    history = [] if saved is None else saved["history"]
    completed = 0 if saved is None else saved["epoch"]
    update_count = 0 if saved is None else saved["optimizer_updates"]
    elapsed = 0.0 if saved is None else saved["committed_seconds"]
    batches = math.ceil(len(bank.training) / config.batch_size)
    optimizer = scheduler = None
    engine = None
    if execution == "cuda-graph":
        from training.gift_acceleration import TrainingEngine
        engine = TrainingEngine(model, kind="generator", scale=target_scale, backend=execution)
    last_phase = None
    final_summary = None if saved is None else saved["affine_summary"]
    for epoch in range(completed + 1, total_epochs + 1):
        phase, phase_epoch = phase_at(epoch, config.generator_phases)
        if phase != last_phase:
            lr = config.generator_learning_rates[phase]
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-8)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.generator_phases[phase] * batches, eta_min=lr * 0.02)
            if saved is not None and epoch == completed + 1 and saved["phase"] == phase:
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
            last_phase = phase
        if saved is not None and epoch == completed + 1:
            store.restore_random_state()
        synchronize(device)
        started = time.perf_counter()
        states, targets = bank.pairs(config.generator_seed, epoch)
        model.train()
        weighted_loss = 0.0
        for first in range(0, len(states), config.batch_size):
            state = states[first:first + config.batch_size].to(device)
            target = targets[first:first + config.batch_size].to(device)
            if engine is not None:
                loss = engine.step(optimizer, state, target)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = (model(state) - target).square().mean() / target_scale
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("generator training loss is nonfinite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
                optimizer.step()
                loss = float(loss.detach())
            scheduler.step()
            update_count += 1
            weighted_loss += loss * len(state)
        # Reuse this epoch's observed windows; analytic refits are counted
        # separately and are not misrepresented as gradient updates or epochs.
        final_fit = fit_affine_minimum_norm(model, states, targets, device=device,
                                            subtract_quadratic=True)
        validation = evaluate(model, valid_state, valid_target, device=device)["mean"]
        if not math.isfinite(validation):
            raise FloatingPointError("generator validation is nonfinite")
        synchronize(device)
        seconds = time.perf_counter() - started
        elapsed += seconds
        row = {"epoch": epoch, "phase": phase + 1, "phase_epoch": phase_epoch,
               "trajectories_visited": len(states), "optimizer_updates": batches,
               "cumulative_optimizer_updates": update_count,
               "normalized_training_mse": weighted_loss / len(states),
               "validation_relative_l2": validation,
               "learning_rate": optimizer.param_groups[0]["lr"], "seconds": seconds}
        history.append(row)
        if epoch == 1 or epoch % log_interval == 0 or epoch == total_epochs:
            print(json.dumps(row, allow_nan=False), flush=True)
        final = epoch == total_epochs
        if (epoch % checkpoint_interval == 0 or phase_epoch == config.generator_phases[phase]
                or final or epoch == stop_after_epoch):
            final_summary = final_fit.summary(identifiability_grid(
                model, bank.training.shape[-1], device=device))
            store.save(f"epoch_{epoch:04d}", {
                "model": clone_state(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "phase": phase,
                "optimizer_updates": update_count, "history": history,
                "state_scale": bank.state_scale, "target_scale": target_scale,
                "committed_seconds": elapsed, "completed": final,
                "affine_summary": final_summary,
            })
        if epoch == stop_after_epoch and not final:
            return {"status": "paused_at_committed_epoch", "epoch": epoch, "output": str(output)}
    if update_count != total_epochs * batches or len(history) != total_epochs:
        raise RuntimeError("completed generator budget is inconsistent")
    artifact = {
        "format_version": 3, "artifact_role": "trained_generator", "condition": "noise_000",
        "model_configuration": model.configuration(), "model_state_dict": clone_state(model),
        "training_regime": "full_data_prediction", "fixture_only": config.fixture_only,
        "training_configuration": asdict(config), "training_data": bank.binding,
        "training_cost": {"epochs": total_epochs, "optimizer_updates": update_count,
                          "affine_fits": total_epochs + 1, "committed_epoch_seconds": elapsed,
                          "timing_scope": "sampling, gradient updates, affine refits and validation; excludes setup, checkpoint writes and paused time"},
        "identifiability": final_summary, "selection": "terminal_epoch",
        "fresh_training": {"pretrained_model_loaded": False, "same_attempt_continuation": resume},
        "source_identity": identity["sources"],
        "execution": identity["execution"],
    }
    with (output / "model.pt").open("xb") as stream:
        torch.save(artifact, stream)
    write_json_new(output / "history.json", history)
    report = {"status": "complete", "role": identity["role"], "configuration": asdict(config),
              "training_data": bank.binding, "training_cost": artifact["training_cost"],
              "model_sha256": digest_file(output / "model.pt"), "runtime": store.runtime,
              "selection": "terminal_epoch", "sources": identity["sources"]}
    write_json_new(output / "COMPLETE.json", report)
    return report


def main_generator(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    action.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--execution", choices=("eager", "cuda-graph"), default="eager",
                        help="Opt-in graph replay; CPU executes eager. Resume requires the same backend.")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"configuration": asdict(PredictionTrainingConfig()),
                          "dataset": str(args.dataset), "output": str(args.output),
                          "execution": args.execution, "selection": "terminal_epoch", "writes": 0}, indent=2))
    else:
        result = run_generator(args.dataset, args.output, device=args.device, resume=args.resume,
                               checkpoint_interval=args.checkpoint_interval,
                               stop_after_epoch=args.stop_after_epoch, execution=args.execution)
        print(json.dumps(result, indent=2), flush=True)
