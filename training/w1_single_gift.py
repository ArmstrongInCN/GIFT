"""One fresh GIFT pipeline with the original full-data optimizer and budgets.

Only the observation window/grid glue is reused from W1-v2. Physical batches,
phase-local schedulers, optimizer resets and terminal model selection follow
the pinned GIFT full-data scripts. No validation score changes training or
chooses an earlier generator/branch. The .04s observations retain .02s RK4.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from training import w1_v2_gift as common
from experiments.formal._shared.gift_generator_training import evaluate
from experiments.formal.train_gift_branches import _validate_derivative, _validate_rollout

PROFILE = "single_native76_n300"


@dataclass(frozen=True)
class NativeGIFTConfig:
    """Pinned author settings plus explicit common-window sampling/monitor glue.

    Fixture overrides exist solely for short CPU recovery tests. Production
    never accepts a different budget, physical batch, model seed or grid.
    """

    generator_phases: tuple[int, int] = (250, 250)
    branch_phases: tuple[int, int, int, int] = (400, 50, 25, 25)
    generator_learning_rates: tuple[float, float] = (.01, .003)
    branch_learning_rates: tuple[float, float, float, float] = (.0015, .0002, .00008, .00008)
    generator_batch: int = 16
    derivative_batch: int = 16
    rollout_batch: int = 5
    generator_seed: int = 2026072301
    branch_seed: int = 20260820
    order_seed: int = 2026100301
    sampling_seed: int = 2026100303
    validation_batch: int = 10
    branch_monitor_interval: int = 10
    stage: str = "formal"
    profile: str = PROFILE
    cutoff: int = 21
    rank: int = 8
    fixture_only: bool = False

    def validate(self):
        """Keep production fixed; permit only declared small, positive CPU fixtures."""
        if (len(self.generator_phases) != 2 or len(self.branch_phases) != 4
                or len(self.generator_learning_rates) != 2 or len(self.branch_learning_rates) != 4
                or min(*self.generator_phases, *self.branch_phases, self.generator_batch,
                       self.derivative_batch, self.rollout_batch, self.validation_batch,
                       self.branch_monitor_interval, self.cutoff, self.rank) < 1
                or min(*self.generator_learning_rates, *self.branch_learning_rates) <= 0):
            raise ValueError("invalid native GIFT phase/batch settings")
        if self.profile != PROFILE or self.stage != "formal":
            raise ValueError("native GIFT profile/stage differs")
        if not self.fixture_only and self != NativeGIFTConfig():
            raise ValueError("production single GIFT must use the pinned native settings")


def native_training_config(config=NativeGIFTConfig()):
    """Return the complete fixed configuration for the orchestrator's protocol lock."""
    return {**asdict(config),
            "generator_monitor": "observed_derivative_each_epoch",
            "branch_monitor": "native_derivative_or_one_second_each10_epochs_and_phase_end",
            "terminal_diagnostic": "one_common_three_second_rollout_per_component",
            "selection": "terminal_epoch", "early_stopping": False,
            "execution": "eager", "gradient_accumulation": False,
            "generator_scheduler": "phase_local_cosine_per_update_floor_0.02_initial_lr",
            "branch_scheduler": "derivative_cosine_per_epoch_floor_0.0001_then_constant"}


def _bank(dataset, config, expected_training_count=300):
    """Reuse the decoded-data receipt and reject a production grid/population change."""
    if expected_training_count not in (50, 300):
        raise ValueError("native fixed-stage GIFT permits the declared N300 or N50 comparison")
    bank = common.PredictionObservationBank(dataset, config)
    grid = common._grid_spec(bank)
    if not config.fixture_only:
        if (len(bank.training) != expected_training_count or len(bank.validation) != 20 or grid["points"] != 76
                or grid["save_interval"] != .04 or bank.binding.get("time_stride") != 2
                or bank.binding.get("subset_seed") != 2026100304):
            raise ValueError("native single GIFT requires its locked population/76-point receipt")
    return bank


def _optimizer(parameters, stage, phase, config, population):
    """Rebuild AdamW at a phase boundary with the original local cosine clock."""
    rates = config.generator_learning_rates if stage == "generator" else config.branch_learning_rates
    optimizer = torch.optim.AdamW(parameters, lr=rates[phase], weight_decay=1e-8 if stage == "generator" else 1e-6)
    if stage == "generator":
        steps = config.generator_phases[phase] * math.ceil(population/config.generator_batch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.02*rates[phase])
    elif phase == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.branch_phases[0], eta_min=.0001)
    else:
        scheduler = None
    return optimizer, scheduler


def _validate_boundary(saved, stage, config, population):
    """Validate the committed phase, physical updates and phase-local scheduler time."""
    if saved is None:
        return
    counts = config.generator_phases if stage == "generator" else config.branch_phases
    epoch = saved["epoch"]
    if type(epoch) is not int or not 0 <= epoch <= sum(counts):
        raise ValueError("native checkpoint epoch outside budget")
    phase, phase_epoch = common.phase_at(epoch, counts) if epoch else (0, 0)
    if saved["phase"] != phase or saved["completed"] != (epoch == sum(counts)):
        raise ValueError("native checkpoint phase/completion differs")
    expected_updates = 0
    for row in saved["history"]:
        row_phase = common.phase_at(row["epoch"], counts)[0]
        batch = config.generator_batch if stage == "generator" else config.derivative_batch if row_phase == 0 else config.rollout_batch
        if row["phase"] != row_phase or row["optimizer_updates"] != math.ceil(population/batch):
            raise ValueError("native checkpoint physical update counts differ")
        expected_updates += row["optimizer_updates"]
    if saved["optimizer_updates"] != expected_updates:
        raise ValueError("native checkpoint cumulative updates differ")
    clock = saved["scheduler"]
    if stage == "generator":
        batches = math.ceil(population/config.generator_batch)
        expected_clock, expected_limit = phase_epoch*batches, counts[phase]*batches
    elif phase == 0:
        expected_clock, expected_limit = phase_epoch, counts[0]
    else:
        if clock is not None:
            raise ValueError("native rollout phase must have no scheduler")
        return
    if clock is None or clock["last_epoch"] != expected_clock or clock["T_max"] != expected_limit:
        raise ValueError("native checkpoint phase-local cosine time differs")


def _boundary(model, optimizer, scheduler, epoch, phase, history, completed, **extra):
    """Keep only this pipeline's complete rolling state; no selected historical model."""
    return {"model": common.clone_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch, "phase": phase, "history": history, "best": None,
            "optimizer_updates": sum(row["optimizer_updates"] for row in history),
            "completed": completed, **extra}


def _finite_model(model):
    """Finite training weights/buffers are required independently of validation quality."""
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise FloatingPointError("native GIFT training produced nonfinite parameters/buffers")


def _nonfinite_gradient_error(error):
    """Recognize PyTorch's explicit clipping rejection, not an arbitrary RuntimeError."""
    message = str(error)
    return message.startswith("The total norm of order ") and "is non-finite, so it cannot be clipped" in message


def _clip_generator_gradients(parameters):
    """Preserve native clip5 and classify its nonfinite rejection as scientific."""
    try:
        torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    except RuntimeError as error:
        if _nonfinite_gradient_error(error):
            raise FloatingPointError(str(error)) from error
        raise


def _json_metrics(value):
    """Keep nonfinite diagnostic values as null without dropping their population."""
    if isinstance(value, dict):
        return {key: _json_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_metrics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _monitor(call, population):
    """Time native validation while preserving training RNG; a gate rejection is diagnostic.

    Native validation raises for an entire rejected batch and cannot identify
    individual failed rows. Mark its whole population unavailable, without
    fabricating IDs. The terminal common rollout separately records true IDs.
    Infrastructure exceptions remain exceptions.
    """
    rng, started = common.capture_rng(), time.perf_counter()
    try:
        try:
            values = _json_metrics(call())
            failed = any(value is None for value in values.values())
            return {**values, "status": "failed_nonfinite" if failed else "finite",
                    "population": population, "seconds": time.perf_counter()-started}
        except (FloatingPointError, ArithmeticError) as error:
            return {"status": "failed_numerical_diagnostic", "reason": repr(error),
                    "population": population, "seconds": time.perf_counter()-started}
        except RuntimeError as error:
            if str(error) != "recursive local correction gate failed during training":
                raise
            return {"status": "failed_native_validation_gate", "reason": str(error),
                    "population": population, "individual_failure_ids_known": False,
                    "seconds": time.perf_counter()-started}
    finally:
        common.restore_rng(rng)


def _sources():
    """Source bytes are provenance, while decoded-data/config/RNG govern recovery."""
    return {**common._sources(), "training/w1_single_gift.py": common.digest_file(Path(__file__))}


def _report_row(stage, row):
    """Print compact monitoring progress; do not dump per-trajectory field arrays."""
    print(json.dumps({"stage": stage, **{key: row[key] for key in
          ("epoch", "phase", "phase_epoch", "optimizer_updates", "normalized_training_mse", "learning_rate_next", "seconds")}}, allow_nan=False), flush=True)


def _export_generator(model, config, bank, epoch, diagnostic, cost):
    value = common._generator_export(model, common.clone_state(model), config, config.generator_seed,
                                    bank, epoch, diagnostic["val_metric"], cost)
    value.update(selection="terminal_epoch", profile=PROFILE, training_regime="w1_single_native_t5_t8",
                 config=native_training_config(config),
                 terminal_validation=diagnostic, validation_status="failed" if diagnostic["val_metric"] is None else "finite")
    return value


def run_generator(dataset, output, *, config=NativeGIFTConfig(), device="cuda", resume=False, stop_after_epoch=None,
                  expected_training_count=300):
    """Train both original generator phases from scratch or this run's own state."""
    config.validate()
    device, total = torch.device(device), sum(config.generator_phases)
    setup_started = time.perf_counter()
    common.configure_determinism(config.generator_seed, strict=True)
    bank = _bank(dataset, config, expected_training_count)
    identity = {"role": "w1_single_native_gift_generator", "configuration": asdict(config),
                "seed": config.generator_seed, "data": bank.binding, "sources": _sources(), "device": str(device)}
    journal = common._Journal(output, identity, resume)
    saved = journal.payload
    _validate_boundary(saved, "generator", config, len(bank.training))
    if saved and saved["completed"] and (journal.output/"COMPLETE.json").exists():
        return json.loads((journal.output/"COMPLETE.json").read_text(encoding="utf-8"))
    model = common.GIFTGenerator(config.cutoff, config.rank, bank.state_scale, False).to(device)
    parameters = common.configure_variable_projection_parameters(model)
    states, targets = bank.pairs(config.generator_seed, 0)
    target_scale = float(targets.square().mean().clamp_min(1e-20))
    if saved is None:
        common.fit_affine_minimum_norm(model, states, targets, device=device, subtract_quadratic=False)
    else:
        if saved["target_scale"] != target_scale:
            raise ValueError("native generator training target scale differs")
        model.load_state_dict(saved["model"], strict=True)
    _finite_model(model)
    valid_state, valid_target = bank.pairs(config.generator_seed, 0, validation=True)
    phase = 0 if saved is None else saved["phase"]
    optimizer, scheduler = _optimizer(parameters, "generator", phase, config, len(bank.training))
    if saved:
        common.restore_optimizer(optimizer, saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        common.restore_rng(journal.rng)
    completed = 0 if saved is None else saved["epoch"]
    if stop_after_epoch is not None and not completed < stop_after_epoch <= total:
        raise ValueError("native generator stop boundary outside remaining budget")
    history = [] if saved is None else saved["history"]
    setup_seconds = (0. if saved is None else saved["setup_seconds"])+time.perf_counter()-setup_started
    if saved is None:
        journal.save(_boundary(model, optimizer, scheduler, 0, phase, history, False,
                               target_scale=target_scale, setup_seconds=setup_seconds))
    for epoch in range(completed+1, total+1):
        current_phase, phase_epoch = common.phase_at(epoch, config.generator_phases)
        if current_phase != phase:
            phase = current_phase
            optimizer, scheduler = _optimizer(parameters, "generator", phase, config, len(bank.training))
        common.synchronize(device)
        started = time.perf_counter()
        states, targets = bank.pairs(config.generator_seed, epoch-1)
        model.train()
        weighted, updates = 0., 0
        for first in range(0, len(states), config.generator_batch):
            last = min(first+config.generator_batch, len(states))
            state, target = states[first:last].to(device), targets[first:last].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(state)-target).square().mean()/target_scale
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("native generator loss nonfinite")
            loss.backward()
            _clip_generator_gradients(parameters)
            optimizer.step()
            scheduler.step()
            _finite_model(model)
            weighted += float(loss.detach())*(last-first)
            updates += 1
        common.fit_affine_minimum_norm(model, states, targets, device=device, subtract_quadratic=True)
        _finite_model(model)
        model.eval()
        validation = _monitor(lambda: evaluate(model, valid_state, valid_target, device=device), len(valid_state))
        common.synchronize(device)
        history.append({"epoch": epoch, "global_trajectory_epoch": epoch, "phase": phase,
                        "phase_epoch": phase_epoch, "optimizer_updates": updates,
                        "cumulative_optimizer_updates": sum(row["optimizer_updates"] for row in history)+updates,
                        "trajectories_visited": len(states), "normalized_training_mse": weighted/len(states),
                        "physical_batch": config.generator_batch, "learning_rate_next": optimizer.param_groups[0]["lr"],
                        "validation": validation, "selection_eligible": False, "seconds": time.perf_counter()-started})
        journal.save(_boundary(model, optimizer, scheduler, epoch, phase, history, epoch == total,
                               target_scale=target_scale, setup_seconds=setup_seconds))
        if epoch % 10 == 0 or phase_epoch == config.generator_phases[phase]:
            _report_row("generator", history[-1])
        if epoch == stop_after_epoch and epoch != total:
            return {"status": "paused_at_committed_epoch", "epoch": epoch, "output": str(journal.output)}
    diagnostic = common._validation(common.FixedBandwidthGridGenerator(model, bank.training.shape[-1], reference_grid=bank.training.shape[-1]),
                                     None, bank, device, config)
    cost = common._cost(history, config.generator_phases, affine_fits=total+1, setup_seconds=setup_seconds)
    cost.update(diagnostic_seconds=diagnostic["seconds"], diagnostic_count=1,
                total_training_seconds=cost["total_training_seconds"]+diagnostic["seconds"],
                monitor_count=len(history), parameter_count=sum(p.numel() for p in model.parameters()),
                timing_scope="committed epochs include observed-derivative monitoring and affine refits; setup and one terminal3s diagnostic recorded separately; excludes snapshot IO and paused time")
    exported = _export_generator(model, config, bank, total, diagnostic, cost)
    for name in ("terminal.pt", "model.pt"):
        common._atomic_torch(journal.output/name, exported)
    report = {"status": "complete", "stage": "generator", "profile": PROFILE, "seed": config.generator_seed,
              "epoch": total, "selected_epoch": total, "selection": "terminal_epoch",
              "val_metric": diagnostic["val_metric"], "validation_status": exported["validation_status"],
              "terminal_validation": diagnostic, "training_cost": cost, "runtime": journal.runtime,
              "config": asdict(config), "output": str(journal.output)}
    common._atomic_json(journal.output/"COMPLETE.json", report)
    return report


def _pipeline_cost(generator_cost, branch_cost, digest):
    return {"generator": generator_cost, "branch": branch_cost,
            "total_trajectory_epochs": generator_cost["trajectory_epochs"]+branch_cost["trajectory_epochs"],
            "total_optimizer_updates": generator_cost["optimizer_updates"]+branch_cost["optimizer_updates"],
            "total_committed_seconds": generator_cost["committed_seconds"]+branch_cost["committed_seconds"],
            "total_training_seconds": generator_cost["total_training_seconds"]+branch_cost["total_training_seconds"],
            "generator_weights_sha256": digest}


def run_branch(dataset, low_model, output, *, config=NativeGIFTConfig(), device="cuda", resume=False, stop_after_epoch=None,
               expected_training_count=300):
    """Freeze this run's terminal generator, then train the original four branch phases."""
    config.validate()
    if config.cutoff != 21:
        raise ValueError("native branch requires cutoff21")
    device, total, offset = torch.device(device), sum(config.branch_phases), sum(config.generator_phases)
    setup_started = time.perf_counter()
    common.configure_determinism(config.branch_seed, strict=False)
    bank = _bank(dataset, config, expected_training_count)
    generator_payload = torch.load(low_model, map_location="cpu", weights_only=True)
    if (generator_payload.get("training_regime") != "w1_single_native_t5_t8"
            or generator_payload.get("selection") != "terminal_epoch"
            or generator_payload.get("epoch") != offset
            or generator_payload.get("training_data", {}).get("observations_sha256") != bank.binding["observations_sha256"]):
        raise ValueError("native branch requires its completed terminal generator and identical observations")
    digest = common.state_digest(generator_payload["model_state_dict"])
    generator = common.load_generator(generator_payload, device=device).eval().requires_grad_(False)
    _finite_model(generator)
    frozen = common.FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    states, targets = bank.pairs(config.branch_seed, offset)
    qtargets = common._project(targets.numpy(), ~common._square_mask(21))
    low = common._compute_low_rhs(frozen, states.numpy(), device)
    scales = common._training_scales(states.numpy(), qtargets, low)
    if not all(math.isfinite(v) and v > 0 for v in scales):
        raise FloatingPointError("native branch training scales nonfinite")
    model = common.HighFrequencyBranch(common.HighFrequencyConfig(), state_scale=scales[0], high_scale=scales[1],
                low_rhs_scale=scales[2], high_rhs_scale=scales[3], frozen_generator=frozen).to(device)
    identity = {"role": "w1_single_native_gift_branch", "configuration": asdict(config), "seed": config.branch_seed,
                "data": bank.binding, "generator_weights_sha256": digest, "scales": list(scales),
                "sources": _sources(), "device": str(device)}
    journal = common._Journal(output, identity, resume)
    saved = journal.payload
    _validate_boundary(saved, "branch", config, len(bank.training))
    if saved and saved["completed"] and (journal.output/"COMPLETE.json").exists():
        return json.loads((journal.output/"COMPLETE.json").read_text(encoding="utf-8"))
    if saved:
        model.load_state_dict(saved["model"], strict=True)
    _finite_model(model)
    valid_state, valid_target = bank.pairs(config.branch_seed, offset, validation=True)
    valid_target = torch.from_numpy(common._project(valid_target.numpy(), ~common._square_mask(21)))
    valid_low = torch.from_numpy(common._compute_low_rhs(frozen, valid_state.numpy(), device))
    valid_derivative = DataLoader(TensorDataset(valid_state, valid_target, valid_low), batch_size=config.derivative_batch, shuffle=False)
    valid_sequence = torch.from_numpy(bank.sequences(config.branch_seed, offset, validation=True))
    valid_rollout = DataLoader(TensorDataset(valid_sequence), batch_size=config.validation_batch, shuffle=False)
    phase = 0 if saved is None else saved["phase"]
    optimizer, scheduler = _optimizer(model.parameters(), "branch", phase, config, len(bank.training))
    if saved:
        common.restore_optimizer(optimizer, saved["optimizer"])
        if scheduler:
            scheduler.load_state_dict(saved["scheduler"])
        common.restore_rng(journal.rng)
    completed = 0 if saved is None else saved["epoch"]
    if stop_after_epoch is not None and not completed < stop_after_epoch <= total:
        raise ValueError("native branch stop boundary outside remaining budget")
    history = [] if saved is None else saved["history"]
    setup_seconds = (0. if saved is None else saved["setup_seconds"])+time.perf_counter()-setup_started
    if saved is None:
        journal.save(_boundary(model, optimizer, scheduler, 0, phase, history, False, setup_seconds=setup_seconds))
    for epoch in range(completed+1, total+1):
        current_phase, phase_epoch = common.phase_at(epoch, config.branch_phases)
        if current_phase != phase:
            phase = current_phase
            optimizer, scheduler = _optimizer(model.parameters(), "branch", phase, config, len(bank.training))
        batch = config.derivative_batch if phase == 0 else config.rollout_batch
        # Exactly one microbatch per effective group: this reuses only the
        # native sparse loss/RK4 implementation, without cross-batch accumulation.
        loss_config = SimpleNamespace(effective_batch=batch, branch_micro_batch=batch)
        common.synchronize(device)
        started = time.perf_counter()
        try:
            loss, updates = common._branch_epoch(model, frozen, bank, optimizer, loss_config, config.branch_seed,
                                                 offset+epoch-1, phase, device, scales)
        except RuntimeError as error:
            if _nonfinite_gradient_error(error):
                raise FloatingPointError(str(error)) from error
            raise
        if not math.isfinite(loss):
            raise FloatingPointError("native branch training loss nonfinite")
        _finite_model(model)
        if scheduler:
            scheduler.step()
        validation = None
        if epoch == 1 or epoch % config.branch_monitor_interval == 0 or phase_epoch == config.branch_phases[phase]:
            if phase == 0:
                validation = _monitor(lambda: _validate_derivative(model, valid_derivative, device, scales[3]), len(bank.validation))
            else:
                validation = _monitor(lambda: _validate_rollout(model, frozen, valid_rollout, device), len(bank.validation))
        common.synchronize(device)
        history.append({"epoch": epoch, "global_trajectory_epoch": offset+epoch, "phase": phase,
                        "phase_epoch": phase_epoch, "optimizer_updates": updates,
                        "cumulative_optimizer_updates": sum(row["optimizer_updates"] for row in history)+updates,
                        "trajectories_visited": len(bank.training), "normalized_training_mse": loss,
                        "physical_batch": batch, "learning_rate_next": optimizer.param_groups[0]["lr"],
                        "validation": validation, "selection_eligible": False, "seconds": time.perf_counter()-started})
        journal.save(_boundary(model, optimizer, scheduler, epoch, phase, history, epoch == total, setup_seconds=setup_seconds))
        if validation:
            _report_row("branch", history[-1])
        if epoch == stop_after_epoch and epoch != total:
            return {"status": "paused_at_committed_epoch", "epoch": epoch, "output": str(journal.output)}
    diagnostic = common._validation(frozen, model, bank, device, config)
    stage_cost = common._cost(history, config.branch_phases, setup_seconds=setup_seconds)
    stage_cost.update(diagnostic_seconds=diagnostic["seconds"], diagnostic_count=1,
                      total_training_seconds=stage_cost["total_training_seconds"]+diagnostic["seconds"],
                      monitor_count=sum(row["validation"] is not None for row in history),
                      timing_scope="committed epochs include native derivative/one-second monitoring and frozen-RHS preparation; setup and one terminal3s diagnostic recorded separately; excludes snapshot IO and paused time")
    cost = _pipeline_cost(generator_payload["training_cost"], stage_cost, digest)
    exported = common._branch_export(model, common.clone_state(model), low_model, digest, config, config.branch_seed,
                                     bank, total, diagnostic["val_metric"], cost)
    exported.update(selection="terminal_epoch", phase="terminal_epoch", profile=PROFILE, data_profile="w1_single_native_t5_t8",
                    config=native_training_config(config),
                    terminal_validation=diagnostic, validation_status="failed" if diagnostic["val_metric"] is None else "finite")
    for name in ("terminal.pt", "model.pt"):
        common._atomic_torch(journal.output/name, exported)
    report = {"status": "complete", "stage": "branch", "profile": PROFILE, "seed": config.branch_seed,
              "epoch": total, "selected_epoch": total, "selection": "terminal_epoch", "val_metric": diagnostic["val_metric"],
              "validation_status": exported["validation_status"], "terminal_validation": diagnostic,
              "training_cost": cost, "runtime": journal.runtime, "config": asdict(config), "output": str(journal.output)}
    common._atomic_json(journal.output/"COMPLETE.json", report)
    return report


def run_pipeline(dataset, output, *, config=NativeGIFTConfig(), device="cuda", resume=False, expected_training_count=300):
    """Run one generator and one branch; diagnostic validation never selects weights."""
    config.validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generator_dir, branch_dir = output/"generator", output/"branch"
    generator = run_generator(dataset, generator_dir, config=config, device=device, resume=resume and generator_dir.exists(),
                              expected_training_count=expected_training_count)
    branch = run_branch(dataset, generator_dir/"model.pt", branch_dir, config=config, device=device,
                        resume=resume and branch_dir.exists(), expected_training_count=expected_training_count)
    result = {"status": "complete", "method": "GIFT", "stage": "formal", "profile": PROFILE,
              "seed": config.branch_seed, "generator_seed": config.generator_seed,
              "generator": generator, "branch": branch, "training_cost": branch["training_cost"],
              "validation_status": branch["validation_status"],
              "paths": {"generator_model": str(generator_dir/"model.pt"), "branch_model": str(branch_dir/"model.pt"),
                        "generator_terminal": str(generator_dir/"terminal.pt"), "branch_terminal": str(branch_dir/"terminal.pt")}}
    common._atomic_json(output/"PIPELINE.json", result)
    common._atomic_json(output/"COMPLETE.json", result)
    common._atomic_json(output/"STATUS.json", {"status": "complete", "method": "GIFT", "stage": "formal",
                                             "profile": PROFILE, "seed": config.branch_seed,
                                             "validation_status": branch["validation_status"], "paths": result["paths"]})
    return result


def main(argv=None):
    """Execute one locked job JSON; preserve typed failures and nonzero process exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    job = json.loads(args.run_config.read_text(encoding="utf-8"))
    if (job.get("method") != "GIFT" or job.get("stage", "formal") != "formal"
            or job.get("profile", PROFILE) != PROFILE or job.get("seed", 20260820) != 20260820
            or job.get("generator_seed", 2026072301) != 2026072301):
        raise ValueError("GIFT job differs from the locked single native profile")
    dataset = job.get("window", job.get("dataset"))
    if dataset is None:
        raise ValueError("GIFT job requires its bound window")
    if "window" in job and "dataset" in job and Path(job["window"]).resolve() != Path(job["dataset"]).resolve():
        raise ValueError("GIFT job window/dataset aliases differ")
    output = Path(job["output"]).resolve()
    config = NativeGIFTConfig()
    try:
        if "config" in job and json.dumps(job["config"], sort_keys=True) != json.dumps(native_training_config(), sort_keys=True):
            raise ValueError("GIFT job scientific config differs from the pinned native recipe")
        output.mkdir(parents=True, exist_ok=True)
        common._atomic_json(output/"STATUS.json", {"status": "running", "method": "GIFT", "seed": config.branch_seed,
                                                 "stage": "formal", "profile": PROFILE})
        result = run_pipeline(dataset, output, config=config, device=job.get("device", "cuda"), resume=args.resume)
        if result["status"] != "complete":
            raise RuntimeError("native GIFT pipeline did not complete")
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        failure = {"method": "GIFT", "seed": config.branch_seed, "generator_seed": config.generator_seed,
                   "stage": "formal", "profile": PROFILE, "status": "failed", "reason": repr(error),
                   "failure_kind": "numerical" if isinstance(error, (FloatingPointError, ArithmeticError)) else "infrastructure"}
        common._atomic_json(output/"FAILURE.json", failure)
        common._atomic_json(output/"STATUS.json", failure)
        raise


if __name__ == "__main__":
    main()
