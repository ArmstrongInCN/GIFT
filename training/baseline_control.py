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
from training.checkpoints import CheckpointStore, digest_file, runtime_identity


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = "fno/fno1000_n64_t0_t10_dt0p02.h5"
TRAIN_SHA = "1322ad0a5be67bccab2a93c55a5a5748f792f80f3a839032bce888959d54e3ee"
PROTOCOLS = {
    "fno2d": dict(epochs=500, trajectories=1000, history=46, rollout=150, micro_batch=10,
                  accumulation=2, learning_rate=0.001, weight_decay=0.0001, seed=0),
    "fno3d": dict(epochs=500, trajectories=1000, history=46, rollout=150, micro_batch=5,
                  accumulation=2, learning_rate=0.001, weight_decay=0.0001, seed=0),
    "uno": dict(epochs=150, trajectories=1000, history=46, rollout=20, micro_batch=16,
                accumulation=1, learning_rate=0.001, weight_decay=0.00001, seed=0),
    "unet": dict(epochs=500, trajectories=1000, history=46, rollout=4, micro_batch=20,
                 accumulation=1, learning_rate=0.001, weight_decay=0.0001, seed=0),
}


def _arguments(model: str, argv=None):
    parser = argparse.ArgumentParser(description=f"Independent {model} training; default is read-only dry run")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run-training", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue this output's own checkpoint only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-file", default=TRAIN_FILE, help="path relative to GIFT_DATA_ROOT")
    parser.add_argument("--data-profile", choices=("released", "regenerated"), default="released",
                        help="released: original SHA; regenerated: completed generated collection with provenance")
    parser.add_argument("--stop-after-epoch", type=int, help="operational pause; does not change total run budget")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="save every N completed epochs; default 10")
    parser.add_argument("--tiny", action="store_true", help="explicit NONFORMAL test budget, never a formal artifact")
    parser.add_argument("--tiny-epochs", type=int, default=2)
    parser.add_argument("--tiny-trajectories", type=int, default=2)
    parser.add_argument("--tiny-batch", type=int, default=1)
    parser.add_argument("--tiny-rollout", type=int)
    args = parser.parse_args(argv)
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
    configured = os.environ.get("GIFT_DATA_ROOT")
    if not configured:
        raise ValueError("set GIFT_DATA_ROOT to the external dataset directory")
    base = Path(configured).expanduser().resolve(strict=True)
    if base == ROOT or ROOT in base.parents:
        raise ValueError("training data must remain outside this repository")
    path = (base / relative).resolve(strict=True)
    if base not in path.parents:
        raise ValueError("--data-file must stay inside GIFT_DATA_ROOT")
    return base, path


def configuration(model: str, args) -> dict:
    config = dict(PROTOCOLS[model])
    config.update(model=model, formal=not args.tiny, scheduler_step=100, scheduler_gamma=0.5,
                  selection="terminal_epoch_not_validation", teacher_forcing=False, detach_rollout=False,
                  checkpoint_interval=args.checkpoint_interval, data_profile=args.data_profile)
    if args.tiny:
        config.update(epochs=args.tiny_epochs, trajectories=args.tiny_trajectories,
                      micro_batch=args.tiny_batch, accumulation=1,
                      rollout=args.tiny_rollout or (16 if model == "fno3d" else 1))
    if any(config[key] < 1 for key in ("epochs", "trajectories", "micro_batch", "rollout")):
        raise ValueError("training counts must be positive")
    if model == "fno3d" and config["rollout"] < 14:
        raise ValueError("FNO-3D's unchanged modes3=8 needs at least 14 future points")
    effective = config["micro_batch"] * config["accumulation"]
    if model != "uno" and config["trajectories"] % effective:
        raise ValueError("this protocol requires complete physical/effective batches")
    return config


def _regenerated_input(base: Path, path: Path, *, full: bool) -> dict:
    """Validate only this consumer's complete generated input, without training.

    A collection may intentionally omit unrelated experiments. Its manifest is
    an integrity/provenance record, not a certificate of numerical reproduction.
    """
    from scripts import assemble_generated_data as assembly
    from scripts import generate_data as gen

    manifest_path = assembly.checked_file(base, "manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema") != "gift.generated-data-collection.v1"
            or manifest.get("data_profile") != "regenerated"
            or manifest.get("status") != "ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED"
            or manifest.get("scientific_arrays_modified") is not False
            or manifest.get("released_truth_used_to_fill_gaps") is not False
            or manifest.get("full_hash_and_finite_checks_performed") is not True
            or manifest.get("assembler_sha256") != digest_file(Path(assembly.__file__))):
        raise ValueError("regenerated collection status/provenance/assembler differs")
    available, missing = manifest.get("available_jobs", []), manifest.get("missing_jobs", [])
    allowed = set(assembly.CLEAN_PATHS) | set(assembly.SAMPLING) | {"noise"}
    if (len(available) != len(set(available)) or len(missing) != len(set(missing))
            or set(available) & set(missing) or set(available) | set(missing) != allowed
            or "fno-training" not in available):
        raise ValueError("regenerated collection job inventory differs")
    records, seen = {}, set()
    for record in manifest["files"]:
        relative = record["path"]
        parts = relative.split("/")
        if (not relative or "\\" in relative or ":" in relative or any(p in ("", ".", "..") for p in parts)
                or relative.casefold() in seen):
            raise ValueError("unsafe/duplicate generated collection path")
        seen.add(relative.casefold())
        checked = assembly.checked_file(base, relative)
        expected = record["sha256"]
        if (checked.stat().st_size != record["bytes"] or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("generated collection file size/SHA declaration differs")
        records[relative] = record
    if path != assembly.checked_file(base, TRAIN_FILE):
        raise ValueError("regenerated baseline requires its canonical collection training file")
    required = {TRAIN_FILE: "generated_scientific_input", "splits.json": "split_metadata",
                "provenance/fno-training/run.json": "generation_provenance",
                "provenance/fno-training/COMPLETE.json": "generation_provenance"}
    consumed = {}
    for relative, category in required.items():
        record = records.get(relative)
        if record is None or record.get("category") != category:
            raise ValueError("missing/wrong generated input or provenance category")
        if relative != TRAIN_FILE or full:
            if digest_file(assembly.checked_file(base, relative)) != record["sha256"]:
                raise ValueError("generated input/provenance SHA256 differs")
        consumed[relative] = record["sha256"]
    attempt = assembly.load_attempt("fno-training", base / "provenance" / "fno-training")
    plan = attempt["scientific"]
    canonical = gen.make_plan(argparse.Namespace(dataset="standard", initial_conditions=None,
        subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))
    if (plan.get("schema") != "gift.data-generation.v1" or plan.get("sources") != canonical["sources"]
            or plan.get("dt") != gen.DT
            or plan.get("extra950_origin") != "specified_initial_conditions_seed_unknown"
            or not isinstance(plan.get("initial_conditions_file_sha256"), str)
            or len(plan["initial_conditions_file_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in plan["initial_conditions_file_sha256"])):
        raise ValueError("generation source/PDE/initial-condition origin differs")
    record = records[TRAIN_FILE]
    if (record.get("generation_job") != "fno-training" or record.get("attempt_id") != attempt["run"]["attempt_id"]
            or record["sha256"] != attempt["receipt"]["data_sha256"]):
        raise ValueError("generated input and generation receipt differ")
    if full:
        with h5py.File(path, "r") as handle:
            if (handle.attrs.get("fixture_only", False)
                    or any(token in str(handle.attrs.get(key, "")).upper()
                           for key in ("purpose", "attempt_id") for token in ("MOCK", "TEST", "PILOT"))):
                raise ValueError("test/mock/pilot data cannot qualify as regenerated training input")
    validated = assembly.validate_clean(attempt, full=full, data_path=path)
    splits = assembly.read_json(base / "splits.json")
    if splits.get("schema") != "gift.generated-splits.v1" or splits["files"].get(TRAIN_FILE) != validated[0][3]:
        raise ValueError("generated collection split metadata differs")
    return dict(profile="regenerated", manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                consumed_files=consumed, generation_attempt_id=attempt["run"]["attempt_id"],
                generation_binding_sha256=attempt["run"]["binding_sha256"],
                generation_sources=plan["sources"], initial_conditions_file_sha256=plan["initial_conditions_file_sha256"],
                extra950_origin=plan["extra950_origin"], full_input_checks_performed=full,
                published_dataset_identity=False, numerical_reproduction_verified=False)


def _validate_data(path: Path, config: dict, *, hash_bytes: bool, data_base: Path | None = None) -> dict:
    profile = config.get("data_profile", "released")
    if profile not in ("released", "regenerated"):
        raise ValueError("unknown data profile")
    provenance = None
    if profile == "regenerated":
        if data_base is None:
            raise ValueError("regenerated input requires its explicit collection root")
        provenance = _regenerated_input(data_base, path, full=hash_bytes)
    with h5py.File(path, "r") as handle:
        field = handle["training/vorticity"]
        shape = tuple(field.shape)
        if len(shape) != 4 or shape[0] < config["trajectories"] or shape[2:] != (64, 64):
            raise ValueError("expected at least the selected trajectories in [trajectory,time,64,64]")
        if field.dtype != np.dtype("float32") or shape[1] < 46 + config["rollout"]:
            raise ValueError("field dtype or available frames differ")
        if config["formal"]:
            expected_ids = np.r_[np.arange(50), np.arange(1200, 2150)]
            if shape != (1000, 501, 64, 64) or not np.array_equal(handle["training/trajectory_index"][:], expected_ids):
                raise ValueError("formal training population/shape differs")
            if not np.allclose(handle["training/time"][:], np.arange(501) * 0.02, rtol=0, atol=2e-12):
                raise ValueError("formal training time axis differs")
    actual = (provenance["consumed_files"][TRAIN_FILE] if provenance is not None
              else digest_file(path)) if hash_bytes else None
    if config["formal"] and profile == "released" and actual is not None and actual != TRAIN_SHA:
        raise ValueError("formal dataset SHA256 differs; no training started")
    return dict(shape=list(shape), dtype="float32", bytes=path.stat().st_size,
                sha256=actual, sha256_verified=hash_bytes,
                provenance=provenance or dict(profile="released" if config["formal"] else "nonformal_test",
                    published_dataset_identity=bool(config["formal"] and hash_bytes),
                    full_input_checks_performed=bool(config["formal"] and hash_bytes),
                    numerical_reproduction_verified=False))


def _configure_profile(model: str, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if model in ("uno", "unet"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=model == "uno")
    if model == "unet":
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
            raise ValueError("U-Net's verified training profile needs cuDNN TF32; remove the global override explicitly")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    # FNO and UNO retain the original default/ambient TF32 flags. Record them;
    # the historical metadata do not establish a stronger universal profile.


def epoch_schedule(config: dict, epoch: int, frames: int):
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
    batch, nx, ny, history = context.shape
    axes = [torch.tensor(np.linspace(0, 1, size), dtype=torch.float32) for size in (nx, ny)]
    axes.append(torch.tensor(np.linspace(0, 1, future + 1)[1:], dtype=torch.float32))
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).to(context.device)
    repeated = context.reshape(batch, nx, ny, 1, history).repeat(1, 1, 1, future, 1)
    return torch.cat((coordinates.unsqueeze(0).expand(batch, -1, -1, -1, -1), repeated), -1)


def objective(model, context, target, family: str, loss_function=None, normalizers=None):
    """Only project-specific closed-loop scheduling and calls to native losses."""
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


def _export_terminal(output, network, normalization, config, identity, resumed):
    """A new-schema weights-only product; never masquerade as a published file."""
    attempt = json.loads((output / "ATTEMPT.json").read_text(encoding="utf-8"))
    latest = json.loads((output / "LATEST.json").read_text(encoding="utf-8"))
    artifact = dict(schema="gift.independent-baseline-weights.v1", status="complete", method=config["model"],
                    model_state_dict={key: value.detach().cpu() for key, value in network.state_dict().items()},
                    normalization=normalization, formal_configuration=config,
                    run_provenance=dict(run_id=attempt["run_id"], identity=identity, terminal_boundary=latest),
                    artifact_role="fresh_terminal" if config["formal"] else "test_only",
                    fresh_training=True, same_run_resume_used=resumed,
                    initialization="random_from_seed_no_published_checkpoint",
                    terminal_epoch=config["epochs"])
    with (output / "model.pt").open("xb") as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())


def save_boundary(epoch: int, config: dict, stop: int) -> bool:
    return (not config["formal"] or epoch % config["checkpoint_interval"] == 0
            or epoch == stop or epoch == config["epochs"])


def main(model_name: str, argv=None) -> None:
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
    if not args.tiny and device.type != "cuda":
        raise ValueError("formal-budget training requires CUDA; --tiny explicitly enables CPU tests")
    output = args.output.expanduser().resolve()
    external = Path(os.environ["GIFT_EXTERNAL_ROOT"]).resolve()
    for protected in (ROOT, data_base, external):
        if output == protected or protected in output.parents:
            raise ValueError("run output must be outside source, data and external-code roots")
    _configure_profile(model_name, config["seed"])
    source.pop("directory")
    identity = dict(kind="gift.independent-baseline-training.v1", configuration=config,
                    data=data, data_file=args.data_file, upstream=source,
                    sources={str(path.relative_to(ROOT)): digest_file(path) for path in (
                        Path(__file__), ROOT / "training" / "checkpoints.py",
                        ROOT / "training" / f"train_{model_name}.py", ROOT / "adapters" / "models.py",
                        ROOT / "external_sources.json")},
                    numerical_profile=runtime_identity(), device=str(device))
    if args.data_profile == "regenerated":
        identity["sources"].update({relative: digest_file(ROOT / relative) for relative in (
            "scripts/assemble_generated_data.py", "scripts/generate_data.py", "src/even_full_spectrum_ns.py")})
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
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])
            history, updates, start_epoch = saved["history"], saved["optimizer_updates"], saved["completed_epochs"]
            normalization = saved["normalization"]
            if saved["status"] == "complete":
                raise ValueError("run is already complete; no training repeated")
        else:
            normalization = _fit_normalization(field, config)
        norms = _normalizer_objects(normalization, device) if model_name == "fno3d" else None

        def payload(epoch, status):
            return dict(status=status, model_state_dict=network.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                        normalization=normalization, completed_epochs=epoch, optimizer_updates=updates,
                        history=history, selection="terminal_epoch_not_validation", formal=not args.tiny)

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
        _export_terminal(output, network, normalization, config, identity, resumed)
    print(json.dumps(dict(status=status, completed_epochs=stop, output=str(output),
                          formal=not args.tiny, published_numerical_reproduction_verified=False)))
