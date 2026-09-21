"""CPU-only export recovery from one explicit immutable complete baseline journal.

Default is a no-read plan. This is not training, continuation or numerical acceptance.
No private audit helper, historical run name, dataset, upstream checkout or LATEST.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

# Metadata contract only; no architecture, solver, objective or optimizer code.
# epochs, rollout, physical batch, accumulation, weight decay, group IDs, states
# The four independent baseline training budgets: (epochs, rollout, physical
# batch, accumulation, weight decay, optimizer param-groups, optimizer states).
# These are the only accepted terminal boundaries for a recovered baseline export.
BUDGETS = {"uno": (500, 20, 16, 1, 1e-5, 36, 36),
           "fno2d": (500, 150, 10, 2, 1e-4, 30, 30),
           "fno3d": (500, 150, 5, 2, 1e-4, 38, 30),
           "unet": (500, 4, 20, 1, 1e-4, 36, 36)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return dict(file=Path(path).name, bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


# Verify the immutable training journal against the ATTEMPT record: identity,
# runtime, the unchanged formal budget, recorded input binding and the complete
# terminal-epoch boundary. Any mismatch rejects the export as unverified.
def validate_metadata(saved, attempt):
    require(saved["schema"] == "gift.training-boundary.v1"
        and attempt["schema"] == "gift.training-attempt.v1", "unsupported journal schema")
    identity = saved["identity"]
    require(identity == attempt["identity"] and saved["runtime"] == attempt["runtime"]
        and saved["runtime"] == identity["numerical_profile"], "ATTEMPT/checkpoint identity or runtime mismatch")
    require(re.fullmatch(r"[a-f0-9]{32}", attempt["run_id"]) is not None
        and identity["kind"] == "gift.independent-baseline-training.v1", "invalid ATTEMPT run identity")
    config, payload = identity["configuration"], saved["payload"]
    model = config["model"]
    require(model in BUDGETS, "unsupported baseline")
    epochs, rollout, batch, accum, decay, groups, states = BUDGETS[model]
    expected = dict(epochs=epochs, trajectories=1000, history=46, rollout=rollout,
        micro_batch=batch, accumulation=accum, learning_rate=.001, weight_decay=decay,
        seed=0, formal=True, scheduler_step=100, scheduler_gamma=.5,
        selection="terminal_epoch_not_validation", teacher_forcing=False, detach_rollout=False)
    require(all(config.get(k) == v for k, v in expected.items())
        and config["data_profile"] in ("released", "regenerated")
        and type(config["checkpoint_interval"]) is int and config["checkpoint_interval"] > 0,
        "not the unchanged complete formal budget")
    require(identity["sources"] and all(isinstance(v, str) and re.fullmatch(r"[a-fA-F0-9]{64}", v)
        for v in identity["sources"].values()), "missing recorded source hashes")
    data = identity["data"]
    require(data["sha256_verified"] is True and re.fullmatch(r"[a-fA-F0-9]{64}", data["sha256"])
        and data["shape"] == [1000, 501, 64, 64] and data["dtype"] == "float32", "missing recorded input binding")
    per_epoch = math.ceil(1000/(batch*accum))
    updates = epochs*per_epoch
    require(saved["boundary"] == "epoch_%04d" % epochs and payload["status"] == "complete"
        and payload["completed_epochs"] == epochs and payload["optimizer_updates"] == updates
        and payload["formal"] is True and payload["selection"] == "terminal_epoch_not_validation",
        "journal is not a complete formal terminal boundary")
    require(len(payload["history"]) == epochs, "incomplete history")
    for epoch, row in enumerate(payload["history"], 1):
        require(row["epoch"] == epoch and row["optimizer_updates"] == epoch*per_epoch
            and row["processed_trajectories"] == 1000 and row["formal"] is True
            and row["learning_rate_used"] == .001*.5**((epoch-1)//100)
            and row["learning_rate_next"] == .001*.5**(epoch//100)
            and all(math.isfinite(row[k]) for k in ("backward_loss_mean_per_trajectory", "full_loss_mean_per_trajectory")),
            "history budget/order/finite loss mismatch")
    optimizer = payload["optimizer_state_dict"]
    require(len(optimizer["param_groups"]) == 1, "optimizer group missing")
    group, state = optimizer["param_groups"][0], optimizer["state"]
    ids = group["params"]
    require(all(type(i) is int for i in ids) and ids == list(range(groups))
        and len(set(ids)) == groups and len(state) == states, "optimizer coverage incomplete")
    inactive = set(range(26, 34)) if model == "fno3d" else set()
    require(set(state) == set(ids)-inactive and all(set(s) == {"step", "exp_avg", "exp_avg_sq"}
        and float(s["step"]) == updates for s in state.values()), "optimizer slot coverage or final step mismatch")
    lr = .001*.5**(epochs//100)
    require(group["lr"] == lr and tuple(group["betas"]) == (.9, .999) and group["weight_decay"] == decay,
        "terminal optimizer settings differ")
    schedule = payload["scheduler_state_dict"]
    require(schedule["last_epoch"] == epochs and schedule["step_size"] == 100
        and schedule["gamma"] == .5 and schedule["_last_lr"] == [lr], "terminal scheduler differs")
    require(set(saved["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "continuation RNG inventory incomplete")
    return config


def check_tensors(payload, model, torch):
    state = payload["model_state_dict"]
    require(isinstance(state, dict) and state and all(isinstance(k, str) and isinstance(v, torch.Tensor)
        and v.device.type == "cpu" and bool(torch.isfinite(v).all()) for k, v in state.items()), "invalid model tensors")
    for slots in payload["optimizer_state_dict"]["state"].values():
        a, b = slots["exp_avg"], slots["exp_avg_sq"]
        require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            and a.shape == b.shape and a.dtype == b.dtype and a.device.type == b.device.type == "cpu"
            and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "invalid Adam moments")
    norm = payload["normalization"]
    if model != "fno3d":
        require(norm == {"kind": "none"}, "unexpected normalizers")
        return
    keys = {"input_mean", "input_std", "output_mean", "output_std"}
    require(set(norm) == keys | {"kind", "eps"} and norm["kind"] == "paper_UnitGaussianNormalizer"
        and norm["eps"] == 1e-5, "FNO3D normalizer schema differs")
    for key in keys:
        value = norm[key]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float32
            and tuple(value.shape) == (64, 64, 46 if key.startswith("input") else 150)
            and bool(torch.isfinite(value).all()) and (not key.endswith("std") or bool((value > 0).all())), "invalid FNO3D normalizer")


def artifact(saved, attempt, checkpoint_record, attempt_record):
    config, payload = saved["identity"]["configuration"], saved["payload"]
    result = dict(schema="gift.recovered-baseline-weights.v1", status="complete", method=config["model"],
        model_state_dict=payload["model_state_dict"], normalization=payload["normalization"],
        formal_configuration=config, terminal_epoch=config["epochs"], artifact_role="recovered_terminal",
        fresh_training=None, same_run_resume_used=None,
        initialization="source_bound_independent_baseline_protocol_not_recorded_in_checkpoint",
        run_provenance=dict(run_id=attempt["run_id"], identity=saved["identity"],
            terminal_boundary={k: checkpoint_record[k] for k in ("file", "sha256")}),
        recovery_export=dict(checkpoint=checkpoint_record, attempt=attempt_record,
            identity_binding="ATTEMPT/SHA/same-directory + matching identity/runtime; checkpoint has no internal run_id",
            unknown_fields=["fresh_training", "same_run_resume_used"],
            source_and_data_hashes="preserved recorded bindings; source, upstream and datasets not reread",
            original_invocation_completion="not inferred; export can recover after crash without an END receipt",
            scientific_acceptance_verified=False, training_or_forward_performed=False))
    # Preserve measured cost when recorded; never infer missing timing from epochs.
    if "training_cost" in payload:
        result["training_cost"] = payload["training_cost"]
    return result


def atomic_create(destination, writer, verify):
    """Flush a new sibling, verify it, then publish with no-overwrite hard-link.

    Failure leaves the uniquely named sibling for inspection; never overwrites.
    Local same-filesystem hard-link support is required (e.g. Windows NTFS).
    """
    # Write to a uniquely named sibling, verify the roundtrip, then publish by
    # hard link which fails if the destination already exists (create-only).
    require(not destination.exists(), "destination already exists")
    temporary = destination.parent / ("." + destination.name + ".recovery-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    verify(temporary)
    os.link(temporary, destination)  # Atomic name creation; fails if destination won a race.
    temporary.unlink()  # Only this tool's exact new sibling, never an old artifact.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--attempt-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PLAN_NOT_EXECUTED", files_read=0, files_written=0,
            note="Explicit checkpoint/ATTEMPT/SHA and new output required; no LATEST, training or dataset read.")))
        return 0
    loaded_torch = sys.modules.get("torch")
    require(loaded_torch is None or not loaded_torch.cuda.is_initialized(),
        "cannot recover inside a process that already initialized CUDA")
    require(all(getattr(args, k) is not None for k in ("checkpoint", "checkpoint_sha256", "attempt", "attempt_sha256", "output")), "explicit inputs, hashes and output required")
    checkpoint, attempt_path = args.checkpoint.resolve(strict=True), args.attempt.resolve(strict=True)
    destination = args.output.resolve()
    require(checkpoint.parent == attempt_path.parent and attempt_path.name == "ATTEMPT.json"
        and re.fullmatch(r"epoch_[0-9]{4}_[a-f0-9]{32}\.pt", checkpoint.name), "explicit same-directory immutable journal required")
    require(destination.parent.is_dir() and destination.suffix == ".pt" and destination not in (checkpoint, attempt_path)
        and not destination.exists(), "explicit new .pt file in existing output directory required")
    before = [record(checkpoint), record(attempt_path)]
    require(before[0]["sha256"] == args.checkpoint_sha256.lower() and before[1]["sha256"] == args.attempt_sha256.lower(), "input SHA mismatch")
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    # Force CPU-only recovery so the export never depends on GPU availability or a
    # CUDA context; the weights are copied, not recomputed.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = validate_metadata(saved, attempt)
    require(checkpoint.name.startswith("epoch_%04d_" % config["epochs"]), "filename budget differs")
    check_tensors(saved["payload"], config["model"], torch)
    product = artifact(saved, attempt, before[0], before[1])
    def verify(path):
        copied = torch.load(path, map_location="cpu", weights_only=True)
        require(set(copied) == set(product), "export fields differ")
        for key in product:
            if key == "model_state_dict":
                require(set(copied[key]) == set(product[key]) and all(torch.equal(copied[key][k], v)
                    for k, v in product[key].items()), "roundtrip state differs")
            elif key == "normalization" and config["model"] == "fno3d":
                require(set(copied[key]) == set(product[key]) and all(torch.equal(copied[key][k], v)
                    if isinstance(v, torch.Tensor) else copied[key][k] == v for k, v in product[key].items()), "roundtrip normalizers differ")
            else:
                require(copied[key] == product[key], "roundtrip metadata differs")
        require([record(checkpoint), record(attempt_path)] == before and not torch.cuda.is_initialized(), "input changed or CUDA initialized")
    atomic_create(destination, lambda stream: torch.save(product, stream), verify)
    print(json.dumps(dict(status="RECOVERY_EXPORT_CREATED_NOT_SCIENTIFIC_ACCEPTANCE", output=str(destination),
        output_record=record(destination), inputs_unchanged=True, cuda_initialized=False,
        torch_num_threads=torch.get_num_threads(), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        checkpoint_internal_run_id_bound=False, same_run_resume_used=None), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
