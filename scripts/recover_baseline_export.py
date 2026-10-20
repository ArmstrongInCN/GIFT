"""Export baseline weights from a completed checkpoint without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid


def record(path):
    """Return file name, size and streaming hash when archive verification is requested."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_metadata(saved, attempt=None):
    """Require a completed supported baseline with consecutive history and its actual budget.

    The budget comes from the checkpoint configuration, not a hardcoded
    reference epoch count. Optional attempt metadata never replaces the
    checkpoint configuration or assigns another attempt as its owner.
    """
    config = saved["identity"]["configuration"]
    payload = saved["payload"]
    if config["model"] not in {"uno", "fno2d", "fno3d", "unet"}:
        raise ValueError("Unsupported baseline model")
    epochs = payload["completed_epochs"]
    if payload["status"] != "complete" or epochs != config["epochs"]:
        raise ValueError("Checkpoint is not a complete terminal boundary")
    history = payload.get("history", [])
    if len(history) != epochs or any(row["epoch"] != i + 1 for i, row in enumerate(history)):
        raise ValueError("Incomplete training history")
    return config


def check_tensors(payload, model, torch):
    """Validate finite model tensors and the normalizers actually used at inference.

    FNO-3D requires finite means and positive standard deviations; the other
    supported baselines must declare no normalization.
    """
    state = payload["model_state_dict"]
    if not state or any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
                        for value in state.values()):
        raise ValueError("Invalid model tensors")
    normalization = payload["normalization"]
    if model == "fno3d":
        for key in ("input_mean", "input_std", "output_mean", "output_std"):
            value = normalization[key]
            if not bool(torch.isfinite(value).all()) or (key.endswith("std") and not bool((value > 0).all())):
                raise ValueError("Invalid FNO3D normalizer")
    elif normalization != {"kind": "none"}:
        raise ValueError("Unexpected model normalization")


def artifact(saved, attempt, checkpoint_record, attempt_record):
    """Build an inference-only export from the saved model, normalizers and configuration.

    Optimizer and random states stay in the training checkpoint. Unknown
    fresh-training/resume claims remain unknown; optional ATTEMPT metadata is
    recorded as export context rather than reassigned run ownership.
    """
    config, payload = saved["identity"]["configuration"], saved["payload"]
    result = {
        "schema": "gift.recovered-baseline-weights.v1", "status": "complete",
        "method": config["model"], "model_state_dict": payload["model_state_dict"],
        "normalization": payload["normalization"], "formal_configuration": config,
        "terminal_epoch": config["epochs"], "artifact_role": "recovered_terminal",
        "fresh_training": None, "same_run_resume_used": None,
        "run_provenance": {
            "identity": saved["identity"], "terminal_boundary": checkpoint_record,
        },
        "recovery_export": {"checkpoint": checkpoint_record, "training_performed": False},
    }
    if attempt:
        result["recovery_export"]["attempt_metadata"] = attempt_record
    if "training_cost" in payload:
        result["training_cost"] = payload["training_cost"]
    return result


def atomic_create(destination, writer, verify):
    """Publish verified bytes to a new file without replacing an existing export.

    writer fills a same-directory staging file; verify reads it after flush
    and fsync. A create-only hard link publishes the verified file. A failed
    write or verification never leaves a partial destination.
    """
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Destination already exists")
    temporary = destination.with_name("." + destination.name + ".recovery-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    verify(temporary)
    os.link(temporary, destination)
    temporary.unlink()


def main(argv=None):
    """Print a no-I/O recovery plan, or export a completed checkpoint on CPU with --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempt", type=Path, help="optional original run metadata")
    parser.add_argument("--checkpoint-sha256", help="optional archive integrity check")
    parser.add_argument("--attempt-sha256", help="optional metadata integrity check")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "PLAN_NOT_EXECUTED", "checkpoint": str(args.checkpoint),
                          "output": str(args.output), "files_read": 0, "files_written": 0}))
        return 0
    if args.checkpoint is None or args.output is None:
        parser.error("--execute requires --checkpoint and --output")
    checkpoint = args.checkpoint.resolve(strict=True)
    destination = args.output.resolve()
    if destination == checkpoint or destination.exists():
        raise ValueError("Output must be a new weights file")
    checkpoint_record = record(checkpoint) if args.checkpoint_sha256 else {"file": checkpoint.name}
    if args.checkpoint_sha256 and checkpoint_record["sha256"] != args.checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA mismatch")
    attempt, attempt_record = None, {}
    if args.attempt:
        attempt_path = args.attempt.resolve(strict=True)
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt_record = record(attempt_path) if args.attempt_sha256 else {"file": attempt_path.name}
        if args.attempt_sha256 and attempt_record["sha256"] != args.attempt_sha256.lower():
            raise ValueError("Attempt SHA mismatch")
    elif args.attempt_sha256:
        parser.error("--attempt-sha256 requires --attempt")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = validate_metadata(saved, attempt)
    check_tensors(saved["payload"], config["model"], torch)
    product = artifact(saved, attempt, checkpoint_record, attempt_record)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def verify(path):
        """Read the staged export and compare every model tensor with the saved checkpoint."""
        exported = torch.load(path, map_location="cpu", weights_only=True)
        if any(not torch.equal(exported["model_state_dict"][key], value)
               for key, value in product["model_state_dict"].items()):
            raise ValueError("Exported model differs from the checkpoint")

    atomic_create(destination, lambda stream: torch.save(product, stream), verify)
    print(json.dumps({"status": "complete", "output": str(destination), "training_performed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
