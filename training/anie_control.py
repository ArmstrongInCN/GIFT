"""Small shared ANIE IO, optimizer and export functions."""
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from gift.publication_metadata import public_metadata
from training.checkpoints import write_model_export
from training.configuration import output_path

ROOT = Path(__file__).resolve().parents[1]


def checked_paths(args):
    """Resolve the selected observation file, upstream checkout and optional run output.

    The dataset path is relative to GIFT_DATA_ROOT. Return four Paths in the
    order (data_root, data_file, source_root, output); output is None for a
    plan without an output argument. Data and source roots are read-only.
    """
    configured = os.environ.get("GIFT_DATA_ROOT")
    if not configured:
        raise ValueError("set GIFT_DATA_ROOT to the explicit external dataset root")
    data_root = Path(configured).expanduser().resolve(strict=True)
    from gift.paths import data_path
    from training.anie_data import TRAIN_FILE
    path = (data_path(data_root, "four_vortex.fno_training_dt0p02", TRAIN_FILE)
            if args.data_file == TRAIN_FILE else data_root / args.data_file).resolve(strict=True)
    if data_root not in path.parents:
        raise ValueError("ANIE input must stay inside GIFT_DATA_ROOT")
    configured_source = args.source_root or os.environ.get("GIFT_ANIE_SOURCE_ROOT")
    if not configured_source:
        raise ValueError("set --source-root or GIFT_ANIE_SOURCE_ROOT to the fixed official ANIE checkout")
    source_root = Path(configured_source).expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve() if args.output else None
    if output:
        output = output_path(output, ROOT, data_root, source_root)
    return data_root, path, source_root, output


def configure_random(seed):
    """Seed Python, NumPy and Torch before constructing the native ANIE model.

    Use float32 without TF32 or cuDNN benchmark selection, matching the
    precision recorded in the ANIE configuration. CUDA is seeded only when
    available; this function does not choose the training device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def native_training_objects(model, config):
    """Create Adam and cosine scheduling for the supplied native model parameters.

    The learning rate, decay, cosine period and minimum rate come from config.
    Resume loads their state after these objects have been constructed.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler_t_max"], eta_min=config["scheduler_eta_min"])
    return optimizer, scheduler


def step_scheduler(scheduler, zero_based_epoch):
    # The official NS trainer uses `if i > 15: scheduler.step()`.
    """Apply the upstream warm-up rule to a completed zero-based epoch.

    Epoch indices 0 through 15 do not advance the cosine schedule; index 16
    is the first scheduler step, after completed epoch 17.
    """
    if zero_based_epoch > 15:
        scheduler.step()


def training_cost(history, updates):
    """Summarize committed epoch wall time and the actual Adam update count.

    Only history rows already retained by the controller contribute seconds;
    setup, checkpoint writes and work discarded after interruption are excluded.
    """
    return dict(completed_epochs=len(history), optimizer_updates=int(updates),
                committed_epoch_seconds=sum(float(row["epoch_seconds"]) for row in history),
                timing_scope="training window loading, native forward/backward/Adam, validation and CUDA synchronization",
                timing_excludes="initial integrity checks, initialization, checkpoint writing, paused and uncommitted work")


def export_terminal(output, model, config, model_config, identity, history, updates, resumed):
    """Export weights from the completed trajectory-epoch schedule.

    Check consecutive history and update counts before writing model.pt.
    Sparse-time and dense151 runs retain distinct schemas; both exports carry
    CPU tensors and provenance, but omit optimizer state needed for resume.
    """
    if (len(history) != config["epochs"]
            or any(row["epoch"] != i + 1 or row["processed_trajectories"] != config["trajectories"]
                   for i, row in enumerate(history))
            or sum(row["optimizer_updates_this_epoch"] for row in history) != updates):
        raise ValueError("ANIE terminal export requires a complete consistent trajectory-epoch history")
    attempt = json.loads((output / "ATTEMPT.json").read_text(encoding="utf-8"))
    latest = json.loads((output / "LATEST.json").read_text(encoding="utf-8"))
    method = config["method"]
    schema = "gift.anie-dense-terminal-weights.v1" if method == "anie_dense151" else "gift.anie-terminal-weights.v1"
    artifact = dict(schema=schema, method=method,
                    status="complete" if config["formal"] else "diagnostic_complete",
                    model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    model_configuration=model_config, formal_configuration=config,
                    completed_epochs=config["epochs"], terminal_epoch=config["epochs"],
                    artifact_role="fresh_terminal" if config["formal"] else "test_only",
                    fresh_training=True, same_run_resume_used=resumed,
                    selection="terminal_epoch_not_validation",
                    training_cost=training_cost(history, updates),
                    run_provenance=dict(run_id=attempt["run_id"], identity=public_metadata(identity),
                                        terminal_boundary=public_metadata(latest)))
    write_model_export(output / "model.pt", artifact)


