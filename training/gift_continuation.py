"""Continue a committed training boundary in a separate output directory."""
from __future__ import annotations

import json
from pathlib import Path

from training.checkpoints import CheckpointStore, validate_resume


def open_branch_run(output, identity, *, resume=False, continue_from=None, transition_record=None):
    """Open a branch journal, optionally copying an unfinished committed boundary.

    resume reopens output in place. continue_from creates a separate output,
    checks the parent training configuration, and copies model, optimizer,
    history and RNG state into its initial boundary. Parent files are untouched.
    transition_record is a legacy argument and is meaningful only with a parent.
    """
    from training.gift_prediction_control import _open_run

    output = Path(output).resolve()
    if continue_from is None:
        if transition_record is not None:
            raise ValueError("A transition record requires --continue-from")
        return _open_run(output, identity, resume)
    if resume:
        raise ValueError("Continuation creates a new run; use --resume for the existing run")
    parent = Path(continue_from).resolve(strict=True)
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("Continuation must use a separate output directory")
    attempt = json.loads((parent / "checkpoints/ATTEMPT.json").read_text(encoding="utf-8"))
    validate_resume(identity, attempt["identity"])
    source = CheckpointStore(parent / "checkpoints", attempt["identity"], resume=True)
    payload = source.payload
    if payload.get("completed") or payload.get("epoch", 0) < 1:
        raise ValueError("Only an unfinished committed training boundary can continue")
    identity = dict(identity, continuation={"parent_run_id": attempt["run_id"],
                    "parent_boundary": source.boundary, "parent_directory": str(parent)})
    output, destination = _open_run(output, identity, False)
    source.restore_random_state()
    destination.save(source.boundary, payload)
    return _open_run(output, identity, True)
