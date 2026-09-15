"""Explicit, source-bound continuation into a new training journal.

Ordinary resume remains strict. A reviewed backend/source transition requires
an exact parent checkpoint hash and a record binding both implementations and
the numerical validation evidence. It never overwrites the parent's journal.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.checkpoints import CheckpointStore, digest_file


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def open_branch_run(output, identity, *, resume=False, continue_from=None,
                    transition_record=None):
    from training.gift_prediction_control import _open_run, write_json_new

    identity = canonical(identity)
    output = Path(output).resolve()
    if continue_from is None:
        if transition_record is not None:
            raise ValueError("A transition record requires --continue-from")
        if resume:
            previous = json.loads((output / "checkpoints/ATTEMPT.json").read_text(encoding="utf-8"))
            if "continuation" in previous["identity"]:
                identity["continuation"] = previous["identity"]["continuation"]
        return _open_run(output, identity, resume)
    if resume or transition_record is None:
        raise ValueError("Explicit continuation needs a transition record and a NEW output")
    parent = Path(continue_from).resolve(strict=True)
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("Continuation must use a separate output directory")
    attempt = json.loads((parent / "checkpoints/ATTEMPT.json").read_text(encoding="utf-8"))
    record_path = Path(transition_record).resolve(strict=True)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    old = attempt["identity"]
    old_execution = old.get("execution", "eager")
    before = {k: v for k, v in old.items() if k not in ("sources", "execution", "continuation")}
    after = {k: v for k, v in identity.items() if k not in ("sources", "execution", "continuation")}
    if before != after:
        raise ValueError("Continuation cannot change data, seed, model, budget, scales or device")
    if (record.get("schema") != "gift.validated-execution-transition.v1"
            or record.get("parent_sources") != old["sources"]
            or record.get("target_sources") != identity["sources"]
            or record.get("parent_execution") != old_execution
            or record.get("target_execution") != identity["execution"]
            or record.get("validation") != "bitwise_equal_model_optimizer_rng_and_numeric_history"):
        raise ValueError("Transition record does not bind these exact implementations")
    evidence = record.get("evidence", [])
    if not evidence:
        raise ValueError("Numerical validation evidence is required")
    for item in evidence:
        evidence_path = (record_path.parent / item["file"]).resolve(strict=True)
        if evidence_path.parent != record_path.parent or digest_file(evidence_path) != item["sha256"]:
            raise ValueError("Transition evidence hash or location differs")
    # The normal loader checks the parent's own source identity, runtime, pointer
    # and checkpoint digest. No identity is edited or passed off as unchanged.
    source = CheckpointStore(parent / "checkpoints", old, resume=True)
    pointer = json.loads((parent / "checkpoints/LATEST.json").read_text(encoding="utf-8"))
    if record.get("parent_checkpoint_sha256") != pointer["sha256"]:
        raise ValueError("Parent advanced or the requested checkpoint hash differs")
    payload = source.payload
    if payload.get("completed") or payload.get("epoch", 0) < 1:
        raise ValueError("Only an unfinished committed training boundary can continue")
    identity["continuation"] = {
        "parent_run_id": attempt["run_id"],
        "parent_checkpoint_sha256": pointer["sha256"],
        "parent_boundary": source.boundary,
        "completed_epochs": payload["epoch"],
        "parent_sources": old["sources"],
        "parent_execution": old_execution,
        "parent_continuation": old.get("continuation"),
        "transition_record_sha256": digest_file(record_path),
        "validation": record["validation"],
        "evidence": [{"sha256": item["sha256"]} for item in evidence],
    }
    output, destination = _open_run(output, identity, False)
    # Saving the unchanged payload after restoring all RNG streams preserves
    # the exact next update; this boundary performs no optimizer step.
    source.restore_random_state()
    destination.save(source.boundary, payload)
    write_json_new(output / "CONTINUATION.json", identity["continuation"])
    return _open_run(output, identity, True)
