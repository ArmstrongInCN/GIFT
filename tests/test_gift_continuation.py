"""Continue unfinished branch state into a new directory while retaining the parent journal."""
import json

import pytest
import torch

from training.checkpoints import CheckpointStore, capture_rng, digest_file
from training.gift_continuation import open_branch_run


def prepare(tmp_path):
    """Create an unfinished parent journal plus a legacy transition record for compatibility tests."""
    parent = tmp_path / "parent"
    old = {"role": "gift_prediction_branch", "seed": 20, "sources": {"a": "old"}}
    store = CheckpointStore(parent / "checkpoints", old)
    payload = {"epoch": 7, "completed": False, "model": {"w": torch.arange(3)},
               "optimizer": {"step": 99}, "history": list(range(7)), "committed_seconds": 12.0}
    store.save("epoch_0007", payload)
    target = {**old, "sources": {"a": "new"}, "execution": "cuda-graph"}
    evidence = tmp_path / "proof.json"
    evidence.write_text('{"pass": true}', encoding="utf-8")
    record = {"schema": "gift.validated-execution-transition.v1",
              "parent_sources": old["sources"], "target_sources": target["sources"],
              "parent_execution": "eager", "target_execution": "cuda-graph",
              "parent_checkpoint_sha256": "legacy-record-not-required",
              "validation": "bitwise_equal_model_optimizer_rng_and_numeric_history",
              "evidence": [{"file": evidence.name, "sha256": digest_file(evidence)}]}
    path = tmp_path / "transition.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return parent, target, path


def test_preserves_payload_rng_and_parent(tmp_path):
    """Continuation must copy committed state/RNG, preserve parent bytes and still reject a changed seed."""
    parent, target, record = prepare(tmp_path)
    original = {p.name: digest_file(p) for p in (parent / "checkpoints").iterdir()}
    old = json.loads((parent / "checkpoints/ATTEMPT.json").read_text())["identity"]
    source = CheckpointStore(parent / "checkpoints", old, resume=True)
    output, store = open_branch_run(tmp_path / "child", target, continue_from=parent,
                                    transition_record=record)
    assert torch.equal(store.payload["model"]["w"], source.payload["model"]["w"])
    assert store.payload["optimizer"] == source.payload["optimizer"]
    assert store.payload["history"] == source.payload["history"]
    assert torch.equal(store.rng["torch_cpu"], source.rng["torch_cpu"])
    assert store.rng["python"] == source.rng["python"]
    assert {p.name: digest_file(p) for p in (parent / "checkpoints").iterdir()} == original
    _, resumed = open_branch_run(output, target, resume=True)
    assert resumed.payload["epoch"] == store.payload["epoch"]
    with pytest.raises(ValueError, match="configuration differs"):
        open_branch_run(output, {**target, "seed": 21}, resume=True)
    assert CheckpointStore(parent / "checkpoints", target, resume=True).payload["epoch"] == 7


@pytest.mark.parametrize("change", ["seed", "overwrite"])
def test_rejects_unbound_transition(tmp_path, change):
    """Reject a changed scientific seed or a continuation output that overlaps its parent."""
    parent, target, path = prepare(tmp_path)
    record = json.loads(path.read_text())
    output = tmp_path / "child"
    if change == "seed": target["seed"] = 21
    if change == "hash": record["parent_checkpoint_sha256"] = "bad"
    if change == "source": record["target_sources"] = {}
    if change == "evidence": record["evidence"][0]["sha256"] = "bad"
    if change == "overwrite": output = parent
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        open_branch_run(output, target, continue_from=parent, transition_record=path)
