"""Small CPU checks for exact state/RNG continuation, scientific inputs and atomic writers."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from training.checkpoints import CheckpointStore, runtime_identity


def test_resume_replays_optimizer_scheduler_loader_and_global_rng(tmp_path):
    """Restored state must reproduce the next update despite unrelated initialization/RNG draws."""
    torch.manual_seed(17)
    random.seed(18)
    np.random.seed(19)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    generator = torch.Generator().manual_seed(20)

    def step():
        """Make one update consuming private loader RNG and global Python/NumPy/Torch randomness."""
        x = torch.randn(4, 3, generator=generator) + torch.randn(4, 3)
        x = x + random.random() + float(np.random.random())
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        scheduler.step()

    step()
    store = CheckpointStore(tmp_path / "checkpoints", {"seed": 17, "steps": 2})
    store.save("step_1", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "loader_rng": generator.get_state()})
    step()
    expected = copy.deepcopy(model.state_dict())
    torch.manual_seed(900)
    restored = CheckpointStore(tmp_path / "checkpoints", {"seed": 17, "steps": 2}, resume=True)
    payload = restored.payload
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    generator.set_state(payload["loader_rng"])
    restored.restore_random_state()
    step()
    assert all(torch.equal(value, expected[key]) for key, value in model.state_dict().items())


def test_resume_rejects_changed_identity_and_corruption(tmp_path):
    """A changed seed or appended checkpoint bytes must prevent restoring the boundary."""
    store = CheckpointStore(tmp_path / "checkpoints", {"seed": 1})
    path = store.save("step_0", {"model": torch.zeros(1)})
    with pytest.raises(ValueError, match="configuration differs"):
        CheckpointStore(tmp_path / "checkpoints", {"seed": 2}, resume=True)
    with path.open("ab") as stream:
        stream.write(b"corrupt test fixture")
    with pytest.raises(ValueError, match="size differs"):
        CheckpointStore(tmp_path / "checkpoints", {"seed": 1}, resume=True)


def test_two_resumers_cannot_silently_fork_one_run(tmp_path):
    """Once one writer commits, another reader of the old pointer must be rejected as stale."""
    store = CheckpointStore(tmp_path / "checkpoints", {"seed": 3})
    store.save("step_0", {"model": torch.zeros(1)})
    first = CheckpointStore(tmp_path / "checkpoints", {"seed": 3}, resume=True)
    second = CheckpointStore(tmp_path / "checkpoints", {"seed": 3}, resume=True)
    first.save("step_1", {"model": torch.ones(1)})
    with pytest.raises(RuntimeError, match="Another writer advanced"):
        second.save("step_1", {"model": torch.zeros(1)})


def test_runtime_identity_records_thread_counts_and_missing_values(monkeypatch):
    """Record actual Torch pools and distinguish missing environment settings from empty values."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "")
    record = runtime_identity()["threading"]
    assert record["torch_num_threads"] == torch.get_num_threads()
    assert record["torch_num_interop_threads"] == torch.get_num_interop_threads()
    assert record["environment"]["OMP_NUM_THREADS"] is None
    assert record["environment"]["MKL_NUM_THREADS"] == ""


def test_source_environment_and_operational_changes_do_not_lock_resume(tmp_path, monkeypatch):
    """Comment/environment/IO changes are compatible, while changing the declared epoch budget is not."""
    identity = {"seed": 4, "configuration": {"epochs": 2, "checkpoint_interval": 1,
                "cache_observations": False, "execution_requested": "auto"}, "sources": {"code": "old"}}
    store = CheckpointStore(tmp_path / "checkpoints", identity)
    store.save("epoch_0001", {"epoch": 1, "history": [{"epoch": 1}], "model": torch.zeros(1)})
    before = (store.directory / "ATTEMPT.json").read_bytes()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    changed = copy.deepcopy(identity)
    changed["sources"]["code"] = "new"
    changed["configuration"].update(checkpoint_interval=3, cache_observations=True, execution_requested="eager")
    resumed = CheckpointStore(store.directory, changed, resume=True)
    assert torch.equal(resumed.payload["model"], torch.zeros(1))
    assert (store.directory / "ATTEMPT.json").read_bytes() == before
    changed["configuration"]["epochs"] = 3
    with pytest.raises(ValueError, match="configuration differs"):
        CheckpointStore(store.directory, changed, resume=True)


def test_legacy_sha_pointer_is_read_without_rewriting_old_record(tmp_path):
    """A legacy digest-only pointer must load without rewriting its original bytes."""
    store = CheckpointStore(tmp_path / "checkpoints", {"seed": 1})
    store.save("step_0", {"model": torch.ones(1)})
    pointer = store.directory / "LATEST.json"
    value = json.loads(pointer.read_text())
    value.pop("bytes")
    value["sha256"] = "old-descriptive-hash"
    pointer.write_text(json.dumps(value))
    before = pointer.read_bytes()
    resumed = CheckpointStore(store.directory, {"seed": 1}, resume=True)
    assert torch.equal(resumed.payload["model"], torch.ones(1))
    assert pointer.read_bytes() == before


def test_invalid_progress_is_rejected(tmp_path):
    """An explicit epoch cursor must agree with the number of committed history rows."""
    store = CheckpointStore(tmp_path / "checkpoints", {"seed": 1})
    store.save("epoch_0001", {"epoch": 1, "history": [], "model": torch.ones(1)})
    with pytest.raises(ValueError, match="history and epoch"):
        CheckpointStore(store.directory, {"seed": 1}, resume=True)


def test_flat_branch_runtime_metadata_does_not_block_resume(tmp_path, monkeypatch):
    """Legacy flat and current nested runtime provenance must remain compatible without weakening seed checks."""
    from types import SimpleNamespace
    from experiments.formal import train_gift_branches as trainer

    data, low = tmp_path / "observations.bin", tmp_path / "generator.bin"
    data.write_bytes(b"small metadata fixture")
    low.write_bytes(b"not actual model weights")
    paths = SimpleNamespace(standard_n64=data, low_model=low)
    scales = (1.0, 1.0, 1.0, 1.0)
    original = trainer._training_identity(7, paths, scales, torch.device("cpu"))
    # Historical journals stored these fields flat; newer producers may group
    # them under runtime. Exercise both representations during migration.
    original.update(original.pop("runtime", {}))
    store = CheckpointStore(tmp_path / "run", original)
    store.save("derivative_1", {"model": torch.ones(1), "phase": "derivative", "next_index": 2})
    before = (store.directory / "ATTEMPT.json").read_bytes()
    monkeypatch.setattr(trainer.h5py, "__version__", "changed-h5py-version")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 17)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 19)
    monkeypatch.setenv("OMP_NUM_THREADS", "23")
    monkeypatch.setenv("MKL_NUM_THREADS", "29")
    changed = trainer._training_identity(7, paths, scales, torch.device("cpu"))
    changed_runtime = changed.get("runtime", changed)
    for key in ("h5py", "torch_threads", "torch_interop_threads", "omp_threads", "mkl_threads"):
        assert changed_runtime[key] != original[key]
    resumed = CheckpointStore(store.directory, changed, resume=True)
    assert torch.equal(resumed.payload["model"], torch.ones(1))
    trainer._preflight_resume(store.directory, 7, paths, torch.device("cpu"), None)
    assert (store.directory / "ATTEMPT.json").read_bytes() == before
    changed["seed"] = 8
    with pytest.raises(ValueError, match="configuration differs"):
        CheckpointStore(store.directory, changed, resume=True)


def test_phase_history_does_not_require_one_global_epoch_sequence(tmp_path):
    """Phase-local epoch history is valid unless the payload declares a contradictory global epoch."""
    store = CheckpointStore(tmp_path / "run", {"seed": 3})
    history = [{"phase": "derivative", "epoch": 1}, {"phase": "derivative", "epoch": 2},
               {"phase": "short", "epoch": 1}]
    store.save("short_1", {"phase": "short", "next_index": 2, "history": history})
    resumed = CheckpointStore(store.directory, {"seed": 3}, resume=True)
    assert resumed.payload["history"] == history
    from training.configuration import validate_training_state
    with pytest.raises(ValueError, match="not consecutive"):
        validate_training_state({"epoch": 3, "history": history})


def test_legacy_input_file_sizes_are_descriptive_without_rewriting(tmp_path):
    """Changing or omitting old file-size provenance must not rewrite or disqualify a saved boundary."""
    identity = {"seed": 3, "dataset_bytes": 100, "data_bytes": 200}
    store = CheckpointStore(tmp_path / "run", identity)
    store.save("initial", {"model": torch.ones(1)})
    before = (store.directory / "ATTEMPT.json").read_bytes()
    CheckpointStore(store.directory, {"seed": 3}, resume=True)
    CheckpointStore(store.directory, {"seed": 3, "dataset_bytes": 300, "data_bytes": 400}, resume=True)
    assert (store.directory / "ATTEMPT.json").read_bytes() == before


def test_legacy_collection_receipts_do_not_qualify_resume(tmp_path):
    """Ignore obsolete collection receipts but still reject a changed loaded-array fingerprint."""
    data = {"condition": "noise_000", "selected_arrays_sha256": "actual-arrays"}
    legacy = dict(data, profile="regenerated", manifest_sha256="old-manifest",
                  consumed_files=[{"file": "input.h5", "sha256": "old-file", "bytes": 100}],
                  generation_attempts=[{"attempt_id": "old-attempt"}])
    store = CheckpointStore(tmp_path / "run", {"seed": 3, "input_provenance": legacy})
    store.save("initial", {"model": torch.ones(1)})
    current = {"seed": 3, "input_provenance": data}
    CheckpointStore(store.directory, current, resume=True)
    current["input_provenance"] = dict(data, selected_arrays_sha256="changed-arrays")
    with pytest.raises(ValueError, match="scientific inputs differ"):
        CheckpointStore(store.directory, current, resume=True)
