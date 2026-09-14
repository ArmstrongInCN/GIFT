"""Resume equivalence and fail-closed checks, not full-budget scientific tests."""
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
    torch.manual_seed(17)
    random.seed(18)
    np.random.seed(19)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    generator = torch.Generator().manual_seed(20)

    def step():
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
    store = CheckpointStore(tmp_path / "checkpoints", {"seed": 1})
    path = store.save("step_0", {"model": torch.zeros(1)})
    with pytest.raises(ValueError, match="identity differs"):
        CheckpointStore(tmp_path / "checkpoints", {"seed": 2}, resume=True)
    with path.open("ab") as stream:
        stream.write(b"corrupt test fixture")
    with pytest.raises(ValueError, match="SHA256"):
        CheckpointStore(tmp_path / "checkpoints", {"seed": 1}, resume=True)


def test_two_resumers_cannot_silently_fork_one_run(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints", {"seed": 3})
    store.save("step_0", {"model": torch.zeros(1)})
    first = CheckpointStore(tmp_path / "checkpoints", {"seed": 3}, resume=True)
    second = CheckpointStore(tmp_path / "checkpoints", {"seed": 3}, resume=True)
    first.save("step_1", {"model": torch.ones(1)})
    with pytest.raises(RuntimeError, match="Another writer advanced"):
        second.save("step_1", {"model": torch.zeros(1)})


def test_runtime_identity_records_thread_counts_and_missing_values(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "")
    record = runtime_identity()["threading"]
    assert record["torch_num_threads"] == torch.get_num_threads()
    assert record["torch_num_interop_threads"] == torch.get_num_interop_threads()
    assert record["environment"]["OMP_NUM_THREADS"] is None
    assert record["environment"]["MKL_NUM_THREADS"] == ""


@pytest.mark.parametrize("name", [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_DYNAMIC", "MKL_DYNAMIC",
])
def test_resume_rejects_changed_thread_environment_without_reading_weights(tmp_path, monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    store = CheckpointStore(tmp_path / "checkpoints", {"fixture": "thread-environment"})
    store.save("step_0", {"model": torch.zeros(1)})
    # An empty entry is still a launch-environment change, not an absent key.
    monkeypatch.setenv(name, "")
    def forbidden_load(*args, **kwargs):
        raise AssertionError("A changed runtime must be rejected before loading weights")
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="numerical runtime differs"):
        CheckpointStore(tmp_path / "checkpoints", {"fixture": "thread-environment"}, resume=True)


def test_missing_thread_binding_is_rejected(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints", {"fixture": "missing-threading"})
    store.save("step_0", {"model": torch.zeros(1)})
    incomplete = copy.deepcopy(store.runtime)
    incomplete.pop("threading")
    # Corrupt only this synthetic fixture; real training records are untouched.
    path = store.directory / "ATTEMPT.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["runtime"] = incomplete
    path.write_text(json.dumps(record), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="numerical runtime differs"):
        CheckpointStore(store.directory, {"fixture": "missing-threading"}, resume=True)
    assert path.read_bytes() == before


def test_fresh_process_resume_checks_actual_torch_thread_pools(tmp_path):
    """Tiny state IO only: no baseline model, dataset, optimizer or GPU training."""
    directory = tmp_path / "subprocess-journal"
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json, sys
from pathlib import Path
import torch
from training.checkpoints import CheckpointStore
torch.set_num_threads(int(sys.argv[3]))
torch.set_num_interop_threads(int(sys.argv[4]))
identity = {"fixture": "new-process-thread-pools-not-scientific-training"}
try:
    store = CheckpointStore(Path(sys.argv[1]), identity, resume=sys.argv[2] != "create")
    if sys.argv[2] == "create":
        store.save("step_0", {"value": torch.tensor([3.0])})
    else:
        assert torch.equal(store.payload["value"], torch.tensor([3.0]))
except ValueError as error:
    if sys.argv[2] != "reject" or "numerical runtime differs" not in str(error):
        raise
    print(json.dumps({"status": "REJECTED_CHANGED_RUNTIME"}))
else:
    if sys.argv[2] == "reject":
        raise AssertionError("Changed numerical runtime was accepted")
    print(json.dumps({"status": "PASS", "threading": store.runtime["threading"]}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # Child imports must use the project under test, not a caller's Python path.
    env.pop("PYTHONPATH", None)
    def invoke(mode, intra=1, inter=1, **changes):
        result = subprocess.run([sys.executable, "-s", "-B", "-c", script,
                                 str(directory), mode, str(intra), str(inter)],
                                cwd=root, env=dict(env, **changes), text=True,
                                capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert invoke("create")["threading"]["torch_num_threads"] == 1
    def inventory():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.iterdir() if path.is_file()}
    before = inventory()
    assert invoke("resume")["status"] == "PASS"
    assert invoke("reject", intra=2)["status"] == "REJECTED_CHANGED_RUNTIME"
    assert invoke("reject", inter=2)["status"] == "REJECTED_CHANGED_RUNTIME"
    assert invoke("reject", OMP_NUM_THREADS="2")["status"] == "REJECTED_CHANGED_RUNTIME"
    assert inventory() == before
