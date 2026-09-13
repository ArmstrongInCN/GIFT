"""Resume equivalence and fail-closed checks, not full-budget scientific tests."""
import copy
import random

import numpy as np
import pytest
import torch

from training.checkpoints import CheckpointStore


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
