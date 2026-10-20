"""All GIFT trainers share one explicit, resumable execution selection policy."""
import json

import pytest
import torch

from training.gift_execution import configure_training_runtime, resolve_execution
from experiments.formal.train_low_generator import parse_args


@pytest.mark.parametrize("device,expected", [("cuda", "cuda-graph"), ("cpu", "eager")])
def test_fresh_auto_backend(device, expected):
    """Resolve fresh automatic selection to CUDA graphs on GPU and eager execution on CPU."""
    assert resolve_execution("auto", device) == expected


@pytest.mark.parametrize("saved", ["eager", "cuda-graph"])
def test_auto_resume_inherits_backend_only(tmp_path, saved):
    """Automatic resume inherits recorded execution while an explicit backend remains a caller choice."""
    (tmp_path/"ATTEMPT.json").write_text(json.dumps({"identity":{"execution":saved}}))
    assert resolve_execution("auto", "cuda", resume=True, checkpoint_directory=tmp_path) == saved
    # An explicit backend is a caller choice; saved training state is checked
    # separately by the controller.
    assert resolve_execution("eager", "cuda", resume=True, checkpoint_directory=tmp_path) == "eager"


def test_bad_record_and_missing_journal_are_not_fresh_training(tmp_path):
    """A missing or invalid resume record must raise rather than silently initialize a fresh run."""
    with pytest.raises(FileNotFoundError):
        resolve_execution("auto", "cuda", resume=True, checkpoint_directory=tmp_path)
    with pytest.raises(ValueError):
        resolve_execution("auto", "cuda", resume=True)
    (tmp_path/"ATTEMPT.json").write_text(json.dumps({"identity":{"execution":"unknown"}}))
    with pytest.raises(ValueError, match="saved training backend"):
        resolve_execution("auto", "cuda", resume=True, checkpoint_directory=tmp_path)
    with pytest.raises(ValueError, match="unknown GIFT"):
        resolve_execution("bad", "cpu")


def test_generator_cli_defaults_to_shared_auto_policy():
    """Retain the shared automatic default while accepting an explicit eager reference run."""
    assert parse_args(["--dry-run"]).execution == "auto"
    assert parse_args(["--dry-run", "--execution", "eager"]).execution == "eager"


@pytest.mark.parametrize("strict", [False, True])
def test_runtime_policy_preserves_precision_rng_and_existing_workspace(monkeypatch, strict):
    """Kernel determinism settings must preserve RNG/precision and respect an existing cuBLAS workspace choice."""
    before = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
              torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    rng = torch.get_rng_state().clone()
    precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.get_float32_matmul_precision())
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    try:
        configure_training_runtime(strict=strict)
        import os
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert torch.are_deterministic_algorithms_enabled() is strict
        assert torch.is_deterministic_algorithms_warn_only_enabled() is (not strict)
        assert torch.equal(rng, torch.get_rng_state())
        assert precision == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                             torch.get_float32_matmul_precision())
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
        configure_training_runtime(strict=strict)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = before[:2]
        torch.use_deterministic_algorithms(before[2], warn_only=before[3])
