"""Recovery-only fixtures; never real training data, weights, models or GPU work."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import recover_baseline_export as recovery

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    config = dict(model="uno", epochs=150, trajectories=1000, history=46, rollout=20,
        micro_batch=16, accumulation=1, learning_rate=.001, weight_decay=1e-5, seed=0,
        formal=True, scheduler_step=100, scheduler_gamma=.5, selection="terminal_epoch_not_validation",
        teacher_forcing=False, detach_rollout=False, checkpoint_interval=10, data_profile="released")
    identity = dict(kind="gift.independent-baseline-training.v1", configuration=config,
        fixture_purpose="SYNTHETIC_NOT_TRAINING", numerical_profile={"fixture": "SYNTHETIC_NOT_TRAINING"},
        sources={"synthetic.py": "a"*64},
        data=dict(sha256_verified=True, sha256="b"*64, shape=[1000, 501, 64, 64], dtype="float32"))
    history = [dict(epoch=e, optimizer_updates=e*63, processed_trajectories=1000, formal=True,
        learning_rate_used=.001*.5**((e-1)//100), learning_rate_next=.001*.5**(e//100),
        backward_loss_mean_per_trajectory=1., full_loss_mean_per_trajectory=1.) for e in range(1, 151)]
    payload = dict(status="complete", completed_epochs=150, optimizer_updates=9450, formal=True,
        selection="terminal_epoch_not_validation", history=history,
        model_state_dict={"SYNTHETIC_NOT_TRAINING": None}, normalization={"kind": "none"},
        optimizer_state_dict=dict(param_groups=[dict(params=list(range(36)), lr=.0005, betas=(.9, .999), weight_decay=1e-5)],
            state={i: dict(step=9450, exp_avg=None, exp_avg_sq=None) for i in range(36)}),
        scheduler_state_dict=dict(last_epoch=150, step_size=100, gamma=.5, _last_lr=[.0005]))
    saved = dict(schema="gift.training-boundary.v1", identity=identity, runtime=identity["numerical_profile"],
        boundary="epoch_0150", payload=payload, rng=dict(python=None, numpy=None, torch_cpu=None, torch_cuda=None))
    attempt = dict(schema="gift.training-attempt.v1", identity=copy.deepcopy(identity),
        runtime=identity["numerical_profile"], run_id="a"*32)
    return saved, attempt


def _cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def test_import_and_default_are_isolated_and_do_not_pollute_environment():
    # Other repository tests may already have imported Torch. Check only a fresh
    # child, never infer cleanliness from the enclosing pytest's sys.modules.
    code = r'''
import contextlib, io, json, os, sys
from types import SimpleNamespace
before = dict(os.environ)
from scripts import recover_baseline_export as recovery
assert dict(os.environ) == before
stream = io.StringIO()
with contextlib.redirect_stdout(stream): assert recovery.main([]) == 0
assert json.loads(stream.getvalue())["files_read"] == 0
assert "torch" not in sys.modules and "tensorflow" not in sys.modules
sys.modules["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
try:
    recovery.main(["--execute"])
except ValueError as error:
    assert "already initialized CUDA" in str(error)
else:
    raise AssertionError("already-initialized CUDA caller accepted")
assert dict(os.environ) == before
'''
    run = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
        capture_output=True, text=True, env=_cpu_environment(), timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


def test_metadata_keeps_unknown_provenance():
    saved, attempt = _fixture()
    recovery.validate_metadata(saved, attempt)
    artifact = recovery.artifact(saved, attempt, dict(file="fixture.pt", sha256="c"*64), {})
    assert artifact["same_run_resume_used"] is None and artifact["fresh_training"] is None
    assert artifact["schema"] == "gift.recovered-baseline-weights.v1"
    assert "optimizer_state_dict" not in artifact


def test_incomplete_identity_history_and_optimizer_are_rejected():
    saved, attempt = _fixture()
    cases = [copy.deepcopy(saved) for _ in range(4)]
    cases[0]["payload"]["status"] = "running"
    cases[1]["payload"]["history"].pop()
    cases[2]["identity"]["configuration"]["formal"] = False
    del cases[3]["payload"]["optimizer_state_dict"]["state"][17]
    for value in cases:
        with pytest.raises(ValueError):
            recovery.validate_metadata(value, attempt)


def test_atomic_creation_preserves_existing_and_failed_files(tmp_path):
    target = tmp_path / "bytes_only_not_weights.pt"
    def verify(path):
        assert path.read_bytes() == b"fixture"
    recovery.atomic_create(target, lambda stream: stream.write(b"fixture"), verify)
    with pytest.raises(ValueError):
        recovery.atomic_create(target, lambda stream: stream.write(b"overwrite"), lambda path: None)
    assert target.read_bytes() == b"fixture"
    failed = tmp_path / "must_not_publish.pt"
    def reject(path):
        raise ValueError("simulated verification interruption")
    with pytest.raises(ValueError):
        recovery.atomic_create(failed, lambda stream: stream.write(b"incomplete_fixture"), reject)
    assert not failed.exists()
    assert len(list(tmp_path.glob(".must_not_publish.pt.recovery-*.tmp"))) == 1


def test_current_consumers_metadata_only_without_model_import():
    class FakeModel:
        def requires_grad_(self, enabled):
            assert enabled is False
            return self
    saved, attempt = _fixture()
    artifact = recovery.artifact(saved, attempt, dict(file="fixture.pt", sha256="c"*64), {})
    paths = [("adapters/prediction.py", "_load"), ("experiments/formal/_shared/fno_runtime.py", "_terminal_contract")]
    for relative, name in paths:
        source = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
        scope = {"load_checkpoint": lambda *args: (FakeModel(), artifact)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), scope)
        if name == "_load":
            scope[name]("uno", "NOT_READ", "cpu")
        else:
            scope[name](dict(artifact, method="fno2d", terminal_epoch=500), "FNO-2D")


def test_synthetic_cpu_tensor_end_to_end(tmp_path):
    # These 36 tiny tensors and 150 invented metadata rows are not a UNO model
    # or evidence of training. The actual tool runs in another CPU-only child.
    code = r'''
import json, runpy, subprocess, sys
from pathlib import Path
from scripts import recover_baseline_export as recovery
fixture = runpy.run_path(sys.argv[1])["_fixture"]
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
directory = Path(sys.argv[2])
saved, attempt = fixture()
state = {"SYNTHETIC_NOT_TRAINING.p%d" % i: torch.tensor([i, -i], dtype=torch.float32) for i in range(36)}
saved["payload"]["model_state_dict"] = state
for item in saved["payload"]["optimizer_state_dict"]["state"].values():
    item["exp_avg"], item["exp_avg_sq"] = torch.zeros(2), torch.ones(2)
saved["rng"].update(torch_cpu=torch.zeros(4, dtype=torch.uint8), torch_cuda=[],
    python=(3, (1, 2), None), numpy=("SYNTHETIC_NOT_TRAINING", [], 0, 0, 0.))
checkpoint = directory / ("epoch_0150_" + "b"*32 + ".pt")
attempt_path = directory / "ATTEMPT.json"
with checkpoint.open("xb") as stream: torch.save(saved, stream)
with attempt_path.open("x", encoding="utf-8") as stream: json.dump(attempt, stream)
before = [recovery.record(checkpoint), recovery.record(attempt_path)]
output = directory / "SYNTHETIC_NOT_TRAINING_recovered.pt"
command = [sys.executable, "-B", str(Path(recovery.__file__)), "--execute", "--checkpoint", str(checkpoint),
    "--checkpoint-sha256", before[0]["sha256"], "--attempt", str(attempt_path),
    "--attempt-sha256", before[1]["sha256"], "--output", str(output)]
run = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
receipt = json.loads(run.stdout)
assert receipt["inputs_unchanged"] and receipt["cuda_initialized"] is False
assert receipt["torch_num_threads"] == 1 and receipt["cuda_visible_devices"] == "-1"
actual = torch.load(output, map_location="cpu", weights_only=True)
expected = recovery.artifact(saved, attempt, before[0], before[1])
assert set(actual) == set(expected) and set(actual["model_state_dict"]) == set(state)
assert all(torch.equal(actual["model_state_dict"][k], value) for k, value in state.items())
assert all(actual[k] == value for k, value in expected.items() if k != "model_state_dict")
assert [recovery.record(checkpoint), recovery.record(attempt_path)] == before
assert not torch.cuda.is_initialized()
evidence = dict(purpose="SYNTHETIC_NOT_TRAINING", status="PASS", tensor_count=36,
    fake_history_rows=150, inputs_unchanged=True, cuda_initialized=False, subprocess_receipt=receipt)
with (directory / "SYNTHETIC_RESULT.json").open("x", encoding="utf-8") as stream: json.dump(evidence, stream, indent=2)
print(json.dumps(evidence))
'''
    run = subprocess.run([sys.executable, "-B", "-c", code, str(Path(__file__)), str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, env=_cpu_environment(), timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    assert result["purpose"] == "SYNTHETIC_NOT_TRAINING" and result["tensor_count"] == 36
    assert result["inputs_unchanged"] and result["cuda_initialized"] is False
