"""CPU-only guards for the W3 fixed-budget queue; fixtures are not experiment results."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts import generate_noise
from scripts import run_w3_parameters as queue
from training import train_pinn
from training.gift_data import input_file


def test_five_percent_values_use_clean_shared_draw(tmp_path):
    """Require actual 5% fields, the original 1% draw and unchanged float32 casting."""
    source = tmp_path / "clean.h5"
    fields = {}
    with h5py.File(source, "x") as handle:
        for group, identity, seed in (("training", 0, 80), ("validation", 50, 81)):
            fields[group] = np.random.RandomState(seed).normal(size=(1, 17, 64, 64)).astype(np.float32)
            handle.create_dataset(group + "/vorticity", data=fields[group])
            handle.create_dataset(group + "/time", data=np.arange(17) * .02)
            handle.create_dataset(group + "/trajectory_index", data=[identity])
    output = tmp_path / "paired"
    generate_noise.main(["--clean", str(source), "--output", str(output), "--pilot", "--execute"])
    rng = np.random.RandomState(0)
    with h5py.File(source, "r") as clean, h5py.File(output / "noise_001.h5", "r") as one, h5py.File(output / "noise_005.h5", "r") as five:
        assert five.attrs["noise_fraction"] == .05
        for group in ("training", "validation"):
            sigma = generate_noise.group_statistics(clean[group + "/vorticity"])["std_ddof0"]
            for start in (0, 16):
                values = fields[group][0, start:start + 16]
                draw = rng.standard_normal(values.shape)
                for fraction, target in ((.01, one), (.05, five)):
                    expected = (values.astype(np.float64) + fraction * sigma * draw).astype(np.float32)
                    np.testing.assert_array_equal(target[group + "/vorticity"][0, start:start + 16], expected)
    assert {path.name for path in output.glob("*.h5")} == {"noise_001.h5", "noise_005.h5"}


def test_only_new_three_conditions_select_inputs(tmp_path):
    """Reject a withdrawn noise condition while resolving the true new observation path."""
    for condition in ("noise_000", "noise_001", "noise_005"):
        _, inputs = train_pinn.configured_inputs(condition, tmp_path)
        assert condition in inputs["sampling"].name
        assert inputs["data"] == input_file(tmp_path, condition, "regenerated")
    with pytest.raises(ValueError, match="condition"):
        train_pinn.configured_inputs("unsupported_condition", tmp_path)


def test_verified_actual_trainer_matches_when_wrapper_has_exited(tmp_path):
    """A surviving actual trainer prevents another queue even if its venv wrapper is absent."""
    record = dict(child_pid=999999, command=["python", "-u", "-B", "-m", "training.train_pinn", "--output", str(tmp_path)])
    matching = dict(ProcessId=1234, ParentProcessId=999999,
        CommandLine='python -u -B -m training.train_pinn --output "' + str(tmp_path) + '"')
    other = dict(ProcessId=1235, ParentProcessId=12, CommandLine="python -m training.train_pinn --output D:/another-run")
    assert queue.matching_children([matching, other], tmp_path, record) == [matching]
    reused = dict(ProcessId=999999, ParentProcessId=12, CommandLine="python -m another_task")
    assert queue.matching_children([reused], tmp_path, record) == []


def test_process_inventory_explicit_utf8_preserves_command_text(monkeypatch):
    """Non-ASCII command lines survive the read-only PowerShell transport without replacement decoding."""
    row = dict(ProcessId=1234, ParentProcessId=12, CommandLine='python -m training.train_pinn --output "D:/试验记录/run"')
    def inventory(command, **options):
        assert "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)" in command[-1]
        assert "Where-Object" in command[-1] and "python(?:w)?" in command[-1]
        assert options["encoding"] == "utf-8" and "errors" not in options
        return SimpleNamespace(stdout=json.dumps([row], ensure_ascii=False))
    monkeypatch.setattr(queue.subprocess, "run", inventory)
    monkeypatch.setattr(queue.os, "name", "nt")
    assert queue.process_inventory() == [row]


def test_generator_continues_own_journal_without_fresh_action(tmp_path, monkeypatch):
    """Resume exactly the owned generator slot; a partial directory without its journal must stop."""
    plan = queue.default_plan(tmp_path)
    output = tmp_path / "training/noise_005/gift"
    (output / "checkpoints").mkdir(parents=True)
    (output / "checkpoints/LATEST.json").write_text("{}")
    calls = []
    def complete(directory, plan, module, arguments, label, **options):
        calls.append((module, arguments, options))
        (output / "gift_noise_005.pt").write_bytes(b"UNIT_TEST_DUMMY_NOT_MODEL_WEIGHTS")
        (output / "manifest.json").write_text('{"status":"complete"}')
        (output / "training_report.json").write_text(json.dumps(dict(
            training_cost=dict(phase_updates=[6000, 6000]), optimization=dict(seed=2026072301))))
        return dict(status="complete")
    monkeypatch.setattr(queue, "run_module", complete)
    slot = queue.train_slot(tmp_path, plan, "noise_005", "GIFT")
    assert slot["status"] == "complete" and len(calls) == 1
    assert "--resume" in calls[0][1] and "--run-training" not in calls[0][1]
    queue.train_slot(tmp_path, plan, "noise_005", "GIFT")
    assert len(calls) == 1
    (tmp_path / "training/noise_001/gift").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="initial full-state journal"):
        queue.train_slot(tmp_path, plan, "noise_001", "GIFT")


def test_scientific_failure_cannot_restart_unique_seed(tmp_path, monkeypatch):
    """A recorded scientific failure fills its original slot without a second child launch."""
    output = tmp_path / "training/noise_005/open"
    output.mkdir(parents=True)
    (output / "FAILURE.json").write_text('{"status":"failed","failure_kind":"scientific"}')
    def forbidden(*args, **kwargs):
        raise AssertionError("failed slot cannot launch another initialization")
    monkeypatch.setattr(queue, "run_module", forbidden)
    result = queue.train_slot(tmp_path, queue.default_plan(tmp_path), "noise_005", "PINN-SR")
    assert result["status"] == "failed" and result["seed"] == 1234


def test_interrupted_cost_is_explicit_upper_bound(tmp_path):
    """Parent loss must not invent an exact child exit time or erase its cost evidence."""
    queue.append_event(tmp_path, dict(event="start", attempt_id="fixture", label="train_fixture",
        command=["test_only"], started_utc=queue.now(), child_pid=1, parent_pid=2))
    queue.close_interrupted_events(tmp_path)
    queue.close_interrupted_events(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "JOB_EVENTS.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[1]["event"] == "interrupted"
    assert rows[1]["process_seconds"] is None and rows[1]["elapsed_upper_bound_seconds"] >= 0


def test_locked_plan_contains_no_selected_candidates_or_reused_weights(tmp_path):
    """Lock nine fresh training instances and native fixed schedules before any score is read."""
    plan = queue.default_plan(tmp_path)
    lock = queue.prepare_run(tmp_path, plan)
    assert plan["conditions"] == {"noise_000": 0., "noise_001": .01, "noise_005": .05}
    assert plan["fresh_training_slots"] == 9 and plan["fresh_measurement_slots"] == 15
    assert lock["input_weights"] == [] and lock["no_test_parameter_selection"]
    assert not plan["parameter_search"] and not plan["repeated_seed_runs"]
    assert plan["generator"]["phase_steps"] == [6000, 6000]
    assert plan["pinn"]["budget"] == dict(pre_nadam=5000, rounds=6, ado_nadam=1000,
        post_nadam=20000, lbf_maxiter=10000, lbf_maxfun=10000)
