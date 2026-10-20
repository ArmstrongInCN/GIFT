"""CPU-only queue tests replace every child; no dataset, GPU or training is used."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_w1_v2 as queue


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_sparse76_n300_new_identity_and_ten_updates(tmp_path):
    """N30076 binds one common physical grid and starts new model runs."""
    plan = queue.default_plan(tmp_path, "sparse76_n300")
    previous = queue.default_plan(tmp_path, "sparse31_n200")
    assert plan["schema"] != previous["schema"] and plan["window"] != previous["window"]
    assert plan["training_count"] == 300 and plan["time_stride"] == 2
    assert plan["observation_dt"] == .04 and plan["future_steps"] == 75
    assert len(plan["query_times"]) == 76 and plan["query_times"][-1] == 3.
    assert plan["fno2d_curriculum"]["horizons"] == [1, 2, 5, 10, 25, 50, 75]
    assert divmod(plan["training_count"], plan["effective_batch"]) == (9, 12)
    for method in ("anie_ns", "anie_f", "fno2d", "fno3d"):
        job = queue.job_configuration(plan, method, "formal", 11, plan["candidates"][method][0])
        assert job["model_spec"]["query_times"] == plan["query_times"]
        assert job["config"]["future_steps"] == 75
        assert job["config"]["closed_loop_horizons"] == [1, 2, 5, 10, 25, 50, 75]
        assert job["config"]["max_epochs"] == 1500


def test_sparse76_lock_and_both_evaluations_have_75_future_points(tmp_path, monkeypatch):
    """Both cohorts receive the locked .04 grid, never old31 predictions."""
    plan = queue.default_plan(tmp_path, "sparse76_n300")
    plan["methods"] = ["anie_f"]
    def job(directory, current_plan, concrete):
        return dict(job=concrete, status="complete", checkpoint=concrete["output"] + "/best.pt",
                    validation_metric=concrete["candidate"], training_cost={})
    def child(directory, module, arguments, label):
        if module == "scripts.generate_w1_v2_confirmation":
            lock = _read(directory / "DEVELOPMENT_LOCK.json")
            assert lock["evaluation_dt"] == .04 and lock["observation_points"] == 76
            assert lock["training_count"] == 300 and lock["future_steps"] == 75
            queue.atomic_json(directory / "confirmation/complete.json", {"status": "complete"})
        else:
            assert arguments[arguments.index("--dt") + 1] == .04
            assert arguments.count("--checkpoint") == 3
            output = Path(arguments[arguments.index("--output") + 1])
            queue.atomic_json(output / "numeric_report.json", {"methods": {}})
        return 0
    monkeypatch.setattr(queue, "execute_job", job)
    monkeypatch.setattr(queue, "call_module", child)
    queue.execute(tmp_path, plan)
    report = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "75个未来帧" in report and "300条" in report and "76点" in report
    assert "5.04" in report and "30个逐帧" not in report


def test_sparse31_n200_is_new_scientific_identity_with_seven_updates_per_epoch(tmp_path):
    """The cheaper task binds grid/data and cannot quietly reuse a dense run."""
    dense = queue.default_plan(tmp_path)
    sparse = queue.default_plan(tmp_path, "sparse31_n200")
    assert dense["schema"] != sparse["schema"]
    assert dense["window"] != sparse["window"]
    assert sparse["training_count"] == 200 and sparse["time_stride"] == 5
    assert sparse["subset_seed"] != sparse["sample_seed"]
    assert len(sparse["query_times"]) == 31 and sparse["query_times"][-1] == 3.
    assert sparse["fno2d_curriculum"]["horizons"] == [1, 1, 2, 4, 10, 20, 30]
    assert (sparse["training_count"] + 31) // 32 == 7
    for method in ("anie_ns", "anie_f", "fno2d", "fno3d"):
        job = queue.job_configuration(sparse, method, "formal", 11, sparse["candidates"][method][0])
        assert job["model_spec"]["query_times"] == sparse["query_times"]
        assert job["config"]["future_steps"] == 30
        assert job["config"]["max_epochs"] == 1500
        assert job["config"]["micro_batch"] == dense["physical_micro_batches"][method]


def test_sparse_queue_binds_generation_and_both_evaluations_to_common_grid(tmp_path, monkeypatch):
    """All raw scores are genuine31 predictions; neither cohort determines LR."""
    plan = queue.default_plan(tmp_path, "sparse31_n200")
    plan["methods"] = ["anie_f"]
    def job(directory, current_plan, concrete):
        return dict(job=concrete, status="complete", checkpoint=concrete["output"] + "/best.pt",
                    validation_metric=concrete["candidate"], training_cost={})
    def child(directory, module, arguments, label):
        if module == "scripts.generate_w1_v2_confirmation":
            lock = _read(directory / "DEVELOPMENT_LOCK.json")
            assert lock["evaluation_dt"] == .1 and lock["observation_points"] == 31
            assert lock["training_count"] == 200 and lock["future_steps"] == 30
            queue.atomic_json(directory / "confirmation/complete.json", {"status": "complete"})
        else:
            assert arguments[arguments.index("--dt") + 1] == .1
            assert arguments.count("--checkpoint") == 3
            output = Path(arguments[arguments.index("--output") + 1])
            queue.atomic_json(output / "numeric_report.json", {"methods": {}})
        return 0
    monkeypatch.setattr(queue, "execute_job", job)
    monkeypatch.setattr(queue, "call_module", child)
    queue.execute(tmp_path, plan)
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "30个未来帧" in text and "200条" in text and "31点" in text
    assert "150个逐帧" not in text


def test_queue_selects_only_validation_then_uses_fresh_formal_jobs_and_keeps_failures(tmp_path, monkeypatch):
    """Development evidence precedes generation; failedseed22 stays in both evaluations."""
    plan = queue.default_plan(tmp_path)
    plan["methods"] = ["fno2d"]
    calls, jobs = [], []

    def execute_job(directory, concrete_plan, job):
        jobs.append(job)
        assert job["window"] == plan["window"]
        assert plan["old_test"] not in json.dumps(job)
        assert "checkpoint" not in job and "initial_checkpoint" not in job["config"]
        if job["stage"] == "formal":
            lock = _read(directory / "DEVELOPMENT_LOCK.json")
            assert lock["selected_configurations"]["fno2d"]["candidate"] == .0003
            assert (directory / "confirmation/complete.json").exists()
            assert job["candidate"] == .0003
            assert job["config"]["seed"] in (11, 22, 33)
            if job["seed"] == 22:
                return dict(job=job, status="failed", validation_metric=None,
                            failure=job["output"] + "/FAILURE.json")
        return dict(job=job, status="complete", checkpoint=job["output"] + "/best.pt",
                    validation_metric=.8 if job["candidate"] == .001 else .2,
                    training_cost={})

    def call_module(directory, module, arguments, label):
        calls.append((module, list(map(str, arguments))))
        if module == "scripts.generate_w1_v2_confirmation":
            assert len(jobs) == 2 and all(job["stage"] == "development" for job in jobs)
            assert (directory / "DEVELOPMENT_LOCK.json").exists()
            queue.atomic_json(directory / "confirmation/complete.json", {"status": "complete"})
        else:
            assert module == "experiments.additional.w1_v2_evaluate"
            assert len(jobs) == 5
            output = Path(arguments[arguments.index("--output") + 1])
            queue.atomic_json(output / "numeric_report.json", {"methods": {}})
            assert arguments.count("--checkpoint") == 2
            assert arguments.count("--failed-run") == 1
            assert all("development" not in str(value) for value in arguments)
        return 0

    monkeypatch.setattr(queue, "execute_job", execute_job)
    monkeypatch.setattr(queue, "call_module", call_module)
    queue.execute(tmp_path, plan)
    assert [job["stage"] for job in jobs] == ["development"] * 2 + ["formal"] * 3
    assert jobs[0]["output"] != jobs[1]["output"]
    assert all(job["output"] not in {jobs[0]["output"], jobs[1]["output"]} for job in jobs[2:])
    assert _read(tmp_path / "ORCHESTRATOR.json")["failed_training_seeds"] == 1
    assert [module for module, _ in calls] == ["scripts.generate_w1_v2_confirmation"] + [
        "experiments.additional.w1_v2_evaluate"] * 2


def test_numerical_failure_is_reused_without_restarting_seed(tmp_path, monkeypatch):
    """A declared failure is evidence, not a cue to retry with a new initialization."""
    plan = queue.default_plan(tmp_path)
    job = queue.job_configuration(plan, "fno2d", "formal", 22, .001)
    failure = Path(job["output"]) / "FAILURE.json"
    queue.atomic_json(failure, dict(method="fno2d", seed=22, stage="formal",
                                   reason="nonfinite gradient", failure_kind="numerical"))
    monkeypatch.setattr(queue, "call_module", lambda *args: pytest.fail("failed seed must not restart"))
    original = failure.read_bytes()
    result = queue.execute_job(tmp_path, plan, job)
    assert result["status"] == "failed" and result["failure"] == str(failure)
    assert failure.read_bytes() == original


def test_infrastructure_failure_needs_attention_and_is_not_a_numerical_seed(tmp_path, monkeypatch):
    """A permission or IO interruption must not become a bad scientific score."""
    plan = queue.default_plan(tmp_path)
    job = queue.job_configuration(plan, "fno2d", "formal", 22, .001)
    queue.atomic_json(Path(job["output"]) / "FAILURE.json", dict(failure_kind="infrastructure", reason="IO"))
    monkeypatch.setattr(queue, "call_module", lambda *args: pytest.fail("repair is required first"))
    with pytest.raises(RuntimeError, match="infrastructure"):
        queue.execute_job(tmp_path, plan, job)


@pytest.mark.parametrize("existing_journal", [False, True])
def test_training_child_resumes_only_its_own_journal(tmp_path, monkeypatch, existing_journal):
    """Formal jobs either start fresh or load their own latest, never development weights."""
    plan = queue.default_plan(tmp_path)
    job = queue.job_configuration(plan, "fno2d", "formal", 11, .0003)
    output = Path(job["output"])
    if existing_journal:
        output.mkdir(parents=True)
        (output / "latest.pt").write_bytes(b"fixture-journal")

    def child(directory, module, arguments, label):
        assert module == "training.w1_v2_train"
        assert ("--resume" in arguments) == existing_journal
        concrete = _read(arguments[arguments.index("--run-config") + 1])
        assert concrete == job
        assert concrete["stage"] == "formal" and concrete["seed"] == 11
        output.mkdir(parents=True, exist_ok=True)
        (output / "best.pt").write_bytes(b"fixture-best")
        queue.atomic_json(output / "STATUS.json", dict(status="complete", best_val=.1))
        return 0

    monkeypatch.setattr(queue, "call_module", child)
    result = queue.execute_job(tmp_path, plan, job)
    assert result["status"] == "complete" and result["validation_metric"] == .1


def test_terminal_run_without_any_finite_validation_remains_a_failed_seed(tmp_path, monkeypatch):
    """Completing the budget without a best model is numerical failure, not infrastructure."""
    plan = queue.default_plan(tmp_path)
    job = queue.job_configuration(plan, "fno2d", "formal", 33, .001)
    output = Path(job["output"])

    def child(directory, module, arguments, label):
        queue.atomic_json(output / "STATUS.json", dict(status="complete", best_val=None,
                         best_epoch=None, terminal_reason="budget_limit"))
        return 0

    monkeypatch.setattr(queue, "call_module", child)
    result = queue.execute_job(tmp_path, plan, job)
    assert result["status"] == "failed" and result["validation_metric"] is None
    failure = _read(result["failure"])
    assert failure["failure_kind"] == "numerical" and failure["seed"] == 33


@pytest.mark.parametrize("failure_kind", ["numerical", "infrastructure"])
def test_child_failure_is_checked_before_accepting_its_completion_marker(tmp_path, monkeypatch, failure_kind):
    """An exception recorded after terminal writes cannot disappear from this queue turn."""
    plan = queue.default_plan(tmp_path)
    job = queue.job_configuration(plan, "fno2d", "formal", 11, .001)
    output = Path(job["output"])

    def child(directory, module, arguments, label):
        output.mkdir(parents=True)
        (output / "best.pt").write_bytes(b"fixture-best")
        queue.atomic_json(output / "STATUS.json", dict(status="complete", best_val=.1))
        queue.atomic_json(output / "FAILURE.json", dict(method="fno2d", seed=11, stage="formal",
                         reason="exception after terminal write", failure_kind=failure_kind))
        return 1

    monkeypatch.setattr(queue, "call_module", child)
    if failure_kind == "infrastructure":
        with pytest.raises(RuntimeError, match="infrastructure"):
            queue.execute_job(tmp_path, plan, job)
    else:
        result = queue.execute_job(tmp_path, plan, job)
        assert result["status"] == "failed"


@pytest.mark.parametrize("same_command", [True, False])
def test_surviving_child_is_waited_only_after_command_identity_matches(tmp_path, monkeypatch, same_command):
    """Mock every OS call: an adopted child blocks duplicate resume; PID reuse does not."""
    import ctypes

    calls = []

    class Function:
        """Allow ctypes signature assignment while recording harmless mock calls."""
        def __init__(self, name, result):
            self.name, self.result = name, result

        def __call__(self, *args):
            calls.append((self.name, args))
            return self.result

    kernel = SimpleNamespace(OpenProcess=Function("open", 123),
                             WaitForSingleObject=Function("wait", 0x102),
                             CloseHandle=Function("close", 1))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel)
    command = ["python.exe", "-u", "-m", "training.w1_v2_train", "--run-config", "fixture-run.json"]
    stdout = " ".join(command) if same_command else "python.exe -m unrelated_job"
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=stdout))
    queue.atomic_json(tmp_path / "ORCHESTRATOR.json", dict(status="running", child_pid=90909,
                      active_job="fixture-formal", command=command, log="fixture.log"))
    queue.wait_orphaned_child(tmp_path)
    waits = [args[1] for name, args in calls if name == "wait"]
    assert waits == ([0, 0xFFFFFFFF] if same_command else [0])
    assert calls[-1] == ("close", (123,))
    assert bool(_read(tmp_path / "ORCHESTRATOR.json").get("adopted_surviving_child")) == same_command
