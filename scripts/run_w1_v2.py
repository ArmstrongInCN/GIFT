"""Sequential, recoverable execution of the complete predeclared W1-v2 protocol.

Each GPU job runs in a separate process. The queue develops two candidates per
method, freezes validation-selected settings, generates independent truth,
trains three fresh seeds, then saves both cohorts' numerical results and plots.
Training failures remain declared seed slots; infrastructure interruptions need
attention and resume their own saved state rather than becoming bad scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from training.w1_v2_train import atomic_json
from gift.paths import default_data_path
from scripts.queue_runtime import child_environment, configured_source, queue_lease, validate_job_source


def now():
    """UTC audit timestamps, independent of scientific trajectory time."""
    return datetime.now(timezone.utc).isoformat()


def default_plan(directory, profile="dense151", *, anie_source=None, fno_source=None):
    """Concrete counts and methods; no scientific option is inferred from test scores."""
    plan = dict(schema="gift.w1-v2.queue.v2", root=str(directory), project_root=str(ROOT),
        window=str(directory / "data/train_validation_5to8.h5"),
        old_test=str(default_data_path("four_vortex.fno_test_dt0p02", "fno/fno_test_n64_n96_n128_dt0p02.h5")),
        protocol_file=str(directory / "PROTOCOL_LOCKED.md"), sample_seed=2026100301,
        development_seed=7, development_epochs=200, formal_seeds=[11, 22, 33],
        formal_max_epochs=1500, formal_min_epochs=500, patience=10,
        relative_improvement=.001, validation_interval=20, effective_batch=32,
        methods=["fno2d", "anie_ns", "anie_f", "fno3d", "GIFT"],
        candidates=dict(fno2d=[.001, .0003], anie_ns=[.0001, .0003], anie_f=[.0001, .0003],
                        fno3d=[.001, .0003], GIFT=[1., .3]),
        physical_micro_batches=dict(fno2d=16, anie_ns=6, anie_f=6, fno3d=8,
                                    GIFT_generator=16, GIFT_branch=4),
        anie_source=configured_source(anie_source, "GIFT_ANIE_SOURCE_ROOT"),
        fno_source=configured_source(fno_source, "GIFT_EXTERNAL_ROOT"),
        fno2d_curriculum=dict(stage_epochs=20, horizons=[1, 5, 10, 20, 50, 100, 150],
                              teacher_forcing=False, detach=False, activation_checkpoint=False),
        gift_development=dict(generator=[50, 50], branch=[80, 10, 5, 5]),
        gift_formal=dict(generator=[250, 250], branch=[800, 100, 50, 50]),
        confirmation=dict(seed=2026100302, count=180, ids=[100000, 100179]),
        evaluation_batch=4, device="cuda", score="future150_mean_per_frame_relative_l2",
        precision="ANIE/FNO float32, TF32 disabled; GIFT native float32 with float64/complex128 affine statistics",
        scientific_failures_preserved=True)
    if profile == "dense151":
        return plan
    if profile not in ("sparse31_n200", "sparse76_n300"):
        raise ValueError("unsupported W1 comparison profile")
    if profile == "sparse76_n300":
        plan.update(schema="gift.w1-v4.queue.sparse76-n300.v1", profile=profile,
                    window=str(directory / "data/train300_validation20_5to8_dt0p04.h5"),
                    training_count=300, time_stride=2, subset_seed=2026100304,
                    query_times=[.02 * index for index in range(0, 151, 2)],
                    observation_dt=.04, observation_points=76, future_steps=75,
                    gift_integration_dt=.02, score="future75_mean_per_frame_relative_l2",
                    fno2d_curriculum=dict(stage_epochs=20, horizons=[1, 2, 5, 10, 25, 50, 75],
                                         teacher_forcing=False, detach=False, activation_checkpoint=False))
        return plan
    # New scientific identity; never reinterpret the paused dense experiment.
    plan.update(schema="gift.w1-v3.queue.sparse31-n200.v1", profile=profile,
                window=str(directory / "data/train200_validation20_5to8_dt0p1.h5"),
                training_count=200, time_stride=5, subset_seed=2026100304,
                query_times=[.02 * index for index in range(0, 151, 5)],
                observation_dt=.1, observation_points=31, future_steps=30,
                gift_integration_dt=.02, score="future30_mean_per_frame_relative_l2",
                fno2d_curriculum=dict(stage_epochs=20, horizons=[1, 1, 2, 4, 10, 20, 30],
                                     teacher_forcing=False, detach=False, activation_checkpoint=False))
    return plan


def status(directory, phase, **fields):
    """Small mutable status; logs and completed scientific results remain separate."""
    atomic_json(directory / "ORCHESTRATOR.json", dict(status=phase, parent_pid=os.getpid(), updated_utc=now(), **fields))


def wait_orphaned_child(directory):
    """Let a surviving previous child finish before resuming the queue.

    Windows process query/wait APIs are read-only. Never use os.kill(pid,0)
    here: its Windows behavior is unsuitable for a safe liveness check.
    """
    import ctypes
    record_path = directory / "ORCHESTRATOR.json"
    if not record_path.exists():
        return
    record = json.loads(record_path.read_text(encoding="utf-8"))
    pid = record.get("child_pid")
    if record.get("status") != "running" or not isinstance(pid, int):
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        return
    try:
        if kernel.WaitForSingleObject(handle, 0) == 0x102:
            inspected = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                capture_output=True, text=True, check=True).stdout.casefold()
            if not inspected.strip():
                raise RuntimeError("cannot identify surviving PID safely")
            expected = record.get("command", [])
            if not expected or not all(str(argument).casefold() in inspected for argument in expected[2:]):
                return  # PID belongs to a different command; do not wait or touch it.
            # Keep the existing command/PID visible for the monitor. A completed
            # same-run journal is authoritative after the child exits.
            status(directory, "running", active_job=record.get("active_job"), child_pid=pid,
                   adopted_surviving_child=True, command=record.get("command"), log=record.get("log"))
            kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
    finally:
        kernel.CloseHandle(handle)


def call_module(directory, module, arguments, label):
    """Run one child with argument-list quoting and a durable append-only log."""
    environment = child_environment()
    sources = {key: environment[key] for key in ("GIFT_EXTERNAL_ROOT", "GIFT_ANIE_SOURCE_ROOT", "GIFT_EXTERNAL_ADAPTER_ROOT")
               if key in environment}
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    started = time.perf_counter()
    with (logs / (label + ".log")).open("a", encoding="utf-8") as log:
        log.write("\n" + json.dumps(dict(started_utc=now(), command=command, source_directories=sources), ensure_ascii=False) + "\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status(directory, "running", active_job=label, child_pid=child.pid, command=command, log=str(logs / (label + ".log")))
        code = child.wait()
        seconds = time.perf_counter() - started
        log.write(json.dumps(dict(finished_utc=now(), exit_code=code, process_seconds=seconds)) + "\n")
    with (directory / "JOB_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(label=label, command=command, exit_code=code, process_seconds=seconds,
                                     finished_utc=now(), source_directories=sources)) + "\n")
    return code


def job_configuration(plan, method, stage, seed, candidate):
    """Make one fixed native-model run specification, retaining candidate identity."""
    root = Path(plan["root"])
    suffix = f"{method}_seed{seed}_candidate{candidate:.8g}"
    output = root / stage / suffix
    if method == "GIFT":
        return dict(method=method, stage=stage, seed=seed, candidate=candidate, output=str(output))
    model_spec = dict(method=method, history=1, source_root=plan["anie_source"] if method.startswith("anie") else plan["fno_source"])
    if method.startswith("anie"):
        model_spec["condition_mode"] = "native" if method == "anie_ns" else "fixed_f"
    if method == "fno2d":
        model_spec["activation_checkpoint"] = False
    if "query_times" in plan:
        model_spec.update(query_times=plan["query_times"], future_steps=plan["future_steps"],
                          observation_dt=plan["observation_dt"])
    config = dict(stage=stage, seed=seed, sample_seed=plan["sample_seed"], learning_rate=candidate,
                  max_epochs=plan["development_epochs"] if stage == "development" else plan["formal_max_epochs"],
                  min_epochs=plan["formal_min_epochs"], patience=plan["patience"],
                  validation_interval=plan["validation_interval"], effective_batch=plan["effective_batch"],
                  micro_batch=plan["physical_micro_batches"][method], fixture=False,
                  closed_loop_curriculum=method == "fno2d")
    if "query_times" in plan:
        config.update(future_steps=plan["future_steps"],
                      closed_loop_horizons=plan["fno2d_curriculum"]["horizons"])
    return dict(method=method, stage=stage, seed=seed, candidate=candidate, window=plan["window"],
                output=str(output), model_spec=model_spec, config=config, device=plan["device"])


def recorded_failure(output, job):
    """An explicit post-commit failure takes precedence over a completion pointer."""
    path = Path(output) / "FAILURE.json"
    if not path.exists():
        return None
    failed = json.loads(path.read_text(encoding="utf-8"))
    if failed.get("failure_kind") == "infrastructure":
        raise RuntimeError("infrastructure failure needs repair before same-run continuation: " + str(path))
    return dict(job=job, status="failed", validation_metric=None, failure=str(path))


def execute_job(directory, plan, job):
    """Reuse only a completed same-configuration run, or resume its own journal."""
    output = Path(job["output"])
    directory.joinpath("configs").mkdir(exist_ok=True)
    config_path = directory / "configs" / (output.name + "_" + job["stage"] + ".json")
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != job:
            raise ValueError("existing concrete run configuration differs")
    else:
        atomic_json(config_path, job)
    failure_path = output / "FAILURE.json"
    failure = recorded_failure(output, job)
    if failure:
        return failure
    if job["method"] == "GIFT":
        completed_path = output / "PIPELINE.json"
        complete = json.loads(completed_path.read_text(encoding="utf-8")) if completed_path.exists() else None
        if complete is None:
            args = ["--resume" if output.exists() else "--run-training", "--stage", "pipeline",
                    "--profile", job["stage"], "--dataset", plan["window"], "--output", output,
                    "--seed", job["seed"], "--lr-multiplier", job["candidate"], "--device", plan["device"]]
            code = call_module(directory, "training.w1_v2_gift", args, job["stage"] + "_" + output.name)
            complete = json.loads(completed_path.read_text(encoding="utf-8")) if completed_path.exists() else None
        else:
            code = 0
        failure = recorded_failure(output, job)
        if failure:
            return failure
        if complete and complete["status"] == "complete":
            return dict(job=job, status="complete", validation_metric=complete["val_metric"],
                        checkpoint=complete["paths"]["branch_model"], training_cost=complete["training_cost"])
    else:
        completed_path = output / "STATUS.json"
        complete = json.loads(completed_path.read_text(encoding="utf-8")) if completed_path.exists() else None
        if not complete or complete["status"] != "complete":
            validate_job_source(job)
            args = ["--run-config", config_path]
            if (output / "latest.pt").exists():
                args.append("--resume")
            elif output.exists():
                raise RuntimeError("run interrupted before its initial recoverable boundary: " + str(output))
            code = call_module(directory, "training.w1_v2_train", args, job["stage"] + "_" + output.name)
            complete = json.loads(completed_path.read_text(encoding="utf-8")) if completed_path.exists() else None
        else:
            code = 0
        failure = recorded_failure(output, job)
        if failure:
            return failure
        if complete and complete["status"] == "complete" and (output / "best.pt").exists():
            return dict(job=job, status="complete", validation_metric=complete["best_val"],
                        checkpoint=str(output / "best.pt"), training_cost=complete)
        if code == 0 and complete and complete["status"] == "complete" and not (output / "best.pt").exists():
            atomic_json(failure_path, dict(method=job["method"], seed=job["seed"], stage=job["stage"],
                        status="failed", failure_kind="numerical", reason="no_finite_validation_checkpoint",
                        training_cost=complete))
            return dict(job=job, status="failed", validation_metric=None, failure=str(failure_path))
    if failure_path.exists():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure.get("failure_kind", "numerical") == "numerical":
            return dict(job=job, status="failed", validation_metric=None, failure=str(failure_path))
    if code == 0 and complete and complete["status"].startswith("failed"):
        atomic_json(failure_path, dict(method=job["method"], seed=job["seed"], stage=job["stage"],
                    status="failed", failure_kind="numerical", reason=complete["status"]))
        return dict(job=job, status="failed", validation_metric=None, failure=str(failure_path))
    raise RuntimeError("child interrupted or had an infrastructure error; inspect log before resuming " + str(output))


def render_report(directory, reports, plan=None):
    """Render completed numbers, keeping failed populations out of normal rankings."""
    sparse = plan is not None and "query_times" in plan
    future = plan["future_steps"] if sparse else 150
    first_time = f"{5 + plan['observation_dt']:g}" if sparse else "5.02"
    lines = ["# W1 共同训练与评价实验结果", "",
             f"共同输入：t=5单个N64涡量场；目标：{first_time}–8秒{future}个未来帧。",
             (f"共同训练：从原1000条中按预定seed{plan['subset_seed']}抽取的固定{plan['training_count']}条；仅5–8秒{plan['observation_points']}点观测。"
              if sparse else "训练IDs0–999，仅5–8秒151点观测。") +
             "validation IDs1000–1019。各方法独立从头训练seed11/22/33。",
             f"主误差为每条轨迹未来{future}个逐帧relative L2的平均，再跨轨迹平均；±为三个训练seed的样本SD。", ""]
    for title, report in reports:
        lines.extend(["## " + title, "", "| 方法 | 未来平均相对L2 | t8相对L2 | 完整seed状态 |", "|---|---:|---:|---|"])
        for method, result in report["methods"].items():
            def number(key):
                record = result["metrics"][key]
                return "失败/不完整" if record["mean"] is None else f"{100*record['mean']:.3f}% ± {100*record['training_seed_sample_sd']:.3f}%"
            lines.append(f"| {method} | {number('future_mean_relative_l2')} | {number('t8_relative_l2')} | {result['status']} |")
        lines.extend(["", "完整数值、逐轨迹误差、失败率和成本见对应评价目录的numeric_report.json；图由已保存数值生成。", ""])
    lines.extend(["旧test为已参与方案开发的探索性复评。新确认集为配置锁定后生成的180条独立同分布轨迹，结果未用于调参。",
                  "FNO2D包含预先固定的闭环horizon课程；GIFT保留分阶段观测导数/rollout目标。此处比较的是完整声明训练流程，不是只换网络结构的消融。", ""])
    if plan is not None and plan.get("profile") == "sparse76_n300":
        lines.extend(["N300/dt0.04任务中，GIFT原.1/.5秒监督lead按预锁nearest/tie-earlier规则改为.08/.48秒；内部RK4仍.02秒，每两步保存一次，共76点。",
                      "固定关键帧为t5.48/6/7/8；t5.5不在共同网格，未用插值替代。完整方案及修改说明见PROTOCOL_LOCKED.md和EXECUTION.md。", ""])
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def execute(directory, plan):
    """Advance all stages, with development locking preceding independent generation."""
    development = []
    for method in plan["methods"]:
        for candidate in plan["candidates"][method]:
            result = execute_job(directory, plan, job_configuration(plan, method, "development", plan["development_seed"], candidate))
            development.append(result)
            atomic_json(directory / "DEVELOPMENT.json", development)
    selected = {}
    for method in plan["methods"]:
        eligible = [row for row in development if row["job"]["method"] == method and row["status"] == "complete"]
        if not eligible:
            raise RuntimeError(method + ": both development candidates failed; no valid configuration can be locked")
        winner = min(eligible, key=lambda row: row["validation_metric"])
        selected[method] = dict(candidate=winner["job"]["candidate"], development_validation=winner["validation_metric"],
                                development_checkpoint=winner["checkpoint"])
    lock = dict(protocol_locked=True, development_locked=True, protocol_file=plan["protocol_file"],
                selected_configurations=selected, formal_seeds=plan["formal_seeds"], score=plan["score"],
                confirmation=plan["confirmation"])
    if "query_times" in plan:
        lock.update(evaluation_query_times=plan["query_times"], evaluation_dt=plan["observation_dt"],
                    observation_points=plan["observation_points"], future_steps=plan["future_steps"],
                    training_count=plan["training_count"], training_subset_seed=plan["subset_seed"],
                    observation_window_file=plan["window"])
    lock_path = directory / "DEVELOPMENT_LOCK.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("development lock differs from completed candidate evidence")
    else:
        atomic_json(lock_path, lock)
    confirmation_dir = directory / "confirmation"
    if not (confirmation_dir / "complete.json").exists():
        args = ["--run-generation", "--lock", lock_path, "--output", confirmation_dir, "--device", plan["device"], "--batch-size", "8"]
        if confirmation_dir.exists():
            args.append("--resume")
        if call_module(directory, "scripts.generate_w1_v2_confirmation", args, "confirmation_generation") != 0:
            raise RuntimeError("confirmation generation interrupted; its saved draw/progress can be resumed")
    formal = []
    for seed in plan["formal_seeds"]:
        for method in plan["methods"]:
            result = execute_job(directory, plan, job_configuration(plan, method, "formal", seed, selected[method]["candidate"]))
            formal.append(result)
            atomic_json(directory / "FORMAL.json", formal)
    report_paths = []
    for cohort, data, group, title in (
        ("old_test", plan["old_test"], "N64/test", "旧test：探索性配对复评"),
        ("confirmation", str(confirmation_dir / "confirmation_n64_dt0p02.h5"), "test", "新增独立同分布确认")):
        output = directory / "evaluation" / cohort
        args = ["--data-file", data, "--group", group, "--cohort", cohort, "--output", output,
                "--device", plan["device"], "--batch-size", plan["evaluation_batch"]]
        if "observation_dt" in plan:
            args.extend(["--dt", plan["observation_dt"]])
        for row in formal:
            args.extend(["--checkpoint", row["checkpoint"]] if row["status"] == "complete" else ["--failed-run", row["failure"]])
        if output.exists():
            args.append("--resume")
        if call_module(directory, "experiments.additional.w1_v2_evaluate", args, "evaluation_" + cohort) != 0:
            raise RuntimeError("evaluation interrupted; saved numbers are preserved for same-attempt resume")
        report_paths.append((title, json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))))
    render_report(directory, report_paths, plan)
    status(directory, "complete", report=str(directory / "RESULTS.md"), formal_runs=len(formal),
           failed_training_seeds=sum(row["status"] == "failed" for row in formal))


def main():
    """New concrete plan or continuation of the exact existing queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--profile", choices=("dense151", "sparse31_n200", "sparse76_n300"))
    parser.add_argument("--anie-source", type=Path, help="ANIE clone; overrides GIFT_ANIE_SOURCE_ROOT")
    parser.add_argument("--fno-source", type=Path, help="upstream clone parent; overrides GIFT_EXTERNAL_ROOT")
    args = parser.parse_args()
    if args.anie_source is not None:
        os.environ["GIFT_ANIE_SOURCE_ROOT"] = configured_source(args.anie_source, "GIFT_ANIE_SOURCE_ROOT")
    if args.fno_source is not None:
        os.environ["GIFT_EXTERNAL_ROOT"] = configured_source(args.fno_source, "GIFT_EXTERNAL_ROOT")
    directory = args.root.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "RUN_PLAN.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        profile = plan.get("profile", "dense151")
        expected_schema = {"dense151": "gift.w1-v2.queue.v2", "sparse31_n200": "gift.w1-v3.queue.sparse31-n200.v1",
                           "sparse76_n300": "gift.w1-v4.queue.sparse76-n300.v1"}
        if (profile not in expected_schema or plan.get("schema") != expected_schema[profile]
                or args.profile is not None and args.profile != profile):
            raise ValueError("existing queue's scientific plan differs")
    else:
        profile = args.profile or "dense151"
        plan = default_plan(directory, profile, anie_source=args.anie_source, fno_source=args.fno_source)
        atomic_json(plan_path, plan)
        protocol_name = {"dense151": "ANIE_MINIMAL_ADAPTATION_PROTOCOL_V2.md",
                         "sparse31_n200": "W1_SPARSE31_N200_PROTOCOL.md",
                         "sparse76_n300": "W1_SPARSE76_N300_PROTOCOL.md"}[profile]
        shutil.copyfile(ROOT / "docs" / protocol_name, plan["protocol_file"])
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    with queue_lease(directory):
        try:
            wait_orphaned_child(directory)
            execute(directory, plan)
        except Exception as error:
            status(directory, "needs_attention", reason=repr(error))
            raise


if __name__ == "__main__":
    main()
