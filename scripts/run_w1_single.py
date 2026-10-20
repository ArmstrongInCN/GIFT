"""Execute the original native-reference N300/76 four-method training recipe.

Four independently initialized author configurations run serially. No candidate
search or repeated seed loop exists. A scientific failure remains a method slot;
an infrastructure interruption stops the queue for a concrete same-run repair.

This queue uses ANIE native_ns_single_v1. It does not reproduce the current
seven-method publication's aligned_ns_single_v1 model. The default current W1
evaluation interface is experiments.additional.w1_current_evaluate; explicit
coordinate-aligned ANIE training uses scripts.run_anie_coordinate_refresh.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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

PROFILE = "single_native76_n300"
SEEDS = {"fno2d": 0, "fno3d": 0, "GIFT": 20260820, "anie_ns": 7}
from experiments.provenance import external_source
from gift.paths import default_data_path
from scripts.queue_runtime import child_environment, configured_source, queue_lease, validate_job_source

OLD_ROOT = external_source('prediction_reservation')
PLAN_DOCUMENT = ROOT / "docs/W1_W2_PROTOCOL.md"


def now():
    """UTC audit clock, separate from the trajectory's physical time."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    """Commit a mutable pointer owned by this run with strict JSON and fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def fixed_json(path, payload):
    """Create one immutable scientific configuration, or verify its identity."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("existing run configuration differs: " + str(path))
    else:
        atomic_json(path, payload)


def file_hash(path):
    """Record provenance without treating byte identity as an environment gate."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def default_plan(directory, *, cost_forecast=None, anie_source=None, fno_source=None):
    """One fixed dataset and four methods; no model score enters this declaration."""
    directory = Path(directory).resolve()
    return dict(schema="gift.w1-single-native.queue.v1", profile=PROFILE,
                root=str(directory), project_root=str(ROOT),
                window=str(directory / "data/train300_validation20_5to8_dt0p04.h5"),
                immutable_window_source=str(external_source('prediction_native') / "data/train300_validation20_5to8_dt0p04.h5"),
                cost_forecast_source=str(Path(cost_forecast).resolve() if cost_forecast is not None
                                         else external_source('prediction_native') / "COST_FORECAST.json"),
                old_test=str(default_data_path("four_vortex.fno_test_dt0p02", "fno/fno_test_n64_n96_n128_dt0p02.h5")),
                protocol_file=str(directory / "PROTOCOL_LOCKED.md"),
                methods=list(SEEDS), seeds=dict(SEEDS), developer_runs=0,
                training_runs_per_method=1, training_count=300, subset_seed=2026100304,
                time_stride=2, observation_dt=.04, observation_points=76, future_steps=75,
                query_times=[.02 * index for index in range(0, 151, 2)], sample_seed=2026100301,
                anie_source=configured_source(anie_source, "GIFT_ANIE_SOURCE_ROOT"),
                fno_source=configured_source(fno_source, "GIFT_EXTERNAL_ROOT"),
                confirmation=dict(seed=2026100302, count=180, ids=[100000, 100179]),
                evaluation_batch=4, device="cuda", score="future75_mean_per_frame_relative_l2",
                historical_events_file=str(OLD_ROOT / "JOB_EVENTS.jsonl"),
                job_events_file=str(directory / "JOB_EVENTS.jsonl"),
                checkpoint_policies=dict(anie_ns="native_validation_mse_best", fno2d="terminal",
                                         fno3d="terminal", GIFT="native_terminal"),
                scientific_failures_preserved=True)


def job_configuration(plan, method):
    """Use each controller's fixed author recipe, never a candidate result."""
    output = Path(plan["root"]) / "training" / f"{method}_seed{SEEDS[method]}"
    common = dict(method=method, profile=PROFILE, stage="formal", seed=SEEDS[method],
                  window=plan["window"], output=str(output), device=plan["device"])
    if method == "GIFT":
        from training.w1_single_gift import native_training_config
        common.update(generator_seed=2026072301, dataset=plan["window"],
                      config=json.loads(json.dumps(native_training_config())))
        return common
    from training.w1_single_train import native_training_config
    spec = dict(method=method, history=1,
                source_root=plan["anie_source"] if method == "anie_ns" else plan["fno_source"],
                query_times=plan["query_times"], future_steps=75, observation_dt=.04)
    if method == "anie_ns":
        spec.update(condition_mode="native", source_profile="native_ns_single_v1")
    if method == "fno2d":
        spec["activation_checkpoint"] = False
    common.update(config=native_training_config(method), model_spec=spec)
    return common


def status(directory, phase, **fields):
    """Expose the active PID and exact command for independent liveness checks."""
    atomic_json(Path(directory) / "ORCHESTRATOR.json",
                dict(status=phase, parent_pid=os.getpid(), updated_utc=now(), **fields))


def wait_surviving_child(directory):
    """Verify a previous child's command and wait without killing it on Windows."""
    from scripts.run_w1_v2 import wait_orphaned_child
    wait_orphaned_child(Path(directory))


def call_module(directory, module, arguments, label):
    """Run a child with argument-list quoting and append-only attempt accounting."""
    directory = Path(directory)
    environment = child_environment()
    sources = {key: environment[key] for key in ("GIFT_EXTERNAL_ROOT", "GIFT_ANIE_SOURCE_ROOT", "GIFT_EXTERNAL_ADAPTER_ROOT")
               if key in environment}
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    started_utc, started = now(), time.perf_counter()
    with (logs / (label + ".log")).open("a", encoding="utf-8") as log:
        log.write("\n" + json.dumps(dict(started_utc=started_utc, command=command, source_directories=sources)) + "\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status(directory, "running", active_job=label, child_pid=child.pid, command=command,
               log=str(logs / (label + ".log")), started_utc=started_utc)
        code = child.wait()
        seconds = time.perf_counter() - started
        event = dict(label=label, command=command, exit_code=code, process_seconds=seconds,
                     started_utc=started_utc, finished_utc=now(), source_directories=sources)
        log.write(json.dumps(event) + "\n")
    with (directory / "JOB_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return code


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def declared_cost_forecast(plan):
    """Import an external timing reservation without changing measured costs.

    The source is a forecast for the same shape and fixed training recipes, not
    a new measurement or a guarantee. Its byte digest records provenance only.
    An explicit source lets a fresh installation supply its own declared
    reservation without storing timing history in the project documents.
    """
    source = Path(plan.get("cost_forecast_source") or
                  external_source('prediction_native') / "COST_FORECAST.json").resolve()
    if source == ROOT or ROOT in source.parents:
        raise ValueError("cost forecast source must be outside GIFT")
    if not source.is_file():
        raise FileNotFoundError("external cost reservation missing; supply --cost-forecast: " + str(source))
    cost = read_json(source)
    if cost.get("schema") != "gift.w1-single-native.cost-forecast.v1":
        raise ValueError("cost reservation does not describe the fixed single-native profile")
    for method in ("anie", "fno2d", "fno3d"):
        budget = cost.get("methods", {}).get(method, {})
        hours = budget.get("estimated_hours_if_full_budget", budget.get("estimated_hours"))
        if (not isinstance(hours, list) or len(hours) != 2
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(value) and value > 0 for value in hours)
                or hours[0] > hours[1]):
            raise ValueError("invalid declared timing range for " + method)
    cost.update(schema="gift.w1-single-native.cost-forecast.v1", implementation_complete=False,
                plan_file=plan["protocol_file"], proposed_run_root=plan["root"],
                forecast_source=dict(path=str(source), sha256=file_hash(source),
                    interpretation="Historical same-shape author-recipe reservation; not a new measurement."))
    return cost


def failure_record(output, job):
    """Preserve scientific failures and stop for unresolved infrastructure errors."""
    path = Path(output) / "FAILURE.json"
    failed = read_json(path)
    if not failed:
        return None
    if failed.get("failure_kind") == "infrastructure":
        raise RuntimeError("infrastructure needs concrete repair before resume: " + str(path))
    return dict(job=job, status="failed", failure=str(path))


def execute_job(directory, job):
    """Train a fresh slot, or continue only that slot's own saved state."""
    directory, output = Path(directory), Path(job["output"])
    config_path = directory / "configs" / (output.name + ".json")
    fixed_json(config_path, job)
    failed = failure_record(output, job)
    if failed:
        return failed
    pointer = output / ("PIPELINE.json" if job["method"] == "GIFT" else "STATUS.json")
    complete = read_json(pointer, {})
    code = 0
    if complete.get("status") != "complete":
        validate_job_source(job)
        module = "training.w1_single_gift" if job["method"] == "GIFT" else "training.w1_single_train"
        arguments = ["--run-config", config_path]
        if output.exists():
            recoverable = (output / "latest.pt").exists() or (output / "generator/latest.pt").exists()
            if not recoverable:
                raise RuntimeError("slot exists without an initial recoverable journal: " + str(output))
            arguments.append("--resume")
        code = call_module(directory, module, arguments, "train_" + output.name)
        complete = read_json(pointer, {})
    failed = failure_record(output, job)
    if failed:
        return failed
    if complete.get("status") == "complete":
        if job["method"] == "GIFT":
            checkpoint = complete.get("paths", {}).get("branch_model")
        else:
            checkpoint = complete.get("selected_checkpoint") or str(output / (
                "best.pt" if job["method"] == "anie_ns" else "terminal_weights.pt"))
        if checkpoint and Path(checkpoint).is_file():
            return dict(job=job, status="complete", checkpoint=str(checkpoint), training_cost=complete)
        raise RuntimeError("completed slot has no declared evaluation weights: " + str(output))
    raise RuntimeError(f"child exited {code} without completion or scientific failure: {output}")


def prepare_run(directory, plan):
    """Copy only immutable observations; initialize no model from old weights."""
    from training.w1_v2_data import WindowBank
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fixed_json(directory / "RUN_PLAN.json", plan)
    protocol = Path(plan["protocol_file"])
    if not protocol.exists():
        shutil.copyfile(PLAN_DOCUMENT, protocol)
    source, window = Path(plan["immutable_window_source"]), Path(plan["window"])
    source_bank = WindowBank(source)
    expected = source_bank.binding
    if expected.get("points") != 76 or len(expected["training"]["ids"]) != 300:
        raise ValueError("source cache is not the N300/76 observed training window")
    del source_bank
    window.parent.mkdir(exist_ok=True)
    if not window.exists():
        pending = window.with_name(window.name + ".copying")
        if pending.exists():
            raise RuntimeError("partial observation copy preserved; inspect before retry: " + str(pending))
        shutil.copyfile(source, pending)
        os.replace(pending, window)
    bank = WindowBank(window)
    if {key: value for key, value in bank.binding.items() if key != "file"} != {
            key: value for key, value in expected.items() if key != "file"}:
        raise ValueError("copied data binding differs")
    expected = dict(bank.binding)
    fixed_json(directory / "DATA_RECEIPT.json", dict(source=str(source), copied_window=str(window),
               data_binding=bank.binding, source_weights_reused=False))
    del bank
    jobs = {method: job_configuration(plan, method) for method in plan["methods"]}
    lock = dict(schema="gift.w1-single-native.protocol-lock.v1", profile=PROFILE,
                protocol_locked=True, configurations_locked=True, training_configurations=jobs,
                protocol_file=plan["protocol_file"], score=plan["score"],
                evaluation_query_times=plan["query_times"], evaluation_dt=.04,
                observation_points=76, future_steps=75, training_count=300,
                training_subset_seed=2026100304, observation_window_file=plan["window"],
                data_binding=expected, confirmation=plan["confirmation"],
                job_events_file=plan["job_events_file"], historical_events_file=plan["historical_events_file"])
    fixed_json(directory / "PROTOCOL_LOCK.json", lock)
    for job in jobs.values():
        fixed_json(directory / "configs" / (Path(job["output"]).name + ".json"), job)
    if not (directory / "COST_FORECAST.json").exists():
        fixed_json(directory / "COST_FORECAST.json", declared_cost_forecast(plan))
    return jobs


def render_report(directory, reports, plan):
    """Combine completed single-model scores without adding seed statistics."""
    directory = Path(directory)
    lines = ["# W1 N300/76点：固定作者配置的单次训练结果", "",
             "每方法从头训练一次，无开发候选或超参数搜索。共同300条训练轨迹、20条validation；t=5单帧输入，预测5.04–8秒75个未来节点。",
             "训练损失、原生阶段和预算各方法不同。本表为单次训练值，无跨seed均值/SD。", ""]
    for cohort, report in reports:
        title = "独立确认180条（主评价）" if cohort == "confirmation" else "旧test180条（探索性复评）"
        lines.extend(["## " + title, "", "完整数值与失败人口见 [numeric_report](evaluation/" + cohort + "/numeric_report.json)。",
                      "", "| 方法 | 实际seed | future75 relative L2 | t8 relative L2 | 未来物理MSE | 失败数/180 |", "|---|---:|---:|---:|---:|---:|"])
        for method, result in report.get("methods", {}).items():
            summary = result.get("summary", {})
            metrics = summary.get("metrics", result.get("metrics", {}))
            def number(key):
                value = metrics.get(key)
                if isinstance(value, dict):
                    value = value.get("mean")
                return "失败/不完整" if value is None else f"{100*value:.3f}%"
            count = summary.get("failed_count", "见numeric_report")
            mse = metrics.get("future_mse", {}).get("mean")
            mse_text = "失败/不完整" if mse is None else f"{mse:.6g}"
            lines.append(f"| {method} | {result.get('seed', SEEDS.get(method))} | {number('future_mean_relative_l2')} | {number('t8_relative_l2')} | {mse_text} | {count}/180 |")
        lines.append("")
        for method, result in report.get("methods", {}).items():
            lines.append(f"- {method}: {result.get('status')}; 测试权重epoch={result.get('selected_epoch', '无合格权重')}" +
                         ("; 原因=" + str(result["reason"]) if result.get("reason") else ""))
        marker = read_json(directory / "evaluation" / cohort / "FIGURES_COMPLETE.json", {})
        for relative in marker.get("files", []):
            if relative.endswith("future_error.png") or relative.endswith("fixed_id_keyframes.png"):
                label = "误差曲线" if relative.endswith("future_error.png") else "固定轨迹关键帧"
                lines.append(f"- [{label}]({(directory / 'evaluation' / cohort / relative).as_posix()})")
        lines.append("")
    events_path = directory / "JOB_EVENTS.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines.extend(["## 当前run实际进程耗时", "", "| 工作 | 次数 | 墙钟小时 |", "|---|---:|---:|"])
    for label in dict.fromkeys(event["label"] for event in events):
        rows = [row for row in events if row["label"] == label]
        lines.append(f"| {label} | {len(rows)} | {sum(row.get('process_seconds', 0.) for row in rows)/3600:.3f} |")
    lines.extend(["", "上述包含各次尝试初始化、训练/原生验证和I/O；中断尝试必须计入。旧run已消耗成本独立列在各评价cost报告，不合并为本run训练时间。",
                  "", "GIFT原.1/.5秒监督lead已改为.08/.48秒；内部RK4 .02秒，保存/观测 .04秒。FNO2D固定课程为1/2/5/10/25/50/75步，最后380轮完整75步；该课程不是原论文设置。",
                  "ANIE原坐标排列恢复；训练/评价统一NS训练分支，eval dropout关闭为显式推理适配。其他模型按终态测试，ANIE按原validation MSE最佳权重测试。",
                  "", "配置与修改说明见PROTOCOL_LOCKED.md、PROTOCOL_LOCK.json及IMPLEMENTATION.md。数值在绘图前保存，固定ID1040/100000、t5.48/6/7/8。"])
    path = directory / "RESULTS.md"
    if path.exists():
        raise RuntimeError("final historical report already exists; verify rather than overwrite")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(directory, plan, jobs):
    """Finish all four slots, independent truth, both cohorts and final reporting."""
    directory = Path(directory)
    slots = []
    for method in plan["methods"]:
        slots.append(execute_job(directory, jobs[method]))
        atomic_json(directory / "FORMAL.json", slots)
    confirmation = directory / "confirmation"
    if not (confirmation / "complete.json").exists():
        arguments = ["--run-generation", "--lock", directory / "PROTOCOL_LOCK.json",
                     "--output", confirmation, "--device", plan["device"], "--batch-size", "8"]
        if confirmation.exists():
            arguments.append("--resume")
        if call_module(directory, "scripts.generate_w1_v2_confirmation", arguments, "confirmation_generation"):
            raise RuntimeError("confirmation infrastructure interruption; preserve/resume its own draw")
    reports = []
    for cohort, data, group in (("confirmation", str(confirmation / "confirmation_n64_dt0p02.h5"), "test"),
                               ("old_test", plan["old_test"], "N64/test")):
        output = directory / "evaluation" / cohort
        arguments = ["--lock", directory / "PROTOCOL_LOCK.json", "--data-file", data,
                     "--group", group, "--cohort", cohort, "--output", output,
                     "--device", plan["device"], "--batch-size", plan["evaluation_batch"],
                     "--job-events", directory / "JOB_EVENTS.jsonl",
                     "--historical-events", plan["historical_events_file"]]
        for slot in slots:
            arguments.extend(["--checkpoint", slot["checkpoint"]] if slot["status"] == "complete"
                             else ["--failed-run", slot["failure"]])
        if output.exists():
            arguments.append("--resume")
        if call_module(directory, "experiments.additional.w1_single_evaluate", arguments, "evaluation_" + cohort):
            raise RuntimeError("evaluation interrupted; retain raw numbers for same-attempt resume")
        reports.append((cohort, read_json(output / "numeric_report.json")))
    if not (directory / "RESULTS.md").exists():
        render_report(directory, reports, plan)
    status(directory, "complete", report=str(directory / "RESULTS.md"), formal_runs=len(slots),
           failed_training_runs=sum(slot["status"] == "failed" for slot in slots), child_pid=None, active_job=None)


def main():
    """Prepare the fixed run, then execute only with explicit --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true",
                        help="train this queue's four native-reference slots; ANIE uses native_ns_single_v1")
    parser.add_argument("--cost-forecast", type=Path,
                        help="external JSON with declared same-profile timing ranges; defaults to prediction_native")
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
        plan = read_json(plan_path)
        if plan.get("schema") != "gift.w1-single-native.queue.v1" or plan.get("profile") != PROFILE:
            raise ValueError("existing queue's scientific plan differs")
        if args.cost_forecast is not None and Path(plan.get("cost_forecast_source", "")).resolve() != args.cost_forecast.resolve():
            raise ValueError("existing queue's declared cost source differs; retain the locked plan")
    else:
        plan = default_plan(directory, cost_forecast=args.cost_forecast,
                            anie_source=args.anie_source, fno_source=args.fno_source)
    with queue_lease(directory):
        try:
            wait_surviving_child(directory)
            jobs = prepare_run(directory, plan)
            if args.run:
                execute(directory, plan, jobs)
            else:
                status(directory, "prepared_not_started", active_job=None, child_pid=None)
                print(json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception as error:
            status(directory, "needs_attention", reason=repr(error), child_pid=None)
            raise


if __name__ == "__main__":
    main()
