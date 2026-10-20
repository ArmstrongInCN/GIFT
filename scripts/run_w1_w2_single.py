"""Extend the completed fixed single-run comparison without retraining its models.

Only U-Net, U-NO and the separately labelled GIFT-N50 are new training slots.
W1 imports the four frozen measurements; W2 evaluates six native predictors on
paired grids. ANIE has no W2 slot. An OS lease and full child receipts prevent
two queues or a silent rerun of a failed scientific outcome.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.run_w1_single import (
    atomic_json, fixed_json, file_hash, read_json, status, queue_lease,
    wait_surviving_child, call_module,
)

from experiments.provenance import external_source
from gift.paths import default_data_path
from scripts.queue_runtime import configured_source, validate_job_source

BASE = external_source('prediction_native')
PROFILE = "single_native76_n300"
NEW_METHODS = ("unet", "uno", "GIFT-N50")
W1_METHODS = ("GIFT", "fno2d", "fno3d", "anie_ns", "uno", "unet", "GIFT-N50")
W2_METHODS = tuple(label for label in W1_METHODS if label != "anie_ns")
WEIGHTS = {"GIFT": BASE / "training/GIFT_seed20260820/branch/model.pt",
           "fno2d": BASE / "training/fno2d_seed0/terminal_weights.pt",
           "fno3d": BASE / "training/fno3d_seed0/terminal_weights.pt",
           "anie_ns": BASE / "training/anie_ns_seed7/best.pt"}


def default_plan(directory, *, fno_source=None):
    """Declare the extension population and workload before reading new scores."""
    directory = Path(directory).resolve()
    return dict(schema="gift.w1w2-native.queue.v1", profile=PROFILE, root=str(directory),
                project_root=str(ROOT), base_root=str(BASE), base_lock=str(BASE / "PROTOCOL_LOCK.json"),
                fno_source=configured_source(fno_source, "GIFT_EXTERNAL_ROOT"),
                extension_lock=str(directory / "EXTENSION_LOCK.json"),
                common_window=str(BASE / "data/train300_validation20_5to8_dt0p04.h5"),
                n50_window=str(directory / "data/train50_validation20_5to8_dt0p04.h5"),
                source_training=str(default_data_path("four_vortex.fno_training_dt0p02", "fno/fno1000_n64_t0_t10_dt0p02.h5")),
                old_test=str(default_data_path("four_vortex.fno_test_dt0p02", "fno/fno_test_n64_n96_n128_dt0p02.h5")),
                n64_confirmation=str(default_data_path("w1_w2.confirmation_n64",
                    str(BASE / "confirmation/confirmation_n64_dt0p02.h5"))),
                new_training_methods=list(NEW_METHODS), w1_methods=list(W1_METHODS), w2_methods=list(W2_METHODS),
                reused_training_methods=list(WEIGHTS), new_runs_per_method=1,
                development_candidates=0, hyperparameter_search=False, repeated_seed_runs=False,
                device="cuda", w1_evaluation_batch=4, w2_evaluation_batch=1,
                observation_dt=.04, observation_points=76, training_count=300,
                subset_seed=2026100304, validation_count=20, anie_in_w2=False,
                job_events_file=str(directory / "JOB_EVENTS.jsonl"),
                cohort_interpretation="same independent held-out180 already evaluated in v5; extension is not newly blind",
                publication_pending=True)


def job_configuration(plan, label, query_times):
    """Keep each fixed native recipe; only the observed channel count is adapted."""
    seed = 20260820 if label == "GIFT-N50" else 0
    job = dict(method=label, profile=PROFILE, stage="formal", seed=seed,
               window=plan["n50_window"] if label == "GIFT-N50" else plan["common_window"],
               output=str(Path(plan["root"]) / "training" / f"{label}_seed{seed}"), device=plan["device"])
    if label == "GIFT-N50":
        from training.w1_single_gift import native_training_config
        job.update(generator_seed=2026072301, expected_training_count=50,
                   config=json.loads(json.dumps(native_training_config())))
    else:
        from training.w1_extended_baselines import native_training_config
        recorded = read_json(Path(plan["root"]) / "configs" / (Path(job["output"]).name + ".json"), {})
        source = plan.get("fno_source", recorded.get("model_spec", {}).get("source_root"))
        job.update(config=native_training_config(label), model_spec=dict(method=label, history=1,
                   source_root=source, query_times=query_times,
                   future_steps=75, observation_dt=.04))
    return job


def prepare_run(directory, plan):
    """Freeze source checkpoints, exact nested subsets and all new configurations."""
    from training.w1_v2_data import WindowBank, prepare_window
    from experiments.additional.w1_single_evaluate import checkpoint_metadata
    directory = Path(directory)
    fixed_json(directory / "RUN_PLAN.json", plan)
    base = read_json(plan["base_lock"])
    bank = WindowBank(plan["common_window"])
    if bank.binding != base["data_binding"]:
        raise ValueError("common observation cache differs from the completed base lock")
    common_binding = bank.binding
    del bank
    reused = {}
    for label, path in WEIGHTS.items():
        metadata = checkpoint_metadata(path, base)
        if metadata["model_spec"]["method"] != label:
            raise ValueError("reused checkpoint label mismatch")
        reused[label] = dict(checkpoint=str(path), sha256=file_hash(path),
                             source_root=str(BASE), new_training_seconds=0.)
    prepare_window(plan["source_training"], plan["n50_window"], training_count=50,
                   time_stride=2, subset_seed=plan["subset_seed"])
    bank = WindowBank(plan["n50_window"])
    n50_binding = bank.binding
    if n50_binding["training"]["ids"] != common_binding["training"]["ids"][:50] or n50_binding["validation"] != common_binding["validation"]:
        raise ValueError("reduced control is not nested first50 with the same20 validation")
    del bank
    jobs = {label: job_configuration(plan, label, common_binding["query_times"]) for label in NEW_METHODS}
    lock = dict(schema="gift.w1w2-native.extension-lock.v1", profile=PROFILE,
                configurations_locked=True, protocol_locked=True,
                base_lock=plan["base_lock"], base_lock_sha256=file_hash(plan["base_lock"]),
                reused_models=reused, training_configurations=jobs,
                common_data_binding=common_binding, n50_data_binding=n50_binding,
                w1_methods=list(W1_METHODS), w2_methods=list(W2_METHODS), anie_in_w2=False,
                cohort_interpretation=plan["cohort_interpretation"],
                w1_score="future75_mean_per_frame_relative_l2", w2_score="future25_mean_per_frame_relative_l2",
                w2_query_times=[.04 * i for i in range(26)],
                w2_fno3d="keep75 trained joint queries and truncate25; frozen training statistics Fourier lift",
                n50_interpretation="same phase epochs/optimizer; smaller data and update budget; not original GIFT-Lite")
    fixed_json(plan["extension_lock"], lock)
    for job in jobs.values():
        fixed_json(directory / "configs" / (Path(job["output"]).name + ".json"), job)
    fixed_json(directory / "DATA_RECEIPT.json", dict(common=common_binding, reduced=n50_binding,
                common_data_reused=True, nested_subset=True, validation_unchanged=True, test_not_used_for_training=True))
    fixed_json(directory / "COST_FORECAST.json", dict(schema="gift.w1w2-native.cost-forecast.v1",
        estimates_are_not_promises=True, new_training_hours={"unet": [.4, 1.], "uno": [3., 4.], "GIFT-N50": [.6, 1.2]},
        w2_truth_hours=[.5, 2.], evaluation_plot_audit_document_hours=[1., 3.],
        basis="historical different-input workloads and N300 native GIFT; replace with same-run committed epoch totals",
        new_gpu_probe=False, reused_training_hours=0.))
    protocol = directory / "PROTOCOL_LOCKED.md"
    if not protocol.exists():
        protocol.write_text("""# W1/W2 单次训练补充协议

基础四方法权重及其W1测量直接复用v5，不再训练：GIFT、FNO2D、FNO3D、ANIE。
新增U-NO与U-Net，各seed0、500轮、Adam lr.001、StepLR100×.5、终态测试；batch16/20、weight_decay1e-5/1e-4。
保留项目训练目标：U-NO每条从t5单初态闭环20步(t5.04–5.80)，U-Net4步(t5.04–5.16)，不detach。
因此这是共同输入/数据人口下的固定配方比较，训练损失、目标覆盖和计算预算不同，不声称架构单因素消融。
另训练GIFT-N50：N300固定子集的前50条；原GIFT生成元和分支seed、阶段轮数/batch/optimizer/scheduler保持。
GIFT-N50仅为减量数据及更新预算对照，不等于历史GIFT-Lite，也不与N300方法声称相同训练预算。
共同原20validation；t5..8每.04共76点；单真初态；W1 future75逐帧relative L2；不选参、无多seed。
GIFT内部RK4仍.02秒/150步，保存.04秒；原.1/.5监督目标变为.08/.48秒，coarse FD只用76观察节点。
W2仅GIFT、FNO2D、FNO3D、U-NO、U-Net、GIFT-N50，ANIE完全不加入跨分辨率实验。
W2 N64训练权重直接N64/N96/N128预测t5..6每.04共26点；future25相同人口逐帧relative L2；无目标网格拟合。
FNO3D保留原75个joint query后取前25，normalizer只对冻结训练统计作确定性周期Fourier网格插值。
确认180条复用v5已评价人口及精确初始参数；新N96/N128真值独立在该网格积分，不重抽样。
本扩展不再是新盲测：基础四方法确认结果在设计时已知。旧test180仍探索性，禁止根据任一test调参。
失败seed及失败轨迹完整计入。数值先保存再绘图；历史结果、权重、外部源码保持原样。
参数、数据和权重来源见EXTENSION_LOCK.json/DATA_RECEIPT.json；全部尝试成本见JOB_EVENTS.jsonl。
""", encoding="utf-8")
    return jobs, lock


def execute_job(directory, job):
    """Run a new slot once, or resume only its own infrastructure journal."""
    output = Path(job["output"])
    failure = read_json(output / "FAILURE.json")
    if failure:
        if failure.get("failure_kind") == "infrastructure":
            raise RuntimeError("repair and preserve infrastructure evidence before resume: " + str(output))
        return dict(method=job["method"], seed=job["seed"], status="failed", failure=str(output / "FAILURE.json"))
    gift = job["method"] == "GIFT-N50"
    state = read_json(output / ("PIPELINE.json" if gift else "STATUS.json"), {})
    if state.get("status") != "complete":
        validate_job_source(job)
        arguments = ["--run-config", Path(directory) / "configs" / (output.name + ".json")]
        if output.exists():
            if not ((output / "latest.pt").is_file() or (output / "generator/latest.pt").is_file()):
                raise RuntimeError("existing slot lacks recoverable full state: " + str(output))
            arguments.append("--resume")
        module = "training.w1_extended_gift_n50" if gift else "training.w1_extended_baselines"
        code = call_module(directory, module, arguments, "train_" + output.name)
        failure = read_json(output / "FAILURE.json")
        if failure:
            if failure.get("failure_kind") == "infrastructure":
                raise RuntimeError("training infrastructure interrupted: " + str(failure))
            return dict(method=job["method"], seed=job["seed"], status="failed", failure=str(output / "FAILURE.json"))
        state = read_json(output / ("PIPELINE.json" if gift else "STATUS.json"), {})
        if state.get("status") != "complete":
            raise RuntimeError(f"training child exited {code} without completed or failed slot: {output}")
    checkpoint = Path(state["paths"]["branch_model"] if gift else state["selected_checkpoint"])
    if not checkpoint.is_file():
        raise RuntimeError("completed training checkpoint missing")
    return dict(method=job["method"], seed=job["seed"], status="complete", checkpoint=str(checkpoint),
                checkpoint_sha256=file_hash(checkpoint), training_cost=state, reused_training=False)


def model_arguments(slots, methods):
    arguments = []
    for method in methods:
        slot = slots[method]
        arguments += ["--checkpoint", method + "=" + slot["checkpoint"]] if slot["status"] == "complete" else [
            "--failed-run", method + "=" + slot["failure"]]
    return arguments


def execute(directory, plan, jobs, lock):
    """Finish training, both W1 cohorts and the ANIE-free paired W2 cohorts."""
    directory = Path(directory)
    slots = {label: dict(method=label, status="complete", checkpoint=str(path),
                checkpoint_sha256=lock["reused_models"][label]["sha256"], reused_training=True,
                new_training_seconds=0.) for label, path in WEIGHTS.items()}
    atomic_json(directory / "FORMAL.json", slots)
    for label in NEW_METHODS:
        slots[label] = execute_job(directory, jobs[label])
        atomic_json(directory / "FORMAL.json", slots)
    for cohort, data, group in (("confirmation", plan["n64_confirmation"], "test"), ("old_test", plan["old_test"], "N64/test")):
        output = directory / "w1_evaluation" / cohort
        arguments = ["--base-lock", plan["base_lock"], "--extension-lock", plan["extension_lock"],
            "--data-file", data, "--group", group, "--cohort", cohort, "--output", output,
            "--reuse-evaluation", BASE / "evaluation" / cohort, "--device", plan["device"],
            "--batch-size", plan["w1_evaluation_batch"], "--job-events", plan["job_events_file"]]
        arguments += model_arguments(slots, W1_METHODS)
        if output.exists():
            arguments.append("--resume")
        if call_module(directory, "experiments.additional.w1_extended_evaluate", arguments, "w1_evaluation_" + cohort):
            raise RuntimeError("W1 evaluation interrupted; preserve raw output for same-run resume")
    confirmation = directory / "w2_confirmation"
    if not (confirmation / "complete.json").is_file():
        arguments = ["--base-lock", plan["base_lock"], "--extension-lock", plan["extension_lock"],
            "--n64-confirmation", plan["n64_confirmation"], "--output", confirmation,
            "--device", plan["device"], "--batch-size", "4", "--run-generation"]
        if confirmation.exists():
            arguments.append("--resume")
        if call_module(directory, "scripts.generate_w2_single_confirmation", arguments, "w2_confirmation_generation"):
            raise RuntimeError("paired W2 reference generation interrupted")
    # Completion filename is checked below by the actual generation receipt.
    generated = confirmation / "confirmation_n96_n128_dt0p02.h5"
    for cohort, data in (("confirmation", generated), ("old_test", Path(plan["old_test"]))):
        output = directory / "w2_evaluation" / cohort
        arguments = ["--base-lock", plan["base_lock"], "--extension-lock", plan["extension_lock"],
            "--data-file", data, "--cohort", cohort, "--output", output, "--device", plan["device"],
            "--batch-size", plan["w2_evaluation_batch"], "--job-events", plan["job_events_file"]]
        if cohort == "confirmation":
            arguments += ["--n64-data-file", plan["n64_confirmation"]]
        arguments += ["--reuse-evaluation", directory / "w1_evaluation" / cohort]
        arguments += model_arguments(slots, W2_METHODS)
        if output.exists():
            arguments.append("--resume")
        if call_module(directory, "experiments.additional.w2_single_evaluate", arguments, "w2_evaluation_" + cohort):
            raise RuntimeError("W2 evaluation interrupted; retain same grids/population/weights")
    status(directory, "evaluation_complete_documents_pending", child_pid=None, active_job=None,
           formal_slots=7, new_training_slots=3, w2_methods=6, anie_in_w2=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--fno-source", type=Path, help="upstream clone parent; overrides GIFT_EXTERNAL_ROOT")
    args = parser.parse_args()
    if args.fno_source is not None:
        os.environ["GIFT_EXTERNAL_ROOT"] = configured_source(args.fno_source, "GIFT_EXTERNAL_ROOT")
    directory = args.root.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan = read_json(directory / "RUN_PLAN.json")
    if plan is None:
        plan = default_plan(directory, fno_source=args.fno_source)
    elif plan.get("schema") != "gift.w1w2-native.queue.v1" or plan.get("profile") != PROFILE:
        raise ValueError("existing queue's scientific plan differs")
    with queue_lease(directory):
        try:
            wait_surviving_child(directory)
            jobs, lock = prepare_run(directory, plan)
            if args.run:
                execute(directory, plan, jobs, lock)
            else:
                status(directory, "prepared_not_started", child_pid=None, active_job=None)
                print(json.dumps(dict(root=str(directory), new_methods=NEW_METHODS, reused=list(WEIGHTS),
                                      w2_methods=W2_METHODS), ensure_ascii=False, indent=2))
        except Exception as error:
            status(directory, "needs_attention", reason=repr(error), child_pid=None)
            raise


if __name__ == "__main__":
    main()
