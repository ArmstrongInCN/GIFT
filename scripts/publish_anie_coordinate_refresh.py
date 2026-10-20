"""Publish the single ANIE coordinate-aligned W1 model and activate its records.

Only completed, independently audited measurements may be published. The other
six W1 methods remain byte-identical source measurements; W2 is not changed.
Figures call the existing manuscript renderer, including its fixed trajectory,
PCHIP guide, field plate, signed residuals, and shared colour limits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np

from experiments.formal._shared.figure_evidence import file_record, sha256, verify_svg
from scripts import plot_w1_original_style as plot
from scripts.publish_w1_w2_single import csv_text, portable, summary_rows
from scripts.publish_original_style_figures import copy_once, numeric_source, write_once
from scripts.verify_published_results import verify_package
from scripts.run_w1_single import atomic_json

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent / "runs/w1w2_native_single76_n300_v6_20261005"
PACKAGE_NAME = "W1_flow_prediction_20261007"
OLD_PACKAGE_NAME = "W1_native_single76_n300_original_style_20261005"
COHORTS = ("confirmation", "old_test")
METHODS = {"GIFT", "GIFT-N50", "fno2d", "fno3d", "uno", "unet", "anie_ns"}
DOCS = (
    "README.md", "EXPERIMENTS.md", "docs/RESULTS.md", "docs/FIGURES.md",
    "docs/CHECKPOINTS.md", "docs/TRAINING_PROTOCOL.md", "docs/PERFORMANCE.md",
    "docs/SETUP.md", "docs/VALIDATION.md", "docs/EXPERIMENT_EXECUTION.md",
    "docs/EXTERNAL_ADAPTATIONS.md",
    "docs/W1_W2_PROTOCOL.md",
)
from experiments.provenance import external_source

OLD_CHECKPOINT = str(external_source('reference_anie_checkpoint'))
OLD_CHECKPOINT_SHA = "7fdd7e521db2f5a70139ea550b46038d322b672319c17aebb920d1b95c0191ae"
PENDING_VISUAL_STATEMENT = "图像外观须另行实际查看并记录VISUAL_QA.json。"
COMPLETED_VISUAL_STATEMENT = "四张PNG已实际查看：两幅七方法七实测节点曲线和两幅8行74轴场板；固定ID100005/1045、t5/6/7/8及共同色阶保持，标签清晰、有限值未裁切。实际外观核验记录另由VISUAL_QA.json绑定。"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_once(path, content):
    """Create a scientific/document artifact, accepting only an identical retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"existing artifact differs: {path}")
    else:
        path.write_text(content, encoding="utf-8")


def completed_inputs(root, base):
    """Bind completed training, full180 cohorts and unchanged six-method sources."""
    status_path = root / "training/anie_ns_seed7/STATUS.json"
    status = read(status_path)
    if status.get("method") != "anie_ns" or status.get("seed") != 7:
        raise ValueError("ANIE publication requires the unique seed7 training slot")
    failed = status.get("status") != "complete"
    if failed:
        failure = read(status_path.parent / "FAILURE.json")
        if (failure.get("failure_kind") not in {"numerical", "resource"}
                or failure.get("method") != "anie_ns" or failure.get("seed") != 7):
            raise ValueError("only a documented scientific failure may replace a complete training outcome")
        checkpoint, checkpoint_hash = None, None
    else:
        failure = None
        if not (0 < status["best_epoch"] <= status["epoch"] <= 5000) or status["updates"] != 10 * status["epoch"]:
            raise ValueError("ANIE fixed-budget progress is inconsistent")
        checkpoint = Path(status["selected_checkpoint"]).resolve(strict=True)
        if checkpoint.parent != status_path.parent.resolve() or checkpoint.name != "best.pt":
            raise ValueError("selected checkpoint is not this run's validation-MSE-best")
        checkpoint_hash = sha256(checkpoint)
    audits = {}
    for filename in ("FINAL_RESULT_AUDIT.json", "FINAL_TRAINING_AUDIT.json"):
        audit = read(root / filename)
        if str(audit.get("status", "")).lower() not in {"passed", "pass", "complete"}:
            raise ValueError(f"publication requires a passed audit: {filename}")
        audits[filename] = sha256(root / filename)
    reports = {}
    for cohort in COHORTS:
        path = root / "w1_evaluation" / cohort / "numeric_report.json"
        current, original = read(path), read(base / "w1_evaluation" / cohort / "numeric_report.json")
        if (current.get("status") != "numerical_complete" or current.get("population") != 180
                or current.get("cohort") != cohort or set(current["methods"]) != METHODS):
            raise ValueError("W1 publication requires exactly seven complete population slots")
        for field in ("trajectory_ids", "absolute_times", "query_times"):
            if current[field] != original[field]:
                raise ValueError("cohort or observed nodes changed")
        for key in METHODS - {"anie_ns"}:
            for field in ("status", "seed", "checkpoint_sha256", "raw_prediction_sha256", "summary"):
                if current["methods"][key][field] != original["methods"][key][field]:
                    raise ValueError(f"the reused {key} measurement changed: {field}")
        anie = current["methods"]["anie_ns"]
        if failed:
            if (anie.get("seed") != 7 or anie.get("status") != "training_failed"
                    or anie["summary"]["failed_count"] != 180 or anie["summary"].get("metrics")
                    or anie["summary"].get("primary_metric") is not None):
                raise ValueError("the scientific failure must retain the full180 slot without finite metrics")
        elif (anie["seed"] != 7 or Path(anie["checkpoint"]).resolve() != checkpoint
              or anie["checkpoint_sha256"].upper() != checkpoint_hash):
            raise ValueError("new ANIE report belongs to a different trained model")
        reports[cohort] = current
    events_path = root / "JOB_EVENTS.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(event["label"], event["started_utc"]) for event in events]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate process attempts in cost journal")
    training_events = [event for event in events if event["label"].lower().startswith("train") and "anie" in event["label"].lower()]
    if not training_events or not all("process_seconds" in event and "exit_code" in event for event in training_events):
        raise ValueError("ANIE training process cost is not closed")
    reused_cost_path = base / "FINAL_COST_CLOSURE.json"
    reused_cost = read(reused_cost_path) if reused_cost_path.is_file() else None
    unchanged_training_seconds = (sum(value for key,value in reused_cost["reused_model_original_training_process_seconds"].items() if key != "anie_ns")
        + sum(record["process_seconds"] for key,record in reused_cost["grouped_process_wall"].items() if key.startswith("train_"))
        if reused_cost is not None else None)
    return dict(status=status, status_path=status_path, checkpoint=checkpoint,
                checkpoint_hash=checkpoint_hash, reports=reports, audits=audits,
                training_process_seconds=sum(event["process_seconds"] for event in training_events),
                events_hash=sha256(events_path), training_attempts=training_events, failed=failed, failure=failure,
                unchanged_training_seconds=unchanged_training_seconds)


def render_figures(root, base, inputs, figures):
    """Use the same renderer; a scientifically failed model gets explicit tiles."""
    if not inputs["failed"]:
        plot.main(["--input", str(root / "w1_evaluation"), "--output", str(figures)])
        return
    sources = plot.renderer_sources()
    measured = {}
    for cohort, trajectory_id in (("confirmation", 100005), ("old_test", 1045)):
        data = plot.read_cohort(base / "w1_evaluation" / cohort, cohort, trajectory_id)
        report_path = root / "w1_evaluation" / cohort / "numeric_report.json"
        old_anie_source = next(record["actual_read_path"] for record in data["raw_provenance"] if record["raw_method_key"] == "anie_ns")
        data.update(report_path=report_path, report_sha256=sha256(report_path), report=inputs["reports"][cohort])
        data["predictions"]["anie"] = np.full_like(data["truth"], np.nan)
        data["residuals"]["anie"] = np.full_like(data["truth"], np.nan)
        data["errors"]["anie"] = np.full(4, np.nan)
        data["sources"] = [record for record in data["sources"] if record["path"] != old_anie_source]
        data["sources"].append(file_record(root / "training/anie_ns_seed7/FAILURE.json", root))
        data["raw_provenance"] = [record for record in data["raw_provenance"] if record["raw_method_key"] != "anie_ns"]
        data["raw_provenance"].append(dict(label="ANIE", raw_method_key="anie_ns", panel_slug="anie",
            scientific_status="training_failed", reason=inputs["failure"]["reason"],
            initial_and_future_fields="NaN failure placeholders; no prediction was fabricated"))
        for row in data["curve_rows"]:
            if row["raw_method_key"] == "anie_ns":
                row.update(mean_relative_l2=None, complete_population=False, finite_count_min=0)
        measured[cohort] = data
    shared_y = max(1.28, 1.08*max(row["mean_relative_l2"] for data in measured.values()
                                for row in data["curve_rows"] if row["mean_relative_l2"] is not None))
    for cohort in COHORTS:
        plot.plot_cohort(figures / cohort, cohort, measured[cohort], sources, shared_y)


def publish(root, base):
    """Export a new scoped package through the existing original-style renderers."""
    inputs = completed_inputs(root, base)
    reports = inputs["reports"]
    figures = root / "original_style_figures/W1"
    if not all((figures / cohort / "figure_manifest.json").is_file() for cohort in COHORTS):
        if any((figures / cohort).exists() for cohort in COHORTS):
            raise FileExistsError("partial plot export is preserved; inspect before another export")
        render_figures(root, base, inputs, figures)
    package = PROJECT / "results/formal" / PACKAGE_NAME
    protocol = root / "PROTOCOL_LOCK.json"
    measurement = dict(schema="gift.w1w2-native.published-measurement.v1", experiment="W1",
        profile="single_native76_n300", status="numerical_complete", single_model_values=True,
        new_blind_test=False, anie_in_w2=False, cohorts=portable(reports),
        anie_coordinates="space_major_time_minor; state and (x,y,t) share each token index",
        source_numeric_report_sha256={cohort: sha256(root / "w1_evaluation" / cohort / "numeric_report.json") for cohort in COHORTS},
        protocol_lock_sha256=sha256(protocol), result_audit_sha256=inputs["audits"]["FINAL_RESULT_AUDIT.json"])
    write_once(package / "summary/summary.json", measurement)
    rows = summary_rows(reports, "W1")
    text_once(package / "summary/metrics.csv", csv_text(rows, list(rows[0])))
    report_hash = sha256(package / "summary/summary.json")
    figure_directories = []
    for cohort in COHORTS:
        source = figures / cohort
        source_manifest = read(source / "figure_manifest.json")
        report_path = root / "w1_evaluation" / cohort / "numeric_report.json"
        if (source_manifest.get("status") != "complete"
                or source_manifest["evidence"]["report_sha256"].upper() != sha256(report_path)):
            raise ValueError("figure is not bound to the current numerical report")
        for record in source_manifest["files"]:
            item = source / record["path"]
            if item.stat().st_size != record["bytes"] or sha256(item) != record["sha256"].upper():
                raise ValueError(f"source figure bytes changed: {item}")
        destination = package / f"figures/{cohort}"
        destination.mkdir(parents=True, exist_ok=True)
        previews, metadata = [], {}
        for item in sorted(source.rglob("*")):
            if not item.is_file() or item.name == "figure_manifest.json":
                continue
            if item.suffix == ".png":
                asset = PROJECT / "assets/figures" / f"w1_aligned_anie_single76_n300_{cohort}_{item.stem}_20261006.png"
                copy_once(item, asset)
                previews.append(file_record(asset, PROJECT))
            elif item.suffix == ".npz":
                metadata[item.name] = numeric_source(item, destination / item.relative_to(source))
            elif item.suffix in {".svg", ".csv", ".json"}:
                copy_once(item, destination / item.relative_to(source))
            else:
                raise ValueError(f"unexpected source plot file: {item}")
        files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name != "figure_manifest.json")
        svg_checks = {path.relative_to(destination).as_posix(): verify_svg(path, editable_text="panels" not in path.parts)
                      for path in files if path.suffix == ".svg"}
        evidence = dict(source_manifest["evidence"], report_sha256=report_hash,
            source_numeric_report_sha256=sha256(report_path),
            source_figure_manifest_sha256=sha256(source / "figure_manifest.json"),
            sources=[file_record(package / "summary/metrics.csv", package)])
        payload = dict(source_manifest, evidence=evidence, source_array_metadata=metadata,
            figure_contract=dict(backend="Python matplotlib", archetype="original manuscript comparison templates",
                editable_vector_curves=True, editable_vector_fields=True, field_previews=previews,
                residual_definition="prediction minus truth", single_training_instance=True, ANIE_in_W2=False),
            files=[file_record(path, destination) for path in files], svg_checks=svg_checks)
        write_once(destination / "figure_manifest.json", payload)
        figure_directories.append(f"figures/{cohort}")
    inventory = sorted(path for path in package.rglob("*") if path.is_file() and path.name != "published.json")
    write_once(package / "published.json", dict(schema="gift.published-results.v1", experiment="W1",
        scope="summaries_and_selected_figures", full_raw_outputs_included=False, resumable_experiment=False,
        protocol_version="single_native76_n300_aligned_anie_20261006", single_model_values=True,
        measurement=dict(report_sha256=report_hash, source_measurement=measurement),
        figure_directories=figure_directories, files=[file_record(path, package) for path in inventory]))
    integrity = verify_package(package, "W1")
    receipt = dict(status="complete", package=str(package), package_manifest_sha256=sha256(package / "published.json"),
                   source_audits=inputs["audits"], reused_methods_unchanged=sorted(METHODS-{"anie_ns"}),
                   ANIE_in_W2=False, integrity=integrity)
    write_once(root / "PUBLICATION.json", receipt)
    render_result_record(root, package, inputs)
    return inputs, package


def metric(record, name):
    value = record["summary"].get("metrics", {}).get(name)
    return value["mean"] if isinstance(value, dict) else value


def render_result_record(root, package, inputs):
    """Write the complete measured W1 table with native budgets and figure links."""
    status = inputs["status"]
    selection = ("训练发生声明的科学失败，无可评价模型。全部180条失败人口按原槽位保留。" if inputs["failed"] else
                 f"ANIE实际完成{status['epoch']}轮、{status['updates']}次更新，采用第{status['best_epoch']}轮validation-MSE-best权重。")
    lines = ["# W1：N300、76点、单初态、固定配方", "",
        "比较GIFT、GIFT-N50、FNO2D、FNO3D、U-NO、U-Net和ANIE的真实单模型值；每方法一个声明种子，无跨seed标准差。",
        "共同观测t5–8每0.04秒共76点，输入仅为每条轨迹自己的t5真实状态，主指标为future75逐帧relative L2均值，排除已知初态。",
        "两个留出人口均已被查看，本次评价是非盲测的事后测量，未用于参数或模型选择；确认与探索人口分别报告。", "",
        "ANIE状态与(x,y,t)坐标使用相同的space-major/time-minor token排列，训练和评价一致。保持NS架构、c=0/y_init条件路径、全76点物理MSE和原优化器/调度/早停。",
        selection, f"ANIE训练进程墙钟：{inputs['training_process_seconds']:.3f}秒（{inputs['training_process_seconds']/3600:.4f}小时），含全部登记尝试、准备、训练、内部validation与checkpoint I/O。", "",
        "GIFT使用0.08/0.48秒监督目标（名义目标0.1/0.5秒），内部RK4 dt=0.02，保存间隔0.04秒。U-NO监督20步至t5.80，U-Net监督4步至t5.16。GIFT-N50使用有序N300名单前50条，沿用阶段epoch但减少总更新。固定损失、监督范围和计算预算不同，不能把结果解释为等预算的纯架构比较。",
        "ANIE仅参加W1；W2六方法保持其声明权重和结果。", ""]
    labels = {spec['group']: spec['label'] for spec in plot.SPECS}
    for cohort in COHORTS:
        report = inputs["reports"][cohort]
        lines += [f"## {cohort}", "", "| 方法 | Ntrain | seed | future75 mean relative L2 | t8 relative L2 | future MSE | 失败/180 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for spec in plot.SPECS:
            key = spec["group"]
            record = report["methods"][key]
            values = [metric(record, field) for field in ("future_mean_relative_l2", "t8_relative_l2", "future_mse")]
            formatted = ["无总体值" if value is None else f"{value:.9g}" for value in values]
            lines.append(f"| {labels[key]} | {50 if key == 'GIFT-N50' else 300} | {record['seed']} | {' | '.join(formatted)} | {record['summary']['failed_count']}/180 |")
        figure_manifest = read(package / "figures" / cohort / "figure_manifest.json")
        lines += ["", f"[完整数值报告]({(root/'w1_evaluation'/cohort/'numeric_report.json').as_posix()})", ""]
        for preview in figure_manifest["figure_contract"]["field_previews"]:
            lines += [f"![{cohort} {Path(preview['path']).stem}]({(PROJECT/preview['path']).as_posix()})", ""]
    lines += ["## 证据与图像", "", f"[发布包]({(package/'published.json').as_posix()})。曲线和场板沿用既有原稿绘图结构、方法、固定ID100005/1045和t5/6/7/8；残差为prediction−truth，有限值不裁切。",
        "FINAL_TRAINING_AUDIT.json与FINAL_RESULT_AUDIT.json绑定真实训练槽位、各180人口、原始预测和范数重算；PUBLICATION.json核对发布包。图像外观须另行实际查看并记录VISUAL_QA.json。",
        "所述实验是公开NS实现的固定配方任务迁移，不是原论文全部数据与设置的完整复现。"]
    content = "\n".join(lines) + "\n"
    record_path = root / "RESULTS.md"
    if record_path.exists() and record_path.read_text(encoding="utf-8") == content.replace(PENDING_VISUAL_STATEMENT, COMPLETED_VISUAL_STATEMENT):
        return
    text_once(record_path, content)


def measured_curve_table(report):
    """Render all seven real saved report nodes without re-evaluating a model."""
    method_keys = ("GIFT", "GIFT-N50", "fno2d", "uno", "unet", "anie_ns", "fno3d")
    indices = plot.nearest_times(np.asarray(report["absolute_times"]), plot.REQUESTED_TIMES)
    lines = ["| 时间 | GIFT | GIFT-N50 | FNO2D | U-NO | U-Net | ANIE | FNO3D |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index in indices:
        cells = []
        for key in method_keys:
            summary = report["methods"][key]["summary"]
            curve = summary.get("mean_relative_l2_by_time")
            value = curve[int(index)] if curve is not None and summary["failed_count"] == 0 else None
            cells.append("无总体值" if value is None else f"{value:.6f}")
        lines.append(f"| {report['absolute_times'][int(index)]:g} | {' | '.join(cells)} |")
    return "\n".join(lines)


def refresh_curve_tables(text, reports):
    """Update only the existing cohort time tables, retaining the record structure."""
    pattern = r"(?m)^\| 时间 \| GIFT \| GIFT-N50 \| FNO2D \| U-NO \| U-Net \| ANIE \| FNO3D \|\n(?:^\|[^\n]*\|\n?)+"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text
    if len(matches) > 2:
        raise ValueError("the current W1 record has duplicate cohort time tables")
    for match, cohort in reversed(list(zip(matches, COHORTS))):
        text = text[:match.start()] + measured_curve_table(reports[cohort]) + "\n" + text[match.end():]
    return text


def reconcile_record_wording(relative, text, root=None):
    """Keep final records precise about coordinates, exposure and source scope."""
    text = text.replace("但其中两个评价人口的成绩已被查看", "但两个评价人口的成绩已被查看")
    text = text.replace("确认集独立于训练和验证，但四种方法的结果此前已被查看，因此不是全部方法的盲测。",
                        "两个评价人口均独立于训练和验证，成绩均已被查看，因此此次评价为非盲测的事后测量。")
    text = text.replace("既有四方法结果在扩展前已知，故不是本次新盲确认", "两个评价人口的成绩均已被查看，故此次评价为非盲测的事后测量")
    text = text.replace("原生token坐标与76个查询点", "与状态逐token一致的时空坐标排列及76个查询点")
    if relative == "docs/RESULTS.md":
        text = text.replace("# 当前单次固定配方 W1/W2 结果的解释（2026-10-05）", "# 当前单次固定配方 W1/W2 结果的解释")
        text = re.sub(r"(?m)^- v6：(?=`w1_evaluation)", f"- 非ANIE方法及W2的数值来源（外部目录 `{BASE.as_posix()}`）：", text)
        text = re.sub(r"(?m)^- v6：(?=`training)", f"- U-NO/U-Net的训练来源（外部目录 `{BASE.as_posix()}`）：", text)
        text = re.sub(r"(?m)^- v6：", "- 声明运行目录中的来源：", text)
        if root is not None and "- ANIE数值与训练来源" not in text:
            marker = "- 项目："
            source = f"- ANIE数值与训练来源（外部目录 `{root.as_posix()}`）：`w1_evaluation/confirmation/numeric_report.json`、`w1_evaluation/old_test/numeric_report.json`、`training/anie_ns_seed7/STATUS.json`、`HISTORY.json`、`best.pt`及`JOB_EVENTS.jsonl`。\n"
            text = text.replace(marker, source + marker)
        text = text.replace("本次没有选择 validation-best：", "该固定配方选择终态：")
        text = text.replace("此前CPU测试已验证", "CPU测试验证了")
        text = text.replace("本记录没有重新推理或新增训练来改变这些结果。", "上述验证限定于这些已声明模型的来源和接口。")
        text = text.replace("本次按当前标准保留，并改为N300、单初态", "本实验采用该配置，使用N300、单初态")
    if relative == "docs/EXTERNAL_ADAPTATIONS.md":
        text = text.replace("## 当前单初態 W1/W2 profile（2026-10-05）", "## 当前单初態 W1/W2 profile")
        text = text.replace("[v5实施记录]", "[单次固定配方实现]")
        text = re.sub(r"当前新三模型已完成各自固定训练，[^\n]*完整范围见 \[当前成本\]\(PERFORMANCE\.md\)。",
            "各方法的训练与评价来源绑定于对应结果包，独立训练和数值审计见声明运行目录；实际计算成本及范围见 [PERFORMANCE.md](PERFORMANCE.md)。", text)
    if relative == "docs/CHECKPOINTS.md" and root is not None:
        text = text.replace("模型身份、配方、种子和数据绑定存于", "其余六方法的模型身份、配方、种子和数据绑定存于") if "其余六方法的模型身份" not in text else text
        marker = "以下为外部实验文件，不包含在Git结果包中。"
        if "ANIE训练配置与模型身份存于" not in text:
            text = text.replace(marker, f"ANIE训练配置与模型身份存于 `{(root/'PROTOCOL_LOCK.json').as_posix()}` 及 `{(root/'training/anie_ns_seed7/ATTEMPT.json').as_posix()}`。" + marker)
    if relative == "docs/TRAINING_PROTOCOL.md" and root is not None:
        text = text.replace("配置来源为", "其余方法配置来源为") if "其余方法配置来源为" not in text else text
        if "ANIE坐标和训练配置来源为" not in text:
            marker = "## Independent N1000"
            sentence = f"ANIE坐标和训练配置来源为 `{(root/'PROTOCOL_LOCK.json').as_posix()}`，具体源码适配见 [ANIE实现说明](ANIE_ADAPTATION.md)。\n\n"
            if marker in text:
                text = text.replace(marker, sentence + marker)
            else:
                first_history = text.find("\n## ", text.find("\n## ")+1)
                if first_history < 0:
                    text += "\n" + sentence
                else:
                    text = text[:first_history] + "\n" + sentence + text[first_history:]
    return text


def document_contents(root, inputs):
    """Compute only current-protocol record changes from actual completed results."""
    epoch = inputs["status"].get("epoch", inputs["status"].get("completed_epochs", 0))
    best = inputs["status"].get("best_epoch")
    seconds = inputs["training_process_seconds"]
    path = inputs["checkpoint"].as_posix() if inputs["checkpoint"] is not None else "无可评价权重"
    digest = inputs["checkpoint_hash"].lower() if inputs["checkpoint_hash"] is not None else "无"
    records = {key: inputs["reports"][key]["methods"]["anie_ns"] for key in COHORTS}
    changed = {}
    for relative in DOCS:
        before = (PROJECT / relative).read_text(encoding="utf-8")
        after = before.replace(OLD_PACKAGE_NAME, PACKAGE_NAME)
        for old, new in (
            ("该人口的四种方法结果此前已被查看", "两个评价人口的结果已被查看"),
            ("确认集四方法成绩此前已被查看", "两个人口的成绩已被查看"),
            ("四方法成绩此前已被查看", "两个评价人口的成绩已被查看"),
            ("其中确认集的部分方法结果此前已被查看", "两个评价人口的结果均已被查看"),
            ("确认集的部分方法结果此前已被查看", "两个评价人口的结果均已被查看"),
            ("some method results on it had already been examined", "results on both evaluation populations had already been examined"),
        ):
            after = after.replace(old, new)
        after = re.sub(r"w1_single76_n300_(confirmation|old_test)_(\w+)_20261005\.png",
                       r"w1_aligned_anie_single76_n300_\1_\2_20261006.png", after)
        after = after.replace(OLD_CHECKPOINT, path).replace(OLD_CHECKPOINT_SHA, digest)
        for old, new in (("actual1028", f"actual{epoch}"), ("第1028轮", f"第{epoch}轮"),
                         ("至1028轮", f"至{epoch}轮"), ("在1028轮", f"在{epoch}轮"), ("训练1028轮", f"训练{epoch}轮"),
                         ("10280次更新", f"{10*epoch}次更新"), ("第28轮", f"第{best}轮"),
                         ("epoch28", f"epoch{best}")):
            after = after.replace(old, new)
        if relative == "EXPERIMENTS.md":
            after = re.sub(r"^\| ANIE \| 20810\.388 \| 5\.7807 \|$",
                           f"| ANIE | {seconds:.3f} | {seconds/3600:.4f} |", after, flags=re.M)
            matches = list(re.finditer(r"^\| 64 \| ANIE \| 300 \| 7 \|.*$", after, re.M))
            if len(matches) != 2:
                raise ValueError("current experiment record must contain exactly two ANIE W1 result rows")
            for match, cohort in reversed(list(zip(matches, COHORTS))):
                record = records[cohort]
                values = [metric(record, key) for key in ("future_mean_relative_l2", "t8_relative_l2", "future_mse")]
                number = lambda value: "无总体值" if value is None else f"{value:.6f}"
                row = f"| 64 | ANIE | 300 | 7 | {' | '.join(map(number, values))} | {record['summary']['failed_count']}/180 |"
                after = after[:match.start()] + row + after[match.end():]
            sentence = "ANIE按space-major/time-minor顺序组织状态与时空坐标，同一token的场特征和(x,y,t)严格对应；训练与评价使用相同排列。"
            if sentence not in after:
                after = after.replace("ANIE使用全76点的MSE。", "ANIE使用全76点的MSE。" + sentence)
            after = refresh_curve_tables(after, inputs["reports"])
            manifests = [root / "original_style_figures/W1" / cohort / "figure_manifest.json" for cohort in COHORTS]
            if all(path.is_file() for path in manifests):
                maximum = max(read(path)["details"]["color_scales"]["selected_residual_maximum_absolute_value"] for path in manifests)
                after = after.replace("最大残差21.354568", f"最大残差{maximum:.6f}")
        if relative == "docs/PERFORMANCE.md":
            after = re.sub(r"^\| ANIE，训练\d+轮、MSE-best epoch\d+ \| 20810\.388 \| 346\.84 \|$",
                           f"| ANIE，训练{epoch}轮、MSE-best epoch{best} | {seconds:.3f} | {seconds/60:.2f} |", after, flags=re.M)
            reused_seconds = inputs.get("unchanged_training_seconds")
            total = (reused_seconds if reused_seconds is not None else 53591.249 - 20810.388) + seconds
            after = after.replace("53591.249秒（14.8865小时）", f"{total:.3f}秒（{total/3600:.4f}小时）")
            after = after.replace("完整范围记录在后者目录的 `FINAL_COST_CLOSURE.json`；",
                f"ANIE训练成本来自 `{(root/'JOB_EVENTS.jsonl').as_posix()}`（SHA-256 `{inputs['events_hash'].lower()}`）。其他方法成本范围记录在原目录的 `FINAL_COST_CLOSURE.json`；")
        if relative == "docs/EXTERNAL_ADAPTATIONS.md":
            after = re.sub(r"^\| ANIE \|.*$", "| ANIE | 使用官方commit2071ad7的NS架构及 `aligned_ns_single_v1`；训练c=0/y_init路径；状态与(x,y,t)统一space-major/time-minor token排列；functional dropout绑定training，评价关闭dropout。共同76查询与micro6/effective32为任务/显存适配。仅参加W1。 |", after, flags=re.M)
            after = after.replace("四模型直接复用v5；新增U-NO/U-Net由公开仓库固定提交构造，外部磁盘源码未编辑。",
                "各方法由固定公开源码构造，外部磁盘源码保持只读；ANIE的坐标排列及逐项修改见 [ANIE实现说明](ANIE_ADAPTATION.md)。")
        if relative == "docs/W1_W2_PROTOCOL.md":
            after = after.replace("GIFT、FNO2D、FNO3D与ANIE的训练配置位于", "GIFT、FNO2D、FNO3D的训练配置位于")
            if (root/'PROTOCOL_LOCK.json').as_posix() not in after:
                after = after.replace("真实目录名称用于定位来源；", f"ANIE的训练及坐标配置位于`{(root/'PROTOCOL_LOCK.json').as_posix()}`。真实目录名称用于定位来源；")
            if "状态与时空坐标统一space-major/time-minor排列" not in after:
                after = after.replace("ANIE 用物理 MSE，不改为 relative L2。", "ANIE 用物理 MSE，不改为 relative L2。状态与时空坐标统一space-major/time-minor排列，同一token严格对应同一空间位置和查询时间，训练及评价一致。")
        if relative == "docs/RESULTS.md":
            anchor = "W1/W2权重、完整raw HDF及优化器/随机状态保存在声明的外部实验目录"
            if "ANIE的训练与两个W1人口的数值审计绑定于" not in after:
                after = after.replace(anchor, f"ANIE的训练与两个W1人口的数值审计绑定于 `{root.as_posix()}`；其余模型和W2的来源见各自结果清单。\n\n" + anchor)
        if relative == "docs/VALIDATION.md":
            if "六个既有模型及W2的完整预测重算入口" not in after:
                after = after.replace("完整预测重算入口是 `scripts.audit_w1_w2_single`", "六个既有模型及W2的完整预测重算入口是 `scripts.audit_w1_w2_single`")
            anchor = "结果证据见 [W1来源清单]"
            if "ANIE的时空token逐项对应与训练配置由" not in after:
                after = after.replace(anchor, f"ANIE的时空token逐项对应与训练配置由 [ANIE实现说明](ANIE_ADAPTATION.md) 声明，其完整训练/数值审计来自 `{root.as_posix()}` 的 `FINAL_TRAINING_AUDIT.json`、`FINAL_RESULT_AUDIT.json`。\n\n" + anchor)
        if relative == "docs/EXPERIMENT_EXECUTION.md":
            anchor = "独立评价接口为 `experiments.additional.w1_extended_evaluate`"
            if "ANIE_ADAPTATION.md" not in after:
                after = after.replace(anchor, f"ANIE配置、模型选择和W1数值绑定于 `{root.as_posix()}`；其时空token排列见 [ANIE实现说明](ANIE_ADAPTATION.md)。其他方法与W2使用各自声明目录。\n\n" + anchor)
        if relative == "docs/SETUP.md":
            if "ANIE使用" not in after or root.as_posix() not in after:
                after = after.replace("模型文件与SHA见", f"ANIE使用 `{root.as_posix()}` 中声明的坐标配置和训练权重。模型文件与SHA见")
        after = reconcile_record_wording(relative, after, root)
        if after != before:
            if not inputs["failed"] and inputs["status"].get("terminal_reason") != "native_early_stopping":
                after = after.replace(f"实际在第{epoch}轮早停", f"实际完成{epoch}轮上限")
                after = after.replace(f"训练在第{epoch}轮早停", f"训练完成{epoch}轮上限")
                after = after.replace(f"训练在第{epoch}轮依原早停规则终止", f"训练完成{epoch}轮上限")
                after = after.replace(f"依patience1000规则在{epoch}轮终止", f"达到{epoch}轮上限终止")
                after = after.replace(f"按原早停规则训练至{epoch}轮", f"按预定上限训练至{epoch}轮")
            changed[relative] = (before, after)
    if inputs["failed"]:
        failure_description = "ANIE训练发生声明的科学数值或资源失败；保留唯一seed7槽位，两个W1人口均按180/180失败记录，无总体精度值，不更换种子或参数。"
        for relative, (before, after) in list(changed.items()):
            after = after.replace(f"第{best}轮", "无可评价权重").replace(f"epoch{best}", "无可评价权重")
            after = after.replace("七种方法均有180/180条有限轨迹", "其余六种方法有180/180条有限轨迹，ANIE记录180/180失败")
            after = after.replace("All seven methods retain 180/180 finite trajectories.", "Six methods retain 180/180 finite trajectories; ANIE retains its failed 180-trajectory slot.")
            after = after.replace("所有方法均0/180非有限预测", "其余六方法均0/180非有限预测，ANIE记录180/180失败")
            after = after.replace("W1七方法、W2六方法在两评价人口相应任务中均为0/180失败。",
                                  "W1中ANIE记录180/180失败，其余六方法均0/180失败；W2六方法均0/180失败。")
            if relative == "EXPERIMENTS.md":
                after = re.sub(r"^\| ANIE \| 300 \| 7 \|.*$",
                    "| ANIE | 300 | 7 | 上限5000；effective32/micro6 | NS Adam lr1e-4/wd1e-4、延迟CosineT101；科学失败，无可评价模型 | 全76点原单位MSE，含初态重建；保留唯一失败槽位 |", after, flags=re.M)
                after = re.sub(r"ANIE使用无可评价权重验证MSE最佳权重，训练在第\d+轮早停。", failure_description, after)
            if relative == "docs/RESULTS.md":
                after = re.sub(r"ANIE使用无可评价权重验证MSE最优模型，训练在第\d+轮依原早停规则终止；ANIE仅参加W1。",
                               failure_description + "ANIE仅参加W1。", after)
            if relative == "docs/CHECKPOINTS.md":
                after = re.sub(r"^\| ANIE \|.*$", "| ANIE | 无可评价权重 | 无 | seed7科学失败；仅W1 |", after, flags=re.M)
                after = re.sub(r"ANIE依原NS validation-MSE-best规则选无可评价权重，按原早停规则训练至\d+轮。", failure_description, after)
            if relative == "docs/TRAINING_PROTOCOL.md":
                after = re.sub(r"^\| ANIE \|.*$",
                    "| ANIE | seed7，effective batch32/micro6，Adam lr1e-4/wd1e-4，延迟CosineT101，上限5000轮；科学失败，无可评价模型 | 官方NS训练c=0/y_init路径，原单位全76点MSE，含初态重建；仅W1。 |", after, flags=re.M)
            if relative == "docs/PERFORMANCE.md":
                after = re.sub(r"^\| ANIE，训练\d+轮、MSE-best 无可评价权重 \|.*$",
                               f"| ANIE，科学失败 | {seconds:.3f} | {seconds/60:.2f} |", after, flags=re.M)
            if relative == "docs/W1_W2_PROTOCOL.md":
                after = re.sub(r"ANIE上限5000轮、每轮10次更新；实际在第\d+轮早停，共\d+次更新，使用无可评价权重验证MSE最佳权重。",
                    f"ANIE上限5000轮，科学失败前实际完成{epoch}轮、记录{inputs['status'].get('updates', 0)}次更新，无可评价权重，保留唯一seed7槽位。", after)
            if relative in {"EXPERIMENTS.md", "docs/RESULTS.md", "docs/TRAINING_PROTOCOL.md", "docs/CHECKPOINTS.md",
                            "docs/W1_W2_PROTOCOL.md"}:
                after += "\n" + failure_description + "\n"
            changed[relative] = (before, after)
    adaptation = f"""# ANIE：时空坐标与状态token对应

ANIE使用作者公开仓库 [emazap7/ANIE](https://github.com/emazap7/ANIE)，固定提交 `2071ad709b260a6c305450affcf7168fd825f692`，source profile为 `aligned_ns_single_v1`、condition为 `native`。使用者自行下载官方源码并按照 [外部适配说明](EXTERNAL_ADAPTATIONS.md) 在仓库外完成适配；GIFT仓库只保留文字说明、来源引用和通用外部模块加载接口。

状态张量按空间优先、时间次优顺序展平。空间索引s、时间索引j对应token `k=s*T+j`；token的状态特征、空间坐标和时间坐标均使用该索引。求解器内部各次更新保持同一排列，解码前按其逆排列还原。训练、validation和W1推理采用同一配置。

| 外部位置或项目接口 | 调整及必要性 |
| --- | --- |
| 使用者的外部ANIE运行接口 | 明确 `aligned_ns_single_v1` 配置；采用native条件路径，训练控制来源沿用官方NS实现。该接口文件保存在GIFT仓库目录外。 |
| 外部适配记录及 [官方solver](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/IE_source/Attentional_IE_solver.py) | 按文字说明将空间和时间坐标与状态统一排列；保留原架构、注意力、积分迭代和输出解码。完整源码补丁不在GIFT仓库中分发。 |
| [官方attention实现](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/IE_source/Galerkin_transformer.py) | evaluation时关闭functional attention dropout，训练概率保持原值；使用者在自己的外部代码中完成修改。 |
| `training/w1_single_train.py` | 允许新对齐profile通过配置校验并按该profile构造原网络；既有控制器、全76点物理MSE、optimizer/scheduler、同次validation-MSE-best及原早停规则保持；不增加选参或种子。 |

已登记的functional attention dropout调整在训练时保留原概率，在evaluation时关闭，与模块training状态一致。未增加归一化、空间decoder修改、初态硬锚定、辅助重建或递归机制。

训练数据为共同N300名单和20条validation，t5–8每0.04秒共76点。输入仅为每条轨迹t5真场；NS训练和评价均使用c=0/y_init条件路径。seed7，effective batch32、micro6，Adam lr1e-4/wd1e-4，延迟CosineT101，eta_min1e-9，上限5000轮、patience1000。{"训练发生科学失败，无可评价权重；全部失败人口保留。" if inputs['failed'] else f"实际完成{epoch}轮、{10*epoch}次更新，选择第{best}轮；"}训练进程墙钟{seconds:.3f}秒。

权重：`{path}`，SHA-256 `{digest}`。完整训练和数值来源：`{root.as_posix()}`。W1两人口各180条单列，主指标排除已知初态，只取future75。ANIE不参加W2。该实验描述固定配方在本任务的表现，不作为原论文数据与全部设置的完整复现。
"""
    adaptation_path = PROJECT / "docs/ANIE_ADAPTATION.md"
    previous_adaptation = adaptation_path.read_text(encoding="utf-8") if adaptation_path.exists() else None
    if previous_adaptation != adaptation:
        changed["docs/ANIE_ADAPTATION.md"] = (previous_adaptation, adaptation)
    return changed


def prepare_document_plan(root):
    """Declare edit scope while training is pending without inventing any result."""
    path = root / "DOCUMENT_DELIVERY_PLAN.json"
    write_once(path, dict(status="prepared", current_documents=list(DOCS),
        implementation_record="docs/ANIE_ADAPTATION.md",
        package=PACKAGE_NAME, base_package=OLD_PACKAGE_NAME,
        prerequisite="completed seed7 STATUS and passed training/result audits; actual measurements only",
        plotting="scripts.plot_w1_original_style; unchanged structure, fixed IDs and signed residuals",
        untouched=["all historical packages", "six reused methods", "W2", "W3", "S1-S3"]))
    return dict(status="prepared", path=str(path))


def update_documents(root, inputs, package):
    """Snapshot current texts and activate only a verified completed publication."""
    verify_package(package, "W1")
    receipt_path = root / "DOCUMENT_ACTIVATION.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        for record in receipt["files"]:
            if sha256(PROJECT / record["path"]) != record["after_sha256"]:
                raise ValueError("activated current document was modified after receipt")
        return receipt["files"]
    changes = document_contents(root, inputs)
    snapshot = root / "document_snapshots_before_activation"
    records = []
    for relative, (before, after) in changes.items():
        path = PROJECT / relative
        if before is not None:
            text_once(snapshot / relative, before)
        if path.exists() and path.read_text(encoding="utf-8") == after:
            continue
        if (before is None and path.exists()) or (before is not None and path.read_text(encoding="utf-8") != before):
            raise ValueError(f"document changed since preparation: {path}")
        path.write_text(after, encoding="utf-8")
        records.append(dict(path=relative, before_sha256=sha256(snapshot / relative) if before is not None else None,
                            after_sha256=sha256(path)))
    write_once(root / "DOCUMENT_ACTIVATION.json", dict(status="complete", files=records,
        package_manifest_sha256=sha256(package / "published.json"),
        formal_record_style="final methods and measured results; no development chronology",
        figures="unchanged original manuscript renderer", W2_changed=False))
    return records


def build_document_audit(root, package):
    """Resolve current W1 references and verify their completed publication bytes.

    This checks evidence and Markdown pointers. It does not claim to inspect PNG
    appearance; actual visual inspection must be recorded separately by a reader.
    """
    activation = read(root / "DOCUMENT_ACTIVATION.json")
    integrity = verify_package(package, "W1")
    documents, references = [], []
    for record in activation["files"]:
        path = PROJECT / record["path"]
        if sha256(path) != record["after_sha256"]:
            raise ValueError(f"current document bytes differ after activation: {path}")
        documents.append(dict(path=record["path"], sha256=record["after_sha256"]))
    for relative in DOCS:
        path = PROJECT / relative
        text = path.read_text(encoding="utf-8")
        if OLD_PACKAGE_NAME in text:
            raise ValueError(f"current W1 record still points to the superseded package: {relative}")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not any(key in target for key in (PACKAGE_NAME, "w1_aligned_anie_single76_n300_", "ANIE_COORDINATE_ADAPTATION_20261006.md")):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve(strict=True)
            references.append(dict(document=relative, target=target, resolved_path=str(target_path),
                                   sha256=sha256(target_path) if target_path.is_file() else None))
    for cohort in COHORTS:
        manifest = read(package / "figures" / cohort / "figure_manifest.json")
        for record in manifest["figure_contract"]["field_previews"]:
            asset = PROJECT / record["path"]
            if asset.stat().st_size != record["bytes"] or sha256(asset) != record["sha256"].upper():
                raise ValueError("linked field preview differs from figure source manifest")
    experiment_record = (PROJECT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    cohort_tables = {}
    table_count = experiment_record.count("| 时间 | GIFT | GIFT-N50 | FNO2D | U-NO | U-Net | ANIE | FNO3D |")
    if table_count not in (1, 2):
        raise ValueError("the current W1 record must retain one or two declared cohort time tables")
    for cohort in COHORTS:
        numeric = read(root / "w1_evaluation" / cohort / "numeric_report.json")
        expected = measured_curve_table(numeric)
        expected_in_record = cohort == "confirmation" or table_count == 2
        if expected_in_record and expected not in experiment_record:
            raise ValueError(f"{cohort} seven-node experiment table differs from its actual measurements")
        csv_path = package / "figures" / cohort / "source_data.csv"
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 49:
            raise ValueError("each published cohort needs all seven nodes for all seven methods")
        seen = set()
        indices = plot.nearest_times(np.asarray(numeric["absolute_times"]), plot.REQUESTED_TIMES)
        expected_slots = {(key, int(index)) for key in METHODS for index in indices}
        for row in rows:
            key, index = row["raw_method_key"], int(row["observation_index"])
            identity = key, index
            if identity in seen or identity not in expected_slots:
                raise ValueError("duplicate or unknown cohort curve measurement")
            seen.add(identity)
            summary = numeric["methods"][key]["summary"]
            means = summary.get("mean_relative_l2_by_time")
            value = means[index] if means is not None and summary["failed_count"] == 0 else None
            actual = float(row["mean_relative_l2"]) if row["mean_relative_l2"] else None
            if (row["cohort"] != cohort or float(row["absolute_time"]) != numeric["absolute_times"][index]
                    or (value is None) != (actual is None)
                    or (value is not None and abs(value-actual) > 2e-12)):
                raise ValueError("published cohort curve source CSV differs from its actual measurements")
        if seen != expected_slots:
            raise ValueError("published cohort curve source CSV omits an actual method/node slot")
        cohort_tables[cohort] = dict(status="passed", report_sha256=sha256(root / "w1_evaluation" / cohort / "numeric_report.json"),
            actual_node_count=7, method_count=7, published_curve_source_sha256=sha256(csv_path),
            source_csv_rows_verified=49, table_in_current_record=expected_in_record,
            precision="six decimal places for the record; full precision for curve source CSV")
    for relative in DOCS:
        content = (PROJECT / relative).read_text(encoding="utf-8")
        if "原生token坐标" in content or re.search(r"(?m)^- v[56]：", content):
            raise ValueError(f"current record retains a superseded coordinate statement or development label: {relative}")
    result = dict(schema="gift.anie-coordinate-refresh.document-audit.v1", status="passed",
        documents=documents, latest_result_references=references, publication_integrity=integrity,
        package_manifest_sha256=sha256(package / "published.json"),
        scientific_content_checks=dict(cohort_time_tables=cohort_tables,
            coordinate_statement="same state and (x,y,t) token order", development_chronology=False),
        original_manuscript_renderer=True, W2_measurements_and_package_unchanged=True,
        visual_inspection="pending; this audit verifies bytes and references only")
    return result


def audit_documents(root, package):
    """Commit the document audit after verifying both content and file pointers."""
    result = build_document_audit(root, package)
    write_once(root / "FINAL_DOCUMENT_AUDIT.json", result)
    return result


def repair_current_documents(root, base):
    """Reconcile document omissions while preserving all scientific measurements.

    Existing document receipts are copied byte-for-byte into a new evidence
    directory before their current pointers are replaced. Publication and the
    scientific process journal stay immutable. CPU engineering wall time is
    appended separately, including the preservation and independent checks.
    """
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    command = ["python", "-m", "scripts.publish_anie_coordinate_refresh", "--root", str(root), "--repair-current-documents"]
    exit_code = 1
    try:
        inputs = completed_inputs(root, base)
        package = PROJECT / "results/formal" / PACKAGE_NAME
        verify_package(package, "W1")
        protected = {str(path): sha256(path) for path in (
            root / "PUBLICATION.json", root / "JOB_EVENTS.jsonl", root / "FINAL_TRAINING_AUDIT.json",
            root / "FINAL_RESULT_AUDIT.json", package / "published.json")}
        prior_activation = read(root / "DOCUMENT_ACTIVATION.json")
        for record in prior_activation["files"]:
            if sha256(PROJECT / record["path"]) != record["after_sha256"]:
                raise ValueError("a current document was modified outside its existing activation receipt")
        evidence = root / ("document_evidence_" + started.strftime("%Y%m%dT%H%M%S%fZ"))
        evidence.mkdir(exist_ok=False)
        for filename in ("DOCUMENT_ACTIVATION.json", "FINAL_DOCUMENT_AUDIT.json"):
            copy_once(root / filename, evidence / filename)
        changes = document_contents(root, inputs)
        changed = []
        for relative, (_, after) in changes.items():
            path = PROJECT / relative
            before = path.read_text(encoding="utf-8") if path.is_file() else None
            if before == after:
                continue
            if before is not None:
                copy_once(path, evidence / "documents" / relative)
            path.write_text(after, encoding="utf-8")
            changed.append(relative)
        root_report = root / "RESULTS.md"
        before_report = root_report.read_text(encoding="utf-8")
        after_report = before_report.replace(PENDING_VISUAL_STATEMENT, COMPLETED_VISUAL_STATEMENT)
        if before_report != after_report:
            copy_once(root_report, evidence / "RESULTS.md")
            root_report.write_text(after_report, encoding="utf-8")
        records = []
        for record in prior_activation["files"]:
            current = dict(record, after_sha256=sha256(PROJECT / record["path"]))
            if record["path"] in changed:
                current["before_sha256"] = sha256(evidence / "documents" / record["path"])
            records.append(current)
        activation = dict(prior_activation, files=records,
            prior_activation_evidence=str(evidence / "DOCUMENT_ACTIVATION.json"),
            prior_activation_sha256=sha256(evidence / "DOCUMENT_ACTIVATION.json"),
            updated_document_scope="actual measured tables, coordinate semantics and neutral source descriptions")
        atomic_json(root / "DOCUMENT_ACTIVATION.json", activation)
        result = build_document_audit(root, package)
        atomic_json(root / "FINAL_DOCUMENT_AUDIT.json", result)
        for path, digest in protected.items():
            if sha256(Path(path)) != digest:
                raise ValueError("scientific measurement or publication bytes changed during documentation work")
        receipt = dict(schema="gift.anie-coordinate-refresh.document-reconciliation.v1", status="complete",
            changed_documents=changed, evidence_directory=str(evidence),
            preserved_prior_receipts={name: sha256(evidence / name) for name in ("DOCUMENT_ACTIVATION.json", "FINAL_DOCUMENT_AUDIT.json")},
            current_document_activation_sha256=sha256(root / "DOCUMENT_ACTIVATION.json"),
            current_document_audit_sha256=sha256(root / "FINAL_DOCUMENT_AUDIT.json"),
            root_result_record_sha256=sha256(root_report), scientific_and_publication_files_unchanged=protected,
            no_training_or_model_inference=True)
        write_once(evidence / "RECONCILIATION.json", receipt)
        exit_code = 0
        return receipt
    finally:
        finished = datetime.now(timezone.utc)
        event = dict(label="reconcile_document_records", command=command, exit_code=exit_code,
            started_utc=started.isoformat(), finished_utc=finished.isoformat(),
            process_seconds=time.perf_counter()-start_clock,
            scope="CPU-only document reconciliation, evidence preservation and reference/content audit; no scientific ledger mutation")
        with (root / "CPU_EXECUTION_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--prepare-doc-plan", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--update-documents", action="store_true")
    parser.add_argument("--repair-current-documents", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    if args.prepare_doc_plan:
        print(json.dumps(prepare_document_plan(root), ensure_ascii=False))
    if args.repair_current_documents:
        if args.publish or args.update_documents or args.prepare_doc_plan:
            parser.error("--repair-current-documents is a separate document-only operation")
        print(json.dumps(repair_current_documents(root, args.base.resolve(strict=True)), ensure_ascii=False, indent=2))
        return
    default_delivery = not (args.prepare_doc_plan or args.publish or args.update_documents)
    if args.publish or args.update_documents or default_delivery:
        inputs, package = publish(root, args.base.resolve(strict=True))
        result = dict(status="published", package=str(package))
        if args.update_documents or default_delivery:
            result["documents"] = update_documents(root, inputs, package)
            result["document_audit"] = audit_documents(root, package)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
