"""Publication must preserve all six reused measurements and actual outcomes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import publish_anie_coordinate_refresh as publication


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    root, base = tmp_path / "refresh", tmp_path / "base"
    checkpoint = root / "training/anie_ns_seed7/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture-weight-identity")
    status = dict(status="complete", method="anie_ns", seed=7, epoch=1200,
                  updates=12000, best_epoch=200, selected_checkpoint=str(checkpoint))
    dump(checkpoint.parent / "STATUS.json", status)
    for name in ("FINAL_RESULT_AUDIT.json", "FINAL_TRAINING_AUDIT.json"):
        dump(root / name, dict(status="passed"))
    record = dict(status="complete", seed=0, checkpoint_sha256="a" * 64,
                  raw_prediction_sha256="b" * 64, summary=dict(population=180, failed_count=0, metrics={}))
    for cohort in publication.COHORTS:
        original = dict(status="numerical_complete", population=180, cohort=cohort,
                        trajectory_ids=list(range(180)), absolute_times=list(range(76)), query_times=list(range(76)),
                        methods={method: deepcopy(record) for method in publication.METHODS})
        original["methods"]["anie_ns"]["seed"] = 7
        new = deepcopy(original)
        new["methods"]["anie_ns"].update(checkpoint=str(checkpoint), checkpoint_sha256=publication.sha256(checkpoint))
        dump(base / "w1_evaluation" / cohort / "numeric_report.json", original)
        dump(root / "w1_evaluation" / cohort / "numeric_report.json", new)
    event = dict(label="train_anie_ns_seed7", started_utc="2026-10-06T00:00:00Z", process_seconds=100., exit_code=0)
    (root / "JOB_EVENTS.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    return root, base, checkpoint


def test_pending_training_cannot_activate_or_publish(evidence):
    root, base, checkpoint = evidence
    status = publication.read(checkpoint.parent / "STATUS.json")
    status["status"] = "running"
    dump(checkpoint.parent / "STATUS.json", status)
    with pytest.raises(FileNotFoundError):
        publication.completed_inputs(root, base)
    assert not (root / "PUBLICATION.json").exists()
    assert not (root / "DOCUMENT_ACTIVATION.json").exists()


def test_completed_evidence_keeps_actual_epochs_and_process_cost(evidence):
    root, base, _ = evidence
    result = publication.completed_inputs(root, base)
    assert result["status"]["epoch"] == 1200
    assert result["status"]["best_epoch"] == 200
    assert result["training_process_seconds"] == 100.


def test_changed_reused_method_rejected(evidence):
    root, base, _ = evidence
    path = root / "w1_evaluation/confirmation/numeric_report.json"
    report = publication.read(path)
    report["methods"]["fno2d"]["summary"]["failed_count"] = 1
    dump(path, report)
    with pytest.raises(ValueError, match="reused fno2d measurement changed"):
        publication.completed_inputs(root, base)


def test_duplicate_process_cost_rejected(evidence):
    root, base, _ = evidence
    path = root / "JOB_EVENTS.jsonl"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate process"):
        publication.completed_inputs(root, base)


def test_scientific_failure_retains_all180_without_checkpoint(evidence):
    root, base, checkpoint = evidence
    failure = dict(status="failed", method="anie_ns", seed=7, failure_kind="numerical", reason="nonfinite gradient",
                   completed_epochs=17, updates=171)
    dump(checkpoint.parent / "STATUS.json", failure)
    dump(checkpoint.parent / "FAILURE.json", failure)
    for cohort in publication.COHORTS:
        path = root / "w1_evaluation" / cohort / "numeric_report.json"
        report = publication.read(path)
        report["methods"]["anie_ns"] = dict(status="training_failed", seed=7,
            summary=dict(population=180, failed_count=180, primary_metric=None, metrics={}))
        dump(path, report)
    result = publication.completed_inputs(root, base)
    assert result["failed"] and result["checkpoint"] is None
    assert all(report["methods"]["anie_ns"]["summary"]["failed_count"] == 180 for report in result["reports"].values())


def test_document_plan_never_reads_results_or_edits_project(tmp_path, monkeypatch):
    empty_project = tmp_path / "project"
    root = tmp_path / "pending"
    empty_project.mkdir()
    root.mkdir()
    monkeypatch.setattr(publication, "PROJECT", empty_project)
    result = publication.prepare_document_plan(root)
    assert result["status"] == "prepared"
    assert list(empty_project.iterdir()) == []
    assert publication.read(root / "DOCUMENT_DELIVERY_PLAN.json")["prerequisite"].startswith("completed seed7")


def test_document_numbers_replaced_from_new_measurements_without_changing_w2(tmp_path, monkeypatch):
    project, root = tmp_path / "project", tmp_path / "refresh"
    project.mkdir()
    experiment = """# Experiment
| ANIE | 20810.388 | 5.7807 |
| 64 | ANIE | 300 | 7 | .9 | .9 | 15 | 0/180 |
| 64 | ANIE | 300 | 7 | .9 | .9 | 15 | 0/180 |
ANIE使用全76点的MSE。
W1_native_single76_n300_original_style_20261005
W2_native_single76_n300_original_style_20261005
"""
    (project / "EXPERIMENTS.md").write_text(experiment, encoding="utf-8")
    monkeypatch.setattr(publication, "PROJECT", project)
    monkeypatch.setattr(publication, "DOCS", ("EXPERIMENTS.md",))
    record = dict(summary=dict(failed_count=0, metrics={name: dict(mean=value) for name,value in
        (("future_mean_relative_l2", .123), ("t8_relative_l2", .234), ("future_mse", 3.456))}))
    inputs = dict(status=dict(epoch=1200, best_epoch=200), training_process_seconds=1234., checkpoint=root / "best.pt",
                  checkpoint_hash="c" * 64, reports={cohort: dict(methods=dict(anie_ns=record)) for cohort in publication.COHORTS},
                  events_hash="d" * 64, failed=False)
    contents = publication.document_contents(root, inputs)
    content = contents["EXPERIMENTS.md"][1]
    assert "| ANIE | 1234.000 | 0.3428 |" in content
    assert content.count("0.123000 | 0.234000 | 3.456000") == 2
    assert "W2_native_single76_n300_original_style_20261005" in content
    assert publication.PACKAGE_NAME in content
    assert "space-major/time-minor" in content
    implementation = contents["docs/ANIE_ADAPTATION.md"][1]
    assert "允许新对齐profile通过配置校验并按该profile构造原网络" in implementation
    assert "训练代码未改" not in implementation


def curve_report(offset):
    """Different cohort means make an accidental cohort interchange detectable."""
    return dict(absolute_times=[5+i*.04 for i in range(76)], methods={key: dict(summary=dict(
        failed_count=0, mean_relative_l2_by_time=[offset + method*.1 + i*.003 for i in range(76)]))
        for method,key in enumerate(("GIFT", "GIFT-N50", "fno2d", "uno", "unet", "anie_ns", "fno3d"))})


def test_both_cohort_tables_use_actual_nodes_and_are_idempotent():
    reports = dict(confirmation=curve_report(0), old_test=curve_report(.01))
    expected_confirmation = publication.measured_curve_table(reports["confirmation"])
    old_lines = expected_confirmation.splitlines()
    cells = old_lines[2].split("|")
    cells[7] = " 9.999999 "
    old_lines[2] = "|".join(cells)
    before = "\n".join(old_lines) + "\n\n#### 探索性测试集\n\n" + publication.measured_curve_table(reports["old_test"]) + "\n\nW2 unchanged text.\n"
    after = publication.refresh_curve_tables(before, reports)
    assert expected_confirmation in after
    assert publication.measured_curve_table(reports["old_test"]) in after
    assert after.count("| 时间 | GIFT | GIFT-N50") == 2
    assert "9.999999" not in after
    assert "W2 unchanged text." in after
    assert publication.refresh_curve_tables(after, reports) == after


def test_one_existing_curve_table_does_not_add_exploration_record_structure():
    reports = dict(confirmation=curve_report(0), old_test=curve_report(.01))
    before = publication.measured_curve_table(reports["confirmation"]) + "\n\n#### 探索性测试集\n\nExisting summary and figure links.\n"
    after = publication.refresh_curve_tables(before, reports)
    assert after == before
    assert after.count("| 时间 | GIFT | GIFT-N50") == 1


def test_failed_curve_slot_has_no_fabricated_numeric_means():
    report = curve_report(0)
    report["methods"]["anie_ns"]["summary"] = dict(failed_count=180, metrics={})
    table = publication.measured_curve_table(report)
    assert table.count("无总体值") == 7


def test_coordinate_and_exposure_wording_has_no_development_labels():
    text = "原生token坐标与76个查询点；既有四方法结果在扩展前已知，故不是本次新盲确认\n- v6：source\n"
    result = publication.reconcile_record_wording("docs/RESULTS.md", text)
    assert "原生token坐标" not in result
    assert "四方法" not in result
    assert "- v6：" not in result
    assert "非盲测的事后测量" in result


def test_document_consumers_only_reference_existing_canonical_documents():
    assert all((publication.PROJECT / path).is_file() for path in publication.DOCS)
    assert "docs/W1_W2_PROTOCOL.md" in publication.DOCS
    assert len(set(publication.DOCS)) == len(publication.DOCS)
    assert publication.PACKAGE_NAME == "W1_flow_prediction_20261007"
