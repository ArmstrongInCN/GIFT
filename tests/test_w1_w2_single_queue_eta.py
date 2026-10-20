"""CPU orchestration tests: new slots only, immutable reuse, and conditional JSON ETA."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_w1_w2_single as queue
from scripts import estimate_w1_w2_single_eta as eta


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_method_inventory_new_only_fixed_recipes_and_anie_free_w2(tmp_path):
    plan = queue.default_plan(tmp_path)
    assert len(queue.W1_METHODS) == 7 and len(queue.W2_METHODS) == 6
    assert "anie_ns" in queue.W1_METHODS and "anie_ns" not in queue.W2_METHODS
    assert tuple(plan["new_training_methods"]) == queue.NEW_METHODS == ("unet", "uno", "GIFT-N50")
    assert set(plan["reused_training_methods"]) == {"GIFT", "fno2d", "fno3d", "anie_ns"}
    assert plan["development_candidates"] == 0 and not plan["hyperparameter_search"]
    assert not plan["repeated_seed_runs"] and not plan["anie_in_w2"]
    times = (np.arange(76)*.04).tolist()
    for method in queue.NEW_METHODS:
        job = queue.job_configuration(plan, method, times)
        assert job["config"]["selection"] in ("terminal", "terminal_epoch")
        if method == "GIFT-N50":
            assert job["expected_training_count"] == 50 and job["seed"] == 20260820
            assert job["config"]["generator_phases"] == [250, 250]
        else:
            assert job["config"]["max_epochs"] == 500 and job["model_spec"]["history"] == 1
    for reused in queue.WEIGHTS:
        with pytest.raises(ValueError):
            queue.job_configuration(plan, reused, times)


def prepare_fixture(tmp_path, monkeypatch, *, bad_prefix=False):
    """Mock data I/O; real locking and hashing still preserve all nested recipes."""
    from training import w1_v2_data as data
    from experiments.additional import w1_single_evaluate as native
    root = tmp_path / "new"
    root.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    files = {label: base / (label+".pt") for label in queue.WEIGHTS}
    for label, path in files.items():
        path.write_bytes(("synthetic immutable "+label).encode())
    monkeypatch.setattr(queue, "WEIGHTS", files)
    common = dict(training=dict(ids=np.random.default_rng(2026100304).permutation(1000)[:300].tolist(), observations_sha256="a"*64),
                  validation=dict(ids=list(range(1000, 1020)), observations_sha256="b"*64),
                  query_times=(np.arange(76)*.04).tolist(), training_count=300)
    reduced = copy.deepcopy(common)
    reduced.update(training_count=50)
    reduced["training"] = dict(ids=common["training"]["ids"][:50], observations_sha256="c"*64)
    if bad_prefix:
        reduced["training"]["ids"] = list(reversed(reduced["training"]["ids"]))
    lock_path = base / "PROTOCOL_LOCK.json"
    write_json(lock_path, dict(data_binding=common))
    plan = queue.default_plan(root)
    plan.update(base_lock=str(lock_path), common_window=str(base / "common.h5"))
    monkeypatch.setattr(data, "WindowBank", lambda path: SimpleNamespace(binding=copy.deepcopy(
        common if str(path) == plan["common_window"] else reduced)))
    prepared = []
    monkeypatch.setattr(data, "prepare_window", lambda *args, **kwargs: prepared.append(kwargs))
    monkeypatch.setattr(native, "checkpoint_metadata", lambda path, lock: dict(model_spec=dict(method=Path(path).stem)))
    return root, plan, files, prepared


def test_prepare_nesting_lock_complete_recipes_and_immutable_reuse(tmp_path, monkeypatch):
    root, plan, files, prepared = prepare_fixture(tmp_path, monkeypatch)
    before = {label: path.read_bytes() for label, path in files.items()}
    jobs, lock = queue.prepare_run(root, plan)
    assert set(jobs) == set(queue.NEW_METHODS)
    assert lock["n50_data_binding"]["training"]["ids"] == lock["common_data_binding"]["training"]["ids"][:50]
    assert lock["n50_data_binding"]["validation"] == lock["common_data_binding"]["validation"]
    assert all(item["new_training_seconds"] == 0 for item in lock["reused_models"].values())
    assert {label: path.read_bytes() for label, path in files.items()} == before
    assert prepared == [dict(training_count=50, time_stride=2, subset_seed=2026100304)]
    assert queue.prepare_run(root, plan)[1] == lock
    assert set(lock["training_configurations"]) == set(queue.NEW_METHODS)
    for label, job in jobs.items():
        saved = json.loads((root / "configs" / (Path(job["output"]).name+".json")).read_text())
        assert saved == job and saved["config"] == lock["training_configurations"][label]["config"]


def test_prepare_rejects_non_nested_n50_even_with_correct_counts(tmp_path, monkeypatch):
    root, plan, _, _ = prepare_fixture(tmp_path, monkeypatch, bad_prefix=True)
    with pytest.raises(ValueError, match="nested first50"):
        queue.prepare_run(root, plan)


@pytest.mark.parametrize("kind", ["numerical", "resource"])
def test_failed_unique_slot_is_retained_without_child_rerun(tmp_path, monkeypatch, kind):
    output = tmp_path / "training/uno_seed0"
    write_json(output / "FAILURE.json", dict(method="uno", seed=0, stage="formal", reason="synthetic", failure_kind=kind))
    monkeypatch.setattr(queue, "call_module", lambda *args: pytest.fail("failed slot must not start a trainer"))
    value = queue.execute_job(tmp_path, dict(method="uno", seed=0, output=str(output)))
    assert value["status"] == "failed" and Path(value["failure"]).is_file()


def test_infrastructure_failure_requires_specific_repair_and_completed_model_never_retrains(tmp_path, monkeypatch):
    output = tmp_path / "training/unet_seed0"
    write_json(output / "FAILURE.json", dict(failure_kind="infrastructure"))
    monkeypatch.setattr(queue, "call_module", lambda *args: pytest.fail("unrepaired infrastructure must not start"))
    with pytest.raises(RuntimeError, match="repair and preserve"):
        queue.execute_job(tmp_path, dict(method="unet", seed=0, output=str(output)))
    complete = tmp_path / "training/uno_seed0"
    complete.mkdir(parents=True)
    weight = complete / "terminal_weights.pt"
    weight.write_bytes(b"completed fixture")
    write_json(complete / "STATUS.json", dict(status="complete", selected_checkpoint=str(weight)))
    assert queue.execute_job(tmp_path, dict(method="uno", seed=0, output=str(complete)))["status"] == "complete"


def test_execute_trains_exactly_new_three_and_keeps_all_w1_failed_and_w2_slots(tmp_path, monkeypatch):
    plan = queue.default_plan(tmp_path)
    received, calls = [], []
    def execute_job(root, job):
        label = job["method"]
        received.append(label)
        return (dict(method=label, status="failed", failure=str(tmp_path / "failed.json")) if label == "GIFT-N50" else
                dict(method=label, status="complete", checkpoint=str(tmp_path / (label+".pt"))))
    monkeypatch.setattr(queue, "execute_job", execute_job)
    monkeypatch.setattr(queue, "call_module", lambda root, module, arguments, label: calls.append((module, list(map(str, arguments)), label)) or 0)
    jobs = {label: dict(method=label) for label in queue.NEW_METHODS}
    lock = dict(reused_models={label: dict(sha256="a"*64) for label in queue.WEIGHTS})
    queue.execute(tmp_path, plan, jobs, lock)
    assert tuple(received) == queue.NEW_METHODS and not set(received) & set(queue.WEIGHTS)
    w1 = [args for module, args, label in calls if module.endswith("w1_extended_evaluate")]
    w2 = [args for module, args, label in calls if module.endswith("w2_single_evaluate")]
    assert len(w1) == len(w2) == 2
    assert all(sum(value in ("--checkpoint", "--failed-run") for value in args) == 7 for args in w1)
    assert all(sum(value in ("--checkpoint", "--failed-run") for value in args) == 6 for args in w2)
    assert all(not any(value.startswith("anie_ns=") for value in args) for args in w2)
    assert all(any(value.startswith("GIFT-N50=") for value in args) for args in w1+w2)
    assert json.loads((tmp_path / "ORCHESTRATOR.json").read_text())["status"] == "evaluation_complete_documents_pending"


def eta_fixture(root):
    forecast = dict(new_training_hours={"unet": [.4, 1.], "uno": [3., 4.], "GIFT-N50": [.6, 1.2]},
                    w2_truth_hours=[.5, 2.], evaluation_plot_audit_document_hours=[1., 3.])
    write_json(root / "COST_FORECAST.json", forecast)
    write_json(root / "ORCHESTRATOR.json", dict(status="running", active_job="train_unet_seed0"))
    return forecast


def test_mature_ten_epoch_rate_excludes_bad_samples_and_uses_actual_total():
    rows = [dict(epoch_total_seconds=2.) for _ in range(10)] + [dict(seconds=float("nan")), dict(seconds=-1.)]
    rate, count = eta.observed_rate(rows)
    assert count == 10 and rate == [2., 2.4]
    assert eta.observed_rate(rows[:9]) == ([2., 2.6], 9)


def test_eta_mature_remainder_infrastructure_block_and_append_only_history(tmp_path):
    forecast = eta_fixture(tmp_path)
    rows = [dict(epoch=epoch, epoch_total_seconds=2.) for epoch in range(1, 11)]
    write_json(tmp_path / "training/unet_seed0/STATUS.json", dict(status="running", epoch=10))
    write_json(tmp_path / "training/unet_seed0/HISTORY.json", dict(training=rows))
    measured = eta.neural_remaining(tmp_path, "unet", forecast)
    assert measured["samples"] == 10 and measured["remaining_seconds"] == [980., 1176.]
    first = eta.estimate(tmp_path)
    assert first["estimated_finish_beijing"] is not None and first["process_liveness_verified_by_script"] is False
    original_history = (tmp_path / "ETA_HISTORY.jsonl").read_bytes()
    write_json(tmp_path / "training/unet_seed0/FAILURE.json", dict(failure_kind="infrastructure", reason="synthetic lost disk"))
    blocked = eta.estimate(tmp_path)
    assert blocked["estimated_finish_beijing"] is None and blocked["previous_finish_beijing"] == first["estimated_finish_beijing"]
    assert (tmp_path / "ETA_HISTORY.jsonl").read_bytes().startswith(original_history)
    assert eta.neural_remaining(tmp_path, "uno", forecast)["basis"] == "historical_different_input_reservation"


def test_eta_is_json_only_and_resource_failure_does_not_promise_retraining(tmp_path):
    tree = ast.parse(Path(eta.__file__).read_text(encoding="utf-8"))
    imports = [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
    imports += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.split(".")[0] in {"torch", "h5py", "subprocess", "psutil"} for name in imports if name)
    forecast = eta_fixture(tmp_path)
    write_json(tmp_path / "training/uno_seed0/FAILURE.json", dict(failure_kind="resource", reason="OOM"))
    value = eta.neural_remaining(tmp_path, "uno", forecast)
    assert value["scientific_failure"] and value["remaining_seconds"] == [0., 0.]


def test_partial_and_wrong_source_figure_receipts_do_not_retire_eta_reservation(tmp_path):
    import hashlib
    directory = tmp_path / "w1_evaluation/confirmation"
    directory.mkdir(parents=True)
    marker = directory / "FIGURES_COMPLETE.json"
    marker.write_text("{", encoding="utf-8")
    assert not eta.completed_evaluation(directory)
    write_json(directory / "numeric_report.json", dict(status="numerical_complete"))
    write_json(marker, dict(file_sha256={}, numeric_report_sha256="a"*64))
    assert not eta.completed_evaluation(directory)
    digest = hashlib.sha256((directory / "numeric_report.json").read_bytes()).hexdigest()
    write_json(marker, dict(file_sha256={"figures/future_error.svg": "b"*64}, numeric_report_sha256=digest))
    assert eta.completed_evaluation(directory)
    write_json(directory / "numeric_report.json", dict(status="incomplete"))
    assert not eta.completed_evaluation(directory)


def test_completion_requires_all_real_audit_identities_not_a_filename(tmp_path):
    import hashlib
    receipt = tmp_path / "COMPLETION_RECEIPT.json"
    receipt.write_text("{", encoding="utf-8")
    assert not eta.completed_delivery(tmp_path)
    record = dict(schema="gift.w1w2-native.completion-receipt.v1", status="complete", audit_hashes={})
    write_json(receipt, record)
    assert not eta.completed_delivery(tmp_path)
    for name in ("FINAL_RESULT_AUDIT.json", "FINAL_TRAINING_AUDIT.json", "FINAL_COST_CLOSURE.json",
                 "FINAL_DOCUMENT_AUDIT.json", "VISUAL_QA.json"):
        path = tmp_path / name
        write_json(path, dict(status="passed"))
        record["audit_hashes"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(receipt, record)
    assert eta.completed_delivery(tmp_path)
    write_json(tmp_path / "VISUAL_QA.json", dict(status="pending"))
    assert not eta.completed_delivery(tmp_path)
