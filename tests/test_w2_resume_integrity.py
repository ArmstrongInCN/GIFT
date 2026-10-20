"""W2 cache recovery checks real receipt/filesystem boundaries using sparse CPU fixtures."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments.additional import w2_single_evaluate as evaluation


class FixtureWindow:
    """Broadcast reference fields avoid allocating or integrating a scientific dataset."""
    def __init__(self, path, grid):
        self.path, self.group, self.grid, self.cohort = path, f"N{grid}/test", grid, "confirmation"
        self.trajectory_ids = np.arange(100000, 100180, dtype=np.int64)

    def truth(self, first, last):
        return np.broadcast_to(np.ones((), np.float32), (last-first, 26, self.grid, self.grid))


def fixture_measure(prediction, truth, ids, *, solver_failed=None):
    """Small deterministic arrays isolate persistence semantics from FFT metric math."""
    failed = np.asarray(solver_failed, bool)
    relative = np.zeros((len(ids), 26), np.float64)
    per_trajectory = dict(trajectory_id=np.asarray(ids))
    for name in ("future_mean_relative_l2", "t6_relative_l2", "future_mse", "future_path_relative_l2", "t0_relative_l2"):
        values = np.zeros(len(ids), np.float64)
        if name != "t0_relative_l2":
            values[failed] = np.inf
        per_trajectory[name] = values
    return dict(per_trajectory=per_trajectory, relative_l2_by_time=relative,
                finite_by_time=np.ones((len(ids), 26), bool), failed=failed, solver_failed=failed)


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    """Use actual six-method orchestration with one inferred slot and five fixed failures."""
    from experiments.additional import w1_extended_evaluate, w1_single_evaluate
    base, extension, checkpoint = (tmp_path / name for name in ("base.json", "extension.json", "weights.pt"))
    base.write_text("{}", encoding="utf-8")
    extension.write_text(json.dumps(dict(schema="gift.w1w2-native.extension-lock.v1", profile="single_native76_n300",
                                        configurations_locked=True)), encoding="utf-8")
    checkpoint.write_bytes(b"synthetic fixed checkpoint; no real model")
    windows = {}
    for grid in evaluation.GRIDS:
        path = tmp_path / f"reference_{grid}.fixture"
        path.write_bytes(b"synthetic immutable reference receipt")
        windows[grid] = FixtureWindow(path, grid)
    payload = dict(seed=0, epoch=500, config={"fixture_only": True}, model_spec={"method": "fno2d"})
    monkeypatch.setattr(w1_single_evaluate, "protocol_lock", lambda path: {"lock_snapshot": {}})
    monkeypatch.setattr(w1_extended_evaluate, "checkpoint_metadata", lambda *args: payload)
    monkeypatch.setattr(evaluation, "paired_windows", lambda *args: windows)
    monkeypatch.setattr(evaluation, "measure_predictions", fixture_measure)
    calls = dict(predict=[], score=[])

    def predict(label, checkpoint, payload, window, raw_path, **kwargs):
        calls["predict"].append(window.grid)
        failed = np.zeros(180, bool)
        failed[7] = True
        cost = dict(inference_seconds=.25, fixture_only=True)
        with h5py.File(raw_path, "x") as raw:
            raw.attrs.update(status="complete", method=label, checkpoint_sha256=evaluation.shared.file_sha256(checkpoint),
                             grid=window.grid, cohort=window.cohort, completed_trajectories=180,
                             inference_cost_json=json.dumps(cost))
            raw.create_dataset("prediction", shape=(180, 26, window.grid, window.grid), dtype="f4", fillvalue=1.)
            raw.create_dataset("trajectory_index", data=window.trajectory_ids)
            raw.create_dataset("time", data=evaluation.QUERY_TIMES)
            raw.create_dataset("absolute_time", data=5+evaluation.QUERY_TIMES)
            raw.create_dataset("solver_failed", data=failed)
            raw.create_dataset("failure_reason", data=np.asarray(["fixture failure" if flag else "" for flag in failed],
                                                                dtype=h5py.string_dtype("utf-8")))
        return cost

    def score(raw_path, window, metric_path, truth_path):
        calls["score"].append(window.grid)
        with h5py.File(raw_path, "r") as raw:
            measured = fixture_measure(None, None, window.trajectory_ids, solver_failed=raw["solver_failed"][:])
        with h5py.File(metric_path, "x") as stored:
            stored.attrs["status"] = "complete"
            for name, value in measured["per_trajectory"].items():
                stored.create_dataset(name, data=value)
            for name in ("relative_l2_by_time", "finite_by_time", "failed", "solver_failed"):
                stored.create_dataset(name, data=measured[name])
        return evaluation._metric_summary(measured["per_trajectory"], measured["relative_l2_by_time"],
                                          measured["failed"], window.trajectory_ids)

    monkeypatch.setattr(evaluation, "predict_checkpoint", predict)
    monkeypatch.setattr(evaluation, "score_raw", score)
    failures = {}
    for label in evaluation.METHODS:
        if label == "fno2d":
            continue
        path = tmp_path / (label + ".failure.json")
        path.write_text(json.dumps(dict(method="GIFT" if label == "GIFT-N50" else label,
            seed=20260820 if label in ("GIFT", "GIFT-N50") else 0,
            reason="synthetic fixed numerical failure", failure_kind="numerical")), encoding="utf-8")
        failures[label] = path
    output = tmp_path / "W2"
    arguments = dict(base_lock_file=base, extension_lock_file=extension, failed_runs=failures, plots=False)
    report = evaluation.evaluate_checkpoints({"fno2d": checkpoint}, windows[64].path, "confirmation", output, **arguments)
    original = (output / "numeric_report.json").read_bytes()

    def resume():
        return evaluation.evaluate_checkpoints({"fno2d": checkpoint}, windows[64].path, "confirmation", output,
                                               resume=True, **arguments)

    return SimpleNamespace(output=output, checkpoint=checkpoint, report=report, original=original,
                           calls=calls, windows=windows, resume=resume)


def replace_json(path, value):
    """Modify only this test's synthetic receipt; no historical project results are used."""
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_completed_report_reuses_all_bound_outputs_and_preserves_failed_population(completed_run):
    run = completed_run
    assert run.resume() == run.report
    assert run.calls == dict(predict=[64, 96, 128], score=[64, 96, 128])
    assert (run.output / "numeric_report.json").read_bytes() == run.original
    for grid in evaluation.GRIDS:
        slots = run.report["methods"][str(grid)]
        assert set(slots) == set(evaluation.METHODS)
        assert slots["fno2d"]["summary"]["failed_count"] == 1
        assert slots["fno2d"]["summary"]["primary_metric"] is None
        assert all(record["summary"]["population"] == 180 for record in slots.values())
        assert all(record["summary"]["failed_count"] == 180 for label, record in slots.items() if label != "fno2d")


def test_summary_reduction_roundoff_is_accepted_without_replacing_recorded_value(completed_run):
    """A float64 reduction ulp is not a changed experimental outcome."""
    run = completed_run
    report_path = run.output / "numeric_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entry = report["methods"]["96"]["fno2d"]
    entry["summary"]["metrics"]["t0_relative_l2"]["mean"] = 1e-15
    replace_json(run.output / "metrics/N96_fno2d.json", entry)
    replace_json(report_path, report)
    before = report_path.read_bytes()
    assert run.resume() == report
    assert report_path.read_bytes() == before
    assert run.calls == dict(predict=[64, 96, 128], score=[64, 96, 128])


def test_w2_fno3d_forward_uses_current_source_location_without_changing_payload(tmp_path, monkeypatch):
    """FNO3D's native normalizer import during forward must use the relocated root too."""
    import torch
    from training import w1_single_train
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", str(upstream))
    seen = []

    def predict(model, initial, spec, normalization, **kwargs):
        seen.append(spec["source_root"])
        return torch.ones(1, 76, 2, 2)

    monkeypatch.setattr(w1_single_train, "forward_model", predict)
    payload = dict(model_spec=dict(method="fno3d", source_root="recorded-source-directory"))
    value = evaluation.forward_predictor("fno3d", None, torch.ones(1, 2, 2), payload, None, "cpu")
    assert value.shape == (1, 26, 2, 2) and seen == [str(upstream)]
    assert payload["model_spec"]["source_root"] == "recorded-source-directory"


def test_relocated_predictor_records_actual_runtime_source_bytes(tmp_path, monkeypatch):
    """New inference receipts describe the loaded source rather than an old machine path."""
    import torch
    from adapters import w1_extended
    upstream = tmp_path / "upstream"
    source = upstream / "unet/Baselines/U-net.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic source receipt only; not an algorithm implementation.\n", encoding="utf-8")
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", str(upstream))
    model = torch.nn.Linear(1, 1)
    model.gift_external_source = dict(directory=str(upstream / "unet"), family="unet")
    seen = []

    def build(spec, device):
        seen.append(spec["source_root"])
        return model

    monkeypatch.setattr(w1_extended, "build_network", build)
    payload = dict(model_spec=dict(method="unet", history=1, source_root="old-machine-directory"),
                   model_state=model.state_dict())
    _, _, audit = evaluation.load_predictor("unet", tmp_path / "synthetic.pt", payload, 96, "cpu")
    assert seen == [str(upstream)]
    assert audit["runtime_source"]["source_root"] == str(upstream)
    assert audit["runtime_source"]["native_files_sha256"]["Baselines/U-net.py"] == evaluation.shared.file_sha256(source)
    assert audit["runtime_source"]["source_hashes_are_provenance_only"]
    assert payload["model_spec"]["source_root"] == "old-machine-directory"


@pytest.mark.parametrize("missing", ["raw", "metric", "record", "failed_record"])
def test_missing_numerical_artifact_continues_same_run_and_keeps_original_receipts(completed_run, missing):
    run = completed_run
    paths = dict(raw=run.output / "raw/N96_fno2d.h5", metric=run.output / "metrics/N96_fno2d.h5",
                 record=run.output / "metrics/N96_fno2d.json", failed_record=run.output / "metrics/N96_unet.json")
    path = paths[missing]
    path.rename(path.with_name(path.name + ".retained"))
    resumed = run.resume()
    assert path.is_file() and resumed["status"] == "numerical_complete"
    assert (run.output / "numeric_report.interrupted_000.json").read_bytes() == run.original
    assert run.calls["predict"] == ([64, 96, 128, 96] if missing == "raw" else [64, 96, 128])
    assert run.calls["score"] == ([64, 96, 128] if missing == "failed_record" else [64, 96, 128, 96])
    assert resumed["methods"]["96"]["fno2d"]["summary"]["failed_count"] == 1


@pytest.mark.parametrize("corruption", ["raw_bytes", "metric_bytes", "metric_population", "report_population",
    "report_checkpoint", "report_missing_method", "record_summary", "failed_population", "checkpoint", "plan", "report_json"])
def test_corrupt_completed_artifacts_raise_without_replacing_numerical_report(completed_run, corruption):
    run = completed_run
    report_path = run.output / "numeric_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if corruption == "raw_bytes":
        with h5py.File(run.output / "raw/N96_fno2d.h5", "r+") as raw:
            raw.attrs["tampered"] = True
    elif corruption == "metric_bytes":
        with h5py.File(run.output / "metrics/N96_fno2d.h5", "r+") as metrics:
            metrics["future_mse"][0] = .5
    elif corruption == "metric_population":
        with h5py.File(run.output / "metrics/N96_fno2d.h5", "r+") as metrics:
            metrics["trajectory_id"][0] = -1
    elif corruption == "report_population":
        report["trajectory_ids"][0] = -1
        replace_json(report_path, report)
    elif corruption == "report_checkpoint":
        report["methods"]["96"]["fno2d"]["checkpoint_sha256"] = "incorrect"
        replace_json(report_path, report)
    elif corruption == "report_missing_method":
        report["methods"]["96"].pop("unet")
        replace_json(report_path, report)
    elif corruption == "record_summary":
        record_path = run.output / "metrics/N96_fno2d.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["summary"]["failed_count"] = 0
        replace_json(record_path, record)
    elif corruption == "failed_population":
        report["methods"]["96"]["unet"]["summary"]["population"] = 179
        replace_json(report_path, report)
    elif corruption == "checkpoint":
        run.checkpoint.write_bytes(b"different synthetic fixed weights")
    elif corruption == "plan":
        path = run.output / "evaluation_plan.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["trajectory_ids"][0] = -1
        replace_json(path, plan)
    else:
        report_path.write_text("{broken", encoding="utf-8")
    before = report_path.read_bytes()
    with pytest.raises(ValueError):
        run.resume()
    assert report_path.read_bytes() == before
    assert run.calls == dict(predict=[64, 96, 128], score=[64, 96, 128])


@pytest.mark.parametrize("tamper", [False, True])
def test_legacy_metrics_without_digest_are_checked_against_raw_and_reference(completed_run, tamper):
    """Legacy validation cannot bless changed arrays by establishing a new digest."""
    run = completed_run
    report_path = run.output / "numeric_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entry = report["methods"]["96"]["fno2d"]
    entry.pop("metric_sha256")
    if tamper:
        # Keep the public summary unchanged while moving an error between two
        # trajectories. Population means alone cannot catch this alteration.
        with h5py.File(run.output / "metrics/N96_fno2d.h5", "r+") as metrics:
            metrics["relative_l2_by_time"][0, 0] = .125
            metrics["relative_l2_by_time"][1, 0] = -.125
        entry["summary"]["mean_relative_l2_by_time"][0] = 0.
    replace_json(run.output / "metrics/N96_fno2d.json", entry)
    replace_json(report_path, report)
    before = report_path.read_bytes()
    if tamper:
        with pytest.raises(ValueError, match="legacy W2 metric arrays differ"):
            run.resume()
    else:
        assert run.resume() == report
    assert report_path.read_bytes() == before
    assert run.calls == dict(predict=[64, 96, 128], score=[64, 96, 128])
