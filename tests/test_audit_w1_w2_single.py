"""Small CPU fixtures for independent raw-field audit and cost identity checks."""
from datetime import datetime, timedelta, timezone
import json

import h5py
import numpy as np
import pytest
import torch

from scripts import audit_w1_w2_single as audit


def write_measurement_fixture(directory, points):
    """Analytic raw fields give unit future error and two distinct failure types.

One trajectory has a NaN early future; another has a finite field but an
explicit solver failure. Scores are specified analytically, not copied from
the evaluator or from the audit's own numerical calculation.
"""
    directory.mkdir()
    (directory / "raw").mkdir()
    (directory / "metrics").mkdir()
    ids, times = np.arange(100000, 100180), np.arange(points)*.04
    truth_path = directory / "raw" / ("truth.h5" if points == 76 else "N64_truth.h5")
    source_path = directory / "reference.h5"
    for path, group in ((truth_path, None), (source_path, "test")):
        with h5py.File(path, "x") as handle:
            handle.attrs.update(status="complete", complete=True, trajectory_id_scheme="canonical")
            node = handle if group is None else handle.create_group(group)
            node.create_dataset("trajectory_index", data=ids)
            node.create_dataset("time", data=times)
            node.create_dataset("absolute_time", data=5+times)
            node.create_dataset("initial_parameters", data=np.ones((180, 4, 4)))
            node.create_dataset("vorticity", shape=(180, points, 64, 64), dtype="f4", fillvalue=1.)
    raw_path = directory / "raw" / "fno2d.h5"
    flags, failed = np.zeros(180, bool), np.zeros(180, bool)
    flags[9], failed[[5, 9]] = True, True
    with h5py.File(raw_path, "x") as handle:
        handle.attrs.update(status="complete", completed_trajectories=180, method="fno2d",
                            checkpoint_sha256="fixed", cohort="confirmation", grid=64)
        handle.create_dataset("trajectory_index", data=ids)
        handle.create_dataset("time", data=times)
        handle.create_dataset("absolute_time", data=5+times)
        handle.create_dataset("prediction", shape=(180, points, 64, 64), dtype="f4", fillvalue=2.)
        handle["prediction"][5, 3, 0, 0] = np.nan
        handle.create_dataset("solver_failed", data=flags)
    relative, finite = np.ones((180, points)), np.ones((180, points), bool)
    relative[5, 3], finite[5, 3] = np.nan, False
    vector = np.ones(180)
    vector[failed] = np.inf
    endpoint = "t8_relative_l2" if points == 76 else "t6_relative_l2"
    arrays = dict(future_mean_relative_l2=vector, future_mse=vector, future_path_relative_l2=vector,
                  t0_relative_l2=np.ones(180), **{endpoint: vector})
    if points == 26:
        for band in ("P21", "Q21", "P22_31", "outer_P31"):
            band_vector = vector.copy() if band == "P21" else np.where(failed, np.inf, 0.)
            for suffix in ("future_mean", "t6"):
                arrays[band+"_relative_l2_"+suffix] = band_vector
    all_mse = np.ones(180)
    all_mse[5] = np.nan
    metric_path = directory / "metrics" / "fno2d.h5"
    with h5py.File(metric_path, "x") as handle:
        handle.attrs["status"] = "complete"
        for key, values in (("relative_l2_by_time", relative), ("finite_by_time", finite),
                            ("failed", failed), ("solver_failed", flags)):
            handle.create_dataset(key, data=values)
        node = handle.create_group("per_trajectory") if points == 76 else handle
        node.create_dataset("trajectory_id", data=ids)
        for key, values in arrays.items():
            node.create_dataset(key, data=values)
        node.create_dataset("all_points_mse_diagnostic", data=all_mse)
    curve = [1.]*points
    curve[3] = None
    summaries = {key: dict(population=180, finite_count=180 if key == "t0_relative_l2" else 178,
                           mean=1. if key == "t0_relative_l2" else None) for key in arrays}
    record = dict(method="fno2d", seed=0, status="failed_trajectories", checkpoint_sha256="fixed",
                  raw_prediction_file="raw/fno2d.h5", raw_prediction_sha256=audit.Digests()(raw_path),
                  metric_file="metrics/fno2d.h5", cost=dict(inference_seconds=0.),
                  summary=dict(population=180, failed_count=2, failure_rate=2/180,
                               failed_ids=[100005, 100009], primary_metric=None, metrics=summaries,
                               mean_relative_l2_by_time=curve))
    source = audit.source_selection(source_path, "test", "confirmation", points, 64)
    return record, source, raw_path, metric_path


@pytest.mark.parametrize("points", (26, 76))
def test_independent_audit_recalculates_actual_fields_and_retains_both_failures(tmp_path, points):
    """Both W1 nested and W2 flat numerical formats match analytic population semantics."""
    directory = tmp_path / f"fixture{points}"
    record, source, _, _ = write_measurement_fixture(directory, points)
    checked = audit.audit_measurement(directory, record, source, cohort="confirmation", grid=64,
        points=points, label="fno2d", checkpoint_sha="fixed", digest=audit.Digests())
    assert checked["status"] == "independent_raw_recomputation_passed"
    assert checked["population"] == 180 and checked["failed_count"] == 2
    assert checked["primary_metric"] is None
    assert checked["max_absolute_numerical_delta"] < 1e-12


def test_numerical_audit_detects_changed_metrics_and_finite_subset_ranking(tmp_path):
    """Raw hashes alone cannot certify an altered metric or an invalid population mean."""
    directory = tmp_path / "fixture"
    record, source, _, metric_path = write_measurement_fixture(directory, 26)
    with h5py.File(metric_path, "r+") as handle:
        handle["future_mse"][0] = 1.01
    with pytest.raises(audit.AuditError, match="numerical values differ"):
        audit.audit_measurement(directory, record, source, cohort="confirmation", grid=64,
            points=26, label="fno2d", checkpoint_sha="fixed", digest=audit.Digests())
    with h5py.File(metric_path, "r+") as handle:
        handle["future_mse"][0] = 1.
    record["summary"]["primary_metric"] = 1.
    with pytest.raises(audit.AuditError, match="primary future mean"):
        audit.audit_measurement(directory, record, source, cohort="confirmation", grid=64,
            points=26, label="fno2d", checkpoint_sha="fixed", digest=audit.Digests())


def test_audit_excludes_initial_error_and_detects_saved_reference_substitution(tmp_path):
    """Initial reconstruction does not contaminate primary forecast norms."""
    truth, prediction = np.ones((2, 26, 64, 64), np.float32), np.ones((2, 26, 64, 64), np.float32)
    prediction[:, 0] = 100.
    measured = audit.independently_measure(prediction, truth, np.zeros(2, bool))
    np.testing.assert_array_equal(measured["per_trajectory"]["future_mean_relative_l2"], np.zeros(2))
    np.testing.assert_array_equal(measured["per_trajectory"]["future_mse"], np.zeros(2))
    np.testing.assert_array_equal(measured["per_trajectory"]["t0_relative_l2"], np.full(2, 99.))
    directory = tmp_path / "fixture"
    record, source, _, _ = write_measurement_fixture(directory, 26)
    with h5py.File(directory / "raw/N64_truth.h5", "r+") as handle:
        handle["vorticity"][0, 4] = 1.001
    with pytest.raises(audit.AuditError, match="truth differs from exact independent source"):
        audit.audit_measurement(directory, record, source, cohort="confirmation", grid=64,
            points=26, label="fno2d", checkpoint_sha="fixed", digest=audit.Digests())


def test_independent_outer_band_norms_on_known_integer_fourier_modes():
    """Only the prescribed untrained high mode changes, leaving the low and22–31 bands intact."""
    axis = np.arange(96)*2*np.pi/96
    field = (2+np.cos(25*axis[:, None])+np.cos(40*axis[None, :])).astype(np.float32)
    truth = np.broadcast_to(field, (2, 26, 96, 96)).copy()
    prediction = truth.copy()
    prediction[:, 1:] += (.5*np.cos(40*axis[None, :])).astype(np.float32)
    scores = audit.independently_measure_bands(prediction, truth, np.zeros(2, bool))
    np.testing.assert_allclose(scores["outer_P31_relative_l2_t6"], .5, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scores["reliable_outer_P31_relative_l2_future_mean"], .5, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scores["Q21_relative_l2_t6"], .5/np.sqrt(2), rtol=1e-6, atol=1e-6)
    assert np.max(scores["P21_relative_l2_future_mean"]) < 2e-7
    assert np.max(scores["P22_31_relative_l2_t6"]) < 2e-7


def event(second=0, *, resume=False, label="train_unet_seed0", module="training.w1_extended_baselines"):
    """A complete actual process receipt with a controlled distinct attempt time."""
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)+timedelta(seconds=second)
    command = ["python", "-m", module, "--run-config", "fixed.json"]
    if resume:
        command.append("--resume")
    return dict(label=label, command=command, started_utc=start.isoformat(),
                finished_utc=(start+timedelta(seconds=10)).isoformat(), process_seconds=10., exit_code=0)


def test_cost_audit_counts_resume_attempts_once_and_rejects_restarts_or_hidden_training():
    """Repeated initialization or duplicate/imported costs cannot masquerade as single-run work."""
    one, resumed = event(), event(20, resume=True)
    checked = audit.event_costs([one, resumed], {"train_unet_seed0"}, "fixture")
    assert checked["actual_training_seconds"] == 20.
    assert checked["by_job"]["train_unet_seed0"]["attempts"] == 2
    with pytest.raises(audit.AuditError, match="duplicated actual"):
        audit.event_costs([one, one], {"train_unet_seed0"}, "fixture")
    with pytest.raises(audit.AuditError, match="freshly initialized again"):
        audit.event_costs([one, event(20)], {"train_unet_seed0"}, "fixture")
    with pytest.raises(audit.AuditError, match="extra/reused/target-grid"):
        audit.event_costs([one, event(20, label="train_N128_unet")], {"train_unet_seed0"}, "fixture")


def test_training_only_normalizer_is_independently_verified_with_welford(tmp_path):
    """A held-out constant1000 cannot influence statistics of training299×2 plus one×3."""
    path = tmp_path / "observations.h5"
    with h5py.File(path, "x") as handle:
        train = handle.create_dataset("training/vorticity", shape=(300, 76, 64, 64), dtype="f4", fillvalue=2.)
        train[0] = 3.
        handle.create_dataset("validation/vorticity", shape=(20, 76, 64, 64), dtype="f4", fillvalue=1000.)
    norm = dict(kind="paper_UnitGaussianNormalizer", validation_or_test_used=False, eps=1e-5)
    for prefix, count in (("input", 1), ("output", 75)):
        norm[prefix+"_mean"] = torch.full((64, 64, count), 2+1/300)
        norm[prefix+"_std"] = torch.full((64, 64, count), 1/np.sqrt(300))
    receipt = audit.independently_check_normalizer(dict(normalization=norm), path)
    assert receipt["status"] == "independent_training_only_statistics_passed"
    assert receipt["fitting_population"] == 300 and receipt["validation_or_test_used"] is False
    norm["input_mean"] = torch.full((64, 64, 1), 1000.)
    with pytest.raises(audit.AuditError, match="numerical values differ"):
        audit.independently_check_normalizer(dict(normalization=norm), path)


def test_immutable_audit_outputs_and_no_hdf_access_before_final_markers(tmp_path, monkeypatch):
    """The final reader preserves existing receipts and refuses still-running raw artifacts."""
    target = tmp_path / "FINAL_RESULT_AUDIT.json"
    audit.save_immutable(target, dict(status="passed", identity="same"))
    first = target.read_bytes()
    audit.save_immutable(target, dict(status="passed", identity="same"))
    assert target.read_bytes() == first
    with pytest.raises(audit.AuditError, match="different evidence"):
        audit.save_immutable(target, dict(status="passed", identity="changed"))
    monkeypatch.setattr(h5py, "File", lambda *args, **kwargs: pytest.fail("unfinished artifacts must never be opened"))
    with pytest.raises(audit.AuditError, match="all numerical/report/figure"):
        audit.audit_run(tmp_path)


def test_scientific_failed_training_slot_remains_a_complete_population_without_prediction(tmp_path):
    """A true numerical training failure is complete scientific evidence, not a missing row."""
    record = dict(status="training_failed", summary=dict(population=180, failed_count=180, primary_metric=None))
    value = audit.audit_measurement(tmp_path, record, {}, cohort="confirmation", grid=128,
                                   points=26, label="uno", checkpoint_sha=None, digest=audit.Digests())
    assert value["status"] == "scientific_training_failure_retained" and value["failed_count"] == 180


def test_final_reader_routes_w1_plan_sources_and_preserves_all_fifty_failed_slots(tmp_path, monkeypatch):
    """W1 source is declared in its plan, and scientific failures still close all50 slots."""
    def save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    plan = dict(base_root=str(tmp_path / "base"), base_lock=str(tmp_path / "base_lock.json"),
                n64_confirmation=str(tmp_path / "confirmation.h5"), old_test=str(tmp_path / "old_test.h5"))
    save(tmp_path / "RUN_PLAN.json", plan)
    save(tmp_path / "base_lock.json", {})
    save(tmp_path / "EXTENSION_LOCK.json", {})
    save(tmp_path / "FORMAL.json", {})
    reports = {}
    failure = dict(status="training_failed", summary=dict(population=180, failed_count=180, primary_metric=None))
    for experiment in ("w1", "w2"):
        for cohort in audit.COHORTS:
            directory = tmp_path / f"{experiment}_evaluation" / cohort
            points, methods = (76, audit.W1_METHODS) if experiment == "w1" else (26, audit.W2_METHODS)
            records = {label: dict(failure) for label in methods}
            report = dict(cohort=cohort, trajectory_ids=audit.expected_ids(cohort).tolist(),
                          time_points=points, future_steps=points-1,
                          query_times=(np.arange(points)*.04).tolist(), absolute_times=(5+np.arange(points)*.04).tolist(),
                          methods=records if experiment == "w1" else {str(n): records for n in audit.GRIDS},
                          ANIE_included=False, native_zero_shot=True, test_statistics_used=False)
            # Real W1 reports intentionally have no data_file/group; their immutable plan does.
            source_path = plan["n64_confirmation"] if cohort == "confirmation" else plan["old_test"]
            group = "test" if cohort == "confirmation" else "N64/test"
            eval_plan = dict(data_file=source_path, group=group) if experiment == "w1" else dict(
                source_files={str(n): dict(path=source_path, group=group, sha256="fixture") for n in audit.GRIDS})
            save(directory / "evaluation_plan.json", eval_plan)
            save(directory / "numeric_report.json", report)
            reports[str(directory)] = report
    monkeypatch.setattr(audit, "completed_report", lambda directory, digest: reports[str(directory)])
    monkeypatch.setattr(audit, "Digests", lambda: lambda path: "fixture")
    monkeypatch.setattr(audit, "audit_training", lambda *args: (
        dict(status="passed", newly_trained_instances=3, method_slots={
            label: dict(status="scientific_training_failure_retained") for label in audit.W1_METHODS}), {}))
    sources = []
    def select(path, group, cohort, points, grid):
        sources.append((path, group, cohort, points, grid))
        return dict(parameters=np.ones((180, 4, 4)))
    monkeypatch.setattr(audit, "source_selection", select)
    result, training = audit.audit_run(tmp_path)
    assert result["status"] == training["status"] == "passed"
    assert result["audited_completed_or_failed_measurements"] == 50
    assert result["ANIE_in_W2"] is False and len(sources) == 8
    assert sources[0] == (plan["n64_confirmation"], "test", "confirmation", 76, 64)
    assert sources[1] == (plan["old_test"], "N64/test", "old_test", 76, 64)
    assert "anie_ns" not in result["measurements"]["w2_confirmation"]["128"]
    first = (tmp_path / "FINAL_RESULT_AUDIT.json").read_bytes()
    audit.audit_run(tmp_path)
    assert (tmp_path / "FINAL_RESULT_AUDIT.json").read_bytes() == first
