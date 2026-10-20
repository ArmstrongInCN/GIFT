"""CPU-only tests of the new single-slot protocol, persistence and lock boundary."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.additional import w1_single_evaluate as evaluation
from scripts import generate_w1_v2_confirmation as generation


def binding():
    """A synthetic receipt contains the exact declared selection, no solution data."""
    return dict(points=76, future_steps=75, query_times=evaluation.QUERY_TIMES.tolist(),
                observation_dt_seconds=.04, training_count=300, subset_seed=2026100304, time_stride=2,
                source_frame_indices=list(range(250, 401, 2)), manifest_sha256="c"*64,
                training=dict(ids=np.random.default_rng(2026100304).permutation(1000)[:300].tolist(), observations_sha256="a"*64),
                validation=dict(ids=list(range(1000, 1020)), observations_sha256="b"*64))


def locked(tmp_path):
    """Create one explicit four-method lock, with no development field."""
    protocol = tmp_path / "protocol.md"
    protocol.write_text("prespecified single-run synthetic protocol", encoding="utf-8")
    jobs = {}
    for method, seed in evaluation.METHOD_SEEDS.items():
        config = dict(profile=evaluation.PROFILE, stage="formal", seed=seed, max_epochs=5000 if method == "anie_ns" else 500)
        if method == "GIFT":
            config["branch_phases"] = [400, 50, 25, 25]
        jobs[method] = dict(method=method, seed=seed, profile=evaluation.PROFILE, stage="formal", config=config)
    value = dict(schema="gift.w1-single-native.protocol-lock.v1", profile=evaluation.PROFILE,
                 protocol_locked=True, configurations_locked=True, protocol_file=str(protocol.resolve()),
                 training_configurations=jobs, evaluation_query_times=evaluation.QUERY_TIMES.tolist(), evaluation_dt=.04,
                 observation_points=76, future_steps=75, training_count=300, training_subset_seed=2026100304,
                 data_binding=binding())
    path = tmp_path / "PROTOCOL_LOCK.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def checkpoint(tmp_path, method="fno2d", *, metric=.2):
    """Store a finite synthetic single-run state for the real metadata preflight."""
    torch = pytest.importorskip("torch")
    spec = dict(method=method, history=1, points=76, future_steps=75,
                query_times=evaluation.QUERY_TIMES.tolist(), observation_dt=.04)
    if method == "anie_ns":
        spec.update(source_profile="native_ns_single_v1", condition_mode="native")
    if method == "GIFT":
        spec.update(family="gift", step=.02, integration_dt=.02, save_interval=.04, duration=3.)
    seed = evaluation.METHOD_SEEDS[method]
    payload = dict(model_state={"weight": torch.tensor([1.])}, model_spec=spec,
                   config=dict(profile=evaluation.PROFILE, stage="formal", seed=seed, max_epochs=5000 if method == "anie_ns" else 500),
                   stage="formal", seed=seed, epoch=525 if method == "anie_ns" else 500,
                   val_metric=metric, data_binding=binding())
    if method == "GIFT":
        payload["config"]["branch_phases"] = [400, 50, 25, 25]
        payload["selection"] = "terminal_epoch"
    elif method.startswith("fno"):
        payload["status"] = "terminal"
    if method == "fno3d":
        payload["normalization"] = {
            "input_mean": torch.zeros(64, 64, 1), "input_std": torch.ones(64, 64, 1),
            "output_mean": torch.zeros(64, 64, 75), "output_std": torch.ones(64, 64, 75)}
    path = tmp_path / (method + ".pt")
    torch.save(payload, path)
    if method == "anie_ns":
        (tmp_path / "STATUS.json").write_text(json.dumps(dict(status="complete", method=method, seed=seed,
                                                             selected_checkpoint=str(path.resolve()), epoch=5000)), encoding="utf-8")
    return path, payload


def failed(method):
    """An explicit scientific failure still occupies its fixed method slot."""
    return dict(method=method, seed=evaluation.METHOD_SEEDS[method], reason="synthetic nonfinite training gradient")


def test_aligned_checkpoint_must_match_the_trained_profile_and_lock(tmp_path):
    """A corrected model cannot be an old checkpoint with inference metadata changed."""
    torch = pytest.importorskip("torch")
    _, lock = locked(tmp_path)
    path, payload = checkpoint(tmp_path, "anie_ns")
    payload["model_spec"]["source_profile"] = "aligned_ns_single_v1"
    lock["training_configurations"]["anie_ns"]["model_spec"] = {
        "source_profile": "aligned_ns_single_v1", "condition_mode": "native"}
    for provenance in ({}, {"upstream": {"source_profile": "native_ns_single_v1"}}):
        payload["source_provenance"] = provenance
        torch.save(payload, path)
        with pytest.raises(ValueError, match="trained source provenance"):
            evaluation.checkpoint_metadata(path, lock)
    payload["source_provenance"] = {"upstream": {"source_profile": "aligned_ns_single_v1"}}
    torch.save(payload, path)
    assert evaluation.checkpoint_metadata(path, lock)["model_spec"]["source_profile"] == "aligned_ns_single_v1"


def synthetic_data(path, cohort="confirmation", *, population=180):
    """HDF fill values represent a constant task without generating physical trajectories."""
    first = 100000 if cohort == "confirmation" else 1040
    with h5py.File(path, "x") as handle:
        handle.attrs.update(trajectory_id_scheme="canonical", complete=True)
        node = handle.create_group("test")
        node.create_dataset("trajectory_index", data=np.arange(first, first+population))
        node.create_dataset("absolute_time", data=5+evaluation.QUERY_TIMES)
        node.create_dataset("vorticity", shape=(population, 76, 64, 64), dtype="f4", fillvalue=1.)


def test_new_confirmation_lock_requires_no_fictitious_development(tmp_path):
    """The new explicit configuration lock is sufficient; legacy locks remain strict."""
    path, value = locked(tmp_path)
    receipt = generation.locked_configuration(path)
    assert receipt["evaluation_points"] == 76
    assert "development_locked" not in receipt["lock_snapshot"]
    assert evaluation.protocol_lock(path)["lock_snapshot"] == value
    plan = generation.confirmation_plan(receipt["evaluation_query_times"], profile=evaluation.PROFILE)
    assert plan["vorticity_shape"] == [180, 151, 64, 64]
    assert plan["evaluation_future_steps"] == 75
    assert "no development search" in plan["purpose"]
    value["configurations_locked"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        generation.locked_configuration(path)
    legacy = dict(protocol_locked=True, development_locked=False, protocol_file=value["protocol_file"],
                  selected_configurations={"anie_ns": {"learning_rate": .0001}})
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        generation.locked_configuration(path)


@pytest.mark.parametrize("bad", ("seed", "profile", "recipe", "coordinate", "observations", "nonfinite"))
def test_frozen_preflight_rejects_old_or_changed_recipe_before_data_access(tmp_path, bad):
    """A plausible shape cannot hide the wrong source profile, data or trained state."""
    torch = pytest.importorskip("torch")
    lock_path, lock = locked(tmp_path)
    path, payload = checkpoint(tmp_path, "anie_ns")
    if bad == "seed":
        payload["seed"] = 11
    elif bad == "profile":
        payload["config"]["profile"] = "old_v2"
    elif bad == "recipe":
        payload["config"]["max_epochs"] = 1500
    elif bad == "coordinate":
        payload["model_spec"]["source_profile"] = "spatially_aligned"
    elif bad == "observations":
        payload["data_binding"]["validation"]["observations_sha256"] = "d"*64
    else:
        payload["model_state"]["weight"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoints([path], tmp_path / "must_not_be_opened.h5", "test", "confirmation", tmp_path / "out",
                                        lock_file=lock_path, failed_runs=[failed(name) for name in evaluation.METHOD_SEEDS if name != "anie_ns"])
    assert not (tmp_path / "out").exists()


def test_terminal_validation_failure_is_not_new_selection_gate(tmp_path):
    """Finite terminal models are scored, while ANIE requires a finite best MSE."""
    _, lock = locked(tmp_path)
    for method in ("fno2d", "fno3d", "GIFT"):
        path, _ = checkpoint(tmp_path, method, metric=None)
        assert evaluation.checkpoint_metadata(path, lock)["val_metric"] is None
    path, _ = checkpoint(tmp_path, "anie_ns", metric=None)
    with pytest.raises(ValueError, match="finite native"):
        evaluation.checkpoint_metadata(path, lock)


def test_real_native_job_config_types_match_all_four_checkpoint_boundaries(tmp_path):
    """Actual controller recipes match JSON-locked lists/tuples without reading data."""
    torch = pytest.importorskip("torch")
    from scripts import run_w1_single
    from training import w1_single_train, w1_single_gift
    lock_path, value = locked(tmp_path)
    plan = run_w1_single.default_plan(tmp_path / "new_run")
    value["training_configurations"] = {method: run_w1_single.job_configuration(plan, method)
                                          for method in evaluation.METHOD_SEEDS}
    lock_path.write_text(json.dumps(value), encoding="utf-8")
    lock = evaluation.protocol_lock(lock_path)["lock_snapshot"]
    for method in evaluation.METHOD_SEEDS:
        path, payload = checkpoint(tmp_path, method)
        payload["config"] = (w1_single_gift.native_training_config() if method == "GIFT" else
                             w1_single_train.native_training_config(method))
        payload["model_spec"].update(value["training_configurations"][method].get("model_spec", {}))
        torch.save(payload, path)
        measured = evaluation.checkpoint_metadata(path, lock)
        assert measured["seed"] == evaluation.METHOD_SEEDS[method]


def test_best_or_terminal_selection_cannot_be_taken_mid_training(tmp_path):
    """A best snapshot from a running ANIE or an early FNO snapshot is ineligible."""
    torch = pytest.importorskip("torch")
    _, lock = locked(tmp_path)
    path, payload = checkpoint(tmp_path, "anie_ns")
    status = tmp_path / "STATUS.json"
    value = json.loads(status.read_text(encoding="utf-8"))
    value["status"] = "running"
    status.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="completed fixed"):
        evaluation.checkpoint_metadata(path, lock)
    path, payload = checkpoint(tmp_path)
    payload["epoch"] = 200
    torch.save(payload, path)
    with pytest.raises(ValueError, match="epoch budget"):
        evaluation.checkpoint_metadata(path, lock)


def test_exactly_four_single_slots_are_required_before_test_read(tmp_path):
    """Dropping a failed method, duplicating a seed or using the old seed is rejected."""
    lock_path, _ = locked(tmp_path)
    path, _ = checkpoint(tmp_path)
    for paths, failures in (([path], []), ([path, path], [failed("anie_ns"), failed("fno3d")]),
                            ([path], [dict(failed("anie_ns"), seed=11), failed("fno3d"), failed("GIFT")])):
        with pytest.raises(ValueError):
            evaluation.evaluate_checkpoints(paths, tmp_path / "absent.h5", "test", "confirmation", tmp_path / "out",
                                            lock_file=lock_path, failed_runs=failures)


def test_numeric_first_full_population_single_slot_and_completed_resume(tmp_path, monkeypatch):
    """The whole180 synthetic cohort is scored only after raw outputs are committed."""
    torch = pytest.importorskip("torch")
    lock_path, _ = locked(tmp_path)
    path, payload = checkpoint(tmp_path)
    data, output = tmp_path / "synthetic.h5", tmp_path / "evaluation"
    synthetic_data(data)
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: torch.nn.Identity())
    from training import w1_single_train
    monkeypatch.setattr(w1_single_train, "forward_model", lambda model, initial, *args, **kwargs:
                        (initial*2).unsqueeze(1).expand(-1, 76, -1, -1))
    original_truth = evaluation.shared.ObservationWindow.truth
    def checked_truth(window):
        with h5py.File(output / "raw/fno2d.h5", "r") as raw:
            assert raw.attrs["status"] == "complete" and raw.attrs["completed_trajectories"] == 180
        assert not (output / "figures").exists()
        return original_truth(window)
    monkeypatch.setattr(evaluation.shared.ObservationWindow, "truth", checked_truth)
    failures = [failed(name) for name in evaluation.METHOD_SEEDS if name != "fno2d"]
    result = evaluation.evaluate_checkpoints([path], data, "test", "confirmation", output, lock_file=lock_path,
                                             failed_runs=failures, plots=False, batch_size=60)
    assert result["population"] == 180
    assert result["methods"]["fno2d"]["summary"]["primary_metric"] == 1.
    assert result["methods"]["GIFT"]["summary"]["failed_count"] == 180
    assert "expected_training_seeds" not in result and "training_seed_sd" not in result
    assert (output / "RESULTS.md").exists()
    raw_hash = evaluation.shared.file_sha256(output / "raw/fno2d.h5")
    def forbidden(*args, **kwargs):
        raise AssertionError("complete autonomous predictions must be reused")
    monkeypatch.setattr(w1_single_train, "forward_model", forbidden)
    evaluation.shared.preserve_interrupted(output / "numeric_report.json")
    recovered = evaluation.evaluate_checkpoints([path], data, "test", "confirmation", output, lock_file=lock_path,
                                                failed_runs=failures, plots=False, resume=True)
    assert recovered["methods"]["fno2d"]["summary"]["primary_metric"] == 1.
    assert evaluation.shared.file_sha256(output / "raw/fno2d.h5") == raw_hash


def test_partial_raw_resume_retains_prefix_and_cost(tmp_path, monkeypatch):
    """One committed trajectory survives a resource interruption without re-prediction."""
    torch = pytest.importorskip("torch")
    path, payload = checkpoint(tmp_path)
    data = tmp_path / "small_synthetic.h5"
    synthetic_data(data, population=2)
    window = evaluation.shared.ObservationWindow(data, "test", np.arange(2), np.arange(76),
                                                 np.arange(100000, 100002), "confirmation", evaluation.QUERY_TIMES)
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: torch.nn.Identity())
    from training import w1_single_train
    calls = []
    def interrupted(model, initial, *args, **kwargs):
        calls.append(len(initial))
        if len(calls) == 2:
            raise RuntimeError("synthetic resource interruption")
        return initial.unsqueeze(1).expand(-1, 76, -1, -1)
    monkeypatch.setattr(w1_single_train, "forward_model", interrupted)
    raw = tmp_path / "partial.h5"
    with pytest.raises(RuntimeError, match="resource"):
        evaluation.predict_checkpoint(path, payload, window, raw, batch_size=1)
    with h5py.File(raw, "r") as saved:
        assert saved.attrs["completed_trajectories"] == 1
        prefix = saved["prediction"][:1]
        prior_cost = json.loads(saved.attrs["inference_cost_json"])
    prior = evaluation.shared.preserve_interrupted(raw)
    calls.clear()
    def continued(model, initial, *args, **kwargs):
        calls.append(len(initial))
        return initial.unsqueeze(1).expand(-1, 76, -1, -1)
    monkeypatch.setattr(w1_single_train, "forward_model", continued)
    cost = evaluation.predict_checkpoint(path, payload, window, raw, batch_size=1, previous_raw=prior)
    assert calls == [1]
    assert cost["inference_seconds"] >= prior_cost["inference_seconds"] and len(cost["inference_segments"]) == 2
    assert cost["precision"] == "float32; TF32 disabled"
    with h5py.File(raw, "r") as saved:
        assert saved.attrs["status"] == "complete" and np.array_equal(saved["prediction"][:1], prefix)


def test_gift_native_optional_reasons_preserve_success_and_failure(tmp_path, monkeypatch):
    """Native None success reasons serialize without hiding a real solver failure."""
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    from experiments.formal._shared import gift_runtime
    path, payload = checkpoint(tmp_path, "GIFT")
    data = tmp_path / "small_synthetic.h5"
    synthetic_data(data, population=2)
    window = evaluation.shared.ObservationWindow(data, "test", np.arange(2), np.arange(76),
                                                 np.arange(100000, 100002), "confirmation", evaluation.QUERY_TIMES)
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: (object(), object()))
    monkeypatch.setattr(evaluation, "training_cost", lambda *args: {})
    prediction = np.ones((2, 76, 64, 64), np.float32)
    prediction[1, 3:] = np.nan
    def native_rollout(**kwargs):
        assert kwargs["step"] == .02 and kwargs["save_interval"] == .04
        assert kwargs["correction"] is True
        return SimpleNamespace(prediction=prediction, failure_step=np.array([-1, 3]),
                               failure_reason=(None, "correction_nonfinite"))
    monkeypatch.setattr(gift_runtime, "rollout_gift", native_rollout)
    raw = tmp_path / "gift.h5"
    evaluation.predict_checkpoint(path, payload, window, raw, batch_size=1)
    with h5py.File(raw, "r") as saved:
        assert saved.attrs["status"] == "complete" and saved.attrs["completed_trajectories"] == 2
        assert saved["solver_failed"][:].tolist() == [False, True]
        assert saved["failure_reason"].asstr()[:].tolist() == ["", "correction_nonfinite"]
        np.testing.assert_array_equal(saved["prediction"][:], prediction)


def test_single_pairing_is_by_ids_not_seed_and_cannot_intersect_failures():
    """Actual unequal method seeds are retained; pairing conditions on trained models."""
    records = []
    for index, (method, seed) in enumerate(evaluation.METHOD_SEEDS.items()):
        records.append(dict(method=method, seed=seed, status="complete", per_trajectory=dict(
            trajectory_id=[2, 1], future_mean_relative_l2=[index+.2, index+.1])))
    pairs = evaluation.paired_statistics(records, bootstrap_replicates=20)
    assert len(pairs) == 6 and pairs[0]["mean_delta"] == -1.
    assert pairs[0]["trajectory_ids"] == [1, 2]
    records[1]["per_trajectory"]["trajectory_id"] = [1, 3]
    with pytest.raises(ValueError, match="populations differ"):
        evaluation.paired_statistics(records)
    records[1]["status"] = "failed_trajectories"
    assert evaluation.paired_statistics(records)[0]["status"] == "failed_training_or_predictions"


def test_job_events_keep_retry_cost_and_historical_cost_separate(tmp_path):
    """An interrupted training process is counted; evaluation and old runs are separate."""
    rows = [dict(label="training_fno2d_seed0", command=["python", "-m", "training.w1_single_train"], process_seconds=10., exit_code=1),
            dict(label="training_fno2d_seed0", command=["python", "-m", "training.w1_single_train"], process_seconds=20., exit_code=0),
            dict(label="evaluation_confirmation", command=["python", "-m", "experiments.additional.w1_single_evaluate"], process_seconds=5.)]
    new, old = tmp_path / "new.jsonl", tmp_path / "old.jsonl"
    new.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    old.write_text(json.dumps(dict(process_seconds=99.)) + "\n" + json.dumps(dict(label="formal_fno2d_seed11_candidate0.001",
                   event="user_requested_stop", status="paused_by_user")), encoding="utf-8")
    cost = evaluation.job_cost_summary(new, old)
    assert cost["method_training_process_seconds"]["fno2d"] == 30.
    assert cost["training_process_seconds_total"] == 30.
    assert cost["all_recorded_new_child_process_seconds"] == 35.
    assert cost["historical_sunk_cost"]["recorded_child_process_seconds"] == 99.
    assert cost["historical_sunk_cost"]["unclosed_or_unmeasured_labels"] == ["formal_fno2d_seed11_candidate0.001"]
    assert not cost["historical_sunk_cost"]["complete_historical_wall_cost_known"]


def test_all_four_training_failures_still_complete_population_report_and_curve(tmp_path):
    """No success threshold may hide four declared scientific failures or abort reporting."""
    pytest.importorskip("matplotlib")
    lock_path, _ = locked(tmp_path)
    data, output = tmp_path / "synthetic.h5", tmp_path / "all_failed"
    synthetic_data(data)
    report = evaluation.evaluate_checkpoints([], data, "test", "confirmation", output, lock_file=lock_path,
                                             failed_runs=[failed(method) for method in evaluation.METHOD_SEEDS], plots=True)
    assert len(report["methods"]) == 4
    assert all(record["summary"]["failed_count"] == 180 for record in report["methods"].values())
    assert (output / "figures/future_error.svg").exists() and (output / "RESULTS.md").exists()
    assert not (output / "raw/truth.h5").exists()
    marker = output / "REPORT_COMPLETE.json"
    evaluation.shared.preserve_interrupted(marker)
    (output / "RESULTS.md").write_text("synthetic incomplete rendering", encoding="utf-8")
    evaluation.render_report(output)
    assert "synthetic incomplete" not in (output / "RESULTS.md").read_text(encoding="utf-8")
    assert (output / "RESULTS.interrupted_000.md").exists()


def test_keyframes_use_actual_seed_and_prespecified_nodes(tmp_path):
    """Small saved numbers verify plotting provenance, including failed method slots."""
    pytest.importorskip("matplotlib")
    output, figures = tmp_path, tmp_path / "figures"
    (output / "raw").mkdir()
    figures.mkdir()
    with h5py.File(output / "raw/truth.h5", "x") as saved:
        saved.create_dataset("trajectory_index", data=[1040])
        saved.create_dataset("vorticity", shape=(1, 76, 64, 64), dtype="f4", fillvalue=1.)
    with h5py.File(output / "raw/anie_ns.h5", "x") as saved:
        saved.create_dataset("trajectory_index", data=[1040])
        saved.create_dataset("prediction", shape=(1, 76, 64, 64), dtype="f4", fillvalue=2.)
    methods = {method: dict(seed=seed, status="training_failed", summary={}) for method, seed in evaluation.METHOD_SEEDS.items()}
    methods["anie_ns"].update(status="complete", raw_prediction_file="raw/anie_ns.h5")
    provenance = evaluation.draw_keyframes(output, figures, dict(cohort="old_test", methods=methods,
                                      method_seeds=evaluation.METHOD_SEEDS, absolute_times=(5+evaluation.QUERY_TIMES).tolist()))
    assert provenance["method_seeds"]["anie_ns"] == 7
    assert provenance["source_time_indices"] == [12, 25, 50, 75]
    assert np.allclose(provenance["absolute_time"], [5.48, 6., 7., 8.])
    assert not provenance["interpolation"] and (figures / "fixed_id_keyframes.svg").exists()
