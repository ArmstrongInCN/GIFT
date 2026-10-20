"""Check full-population scoring, paired seeds, target isolation and lock rules."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.additional import w1_v2_evaluate as evaluation
from scripts import generate_w1_v2_confirmation as generation


def fields(batch=2, points=151):
    """Constant nonzero truth permits exact independently computed metrics."""
    return np.ones((batch, points, 64, 64), dtype=np.float32)


def run_record(method, seed, value, *, ids=(9, 3)):
    """Construct real scorer outputs for seed aggregation and ID alignment tests."""
    truth = fields(len(ids))
    measured = evaluation.evaluate_predictions(truth * (1. + value), truth, np.asarray(ids))
    return {"method": method, "seed": seed, "stage": "formal",
            "status": measured["summary"]["status"], "summary": measured["summary"],
            "per_trajectory": measured["per_trajectory"]}


def test_primary_excludes_t0_and_distinguishes_frame_mean_from_path_error():
    """Unequal future errors/energies distinguish the declared metric formulas."""
    truth = fields(1)
    truth[:, 76:] *= 2.
    prediction = truth.copy()
    prediction[:, 0] = 100.
    prediction[:, 1:76] *= 1.2
    prediction[:, 76:] *= 1.4
    result = evaluation.evaluate_predictions(prediction, truth, np.array([1040]))
    scores = result["per_trajectory"]
    assert result["summary"]["primary_metric"] == pytest.approx(.3, abs=1e-7)
    assert scores["future_path_relative_l2"][0] == pytest.approx(np.sqrt((.2**2 + 4*.4**2)/5), abs=1e-7)
    assert scores["future_mse"][0] == pytest.approx((.2**2 + .8**2)/2, abs=1e-7)
    assert scores["t8_relative_l2"][0] == pytest.approx(.4, abs=1e-7)
    assert scores["t0_relative_l2"][0] == 99.


def test_failure_cannot_disappear_from_endpoint_population_or_json(tmp_path):
    """An earlier NaN invalidates the trajectory even with a finite t8 output."""
    truth = fields()
    prediction = truth.copy()
    prediction[1, 50, 4, 9] = np.nan
    result = evaluation.evaluate_predictions(prediction, truth, np.array([1040, 1041]))
    assert result["summary"]["failed_ids"] == [1041]
    assert result["summary"]["failure_rate"] == .5
    assert result["summary"]["primary_metric"] is None
    assert result["summary"]["selection_metric"] == np.inf
    assert result["first_nonfinite_future_index"].tolist() == [-1, 49]
    assert np.isinf(result["per_trajectory"]["t8_relative_l2"][1])
    evaluation.write_json(tmp_path / "summary.json", result["summary"])
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["selection_metric"] is None
    assert saved["metrics"]["future_mean_relative_l2"]["finite_subset_mean_diagnostic"] == 0.
    explicit = evaluation.evaluate_predictions(truth, truth, np.array([1040, 1041]), solver_failed=[True, False])
    assert explicit["summary"]["failed_ids"] == [1040]


def test_nan_t0_diagnostic_does_not_corrupt_future_seed_curve():
    """Finite future prediction remains scoreable when only t0 reconstruction fails."""
    truth = fields()
    prediction = truth.copy()
    prediction[:, 0] = np.nan
    measured = evaluation.evaluate_predictions(prediction, truth, np.array([9, 3]))
    runs = [{"method": "anie_ns", "seed": seed, "status": "complete", "summary": measured["summary"]}
            for seed in (11, 22, 33)]
    aggregate = evaluation.aggregate_runs(runs)["anie_ns"]
    assert aggregate["status"] == "complete"
    assert aggregate["metrics"]["future_mean_relative_l2"]["mean"] == 0.
    assert np.isnan(aggregate["mean_relative_l2_by_time"][0])
    assert aggregate["mean_relative_l2_by_time"][1:] == [0.] * 150


def test_seed_population_and_pairing_cannot_select_successful_subset():
    """Missing seeds and explicit training failures invalidate the seed average."""
    runs = [run_record("fno2d", seed, value) for seed, value in zip((11, 22, 33), (.1, .2, .3))]
    scores = evaluation.aggregate_runs(runs)["fno2d"]["metrics"]["future_mean_relative_l2"]
    assert scores["mean"] == pytest.approx(.2, abs=1e-7)
    assert scores["training_seed_sample_sd"] == pytest.approx(.1, abs=1e-7)
    missing = evaluation.aggregate_runs(runs[:2])["fno2d"]
    assert missing["missing_training_seeds"] == [33]
    assert missing["metrics"]["future_mean_relative_l2"]["mean"] is None
    failed = dict(method="fno2d", seed=33, status="training_failed", reason="nonfinite gradient",
                  summary={"primary_metric": None})
    incomplete = evaluation.aggregate_runs(runs[:2] + [failed])["fno2d"]
    assert incomplete["failed_training_seeds"] == [33]
    assert incomplete["invalid_seed_runs"] == [33]
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.aggregate_runs(runs + runs[:1])


def test_paired_statistics_align_ids_and_condition_intervals_on_models():
    """ID order differs, but every model contributes and the cohort is never intersected."""
    runs = [run_record("anie_ns", seed, .4, ids=(9, 3)) for seed in (11, 22, 33)]
    runs += [run_record("fno2d", seed, .2, ids=(3, 9)) for seed in (11, 22, 33)]
    paired = evaluation.paired_statistics(runs, bootstrap_replicates=20)[0]
    assert paired["trajectory_ids"] == [3, 9]
    assert paired["mean_delta"] == pytest.approx(.2, abs=1e-7)
    assert paired["bootstrap_95_percent_interval"] == pytest.approx([.2, .2], abs=1e-7)
    assert "conditional" in paired["bootstrap_scope"]
    runs[-1]["per_trajectory"]["trajectory_id"] = np.array([3, 10])
    with pytest.raises(ValueError, match="cohorts differ"):
        evaluation.paired_statistics(runs)


@pytest.mark.parametrize("bad", ("layout", "ids", "truth"))
def test_scoring_rejects_structural_data_mistakes(bad):
    """Wrong axes, repeated identities and invalid reference fields are hard errors."""
    truth = fields()
    prediction, ids = truth.copy(), np.array([3, 9])
    if bad == "layout":
        prediction = prediction.transpose(0, 2, 3, 1)
    elif bad == "ids":
        ids = np.array([3, 3])
    else:
        truth[0, 20, 0, 0] = np.inf
    with pytest.raises(ValueError):
        evaluation.evaluate_predictions(prediction, truth, ids)


def checkpoint(tmp_path, *, stage="formal", seed=11):
    """Only metadata is needed for safety checks; prediction construction is patched."""
    torch = pytest.importorskip("torch")
    payload = dict(model_state={}, model_spec={"method": "fno2d", "history": 1},
                   config={"stage": stage}, stage=stage, epoch=20, seed=seed, val_metric=.2)
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path, payload


def test_development_test_access_is_rejected_before_opening_data(tmp_path):
    """The stage guard runs before even accessing an absent test-container path."""
    path, _ = checkpoint(tmp_path, stage="development", seed=7)
    with pytest.raises(ValueError, match="development candidates"):
        evaluation.evaluate_checkpoints([path], tmp_path / "absent-test.h5", "N64/test", "old_test",
                                        tmp_path / "output", expected_seeds=(7,), plots=False)
    assert not (tmp_path / "output").exists()


def test_formal_test_seed_population_cannot_shrink_to_successful_seeds(tmp_path):
    """An absent real test file stays unopened when the formal population is changed."""
    path, _ = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="prespecified training seeds"):
        evaluation.evaluate_checkpoints([path], tmp_path / "absent-test.h5", "N64/test", "old_test",
                                        tmp_path / "output", expected_seeds=(11,), plots=False)
    assert not (tmp_path / "output").exists()


def test_numeric_commit_precedes_target_read_and_plot(tmp_path, monkeypatch):
    """A synthetic constant task checks end-to-end persistence without real test data."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from training import w1_v2_train
    data = tmp_path / "synthetic_validation.h5"
    with h5py.File(data, "x") as handle:
        handle.attrs["trajectory_id_scheme"] = "canonical"
        node = handle.create_group("validation")
        node.create_dataset("vorticity", shape=(20, 151, 64, 64), dtype="f4", fillvalue=1.)
        node.create_dataset("trajectory_index", data=np.arange(1000, 1020))
        node.create_dataset("time", data=evaluation.LOCAL_TIMES)
        node.create_dataset("absolute_time", data=evaluation.ABSOLUTE_TIMES)
    path, payload = checkpoint(tmp_path)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(evaluation, "_load_checkpoint", lambda path, device: (torch.nn.Identity().eval(), payload))
    monkeypatch.setattr(w1_v2_train, "forward_model", lambda model, initial, spec, normalization, device, training:
                        (initial * 2).unsqueeze(1).expand(-1, 151, -1, -1))
    original_truth = evaluation.ObservationWindow.truth

    def checked_truth(window):
        with h5py.File(output / "raw" / "run_00.h5", "r") as raw:
            assert raw.attrs["status"] == "complete"
            assert raw.attrs["completed_trajectories"] == 20
        assert not (output / "figures").exists()
        return original_truth(window)

    monkeypatch.setattr(evaluation.ObservationWindow, "truth", checked_truth)
    report = evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                              expected_seeds=(11,), bootstrap_replicates=0, plots=True)
    assert report["methods"]["fno2d"]["metrics"]["future_mean_relative_l2"]["mean"] == 1.
    assert (output / "raw" / "truth.h5").is_file()
    assert (output / "metrics" / "run_00.h5").is_file()
    assert (output / "numeric_report.json").is_file()
    assert (output / "figures" / "future_error.svg").is_file()
    before = evaluation.file_sha256(output / "raw" / "run_00.h5")

    def forbidden_prediction(*args, **kwargs):
        raise AssertionError("completed raw predictions must not be recomputed")

    monkeypatch.setattr(w1_v2_train, "forward_model", forbidden_prediction)
    recovered = evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                                 expected_seeds=(11,), bootstrap_replicates=0, plots=True, resume=True)
    assert recovered["methods"]["fno2d"]["metrics"]["future_mean_relative_l2"]["mean"] == 1.
    assert evaluation.file_sha256(output / "raw" / "run_00.h5") == before
    assert not list(output.glob("figures_retry_*"))
    # A final-report interruption must reuse both complete raw and metric files.
    evaluation.preserve_interrupted(output / "numeric_report.json")
    recovered = evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                                 expected_seeds=(11,), bootstrap_replicates=0, plots=False, resume=True)
    assert recovered["methods"]["fno2d"]["metrics"]["future_mean_relative_l2"]["mean"] == 1.
    assert evaluation.file_sha256(output / "raw" / "run_00.h5") == before


def test_partial_resume_only_predicts_remaining_ids_and_retains_actual_cost(tmp_path, monkeypatch):
    """An interrupted third batch preserves eight outputs, then predicts only twelve."""
    torch = pytest.importorskip("torch")
    from training import w1_v2_train
    data = tmp_path / "synthetic.h5"
    with h5py.File(data, "x") as handle:
        handle.attrs["trajectory_id_scheme"] = "canonical"
        node = handle.create_group("validation")
        node.create_dataset("vorticity", shape=(20, 151, 64, 64), dtype="f4", fillvalue=1.)
        node.create_dataset("trajectory_index", data=np.arange(1000, 1020))
        node.create_dataset("absolute_time", data=evaluation.ABSOLUTE_TIMES)
    path, payload = checkpoint(tmp_path)
    output = tmp_path / "interrupted"
    monkeypatch.setattr(evaluation, "_load_checkpoint", lambda path, device: (torch.nn.Identity().eval(), payload))
    calls = []

    def interruption(model, initial, *args, **kwargs):
        calls.append(len(initial))
        if len(calls) == 3:
            raise RuntimeError("synthetic interruption")
        return (initial * 2).unsqueeze(1).expand(-1, 151, -1, -1)

    monkeypatch.setattr(w1_v2_train, "forward_model", interruption)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                          expected_seeds=(11,), bootstrap_replicates=0, plots=False)
    with h5py.File(output / "raw" / "run_00.h5", "r") as raw:
        assert raw.attrs["completed_trajectories"] == 8
        committed = raw["prediction"][:8]
        prior_cost = json.loads(raw.attrs["inference_cost_json"])
    calls.clear()

    def continuation(model, initial, *args, **kwargs):
        calls.append(len(initial))
        return (initial * 2).unsqueeze(1).expand(-1, 151, -1, -1)

    monkeypatch.setattr(w1_v2_train, "forward_model", continuation)
    report = evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                              expected_seeds=(11,), bootstrap_replicates=0, plots=False, resume=True)
    assert calls == [4, 4, 4]
    assert (output / "raw" / "run_00.interrupted_000.h5").is_file()
    with h5py.File(output / "raw" / "run_00.h5", "r") as raw:
        assert np.array_equal(raw["prediction"][:8], committed)
        cost = json.loads(raw.attrs["inference_cost_json"])
        assert len(cost["inference_segments"]) == 2
        assert cost["inference_seconds"] >= prior_cost["inference_seconds"]
        assert cost["inference_measurement_resumed"] is True
    assert report["methods"]["fno2d"]["metrics"]["future_mean_relative_l2"]["mean"] == 1.


def test_prespecified_keyframes_use_seed11_and_saved_numbers(tmp_path):
    """The figure uses the predeclared first confirmation ID and actual raw prediction."""
    pytest.importorskip("matplotlib")
    output = tmp_path / "figure"
    (output / "raw").mkdir(parents=True)
    truth = fields(1)
    for name, key, value in (("truth", "vorticity", truth), ("run_00", "prediction", truth * 1.4)):
        with h5py.File(output / "raw" / (name + ".h5"), "x") as handle:
            handle.create_dataset("trajectory_index", data=np.array([100000]))
            handle.create_dataset(key, data=value)
    records = [run_record("anie_ns", seed, .4, ids=(100000,)) for seed in (11, 22, 33)]
    records[0]["raw_prediction_file"] = "raw/run_00.h5"
    evaluation.write_json(output / "numeric_report.json", {"cohort": "confirmation", "methods": evaluation.aggregate_runs(records)})
    evaluation.draw_figures(output)
    provenance = json.loads((output / "figures" / "provenance.json").read_text())
    assert provenance["keyframes"]["trajectory_id"] == 100000
    assert provenance["keyframes"]["training_seed"] == 11
    assert provenance["keyframes"]["absolute_time"] == [5.5, 6., 7., 8.]
    assert (output / "figures" / "fixed_id_keyframes.svg").is_file()


def test_confirmation_plan_and_lock_are_prespecified(tmp_path):
    """The independent cohort cannot be generated before both locks are concrete."""
    plan = generation.confirmation_plan()
    assert plan["population"] == 180
    assert plan["seed"] == 2026100302
    assert plan["trajectory_ids"] == list(range(100000, 100180))
    assert plan["uncompressed_field_bytes"] == 445317120
    assert np.allclose(plan["absolute_time"], evaluation.ABSOLUTE_TIMES)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("fixed t5 single-field protocol", encoding="utf-8")
    lock = tmp_path / "lock.json"
    value = dict(protocol_locked=True, development_locked=False, protocol_file=str(protocol.resolve()),
                 selected_configurations={"anie_ns": {"learning_rate": 1e-4}})
    lock.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        generation.locked_configuration(lock)
    value["development_locked"] = True
    lock.write_text(json.dumps(value), encoding="utf-8")
    assert generation.locked_configuration(lock)["lock_snapshot"] == value
    assert np.array_equal(generation.draw_parameters(2, generation.SEED),
                          generation.draw_parameters(180, generation.SEED)[:2])
    assert not np.array_equal(generation.draw_parameters(2, generation.SEED),
                              generation.draw_parameters(2, 2026080801))


def test_confirmation_solver_short_cpu_contract():
    """Four real ETDRK4 steps verify field axes, finite values and source metadata."""
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output, metadata = generation.integrate_batch(generation.draw_parameters(1, generation.SEED),
                                                       saved_steps=np.array([0, 4]), device="cpu")
    finally:
        torch.set_num_threads(previous)
    assert output.shape == (1, 2, 64, 64)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.abs(output.mean(axis=(-2, -1))).max() < 1e-5
    assert not np.array_equal(output[:, 0], output[:, 1])
    assert metadata["internal_dt"] == .005
    assert metadata["all_native_dft_coefficients_retained"] is True
    assert metadata["padding_grid"] == 99


def sparse_payload(method="fno2d", *, dt=.1):
    """A complete synthetic training receipt, without opening real training/test data."""
    contract = evaluation.sparse_training_contract(evaluation.evaluation_grid(dt=dt))
    count, stride = contract["training_count"], contract["time_stride"]
    grid = [.02*i for i in range(0, 151, stride)]
    points = len(grid)
    binding = dict(points=points, future_steps=points-1, query_times=grid, observation_dt_seconds=dt,
                   training_count=count, subset_seed=2026100304, time_stride=stride,
                   source_frame_indices=list(range(250, 401, stride)), manifest_sha256="c"*64,
                   training=dict(ids=np.random.default_rng(2026100304).permutation(1000)[:count].tolist(), observations_sha256="a"*64),
                   validation=dict(ids=list(range(1000, 1020)), observations_sha256="b"*64))
    spec = dict(method=method, history=1, points=points, future_steps=points-1, query_times=grid, observation_dt=dt)
    if method == "GIFT":
        spec.update(family="gift", duration=3., integration_dt=.02, step=.02, save_interval=dt)
    return dict(model_state={}, model_spec=spec, config=dict(stage="formal"), stage="formal", epoch=20,
                seed=11, val_metric=.2, data_binding=binding)


def sparse_checkpoint(tmp_path, method="fno2d", *, dt=.1):
    """Persist the complete sparse receipt for the real preflight path."""
    torch = pytest.importorskip("torch")
    path = tmp_path / (method + ".pt")
    payload = sparse_payload(method, dt=dt)
    torch.save(payload, path)
    return path, payload


def synthetic_validation(path, *, points=151, start=5.):
    """A fill-value dataset avoids writing a large solution while retaining real HDF axes."""
    with h5py.File(path, "x") as handle:
        handle.attrs["trajectory_id_scheme"] = "canonical"
        node = handle.create_group("validation")
        node.create_dataset("vorticity", shape=(20, points, 64, 64), dtype="f4", fillvalue=1.)
        node.create_dataset("trajectory_index", data=np.arange(1000, 1020))
        node.create_dataset("absolute_time", data=start + np.arange(points)*.02)


@pytest.mark.parametrize("dt", (.1, .04))
def test_sparse_scores_real_futures_and_preserves_failures(dt):
    """The sparse primary denominator is30/75, and t0 reconstruction stays diagnostic."""
    grid = evaluation.evaluation_grid(dt=dt)
    points = len(grid)
    truth = fields(points=points)
    prediction = truth * 1.2
    prediction[:, 0] = 100.
    measured = evaluation.evaluate_predictions(prediction, truth, np.array([1040, 1041]), query_times=grid)
    assert measured["summary"]["primary_metric"] == pytest.approx(.2, abs=1e-7)
    assert measured["summary"]["future_steps"] == points-1
    assert measured["summary"]["time_points"] == points
    assert "all151_mse_diagnostic" not in measured["summary"]["metrics"]
    assert measured["summary"]["metrics"]["all_points_mse_diagnostic"]["mean"] == pytest.approx(
        (99.**2+(points-1)*.2**2)/points, abs=1e-6)
    prediction[1, 7, 0, 0] = np.nan
    failed = evaluation.evaluate_predictions(prediction, truth, np.array([1040, 1041]), query_times=grid)
    assert failed["summary"]["failed_ids"] == [1041]
    assert failed["summary"]["primary_metric"] is None
    assert np.isinf(failed["per_trajectory"]["t8_relative_l2"][1])
    with pytest.raises(ValueError, match="151"):
        evaluation.evaluate_predictions(prediction, truth, np.array([1040, 1041]))


@pytest.mark.parametrize("dt", (.1, .04))
@pytest.mark.parametrize("points,start", ((151, 5.), (501, 0.)))
def test_sparse_window_selects_actual_dense_nodes_without_interpolation(tmp_path, points, start, dt):
    """Source151/501 labels select real31/76 nodes, with no synthesized fields."""
    grid = evaluation.evaluation_grid(dt=dt)
    stride = int(round(dt/.02))
    expected = np.arange(250, 401, stride) if points == 501 else np.arange(0, 151, stride)
    path = tmp_path / "synthetic.h5"
    synthetic_validation(path, points=points, start=start)
    window = evaluation.observation_window(path, "validation", "validation", query_times=grid)
    assert np.array_equal(window.columns, expected)
    assert window.truth().shape == (20, len(grid), 64, 64)
    assert window.initial(0, 2).shape == (2, 64, 64)
    with h5py.File(path, "r+") as handle:
        handle["validation/absolute_time"][int(expected[8])] += .001
    with pytest.raises(ValueError, match="interpolation"):
        evaluation.observation_window(path, "validation", "validation", query_times=grid)


@pytest.mark.parametrize("bad", ("grid", "hash", "ids", "subset_seed", "frames", "manifest", "dt",
                                 "gift_step", "gift_saving", "fno3d_normalizer"))
@pytest.mark.parametrize("dt", (.1, .04))
def test_sparse_checkpoint_preflight_rejects_semantic_mismatch(bad, dt):
    """Shape-only sparse output cannot bypass training grid, cohort or native-step constraints."""
    payload = sparse_payload("GIFT" if bad.startswith("gift_") else "fno2d", dt=dt)
    if bad == "grid":
        payload["model_spec"]["query_times"] = evaluation.LOCAL_TIMES.tolist()
    elif bad == "hash":
        payload["data_binding"]["validation"].pop("observations_sha256")
    elif bad == "ids":
        payload["data_binding"]["training"]["ids"][-1] = 1040
    elif bad == "subset_seed":
        payload["data_binding"]["subset_seed"] = 17
    elif bad == "frames":
        payload["data_binding"]["source_frame_indices"][-1] = 399
    elif bad == "manifest":
        payload["data_binding"]["manifest_sha256"] = "invalid"
    elif bad == "dt":
        payload["data_binding"]["observation_dt_seconds"] = .02
    elif bad == "gift_step":
        payload["model_spec"]["step"] = .1
    elif bad == "gift_saving":
        payload["model_spec"]["save_interval"] = .02
    else:
        payload["model_spec"]["method"] = "fno3d"
        payload["normalization"] = {prefix+"_"+suffix: np.ones((64, 64, 1 if prefix == "input" else 150))
                                     for prefix in ("input", "output") for suffix in ("mean", "std")}
    with pytest.raises(ValueError):
        evaluation.validate_checkpoint_grid(payload, evaluation.evaluation_grid(dt=dt))


@pytest.mark.parametrize("dt", (.1, .04))
def test_sparse_common_binding_ignores_gift_extras_but_checks_observations(dt):
    """GIFT derivative anchors do not change the common supervised trajectory receipt."""
    left, right = sparse_payload(dt=dt), sparse_payload("GIFT", dt=dt)
    right["data_binding"].update(observations_sha256="c"*64, generator_derivative_centers=list(range(2, 29)))
    right["data_binding"]["training"]["native_extra"] = "irrelevant sampler metadata"
    assert evaluation.common_data_identity(left) == evaluation.common_data_identity(right)
    right["data_binding"]["training"]["observations_sha256"] = "d"*64
    assert evaluation.common_data_identity(left) != evaluation.common_data_identity(right)


@pytest.mark.parametrize("dt", (.1, .04))
def test_sparse_numeric_commit_and_resume_are_grid_bound(tmp_path, monkeypatch, dt):
    """The real sparse scoring/persistence path reuses completed raw outputs on resume."""
    torch = pytest.importorskip("torch")
    from training import w1_v2_train
    data = tmp_path / "synthetic.h5"
    synthetic_validation(data)
    grid = evaluation.evaluation_grid(dt=dt)
    points = len(grid)
    path, payload = sparse_checkpoint(tmp_path, dt=dt)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(evaluation, "_load_checkpoint", lambda path, device: (torch.nn.Identity().eval(), payload))
    monkeypatch.setattr(w1_v2_train, "forward_model", lambda model, initial, *args, **kwargs:
                        (initial*2).unsqueeze(1).expand(-1, points, -1, -1))
    query = [.02*i for i in range(0, 151, int(round(dt/.02)))]
    report = evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                              expected_seeds=(11,), bootstrap_replicates=0, plots=False, query_times=query)
    assert report["future_steps"] == points-1
    assert report["methods"]["fno2d"]["metrics"]["future_mean_relative_l2"]["mean"] == 1.
    with h5py.File(output / "raw" / "run_00.h5", "r") as handle:
        assert handle["prediction"].shape == (20, points, 64, 64)
        assert np.array_equal(handle["time"][:], grid)
        assert handle.attrs["future_steps"] == points-1
    with h5py.File(output / "metrics" / "run_00.h5", "r") as handle:
        assert handle["relative_l2_by_time"].shape == (20, points)
    before = evaluation.file_sha256(output / "raw" / "run_00.h5")

    def forbidden(*args, **kwargs):
        raise AssertionError("completed sparse outputs must not be regenerated")

    monkeypatch.setattr(w1_v2_train, "forward_model", forbidden)
    recovered = evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                                 expected_seeds=(11,), bootstrap_replicates=0, plots=False, resume=True,
                                                 query_times=np.arange(points)*dt)
    assert recovered["future_steps"] == points-1
    assert evaluation.file_sha256(output / "raw" / "run_00.h5") == before
    with pytest.raises(ValueError, match="grid"):
        evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                          expected_seeds=(11,), plots=False, resume=True)


@pytest.mark.parametrize("dt", (.1, .04))
def test_sparse_gift_uses_declared_saving_with_point_zero_two_integrator(tmp_path, monkeypatch, dt):
    """The native GIFT call emits real sparse fields and retains explicit trajectory failures."""
    from types import SimpleNamespace
    from experiments.formal._shared import gift_runtime
    data = tmp_path / "synthetic.h5"
    synthetic_validation(data)
    grid = evaluation.evaluation_grid(dt=dt)
    path, payload = sparse_checkpoint(tmp_path, "GIFT", dt=dt)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(evaluation, "_load_checkpoint", lambda path, device: ((object(), object()), payload))
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        initial = kwargs["initial_state"]
        assert initial.shape == (20, 64, 64)
        assert kwargs["step"] == .02
        assert kwargs["save_interval"] == dt
        assert kwargs["duration"] == 3.
        assert "truth" not in kwargs
        return SimpleNamespace(prediction=np.repeat(initial[:, None], len(grid), axis=1),
                               failure_step=np.array([7]+[-1]*19), failure_reason=("synthetic finite failure",)+("",)*19)

    monkeypatch.setattr(gift_runtime, "rollout_gift", native)
    report = evaluation.evaluate_checkpoints([path], data, "validation", "validation", output,
                                              expected_seeds=(11,), bootstrap_replicates=0, plots=False,
                                              query_times=grid)
    assert len(calls) == 1
    seed = report["methods"]["GIFT"]["seed_records"][0]
    assert seed["summary"]["failed_ids"] == [1000]
    assert seed["summary"]["failure_rate"] == .05
    assert report["methods"]["GIFT"]["metrics"]["future_mean_relative_l2"]["mean"] is None


@pytest.mark.parametrize("dt", (.1, .04))
def test_sparse_seed_statistics_and_keyframes_use_the_same_grid(tmp_path, dt):
    """Three independent training seeds retain sampleSD; keyframe labels match actual nodes."""
    pytest.importorskip("matplotlib")
    grid = evaluation.evaluation_grid(dt=dt)
    truth = fields(1, points=len(grid))
    records = []
    for seed, value in zip((11, 22, 33), (.1, .2, .3)):
        measured = evaluation.evaluate_predictions(truth*(1+value), truth, np.array([100000]), query_times=grid)
        records.append(dict(method="anie_ns", seed=seed, stage="formal", status="complete", **measured))
    groups = evaluation.aggregate_runs(records, query_times=grid)
    stats = groups["anie_ns"]["metrics"]["future_mean_relative_l2"]
    assert stats["mean"] == pytest.approx(.2, abs=1e-7)
    assert stats["training_seed_sample_sd"] == pytest.approx(.1, abs=1e-7)
    assert len(groups["anie_ns"]["mean_relative_l2_by_time"]) == len(grid)
    with pytest.raises(ValueError, match="grid"):
        evaluation.aggregate_runs(records, query_times=evaluation.LOCAL_TIMES)
    output = tmp_path / "figure"
    (output / "raw").mkdir(parents=True)
    for name, key, value in (("truth", "vorticity", truth), ("run_00", "prediction", truth*1.1)):
        with h5py.File(output / "raw" / (name+".h5"), "x") as handle:
            handle.create_dataset("trajectory_index", data=np.array([100000]))
            handle.create_dataset("time", data=grid)
            handle.create_dataset("absolute_time", data=5+grid)
            handle.create_dataset(key, data=value)
    records[0]["raw_prediction_file"] = "raw/run_00.h5"
    evaluation.write_json(output / "numeric_report.json", dict(cohort="confirmation", absolute_times=5+grid,
                                                                methods=evaluation.aggregate_runs(records, query_times=grid)))
    evaluation.draw_figures(output)
    saved = json.loads((output / "figures" / "provenance.json").read_text())
    assert saved["keyframes"]["absolute_time"] == [5.48 if dt == .04 else 5.5, 6., 7., 8.]
    assert saved["keyframes"]["source_time_indices"] == ([12, 25, 50, 75] if dt == .04 else [5, 10, 20, 30])
    assert saved["keyframes"]["trajectory_id"] == 100000
    assert saved["keyframes"]["training_seed"] == 11


@pytest.mark.parametrize("dt", (.1, .04))
def test_sparse_confirmation_lock_keeps_dense_reference_and_declares_exact_scoring(tmp_path, dt):
    """A sparse evaluation lock changes scored nodes, not independent physics or draws."""
    protocol = tmp_path / "protocol.md"
    grid = evaluation.evaluation_grid(dt=dt)
    contract = evaluation.sparse_training_contract(grid)
    stride = contract["time_stride"]
    protocol.write_text("locked sparse protocol", encoding="utf-8")
    value = dict(protocol_locked=True, development_locked=True, protocol_file=str(protocol.resolve()),
                 selected_configurations={"anie_ns": {"learning_rate": 1e-4}},
                 evaluation_query_times=[.02*i for i in range(0, 151, stride)], evaluation_dt=dt,
                 observation_points=len(grid), future_steps=len(grid)-1,
                 training_count=contract["training_count"], training_subset_seed=2026100304)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    lock = generation.locked_configuration(path)
    plan = generation.confirmation_plan(lock["evaluation_query_times"])
    assert plan["vorticity_shape"] == [180, 151, 64, 64]
    assert plan["save_interval"] == .02
    assert plan["internal_dt"] == .005
    assert plan["evaluation_dt"] == dt
    assert plan["evaluation_points"] == len(grid)
    assert plan["evaluation_future_steps"] == len(grid)-1
    assert plan["trajectory_ids"] == generation.confirmation_plan()["trajectory_ids"]
    assert np.allclose(np.asarray(plan["absolute_time"])[::stride], plan["evaluation_absolute_time"], rtol=0, atol=1e-12)
    value["observation_points"] = 151
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="observation_points"):
        generation.locked_configuration(path)


@pytest.mark.parametrize("cohort,first", (("old_test", 1040), ("confirmation", 100000)))
def test_sparse76_cohorts_keep_separate_complete180_populations(tmp_path, cohort, first):
    """Synthetic metadata checks distinct fixed180 cohorts without reading real test solutions."""
    path = tmp_path / (cohort+".h5")
    with h5py.File(path, "x") as handle:
        handle.attrs.update(trajectory_id_scheme="canonical", complete=True)
        node = handle.create_group("test")
        node.create_dataset("trajectory_index", data=np.arange(first, first+180))
        node.create_dataset("absolute_time", data=evaluation.ABSOLUTE_TIMES)
        node.create_dataset("vorticity", shape=(180, 151, 64, 64), dtype="f4", fillvalue=1.)
    window = evaluation.observation_window(path, "test", cohort, query_times=evaluation.evaluation_grid(dt=.04))
    assert window.trajectory_ids.tolist() == list(range(first, first+180))
    assert window.columns.tolist() == list(range(0, 151, 2))
    assert window.initial(0, 1).shape == (1, 64, 64)
    with h5py.File(path, "r+") as handle:
        handle["test/trajectory_index"][-1] += 1
    with pytest.raises(ValueError, match="complete declared"):
        evaluation.observation_window(path, "test", cohort, query_times=evaluation.evaluation_grid(dt=.04))


def test_sparse76_all_failed_seeds_keep_declared_grid_and_cannot_pair_with31():
    """A no-checkpoint failure report still labels75 futures; mixed grids never form a ranking."""
    grid = evaluation.evaluation_grid(dt=.04)
    failed = [dict(method="anie_ns", seed=seed, status="training_failed", summary={"primary_metric": None})
              for seed in (11, 22, 33)]
    group = evaluation.aggregate_runs(failed, query_times=grid)["anie_ns"]
    assert group["time_points"] == 76
    assert group["future_steps"] == 75
    assert group["failed_training_seeds"] == [11, 22, 33]
    assert group["metrics"]["future_mean_relative_l2"]["mean"] is None
    records = []
    for method, dt in (("anie_ns", .04), ("fno2d", .1)):
        times = evaluation.evaluation_grid(dt=dt)
        truth = fields(1, points=len(times))
        result = evaluation.evaluate_predictions(truth*1.2, truth, np.array([100000]), query_times=times)
        for seed in (11, 22, 33):
            records.append(dict(method=method, seed=seed, status="complete", **result))
    with pytest.raises(ValueError, match="different physical prediction grids"):
        evaluation.paired_statistics(records, bootstrap_replicates=0)
