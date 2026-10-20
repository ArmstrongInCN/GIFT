"""CPU-only scientific boundary tests for the native single-run W2 extension."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from experiments.additional import w2_single_evaluate as evaluation
from scripts import generate_w2_single_confirmation as generation


def data_file(path, *, cohort="confirmation", grids=(64, 96, 128), perturb_grid=None, dense=True):
    """Write same-parameter paired references; chunkless fill avoids a physical simulation."""
    ids = np.arange(100000, 100180) if cohort == "confirmation" else np.arange(1040, 1220)
    times = 5+np.arange(51)*.02 if dense else 5+evaluation.QUERY_TIMES
    with h5py.File(path, "x") as handle:
        handle.attrs.update(complete=True, trajectory_id_scheme="canonical")
        for grid in grids:
            node = handle.create_group(f"N{grid}/test")
            node.create_dataset("trajectory_index", data=ids)
            node.create_dataset("absolute_time", data=times)
            parameters = np.ones((180, 4, 4), np.float64)
            if grid == perturb_grid:
                parameters[0, 0, 0] += 1e-12
            node.create_dataset("initial_parameters", data=parameters)
            node.create_dataset("vorticity", shape=(180, len(times), grid, grid), dtype="f4", fillvalue=1.)


def test_paired_window_reads_actual_dt04_and_only_one_initial_field(tmp_path):
    """Exact .02 nodes are selected at stride2, without inventing targets or history."""
    path = tmp_path / "paired.h5"
    data_file(path)
    windows = evaluation.paired_windows(path, None, "confirmation")
    assert windows[128].initial(0, 2).shape == (2, 128, 128)
    assert windows[96].truth(0, 1).shape == (1, 26, 96, 96)
    assert np.array_equal(windows[64].columns, np.arange(0, 51, 2))
    assert np.array_equal(windows[128].parameters, windows[64].parameters)


def test_pairing_rejects_replaced_or_perturbed_initial_parameters(tmp_path):
    """Cross-resolution identity means exact common continuous parameters, not just IDs."""
    path = tmp_path / "paired.h5"
    data_file(path, perturb_grid=96)
    with pytest.raises(ValueError, match="exact saved parameters"):
        evaluation.paired_windows(path, None, "confirmation")


def test_observations_reject_missing_saved_nodes(tmp_path):
    """Sparse .1-second references cannot be interpolated into the .04 W2 protocol."""
    path = tmp_path / "paired.h5"
    data_file(path)
    with h5py.File(path, "r+") as handle:
        handle["N96/test/absolute_time"][4] += .001
    with pytest.raises(ValueError, match="all actual"):
        evaluation.observation_window(path, "N96/test", 96, "confirmation")


@pytest.mark.parametrize("grid", (96, 128))
def test_periodic_lift_preserves_physical_amplitude_and_nyquist_rule(grid):
    """A known Fourier wave keeps its physical amplitude across unnormalised FFT grids."""
    axis = torch.arange(64, dtype=torch.float64)*2*torch.pi/64
    field = 3. + 2.*torch.cos(5*axis[:, None]) + .5*torch.sin(7*axis[None, :])
    lifted = evaluation.periodic_resize(field, grid)
    target_axis = torch.arange(grid, dtype=torch.float64)*2*torch.pi/grid
    expected = 3.+2.*torch.cos(5*target_axis[:, None])+.5*torch.sin(7*target_axis[None, :])
    torch.testing.assert_close(lifted, expected, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(evaluation.periodic_resize(lifted, 64), field, atol=3e-13, rtol=3e-13)
    nyquist = torch.cos(32*axis)[None, :].expand(64, 64)
    target_nyquist = torch.cos(32*target_axis)[None, :].expand(grid, grid)
    torch.testing.assert_close(evaluation.periodic_resize(nyquist, grid), target_nyquist, atol=3e-13, rtol=3e-13)


def normalizer():
    """Frozen training-only statistics have one input and the original75 future channels."""
    return dict(kind="paper_UnitGaussianNormalizer", validation_or_test_used=False, eps=1e-5,
                input_mean=torch.ones(64, 64, 1), input_std=torch.ones(64, 64, 1)*2,
                output_mean=torch.arange(75.).reshape(1, 1, 75).expand(64, 64, 75).clone(),
                output_std=torch.ones(64, 64, 75)*3)


def test_fno3d_lift_keeps_full_training_time_axis_without_target_statistics():
    """Space changes; all75 temporal statistics and train-only evidence remain intact."""
    source = normalizer()
    lifted, audit = evaluation.lift_fno3d_normalization(source, 96)
    assert lifted["input_mean"].shape == (96, 96, 1)
    assert lifted["output_mean"].shape == (96, 96, 75)
    torch.testing.assert_close(lifted["output_mean"][0, 0], torch.arange(75.))
    assert audit["output_steps"] == 75 and audit["test_statistics_used"] is False
    assert audit["temporal_interpolation"] is False
    source["validation_or_test_used"] = True
    with pytest.raises(ValueError, match="training-only"):
        evaluation.lift_fno3d_normalization(source, 96)


def test_joint_fno3d_predicts_original75_then_scores25(monkeypatch):
    """Changing to26 time tokens would alter the spacetime operator; only its output is sliced."""
    from training import w1_single_train
    calls = []
    def forward(model, initial, spec, normalization, **kwargs):
        calls.append(kwargs)
        return torch.arange(76.).reshape(1, 76, 1, 1).expand(2, 76, 96, 96)
    monkeypatch.setattr(w1_single_train, "forward_model", forward)
    value = evaluation.forward_predictor("fno3d", None, torch.ones(2, 96, 96), dict(model_spec={}), None, "cpu")
    assert value.shape == (2, 26, 96, 96)
    assert value[0, -1, 0, 0] == 25.
    assert calls[0]["horizon"] is None


def test_future25_primary_excludes_initial_reconstruction_and_retains_failures():
    """A poor t0 alone cannot inflate forecast error; one failed future invalidates full mean."""
    rng = np.random.default_rng(17)
    truth = rng.normal(size=(3, 26, 64, 64)).astype(np.float32)
    prediction = truth.copy()
    prediction[:, 0] *= 4
    measured = evaluation.measure_predictions(prediction, truth, np.arange(3))
    np.testing.assert_array_equal(measured["per_trajectory"]["future_mean_relative_l2"], np.zeros(3))
    np.testing.assert_allclose(measured["per_trajectory"]["t0_relative_l2"], 3.)
    prediction[1, 5, 0, 0] = np.nan
    failed = evaluation.measure_predictions(prediction, truth, np.arange(3))
    assert failed["failed"].tolist() == [False, True, False]
    summary = evaluation.shared._summary(failed["per_trajectory"]["future_mean_relative_l2"])
    assert summary["population"] == 3 and summary["mean"] is None
    assert summary["finite_count"] == 2


@pytest.mark.parametrize("method", ("uno", "unet"))
def test_native_uno_and_unet_accept_all_three_grids_without_new_weights(method):
    """Public layers are applied to larger incoming grids; no interpolation or refit is needed."""
    configured = os.environ.get("GIFT_EXTERNAL_ROOT")
    if not configured:
        pytest.skip("set GIFT_EXTERNAL_ROOT to the public pinned dependencies")
    root = Path(configured).expanduser().resolve()
    if not (root / "uno/navier_stokes_uno2d.py").is_file() or not (root / "unet/Baselines/U-net.py").is_file():
        pytest.skip("public pinned dependencies not installed")
    # Public U-NO uses absolute module names. A fresh process prevents another
    # test's differently configured upstream checkout from occupying those names.
    program = """import sys
import torch
from adapters.w1_extended import build_network, forward_model
torch.set_num_threads(1)
spec = dict(method=sys.argv[1], history=1, source_root=sys.argv[2], future_steps=75)
model = build_network(spec, 'cpu').eval()
shapes = {key: tuple(tensor.shape) for key, tensor in model.state_dict().items()}
with torch.inference_mode():
    for grid in (64, 96, 128):
        value = forward_model(model, torch.ones(1, grid, grid), spec, steps=1)
        assert value.shape == (1, 2, grid, grid)
        assert torch.isfinite(value).all()
        assert {key: tuple(tensor.shape) for key, tensor in model.state_dict().items()} == shapes
"""
    result = subprocess.run([sys.executable, "-B", "-c", program, method, str(root)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=180)
    assert result.returncode == 0, result.stderr


def test_w1_prefix_reuse_ignores_failure_after_w2_window(tmp_path, monkeypatch):
    """A saved late t8 failure is outside W2 and does not discard a finite t5..6 prefix."""
    source, paired = tmp_path / "source.h5", tmp_path / "paired.h5"
    data_file(paired, grids=(64,))
    window = evaluation.observation_window(paired, "N64/test", 64, "confirmation")
    with h5py.File(source, "x") as handle:
        handle.create_dataset("prediction", shape=(180, 76, 64, 64), dtype="f4", fillvalue=1.)
        handle.create_dataset("solver_failed", data=np.ones(180, bool))
        handle["prediction"][0, 75] = np.nan
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"synthetic finite-state receipt")
    output = tmp_path / "prefix.h5"
    cost = evaluation.predict_checkpoint("fno2d", checkpoint, dict(seed=0), window, output,
                                         device="cpu", batch_size=20, reuse_source=source)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["status"] == "complete"
        assert not handle["solver_failed"][:].any()
        assert np.isfinite(handle["prediction"][:]).all()
    assert cost["inference_seconds"] == 0
    assert cost["adaptation_audit"]["no_additional_model_forward"] is True


def test_partial_w2_resume_preserves_completed_prefix_and_prior_cost(tmp_path, monkeypatch):
    """Resume reads committed outputs/cost and saves the incomplete attempt separately."""
    paired, source = tmp_path / "paired.h5", tmp_path / "source.h5"
    data_file(paired, grids=(64,))
    window = evaluation.observation_window(paired, "N64/test", 64, "confirmation")
    with h5py.File(source, "x") as handle:
        handle.create_dataset("prediction", shape=(180, 76, 64, 64), dtype="f4", fillvalue=1.)
    checkpoint, raw_path = tmp_path / "weights.pt", tmp_path / "prediction.h5"
    checkpoint.write_bytes(b"fixed synthetic weights")
    evaluation.predict_checkpoint("fno2d", checkpoint, dict(seed=0), window, raw_path,
                                  device="cpu", batch_size=20, reuse_source=source)
    with h5py.File(raw_path, "r+") as handle:
        handle.attrs.update(status="incomplete", completed_trajectories=20)
        handle["prediction"][:20] = 11.
        cost = json.loads(handle.attrs["inference_cost_json"])
        cost["inference_seconds"] = 2.5
        handle.attrs["inference_cost_json"] = json.dumps(cost)
    previous = evaluation.shared.preserve_interrupted(raw_path)
    cost = evaluation.predict_checkpoint("fno2d", checkpoint, dict(seed=0), window, raw_path,
                                         device="cpu", batch_size=20, reuse_source=source, previous_raw=previous)
    assert previous.exists() and raw_path.exists()
    with h5py.File(raw_path, "r") as handle:
        assert np.all(handle["prediction"][:20] == 11.)
        assert np.all(handle["prediction"][20:] == 1.)
        assert handle.attrs["completed_trajectories"] == 180
    assert cost["inference_seconds"] == 2.5
    # A crash after the final flush but before status=complete must not trigger
    # an unnecessary model forward or permanently block this own-run resume.
    with h5py.File(raw_path, "r+") as handle:
        handle.attrs["status"] = "incomplete"
    previous_complete = evaluation.shared.preserve_interrupted(raw_path)
    monkeypatch.setattr(evaluation, "load_predictor", lambda *args: pytest.fail("flushed180 needs no new model"))
    recovered = evaluation.predict_checkpoint("fno2d", checkpoint, dict(seed=0), window, raw_path,
                                               device="cpu", batch_size=20, previous_raw=previous_complete)
    with h5py.File(raw_path, "r") as handle:
        assert handle.attrs["status"] == "complete"
        assert handle.attrs["completed_trajectories"] == 180
        assert np.all(handle["prediction"][:20] == 11.)
    assert recovered["inference_seconds"] == 2.5


def test_w2_method_scope_excludes_anie_and_requires_explicit_budget_label():
    """User's exclusion is structural and GIFT-N50 cannot silently replace GIFT."""
    assert "anie_ns" not in evaluation.METHODS and len(evaluation.METHODS) == 6
    assert evaluation.labelled_paths(["GIFT=x.pt", "GIFT-N50=y.pt"]) == {"GIFT": Path("x.pt"), "GIFT-N50": Path("y.pt")}
    with pytest.raises(ValueError):
        evaluation.labelled_paths(["anie_ns=best.pt"])
    with pytest.raises(ValueError):
        evaluation.labelled_paths(["GIFT=x.pt", "GIFT=y.pt"])


def test_high_resolution_generation_keeps_exact_parameters_and_original_solver():
    """A tiny CPU numerical test checks the new grid without producing a test cohort."""
    from scripts.generate_data import draw_parameters
    from even_full_spectrum_ns import build_initial_hat
    parameters = draw_parameters(1, 17)
    values, metadata = generation.integrate_batch(parameters, 96, saved_steps=np.asarray([0, 1], np.int64), device="cpu")
    initial = torch.fft.ifft2(build_initial_hat(parameters, grid=96, device="cpu")).real.numpy()
    np.testing.assert_array_equal(values[:, 0], initial)
    assert values.shape == (1, 2, 96, 96) and np.isfinite(values).all()
    assert metadata["state_mask_applied"] is False and metadata["internal_dt"] == .005
    assert metadata["padding_grid"] == 147


def test_generation_plan_declares_existing_not_new_blind_confirmation():
    """The extension explicitly discloses previously known four-model results."""
    plan = generation.generation_plan(dict(n64_source_file="unchanged.h5"))
    assert plan["generated_grids"] == [96, 128]
    assert "not a fresh blind" in plan["interpretation"]
    assert plan["models_or_test_statistics_used"] is False
    assert len(plan["absolute_times"]) == 51
    assert plan["population_per_grid"] == 180


def test_all_failed_slots_complete_reports_and_resume_without_retraining(tmp_path, monkeypatch):
    """Actual six-slot orchestration persists each failure on each grid and completes artifacts."""
    from experiments.additional import w1_single_evaluate
    monkeypatch.setattr(w1_single_evaluate, "protocol_lock", lambda path: {"lock_snapshot": {}})
    base, extension, paired = tmp_path / "base.json", tmp_path / "extension.json", tmp_path / "paired.h5"
    base.write_text("{}", encoding="utf-8")
    extension.write_text(json.dumps(dict(schema="gift.w1w2-native.extension-lock.v1", profile="single_native76_n300",
                                        configurations_locked=True)), encoding="utf-8")
    data_file(paired)
    failed = {}
    for label in evaluation.METHODS:
        path = tmp_path / (label+"_failure.json")
        path.write_text(json.dumps(dict(method="GIFT" if label == "GIFT-N50" else label,
                                       seed=20260820 if label in ("GIFT", "GIFT-N50") else 0,
                                       reason="synthetic scientific nonfinite gradient", failure_kind="scientific")), encoding="utf-8")
        failed[label] = path
    output = tmp_path / "W2"
    report = evaluation.evaluate_checkpoints({}, paired, "confirmation", output, base_lock_file=base,
        extension_lock_file=extension, failed_runs=failed, plots=False)
    assert report["ANIE_included"] is False
    assert set(report["methods"]["128"]) == set(evaluation.METHODS)
    assert all(entry["summary"]["failed_count"] == 180 for group in report["methods"].values() for entry in group.values())
    assert (output / "REPORT_COMPLETE.json").is_file()
    initial_report = (output / "numeric_report.json").read_bytes()
    resumed = evaluation.evaluate_checkpoints({}, paired, "confirmation", output, base_lock_file=base,
        extension_lock_file=extension, failed_runs=failed, plots=False, resume=True)
    assert resumed == report
    assert (output / "numeric_report.json").read_bytes() == initial_report
    assert len(list((output / "metrics").glob("*.json"))) == 18
