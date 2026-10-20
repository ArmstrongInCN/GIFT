"""Scientific contracts of the added ANIE evaluation, without model training."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.additional.w1_anie_dense.run import (
    batch_metrics, population_summary, predict_and_measure,
    validate_formal_metadata,
    configure_inference_precision,
    actual_time_grid, validate_time_grid_metadata,
    anie10_reference_rows, draw_comparison, digest_file, TIMES, REPORT_STEPS, REFERENCE_METHODS,
)


def test_dense_time_grid_validates_recorded_gpu_bytes_without_cpu_normalized_recomputation():
    """One-ULP normalized-vector differences must be accepted without rebuilding historical GPU vectors on CPU."""
    identity = formal_metadata()["run_provenance"]["identity"]
    grid = identity["time_grid"]
    normalized = np.asarray(grid["normalized_float32_values"], dtype=np.float32)
    normalized[1:-1] = np.nextafter(normalized[1:-1], np.float32(np.inf))
    grid.update(declared_backend="cuda", device="cuda:0", normalized_float32_values=normalized.tolist())
    # Synthetic one-ULP differences model the CPU/publication versus GPU grid boundary.
    validate_time_grid_metadata(identity)


def test_dense_time_grid_rejects_changed_serialized_tokens():
    """A materially changed normalized query coordinate must fail its physical-time relationship check."""
    identity = formal_metadata()["run_provenance"]["identity"]
    identity["time_grid"]["normalized_float32_values"][42] += .001
    with pytest.raises(ValueError, match="normalized time grid"):
        validate_time_grid_metadata(identity)


def test_dense_physical_query_vector_matches_configuration_without_hashes():
    """Actual physical query values must match training settings without an extra time-vector hash gate."""
    identity = formal_metadata()["run_provenance"]["identity"]
    assert not any(key.endswith("sha256") for key in identity["time_grid"])
    validate_time_grid_metadata(identity)
    identity["time_grid"]["physical_float32_values"][42] += .001
    with pytest.raises(ValueError, match="time-grid bits"):
        validate_time_grid_metadata(identity)


def test_dense_gate_cannot_accept_a_relabelled_sparse_identity():
    """Sparse ten-point metadata cannot qualify as the separately named 151-point training protocol."""
    wrong = formal_metadata()
    wrong["schema"] = "gift.anie-terminal-weights.v1"
    wrong["method"] = "anie"
    wrong["run_provenance"]["identity"]["kind"] = "gift.anie-training.v1"
    with pytest.raises(ValueError, match="metadata"):
        validate_formal_metadata(wrong)


def test_published_anie10_reference_is_verified_without_recomputation():
    """Read the archived ANIE-10 comparison rows and retain their provenance without rerunning its predictor."""
    from pathlib import Path
    import os
    configured = os.environ.get("GIFT_ANIE_SPARSE_REFERENCE")
    if not configured:
        pytest.skip("Archived sparse-grid evidence is external; configure GIFT_ANIE_SPARSE_REFERENCE")
    directory = Path(configured).resolve(strict=True)
    numeric = json.loads((directory / "measurement/numeric_report.json").read_text())
    rows, evidence = anie10_reference_rows(directory, numeric["inputs"]["independent_observations"]["sha256"])
    assert len(rows) == 6 and all(row["method"] == "ANIE-10" for row in rows)
    assert rows[-1]["mean_relative_l2"] == pytest.approx(.9710698041776531)
    assert evidence["reference_recomputed"] is False


def test_dense_figure_reads_bom_and_commits_svg_evidence(tmp_path):
    """Synthetic CSV exercises the real BOM regression, not scientific prediction."""
    from experiments.formal._shared.common import write_csv_new, write_json_new, file_record
    from scripts.verify_results import verify_figures
    output = tmp_path / "synthetic_figure_fixture"
    (output / "summary").mkdir(parents=True)
    rows = []
    for method in (*REFERENCE_METHODS, "ANIE-10", "ANIE-151"):
        for step in REPORT_STEPS:
            rows.append({"method": method, "absolute_time": float(TIMES[step]), "mean_relative_l2": .1,
                         "training_seed_sd": None, "n_training_seeds": 1, "population": 180,
                         "finite_count": 180, "complete_population": True, "evidence_kind": "fixture_only"})
    comparison = output / "summary/comparison.csv"
    write_csv_new(comparison, rows)
    write_json_new(output / "numeric_report.json", {"experiment": "W1-ANIE-151", "status": "numerical_complete",
                   "inputs": {"fixture_only": True}, "numeric_files": [file_record(comparison, output)]})
    draw_comparison(output)
    check = verify_figures(output / "figures", {"experiment": "W1-ANIE-151",
                                              "report_sha256": digest_file(output / "numeric_report.json").upper()})
    assert check["svg_files_verified"] == 1


def formal_metadata():
    """Build a synthetic terminal metadata record for structural validation; no training or real weights are supplied."""
    import torch
    from training.checkpoints import runtime_identity
    configure_inference_precision()
    config = {
        "method": "anie_dense151", "protocol": "anie.dense151.v1",
        "epochs": 500, "trajectories": 1000, "validation_trajectories": 20,
        "dt": .02, "window_seconds": 3., "frames": 151,
        "training_time_points": 151, "prediction_time_points": 151,
        "training_frame_indices": np.arange(151, dtype=int).tolist(),
        "training_local_times": (np.arange(151, dtype=int) * .02).tolist(),
        "effective_batch": 32, "micro_batch": 4, "accumulation": 8,
        "seed": 20260820, "precision": "float32", "amp": False,
        "matmul_tf32": False, "cudnn_tf32": False, "initial_mode": "single_observed_field",
        "initialization": "random_from_seed_no_published_checkpoint",
        "loss": "torch.nn.functional.mse_loss_mean_151_observed_times_including_initial",
        "drop_last": False, "teacher_forcing": False,
        "normalization": "none_raw_observed_vorticity",
        "selection": "terminal_epoch_not_validation", "formal": True,
        "fixture_only": False,
    }
    model = {"frames": 151}
    identity = {
        "kind": "gift.anie-dense-training.v1", "configuration": config,
        "model_configuration": model, "sources": {"adapters/anie.py": "a" * 64},
        "numerical_profile": runtime_identity(),
        "time_grid": actual_time_grid(torch.as_tensor(config["training_local_times"], dtype=torch.float32)),
        "data": {"cohort": "four_vortex", "fixture_only": False,
                 "training_ids": list(range(1000)), "validation_ids": list(range(1000, 1020)),
                 "provenance": {"full_input_checks_performed": True}},
    }
    return {
        "schema": "gift.anie-dense-terminal-weights.v1", "status": "complete",
        "method": "anie_dense151", "artifact_role": "fresh_terminal", "fresh_training": True,
        "completed_epochs": 500, "terminal_epoch": 500,
        "formal_configuration": config, "model_configuration": model,
        "run_provenance": {"run_id": "new-native-run", "identity": identity},
    }


def test_inference_precision_configuration_disables_both_tf32_modes(monkeypatch):
    """The inference setup must disable matmul and cuDNN TF32 independently."""
    import torch
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    configure_inference_precision()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_metadata_requires_full_terminal_budget_and_population():
    """This named formal comparison requires its recorded terminal budget and disjoint declared population."""
    correct = formal_metadata()
    validate_formal_metadata(correct)
    for key, value in (("completed_epochs", 499), ("status", "diagnostic_complete"),
                       ("artifact_role", "test_only"), ("fresh_training", False)):
        bad = deepcopy(correct)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_formal_metadata(bad)
    bad = deepcopy(correct)
    bad["run_provenance"]["identity"]["data"]["training_ids"] = list(range(50))
    with pytest.raises(ValueError, match="population"):
        validate_formal_metadata(bad)


def test_metadata_rejects_other_time_sampling_and_effective_batch():
    """A changed training sample grid or effective batch must not be relabelled as the declared ANIE protocol."""
    correct = formal_metadata()
    for key, value in (("training_time_points", 10), ("effective_batch", 16),
                       ("selection", "validation_best"), ("teacher_forcing", True)):
        bad = deepcopy(correct)
        bad["formal_configuration"][key] = value
        with pytest.raises(ValueError):
            validate_formal_metadata(bad)
    bad = deepcopy(correct)
    bad["formal_configuration"]["training_frame_indices"][2] = 32
    with pytest.raises(ValueError, match="sampling"):
        validate_formal_metadata(bad)


def test_historical_data_audit_flag_does_not_gate_independent_inference():
    """A historical package-audit flag must not block otherwise compatible standalone inference metadata."""
    metadata = formal_metadata()
    metadata["run_provenance"]["identity"]["data"]["provenance"]["full_input_checks_performed"] = False
    assert validate_formal_metadata(metadata)["data"]["provenance"]["full_input_checks_performed"] is False


def test_nonfinite_futures_are_retained_and_have_correct_step_indices():
    """Failed future fields remain nonfinite and their first zero-based future index is reported correctly."""
    truth = np.ones((2, 151, 64, 64), dtype=np.float32)
    prediction = truth.copy()
    prediction[0, 1, 0, 0] = np.nan
    prediction[1, 150, 0, 0] = np.inf
    measured = batch_metrics(prediction, truth, truth[:, 0])
    np.testing.assert_array_equal(measured["first_nonfinite_future_step_zero_based"], [0, 149])
    np.testing.assert_array_equal(measured["all_150_future_fields_finite"], [False, False])
    assert measured["finite_by_trajectory_future_step"].shape == (2, 150)
    assert np.isnan(measured["full_relative_l2"][0, 1])
    summary = population_summary(measured["full_relative_l2"][:, 1])
    assert summary["population"] == 2 and summary["finite_count"] == 1
    assert summary["mean"] is None
    assert summary["finite_subset_mean_diagnostic"] == 0.


def test_initial_decoder_and_raw_solver_initial_output_remain_separate():
    """Decoder reconstruction error must stay distinct from the raw solver t0 error."""
    truth = np.ones((1, 151, 64, 64), dtype=np.float32)
    prediction = truth.copy()
    prediction[:, 0] = .5
    decoder = np.full((1, 64, 64), .75, dtype=np.float32)
    measured = batch_metrics(prediction, truth, decoder)
    np.testing.assert_allclose(measured["initial_decoder_error"], [.25])
    np.testing.assert_allclose(measured["solver_t0_error"], [.5])
    np.testing.assert_array_equal(prediction[:, 0], .5)
    assert measured["all_150_future_fields_finite"].all()


def test_changed_future_targets_do_not_change_single_initial_prediction():
    """Changing future truth must affect metrics only, never the autonomous prediction from one initial field."""
    class InitialOnlyPredictor:
        """Initial-state-only fixture that reveals any accidental future-truth input to inference."""
        def __init__(self):
            """Start a counter for full-sequence predictor calls."""
            self.calls = 0

        def __call__(self, initial):
            """Repeat the initial field over 151 frames while counting exactly one prediction call."""
            self.calls += 1
            assert initial.shape == (1, 64, 64)
            return np.repeat(initial[:, None], 151, axis=1)

        def initial_decoder(self, initial):
            """Return a deliberately imperfect reconstruction to distinguish decoder and solver errors."""
            return initial * .75

    predictor = InitialOnlyPredictor()
    initial = np.ones((1, 64, 64), dtype=np.float32)
    truth = np.repeat(initial[:, None], 151, axis=1)
    first, _, metrics1 = predict_and_measure(predictor, initial, truth)
    changed = truth.copy()
    changed[:, 1:] = 2.
    second, _, metrics2 = predict_and_measure(predictor, initial, changed)
    np.testing.assert_array_equal(first, second)
    assert predictor.calls == 2  # one full-sequence call per evaluation batch
    assert metrics1["full_relative_l2"][0, -1] == 0.
    assert metrics2["full_relative_l2"][0, -1] == .5


def test_nonfinite_truth_cannot_be_confused_with_model_failure():
    """Invalid observed truth must fail validation rather than count as an autonomous model failure."""
    truth = np.ones((1, 151, 64, 64), dtype=np.float32)
    prediction = truth.copy()
    truth[0, 100, 0, 0] = np.nan
    with pytest.raises(ValueError, match="truth"):
        batch_metrics(prediction, truth, prediction[:, 0])


