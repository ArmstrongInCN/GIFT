"""Scientific contracts of the added ANIE evaluation, without model training."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.additional.w1_anie.run import (
    batch_metrics, population_summary, predict_and_measure,
    validate_formal_metadata,
    configure_inference_precision,
)


def formal_metadata():
    """Build a synthetic terminal metadata record for structural validation; no training or real weights are supplied."""
    from training.checkpoints import runtime_identity
    configure_inference_precision()
    config = {
        "epochs": 500, "trajectories": 1000, "validation_trajectories": 20,
        "dt": .02, "window_seconds": 3., "frames": 151,
        "training_time_points": 10, "prediction_time_points": 151,
        "training_frame_indices": np.linspace(0, 150, 10, dtype=int).tolist(),
        "training_local_times": (np.linspace(0, 150, 10, dtype=int) * .02).tolist(),
        "effective_batch": 32, "micro_batch": 1, "accumulation": 32,
        "drop_last": False, "teacher_forcing": False,
        "normalization": "none_raw_observed_vorticity",
        "selection": "terminal_epoch_not_validation", "formal": True,
        "fixture_only": False,
    }
    model = {"frames": 151}
    identity = {
        "kind": "gift.anie-training.v1", "configuration": config,
        "model_configuration": model, "sources": {"adapters/anie.py": "a" * 64},
        "numerical_profile": runtime_identity(),
        "data": {"cohort": "four_vortex", "fixture_only": False,
                 "training_ids": list(range(1000)), "validation_ids": list(range(1000, 1020)),
                 "provenance": {"full_input_checks_performed": True}},
    }
    return {
        "schema": "gift.anie-terminal-weights.v1", "status": "complete",
        "method": "anie", "artifact_role": "fresh_terminal", "fresh_training": True,
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
    for key, value in (("training_time_points", 151), ("effective_batch", 16),
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


@pytest.mark.parametrize("variant", ["w1_anie", "w1_anie_dense"])
def test_standalone_export_loads_without_training_data_or_journal(tmp_path, monkeypatch, variant):
    """Both ANIE variants must load their standalone export without old data, journal, source hashes or runtime match.

    The tiny linear fixture tests export routing/state restoration only; native
    ANIE numerical behavior is covered separately by adapter tests.
    """
    import importlib
    import torch
    from adapters import anie
    module = importlib.import_module("experiments.additional." + variant + ".run")
    metadata_module = importlib.import_module("test_anie_dense_w1_evaluation") if variant.endswith("dense") else None
    if metadata_module is not None:
        metadata = metadata_module.formal_metadata()
    else:
        metadata = formal_metadata()
    class TinyPredictor(torch.nn.Linear):
        """A two-input linear state fixture that replaces native ANIE in the standalone-loader test."""
        def __init__(self, *, config, source_root):
            """Allocate known small weights while accepting the adapter constructor's keyword interface."""
            super().__init__(2, 1)
        def source_identity(self):
            """Label the mocked upstream record as test-only provenance."""
            return {"fixture_only": True}
    monkeypatch.setattr(anie, "ANIEConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(anie, "ANIEPredictor", TinyPredictor)
    predictor = TinyPredictor(config={}, source_root=tmp_path)
    metadata["model_state_dict"] = predictor.state_dict()
    metadata["run_provenance"]["identity"]["sources"] = {"old/missing.py": "OLD"}
    metadata["run_provenance"]["identity"]["numerical_profile"] = {"python": "old-host"}
    metadata["run_provenance"].pop("terminal_boundary", None)
    export = tmp_path / "standalone.pt"
    torch.save(metadata, export)
    loaded, _, provenance = module.load_formal_predictor(export, tmp_path / "missing_data",
                                                       tmp_path / "missing_source", "cpu")
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, predictor.state_dict()[name])
    assert "journal" not in provenance
    assert "historical metadata" in provenance["training_provenance_role"]


def test_sparse_figure_reads_bom_into_independent_output(tmp_path):
    """BOM comparison tables must render into a separate committed SVG directory without altering measurements."""
    from experiments.additional.w1_anie import run
    from experiments.formal._shared.common import write_csv_new, write_json_new, file_record
    from scripts.verify_results import verify_figures
    result = tmp_path / "measurements"
    (result / "summary").mkdir(parents=True)
    rows = [{"method": method, "absolute_time": float(run.TIMES[step]), "mean_relative_l2": .1,
             "training_seed_sd": None, "n_training_seeds": 1, "population": 180,
             "finite_count": 180, "complete_population": True, "evidence_kind": "fixture_only"}
            for method in (*run.REFERENCE_METHODS, "ANIE") for step in run.REPORT_STEPS]
    table = result / "summary/comparison.csv"
    write_csv_new(table, rows)
    write_json_new(result / "numeric_report.json", {"experiment": "W1-ANIE", "status": "numerical_complete",
        "numeric_files": [file_record(table, result)]})
    original = table.read_bytes()
    assert original.startswith(b"\xef\xbb\xbf")
    output = tmp_path / "standalone_figures"
    run.draw_comparison(result, output)
    assert table.read_bytes() == original and not (result / "figures").exists()
    assert verify_figures(output, {"experiment": "W1-ANIE",
        "report_sha256": run.digest_file(result / "numeric_report.json").upper()})["svg_files_verified"] == 1
