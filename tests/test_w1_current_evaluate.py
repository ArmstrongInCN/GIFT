"""Current W1 dispatch fixes the completed seven slots and never imports old raw."""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from experiments.additional import w1_current_evaluate as current
from experiments.additional import w1_extended_evaluate as evaluation


def test_fixed_sources_use_aligned_anie_without_editing_original_locks(tmp_path):
    native, extra, aligned = (tmp_path / name for name in ("native", "extension", "aligned"))
    for root in (native, extra, aligned):
        root.mkdir()
    best = aligned / "training/anie_ns_seed7/best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"Fixed selected checkpoint fixture")
    base = {"training_configurations": {"anie_ns": {"source_profile": "aligned_ns_single_v1"}}}
    extension = {"reused_models": {"anie_ns": {"checkpoint": "prior_selection", "sha256": "prior_digest"},
                                  "GIFT": {"seed": 20260820}}}
    (aligned / "PROTOCOL_LOCK.json").write_text(json.dumps(base))
    extension_path = extra / "EXTENSION_LOCK.json"
    extension_path.write_text(json.dumps(extension))
    before = extension_path.read_bytes()
    slots, locked, combined = current.fixed_sources(native, extra, aligned)
    assert set(slots) == set(evaluation.METHOD_SEEDS)
    assert slots["anie_ns"] == best and locked == base
    assert combined["reused_models"]["anie_ns"]["sha256"] == evaluation.shared.file_sha256(best)
    assert combined["reused_models"]["GIFT"] == extension["reused_models"]["GIFT"]
    assert extension_path.read_bytes() == before


def test_fresh_mode_predicts_before_reading_future_truth_and_keeps_all_failures(tmp_path, monkeypatch):
    events = []
    ids = np.arange(100000, 100180)
    window = SimpleNamespace(path=tmp_path / "cohort.h5", cohort="confirmation", trajectory_ids=ids,
                             query_times=evaluation.QUERY_TIMES, local_times=evaluation.QUERY_TIMES,
                             absolute_times=5 + evaluation.QUERY_TIMES)
    def truth():
        events.append("truth")
        return np.ones((180, 76, 8, 8), dtype=np.float32)
    window.truth = truth
    monkeypatch.setattr(evaluation.shared, "observation_window", lambda *args, **kwargs: window)
    checkpoint = tmp_path / "fixed.pt"
    checkpoint.write_bytes(b"Fixed completed state fixture")
    payload = {"seed": 0, "epoch": 500, "model_spec": {"method": "fno2d"}, "config": {}, "data_binding": {}}
    monkeypatch.setattr(evaluation, "checkpoint_metadata", lambda *args: payload)
    def predict(path, value, observed, raw_path, **kwargs):
        assert "truth" not in events
        events.append("prediction")
        with h5py.File(raw_path, "x") as raw:
            raw.attrs.update(status="complete", completed_trajectories=180)
        return {"inference_seconds": 1.0}
    monkeypatch.setattr(evaluation.native, "predict_checkpoint", predict)
    # Keep the metric commit real: continuation checks require an existing file.
    def score(raw, observed, metric):
        assert events == ["prediction", "truth"]
        with h5py.File(metric, "x") as saved:
            saved.attrs["status"] = "complete"
        return {"summary": {"status": "complete", "population": 180, "failed_count": 0, "primary_metric": 0., "metrics": {}}, "per_trajectory": {}}
    monkeypatch.setattr(evaluation.shared, "score_raw_predictions", score)
    failures = {method: {"method": method, "seed": seed, "reason": "Preserved scientific failure",
                         "failure_kind": "numerical", "stage": "formal"}
                for method, seed in evaluation.METHOD_SEEDS.items() if method != "fno2d"}
    result = evaluation.evaluate({"fno2d": checkpoint}, failures, base_lock={}, extension_lock={},
                                 data_file=window.path, group="test", cohort="confirmation", output=tmp_path / "fresh",
                                 plots=False, device="cpu")
    assert events == ["prediction", "truth"]
    assert len(result["methods"]) == 7 and result["methods"]["fno2d"]["new_inference_performed"] is True
    assert all(result["methods"][method]["summary"]["failed_count"] == 180 for method in failures)
    plan = json.loads((tmp_path / "fresh/evaluation_plan.json").read_text())
    assert plan["reuse_evaluation"] is None
