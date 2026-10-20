"""CPU audit of frozen extended checkpoints, whole population and artifact resume."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from experiments.additional import w1_extended_evaluate as evaluation
from training import w1_extended_baselines as training


@pytest.fixture(autouse=True)
def cpu_only_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def data_binding(count=300):
    """Exact declared IDs/grid with synthetic observation hashes, without test fields."""
    return dict(points=76, future_steps=75, query_times=evaluation.QUERY_TIMES.tolist(),
                observation_dt_seconds=.04, training_count=count, subset_seed=2026100304, time_stride=2,
                source_frame_indices=list(range(250, 401, 2)), manifest_sha256="c"*64,
                training=dict(ids=np.random.default_rng(2026100304).permutation(1000)[:count].tolist(), observations_sha256="a"*64),
                validation=dict(ids=list(range(1000, 1020)), observations_sha256="b"*64))


def frozen_checkpoint(tmp_path, method="uno"):
    """Bind one genuine recipe to a finite synthetic state for metadata preflight."""
    config = training.native_training_config(method)
    spec = dict(method=method, history=1, points=76, future_steps=75,
                query_times=evaluation.QUERY_TIMES.tolist(), observation_dt=.04, source_root="synthetic_public_source")
    payload = dict(schema=training.WEIGHTS_SCHEMA, model_state={"weight": torch.ones(1)},
                   model_spec=spec, config=config, data_binding=data_binding(), seed=0,
                   stage="formal", epoch=500, val_metric=None, status="terminal", selection="terminal")
    base = dict(data_binding=data_binding())
    extension = dict(training_configurations={method: dict(config=copy.deepcopy(config), model_spec=copy.deepcopy(spec))})
    path = tmp_path / (method + ".pt")
    torch.save(payload, path)
    return path, payload, base, extension


@pytest.mark.parametrize("method", ["uno", "unet"])
def test_actual_recipe_sparse76_metadata_uses_explicit_grid(method, tmp_path):
    """Synthetic common identity must not silently take the legacy151 default."""
    path, payload, base, extension = frozen_checkpoint(tmp_path, method)
    saved = evaluation.checkpoint_metadata(method, path, base, extension)
    assert saved["model_spec"]["future_steps"] == 75 and saved["val_metric"] is None
    assert saved["config"]["training_future_steps"] == (20 if method == "uno" else 4)


@pytest.mark.parametrize("alter", ["none_state", "empty_state", "nonfinite", "seed", "epoch", "recipe", "observations", "schema", "selection"])
def test_fixed_metadata_rejects_missing_state_and_scientifically_different_slot(tmp_path, alter):
    path, payload, base, extension = frozen_checkpoint(tmp_path)
    if alter == "none_state":
        payload["model_state"] = None
    elif alter == "empty_state":
        payload["model_state"] = {}
    elif alter == "nonfinite":
        payload["model_state"]["weight"][0] = float("nan")
    elif alter == "seed":
        payload["seed"] = 11
    elif alter == "epoch":
        payload["epoch"] = 200
    elif alter == "recipe":
        payload["config"]["training_future_steps"] = 75
    elif alter == "schema":
        payload["schema"] = "old_46_history_schema"
    elif alter == "selection":
        payload["selection"] = "validation_best"
    else:
        payload["data_binding"]["training"]["observations_sha256"] = "d"*64
    torch.save(payload, path)
    with pytest.raises(ValueError):
        evaluation.checkpoint_metadata("uno", path, base, extension)


def test_n50_exact_population_binding_and_same_gift_recipe(tmp_path):
    """The reduced data control shares GIFT architecture but never N300 observations."""
    from training.w1_single_gift import native_training_config
    config = native_training_config()
    spec = dict(method="GIFT", family="gift", history=1, points=76, future_steps=75,
                query_times=evaluation.QUERY_TIMES.tolist(), step=.02, integration_dt=.02,
                duration=3., save_interval=.04, rollout_integration_steps=[0, 4, 10, 20, 24, 30, 40, 50])
    binding = data_binding(50)
    payload = dict(model_spec=spec, model_state={"weight": torch.ones(1)}, config=config,
                   seed=20260820, stage="formal", epoch=500, val_metric=None, selection="terminal_epoch",
                   data_binding=binding)
    extension = dict(n50_data_binding=copy.deepcopy(binding), training_configurations={"GIFT-N50":
                     dict(config=json.loads(json.dumps(config)), model_spec=copy.deepcopy(spec))})
    path = tmp_path / "gift50.pt"
    torch.save(payload, path)
    assert evaluation.checkpoint_metadata("GIFT-N50", path, dict(data_binding=data_binding()), extension)["epoch"] == 500
    payload["data_binding"]["training"]["ids"][0] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="first50"):
        evaluation.checkpoint_metadata("GIFT-N50", path, dict(data_binding=data_binding()), extension)


def metric_fixture(directory, *, status="complete"):
    metric = directory / "metric.h5"
    with h5py.File(metric, "x") as saved:
        saved.attrs["status"] = status
        saved.create_dataset("relative_l2_by_time", shape=(180, 76), dtype="f8", fillvalue=.1)
        saved.create_dataset("time", data=evaluation.QUERY_TIMES)
    record = dict(checkpoint_sha256="model", raw_prediction_sha256="raw",
                  metric_sha256=evaluation.shared.file_sha256(metric))
    record_path = directory / "metric.json"
    evaluation.shared.write_json(record_path, record)
    return record_path, metric, record


def test_incomplete_metrics_are_never_reused_and_complete_bytes_are_bound(tmp_path):
    record_path, metric, record = metric_fixture(tmp_path, status="incomplete")
    assert evaluation._metric_record(record_path, metric, "model", "raw") is None
    with h5py.File(metric, "r+") as saved:
        saved.attrs["status"] = "complete"
    record["metric_sha256"] = evaluation.shared.file_sha256(metric)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert evaluation._metric_record(record_path, metric, "model", "raw") == record
    with h5py.File(metric, "r+") as saved:
        saved["relative_l2_by_time"][0, 1] = 100.
    with pytest.raises(ValueError, match="metric bytes"):
        evaluation._metric_record(record_path, metric, "model", "raw")


def synthetic_data(path, *, truth=False):
    """Sparse allocated constant fields exercise full180 identity without solving PDEs."""
    ids = np.arange(100000, 100180)
    with h5py.File(path, "x") as handle:
        handle.attrs.update(trajectory_id_scheme="canonical", complete=True)
        node = handle if truth else handle.create_group("test")
        if truth:
            handle.attrs["status"] = "complete"
        node.create_dataset("trajectory_index", data=ids)
        node.create_dataset("absolute_time", data=5+evaluation.QUERY_TIMES)
        node.create_dataset("time", data=evaluation.QUERY_TIMES)
        node.create_dataset("vorticity", shape=(180, 76, 64, 64), dtype="f4", fillvalue=1.,
                            chunks=(1, 1, 64, 64), compression="lzf")


def all_failures():
    return {method: dict(method=method, seed=seed, reason="synthetic numerical failure",
                         failure_kind="numerical", stage="formal")
            for method, seed in evaluation.METHOD_SEEDS.items()}


def reused_artifacts(tmp_path):
    data = tmp_path / "cohort.h5"
    synthetic_data(data)
    directory = tmp_path / "base_measurement"
    (directory / "raw").mkdir(parents=True)
    synthetic_data(directory / "raw/truth.h5", truth=True)
    evaluation.shared.write_json(directory / "numeric_report.json", dict(
        trajectory_ids=list(range(100000, 100180)), cohort="confirmation", query_times=evaluation.QUERY_TIMES.tolist()))
    return data, directory


def test_exact_imported_truth_compares_values_as_well_as_ids(tmp_path):
    data, directory = reused_artifacts(tmp_path)
    window = evaluation.shared.observation_window(data, "test", "confirmation", query_times=evaluation.QUERY_TIMES)
    evaluation._validate_imported_truth(window, directory / "raw/truth.h5")
    with h5py.File(data, "r+") as source:
        source["test/vorticity"][0, 20, 0, 0] = 2.
    with pytest.raises(ValueError, match="truth values"):
        evaluation._validate_imported_truth(window, directory / "raw/truth.h5")


def test_whole180_failed_slots_resume_and_corrupt_markdown_are_preserved(tmp_path):
    """Failed slots remain whole population; completion markers bind actual report text."""
    data, previous = reused_artifacts(tmp_path)
    output = tmp_path / "out"
    options = dict(base_lock={}, extension_lock={}, data_file=data, group="test", cohort="confirmation",
                   output=output, reuse_evaluation=previous, plots=False, device="cpu")
    first = evaluation.evaluate({}, all_failures(), **options)
    assert len(first["methods"]) == 7
    assert all(record["summary"]["failed_count"] == 180 and record["summary"]["primary_metric"] is None
               for record in first["methods"].values())
    original = (output / "RESULTS.md").read_bytes()
    (output / "RESULTS.md").write_text("corrupt or incomplete rendering", encoding="utf-8")
    second = evaluation.evaluate({}, all_failures(), resume=True, **options)
    assert second == first and (output / "RESULTS.md").read_bytes() == original
    assert list(output.glob("RESULTS.interrupted_*.md"))
    partial = copy.deepcopy(first)
    del partial["methods"]["uno"]
    (output / "numeric_report.json").write_text(json.dumps(partial), encoding="utf-8")
    recovered = evaluation.evaluate({}, all_failures(), resume=True, **options)
    assert len(recovered["methods"]) == 7 and list(output.glob("numeric_report.interrupted_*.json"))


def test_all_seven_slots_required_before_any_test_access(tmp_path):
    with pytest.raises(ValueError, match="exactly seven"):
        evaluation.evaluate({}, {}, base_lock={}, extension_lock={}, data_file=tmp_path / "must_not_open.h5",
                            group="test", cohort="confirmation", output=tmp_path / "out", reuse_evaluation=tmp_path / "absent")
    assert not (tmp_path / "out").exists()


def test_svg_text_and_figure_markers_bind_image_bytes_and_preserve_retries(tmp_path, monkeypatch):
    """Export editable SVG text; corrupt render bytes cannot count as completed."""
    output = tmp_path / "figures_only"
    output.mkdir()
    report = dict(absolute_times=(5+evaluation.QUERY_TIMES).tolist(), methods={"uno":
                  dict(status="complete", summary=dict(mean_relative_l2_by_time=(evaluation.QUERY_TIMES*.01).tolist()))})
    evaluation.shared.write_json(output / "numeric_report.json", report)
    monkeypatch.setattr(evaluation, "draw_keyframes", lambda *args: {"fixture_only": True})
    evaluation.draw_figures(output)
    assert evaluation.figures_complete(output)
    marker = json.loads((output / "FIGURES_COMPLETE.json").read_text())
    svg = next(output / path for path in marker["files"] if path.endswith(".svg"))
    assert "<text" in svg.read_text(encoding="utf-8")
    png = next(output / path for path in marker["files"] if path.endswith(".png"))
    png.write_bytes(b"interrupted image data")
    assert not evaluation.figures_complete(output)
    evaluation.draw_figures(output)
    assert evaluation.figures_complete(output) and list(output.glob("FIGURES_COMPLETE.interrupted_*.json"))
    assert (output / "figures_retry_001/future_error.svg").is_file()
