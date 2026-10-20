"""CPU synthetic compact publications: single-model tables and bound selected arrays."""
import copy
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import publish_w1_w2_single as publication

W1_LABELS = ("GIFT", "fno2d", "fno3d", "anie_ns", "uno", "unet", "GIFT-N50")
W2_LABELS = tuple(label for label in W1_LABELS if label != "anie_ns")
SEEDS = {label: (20260820 if label.startswith("GIFT") else 7 if label == "anie_ns" else 0) for label in W1_LABELS}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def record(label, count, experiment):
    endpoint = "t8_relative_l2" if experiment == "W1" else "t6_relative_l2"
    return dict(seed=SEEDS[label], status="complete", checkpoint_sha256="a"*64,
                summary=dict(failed_count=0, primary_metric=.1,
                    mean_relative_l2_by_time=np.linspace(0., .2, count).tolist(),
                    metrics=dict(future_mean_relative_l2=dict(mean=.1), future_mse=dict(mean=.003),
                                 **{endpoint: dict(mean=.2)})))


def report(experiment, cohort):
    count = 76 if experiment == "W1" else 26
    value = dict(schema="synthetic_fixture_not_a_physical_result", status="numerical_complete", cohort=cohort,
                 query_times=(np.arange(count)*.04).tolist(), trajectory_ids=list(range(
                 100000 if cohort == "confirmation" else 1040, 100180 if cohort == "confirmation" else 1220)))
    if experiment == "W1":
        value["methods"] = {label: record(label, count, experiment) for label in W1_LABELS}
    else:
        value["methods"] = {str(grid): {label: record(label, count, experiment) for label in W2_LABELS} for grid in (64, 96, 128)}
    return value


def test_summary_rows_keep_each_method_cohort_grid_once_without_seed_sd():
    for experiment, expected in (("W1", 14), ("W2", 36)):
        rows = publication.summary_rows({cohort: report(experiment, cohort) for cohort in ("confirmation", "old_test")}, experiment)
        assert len(rows) == expected
        assert len({(row["cohort"], row["grid"], row["method"]) for row in rows}) == expected
        assert all(row["future_mean_relative_l2"] == .1 and row["endpoint_relative_l2"] == .2 for row in rows)
        assert all(row["seed"] == SEEDS[row["method"]] for row in rows)
        assert not any("sd" in key.lower() for row in rows for key in row)
        assert all(row["training_count"] == (50 if row["method"] == "GIFT-N50" else 300) for row in rows)
        if experiment == "W2":
            assert all(row["method"] != "anie_ns" for row in rows)


def raw_fields(path, ids, count, grid, key, *, prediction=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as saved:
        saved.attrs.update(status="complete", completed_trajectories=180)
        saved.create_dataset("trajectory_index", data=ids)
        saved.create_dataset("time", data=np.arange(count)*.04)
        saved.create_dataset("absolute_time", data=5+np.arange(count)*.04)
        field = saved.create_dataset(key, shape=(180, count, grid, grid), dtype="f4",
                                     chunks=(1, 1, grid, grid), compression="lzf", fillvalue=2. if prediction else 1.)
        if prediction:
            field[0, 12, 0, 0] = np.nan


def selected_fixture(root, experiment, cohort):
    count = 76 if experiment == "W1" else 26
    value = report(experiment, cohort)
    directory = root / ("w1_evaluation" if experiment == "W1" else "w2_evaluation") / cohort
    grids = (64,) if experiment == "W1" else (64, 96, 128)
    for grid in grids:
        raw_fields(directory / "raw" / ("truth.h5" if experiment == "W1" else f"N{grid}_truth.h5"),
                   value["trajectory_ids"], count, grid, "vorticity")
        methods = value["methods"] if experiment == "W1" else value["methods"][str(grid)]
        for label, entry in methods.items():
            relative = "raw/" + f"N{grid}_{label}.h5"
            raw_fields(directory / relative, value["trajectory_ids"], count, grid, "prediction", prediction=True)
            entry["raw_prediction_file"] = relative
    write_json(directory / "numeric_report.json", value)
    return directory, value


@pytest.mark.parametrize("experiment", ["W1", "W2"])
def test_selected_fields_actual_nodes_preserve_nan_and_repeat_npz_exactly(tmp_path, experiment):
    directory, value = selected_fixture(tmp_path, experiment, "confirmation")
    arrays = publication.selected_fields(directory, value, experiment)
    wanted = [5.48, 6., 7., 8.] if experiment == "W1" else [5.24, 5.48, 5.76, 6.]
    assert arrays["trajectory_id"].tolist() == [100000] and arrays["absolute_time"].tolist() == wanted
    assert arrays["N64_truth"].shape == (4, 64, 64)
    nan_column = 0 if experiment == "W1" else 1
    assert np.isnan(arrays["N64_uno"][nan_column, 0, 0])
    archive = tmp_path / "source_fields.npz"
    publication.fixed_npz(archive, arrays)
    before = archive.read_bytes()
    publication.fixed_npz(archive, arrays)
    assert archive.read_bytes() == before
    changed = copy.deepcopy(arrays)
    changed["N64_truth"][0, 0, 0] = 9.
    with pytest.raises(ValueError, match="selected field evidence"):
        publication.fixed_npz(archive, changed)
    value["query_times"][12] += .01
    with pytest.raises(ValueError, match="exact saved nodes"):
        publication.selected_fields(directory, value, experiment)


def field_exports(directory, value, experiment):
    """Synthetic plots are generated only with Python and contain editable vector text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"
    figures = directory / "figures_retry_001"
    figures.mkdir()
    stems = ["future_error"] if experiment == "W1" else ["future25_error", "cross_resolution_error", "outer_P31_t6"]
    for stem in stems:
        fig, axis = plt.subplots(figsize=(3, 2))
        axis.plot(value["query_times"], np.linspace(0., .2, len(value["query_times"])))
        axis.set(xlabel="Local time (s)", ylabel="Synthetic fixture error")
        fig.tight_layout()
        fig.savefig(figures / (stem+".svg"))
        plt.close(fig)
    arrays = publication.selected_fields(directory, value, experiment)
    png_stems = ["fixed_id_keyframes"] if experiment == "W1" else [f"fixed_id_keyframes_N{grid}" for grid in (64, 96, 128)]
    for index, stem in enumerate(png_stems):
        grid = 64 if experiment == "W1" else (64, 96, 128)[index]
        fig, axis = plt.subplots(figsize=(2, 2))
        axis.imshow(arrays[f"N{grid}_truth"][0], origin="lower")
        axis.set_title("Synthetic constant field")
        fig.savefig(figures / (stem+".png"))
        plt.close(fig)
    paths = [path for path in figures.iterdir() if path.is_file()]
    write_json(directory / "FIGURES_COMPLETE.json", dict(numeric_report_sha256=publication.sha256(directory / "numeric_report.json"),
        files=[path.relative_to(directory).as_posix() for path in paths],
        file_sha256={path.relative_to(directory).as_posix(): publication.sha256(path) for path in paths}))


@pytest.mark.parametrize("experiment", ["W1", "W2"])
def test_complete_compact_publication_verifier_with_real_synthetic_rows(tmp_path, monkeypatch, experiment):
    root, target_root = tmp_path / "run", tmp_path / "temporary_project"
    root.mkdir()
    monkeypatch.setattr(publication, "ROOT", target_root)
    write_json(root / "EXTENSION_LOCK.json", dict(profile="single_native76_n300", fixture_only=True))
    write_json(root / "FINAL_RESULT_AUDIT.json", dict(status="PASS_NUMERICAL_RECOMPUTATION", fixture_only=True))
    for cohort in ("confirmation", "old_test"):
        directory, value = selected_fixture(root, experiment, cohort)
        field_exports(directory, value, experiment)
    published = publication.publish_package(root, experiment)
    assert published["integrity"]["status"] == "PASS_PUBLISHED_INTEGRITY_ONLY"
    package = Path(published["path"])
    measurement = json.loads((package / "summary/summary.json").read_text())
    assert measurement["single_model_values"] and not measurement["new_blind_test"]
    with (package / "summary/metrics.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == (14 if experiment == "W1" else 36)
    assert not any("sd" in key.lower() for key in rows[0])
    assert (package / "figures/confirmation/source_fields.npz").is_file()
    assert not list(package.rglob("*.h5")) and not list(package.rglob("*.png"))
    assert len(list((target_root / "assets/figures").glob("*.png"))) == (2 if experiment == "W1" else 6)
    manifest = json.loads((package / "published.json").read_text())
    assert manifest["full_raw_outputs_included"] is False and manifest["resumable_experiment"] is False
    assert publication.publish_package(root, experiment)["integrity"] == published["integrity"]


def test_source_figure_checks_retry_image_and_numeric_source_sha(tmp_path):
    directory, value = selected_fixture(tmp_path, "W1", "confirmation")
    field_exports(directory, value, "W1")
    path = publication.source_figure(directory, "future_error", "svg")
    assert path.parent.name == "figures_retry_001"
    report_path = directory / "numeric_report.json"
    report_path.write_text(report_path.read_text()+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="numeric|report|source"):
        publication.source_figure(directory, "future_error", "svg")


def cost_fixture(root):
    base = root / "prior"
    base.mkdir(parents=True)
    lock = dict(base_lock=str(base / "PROTOCOL_LOCK.json"), reused_models={})
    old = []
    for index, label in enumerate(("fno2d", "fno3d", "anie_ns", "GIFT"), 1):
        seed = SEEDS[label]
        directory = base / "training" / f"{label}_seed{seed}"
        checkpoint = directory / "branch/model.pt" if label == "GIFT" else directory / "weights.pt"
        lock["reused_models"][label] = dict(checkpoint=str(checkpoint))
        old.append(dict(label="train_" + directory.name, started_utc=f"2026-10-04T0{index}:00:00Z", process_seconds=100*index, exit_code=0))
    write_json(root / "EXTENSION_LOCK.json", lock)
    (base / "JOB_EVENTS.jsonl").write_text("".join(json.dumps(item)+"\n" for item in old), encoding="utf-8")
    current = [dict(label=label, started_utc=f"2026-10-05T00:00:0{index}Z", process_seconds=seconds, exit_code=0)
               for index, (label, seconds) in enumerate((("train_unet_seed0", 10), ("train_uno_seed0", 20),
                  ("train_GIFT-N50_seed20260820", 30), ("w1_evaluation_confirmation", 5), ("training_final_audit", 2)))]
    (root / "JOB_EVENTS.jsonl").write_text("".join(json.dumps(item)+"\n" for item in current), encoding="utf-8")
    return current


def test_cost_closure_counts_process_once_and_reused_training_separately(tmp_path):
    current = cost_fixture(tmp_path)
    value = publication.cost_closure(tmp_path)
    assert value["new_training_process_seconds"] == 60 and value["current_all_process_seconds"] == 67
    assert sum(value["reused_model_original_training_process_seconds"].values()) == 1000
    assert value["reused_training_new_process_seconds"] == 0 and value["nested_inference_times_not_added_to_parent_wall"]
    assert publication.cost_closure(tmp_path) == value
    with (tmp_path / "JOB_EVENTS.jsonl").open("a", encoding="utf-8") as saved:
        saved.write(json.dumps(current[0])+"\n")
    with pytest.raises(ValueError, match="duplicate"):
        publication.cost_closure(tmp_path)


def test_main_requires_both_scientific_audits_before_writing_publication(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["publisher", "--root", str(tmp_path)])
    monkeypatch.setattr(publication, "cost_closure", lambda *args: pytest.fail("unaccepted audit cannot publish costs or results"))
    write_json(tmp_path / "FINAL_RESULT_AUDIT.json", dict(status="FAIL_NUMERICAL_RECOMPUTATION"))
    with pytest.raises(ValueError, match="full scientific audit"):
        publication.main()
    write_json(tmp_path / "FINAL_RESULT_AUDIT.json", dict(status="PASS_NUMERICAL_RECOMPUTATION"))
    write_json(tmp_path / "FINAL_TRAINING_AUDIT.json", dict(status="FAIL_INCOMPLETE_SLOT"))
    with pytest.raises(ValueError, match="FINAL_TRAINING_AUDIT"):
        publication.main()
