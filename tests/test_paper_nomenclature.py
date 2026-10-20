"""CPU contracts for explicit numbering, source identities and current bindings."""
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from experiments import paper_nomenclature as names
from scripts import run_experiment, verify_published_results, verify_results


def test_every_paper_identity_is_canonical_and_invalid_id_is_rejected():
    assert [names.resolve_experiment_id(code) for code in names.PAPER_EXPERIMENT_ORDER] == [
        "W1", "W2", "W3", "S1", "S2", "S3"]
    assert names.resolve_experiment_id("S3") == "S3"
    assert names.EXPERIMENT_ID_CHOICES == names.PAPER_EXPERIMENT_ORDER
    with pytest.raises(ValueError, match="unknown experiment"):
        names.resolve_experiment_id("unknown")


def test_display_labels_do_not_merge_current_and_auxiliary_lite_keys():
    source_keys = ("GIFT-N50", "GIFT-Lite")
    assert source_keys[0] != source_keys[1]
    assert [names.method_display_name(key) for key in source_keys] == ["GIFT-Lite", "GIFT-Lite"]
    assert names.method_display_name("fno2d") == names.method_display_name("FNO2D") == "FNO-2D"
    assert names.method_display_name("fno3d") == "FNO-3D"
    assert names.method_display_name("PDE-FIND-KC") == "PDE-FIND-KC"
    assert names.method_display_name("PINN-SR-KC") == "PINN-SR-KC"
    assert "GIFT-N50" in names.prediction_method_order()
    assert "GIFT-Lite" not in names.prediction_method_order()
    assert "anie_ns" not in names.prediction_method_order(include_anie=False)


@pytest.mark.parametrize("experiment", names.PAPER_EXPERIMENT_ORDER)
def test_canonical_launcher_uses_same_identity_module(experiment, monkeypatch):
    calls = []
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda command, **options:
                        calls.append((command, options)) or SimpleNamespace(returncode=0))
    with pytest.raises(SystemExit) as result:
        run_experiment.main(["--device", "cpu", experiment, "--output", "fixture"])
    assert result.value.code == 0 and len(calls) == 1
    command, options = calls[0]
    expected = run_experiment.CURRENT_MODULES.get(experiment, "experiments.formal." + run_experiment.MODULES[experiment] + ".run")
    assert command[3] == expected
    assert command[-2:] == ["--output", "fixture"]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "-1"


@pytest.mark.parametrize("arguments", [["unknown"], ["--numbering", "other", "W1"],
                                        ["W1", "--device", "cpu"]])
def test_invalid_wrapper_arguments_never_dispatch(arguments, monkeypatch):
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("invalid numbering must not start a child"))
    with pytest.raises(SystemExit) as result:
        run_experiment.main(arguments)
    assert result.value.code != 0


@pytest.mark.parametrize("experiment", ["W1", "W2"])
def test_reference_protocol_requires_explicit_selection(experiment, monkeypatch):
    calls = []
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda command, **options:
                        calls.append(command) or SimpleNamespace(returncode=0))
    with pytest.raises(SystemExit) as result:
        run_experiment.main(["--protocol", "reference", experiment, "--output", "fixture"])
    assert result.value.code == 0
    assert calls[0][3] == "experiments.formal." + run_experiment.MODULES[experiment] + ".run"


@pytest.mark.parametrize("module,checker", [(verify_results, "verify_result"),
                                            (verify_published_results, "verify_package")])
def test_verification_uses_canonical_identity(module, checker, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(module, checker, lambda directory, source:
                        calls.append((directory, source)) or {"experiment": source, "status": "fixture"})
    module.main(["--experiment", "S3", "--result-dir", "fixture"])
    assert len(calls) == 1 and calls[0][1] == "S3"
    assert json.loads(capsys.readouterr().out)["experiment"] == "S3"
    with pytest.raises(SystemExit) as result:
        module.main(["--experiment", "unknown", "--result-dir", "fixture"])
    assert result.value.code != 0 and len(calls) == 1


def test_index_uses_only_canonical_identity_and_direct_scientific_packages():
    repo = Path(__file__).resolve().parents[1]
    index = json.loads((repo / "experiments/EXPERIMENT_INDEX.json").read_text(encoding="utf-8"))
    records = index["experiments"]
    assert tuple(row["id"] for row in records) == names.PAPER_EXPERIMENT_ORDER
    assert "seed_stability" not in json.dumps(index)
    for record in records:
        assert "source_experiment_id" not in record
        assert "presentation_manifest" not in record
        assert (repo / record["entrypoint"]).is_file()
        assert "result_directory" in record
        manifest_path = repo / record["result_directory"] / "published.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record["id"] == "W3":
            assert record["status"] == "complete" and record["noise_levels"] == [0]
            assert manifest["schema"] == "gift.w3-single-trajectory.numeric-package.v1"
            assert manifest["protocol"] == "single_trajectory_terminal_v1"
            assert manifest["status"] == "published"
            assert manifest["methods"] == record["methods"]
            assert manifest["case_count"] == manifest["trajectory_count"] == len(record["conditions"]) == 50
            assert manifest["slot_count"] == record["completed_method_condition_slots"] == 250
            assert manifest["parameter_rows"] == record["parameter_rows"] == 750
            assert manifest["failed_slots"] == manifest["failed_rows"] == 0
            package = manifest_path.parent
            files = verify_results.verify_files(package, manifest["files"])
            assert set(files) | {"published.json"} == {p.relative_to(package).as_posix()
                                                       for p in package.rglob("*") if p.is_file()}
            assert manifest["parameter_csv_sha256"].lower() == hashlib.sha256(
                (package / "summary/PARAMETERS_50_FINAL.csv").read_bytes()).hexdigest()
            # Source provenance describes the external original; the files
            # inventory above binds the actual public copies independently.
            assert re.fullmatch(r"[a-fA-F0-9]{64}", manifest["source_lock_sha256"])
            lock = json.loads((package / "summary/PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
            assert lock["profile"] == manifest["protocol"] and lock["case_count"] == manifest["case_count"]
            assert lock["parameter_csv_sha256"] == manifest["parameter_csv_sha256"]
            for name, digest in manifest["audits"].items():
                assert re.fullmatch(r"[a-fA-F0-9]{64}", digest)
                assert "summary/" + name in files
        else:
            assert manifest["schema"] == "gift.published-results.v1"
            assert manifest["experiment"] == record["id"]
        if "publication_manifest_sha256" in record:
            assert record["publication_manifest_sha256"].lower() == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    w1, w2 = records[:2]
    for record in (w1, w2):
        assert record["initial_history_frames"] == 1 and record["training_observation_points"] == 76
        assert record["training_count_exceptions"] == {"GIFT-N50": 50}
        assert record["method_display_names"]["GIFT-N50"] == "GIFT-Lite"
    assert "anie_ns" in w1["methods"] and "anie_ns" not in w2["methods"]
    assert w2["ANIE_included"] is False
    assert w1["current_ANIE_checkpoint"]["source_profile"] == "aligned_ns_single_v1"
    assert w1["current_ANIE_checkpoint"]["selected_epoch"] == 29


def test_verifier_rejects_a_coherent_label_change_without_new_measurements(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    payload = json.loads((repo / "results/formal/W1_flow_prediction_20261007/published.json").read_text(encoding="utf-8"))
    payload["experiment"] = "W2"
    (tmp_path / "published.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Not an explicitly scoped published result package"):
        verify_published_results.verify_package(tmp_path, "W1")
