"""CPU metadata/negative gates only: no scientific generation, model or training."""
import copy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from experiments.formal import train_gift_branches as high
from experiments.formal.train_low_generator import parse_args as low_arguments
from experiments.formal._shared.gift_generator_training import GeneratorTrainingConfig
from scripts import assemble_generated_data as assembly
from scripts import generate_data as gen
from training import gift_data as gate
from training.checkpoints import digest_file, runtime_identity


def low_metadata(profile="released"):
    binding = dict(profile=profile, sha256=gate.RELEASED_SHA["noise_000"], full_input_checks_performed=True)
    payload = dict(format_version=3, artifact_role="trained_generator",
                   fresh_training=dict(pretrained_model_loaded=False),
                   training_configuration=asdict(GeneratorTrainingConfig()),
                   training_data=dict(sha256=binding["sha256"].upper(), group="training", trajectory_ids=list(range(50))))
    return payload, binding


def test_historical_fresh_v3_released_metadata_compatibility():
    payload, binding = low_metadata()
    gate.validate_low_prerequisite(payload, binding, asdict(GeneratorTrainingConfig()))
    payload["training_data"]["provenance"] = binding
    gate.validate_low_prerequisite(payload, binding, asdict(GeneratorTrainingConfig()))


@pytest.mark.parametrize("kind", ["published", "wrong_data", "tiny", "wrong_ids", "regenerated_unbound", "collection_changed"])
def test_low_prerequisite_rejects(kind):
    payload, binding = low_metadata()
    if kind == "published":
        payload["fresh_training"] = {}
    elif kind == "wrong_data":
        payload["training_data"]["sha256"] = "0" * 64
    elif kind == "tiny":
        payload["training_configuration"]["phase_steps"] = [1, 1]
    elif kind == "wrong_ids":
        payload["training_data"]["trajectory_ids"] = list(range(1, 51))
    elif kind == "regenerated_unbound":
        binding["profile"] = "regenerated"
    else:
        payload["training_data"]["provenance"] = dict(binding, manifest_sha256="old")
    with pytest.raises(ValueError, match="prerequisite"):
        gate.validate_low_prerequisite(payload, binding, asdict(GeneratorTrainingConfig()))


def test_released_and_missing_manifest_reject_before_opening_model(tmp_path):
    (tmp_path / gate.DATASETS["noise_000"]).write_bytes(b"not released bytes")
    with pytest.raises(ValueError, match="SHA256"):
        gate.validate_input(tmp_path, "noise_000", "released", full=True)
    with pytest.raises(FileNotFoundError):
        gate.validate_input(tmp_path, "noise_000", "regenerated", full=True)
    plan = gate.validate_input(tmp_path, "noise_000", "released", full=False)
    assert plan["sha256"] is None and not plan["published_dataset_identity"]


def test_collection_controller_metadata_only_with_mocked_scientific_boundary(tmp_path, monkeypatch):
    """Mocks scientific validation deliberately; not evidence of generated field acceptance."""
    relative = gate.DATASETS["noise_000"]
    with h5py.File(tmp_path / relative, "w"):
        pass
    plan = gen.make_plan(SimpleNamespace(dataset="standard", initial_conditions=None,
        subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))
    attempt = dict(name="standard", root=tmp_path / "provenance/standard", scientific=plan,
                   run=dict(attempt_id="metadata-only", binding_sha256="f" * 64),
                   receipt=dict(data_sha256=digest_file(tmp_path / relative)))
    groups = {"MOCKED_BOUNDARY_NOT_SCIENTIFIC_VALIDATION": True}
    paths = [(relative, "generated_scientific_input"), ("splits.json", "split_metadata"),
             ("provenance/standard/run.json", "generation_provenance"),
             ("provenance/standard/COMPLETE.json", "generation_provenance")]
    for name, _ in paths[1:]:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        value = dict(schema="gift.generated-splits.v1", files={relative: groups}) if name == "splits.json" else {}
        file.write_text(json.dumps(value), encoding="utf-8")
    records = [dict(path=name, bytes=(tmp_path / name).stat().st_size, sha256=digest_file(tmp_path / name),
                    category=category) for name, category in paths]
    records[0].update(generation_job="standard", attempt_id="metadata-only")
    allowed = set(assembly.CLEAN_PATHS) | set(assembly.SAMPLING) | {"noise"}
    manifest = dict(schema="gift.generated-data-collection.v1", data_profile="regenerated",
        status="ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED", scientific_arrays_modified=False,
        released_truth_used_to_fill_gaps=False, full_hash_and_finite_checks_performed=True,
        assembler_sha256=digest_file(Path(assembly.__file__)), available_jobs=["standard"],
        missing_jobs=sorted(allowed - {"standard"}), files=records)
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(assembly, "load_attempt", lambda *args: attempt)
    calls = []

    def mock_validate(value, *, full, data_path):
        calls.append((value, full, data_path))
        return [(data_path, relative, records[0]["sha256"], groups)]

    monkeypatch.setattr(assembly, "validate_clean", mock_validate)
    result = gate.validate_input(tmp_path, "noise_000", "regenerated", full=True)
    assert calls and result["sha256"] == records[0]["sha256"]
    assert not result["published_dataset_identity"] and not result["numerical_reproduction_verified"]
    original = copy.deepcopy(manifest)
    for key, value in [("assembler_sha256", "0" * 64), ("released_truth_used_to_fill_gaps", True)]:
        changed = dict(original, **{key: value})
        target.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="collection"):
            gate.validate_input(tmp_path, "noise_000", "regenerated", full=True)


def test_branch_identity_covers_runtime_and_rejects_before_preparation(tmp_path):
    data, low = tmp_path / "input.bin", tmp_path / "low.bin"
    data.write_bytes(b"metadata fixture")
    low.write_bytes(b"metadata fixture, not model weights")
    paths = SimpleNamespace(standard_n64=data, low_model=low)
    identity = high._training_identity(20260820, paths, (1., 2., 3., 4.), torch.device("cpu"))
    assert "experiments/formal/_shared/gift_runtime.py" in identity["sources"]
    assert "src/gift/model.py" in identity["sources"]
    (tmp_path / "ATTEMPT.json").write_text(json.dumps(dict(identity=identity, runtime=runtime_identity())), encoding="utf-8")
    high._preflight_resume(tmp_path, 20260820, paths, torch.device("cpu"), None)
    data.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="before data preparation"):
        high._preflight_resume(tmp_path, 20260820, paths, torch.device("cpu"), None)


@pytest.mark.parametrize("name", ["m2_recursive_prediction", "m3_cross_resolution", "s1_high_frequency_branch",
                                 "s2_recursive_local_correction", "s3_seed_stability"])
def test_five_experiment_explicit_low_argument(name, monkeypatch):
    module = importlib.import_module(f"experiments.formal.{name}.run")
    monkeypatch.setattr("sys.argv", ["run.py", "--low-model", "explicit_low.pt"])
    assert module.parse_args().low_model == Path("explicit_low.pt")


def test_low_profile_default_and_explicit():
    assert low_arguments(["--dry-run"]).data_profile == "released"
    assert low_arguments(["--dry-run", "--data-profile", "regenerated"]).data_profile == "regenerated"


def test_new_high_payload_preserves_input_binding_without_model_execution(tmp_path):
    class MetadataOnlyModel:
        state_scale = high_scale = low_rhs_scale = high_rhs_scale = torch.tensor(1.)

        def state_dict(self):
            return {}

        def export_config(self):
            return {"fixture_only": True}

    binding = dict(profile="regenerated", sha256="a" * 64, manifest_sha256="b" * 64)
    target = tmp_path / "metadata_only.pt"
    high._write_checkpoint(target, MetadataOnlyModel(), seed=20260820, phase="metadata_only",
        epoch=0, selection_metric=0., frozen_generator_sha256="c" * 64, training_data=binding)
    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert payload["training_data"] == binding and payload["data_profile"] == "regenerated"
    assert payload["model_state"] == {}  # No real model or trained weights in this test.


def test_noise_collection_path_override_still_rejects_wrong_protocol(tmp_path):
    parent = gate.DATASETS["noise_000"]
    attempt = dict(name="noise", scientific=dict(input_sha256="d" * 64, seed=999))
    with pytest.raises(ValueError, match="Noise protocol"):
        assembly.validate_derivative(attempt, {parent: "d" * 64}, full=True,
                                     data_paths={"noise_001.h5": tmp_path / "absent.h5"})


def test_noise_path_override_metadata_only_no_finite_or_hash_claim(tmp_path):
    binding = dict(input_sha256="d" * 64, seed=0, rng="MT19937", scaling="std(clean_group,ddof=0)",
                   group_order=["training", "validation"], frame_chunk=16,
                   fractions={"noise_001": .01, "noise_010": .10})
    paths = {}
    for name in ("noise_001.h5", "noise_010.h5"):
        path = tmp_path / name
        paths[name] = path
        with h5py.File(path, "w") as handle:
            handle.attrs.update(status="COMPLETE_GENERATION", attempt_id="METADATA_ONLY_MOCK_NOT_SCIENCE")
            for split, ids in (("training", np.arange(50)), ("validation", np.arange(50, 70))):
                group = handle.create_group(split)
                group.create_dataset("trajectory_index", data=ids)
                group.create_dataset("time", data=np.arange(501) * .02)
                group.create_dataset("vorticity", shape=(len(ids), 501, 64, 64), dtype="f4", fillvalue=np.nan)
    attempt = dict(name="noise", scientific=binding, run=dict(attempt_id="METADATA_ONLY_MOCK_NOT_SCIENCE"),
                   receipt=dict(outputs={name: "0" * 64 for name in paths}))
    records = assembly.validate_derivative(attempt, {gate.DATASETS["noise_000"]: "d" * 64},
                                            full=False, data_paths=paths)
    assert len(records) == 2 and {row[0] for row in records} == set(paths.values())
