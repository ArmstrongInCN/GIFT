"""CPU metadata/negative gates only: no scientific generation, model or training."""
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
from training import gift_data as gate
from training.checkpoints import digest_file, runtime_identity


def low_metadata(profile="released"):
    """Supply synthetic v3 generator state and historical training provenance for prerequisite checks."""
    binding = dict(profile=profile, sha256=gate.RELEASED_SHA["noise_000"], full_input_checks_performed=True)
    payload = dict(format_version=3, model_state_dict={"weight": torch.ones(1)},
                   model_configuration={"cutoff": 21, "rank": 64},
                   artifact_role="trained_generator", fresh_training=dict(pretrained_model_loaded=False),
                   training_configuration=asdict(GeneratorTrainingConfig()),
                   training_data=dict(sha256=binding["sha256"].upper(), group="training", trajectory_ids=list(range(50))))
    return payload, binding


def test_historical_fresh_v3_released_metadata_compatibility():
    """Both legacy and nested provenance representations must remain usable with matching generator structure."""
    payload, binding = low_metadata()
    gate.validate_low_prerequisite(payload, binding, {"cutoff": 21, "rank": 64})
    payload["training_data"]["provenance"] = binding
    gate.validate_low_prerequisite(payload, binding, {"cutoff": 21, "rank": 64})


@pytest.mark.parametrize("kind", ["published", "wrong_data", "tiny", "wrong_ids", "regenerated_unbound", "collection_changed"])
def test_low_training_metadata_does_not_qualify_usable_weights(kind):
    """Usable generator state must not require a particular historical budget, file hash or collection receipt."""
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
    gate.validate_low_prerequisite(payload, binding, {"cutoff": 21, "rank": 64})


@pytest.mark.parametrize("change", ["missing_state", "cutoff", "rank", "version"])
def test_generator_actual_structure_is_checked(change):
    """Reject missing weights, an unsupported format or mismatched cutoff/rank before branch use."""
    payload, binding = low_metadata()
    if change == "missing_state":
        payload["model_state_dict"] = {}
    elif change == "version":
        payload["format_version"] = 99
    else:
        payload["model_configuration"][change] = 0
    with pytest.raises(ValueError):
        gate.validate_low_prerequisite(payload, binding, {"cutoff": 21, "rank": 64})


def observed_file(path):
    """Write finite five-frame train/validation observations with disjoint IDs and a 4-grid."""
    with h5py.File(path, "w") as handle:
        for split, ids in (("training", np.arange(2)), ("validation", np.arange(2, 3))):
            group = handle.create_group(split)
            group["trajectory_index"] = ids
            group["time"] = np.arange(5) * .02
            group["vorticity"] = np.ones((len(ids), 5, 4, 4), dtype=np.float32)


def test_invalid_hdf_is_rejected_and_manifest_is_optional(tmp_path):
    """Invalid HDF5 must fail; valid observations must work without claiming published byte identity."""
    path = tmp_path / gate.DATASETS["noise_000"]
    path.write_bytes(b"invalid HDF")
    with pytest.raises(OSError):
        gate.validate_input(tmp_path, "noise_000", "released", full=True)
    observed_file(path)
    for profile in ("released", "regenerated"):
        plan = gate.validate_input(tmp_path, "noise_000", profile, full=True)
        assert plan["sha256"] is None and not plan["published_dataset_identity"]


def test_normal_input_does_not_invoke_collection_assembler(tmp_path, monkeypatch):
    """Ordinary training reads its observations directly and must not invoke collection-assembly validation."""
    observed_file(tmp_path / gate.DATASETS["noise_000"])
    monkeypatch.setattr(assembly, "validate_clean", lambda *a, **k: pytest.fail("ordinary consumer must not run assembler"))
    (tmp_path / "manifest.json").write_text("irrelevant metadata", encoding="utf-8")
    result = gate.validate_input(tmp_path, "noise_000", "regenerated", full=True)
    assert result["sha256"] is None and result["full_input_checks_performed"]


def test_branch_preflight_keeps_provenance_and_defers_array_content_check(tmp_path):
    """Metadata preflight accepts source/runtime/file-byte changes; decoded scientific state is checked later."""
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
    high._preflight_resume(tmp_path, 20260820, paths, torch.device("cpu"), None)
    historical = json.loads((tmp_path / "ATTEMPT.json").read_text())
    historical["identity"]["runtime"] = {"h5py": "old-version", "torch_threads": 99}
    (tmp_path / "ATTEMPT.json").write_text(json.dumps(historical), encoding="utf-8")
    high._preflight_resume(tmp_path, 20260820, paths, torch.device("cpu"), None)
    # Old journals had descriptive environment fields at the top level.
    legacy = historical["identity"]
    legacy.pop("runtime")
    legacy.pop("provenance")
    legacy.update(h5py="old-version", torch_threads=99, torch_interop_threads=98,
                  omp_threads="old", mkl_threads="old", device="cpu", gpu=None,
                  data_sha256="OLD-FILE-BYTES", frozen_generator_sha256="OLD-WEIGHT-FILE")
    (tmp_path / "ATTEMPT.json").write_text(json.dumps(dict(identity=legacy, runtime={})), encoding="utf-8")
    high._preflight_resume(tmp_path, 20260820, paths, torch.device("cpu"), None)


@pytest.mark.parametrize("name", ["w1_recursive_prediction", "w2_cross_resolution", "s1_high_frequency_branch",
                                 "s2_recursive_local_correction"])
def test_prediction_experiment_explicit_low_argument(name, monkeypatch):
    """Prediction experiment parsers must retain an explicitly selected low-generator file."""
    module = importlib.import_module(f"experiments.formal.{name}.run")
    monkeypatch.setattr("sys.argv", ["run.py", "--low-model", "explicit_low.pt"])
    assert module.parse_args().low_model == Path("explicit_low.pt")


def test_low_profile_default_and_explicit():
    """The low-generator CLI defaults to released observations and accepts regenerated routing."""
    assert low_arguments(["--dry-run"]).data_profile == "released"
    assert low_arguments(["--dry-run", "--data-profile", "regenerated"]).data_profile == "regenerated"


def test_new_high_payload_preserves_input_binding_without_model_execution(tmp_path):
    """Synthetic branch export must preserve provenance and distinguish old file hashes from loaded-weight fingerprints."""
    class MetadataOnlyModel:
        """Empty-state export interface isolates metadata serialization without running a branch model."""
        state_scale = high_scale = low_rhs_scale = high_rhs_scale = torch.tensor(1.)

        def state_dict(self):
            """Return no learned tensors so the test cannot be mistaken for trained-weight evidence."""
            return {}

        def export_config(self):
            """Mark the exported empty-state interface as a fixture."""
            return {"fixture_only": True}

    binding = dict(profile="regenerated", sha256="a" * 64, manifest_sha256="b" * 64)
    target = tmp_path / "metadata_only.pt"
    high._write_checkpoint(target, MetadataOnlyModel(), seed=20260820, phase="metadata_only",
        epoch=0, selection_metric=0., frozen_generator_sha256="c" * 64, training_data=binding)
    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert payload["training_data"] == binding and payload["data_profile"] == "regenerated"
    assert payload["model_state"] == {}  # No real model or trained weights in this test.

    modern = tmp_path / "metadata_only_new_binding.pt"
    high._write_checkpoint(modern, MetadataOnlyModel(), seed=20260820, phase="metadata_only",
        epoch=0, selection_metric=0., generator_weights_sha256="d" * 64, training_data=binding)
    actual = torch.load(modern, map_location="cpu", weights_only=True)
    assert actual["frozen_generator_sha256"] is None
    assert actual["generator_weights_sha256"] == "d" * 64


def test_noise_collection_path_override_still_rejects_wrong_protocol(tmp_path):
    """Explicit assembly-file overrides must not permit a scientifically different noise seed/protocol."""
    parent = gate.DATASETS["noise_000"]
    attempt = dict(name="noise", scientific=dict(input_sha256="d" * 64, seed=999))
    with pytest.raises(ValueError, match="Noise protocol"):
        assembly.validate_derivative(attempt, {parent: "d" * 64}, full=True,
                                     data_paths={"noise_001.h5": tmp_path / "absent.h5"})


def test_noise_path_override_metadata_only_no_finite_or_hash_claim(tmp_path):
    """Metadata-only assembly inspection may locate sparse NaN fields but must not certify numeric contents."""
    binding = dict(input_sha256="d" * 64, seed=0, rng="MT19937", scaling="std(clean_group,ddof=0)",
                   group_order=["training", "validation"], frame_chunk=16,
                   fractions={"noise_001": .01, "noise_005": .05})
    paths = {}
    for name in ("noise_001.h5", "noise_005.h5"):
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
