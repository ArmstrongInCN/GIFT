"""CPU copy/guard tests with TINY MOCK H5s, not real generation acceptance.

Only the schedules and names are full-sized; grids/populations are deliberately
small and the parent has explicit fixture markers. Scientific-boundary patches
are local to tests. No public CLI flag accepts these fixtures as formal inputs.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import assemble_generated_data as assembly  # noqa: E402
from scripts import derive_dense_frames as derive  # noqa: E402
from scripts import generate_data as gen  # noqa: E402


def mock_parent(tmp_path, name="fno-test"):
    """Finite two-row/two-pixel fixture, explicitly NOT a solver output."""
    root = tmp_path / (name + "_mock_" + uuid.uuid4().hex)
    root.mkdir()
    canonical = gen.make_plan(argparse.Namespace(dataset="standard", initial_conditions=None,
        subset=None, split="all", pilot_steps=None, batch_size=None, device="cpu"))
    groups = []
    names = ("training", "validation") if name == "fno-training" else ("N64/test", "N96/test", "N128/test")
    for index, group_name in enumerate(names):
        parameters = np.arange(32, dtype=np.float64).reshape(2, 4, 4) / 100 + index
        steps = list(range(0, 2001, 4)) if name == "fno-training" else list(range(820, (1600 if group_name == "N64/test" else 1200) + 1, 4))
        groups.append({"name": group_name, "grid": 2, "split": group_name.split("/")[-1],
                       "ids": [index * 2, index * 2 + 1], "steps": steps, "batch_size": 2,
                       "parameters": parameters.tolist(), "parameters_sha256": gen.array_hash(parameters)})
    plan = {**canonical, "dataset": name, "groups": groups,
            "initial_conditions_file_sha256": "a" * 64 if name == "fno-training" else None,
            "extra950_origin": "specified_initial_conditions_seed_unknown" if name == "fno-training" else None}
    binding = {"plan": plan, "runtime": {"test_fixture_only": True, "python": "mock", "numpy": "mock",
               "h5py": "mock", "torch": "not imported", "device": "cpu", "threads": 1, "interop_threads": 1}}
    run = {"attempt_id": str(uuid.uuid4()), "binding": binding, "binding_sha256": derive.digest_json(binding)}
    gen.write_json_new(root / "run.json", run)
    with h5py.File(root / "data.h5", "x") as handle:
        handle.attrs.update(status="COMPLETE_GENERATION", attempt_id=run["attempt_id"],
            binding_sha256=run["binding_sha256"], generated_from="initial_conditions_not_saved_truth", fixture_only=True)
        gen.write_generation_metadata(handle, plan, complete=True)
        for info in groups:
            group = handle.create_group(info["name"])
            group.create_dataset("trajectory_index", data=np.asarray(info["ids"], dtype=np.int64))
            group.create_dataset("time", data=gen.stored_times(plan, info))
            group.create_dataset(derive.parameter_name(info["name"]), data=np.asarray(info["parameters"], dtype=np.float64))
            values = np.arange(2 * len(info["steps"]) * 4, dtype=np.float32).reshape(2, -1, 2, 2)
            values[..., 0, 0], values[..., 0, 1] = np.float32(-0.0), np.float32(0.0)
            group.create_dataset("vorticity", data=values)
            group.attrs["solver_metadata"] = gen.canonical({"fixture_only": True, "not_a_solver_run": True})
    receipt = {"status": "COMPLETE_GENERATION", "attempt_id": run["attempt_id"],
               "binding_sha256": run["binding_sha256"], "data_sha256": gen.sha256(root / "data.h5"),
               "reference_comparison_performed": False, "full_budget_reproduction_claim": False}
    gen.write_json_new(root / "COMPLETE.json", receipt)
    return {"name": name, "root": root, "run": run, "receipt": receipt, "scientific": plan}


def mock_group_boundary(monkeypatch, parent):
    """Keep exact full temporal mappings while substituting tiny spatial/ID axes."""
    def expected(dataset):
        is_parent = dataset == parent["name"]
        groups = parent["scientific"]["groups"]
        if dataset == "short-test":
            groups = groups[:1]
        elif dataset == "cross-resolution":
            groups = groups[1:]
        result = []
        for group in groups:
            name = "test" if dataset == "short-test" else group["name"]
            steps = group["steps"] if is_parent else list(range(0, 2001, 20)) if dataset == "fno-training-coarse" else list(range(820, 1201, 20))
            result.append((name, 2, np.asarray(group["ids"], dtype=np.int64), steps, group["parameters_sha256"]))
        return result
    monkeypatch.setattr(assembly, "expected_groups", expected)


def refreshed_receipt(parent):
    parent["receipt"]["data_sha256"] = gen.sha256(parent["root"] / "data.h5")
    gen.atomic_json(parent["root"] / "COMPLETE.json", parent["receipt"])


@pytest.mark.parametrize("target", tuple(derive.PARENTS))
def test_three_integer_mappings_signed_zero_and_reader_metadata(tmp_path, monkeypatch, target):
    imports_before = {name: name in sys.modules for name in ("torch", "tensorflow")}
    parent = mock_parent(tmp_path, derive.PARENTS[target])
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan(target, parent)
    output = tmp_path / (target + "_derived_mock")
    assert derive.copy_attempt(parent, plan, output) == 0
    expected_columns = list(range(0, 501, 5)) if target == "fno-training-coarse" else list(range(0, 96, 5))
    expected_times = np.arange(0, 2001, 20).astype(np.float64) * .005 if target == "fno-training-coarse" else np.round(np.arange(41, 61, dtype=np.float64) / 10, 12)
    with h5py.File(parent["root"] / "data.h5", "r") as source, h5py.File(output / "data.h5", "r") as child:
        assert child.attrs["status"] == derive.STATUS
        assert "metadata_json" not in child.attrs  # Dense fno native dt=.02 is not inherited.
        assert json.loads(child.attrs["derivation_metadata_json"])["stored_dt"] == .1
        for group in plan["groups"]:
            assert group["parent_columns"] == expected_columns
            src, dst = source[group["parent_group"]], child[group["name"]]
            assert np.array_equal(dst["time"][:], expected_times)
            for row in range(2):
                assert derive.bits_equal(dst["vorticity"][row], src["vorticity"][row, expected_columns])
            assert np.signbit(dst["vorticity"][:, :, 0, 0]).all()
            assert not np.signbit(dst["vorticity"][:, :, 0, 1]).any()
            assert dst[derive.parameter_name(group["name"])][:].tobytes() == src[derive.parameter_name(group["parent_group"])][:].tobytes()
    complete = assembly.read_json(output / "COMPLETE.json")
    assert complete["parent"] == derive.parent_identity(parent)
    assert complete["new_solver_invocation"] is False
    assert complete["full_budget_reproduction_claim"] is False
    assert {name: name in sys.modules for name in imports_before} == imports_before


def test_bit_comparison_distinguishes_signed_zero():
    assert not derive.bits_equal(np.asarray([-0.0], dtype="f4"), np.asarray([0.0], dtype="f4"))


def test_separate_process_resume_preserves_own_identity(tmp_path, monkeypatch):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan("short-test", parent)
    spec = tmp_path / "mock_spec.json"
    gen.write_json_new(spec, {"parent_root": str(parent["root"]), "plan": plan})
    output = tmp_path / "resumed_mock"
    # This invokes the copy engine, not the formal CLI. The parent remains marked
    # as a mock, and public load_parent rejects it (tested separately below).
    code = """
import sys
from pathlib import Path
from scripts import derive_dense_frames as d, assemble_generated_data as a
spec = a.read_json(sys.argv[1]); root = Path(spec['parent_root'])
run = a.read_json(root/'run.json'); receipt = a.read_json(root/'COMPLETE.json')
parent = dict(name='fno-test',root=root,run=run,receipt=receipt,scientific=run['binding']['plan'])
assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules
d.copy_attempt(parent,spec['plan'],Path(sys.argv[2]),resume=sys.argv[3]=='resume',
               checkpoint_every=1,stop_after_chunks=1 if sys.argv[3]=='pause' else None)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    first = subprocess.run([sys.executable, "-s", "-B", "-c", code, str(spec), str(output), "pause"],
                           cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert '"status": "PAUSED"' in first.stdout and not (output / "COMPLETE.json").exists()
    before = (output / "run.json").read_bytes()
    subprocess.run([sys.executable, "-s", "-B", "-c", code, str(spec), str(output), "resume"],
                   cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert (output / "run.json").read_bytes() == before
    receipt = assembly.read_json(output / "COMPLETE.json")
    assert receipt["attempt_id"] == assembly.read_json(output / "run.json")["attempt_id"]
    assert receipt["data_sha256"] == gen.sha256(output / "data.h5")
    with pytest.raises(ValueError, match="already complete"):
        derive.copy_attempt(parent, plan, output, resume=True)


@pytest.mark.parametrize("corruption", ("foreign_cursor", "changed_source", "committed_bits", "unselected_parent_nan"))
def test_resume_and_nonfinite_guards(tmp_path, monkeypatch, corruption):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan("short-test", parent)
    output = tmp_path / "paused_mock"
    derive.copy_attempt(parent, plan, output, stop_after_chunks=1)
    if corruption == "foreign_cursor":
        second = tmp_path / "other_attempt"
        derive.copy_attempt(parent, plan, second, stop_after_chunks=1)
        latest = assembly.read_json(second / "latest.json")
        assembly.copy_new(second / latest["file"], output / latest["file"], latest["sha256"])
        gen.atomic_json(output / "latest.json", latest)
        match = "Foreign cursor"
    elif corruption == "changed_source":
        plan = copy.deepcopy(plan)
        plan["sources"]["scripts/derive_dense_frames.py"] = "0" * 64
        match = "Resume parent/source"
    elif corruption == "committed_bits":
        with h5py.File(output / "data.h5", "r+") as handle:
            handle["test/vorticity"][0, 0, 0, 0] = np.float32(0.0)
        match = "differs bitwise"
    else:
        # Even an UNSELECTED parent frame must not be silently trusted at completion.
        with h5py.File(parent["root"] / "data.h5", "r+") as handle:
            handle["N64/test/vorticity"][0, -1, 0, 0] = np.nan
        match = "Parent changed"
    with pytest.raises(ValueError, match=match):
        derive.copy_attempt(parent, plan, output, resume=True)
    assert not (output / "COMPLETE.json").exists()


def test_public_parent_rejects_incomplete_wrong_parent_wrong_source_and_mock(tmp_path):
    parent = mock_parent(tmp_path)
    with pytest.raises(ValueError, match="Fixture/mock"):
        derive.load_parent("short-test", parent["root"], full=True)
    with pytest.raises(ValueError, match="complete new native"):
        derive.load_parent("fno-training-coarse", parent["root"], full=False)
    parent["receipt"]["status"] = "INCOMPLETE"
    gen.atomic_json(parent["root"] / "COMPLETE.json", parent["receipt"])
    with pytest.raises(ValueError, match="incomplete/pilot"):
        derive.load_parent("short-test", parent["root"], full=False)
    parent["receipt"]["status"] = "COMPLETE_GENERATION"
    gen.atomic_json(parent["root"] / "COMPLETE.json", parent["receipt"])
    parent["scientific"]["sources"]["scripts/generate_data.py"] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        derive.inspect_parent(parent, full=False)


@pytest.mark.parametrize("storage", ("external_link", "virtual_axis", "external_storage", "nonfinite_unselected"))
def test_full_parent_scans_unselected_frames_and_all_dataset_storage(tmp_path, monkeypatch, storage):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    # Isolate the actual production H5/finite guards on a tiny marked fixture.
    # Formal acceptance still rejects its fixture/runtime marker without this patch.
    monkeypatch.setattr(derive, "reject_fixture", lambda *args: None)
    root = parent["root"]
    if storage == "external_link":
        with h5py.File(root / "external.h5", "x") as external:
            external.create_dataset("times", data=gen.stored_times(parent["scientific"], parent["scientific"]["groups"][0]))
        with h5py.File(root / "data.h5", "r+") as handle:
            del handle["N64/test/time"]
            handle["N64/test/time"] = h5py.ExternalLink(str(root / "external.h5"), "/times")
        match = "External/soft"
    elif storage == "virtual_axis":
        with h5py.File(root / "data.h5", "r+") as handle:
            del handle["N64/test/time"]
            layout = h5py.VirtualLayout(shape=(196,), dtype="f8")
            layout[:] = h5py.VirtualSource("absent.h5", "times", shape=(196,))
            handle["N64/test"].create_virtual_dataset("time", layout)
        match = "Virtual/external"
    elif storage == "external_storage":
        with h5py.File(root / "data.h5", "r+") as handle:
            del handle["N64/test/trajectory_index"]
            handle["N64/test"].create_dataset("trajectory_index", shape=(2,), dtype="i8",
                external=[(str(root / "external_values.bin"), 0, h5py.h5f.UNLIMITED)])
        match = "Virtual/external"
    else:
        with h5py.File(root / "data.h5", "r+") as handle:
            handle["N64/test/vorticity"][0, -1, 0, 0] = np.nan
        match = "Nonfinite/unwritten"
    refreshed_receipt(parent)
    with pytest.raises(ValueError, match=match):
        derive.inspect_parent(parent, full=True)


def test_selected_nan_cannot_complete(tmp_path, monkeypatch):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    with h5py.File(parent["root"] / "data.h5", "r+") as handle:
        handle["N64/test/vorticity"][0, 0, 1, 1] = np.nan
    refreshed_receipt(parent)
    plan = derive.make_plan("short-test", parent)
    output = tmp_path / "nan_mock"
    with pytest.raises(ValueError, match="Nonfinite"):
        derive.copy_attempt(parent, plan, output)
    assert not (output / "COMPLETE.json").exists()


def test_assembler_requires_same_sha_parent_receipt_and_exact_mapping(tmp_path, monkeypatch):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan("short-test", parent)
    output = tmp_path / "child_mock"
    derive.copy_attempt(parent, plan, output)
    attempt = assembly.load_attempt("short-test", output)
    key = assembly.CLEAN_PATHS["fno-test"]
    digest = parent["receipt"]["data_sha256"]
    for parents, available in (({}, {}), ({"fno-test": parent}, {}), ({"fno-test": parent}, {key: "0" * 64})):
        with pytest.raises(ValueError, match="same-SHA newly generated"):
            derive.validate_for_assembly(attempt, parents, available, full=True)
    # Positive association test patches ONLY the mock-parent scientific boundary;
    # actual child schema/hash/full byte comparisons below remain production code.
    monkeypatch.setattr(derive, "inspect_parent", lambda parent, full: parent["root"] / "data.h5")
    assert len(derive.validate_for_assembly(attempt, {"fno-test": parent}, {key: digest}, full=True)) == 1
    broken = copy.deepcopy(attempt)
    broken["scientific"]["groups"][0]["parent_columns"][0] = 1
    with pytest.raises(ValueError, match="exact integer mapping"):
        derive.validate_for_assembly(broken, {"fno-test": parent}, {key: digest}, full=False)
    changed = copy.deepcopy(parent)
    changed["run"]["attempt_id"] = str(uuid.uuid4())
    with pytest.raises(ValueError, match="parent receipt"):
        derive.validate_for_assembly(attempt, {"fno-test": changed}, {key: digest}, full=False)
    attempt["receipt"]["data_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        derive.validate_for_assembly(attempt, {"fno-test": parent}, {key: digest}, full=True)


def test_assembler_dispatch_and_manifest_records_derivation_without_relabeling(tmp_path, monkeypatch):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan("short-test", parent)
    child = tmp_path / "assembled_child_mock"
    derive.copy_attempt(parent, plan, child)
    key, digest = assembly.CLEAN_PATHS["fno-test"], parent["receipt"]["data_sha256"]
    monkeypatch.setattr(derive, "inspect_parent", lambda parent, full: parent["root"] / "data.h5")
    # Mock the dense scientific boundary only; dispatch, receipts, child bit checks,
    # create-only collection copying and provenance/manifest writing run normally.
    monkeypatch.setattr(assembly, "validate_clean", lambda attempt, full: [(parent["root"] / "data.h5", key, digest, {})])
    collection = tmp_path / "new_mock_collection"
    args = ["--job", "short-test=" + str(child), "--job", "fno-test=" + str(parent["root"]), "--output", str(collection), "--execute"]
    assert assembly.main(args) == 0
    manifest = assembly.read_json(collection / "manifest.json")
    record = next(item for item in manifest["files"] if item.get("generation_job") == "short-test")
    assert record["generation_method"] == "integer_frame_derivation"
    assert record["derivation_parent"] == plan["parent"]
    assert manifest["status"] == "ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED"
    assert assembly.read_json(collection / "provenance/short-test/COMPLETE.json")["status"] == "COMPLETE_DERIVATION"
    with pytest.raises(FileExistsError):
        assembly.main(args)


def test_plan_only_writes_nothing_and_derivation_source_status_are_guarded(tmp_path, monkeypatch, capsys):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    calls = []

    def mock_load(target, directory, *, full):
        calls.append(full)
        return parent

    monkeypatch.setattr(derive, "load_parent", mock_load)
    output = tmp_path / "plan_only"
    assert derive.main(["--dataset", "short-test", "--parent", str(parent["root"]), "--output", str(output)]) == 0
    assert calls == [False] and not output.exists()
    assert json.loads(capsys.readouterr().out)["full_hash_and_finite_checks_performed"] is False
    plan = derive.make_plan("short-test", parent)
    derive.copy_attempt(parent, plan, output)
    expected_sources = derive.source_hashes()
    monkeypatch.setattr(derive, "source_hashes", lambda: {**expected_sources, "scripts/derive_dense_frames.py": "0" * 64})
    with pytest.raises(ValueError, match="schema/slot/source"):
        assembly.load_attempt("short-test", output)
    monkeypatch.setattr(derive, "source_hashes", lambda: expected_sources)
    receipt = assembly.read_json(output / "COMPLETE.json")
    receipt["status"] = "COMPLETE_GENERATION"
    gen.atomic_json(output / "COMPLETE.json", receipt)
    with pytest.raises(ValueError, match="must not claim an integration"):
        assembly.load_attempt("short-test", output)


@pytest.mark.parametrize("phase", ("partial_stage", "before_publish", "after_publish"))
def test_interrupted_completion_is_atomic_and_resumable(tmp_path, monkeypatch, phase):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan("short-test", parent)
    output = tmp_path / ("atomic_mock_" + phase)
    real_stage, real_link = derive.write_completion_stage, os.link

    def partial_stage(path, receipt):
        with path.open("x", encoding="utf-8") as stream:
            stream.write('{"status":')
        raise InterruptedError("simulated interruption during staging write")

    def interrupted_publish(source, destination):
        if phase == "after_publish":
            real_link(source, destination)
        raise InterruptedError("simulated interruption at atomic publish")

    if phase == "partial_stage":
        monkeypatch.setattr(derive, "write_completion_stage", partial_stage)
    else:
        monkeypatch.setattr(os, "link", interrupted_publish)
    with pytest.raises(InterruptedError):
        derive.copy_attempt(parent, plan, output)
    staged = {path.name: path.read_bytes() for path in output.glob("COMPLETE.*.staging.json")}
    assert len(staged) == 1
    run_before = (output / "run.json").read_bytes()
    latest = assembly.read_json(output / "latest.json")
    cursor = assembly.read_json(output / latest["file"])
    assert cursor["next_chunk"] == len(derive.chunks(plan))
    monkeypatch.setattr(derive, "write_completion_stage", real_stage)
    monkeypatch.setattr(os, "link", real_link)
    if phase == "after_publish":
        complete_before = (output / "COMPLETE.json").read_bytes()
        assert json.loads(complete_before)["status"] == derive.STATUS
        with pytest.raises(ValueError, match="already complete"):
            derive.copy_attempt(parent, plan, output, resume=True)
        assert (output / "COMPLETE.json").read_bytes() == complete_before
    else:
        assert not (output / "COMPLETE.json").exists()
        assert derive.copy_attempt(parent, plan, output, resume=True) == 0
    assert (output / "run.json").read_bytes() == run_before
    for name, value in staged.items():
        assert (output / name).read_bytes() == value  # No staging removal/refresh.
    complete = assembly.read_json(output / "COMPLETE.json")
    assert complete["data_sha256"] == gen.sha256(output / "data.h5")
    # Publication is create-only even when called directly with an existing receipt.
    with pytest.raises(FileExistsError):
        derive.publish_completion(output, complete)


def test_writer_lock_covers_closed_h5_hash_and_publication(tmp_path, monkeypatch):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan("short-test", parent)
    output = tmp_path / "locked_mock"
    spec = tmp_path / "locked_mock_spec.json"
    gen.write_json_new(spec, {"parent_root": str(parent["root"]), "plan": plan})
    code = """
import sys
from pathlib import Path
from scripts import derive_dense_frames as d, assemble_generated_data as a
spec=a.read_json(sys.argv[1]); root=Path(spec['parent_root'])
run=a.read_json(root/'run.json'); receipt=a.read_json(root/'COMPLETE.json')
parent=dict(name='fno-test',root=root,run=run,receipt=receipt,scientific=run['binding']['plan'])
try:
    d.copy_attempt(parent,spec['plan'],Path(sys.argv[2]),resume=True)
except RuntimeError as error:
    assert 'Another derivation writer' in str(error)
    print('CONCURRENT_WRITER_REJECTED')
else:
    raise AssertionError('second writer was not rejected')
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    real_hash, real_publish = gen.sha256, derive.publish_completion
    checked = []

    def hash_after_h5_close(path):
        if Path(path) == output / "data.h5":
            # HDF5's own writer lock is already gone at this previously unsafe
            # boundary. The attempt lock must still exclude both writers below.
            with h5py.File(path, "r+") as handle:
                assert handle.attrs["status"] == derive.STATUS
            with pytest.raises(RuntimeError, match="Another derivation writer"):
                derive.copy_attempt(parent, plan, output, resume=True)
            result = subprocess.run([sys.executable, "-s", "-B", "-c", code, str(spec), str(output)],
                cwd=ROOT, env=env, text=True, capture_output=True, check=True, timeout=30)
            assert "CONCURRENT_WRITER_REJECTED" in result.stdout
            checked.append("post_h5_hash")
        return real_hash(path)

    def publish_under_lock(path, receipt):
        with pytest.raises(RuntimeError, match="Another derivation writer"):
            with derive.writer_lock(output):
                pytest.fail("nested writer lock acquired")
        checked.append("final_publication")
        real_publish(path, receipt)

    monkeypatch.setattr(gen, "sha256", hash_after_h5_close)
    monkeypatch.setattr(derive, "publish_completion", publish_under_lock)
    assert derive.copy_attempt(parent, plan, output) == 0
    assert checked == ["post_h5_hash", "final_publication"]
    # The persistent file is not a stale sentinel and can be locked after exit.
    with derive.writer_lock(output):
        assert (output / ".writer.lock").exists()
    with pytest.raises(ValueError, match="already complete"):
        derive.copy_attempt(parent, plan, output, resume=True)


def test_source_change_during_final_hash_does_not_publish_completion(tmp_path, monkeypatch):
    parent = mock_parent(tmp_path)
    mock_group_boundary(monkeypatch, parent)
    plan = derive.make_plan("short-test", parent)
    output = tmp_path / "source_change_mock"
    original_sources, real_hash = derive.source_hashes(), gen.sha256
    changed = []

    def final_hash(path):
        digest = real_hash(path)
        if Path(path) == output / "data.h5":
            changed.append(True)
        return digest

    monkeypatch.setattr(gen, "sha256", final_hash)
    monkeypatch.setattr(derive, "source_hashes", lambda: {**original_sources,
        "scripts/derive_dense_frames.py": "0" * 64} if changed else original_sources)
    with pytest.raises(ValueError, match="Source files changed"):
        derive.copy_attempt(parent, plan, output)
    assert changed and not (output / "COMPLETE.json").exists()
    assert not list(output.glob("COMPLETE.*.staging.json"))
    monkeypatch.setattr(gen, "sha256", real_hash)
    monkeypatch.setattr(derive, "source_hashes", lambda: original_sources)
    assert derive.copy_attempt(parent, plan, output, resume=True) == 0
