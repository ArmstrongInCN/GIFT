"""Bounded fail-closed assembly tests; mock files are never scientific data."""
import argparse
import hashlib
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import assemble_generated_data as assembly  # noqa: E402
from scripts import generate_data as gen  # noqa: E402


def test_rejects_pilot_and_missing_generation_receipt(tmp_path):
    released_like=tmp_path/"no_generation_receipt"
    released_like.mkdir()
    with pytest.raises(FileNotFoundError):
        assembly.load_attempt("standard",released_like)
    pilot=tmp_path/"pilot"
    pilot.mkdir()
    gen.write_json_new(pilot/"run.json",{})
    gen.write_json_new(pilot/"COMPLETE.json",{"status":"COMPLETE_PILOT"})
    with pytest.raises(ValueError,match="pilot"):
        assembly.load_attempt("standard",pilot)


def test_noise_cannot_take_released_parent_to_fill_gap():
    attempt={"name":"noise","scientific":{"input_sha256":"released-parent-hash"}}
    with pytest.raises(ValueError,match="newly generated parent"):
        assembly.validate_derivative(attempt,{},full=False)
    with pytest.raises(ValueError,match="newly generated parent"):
        assembly.validate_derivative(attempt,{assembly.CLEAN_PATHS["standard"]:"different-new-parent-hash"},full=False)


def test_copy_is_create_only_and_byte_preserving(tmp_path):
    source,target=tmp_path/"source.txt",tmp_path/"target.txt"
    source.write_text("unit fixture, not scientific data",encoding="utf-8")
    before=gen.sha256(source)
    assembly.copy_new(source,target,before)
    assert source.read_bytes()==target.read_bytes()
    with pytest.raises(FileExistsError):
        assembly.copy_new(source,target,before)


def test_full_population_metadata_does_not_bypass_unwritten_values(tmp_path):
    plan=gen.make_plan(argparse.Namespace(dataset="standard",initial_conditions=None,
        subset=None,split="all",pilot_steps=None,batch_size=None,device="cpu"))
    binding={"plan":plan,"runtime":{"test_fixture_only":True}}
    binding_hash=hashlib.sha256(gen.canonical(binding).encode()).hexdigest()
    run={"binding":binding,"binding_sha256":binding_hash,"attempt_id":"MOCK_NOT_SCIENCE"}
    path=tmp_path/"data.h5"
    with h5py.File(path,"x") as handle:
        handle.attrs.update(status="COMPLETE_GENERATION",attempt_id=run["attempt_id"],
            binding_sha256=binding_hash,generated_from="initial_conditions_not_saved_truth",fixture_only=True)
        for info in plan["groups"]:
            group=handle.create_group(info["name"])
            group.create_dataset("trajectory_index",data=np.asarray(info["ids"],dtype=np.int64))
            group.create_dataset("time",data=gen.stored_times(plan,info))
            group.create_dataset("initial_condition_parameters",data=np.asarray(info["parameters"],dtype=np.float64))
            group.create_dataset("vorticity",shape=(len(info["ids"]),len(info["steps"]),64,64),
                dtype="f4",chunks=(1,1,64,64),fillvalue=np.nan)
    attempt={"name":"standard","root":tmp_path,"run":run,"scientific":plan,
             "receipt":{"data_sha256":gen.sha256(path)}}
    # A metadata-only plan is allowed to inspect a schema, never to assert eligibility.
    assert len(assembly.validate_clean(attempt,full=False))==1
    with pytest.raises(ValueError,match="Nonfinite/unwritten"):
        assembly.validate_clean(attempt,full=True)
