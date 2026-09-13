"""Assemble completed independent generation attempts into a NEW input collection.

Only this candidate's generation receipts are accepted. Never imports models,
copies released truth to fill gaps, edits arrays, or relaxes training hash gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import generate_data as gen

CLEAN_PATHS = {
    "standard": "standard_ns_n64_full_spectrum.h5",
    "fno-training": "fno/fno1000_n64_t0_t10_dt0p02.h5",
    "fno-training-coarse": "fair_short_horizon/fno1000_n64_t0_t10_dt0p1.h5",
    "fno-test": "fno/fno_test_n64_n96_n128_dt0p02.h5",
    "cross-resolution": "cross_resolution/cross_resolution_n96_n128_t4p1_t6p0_dt0p1.h5",
    "short-test": "fair_short_horizon/standard_ns_n64_full_spectrum_test_t4p1_t6p0_dt0p1.h5",
}
SAMPLING = {"sampling-clean": "noise_000", "sampling-001": "noise_001", "sampling-010": "noise_010"}
FULL_TRAIN_PARAMETER_HASH = "68e8d17c07e26c4034753022e8106432eacbdb3582da21470ac2b9be3eaacce8"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_file(root, relative):
    if (root / relative).is_symlink():
        raise ValueError("Linked generation files are not assembly inputs")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError("Expected an ordinary file inside its own generation attempt")
    return path


def load_attempt(name, directory):
    root = directory.resolve(strict=True)
    run, receipt = read_json(root / "run.json"), read_json(root / "COMPLETE.json")
    if receipt.get("status") != "COMPLETE_GENERATION":
        raise ValueError(f"{name}: incomplete/pilot generation cannot be assembled")
    binding = run["binding"]
    digest = hashlib.sha256(gen.canonical(binding).encode()).hexdigest()
    if (digest != run["binding_sha256"] or digest != receipt["binding_sha256"] or
            run["attempt_id"] != receipt["attempt_id"]):
        raise ValueError(f"{name}: generation receipt/binding identity mismatch")
    scientific = binding.get("plan", binding)
    if scientific.get("pilot") or scientific.get("subset") is not None:
        raise ValueError(f"{name}: pilot/subset inputs are not full-population data")
    sources = scientific["sources"]
    expected_script = "scripts/generate_data.py" if name in CLEAN_PATHS else "scripts/generate_noise.py" if name == "noise" else "scripts/generate_sampling.py"
    if expected_script not in sources:
        raise ValueError(f"{name}: wrong generation source")
    for relative, expected in sources.items():
        path = (gen.ROOT / relative).resolve(strict=True)
        if gen.ROOT not in path.parents or gen.sha256(path) != expected.lower():
            raise ValueError(f"{name}: generator/solver source hash differs: {relative}")
    return {"name": name, "root": root, "run": run, "receipt": receipt, "scientific": scientific}


def expected_groups(dataset):
    seeds = gen.seed_parameters()
    if dataset == "standard":
        return [(split,64,seeds[split][0], list(range(1000,2001,100)) if split=="test" else list(range(0,2001,4)),gen.array_hash(seeds[split][1])) for split in seeds]
    if dataset in ("fno-training","fno-training-coarse"):
        steps=list(range(0,2001,4 if dataset=="fno-training" else 20))
        return [("training",64,np.r_[np.arange(50),np.arange(1200,2150)],steps,FULL_TRAIN_PARAMETER_HASH),
                ("validation",64,seeds["validation"][0],steps,gen.array_hash(seeds["validation"][1]))]
    grids = (64,96,128) if dataset=="fno-test" else (96,128) if dataset=="cross-resolution" else (64,)
    return [(f"N{n}/test" if dataset!="short-test" else "test",n,seeds["test"][0],
             list(range(820,(1600 if dataset=="fno-test" and n==64 else 1200)+1,4 if dataset=="fno-test" else 20)),
             gen.array_hash(seeds["test"][1])) for n in grids]


def require_finite(field):
    for row in range(field.shape[0]):
        for start in range(0,field.shape[1],16):
            if not np.isfinite(field[row,start:start+16]).all():
                raise ValueError("Nonfinite/unwritten scientific field rejected")


def validate_clean(attempt, *, full, data_path=None):
    """Read a generation attempt, or an already range/hash-checked collection file.

    The optional path is a read-only consumer entry point; callers must enforce
    their collection boundary. All scientific/receipt checks remain identical.
    """
    name, plan = attempt["name"], attempt["scientific"]
    if plan.get("dataset") != name:
        raise ValueError("Generation dataset and assembly slot differ")
    expected = expected_groups(name)
    canonical_protocol=gen.make_plan(argparse.Namespace(dataset="standard",initial_conditions=None,
        subset=None,split="all",pilot_steps=None,batch_size=None,device="cpu"))["physical_protocol"]
    if plan.get("physical_protocol") != canonical_protocol:
        raise ValueError("Physical/numerical protocol differs")
    groups = {g["name"]:g for g in plan["groups"]}
    if set(groups) != {x[0] for x in expected}:
        raise ValueError("Missing or extra population/resolution groups")
    path = checked_file(attempt["root"], "data.h5") if data_path is None else Path(data_path)
    digest = attempt["receipt"]["data_sha256"]
    if full and gen.sha256(path) != digest:
        raise ValueError("Generated data hash differs from completion receipt")
    splits={}
    with h5py.File(path,"r") as handle:
        if (handle.attrs.get("status") != "COMPLETE_GENERATION" or
                handle.attrs.get("attempt_id") != attempt["run"]["attempt_id"] or
                handle.attrs.get("binding_sha256") != attempt["run"]["binding_sha256"] or
                handle.attrs.get("generated_from") != "initial_conditions_not_saved_truth"):
            raise ValueError("H5 completion/provenance binding differs")
        for group_name,n,ids,steps,parameters_hash in expected:
            planned=groups[group_name]
            if (planned["ids"] != ids.tolist() or planned["steps"] != steps or
                    planned["grid"] != n or planned["parameters_sha256"] != parameters_hash or
                    gen.array_hash(planned["parameters"]) != parameters_hash):
                raise ValueError("Population/steps/initial-condition parameters differ")
            group=handle[group_name]
            if Path(group.file.filename).resolve() != path.resolve():
                raise ValueError("Scientific groups must not link to another H5 file")
            times=gen.stored_times(plan,planned)
            if (not np.array_equal(group["trajectory_index"][:],ids) or group["trajectory_index"].dtype!=np.dtype("int64") or
                    not np.array_equal(group["time"][:],times) or group["time"].dtype!=np.dtype("float64")):
                raise ValueError("Stored identity/time axes differ")
            field=group["vorticity"]
            if Path(field.file.filename).resolve() != path.resolve() or field.is_virtual or field.external:
                raise ValueError("Scientific values must be stored in the hashed H5, not external/virtual storage")
            if field.shape!=(len(ids),len(times),n,n) or field.dtype!=np.dtype("float32"):
                raise ValueError("Stored vorticity schema differs")
            parameter_name="initial_parameters" if group_name.startswith("N") else "initial_condition_parameters"
            if gen.array_hash(group[parameter_name][:])!=parameters_hash:
                raise ValueError("Stored initial parameters differ")
            if full:
                require_finite(field)
            splits[group_name]={"trajectory_ids":ids.tolist(),"times":times.tolist(),"shape":list(field.shape),"dtype":"float32"}
        if name=="fno-test":
            metadata=json.loads(handle.attrs["metadata_json"])
            if (metadata.get("schema_version")!="gift.fno.test-data.v1" or metadata.get("status")!="complete" or
                    metadata.get("stored_dt")!=.02 or metadata.get("interpolation_used") is not False):
                raise ValueError("FNO native-test reader metadata differs")
    return [(path,CLEAN_PATHS[name],digest,splits)]


def validate_derivative(attempt, available, *, full, data_paths=None):
    """Validate own outputs; optional paths are range/hash-checked collection inputs."""
    name,binding=attempt["name"],attempt["scientific"]
    if name=="noise":
        required=CLEAN_PATHS["standard"]
    else:
        condition=SAMPLING[name]
        required=CLEAN_PATHS["standard"] if condition=="noise_000" else "m1_parameter_identification/"+condition+".h5"
    if required not in available or binding["input_sha256"]!=available[required]:
        raise ValueError(f"{name}: requires its own newly generated parent input in this collection; released input cannot fill the gap")
    if name!="noise":
        if (binding.get("seed")!=1234 or binding.get("sensors")!=500 or binding.get("time_samples")!=60 or
                binding.get("lhs_points")!=60000 or binding.get("velocity_fft")!="explicit_float64_complex128_original_cache_semantics"):
            raise ValueError("Sampling protocol differs")
        from scripts.generate_sampling import sampling_design
        path=checked_file(attempt["root"],"sampling.npz") if data_paths is None else Path(data_paths["sampling.npz"])
        digest=attempt["receipt"]["sampling_sha256"]
        if full and gen.sha256(path)!=digest:
            raise ValueError("Sampling output hash mismatch")
        expected_shapes={"measurement_coordinates":(30000,3),"targets":(30000,3),"train_indices":(24000,),
                         "validation_indices":(6000,),"lhs_coordinates":(60000,3),"sensor_flat":(500,),"time_indices":(60,),
                         "lower":(3,),"upper":(3,),"trajectory_id":(1,)}
        with np.load(path,allow_pickle=False) as arrays:
            if set(arrays.files)!=set(expected_shapes):
                raise ValueError("Sampling schema differs")
            for key,shape in expected_shapes.items():
                dtype=np.dtype("int64") if key in ("train_indices","validation_indices","sensor_flat","time_indices","trajectory_id") else np.dtype("float32")
                if arrays[key].shape!=shape or arrays[key].dtype!=dtype or not np.isfinite(arrays[key]).all():
                    raise ValueError("Sampling shape/dtype/finiteness differs")
            design=sampling_design((np.arange(501)*.02).astype(np.float32),64,np.random.RandomState(1234))
            if any(not np.array_equal(arrays[key],value) for key,value in design.items()):
                raise ValueError("Sampling random design/split differs")
            split_record={"measurement_train_indices":arrays["train_indices"].tolist(),
                          "measurement_validation_indices":arrays["validation_indices"].tolist(),
                          "sensor_flat":arrays["sensor_flat"].tolist(),"time_indices":arrays["time_indices"].tolist(),
                          "trajectory_id":0,"flatten_order":"sensor,time,C"}
        return [(path,"auxiliary/pinn_sampling/"+SAMPLING[name]+"_seed1234.npz",digest,split_record)]
    if (binding.get("seed")!=0 or binding.get("rng")!="MT19937" or binding.get("scaling")!="std(clean_group,ddof=0)" or
            binding.get("group_order")!=["training","validation"] or binding.get("frame_chunk")!=16 or
            binding.get("fractions")!={"noise_001":.01,"noise_010":.10}):
        raise ValueError("Noise protocol differs")
    result=[]
    for condition in ("noise_001","noise_010"):
        path=checked_file(attempt["root"],condition+".h5") if data_paths is None else Path(data_paths[condition+".h5"])
        digest=attempt["receipt"]["outputs"][condition+".h5"]
        if full and gen.sha256(path)!=digest:
            raise ValueError("Noise output hash mismatch")
        splits={}
        with h5py.File(path,"r") as handle:
            if handle.attrs.get("status")!="COMPLETE_GENERATION" or handle.attrs.get("attempt_id")!=attempt["run"]["attempt_id"]:
                raise ValueError("Noise H5 completion binding differs")
            for group_name,ids in (("training",np.arange(50)),("validation",np.arange(50,70))):
                group=handle[group_name]
                field=group["vorticity"]
                if (Path(group.file.filename).resolve()!=path.resolve() or
                        Path(field.file.filename).resolve()!=path.resolve() or field.is_virtual or field.external):
                    raise ValueError("Noise values must be stored in the hashed H5, not external/virtual storage")
                times=np.arange(501)*.02
                if (group["vorticity"].shape!=(len(ids),501,64,64) or group["vorticity"].dtype!=np.dtype("float32") or
                        not np.array_equal(group["trajectory_index"][:],ids) or
                        not np.allclose(group["time"][:],times,atol=2e-12,rtol=0)):
                    raise ValueError("Noise schema/IDs/time mismatch")
                if full:
                    require_finite(group["vorticity"])
                splits[group_name]={"trajectory_ids":ids.tolist(),"times":group["time"][:].tolist(),"shape":list(group["vorticity"].shape),"dtype":"float32"}
        result.append((path,"m1_parameter_identification/"+condition+".h5",digest,splits))
    return result


def copy_new(source, destination, expected):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src,dst,length=8*1024*1024)
    if gen.sha256(destination)!=expected:
        raise ValueError("Copied file hash mismatch; incomplete collection retained for inspection")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",action="append",required=True,help="Repeat NAME=attempt_directory; see docs/DATA_GENERATION.md")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true",help="Without this flag: schema/provenance plan, no full-array scan/copy")
    args=parser.parse_args(argv)
    output=gen.safe_output(args.output)
    if output.exists():
        raise FileExistsError("Assembly output must be a new directory; no refresh/overwrite")
    attempts={}
    allowed=set(CLEAN_PATHS)|set(SAMPLING)|{"noise"}
    for item in args.job:
        name,separator,directory=item.partition("=")
        if not separator or name not in allowed or name in attempts:
            raise ValueError("Invalid/duplicate job slot")
        attempts[name]=load_attempt(name,Path(directory))
        if output==attempts[name]["root"] or output in attempts[name]["root"].parents or attempts[name]["root"] in output.parents:
            raise ValueError("Collection output must be separate from generation attempts")
    files,available,splits=[],{},{}
    order=list(CLEAN_PATHS)+["noise"]+list(SAMPLING)
    for name in order:
        if name not in attempts:
            continue
        attempt=attempts[name]
        records=validate_clean(attempt,full=args.execute) if name in CLEAN_PATHS else validate_derivative(attempt,available,full=args.execute)
        for path,relative,digest,groups in records:
            files.append((path,relative,digest,name))
            available[relative]=digest
            splits[relative]=groups
    if not args.execute:
        print(json.dumps({"status":"ASSEMBLY_PLAN_ONLY","files":list(available),"full_hash_and_finite_checks_performed":False},indent=2))
        return 0
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for source,relative,digest,name in files:
        copy_new(source,output/relative,digest)
        records.append({"path":relative,"bytes":source.stat().st_size,"sha256":digest,
                        "category":"generated_scientific_input","generation_job":name,
                        "attempt_id":attempts[name]["run"]["attempt_id"]})
    for name,attempt in attempts.items():
        for filename in ("run.json","COMPLETE.json"):
            source=attempt["root"]/filename
            relative="provenance/"+name+"/"+filename
            digest=gen.sha256(source)
            copy_new(source,output/relative,digest)
            records.append({"path":relative,"bytes":source.stat().st_size,"sha256":digest,"category":"generation_provenance"})
    gen.write_json_new(output/"splits.json",{"schema":"gift.generated-splits.v1","files":splits})
    records.append({"path":"splits.json","bytes":(output/"splits.json").stat().st_size,"sha256":gen.sha256(output/"splits.json"),"category":"split_metadata"})
    manifest={"schema":"gift.generated-data-collection.v1","data_profile":"regenerated",
              "status":"ASSEMBLED_SCHEMA_VERIFIED_NOT_EXPERIMENT_ACCEPTED","files":records,
              "available_jobs":list(attempts),"missing_jobs":sorted(allowed-set(attempts)),
              "scientific_arrays_modified":False,"released_truth_used_to_fill_gaps":False,
              "full_hash_and_finite_checks_performed":True,"reference_field_comparison_performed":False,
              "training_profile_integration_verified":False,"initialization_included":False,
              "assembler_sha256":gen.sha256(__file__)}
    gen.write_json_new(output/"manifest.json",manifest)
    print(json.dumps({"status":manifest["status"],"files":len(records),"output":str(output),"manifest_sha256":gen.sha256(output/"manifest.json")},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
