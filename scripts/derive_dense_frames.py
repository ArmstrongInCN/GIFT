"""Select exact integer frames from a completed NEW dense generation attempt.

This is derivation, not another solver invocation or reference-data recovery.
Fields are copied without interpolation or model imports. Qualification recreates
seeded initial parameters; the copy operation itself has no RNG state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import generate_data as gen

SCHEMA = "gift.dense-frame-derivation.v1"
STATUS = "COMPLETE_DERIVATION"
ORIGIN = "integer_frames_of_new_generation"
# Each dense target reuses a complete native generation as its only parent: the
# coarse/fair-horizon/cross-resolution views are exact sub-selections of it.
PARENTS = {"fno-training-coarse": "fno-training", "short-test": "fno-test",
           "cross-resolution": "fno-test"}
BLOCK_FRAMES = 16


def digest_json(value):
    return hashlib.sha256(gen.canonical(value).encode()).hexdigest()


def parameter_name(group_name):
    return "initial_parameters" if group_name.startswith("N") else "initial_condition_parameters"


def source_hashes():
    return {relative: gen.sha256(gen.ROOT / relative) for relative in (
        "scripts/derive_dense_frames.py", "scripts/assemble_generated_data.py",
        "scripts/generate_data.py", "src/even_full_spectrum_ns.py")}


def local_storage(handle):
    """Reject links/storage outside the hashed H5, including on small axes.

    Inspect links before dereferencing them. Hard links within the file are safe;
    object addresses prevent recursion through hard-linked group cycles.
    """
    # Only data stored inside the hashed H5 may be a derivation input or output;
    # external/soft/virtual storage would escape the byte-hash the receipt binds.
    visited = set()

    def visit(group):
        address = h5py.h5o.get_info(group.id).addr
        if address in visited:
            return
        visited.add(address)
        for name in group:
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                raise ValueError("External/soft H5 links are not derivation inputs or outputs")
            item = group[name]
            if isinstance(item, h5py.Group):
                visit(item)
            elif item.is_virtual or item.external:
                raise ValueError("Virtual/external dataset storage is forbidden")

    visit(handle)


def reject_fixture(runtime, handle=None):
    # Public acceptance has no reduced-size/mock override. Tests patch boundaries
    # explicitly instead of introducing an undocumented production bypass.
    # Fixture/mock receipts are never accepted as scientific generation evidence.
    if any(marker in gen.canonical(runtime).lower() for marker in ("fixture", "mock")):
        raise ValueError("Fixture/mock receipts are not scientific generation evidence")
    if handle is not None and any(marker in key.lower() for key in handle.attrs
                                  for marker in ("fixture", "mock")):
        raise ValueError("Fixture/mock H5 is not scientific generation evidence")


def inspect_parent(attempt, *, full):
    """Require the complete native integrator protocol, never a naked/derived H5."""
    from scripts import assemble_generated_data as assembly

    plan, run, receipt = attempt["scientific"], attempt["run"], attempt["receipt"]
    if (attempt["name"] not in set(PARENTS.values()) or
            plan.get("dataset") != attempt["name"] or
            receipt.get("status") != "COMPLETE_GENERATION" or
            plan.get("schema") != "gift.data-generation.v1" or
            plan.get("pilot") is not False or plan.get("subset") is not None or
            plan.get("dt") != gen.DT):
        raise ValueError("Parent must be a complete new native fno-training/fno-test generation")
    expected_sources = {key: value for key, value in source_hashes().items()
                        if key in ("scripts/generate_data.py", "src/even_full_spectrum_ns.py")}
    if plan.get("sources") != expected_sources:
        raise ValueError("Parent generator/solver source identity differs")
    runtime = run["binding"].get("runtime", {})
    reject_fixture(runtime)
    if not {"python", "numpy", "h5py", "torch", "device", "threads", "interop_threads"} <= set(runtime):
        raise ValueError("Parent integrator runtime receipt is incomplete")
    if (receipt.get("reference_comparison_performed") is not False or
            receipt.get("full_budget_reproduction_claim") is not False):
        raise ValueError("Parent receipt is not the native generation-only completion schema")
    if plan["dataset"] == "fno-training":
        initial_hash = plan.get("initial_conditions_file_sha256", "")
        if (not isinstance(initial_hash, str) or len(initial_hash) != 64 or
                any(c not in "0123456789abcdef" for c in initial_hash) or
                plan.get("extra950_origin") != "specified_initial_conditions_seed_unknown"):
            raise ValueError("FNO parent must preserve the explicit extra-950 parameter provenance")
    elif plan.get("initial_conditions_file_sha256") is not None or plan.get("extra950_origin") is not None:
        raise ValueError("Native test parent must use its original seeded test parameters")
    path = assembly.checked_file(attempt["root"], "data.h5")
    with h5py.File(path, "r") as handle:
        local_storage(handle)
        reject_fixture(runtime, handle)
        for group in plan["groups"]:
            obj = handle[group["name"]]
            parameters = obj[parameter_name(group["name"])]
            if (parameters.shape != (len(group["ids"]), 4, 4) or
                    parameters.dtype != np.dtype("float64") or not np.isfinite(parameters[:]).all()):
                raise ValueError("Parent parameter dtype/shape/finiteness differs")
            if not isinstance(json.loads(obj.attrs["solver_metadata"]), dict):
                raise ValueError("Missing parent solver metadata")
    assembly.validate_clean(attempt, full=full)
    return path


def load_parent(target, directory, *, full):
    from scripts import assemble_generated_data as assembly

    root = Path(directory).resolve(strict=True)
    for name in ("run.json", "COMPLETE.json"):
        assembly.checked_file(root, name)
    attempt = assembly.load_attempt(PARENTS[target], root)
    inspect_parent(attempt, full=full)
    return attempt


def parent_identity(parent):
    return {"dataset": parent["name"], "attempt_id": parent["run"]["attempt_id"],
            "binding_sha256": parent["run"]["binding_sha256"],
            "data_sha256": parent["receipt"]["data_sha256"],
            "run_sha256": gen.sha256(parent["root"] / "run.json"),
            "completion_sha256": gen.sha256(parent["root"] / "COMPLETE.json")}


def make_plan(target, parent):
    from scripts import assemble_generated_data as assembly

    if PARENTS[target] != parent["name"]:
        raise ValueError("Wrong dense parent for this target")
    parents = {group["name"]: group for group in parent["scientific"]["groups"]}
    groups = []
    for name, grid, ids, steps, parameters_hash in assembly.expected_groups(target):
        source_name = "N64/test" if target == "short-test" else name
        source = parents[source_name]
        lookup = {step: column for column, step in enumerate(source["steps"])}
        if (len(lookup) != len(source["steps"]) or source["ids"] != ids.tolist() or
                source["grid"] != grid or source["parameters_sha256"] != parameters_hash or
                gen.array_hash(source["parameters"]) != parameters_hash):
            raise ValueError("Parent IDs/grid/parameters/steps do not match the target")
        if any(type(step) is not int or step not in lookup for step in steps):
            raise ValueError("Target must select existing integer solver steps")
        groups.append({"name": name, "parent_group": source_name, "grid": grid,
                       "ids": ids.tolist(), "steps": steps,
                       "parent_columns": [lookup[step] for step in steps],
                       "parameters": source["parameters"], "parameters_sha256": parameters_hash})
    return {"schema": SCHEMA, "dataset": target, "pilot": False, "subset": None,
            # Dense derivation copies exact integer solver frames with no
            # interpolation; stored_dt is 0.1 (every 20th native step at dt 0.005).
            "operation": "integer_frame_selection", "interpolation_used": False,
            "field_copy": "float32_bitwise_no_cast", "dt": gen.DT, "stored_dt": .1,
            "physical_protocol": parent["scientific"]["physical_protocol"],
            "parent": parent_identity(parent), "groups": groups, "sources": source_hashes()}


def metadata(plan):
    # Do not copy the dense native-test metadata, whose stored_dt is 0.02.
    return {"schema_version": SCHEMA, "dataset": plan["dataset"], "stored_dt": .1,
            "interpolation_used": False, "parent": plan["parent"],
            "copied_reference_truth": False, "new_solver_invocation": False}


def chunks(plan):
    return [(group, row, start, min(start + BLOCK_FRAMES, len(group["steps"])))
            for group in plan["groups"] for row in range(len(group["ids"]))
            for start in range(0, len(group["steps"]), BLOCK_FRAMES)]


def bits_equal(left, right):
    """Numerical equality alone would silently consider +0 and -0 equal."""
    # Compare raw little-endian bytes, so a derived frame is accepted only if it is
    # bitwise identical to its parent; floating-point equivalence is not enough.
    return (left.dtype == right.dtype == np.dtype("float32") and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def compare_chunk(source, output, task):
    group, row, start, stop = task
    expected = source[group["parent_group"] + "/vorticity"][row, group["parent_columns"][start:stop]]
    actual = output[group["name"] + "/vorticity"][row, start:stop]
    if not np.isfinite(expected).all() or not np.isfinite(actual).all() or not bits_equal(actual, expected):
        raise ValueError("Derived field is nonfinite or differs bitwise from its own parent")


def check_inherited_metadata(source, output, plan):
    for group in plan["groups"]:
        if (output[group["name"]].attrs.get("inherited_parent_solver_metadata") !=
                source[group["parent_group"]].attrs["solver_metadata"]):
            raise ValueError("Inherited solver metadata differs from the declared parent")


def inspect_output(handle, plan, run, *, statuses):
    local_storage(handle)
    if (handle.attrs.get("status") not in statuses or
            handle.attrs.get("attempt_id") != run["attempt_id"] or
            handle.attrs.get("binding_sha256") != run["binding_sha256"] or
            handle.attrs.get("generated_from") != ORIGIN or
            handle.attrs.get("internal_dt") != gen.DT or handle.attrs.get("stored_dt") != .1 or
            json.loads(handle.attrs.get("derivation_metadata_json", "null")) != metadata(plan) or
            "metadata_json" in handle.attrs):
        raise ValueError("Derived H5 identity/metadata differs")
    splits = {}
    expected_datasets = set()
    for group in plan["groups"]:
        name, n = group["name"], group["grid"]
        obj = handle[name]
        names = {"trajectory_index", "time", "vorticity", parameter_name(name)}
        expected_datasets.update(name + "/" + item for item in names)
        if set(obj) != names:
            raise ValueError("Unexpected derived group schema")
        times = gen.stored_times(plan, group)
        if (obj["trajectory_index"].dtype != np.dtype("int64") or
                not np.array_equal(obj["trajectory_index"][:], group["ids"]) or
                obj["time"].dtype != np.dtype("float64") or not np.array_equal(obj["time"][:], times) or
                obj[parameter_name(name)].dtype != np.dtype("float64") or
                obj[parameter_name(name)].shape != (len(group["ids"]), 4, 4) or
                gen.array_hash(obj[parameter_name(name)][:]) != group["parameters_sha256"] or
                obj["vorticity"].dtype != np.dtype("float32") or
                obj["vorticity"].shape != (len(group["ids"]), len(times), n, n)):
            raise ValueError("Derived axes/parameters/shape/dtype differ")
        splits[name] = {"trajectory_ids": group["ids"], "times": times.tolist(),
                        "shape": list(obj["vorticity"].shape), "dtype": "float32"}
    present = set()
    handle.visititems(lambda name, obj: present.add(name) if isinstance(obj, h5py.Dataset) else None)
    if present != expected_datasets:
        raise ValueError("Missing/extra derived datasets")
    return splits


@contextmanager
def writer_lock(output):
    """One nonblocking writer per attempt, including final hash/publication.

    The lock file is persistent, not a stale-PID sentinel. OS locks release on
    process exit; no polling, lock-file removal or force-unlock is performed.
    """
    lock_path = output / ".writer.lock"
    if lock_path.is_symlink() or lock_path.resolve().parent != output.resolve():
        raise ValueError("Writer lock must be an ordinary file in its own attempt")
    with lock_path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another derivation writer holds this attempt lock") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another derivation writer holds this attempt lock") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_completion_stage(path, receipt):
    """A torn write may leave this staging file, never a partial COMPLETE.json."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish_completion(output, receipt):
    """Atomic create-only publication while holding the attempt writer lock.

    Same-directory hard-link creation publishes only the closed, synced inode,
    and fails if COMPLETE already exists. Keep staging files, including partial
    ones from interrupted writes. Unsupported filesystems fail closed; never fall
    back to a direct write or replace another completion receipt.
    """
    # Stage to a uniquely named file, sync it, then publish via a same-directory
    # hard link that fails if COMPLETE.json already exists (create-only guarantee).
    stage = output / ("COMPLETE." + uuid.uuid4().hex + ".staging.json")
    write_completion_stage(stage, receipt)
    os.link(stage, output / "COMPLETE.json")


def copy_attempt(parent, plan, output, *, resume=False, checkpoint_every=32, stop_after_chunks=None):
    """Hold the same-attempt writer lock through the entire mutation lifecycle."""
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with writer_lock(output):
        return _copy_attempt_locked(parent, plan, output, resume=resume,
                                    checkpoint_every=checkpoint_every, stop_after_chunks=stop_after_chunks)


def _copy_attempt_locked(parent, plan, output, *, resume, checkpoint_every, stop_after_chunks):
    """Copy-only state machine; the public CLI validates the full parent first.

    Immutable checkpoint JSONs bind the next block to this attempt. Every committed
    prefix block is rechecked against the same parent on resume. Uncommitted blocks
    may be rewritten; no foreign cursor, data refresh or completed attempt is reused.
    """
    from scripts import assemble_generated_data as assembly

    # Recheck only AFTER acquiring the writer lock: a pre-lock observation may
    # have preceded another invocation's completion publication.
    if (output / "COMPLETE.json").exists():
        raise ValueError("Attempt already complete; derivation is create-only")
    source_path = assembly.checked_file(parent["root"], "data.h5")
    binding = {"plan": plan, "runtime": {"python": platform.python_version(),
               "platform": platform.platform(), "numpy": np.__version__, "h5py": h5py.__version__},
               "block_frames": BLOCK_FRAMES}
    binding_hash, tasks = digest_json(binding), chunks(plan)
    if resume:
        run = assembly.read_json(assembly.checked_file(output, "run.json"))
        if run["binding"] != binding or run["binding_sha256"] != binding_hash:
            raise ValueError("Resume parent/source/protocol/runtime mismatch")
        latest = assembly.read_json(assembly.checked_file(output, "latest.json"))
        filename = latest["file"]
        if Path(filename).name != filename or not filename.startswith("checkpoint_") or not filename.endswith(".json"):
            raise ValueError("Checkpoint must be local to its own attempt")
        cursor_path = assembly.checked_file(output, filename)
        if gen.sha256(cursor_path) != latest["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        cursor = assembly.read_json(cursor_path)
        if cursor["attempt_id"] != run["attempt_id"] or cursor["binding_sha256"] != binding_hash:
            raise ValueError("Foreign cursor rejected")
        position = cursor["next_chunk"]
        if type(position) is not int or not 0 <= position <= len(tasks):
            raise ValueError("Invalid cursor position")
        path, mode = assembly.checked_file(output, "data.h5"), "r+"
    else:
        run = {"attempt_id": str(uuid.uuid4()), "binding_sha256": binding_hash, "binding": binding}
        gen.write_json_new(output / "run.json", run)
        position, path, mode = 0, output / "data.h5", "x"
    with h5py.File(source_path, "r") as source, h5py.File(path, mode, locking=True) as handle:
        if mode == "x":
            handle.attrs.update(attempt_id=run["attempt_id"], binding_sha256=binding_hash,
                                status="INCOMPLETE", generated_from=ORIGIN,
                                internal_dt=gen.DT, stored_dt=.1,
                                derivation_metadata_json=gen.canonical(metadata(plan)))
            for group in plan["groups"]:
                obj = handle.create_group(group["name"])
                obj.create_dataset("trajectory_index", data=np.asarray(group["ids"], dtype=np.int64))
                obj.create_dataset("time", data=gen.stored_times(plan, group))
                # Copy parameter bytes under the destination reader's required name.
                obj.create_dataset(parameter_name(group["name"]),
                                   data=source[group["parent_group"]][parameter_name(group["parent_group"])][:])
                obj.attrs["inherited_parent_solver_metadata"] = source[group["parent_group"]].attrs["solver_metadata"]
                n = group["grid"]
                obj.create_dataset("vorticity", shape=(len(group["ids"]), len(group["steps"]), n, n),
                                   dtype="f4", chunks=(1, 1, n, n), compression="lzf", shuffle=True, fillvalue=np.nan)
        inspect_output(handle, plan, run, statuses={"INCOMPLETE", STATUS})
        check_inherited_metadata(source, handle, plan)

        def save(next_chunk):
            handle.flush()
            checkpoint = output / ("checkpoint_" + uuid.uuid4().hex + ".json")
            gen.write_json_new(checkpoint, {"attempt_id": run["attempt_id"],
                               "binding_sha256": binding_hash, "next_chunk": next_chunk})
            gen.atomic_json(output / "latest.json", {"file": checkpoint.name, "sha256": gen.sha256(checkpoint)})

        if not resume:
            save(0)
        for task in tasks[:position]:
            compare_chunk(source, handle, task)
        for index in range(position, len(tasks)):
            if stop_after_chunks is not None and index - position >= stop_after_chunks:
                save(index)
                print(json.dumps({"status": "PAUSED", "attempt_id": run["attempt_id"],
                                  "next_chunk": index, "total_chunks": len(tasks)}), flush=True)
                return 0
            group, row, start, stop = tasks[index]
            values = source[group["parent_group"] + "/vorticity"][row, group["parent_columns"][start:stop]]
            if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
                raise ValueError("Nonfinite or non-float32 parent field")
            handle[group["name"] + "/vorticity"][row, start:stop] = values
            if (index + 1) % checkpoint_every == 0:
                save(index + 1)
        save(len(tasks))
        # A completion means every selected float32 bit was checked, not merely
        # that the copy loop ended. The full parent was checked by the CLI.
        for task in tasks:
            compare_chunk(source, handle, task)
        if parent_identity(parent) != plan["parent"] or gen.sha256(source_path) != plan["parent"]["data_sha256"]:
            raise ValueError("Parent changed during derivation")
        handle.attrs["status"] = STATUS
        handle.flush()
    receipt = {"status": STATUS, "schema": SCHEMA,
                       "attempt_id": run["attempt_id"], "binding_sha256": binding_hash,
                       "data_sha256": gen.sha256(path), "parent": plan["parent"],
                       "selected_fields_bitwise_verified": True, "reference_comparison_performed": False,
                       "new_solver_invocation": False, "full_budget_reproduction_claim": False}
    if source_hashes() != plan["sources"]:
        raise ValueError("Source files changed during derivation; completion not published")
    publish_completion(output, receipt)
    print(json.dumps({"status": STATUS, "attempt_id": run["attempt_id"], "output": str(output)}), flush=True)
    return 0


def load_derivation(name, root):
    """Distinct assembler dispatch: never relabel derivation as integration."""
    from scripts import assemble_generated_data as assembly

    run = assembly.read_json(assembly.checked_file(root, "run.json"))
    receipt = assembly.read_json(assembly.checked_file(root, "COMPLETE.json"))
    binding, plan = run["binding"], run["binding"]["plan"]
    if (name not in PARENTS or plan.get("schema") != SCHEMA or plan.get("dataset") != name or
            receipt.get("status") != STATUS or receipt.get("schema") != SCHEMA or
            plan.get("pilot") is not False or plan.get("subset") is not None or
            plan.get("sources") != source_hashes() or binding.get("block_frames") != BLOCK_FRAMES):
        raise ValueError("Invalid derivation schema/slot/source/protocol")
    if (digest_json(binding) != run["binding_sha256"] or receipt["binding_sha256"] != run["binding_sha256"] or
            receipt["attempt_id"] != run["attempt_id"] or receipt.get("parent") != plan["parent"] or
            receipt.get("selected_fields_bitwise_verified") is not True or
            receipt.get("reference_comparison_performed") is not False or
            receipt.get("new_solver_invocation") is not False or
            receipt.get("full_budget_reproduction_claim") is not False):
        raise ValueError("Derivation receipt/binding identity mismatch")
    reject_fixture(binding["runtime"])
    return {"name": name, "root": root, "run": run, "receipt": receipt, "scientific": plan}


def validate_for_assembly(attempt, attempts, available, *, full):
    from scripts import assemble_generated_data as assembly

    name, plan = attempt["name"], attempt["scientific"]
    parent_name = PARENTS[name]
    if (parent_name not in attempts or assembly.CLEAN_PATHS[parent_name] not in available or
            available[assembly.CLEAN_PATHS[parent_name]] != plan["parent"]["data_sha256"]):
        raise ValueError("Derivation requires its same-SHA newly generated dense parent in this collection")
    parent = attempts[parent_name]
    # The assembler already full-validated this earlier parent slot on execution.
    source_path = inspect_parent(parent, full=False)
    if plan != make_plan(name, parent):
        raise ValueError("Derivation parent receipt or exact integer mapping differs")
    path = assembly.checked_file(attempt["root"], "data.h5")
    digest = attempt["receipt"]["data_sha256"]
    if full and gen.sha256(path) != digest:
        raise ValueError("Derived data hash differs from completion receipt")
    with h5py.File(path, "r") as handle, h5py.File(source_path, "r") as source:
        reject_fixture(attempt["run"]["binding"]["runtime"], handle)
        splits = inspect_output(handle, plan, attempt["run"], statuses={STATUS})
        check_inherited_metadata(source, handle, plan)
        if full:
            for task in chunks(plan):
                compare_chunk(source, handle, task)
    return [(path, assembly.CLEAN_PATHS[name], digest, splits)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(PARENTS), required=True)
    parser.add_argument("--parent", type=Path, required=True, help="Completed new dense attempt directory, not an H5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--execute", action="store_true", help="Otherwise read-only schema plan, no full-array/hash verification")
    args = parser.parse_args(argv)
    if args.checkpoint_every < 1 or (args.stop_after_chunks is not None and args.stop_after_chunks < 0):
        parser.error("Invalid checkpoint/pause interval")
    output, parent_root = gen.safe_output(args.output), args.parent.resolve(strict=True)
    if output == parent_root or output in parent_root.parents or parent_root in output.parents:
        raise ValueError("Output must be separate from its parent attempt")
    if output.exists() and not args.resume:
        raise FileExistsError("Use a NEW output directory, or resume its own incomplete attempt")
    parent = load_parent(args.dataset, parent_root, full=args.execute)
    plan = make_plan(args.dataset, parent)
    if not args.execute:
        print(json.dumps({"status": "DERIVATION_PLAN_ONLY", "plan": plan,
                          "full_hash_and_finite_checks_performed": False}, indent=2))
        return 0
    return copy_attempt(parent, plan, output, resume=args.resume,
                        checkpoint_every=args.checkpoint_every, stop_after_chunks=args.stop_after_chunks)


if __name__ == "__main__":
    raise SystemExit(main())
