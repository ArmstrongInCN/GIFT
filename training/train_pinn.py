"""KC-only phase coordinator. Tiny integration passed; full budget is unverified.

Amendment002: every NAdam stage updates network AND coefficients using the same
optimizer state. This file schedules externally supplied algorithms; it contains
no upstream network, NAdam, L-BFGS-B or STRidge implementation.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import random
import re
import uuid

import numpy as np

from adapters.pinn_runtime import _json, _writer, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET = {"pre_nadam": 5000, "rounds": 6, "ado_nadam": 1000,
                  "post_nadam": 20000, "lbf_maxiter": 10000, "lbf_maxfun": 10000}
PHASES = ("pre_nadam", "pre_lbfgs", "ado_stridge", "ado_nadam", "final_mask", "post_nadam", "complete")
TINY_INTEGRATION_VALIDATED = True
LIBRARY = ["w", "advection", "laplacian", "q"]


def budget_for(overrides=None, diagnostic=False):
    values = dict(DEFAULT_BUDGET)
    if overrides:
        if set(overrides) - set(values):
            raise ValueError("Unknown phase budget")
        values.update(overrides)
    if any(type(v) is not int or v <= 0 for v in values.values()):
        raise ValueError("Phase budgets must be positive integers")
    if values["rounds"] > 6:
        raise ValueError("The external STRidge contract supports at most six ADO rounds")
    if values != DEFAULT_BUDGET and not diagnostic:
        raise ValueError("Modified budgets require explicit diagnostic/test-only status")
    return values


def callable_binding(value):
    return {"name": value.__module__ + "." + value.__qualname__, "source_sha256": sha256(inspect.getsourcefile(value))}


class TensorFlowBackend:
    """Explicit real-graph bridge; constructing this class performs graph setup."""
    kind = "real_tensorflow_kc"

    def __init__(self, initialization, sampling, data, external_root=None, train_rows=None, physics_rows=None, chunk=32768, device="cpu"):
        from adapters.pinn_runtime import PINNRuntime, read_inputs
        initial, self.coordinates, self.targets, self.physics, lower, upper, bound = read_inputs(
            initialization, sampling, data, train_rows, physics_rows)
        self.runtime = PINNRuntime("known", lower, upper, initial, external_root, device=device)
        self.chunk = chunk
        if chunk <= 0:
            raise ValueError("Positive chunk required")
        self.identity = {"inputs": bound, "chunk": chunk, "profile": self.runtime.profile,
                         "external_sources": self.runtime.source_records,
                         "parameter_order": [v.name for v in self.runtime.parameters],
                         "source_sha256": sha256(ROOT / "adapters/pinn_runtime.py")}
        tf = self.runtime.tf
        with self.runtime.graph.as_default():
            self.flat_input = tf.compat.v1.placeholder(tf.float32, [None])
            offset, operations = 0, []
            for variable in self.runtime.parameters:
                count = int(np.prod(variable.shape.as_list()))
                operations.append(variable.assign(tf.reshape(self.flat_input[offset:offset+count], variable.shape)))
                offset += count
            self.assign_operation = tf.group(*operations)
            self.parameter_count = offset

    @property
    def layout(self):
        return [{"name": v.name, "shape": v.shape.as_list(), "dtype": v.dtype.name} for v in self.runtime.variables]

    def state(self):
        return self.runtime.state()

    def restore(self, state):
        self.runtime.restore(state)

    def flat(self):
        return np.concatenate([v.reshape(-1) for v in self.runtime.session.run(self.runtime.parameters)]).astype(np.float64)

    def assign_flat(self, value):
        value = np.asarray(value, np.float32)
        if value.shape != (self.parameter_count,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid parameter vector")
        self.runtime.session.run(self.assign_operation, {self.flat_input: value})

    def objective(self, value):
        self.assign_flat(value)
        metrics, gradients = self.runtime.objective(self.coordinates, self.targets, self.physics, chunk=self.chunk)
        return metrics["loss"], np.concatenate([g.reshape(-1) for g in gradients]).astype(np.float64)

    def nadam(self, lr, physics, l1):
        return self.runtime.step(self.coordinates, self.targets, self.physics, chunk=self.chunk,
                                 learning_rate=lr, physics_weight=physics, l1_weight=l1)

    def coefficients(self):
        return self.runtime.session.run(self.runtime.parameters[-1]).reshape(-1)

    def mask(self):
        return self.runtime.mask.reshape(-1).copy()

    def assign_coefficients(self, value, mask=None):
        self.runtime.set_coefficients(np.asarray(value, np.float32).reshape(4, 1),
                                      None if mask is None else np.asarray(mask, np.float32).reshape(4, 1))

    def collect(self):
        libraries, targets = [], []
        for start in range(0, len(self.physics), self.chunk):
            library, omega_t = self.runtime.library(self.physics[start:start+self.chunk])
            libraries.append(library)
            targets.append(omega_t)
        return np.concatenate(libraries, axis=0), np.concatenate(targets, axis=0)

    def close(self):
        self.runtime.close()


class Coordinator:
    """One NAdam update, one solver boundary or one STRidge call per advance."""
    def __init__(self, backend, stridge, solver_factory=None, budget=None, diagnostic=False,
                 mode="known", external_root=None, nadam_interval=1000, lbfgs_interval=100):
        if mode != "known":
            raise ValueError("Full phase scheduling is KC/known only; open remains blocked")
        self.budget = budget_for(budget, diagnostic)
        if backend.kind != "real_tensorflow_kc" and not diagnostic:
            raise ValueError("Mock backends must be explicitly diagnostic/test-only")
        if backend.identity.get("inputs", {}).get("selection", {}).get("diagnostic_subset") and not diagnostic:
            raise ValueError("Limited input rows require explicit diagnostic/test-only status")
        if nadam_interval <= 0 or lbfgs_interval <= 0:
            raise ValueError("Positive checkpoint intervals required")
        if solver_factory is None:
            from training.pinn_lbfgsb import LBFGSB
            solver_factory = LBFGSB
        self.backend, self.stridge, self.solver_factory = backend, stridge, solver_factory
        self.external_root, self.diagnostic = external_root, diagnostic
        self.intervals = {"nadam": nadam_interval, "lbfgs": lbfgs_interval}
        self.identity = json.loads(json.dumps({"protocol": "Amendment002_joint_nadam_final_mask_after_round6",
            "source_sha256": sha256(Path(__file__)), "backend": backend.identity, "kind": backend.kind,
            "stridge": callable_binding(stridge), "lbfgsb": callable_binding(solver_factory),
            "budget": self.budget, "diagnostic_test_only": diagnostic, "checkpoint_intervals": self.intervals}))
        self.state = {"phase": "pre_nadam", "phase_step": 0, "round": 0, "nadam_total": 0,
                      "serial": 0, "l0_penalty": None, "final_mask": False, "stridge_records": [], "lbfgs_summary": None}
        self.solver, self.lbf_initial = None, None
        self.checkpoint_due = True
        if backend.state()["iteration"] != 0:
            raise ValueError("Fresh coordinator requires a fresh runtime, not trained state")

    def _solver(self, initial):
        return self.solver_factory(initial, {"phase": "pre_lbfgs", "run_identity": self.identity},
                                   options={"maxiter": self.budget["lbf_maxiter"], "maxfun": self.budget["lbf_maxfun"]})

    def _phase(self, name):
        self.state["phase"], self.state["phase_step"] = name, 0

    def advance(self):
        state, phase = self.state, self.state["phase"]
        if phase == "complete":
            raise RuntimeError("Completed phase schedule cannot be executed again")
        due, event = False, {"phase": phase, "round": state["round"]}
        if phase in ("pre_nadam", "ado_nadam", "post_nadam"):
            pre = phase == "pre_nadam"
            event["metrics"] = self.backend.nadam(1e-3 if pre else 1e-4, 1.0 if pre else 2.0, 1e-7 if pre else 0.0)
            state["phase_step"] += 1
            state["nadam_total"] += 1
            due = state["phase_step"] % self.intervals["nadam"] == 0
            if state["phase_step"] == self.budget[phase]:
                if pre:
                    self._phase("pre_lbfgs")
                elif phase == "post_nadam":
                    self._phase("complete")
                elif state["round"] + 1 == self.budget["rounds"]:
                    self._phase("final_mask")
                else:
                    state["round"] += 1
                    self._phase("ado_stridge")
        elif phase == "pre_lbfgs":
            if self.solver is None:
                self.lbf_initial = self.backend.flat()
                self.solver = self._solver(self.lbf_initial)
            event["solver"] = self.solver.advance(self.backend.objective)
            count = self.solver.counts
            due = ((self.solver.task.startswith("FG") and count["fg_requests"] % self.intervals["lbfgs"] == 0)
                   or (self.solver.task.startswith("NEW_X") and count["iterations"] % self.intervals["lbfgs"] == 0))
            if self.solver.done:
                self.backend.assign_flat(self.solver.x)
                state["lbfgs_summary"] = {"task": self.solver.task, "counts": dict(count),
                                           "loss": float(self.solver.arrays["f"]), "converged": self.solver.task.startswith("CONV")}
                # Original minimize termination (including budget-stop) is
                # retained, not relabelled as convergence or acceptance.
                self.solver, self.lbf_initial = None, None
                self._phase("ado_stridge")
        elif phase == "ado_stridge":
            if not np.array_equal(self.backend.mask(), np.ones(4, np.float32)):
                raise ValueError("ADO must retain all-one mask")
            matrix, target = self.backend.collect()
            result = self.stridge(matrix, target, self.backend.coefficients(), state["l0_penalty"],
                                  iteration=state["round"], external_root=self.external_root)
            coefficients = np.asarray(result["coefficients"], np.float64)
            penalty = float(result["l0_penalty"])
            if coefficients.shape != (4,) or not np.all(np.isfinite(coefficients)) or not np.isfinite(penalty) or penalty < 0:
                raise ValueError("Invalid STRidge output")
            if state["l0_penalty"] is not None and penalty != state["l0_penalty"]:
                raise ValueError("STRidge changed the inherited l0 baseline")
            self.backend.assign_coefficients(coefficients.astype(np.float32))
            state["l0_penalty"] = penalty
            record = {k: v for k, v in result.items() if k != "coefficients"}
            record.update(iteration=state["round"], coefficients=coefficients.tolist())
            state["stridge_records"].append(record)
            self._phase("ado_nadam")
        elif phase == "final_mask":
            coefficients = self.backend.coefficients()
            self.backend.assign_coefficients(coefficients, (coefficients != 0.0).astype(np.float32))
            state["final_mask"] = True
            self._phase("post_nadam")
        state["serial"] += 1
        self.checkpoint_due = due or phase != state["phase"]
        event.update(next_phase=state["phase"], serial=state["serial"], checkpoint_due=self.checkpoint_due,
                     diagnostic_test_only=self.diagnostic, scientific_acceptance=False)
        return event

    def snapshot(self):
        model = self.backend.state()
        arrays = {"model_%03d" % i: value for i, value in enumerate(model["variables"])}
        arrays["model_mask"] = model["mask"]
        metadata = {"identity": self.identity, "fsm": self.state, "model_iteration": model["iteration"],
                    "model_layout": self.backend.layout, "lbfgsb": None}
        if self.solver is not None:
            arrays.update({"lbfgsb_"+key: value for key, value in self.solver.arrays.items()})
            arrays["lbfgsb_initial"] = self.lbf_initial
            metadata["lbfgsb"] = self.solver.metadata()
        return arrays, metadata

    def restore(self, arrays, metadata):
        if metadata["identity"] != self.identity or metadata["model_layout"] != self.backend.layout:
            raise ValueError("Coupled run identity or model schema differs")
        state = metadata["fsm"]
        if state["phase"] not in PHASES or any(type(state[k]) is not int or state[k] < 0 for k in ("phase_step", "round", "nadam_total", "serial")):
            raise ValueError("Invalid phase state")
        if state["round"] >= self.budget["rounds"] or metadata["model_iteration"] != state["nadam_total"]:
            raise ValueError("Round or NAdam counters disagree")
        model_keys = ["model_%03d" % i for i in range(len(self.backend.layout))]
        expected = set(model_keys + ["model_mask"])
        self.solver, self.lbf_initial = None, None
        if metadata["lbfgsb"] is not None:
            if state["phase"] != "pre_lbfgs":
                raise ValueError("Live L-BFGS cursor outside its phase")
            self.lbf_initial = arrays["lbfgsb_initial"].copy()
            self.solver = self._solver(self.lbf_initial)
            solver_arrays = {key: arrays["lbfgsb_"+key] for key in self.solver.arrays}
            self.solver.restore(solver_arrays, metadata["lbfgsb"])
            expected.update("lbfgsb_"+key for key in self.solver.arrays)
            expected.add("lbfgsb_initial")
        if set(arrays) != expected:
            raise ValueError("Unexpected coupled state arrays")
        self.backend.restore({"variables": [arrays[k] for k in model_keys], "mask": arrays["model_mask"],
                              "iteration": metadata["model_iteration"]})
        self.state = json.loads(json.dumps(state))
        self.checkpoint_due = False


class CoupledJournal:
    """One atomic commit covers TF state, FSM, STRidge l0/mask and LBF cursor."""
    def __init__(self, directory, coordinator, resume=False):
        self.directory, self.coordinator = Path(directory).resolve(), coordinator
        self.latest = None
        if not resume:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.run_id = uuid.uuid4().hex
            _json(self.directory / "RUN.json", {"run_id": self.run_id, "identity": coordinator.identity})
        else:
            header = json.loads((self.directory / "RUN.json").read_text(encoding="utf-8"))
            if header["identity"] != coordinator.identity:
                raise ValueError("Not the same phase schedule/source/data/numeric run")
            self.run_id = header["run_id"]
            self._load()

    def _pointer(self):
        path = self.directory / "LATEST.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(self):
        with _writer(self.directory):
            serial = self.coordinator.state["serial"]
            if self._pointer() != self.latest or (self.latest and serial <= self.latest["serial"]):
                raise ValueError("Stale writer or already committed phase boundary")
            name = "state_%09d_%s" % (serial, uuid.uuid4().hex)
            folder = self.directory / name
            folder.mkdir(exist_ok=False)
            arrays, metadata = self.coordinator.snapshot()
            rng = np.random.get_state()
            arrays["numpy_rng"] = rng[1]
            if any(value.dtype.kind == "O" for value in arrays.values()):
                raise ValueError("Object arrays are forbidden")
            with (folder / "state.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            _json(folder / "state.json", {"run_id": self.run_id, "snapshot": metadata,
                  "python_rng": random.getstate(), "numpy_rng": [rng[0], rng[2], rng[3], rng[4]],
                  "npz_sha256": sha256(folder / "state.npz")})
            latest = {"run_id": self.run_id, "serial": serial, "directory": name, "metadata_sha256": sha256(folder / "state.json")}
            temporary = self.directory / ("LATEST_%s.tmp" % uuid.uuid4().hex)
            _json(temporary, latest)
            os.replace(str(temporary), str(self.directory / "LATEST.json"))
            self.latest = latest
            return folder

    def _load(self):
        latest = self._pointer()
        if not latest or latest.get("run_id") != self.run_id or not re.fullmatch(r"state_[0-9]{9}_[a-f0-9]{32}", latest.get("directory", "")):
            raise ValueError("Invalid same-run phase pointer")
        folder = (self.directory / latest["directory"]).resolve(strict=True)
        if folder.parent != self.directory or sha256(folder / "state.json") != latest["metadata_sha256"]:
            raise ValueError("Phase metadata integrity failed")
        record = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if record["run_id"] != self.run_id or record["snapshot"]["fsm"]["serial"] != latest["serial"] or sha256(folder / "state.npz") != record["npz_sha256"]:
            raise ValueError("Coupled state integrity failed")
        with np.load(folder / "state.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        numpy_rng = arrays.pop("numpy_rng")
        self.coordinator.restore(arrays, record["snapshot"])
        rng = record["numpy_rng"]
        np.random.set_state((rng[0], numpy_rng, rng[1], rng[2], rng[3]))
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        random.setstate(tuples(record["python_rng"]))
        self.latest = latest


def configured_inputs(condition, data_root=None):
    configured = data_root or os.environ.get("GIFT_DATA_ROOT")
    if not configured:
        raise ValueError("Set GIFT_DATA_ROOT to the external canonical data package")
    root = Path(configured).resolve(strict=True)
    if not root.is_dir() or root == ROOT or ROOT in root.parents:
        raise ValueError("Canonical data package must be an external directory")
    if condition not in ("noise_000", "noise_001", "noise_010"):
        raise ValueError("Unknown canonical noise condition")
    return root, {
        "initialization": root / "auxiliary/pinn_initialization/network_initialization_tf115_seed1234.npz",
        "sampling": root / ("auxiliary/pinn_sampling/" + condition + "_seed1234.npz"),
        "data": root / ("standard_ns_n64_full_spectrum.h5" if condition == "noise_000"
                        else "m1_parameter_identification/" + condition + ".h5"),
    }


def validate_output(output, data_root, inputs, resume=False):
    output = Path(output).resolve()
    protected = [ROOT.resolve(), Path(data_root).resolve()] + [Path(path).resolve().parent for path in inputs.values()]
    if any(output == path or path in output.parents for path in protected):
        raise ValueError("Training output must be outside source/data/initialization directories")
    if output.exists() and not resume:
        raise FileExistsError("Fresh training requires a new output directory")
    if resume and not output.is_dir():
        raise ValueError("Resume requires this run's existing CoupledJournal directory")
    return output


def terminal_export(journal):
    """Create-only numeric terminal files; completion is a separate final commit."""
    control = journal.coordinator
    if control.state["phase"] != "complete":
        raise ValueError("Cannot export an unfinished phase schedule")
    arrays, metadata = control.snapshot()
    if metadata["lbfgsb"] is not None:
        raise ValueError("Terminal state cannot contain a live LBF cursor")
    raw, mask = control.backend.coefficients(), control.backend.mask()
    arrays.update(raw_coefficients=raw, effective_coefficients=raw * mask)
    if any(value.dtype.kind == "O" or (value.dtype.kind == "f" and not np.isfinite(value).all()) for value in arrays.values()):
        raise ValueError("Terminal arrays must be safe finite numeric state")
    directory = journal.directory
    with _writer(directory):
        if journal._pointer() != journal.latest or journal.latest["serial"] != control.state["serial"]:
            raise ValueError("Terminal export requires the latest committed same-run boundary")
        numeric = directory / "terminal_state.npz"
        if numeric.exists():
            with np.load(numeric, allow_pickle=False) as archive:
                if set(archive.files) != set(arrays) or any(archive[key].dtype != value.dtype or archive[key].shape != value.shape
                        or archive[key].tobytes() != value.tobytes() for key, value in arrays.items()):
                    raise ValueError("Existing terminal state does not match this completed run")
        else:
            with numeric.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
        result = {"schema": "gift.pinn-kc-terminal.v1", "run_id": journal.run_id, "mode": "known",
                  "training_completed": True, "scientific_acceptance": False,
                  "scientific_status": "not_compared_with_original_reference",
                  "diagnostic_test_only": control.diagnostic, "eligible_for_formal_M1": not control.diagnostic,
                  "requested_budget": control.budget, "original_default_budget": control.budget == DEFAULT_BUDGET,
                  "protocol": {"amendment": "002", "initialization_seed": 1234, "float_graph": "float32",
                     "LBF_interface": "float64", "chunk": control.backend.identity.get("chunk"),
                     "pre": {"learning_rate": 1e-3, "physics_weight": 1.0, "l1_weight": 1e-7},
                     "ado_and_post": {"learning_rate": 1e-4, "physics_weight": 2.0, "l1_weight": 0.0},
                     "all_nadam_updates": "joint_network_and_coefficients_same_optimizer",
                     "mask": "exact_nonzero_after_last_ADO_NAdam", "post_LBF": False},
                  "library": LIBRARY, "raw_coefficients": raw.tolist(), "mask": mask.tolist(),
                  "effective_coefficients": (raw * mask).tolist(), "terminal_snapshot": metadata,
                  "terminal_state_sha256": sha256(numeric), "checkpoint": journal.latest,
                  "tensor_format": "numeric_npz_allow_pickle_false_no_meta_graph"}
        destination = directory / "result.json"
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != result:
                raise ValueError("Existing result differs; no overwrite is permitted")
        else:
            _json(destination, result)
        commit = {"schema": "gift.pinn-kc-completion.v1", "run_id": journal.run_id,
                  "result_sha256": sha256(destination), "terminal_state_sha256": sha256(numeric),
                  "diagnostic_test_only": control.diagnostic, "scientific_acceptance": False}
        completion = directory / "COMPLETED.json"
        if completion.exists():
            if json.loads(completion.read_text(encoding="utf-8")) != commit:
                raise ValueError("Existing completion commit is invalid")
        else:
            temporary = directory / ("COMPLETED_%s.tmp" % uuid.uuid4().hex)
            _json(temporary, commit)
            # Link is create-only and makes the fully fsynced commit visible
            # atomically. The harmless temporary link is deliberately retained.
            os.link(str(temporary), str(completion))
        return commit


def execute_schedule(control, output, resume=False, verify_inputs=None):
    journal = CoupledJournal(output, control, resume=resume)
    already_complete = control.state["phase"] == "complete"
    if not resume:
        journal.save()
    while control.state["phase"] != "complete":
        event = control.advance()
        if control.checkpoint_due:
            journal.save()
            print(json.dumps({"phase": control.state["phase"], "round": control.state["round"],
                "nadam_total": control.state["nadam_total"], "checkpoint_serial": event["serial"],
                "diagnostic_test_only": control.diagnostic, "scientific_acceptance": False}), flush=True)
    if verify_inputs is not None:
        verify_inputs()
    return {"already_complete": already_complete, "completion": terminal_export(journal)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("known", "open"), default="known")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--condition", choices=("noise_000", "noise_001", "noise_010"), default="noise_000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--train-rows", type=int)
    parser.add_argument("--physics-rows", type=int)
    for key in DEFAULT_BUDGET:
        parser.add_argument("--"+key.replace("_", "-"), type=int, default=DEFAULT_BUDGET[key])
    args = parser.parse_args()
    if args.mode != "known":
        parser.error("Full PINN scheduling is KC/known only; open remains blocked")
    budget = budget_for({key: getattr(args, key) for key in DEFAULT_BUDGET}, args.diagnostic)
    if args.chunk <= 0 or any(value is not None and value <= 0 for value in (args.train_rows, args.physics_rows)):
        parser.error("Chunk and optional row limits must be positive")
    if not args.diagnostic and (args.chunk != 32768 or args.train_rows is not None or args.physics_rows is not None):
        parser.error("Original chunk32768 and complete inputs are required; subsets/changed chunk require --diagnostic")
    if not args.execute:
        print(json.dumps({"plan_only": True, "condition": args.condition, "mode": "known", "device": args.device, "budget": budget,
                          "diagnostic_test_only": args.diagnostic, "scientific_acceptance": False,
                          "phases": list(PHASES), "tiny_integration_validated": TINY_INTEGRATION_VALIDATED,
                          "full_budget_validated": False, "chunk": args.chunk,
                          "train_rows": args.train_rows, "physics_rows": args.physics_rows}))
        return
    data_root, inputs = configured_inputs(args.condition)
    output = validate_output(args.output, data_root, inputs, args.resume)
    from adapters.pinn_stridge import fit_stridge
    backend = TensorFlowBackend(inputs["initialization"], inputs["sampling"], inputs["data"],
                                train_rows=args.train_rows, physics_rows=args.physics_rows, chunk=args.chunk, device=args.device)
    try:
        control = Coordinator(backend, fit_stridge, budget=budget, diagnostic=args.diagnostic)
        def verify_inputs():
            for key, path in inputs.items():
                if sha256(path) != backend.identity["inputs"][key]["sha256"]:
                    raise ValueError("Bound input changed during execution: " + key)
            if sha256(__file__) != control.identity["source_sha256"]:
                raise ValueError("Controller source changed during execution")
        print(json.dumps(execute_schedule(control, output, args.resume, verify_inputs)), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
