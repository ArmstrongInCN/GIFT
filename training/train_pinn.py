"""Independent native PINN-SR training, including open and known equation libraries.

One command trains one noise condition and one library. All NAdam variables,
phase counters, sparse masks and STRidge state are checkpointed together.
L-BFGS uses the upstream public interface and resumes only at phase boundaries.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from adapters.pinn_runtime import _json, _writer, sha256

DEFAULT_BUDGET = {"pre_nadam": 5000, "rounds": 6, "ado_nadam": 1000,
                  "post_nadam": 20000, "lbf_maxiter": 10000, "lbf_maxfun": 10000}
PHASES = ("pre_nadam", "pre_lbfgs", "ado_stridge", "ado_nadam", "final_mask", "post_nadam", "complete")


def budget_for(overrides=None, diagnostic=False):
    values = dict(DEFAULT_BUDGET)
    if overrides:
        if set(overrides) - set(values):
            raise ValueError("unknown phase budget")
        values.update(overrides)
    if any(type(value) is not int or value <= 0 for value in values.values()) or values["rounds"] > 6:
        raise ValueError("positive counts and at most six ADO rounds required")
    if values != DEFAULT_BUDGET and not diagnostic:
        raise ValueError("changed budgets require explicit diagnostic/test-only status")
    return values


def callable_binding(value):
    return {"name": value.__module__ + "." + value.__qualname__, "source_sha256": sha256(inspect.getsourcefile(value))}


class TensorFlowBackend:
    """Only feeds, state IO and calls to the native graph, not a second loss."""
    kind = "native_tensorflow_pinn"
    def __init__(self, sampling, data, mode="known", external_root=None,
                 train_rows=None, physics_rows=None, device="cpu",
                 data_profile="released", data_root=None, condition=None):
        from adapters.pinn_runtime import PINNRuntime, read_inputs
        if data_profile == "released":
            selected = read_inputs(sampling, data, train_rows, physics_rows)
        elif data_profile == "regenerated":
            from training.pinn_data import read_regenerated_inputs
            if data_root is None or condition is None:
                raise ValueError("Regenerated PINN inputs need an explicit collection and condition")
            selected = read_regenerated_inputs(sampling, data, data_root, condition, train_rows, physics_rows)
        else:
            raise ValueError("Unknown PINN data profile")
        self.coordinates, self.targets, self.physics, lower, upper, bound = selected
        self.runtime = PINNRuntime(mode, lower, upper, external_root, device)
        self.identity = {"inputs": bound, "mode": mode, "batching": "complete_population",
                         "profile": self.runtime.profile, "external_sources": self.runtime.source_records,
                         "source_sha256": sha256(ROOT / "adapters/pinn_runtime.py")}
        if data_profile == "regenerated":
            self.identity["input_gate_sha256"] = sha256(ROOT / "training/pinn_data.py")
        self.feeds = {}

    @property
    def layout(self):
        return [{"name": variable.name, "shape": variable.shape.as_list(), "dtype": variable.dtype.name}
                for variable in self.runtime.variables]

    def state(self):
        return self.runtime.state()

    def restore(self, state):
        self.runtime.restore(state)

    def feed(self, lr, physics, l1):
        key = (lr, physics, l1)
        if key not in self.feeds:
            self.feeds[key] = self.runtime.feed(self.coordinates, self.targets, self.physics,
                                              physics_weight=physics, l1_weight=l1, learning_rate=lr)
        self.feeds[key][self.runtime.model.NonZeroMask_w_tf] = self.runtime.mask
        return self.feeds[key]

    def nadam(self, lr, physics, l1):
        self.runtime.step(self.feed(lr, physics, l1))

    def coefficients(self):
        return self.runtime.session.run(self.runtime.model.lambda_w).reshape(-1)

    def mask(self):
        return self.runtime.mask.reshape(-1).copy()

    def assign_coefficients(self, value, mask=None):
        self.runtime.set_coefficients(value, mask)

    def collect(self):
        return self.runtime.library(self.physics)

    def lbfgs(self, budget):
        from training.pinn_lbfgsb import run_lbfgs
        return run_lbfgs(self.runtime, self.feed(1e-3, 1., 1e-7),
                         maxiter=budget["lbf_maxiter"], maxfun=budget["lbf_maxfun"])

    def close(self):
        self.runtime.close()


class Coordinator:
    """Native updates with a small, explicit resumable experimental schedule."""
    def __init__(self, backend, stridge, budget=None, diagnostic=False,
                 mode="known", external_root=None, nadam_interval=1000):
        if mode not in ("known", "open") or nadam_interval < 1:
            raise ValueError("invalid mode or checkpoint interval")
        if backend.kind != "native_tensorflow_pinn" and not diagnostic:
            raise ValueError("mock backends require diagnostic status")
        if backend.identity.get("inputs", {}).get("selection", {}).get("diagnostic_subset") and not diagnostic:
            raise ValueError("row subsets require diagnostic status")
        self.budget = budget_for(budget, diagnostic)
        self.backend, self.stridge = backend, stridge
        self.mode, self.columns, self.diagnostic = mode, 4 if mode == "known" else 90, diagnostic
        self.external_root, self.nadam_interval = external_root, nadam_interval
        self.identity = json.loads(json.dumps({
            "protocol": "native_loss_joint_nadam_phase_boundary_lbfgs", "mode": mode,
            "source_sha256": sha256(Path(__file__)), "backend": backend.identity,
            "stridge": callable_binding(stridge), "budget": self.budget,
            "lbfgs_source_sha256": sha256(ROOT / "training/pinn_lbfgsb.py"),
            "diagnostic_test_only": diagnostic, "nadam_checkpoint_interval": nadam_interval}))
        self.state = {"phase": "pre_nadam", "phase_step": 0, "round": 0, "nadam_total": 0,
                      "serial": 0, "l0_penalty": None, "final_mask": False, "stridge_records": [],
                      "lbfgs_summary": None, "cost": {"phase_seconds": {}, "nadam_updates": 0,
                      "stridge_calls": 0, "timing_excludes": "setup, checkpoint IO, paused time and discarded uncommitted work"}}
        self.checkpoint_due = True
        if backend.state()["iteration"] != 0:
            raise ValueError("new coordinator requires native fresh initialization")

    def _phase(self, name):
        self.state["phase"], self.state["phase_step"] = name, 0

    def advance(self):
        state, started = self.state, time.perf_counter()
        phase = state["phase"]
        if phase == "complete":
            raise ValueError("schedule already complete")
        due = False
        if phase in ("pre_nadam", "ado_nadam", "post_nadam"):
            pre = phase == "pre_nadam"
            self.backend.nadam(1e-3 if pre else 1e-4, 1. if pre else 2., 1e-7 if pre else 0.)
            state["phase_step"] += 1
            state["nadam_total"] += 1
            due = state["phase_step"] % self.nadam_interval == 0
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
            # This call is one phase. If interrupted, no partial solver state is
            # committed and --resume restores the saved phase-start TF variables.
            state["lbfgs_summary"] = self.backend.lbfgs(self.budget)
            self._phase("ado_stridge")
        elif phase == "ado_stridge":
            if not np.array_equal(self.backend.mask(), np.ones(self.columns, np.float32)):
                raise ValueError("ADO requires all-one mask")
            matrix, target = self.backend.collect()
            result = self.stridge(matrix, target, self.backend.coefficients(), state["l0_penalty"],
                                  iteration=state["round"], external_root=self.external_root)
            coefficients, penalty = np.asarray(result["coefficients"], np.float64), float(result["l0_penalty"])
            if coefficients.shape != (self.columns,) or not np.isfinite(coefficients).all() or not np.isfinite(penalty) or penalty < 0:
                raise ValueError("invalid sparse regression output")
            if state["l0_penalty"] is not None and penalty != state["l0_penalty"]:
                raise ValueError("STRidge changed the inherited l0 baseline")
            self.backend.assign_coefficients(coefficients.astype(np.float32))
            state["l0_penalty"] = penalty
            record = {key: value for key, value in result.items() if key != "coefficients"}
            record.update(iteration=state["round"], coefficients=coefficients.tolist())
            state["stridge_records"].append(record)
            state["cost"]["stridge_calls"] += 1
            self._phase("ado_nadam")
        elif phase == "final_mask":
            coefficients = self.backend.coefficients()
            self.backend.assign_coefficients(coefficients, (coefficients != 0).astype(np.float32))
            state["final_mask"] = True
            self._phase("post_nadam")
        seconds = state["cost"]["phase_seconds"]
        seconds[phase] = seconds.get(phase, 0.) + time.perf_counter()-started
        state["cost"]["nadam_updates"] = state["nadam_total"]
        state["serial"] += 1
        self.checkpoint_due = due or phase != state["phase"]
        return {"phase": phase, "next_phase": state["phase"], "serial": state["serial"],
                "nadam_total": state["nadam_total"], "checkpoint_due": self.checkpoint_due}

    def snapshot(self):
        model = self.backend.state()
        arrays = {"model_%03d" % index: value for index, value in enumerate(model["variables"])}
        arrays["model_mask"] = model["mask"]
        return arrays, {"identity": self.identity, "fsm": self.state, "model_iteration": model["iteration"],
                        "model_layout": self.backend.layout}

    def restore(self, arrays, metadata):
        if metadata["identity"] != self.identity or metadata["model_layout"] != self.backend.layout:
            raise ValueError("run identity or native model schema differs")
        state = metadata["fsm"]
        if state["phase"] not in PHASES or any(type(state[key]) is not int or state[key] < 0
                                              for key in ("phase_step", "round", "nadam_total", "serial")):
            raise ValueError("invalid phase state")
        if state["round"] >= self.budget["rounds"] or metadata["model_iteration"] != state["nadam_total"]:
            raise ValueError("round or NAdam counters disagree")
        keys = ["model_%03d" % index for index in range(len(self.backend.layout))]
        if set(arrays) != set(keys + ["model_mask"]):
            raise ValueError("unexpected checkpoint arrays")
        self.backend.restore({"variables": [arrays[key] for key in keys], "mask": arrays["model_mask"],
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
        result = {"schema": "gift.pinn-terminal.v1", "run_id": journal.run_id, "mode": control.mode,
                  "training_completed": True, "scientific_acceptance": False,
                  "scientific_status": "training_complete; equation_readout_not_yet_evaluated",
                  "diagnostic_test_only": control.diagnostic, "eligible_for_formal_M1": not control.diagnostic,
                  "requested_budget": control.budget, "standard_budget": control.budget == DEFAULT_BUDGET,
                  "protocol": {"initialization": "native_Xavier_seed1234", "float_graph": "float32",
                     "LBF_interface": "upstream_public_ScipyOptimizerInterface", "batching": "complete_population",
                     "pre": {"learning_rate": 1e-3, "physics_weight": 1.0, "l1_weight": 1e-7},
                     "ado_and_post": {"learning_rate": 1e-4, "physics_weight": 2.0, "l1_weight": 0.0},
                     "all_nadam_updates": "joint_network_and_coefficients_same_optimizer",
                     "mask": "exact_nonzero_after_last_ADO_NAdam", "post_LBF": False},
                  "library": control.backend.runtime.library_names, "training_cost": control.state["cost"], "raw_coefficients": raw.tolist(), "mask": mask.tolist(),
                  "effective_coefficients": (raw * mask).tolist(), "terminal_snapshot": metadata,
                  "terminal_state_sha256": sha256(numeric), "checkpoint": journal.latest,
                  "tensor_format": "numeric_npz_allow_pickle_false_no_meta_graph"}
        destination = directory / "result.json"
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != result:
                raise ValueError("Existing result differs; no overwrite is permitted")
        else:
            _json(destination, result)
        commit = {"schema": "gift.pinn-completion.v1", "run_id": journal.run_id,
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



def execute_schedule(control, output, resume=False, verify_inputs=None, stop_after_updates=None):
    journal = CoupledJournal(output, control, resume=resume)
    if not resume:
        journal.save()
    if stop_after_updates is not None and control.state["nadam_total"] >= stop_after_updates:
        raise ValueError("no remaining updates before requested pause")
    while control.state["phase"] != "complete":
        event = control.advance()
        pause = stop_after_updates is not None and control.state["nadam_total"] >= stop_after_updates
        if control.checkpoint_due or pause:
            journal.save()
            print(json.dumps({"phase": control.state["phase"], "round": control.state["round"],
                "nadam_total": control.state["nadam_total"], "checkpoint_serial": event["serial"],
                "training_cost": control.state["cost"], "diagnostic_test_only": control.diagnostic}), flush=True)
        if pause and control.state["phase"] != "complete":
            return {"status": "paused_at_committed_boundary", "checkpoint": journal.latest}
    if verify_inputs is not None:
        verify_inputs()
    return {"status": "complete", "completion": terminal_export(journal)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("known", "open"), default="known")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--condition", choices=("noise_000", "noise_001", "noise_010"), default="noise_000")
    parser.add_argument("--data-profile", choices=("released", "regenerated"), default="released")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--train-rows", type=int)
    parser.add_argument("--physics-rows", type=int)
    parser.add_argument("--nadam-interval", type=int, default=1000)
    parser.add_argument("--stop-after-updates", type=int)
    for key in DEFAULT_BUDGET:
        parser.add_argument("--"+key.replace("_", "-"), type=int, default=DEFAULT_BUDGET[key])
    args = parser.parse_args()
    budget = budget_for({key: getattr(args, key) for key in DEFAULT_BUDGET}, args.diagnostic)
    if args.nadam_interval < 1 or (args.stop_after_updates is not None and args.stop_after_updates < 1):
        parser.error("positive checkpoint interval and stopping count required")
    if not args.diagnostic and (args.train_rows is not None or args.physics_rows is not None):
        parser.error("row subsets require --diagnostic")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if not args.execute:
        print(json.dumps({"plan_only": True, "mode": args.mode, "condition": args.condition,
                          "data_profile": args.data_profile,
                          "device": args.device, "budget": budget, "phases": PHASES,
                          "batching": "complete_population", "initialization": "native_Xavier_seed1234",
                          "lbfgs_resume": "restart_uncommitted_phase_from_saved_start",
                          "diagnostic_test_only": args.diagnostic, "training_executed": False}))
        return
    data_root, inputs = configured_inputs(args.condition)
    output = validate_output(args.output, data_root, inputs, args.resume)
    from adapters.pinn_stridge import fit_stridge
    backend = TensorFlowBackend(inputs["sampling"], inputs["data"], mode=args.mode,
                                train_rows=args.train_rows, physics_rows=args.physics_rows, device=args.device,
                                data_profile=args.data_profile, data_root=data_root, condition=args.condition)
    try:
        control = Coordinator(backend, fit_stridge, budget=budget, diagnostic=args.diagnostic,
                              mode=args.mode, nadam_interval=args.nadam_interval)
        def verify_inputs():
            for key, path in inputs.items():
                if sha256(path) != backend.identity["inputs"][key]["sha256"]:
                    raise ValueError("bound input changed during execution: " + key)
            if sha256(__file__) != control.identity["source_sha256"]:
                raise ValueError("controller source changed during execution")
            for relative, expected in (
                ("adapters/pinn_runtime.py", backend.identity["source_sha256"]),
                ("training/pinn_lbfgsb.py", control.identity["lbfgs_source_sha256"]),
            ):
                if sha256(ROOT / relative) != expected:
                    raise ValueError("training source changed during execution")
            if callable_binding(fit_stridge) != control.identity["stridge"]:
                raise ValueError("STRidge source changed during execution")
            from adapters.pinn_runtime import sources
            if sources()[0] != backend.identity["external_sources"]:
                raise ValueError("External PINN sources changed during execution")
            if args.data_profile == "regenerated":
                if sha256(ROOT / "training/pinn_data.py") != backend.identity["input_gate_sha256"]:
                    raise ValueError("Generated input gate changed during execution")
                from training.pinn_data import read_regenerated_inputs
                checked = read_regenerated_inputs(inputs["sampling"], inputs["data"], data_root,
                    args.condition, args.train_rows, args.physics_rows)[-1]
                if checked != backend.identity["inputs"]:
                    raise ValueError("Generated input provenance changed during execution")
        print(json.dumps(execute_schedule(control, output, args.resume, verify_inputs,
                                          args.stop_after_updates)), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
