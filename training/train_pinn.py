"""Independent native PINN-SR training, including open and known equation libraries.

One command trains one noise condition and one library. ADO holds the sparse
coefficients fixed as in the paper; pre/post use the native joint optimizer.
NAdam state, phase counters and regression support are checkpointed together.
External operations remain outside GIFT; L-BFGS resumes at phase boundaries.
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
from adapters import source_file
from gift.publication_metadata import public_metadata
from training.configuration import validate_resume, output_path

DEFAULT_BUDGET = {"pre_nadam": 5000, "rounds": 6, "ado_nadam": 1000,
                  "post_nadam": 20000, "lbf_maxiter": 10000, "lbf_maxfun": 10000}
PHASES = ("pre_nadam", "pre_lbfgs", "ado_stridge", "ado_nadam", "final_mask", "post_nadam", "complete")
ALGORITHM_PROFILE = "paper_ado_v1"


def budget_for(overrides=None, diagnostic=False):
    """Return positive native phase budgets after applying named count overrides.

    Counts are NAdam updates, ADO rounds or L-BFGS limits according to key;
    diagnostic is retained in the API while artifact status is set by callers.
    """
    values = dict(DEFAULT_BUDGET)
    if overrides:
        if set(overrides) - set(values):
            raise ValueError("unknown phase budget")
        values.update(overrides)
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("positive phase counts required")
    return values


def callable_binding(value):
    """Record the callable's qualified name and source digest as descriptive provenance."""
    return {"name": value.__module__ + "." + value.__qualname__, "source_sha256": sha256(inspect.getsourcefile(value))}


class TensorFlowBackend:
    """Only feeds, state IO and calls to the native graph, not a second loss."""
    kind = "native_tensorflow_pinn"
    def __init__(self, sampling, data, mode="known", external_root=None,
                 train_rows=None, physics_rows=None, device="cpu",
                 data_profile="released", data_root=None, condition=None):
        """Read selected PINN populations and construct the native TensorFlow graph.

        coordinates and physics have columns [x,y,t] in physical units; targets
        have [u,v,w] with the same measurement row count. Physics collocation
        may have more rows. mode selects a four- or ninety-column library.
        Optional leading-row subsets are for diagnostics; graph feeds are cached
        after construction and are not serialized as model state.
        """
        from adapters.pinn_runtime import PINNRuntime, read_inputs
        if data_profile == "released":
            selected = read_inputs(sampling, data, train_rows, physics_rows, condition=condition)
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
                         "algorithm_profile": ALGORITHM_PROFILE,
                         "profile": self.runtime.profile, "external_sources": self.runtime.source_records,
                         "source_sha256": sha256(source_file(ROOT, "adapters/pinn_runtime.py"))}
        if data_profile == "regenerated":
            self.identity["input_gate_sha256"] = sha256(ROOT / "training/pinn_data.py")
        self.feeds = {}
        self.phase = "pre_nadam"

    @property
    def layout(self):
        """Describe native variable names, shapes and dtypes for exact state restoration."""
        return [{"name": variable.name, "shape": variable.shape.as_list(), "dtype": variable.dtype.name}
                for variable in self.runtime.variables]

    def state(self):
        """Read native TF variables, NAdam slots, mask and iteration as numeric arrays."""
        return self.runtime.state()

    def restore(self, state):
        """Assign a validated numeric snapshot back into the existing native graph."""
        self.runtime.restore(state)

    def feed(self, lr, physics, l1):
        """Cache full-population feeds by learning rate and physics/L1 loss weights.

        Refresh the sparse coefficient mask on every call so final masking takes
        effect even when a previously constructed feed dictionary is reused.
        """
        key = (lr, physics, l1)
        if key not in self.feeds:
            self.feeds[key] = self.runtime.feed(self.coordinates, self.targets, self.physics,
                                              physics_weight=physics, l1_weight=l1, learning_rate=lr)
        self.feeds[key][self.runtime.model.NonZeroMask_w_tf] = self.runtime.mask
        return self.feeds[key]

    def nadam(self, lr, physics, l1):
        """Dispatch paper-fixed coefficients in ADO and joint native pre/post updates."""
        self._require_repaired_runtime()
        phase = "ado" if self.phase == "ado_nadam" else "pre" if self.phase == "pre_nadam" else "post"
        self.runtime.step(self.feed(lr, physics, l1), phase=phase)

    def _require_repaired_runtime(self):
        """Reject an old external runtime before it can perform incompatible training."""
        if (self.runtime.profile.get("variable_semantics") != "TF1_RefVariable"
                or any(not callable(getattr(self.runtime, name, None)) for name in
                       ("apply_sparse_result", "finalize_sparse_support", "lbfgs_result"))):
            raise RuntimeError("PINN training requires the repaired outside runtime: TF1 reference "
                               "variables, recorded SciPy termination and explicit sparse support. "
                               "Set GIFT_EXTERNAL_ADAPTER_ROOT to that runtime.")

    def coefficients(self):
        """Return the native coefficient vector flattened to the selected library width."""
        return self.runtime.session.run(self.runtime.model.lambda_w).reshape(-1)

    def mask(self):
        """Return a copy of the flattened sparse coefficient mask."""
        return self.runtime.mask.reshape(-1).copy()

    def assign_coefficients(self, value, mask=None):
        """Assign the native coefficient vector and optionally its nonzero mask."""
        self.runtime.set_coefficients(value, mask)

    def apply_sparse_result(self, value):
        """Commit one regression result with a stable variable and independent support."""
        self._require_repaired_runtime()
        self.runtime.apply_sparse_result(value)

    def collect(self):
        """Return the native PDE library [physics_rows,columns] and w_t target.

        The target is the network's coordinate derivative at collocation points,
        not a finite-difference label from an independent truth trajectory.
        """
        return self.runtime.library(self.physics)

    def lbfgs(self, budget):
        """Run one native pre-ADO L-BFGS phase with configured iteration/evaluation limits."""
        self._require_repaired_runtime()
        from training.pinn_lbfgsb import run_lbfgs
        return run_lbfgs(self.runtime, self.feed(1e-3, 1., 1e-7),
                         maxiter=budget["lbf_maxiter"], maxfun=budget["lbf_maxfun"])

    def close(self):
        """Release the TensorFlow session and graph resources owned by this backend."""
        self.runtime.close()


class Coordinator:
    """Native updates with a small, explicit resumable experimental schedule."""
    def __init__(self, backend, stridge, budget=None, diagnostic=False,
                 mode="known", external_root=None, nadam_interval=1000):
        """Initialize a fresh PINN phase cursor with native backend and STRidge callable.

        Known/open modes use 4/90 coefficients. Surrogate backends and selected
        row subsets require diagnostic status. The NAdam interval controls durable
        boundaries, while phase budgets govern actual optimization work.
        """
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
            "protocol": "native_loss_paper_fixed_coefficient_ado_phase_boundary_lbfgs", "mode": mode,
            "algorithm_profile": ALGORITHM_PROFILE,
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
        """Move to a named schedule phase and reset its phase-local update cursor."""
        self.state["phase"], self.state["phase_step"] = name, 0

    def advance(self):
        """Execute one NAdam update or one indivisible non-NAdam phase.

        Pretraining runs NAdam then L-BFGS; ADO alternates STRidge and NAdam,
        inheriting the first l0 penalty. ADO optimizes the field network with
        coefficients fixed; posttraining uses the final regression support.
        Return the completed event and whether its resulting
        state needs a checkpoint; L-BFGS has no resumable internal iteration state.
        """
        state, started = self.state, time.perf_counter()
        phase = state["phase"]
        if phase == "complete":
            raise ValueError("schedule already complete")
        due = False
        if phase in ("pre_nadam", "ado_nadam", "post_nadam"):
            pre = phase == "pre_nadam"
            self.backend.phase = phase
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
            sparse_values = coefficients.astype(np.float32)
            if not np.isfinite(sparse_values).all():
                raise FloatingPointError("Sparse coefficients exceed the native float32 representation")
            if callable(getattr(self.backend, "apply_sparse_result", None)):
                self.backend.apply_sparse_result(sparse_values)
            else:
                self.backend.assign_coefficients(sparse_values)
            state["l0_penalty"] = penalty
            record = {key: value for key, value in result.items() if key != "coefficients"}
            record.update(iteration=state["round"], coefficients=coefficients.tolist())
            state["stridge_records"].append(record)
            state["cost"]["stridge_calls"] += 1
            self._phase("ado_nadam")
        elif phase == "final_mask":
            # The regression support is independent of any raw coefficient
            # updates or momentum. Never rediscover it from an evolved vector.
            coefficients = np.asarray(state["stridge_records"][-1]["coefficients"], np.float32)
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
        """Return numeric native arrays and JSON metadata for one coupled boundary.

        Model arrays use stable ordinal names; layout in metadata maps those
        ordinals to native variables. FSM, sparse-regression history and model
        iteration must travel with the arrays, not as independent checkpoints.
        """
        model = self.backend.state()
        arrays = {"model_%03d" % index: value for index, value in enumerate(model["variables"])}
        arrays["model_mask"] = model["mask"]
        return arrays, {"identity": self.identity, "fsm": self.state, "model_iteration": model["iteration"],
                        "model_layout": self.backend.layout}

    def restore(self, arrays, metadata):
        """Validate and restore a coupled model/FSM snapshot before resuming work.

        Derive expected NAdam updates and STRidge calls from phase, round and
        local cursor, rejecting contradictory serial/history/mask state. Native
        layout and scientific inputs must agree before assigning arrays. A
        completed run keeps its original provenance when only export is repaired.
        """
        validate_resume(self.identity, metadata["identity"])
        if metadata["model_layout"] != self.backend.layout:
            raise ValueError("Native model schema differs")
        state = metadata["fsm"]
        if state["phase"] not in PHASES or any(type(state[key]) is not int or state[key] < 0
                                              for key in ("phase_step", "round", "nadam_total", "serial")):
            raise ValueError("invalid phase state")
        if state["round"] >= self.budget["rounds"] or metadata["model_iteration"] != state["nadam_total"]:
            raise ValueError("round or NAdam counters disagree")
        phase, step, round_ = state["phase"], state["phase_step"], state["round"]
        pre, ado, rounds, post = (self.budget[key] for key in ("pre_nadam", "ado_nadam", "rounds", "post_nadam"))
        if phase in ("pre_nadam", "ado_nadam", "post_nadam"):
            if step >= self.budget[phase]:
                raise ValueError("NAdam phase cursor has reached its budget")
        elif step != 0:
            raise ValueError("Non-NAdam phases require a zero cursor")
        expected_updates = {
            "pre_nadam": step, "pre_lbfgs": pre,
            "ado_stridge": pre + round_ * ado, "ado_nadam": pre + round_ * ado + step,
            "final_mask": pre + rounds * ado, "post_nadam": pre + rounds * ado + step,
            "complete": pre + rounds * ado + post,
        }[phase]
        calls = (0 if phase in ("pre_nadam", "pre_lbfgs") else
                 round_ if phase == "ado_stridge" else
                 round_ + 1 if phase == "ado_nadam" else rounds)
        overhead = (0 if phase in ("pre_nadam", "pre_lbfgs") else
                    1 + calls + int(phase in ("post_nadam", "complete")))
        if (state["nadam_total"] != expected_updates or state["serial"] != expected_updates + overhead
                or state["cost"]["nadam_updates"] != expected_updates
                or state["cost"]["stridge_calls"] != calls
                or len(state["stridge_records"]) != calls
                or any(row["iteration"] != i for i, row in enumerate(state["stridge_records"]))
                or bool(state["final_mask"]) != (phase in ("post_nadam", "complete"))
                or (phase in ("pre_nadam", "pre_lbfgs") and round_ != 0)
                or (phase in ("final_mask", "post_nadam", "complete") and round_ != rounds - 1)):
            raise ValueError("Phase progress and cumulative counters disagree")
        if any(np.asarray(record["coefficients"]).shape != (self.columns,)
               or not np.isfinite(np.asarray(record["coefficients"], np.float32)).all()
               for record in state["stridge_records"]):
            raise ValueError("Checkpoint regression vectors differ from the finite library layout")
        keys = ["model_%03d" % index for index in range(len(self.backend.layout))]
        if set(arrays) != set(keys + ["model_mask"]):
            raise ValueError("unexpected checkpoint arrays")
        # The coupled FSM already stores each regression vector, so recover
        # the runtime's saved regression/support without duplicating NPZ state.
        sparse = (None if not state["stridge_records"] else
                  np.asarray(state["stridge_records"][-1]["coefficients"], np.float32).reshape(self.columns, 1))
        support = None if sparse is None else (sparse != 0).astype(np.float32)
        expected_mask = support if state["final_mask"] else np.ones((self.columns, 1), np.float32)
        if not np.array_equal(arrays["model_mask"], expected_mask):
            raise ValueError("Checkpoint mask differs from its coupled regression support")
        self.backend.restore({"variables": [arrays[key] for key in keys], "mask": arrays["model_mask"],
                              "iteration": metadata["model_iteration"],
                              "sparse_coefficients": sparse, "sparse_support": support})
        self.backend.phase = phase
        self.state = json.loads(json.dumps(state))
        if phase == "complete":
            # A completed export describes the committed run. Keep its original
            # provenance when a later source/environment change only repairs IO.
            self.identity = metadata["identity"]
        self.checkpoint_due = False


class CoupledJournal:
    """One atomic commit covers TF state, FSM, STRidge l0/mask and LBF cursor."""
    def __init__(self, directory, coordinator, resume=False):
        """Create a run header or reopen and restore its latest coupled phase boundary."""
        self.directory, self.coordinator = Path(directory).resolve(), coordinator
        self.latest = None
        if not resume:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.run_id = uuid.uuid4().hex
            _json(self.directory / "RUN.json", {"run_id": self.run_id, "identity": coordinator.identity})
        else:
            header = json.loads((self.directory / "RUN.json").read_text(encoding="utf-8"))
            validate_resume(coordinator.identity, header["identity"])
            self.run_id = header["run_id"]
            self._load()

    def _pointer(self):
        """Read the current small JSON pointer, returning None before the first commit."""
        path = self.directory / "LATEST.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(self):
        """Flush finite numeric arrays, FSM and RNG, then advance the latest pointer.

        The writer lock and serial prevent competing or repeated commits. NPZ
        is written first, JSON next, and LATEST last, so interruption leaves the
        previous complete model/FSM/RNG boundary available.
        """
        with _writer(self.directory):
            serial = self.coordinator.state["serial"]
            if self._pointer() != self.latest or (self.latest and serial <= self.latest["serial"]):
                raise ValueError("Stale writer or already committed phase boundary")
            name = "state_%09d_%s" % (serial, uuid.uuid4().hex)
            folder = self.directory / name
            arrays, metadata = self.coordinator.snapshot()
            rng = np.random.get_state()
            arrays["numpy_rng"] = rng[1]
            if any(value.dtype.kind == "O" for value in arrays.values()):
                raise ValueError("Object arrays are forbidden")
            if any(value.dtype.kind in "fc" and not np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError("Nonfinite PINN state; the previous checkpoint remains current")
            folder.mkdir(exist_ok=False)
            with (folder / "state.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            _json(folder / "state.json", {"run_id": self.run_id, "snapshot": metadata,
                  "python_rng": random.getstate(), "numpy_rng": [rng[0], rng[2], rng[3], rng[4]],
                  "npz_bytes": (folder / "state.npz").stat().st_size})
            latest = {"run_id": self.run_id, "serial": serial, "directory": name}
            temporary = self.directory / ("LATEST_%s.tmp" % uuid.uuid4().hex)
            _json(temporary, latest)
            os.replace(str(temporary), str(self.directory / "LATEST.json"))
            self.latest = latest
            return folder

    def _load(self):
        """Restore arrays/FSM first, then Python and NumPy RNG from the named boundary.

        New JSON records include NPZ size; old records may omit it. Object arrays
        are not deserialized. The pointer and snapshot serial must agree.
        """
        latest = self._pointer()
        if not latest or latest.get("run_id") != self.run_id or not re.fullmatch(r"state_[0-9]{9}_[a-f0-9]{32}", latest.get("directory", "")):
            raise ValueError("Invalid same-run phase pointer")
        folder = (self.directory / latest["directory"]).resolve(strict=True)
        if folder.parent != self.directory:
            raise ValueError("Phase metadata integrity failed")
        record = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if record["run_id"] != self.run_id or record["snapshot"]["fsm"]["serial"] != latest["serial"]:
            raise ValueError("Coupled state integrity failed")
        if "npz_bytes" in record and (folder / "state.npz").stat().st_size != record["npz_bytes"]:
            raise ValueError("Checkpoint file size differs")
        with np.load(folder / "state.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        numpy_rng = arrays.pop("numpy_rng")
        self.coordinator.restore(arrays, record["snapshot"])
        rng = record["numpy_rng"]
        np.random.set_state((rng[0], numpy_rng, rng[1], rng[2], rng[3]))
        def tuples(value):
            """Restore nested Python RNG tuples after JSON has converted them to lists."""
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        random.setstate(tuples(record["python_rng"]))
        self.latest = latest


def configured_inputs(condition, data_root=None):
    """Return the selected data root and sampling/HDF5 paths for one noise condition."""
    configured = data_root or os.environ.get("GIFT_DATA_ROOT")
    if not configured:
        raise ValueError("Set GIFT_DATA_ROOT to the external canonical data package")
    root = Path(configured).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Data root must be a directory")
    if condition not in ("noise_000", "noise_001", "noise_005"):
        raise ValueError("Unknown canonical noise condition")
    return root, {
        "sampling": root / ("auxiliary/pinn_sampling/" + condition + "_seed1234.npz"),
        "data": root / ("standard_ns_n64_full_spectrum.h5" if condition == "noise_000"
                        else "w3_parameter_identification/" + condition + ".h5"),
    }


def validate_output(output, data_root, inputs, resume=False):
    """Require a protected-root-safe new run folder or an existing resume journal."""
    output = output_path(output, ROOT, data_root, *[Path(path).resolve().parent for path in inputs.values()])
    if output.exists() and not resume:
        raise FileExistsError("Fresh training requires a new output directory")
    if resume and not output.is_dir():
        raise ValueError("Resume requires this run's existing CoupledJournal directory")
    return output


def terminal_export(journal):
    """Publish numeric final state, equation readout and a separate completion record.

    Require the complete FSM to match the latest committed journal boundary.
    Existing products must contain the same numeric state; missing products
    are created atomically. Completion reports that optimization finished,
    while scientific_acceptance stays false until equation evaluation is
    performed. Effective coefficients are the raw vector times its final mask.
    """
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
            temporary = numeric.with_name(numeric.name + "." + uuid.uuid4().hex + ".tmp")
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, numeric)
            temporary.unlink()
        result = {"schema": "gift.pinn-terminal.v1", "run_id": journal.run_id, "mode": control.mode,
                  "training_completed": True, "scientific_acceptance": False,
                  "scientific_status": "training_complete; equation_readout_not_yet_evaluated",
                  "diagnostic_test_only": control.diagnostic, "eligible_for_formal_W3": not control.diagnostic,
                  "requested_budget": control.budget, "standard_budget": control.budget == DEFAULT_BUDGET,
                  "protocol": {"initialization": "native_Xavier_seed1234", "float_graph": "float32",
                     "algorithm_profile": ALGORITHM_PROFILE,
                     "variable_semantics": "TF1_RefVariable",
                     "LBF_interface": "upstream_public_ScipyOptimizerInterface", "batching": "complete_population",
                     "pre": {"learning_rate": 1e-3, "physics_weight": 1.0, "l1_weight": 1e-7},
                     "ado_and_post": {"learning_rate": 1e-4, "physics_weight": 2.0, "l1_weight": 0.0},
                     "nadam_updates": {"pre": "joint_network_and_coefficients",
                        "ado": "network_only_fixed_STRidge_coefficients_paper_Eq6b",
                        "post": "joint_network_and_effective_nonzero_coefficients_paper_Eq8",
                        "optimizer": "same_native_NAdam_with_inherited_state"},
                     "mask": "exact_nonzero_of_final_STRidge_float32_coefficients",
                     "post_LBF": False},
                  "library": control.backend.runtime.library_names, "training_cost": control.state["cost"], "raw_coefficients": raw.tolist(), "mask": mask.tolist(),
                  "effective_coefficients": (raw * mask).tolist(), "terminal_snapshot": public_metadata(metadata),
                  "terminal_state_sha256": sha256(numeric), "checkpoint": journal.latest,
                  "tensor_format": "numeric_npz_allow_pickle_false_no_meta_graph"}
        destination = directory / "result.json"
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != result:
                raise ValueError("Existing result differs; no overwrite is permitted")
        else:
            temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
            _json(temporary, result)
            os.link(temporary, destination)
            temporary.unlink()
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
            # Link is create-only and exposes a fully flushed commit atomically.
            os.link(str(temporary), str(completion))
            temporary.unlink()
        return commit



def execute_schedule(control, output, resume=False, verify_inputs=None, stop_after_updates=None):
    """Drive phase events, commit requested boundaries and export completed training.

    A requested pause commits the resulting state before returning. When an
    uncommitted L-BFGS phase is interrupted, resume starts it from the previous
    phase boundary. verify_inputs is an optional caller-supplied final check.
    """
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
    """Parse one PINN condition/library run; construct TensorFlow only for --execute.

    Plan mode prints budgets and phase order without creating outputs.
    Diagnostic row restrictions are explicit, and the native backend is
    closed even if schedule execution raises.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("known", "open"), default="known")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--condition", choices=("noise_000", "noise_001", "noise_005"), default="noise_000")
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
                          "algorithm_profile": ALGORITHM_PROFILE,
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
        print(json.dumps(execute_schedule(control, output, args.resume, None,
                                          args.stop_after_updates)), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
