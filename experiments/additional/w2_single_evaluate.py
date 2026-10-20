"""Evaluate fixed N64-trained native predictors on paired N64/N96/N128 grids.

W2 scores the exact first 25 future .04-second nodes of the t5..6 window.
FNO3D retains its trained 75-query joint output and only then selects the prefix.
U-NO/U-Net keep their own learned layers and incoming target-grid fields.
ANIE is intentionally outside W2's user-requested scope.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import h5py
import numpy as np

from experiments.additional import w1_v2_evaluate as shared

METHODS = ("GIFT", "fno2d", "fno3d", "uno", "unet", "GIFT-N50")
GRIDS = (64, 96, 128)
QUERY_TIMES = np.arange(26, dtype=np.float64) * .04
DISPLAY_NAMES = {"GIFT": "GIFT", "GIFT-N50": "GIFT-N50", "fno2d": "FNO2D",
                 "fno3d": "FNO3D", "uno": "U-NO", "unet": "U-Net"}
COLORS = {"GIFT": "#0072B2", "GIFT-N50": "#56B4E9", "fno2d": "#D55E00",
          "fno3d": "#CC79A7", "uno": "#009E73", "unet": "#E69F00"}


@dataclass
class ObservationWindow:
    """Exact complete-population rows/nodes, with prediction inputs separate from truth."""
    path: Path
    group: str
    grid: int
    rows: np.ndarray
    columns: np.ndarray
    trajectory_ids: np.ndarray
    parameters: np.ndarray
    cohort: str

    def initial(self, first, last):
        """Read only the observed t5 field; predictors cannot access future targets."""
        with h5py.File(self.path, "r") as handle:
            fields = handle[self.group + "/vorticity"]
            return np.stack([fields[int(row), int(self.columns[0])] for row in self.rows[first:last]]).astype(np.float32)

    def truth(self, first, last):
        """Read exact t5..6 reference nodes only after autonomous raw output is committed."""
        with h5py.File(self.path, "r") as handle:
            fields = handle[self.group + "/vorticity"]
            return np.stack([fields[int(row), self.columns] for row in self.rows[first:last]]).astype(np.float32)


def observation_window(path, group, grid, cohort):
    """Require every fixed cohort member and all 26 actual nodes; never interpolate."""
    if grid not in GRIDS or cohort not in ("confirmation", "old_test"):
        raise ValueError("W2 requires a prescribed grid/cohort")
    path = Path(path).resolve(strict=True)
    expected = np.arange(100000, 100180) if cohort == "confirmation" else np.arange(1040, 1220)
    with h5py.File(path, "r") as handle:
        node, fields = handle[group], handle[group + "/vorticity"]
        ids = node["trajectory_index"][:]
        if (handle.attrs.get("trajectory_id_scheme") != "canonical" or ids.dtype.kind not in "iu"
                or ids.ndim != 1 or len(ids) != 180 or set(map(int, ids)) != set(map(int, expected))):
            raise ValueError("W2 requires exactly the complete canonical180 population")
        if cohort == "confirmation" and not bool(handle.attrs.get("complete", False)):
            raise ValueError("confirmation references must be complete")
        if fields.shape[0] != 180 or fields.ndim != 4 or fields.shape[2:] != (grid, grid) or fields.dtype != np.dtype("float32"):
            raise ValueError("W2 references must be float32 [180,time,N,N]")
        times = node["absolute_time"][:] if "absolute_time" in node else node["time"][:]
        if times.shape != (fields.shape[1],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("W2 reference time labels must be increasing actual absolute nodes")
        columns = np.asarray([np.argmin(np.abs(times-target)) for target in 5+QUERY_TIMES], dtype=np.int64)
        if len(set(map(int, columns))) != 26 or not np.allclose(times[columns], 5+QUERY_TIMES, rtol=0, atol=1e-6):
            raise ValueError("W2 requires all actual t5..6 .04 nodes; interpolation is forbidden")
        by_id = {int(value): row for row, value in enumerate(ids)}
        rows = np.asarray([by_id[int(value)] for value in expected], np.int64)
        parameter_name = "initial_parameters" if "initial_parameters" in node else "initial_condition_parameters"
        parameters = node[parameter_name][:][rows]
        if parameters.shape != (180, 4, 4) or not np.isfinite(parameters).all():
            raise ValueError("paired initial-condition parameters must be finite [180,4,4]")
    return ObservationWindow(path, group, grid, rows, columns, expected.astype(np.int64), parameters, cohort)


def paired_windows(data_file, n64_data_file, cohort):
    """Bind the same continuous initial parameters and exact IDs on all three grids."""
    windows = {}
    for grid in GRIDS:
        path = n64_data_file if grid == 64 and n64_data_file is not None else data_file
        group = "test" if grid == 64 and n64_data_file is not None else f"N{grid}/test"
        windows[grid] = observation_window(path, group, grid, cohort)
    reference = windows[64]
    for grid in (96, 128):
        if (not np.array_equal(windows[grid].trajectory_ids, reference.trajectory_ids)
                or not np.array_equal(windows[grid].parameters, reference.parameters)):
            raise ValueError("W2 cross-resolution grids must share exact saved parameters and trajectory IDs")
    return windows


def periodic_resize(value, grid):
    """Deterministically change only spatial grid size using periodic Fourier modes.

The even-grid Nyquist coefficient is split/merged by the project's established
solver convention. Unnormalised DFT amplitudes are rescaled automatically.
Leading axes, dtype and device stay unchanged; no learned or test statistic is
created. This operation is used solely to lift frozen FNO3D training statistics.
"""
    import torch
    from even_full_spectrum_ns import split_even_native_dft, merge_odd_dft_to_even_native
    source = value.shape[-1]
    if source == grid:
        return value.clone()
    padded = 3 * (max(source, grid) // 2) + 1
    spectrum = split_even_native_dft(torch.fft.fft2(value, dim=(-2, -1)), padded)
    return torch.fft.ifft2(merge_odd_dft_to_even_native(spectrum, grid), dim=(-2, -1)).real.to(value.dtype)


def lift_fno3d_normalization(normalization, grid):
    """Lift only frozen N64 one-input/75-output statistics to the target grid."""
    import torch
    if normalization.get("kind") != "paper_UnitGaussianNormalizer" or normalization.get("validation_or_test_used", False):
        raise ValueError("W2 FNO3D requires frozen training-only normalizer metadata")
    lifted = dict(normalization)
    for key in ("input_mean", "input_std", "output_mean", "output_std"):
        tensor = normalization[key].detach().cpu().to(torch.float32)
        steps = 1 if key.startswith("input") else 75
        if tuple(tensor.shape) != (64, 64, steps):
            raise ValueError("FNO3D training-statistic axes must be [64,64,1 or75]")
        tensor = periodic_resize(tensor.permute(2, 0, 1).contiguous(), grid).permute(1, 2, 0).contiguous()
        if not bool(torch.isfinite(tensor).all()) or (key.endswith("std") and not bool((tensor > 0).all())):
            raise FloatingPointError("invalid deterministic lifted FNO3D training statistic: " + key)
        lifted[key] = tensor
    return lifted, dict(source_grid=64, target_grid=grid, spatial_method="identity" if grid == 64 else
                       "deterministic periodic Fourier interpolation of frozen training statistics",
                       input_steps=1, output_steps=75, test_statistics_used=False,
                       temporal_interpolation=False, target_grid_fit=False, std_clamping=False)


def load_predictor(label, checkpoint, payload, grid, device):
    """Load fixed learned tensors and apply only their existing grid interface."""
    import torch
    from experiments.model_locations import generator_checkpoint, runtime_model_spec
    if label in ("GIFT", "GIFT-N50"):
        from gift import FixedBandwidthGridGenerator, load_generator
        from training.checkpoints import state_digest
        from experiments.formal._shared.high_frequency import load_high_frequency_model
        spec = payload["model_spec"]
        generator_path = generator_checkpoint(checkpoint, payload)
        generator = load_generator(generator_path, device=device).eval().requires_grad_(False)
        if state_digest(generator.state_dict()) != spec["generator_weights_sha256"]:
            raise ValueError("GIFT branch and generator tensor binding differs")
        low = FixedBandwidthGridGenerator(generator, grid, reference_grid=64)
        branch, _ = load_high_frequency_model(checkpoint, torch.device(device), low)
        return (low, branch), None, dict(source_grid=64, target_grid=grid, constant_spectrum_factor=(grid/64)**2,
            learned_weights_unchanged=True, target_grid_fit=False,
            runtime_generator_checkpoint=str(generator_path),
            runtime_generator_checkpoint_sha256=shared.file_sha256(generator_path),
            generator_weights_sha256=spec["generator_weights_sha256"])
    runtime_spec = runtime_model_spec(payload["model_spec"])
    if label in ("uno", "unet"):
        from adapters.w1_extended import build_network
        model = build_network(runtime_spec, "cpu")
    else:
        from training.w1_single_train import build_network
        model = build_network(runtime_spec, "cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    if label == "fno3d":
        normalization, audit = lift_fno3d_normalization(payload["normalization"], grid)
    else:
        normalization, audit = payload.get("normalization"), dict(source_grid=64, target_grid=grid,
            spatial_resize_of_observations=False, learned_weights_unchanged=True, target_grid_fit=False)
    if label == "unet":
        audit["pixel_kernel_physical_scale_changes"] = True
        audit["native_convolution_input_grid"] = grid
    audit["runtime_source"] = _predictor_source(model, runtime_spec)
    return model, normalization, audit


def _predictor_source(model, runtime_spec):
    """Record the actual inference source files after machine-specific relocation.

    Historical checkpoint specs remain unchanged. These locations and hashes
    describe this new evaluation only and never determine permission to run.
    """
    from adapters import source_file
    method = runtime_spec["method"]
    upstream = model.gift_external_source
    names = {"fno2d": ["fourier_2d_time.py"],
             "fno3d": ["fourier_3d.py", "utilities3.py"],
             "uno": ["navier_stokes_uno2d.py", "integral_operators.py", "Adam.py", "utilities3.py"],
             "unet": ["Baselines/U-net.py"]}[method]
    project = Path(__file__).resolve().parents[2]
    interfaces = ["adapters/models.py", "training/w1_single_train.py"]
    if method in ("uno", "unet"):
        interfaces.append("adapters/w1_extended.py")
    return dict(source_root=runtime_spec["source_root"], upstream=upstream,
        native_files_sha256={name: shared.file_sha256(Path(upstream["directory"]) / name) for name in names},
        interface_files_sha256={name: shared.file_sha256(source_file(project, name)) for name in interfaces},
        source_hashes_are_provenance_only=True)


def forward_predictor(label, model, initial, payload, normalization, device):
    """Predict [B,26,N,N] without passing targets or additional observed history.

Autoregressive methods run exactly25 calls. FNO3D evaluates all trained75
future query coordinates before taking the first25; changing that time axis
would change its learned joint spacetime operator and is not this experiment.
"""
    import torch
    from experiments.model_locations import runtime_model_spec
    spec = runtime_model_spec(payload["model_spec"])
    if label in ("uno", "unet"):
        from adapters.w1_extended import forward_model
        with torch.inference_mode():
            return shared._numpy(forward_model(model, initial, spec, device=device, steps=25))
    from training.w1_single_train import forward_model
    with torch.inference_mode():
        value = forward_model(model, initial, spec, normalization, device=device,
                              training=False, horizon=25 if label == "fno2d" else None)
    return shared._numpy(value[:, :26])


def raw_status(path, window, label, checkpoint_sha):
    """Validate own-run saved raw identity and committed progress before reuse."""
    with h5py.File(path, "r") as raw:
        required = {"prediction", "trajectory_index", "time", "absolute_time", "solver_failed", "failure_reason"}
        if not required.issubset(raw):
            raise ValueError("saved W2 raw output is missing required arrays")
        if (raw.attrs.get("method") != label or raw.attrs.get("checkpoint_sha256") != checkpoint_sha
                or raw.attrs.get("grid") != window.grid or raw.attrs.get("cohort") != window.cohort
                or raw["prediction"].shape != (180, 26, window.grid, window.grid)
                or raw["prediction"].dtype != np.dtype("float32")
                or raw["solver_failed"].shape != (180,) or raw["failure_reason"].shape != (180,)
                or not np.array_equal(raw["trajectory_index"][:], window.trajectory_ids)
                or not np.array_equal(raw["time"][:], QUERY_TIMES)
                or not np.array_equal(raw["absolute_time"][:], 5+QUERY_TIMES)):
            raise ValueError("saved W2 raw weights, grid, cohort or physical nodes differ")
        completed = int(raw.attrs.get("completed_trajectories", 0))
        if not 0 <= completed <= 180:
            raise ValueError("invalid saved W2 progress")
        status = "complete" if raw.attrs.get("status") == "complete" and completed == 180 else "incomplete"
        cost = json.loads(raw.attrs.get("inference_cost_json", "{}"))
    return status, cost


def reuse_n64_source(directories, label, checkpoint_sha, window):
    """Find a completed same-checkpoint W1 trajectory prefix without new inference."""
    for directory in directories:
        directory = Path(directory)
        report_path = directory / "numeric_report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("cohort") != window.cohort:
            continue
        entry = report.get("methods", {}).get(label)
        if entry is None or entry.get("checkpoint_sha256") != checkpoint_sha:
            continue
        path = directory / entry.get("raw_prediction_file", "")
        if not path.is_file() or shared.file_sha256(path) != entry.get("raw_prediction_sha256"):
            raise ValueError("W1 prefix source raw data differ from its completed numerical receipt")
        with h5py.File(path, "r") as raw:
            if (raw.attrs.get("status") != "complete" or raw["prediction"].shape != (180, 76, 64, 64)
                    or not np.array_equal(raw["trajectory_index"][:], window.trajectory_ids)
                    or not np.array_equal(raw["time"][:26], QUERY_TIMES)):
                raise ValueError("W1 prefix source is incomplete or has different nodes/population")
        return path
    return None


def predict_checkpoint(label, checkpoint, payload, window, raw_path, *, device="cpu", batch_size=1,
                       previous_raw=None, reuse_source=None):
    """Commit raw prefix predictions and actual costs, retaining interrupted evidence."""
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_sha = shared.file_sha256(checkpoint)
    grid, failures, reasons = window.grid, np.zeros(180, bool), [""]*180
    completed, prior_cost = 0, {}
    if previous_raw is not None:
        with h5py.File(previous_raw, "r") as prior:
            completed = int(prior.attrs.get("completed_trajectories", 0))
            prior_cost = json.loads(prior.attrs.get("inference_cost_json", "{}"))
            if completed and not prior_cost:
                raise ValueError("committed W2 prediction prefix requires its original cost receipt")
            failures[:completed] = prior["solver_failed"][:completed]
            reasons[:completed] = prior["failure_reason"].asstr()[:completed].tolist()
    if not 0 <= completed <= 180:
        raise ValueError("partial W2 prefix progress must be inside180")
    setup_start = time.perf_counter()
    if completed == 180:
        # The last flushed prefix can survive a crash before the final status
        # marker. Preserve that old file and rebuild its completion receipt
        # from saved outputs/cost without another model load or forward.
        model, normalization, audit = None, None, prior_cost.get("adaptation_audit", {})
    elif reuse_source is None:
        model, normalization, audit = load_predictor(label, checkpoint, payload, grid, device)
    else:
        model, normalization, audit = None, None, dict(source_grid=64, target_grid=64,
            reused_w1_prefix=str(reuse_source), source_raw_sha256=shared.file_sha256(reuse_source),
            no_additional_model_forward=True, later_t8_failure_not_assigned_to_t6_prefix=True)
    setup_seconds, inference_seconds = time.perf_counter()-setup_start, 0.
    if torch.device(device).type == "cuda" and reuse_source is None and completed < 180:
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with h5py.File(raw_path, "x") as raw:
        raw.attrs.update(status="incomplete", method=label, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                         grid=grid, cohort=window.cohort, inference_input="omega(t5) only", time_points=26,
                         future_steps=25, prediction_dt=.04, adaptation_audit_json=json.dumps(audit, allow_nan=False))
        raw.create_dataset("trajectory_index", data=window.trajectory_ids)
        raw.create_dataset("time", data=QUERY_TIMES)
        raw.create_dataset("absolute_time", data=5+QUERY_TIMES)
        predictions = raw.create_dataset("prediction", shape=(180, 26, grid, grid), dtype="f4",
                                         chunks=(1, 1, grid, grid), compression="lzf", fillvalue=np.nan)
        stored_failures = raw.create_dataset("solver_failed", data=failures)
        stored_reasons = raw.create_dataset("failure_reason", data=np.asarray(reasons, dtype=h5py.string_dtype("utf-8")))
        if completed:
            with h5py.File(previous_raw, "r") as prior:
                for first in range(0, completed, batch_size):
                    predictions[first:min(first+batch_size, completed)] = prior["prediction"][first:min(first+batch_size, completed)]

        def commit(end):
            """Persist outputs and cumulative segment cost before marking progress."""
            peak = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" and reuse_source is None and completed < 180 else None
            cost = dict(inference_seconds=prior_cost.get("inference_seconds", 0.)+inference_seconds,
                        prediction_and_io_seconds=prior_cost.get("prediction_and_io_seconds", 0.)+time.perf_counter()-started,
                        model_setup_seconds=prior_cost.get("model_setup_seconds", 0.)+setup_seconds,
                        batch_size=batch_size, device=str(device), peak_cuda_allocated_bytes=max(peak or 0, prior_cost.get("peak_cuda_allocated_bytes") or 0),
                        forward_time_points=76 if label == "fno3d" else 26, scored_time_points=26,
                        grid=grid, seed=int(payload["seed"]), adaptation_audit=audit,
                        cost_scope="actual W2 inference and I/O; reused W1 prediction costs remain accounted in their original run",
                        inference_segments=prior_cost.get("inference_segments", [])+[
                            dict(first_trajectory=completed, committed_until=end, inference_seconds=inference_seconds)])
            stored_failures[:], stored_reasons[:] = failures, reasons
            raw.attrs["inference_cost_json"] = json.dumps(shared._json_ready(cost), allow_nan=False)
            raw.flush()
            raw.attrs["completed_trajectories"] = end
            raw.flush()
            return cost

        cost = commit(completed)
        for first in range(completed, 180, batch_size):
            last = min(first+batch_size, 180)
            if reuse_source is not None:
                with h5py.File(reuse_source, "r") as source:
                    value = source["prediction"][first:last, :26]
                # A failed future75 after t6 is outside W2. Finite prefix output
                # is scored on its actual short window; no successful row drops.
                bad = ~np.isfinite(value[:, 1:]).all(axis=(1, 2, 3))
                failures[first:last] = bad
                reasons[first:last] = ["nonfinite saved t5..6 prediction prefix" if item else "" for item in bad]
            else:
                initial = window.initial(first, last)
                if not np.isfinite(initial).all():
                    raise ValueError("nonfinite initial field cannot be scored")
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                before = time.perf_counter()
                try:
                    if label in ("GIFT", "GIFT-N50"):
                        from experiments.formal._shared.gift_runtime import rollout_gift
                        result = rollout_gift(low=model[0], branch=model[1], initial_state=initial,
                            trajectory_ids=window.trajectory_ids[first:last], duration=1., save_interval=.04, step=.02,
                            device=torch.device(device), batch_size=batch_size, correction=True, execution="eager")
                        value = shared._numpy(result.prediction)
                        failures[first:last] = np.asarray(result.failure_step) >= 0
                        reasons[first:last] = ["" if item is None else item for item in result.failure_reason]
                    else:
                        value = forward_predictor(label, model, torch.from_numpy(initial).to(device), payload, normalization, device)
                    if value.shape != (last-first, 26, grid, grid):
                        raise ValueError("predictor returned incompatible W2 grid/time axes")
                except (ArithmeticError, FloatingPointError) as error:
                    value = np.full((last-first, 26, grid, grid), np.nan, np.float32)
                    failures[first:last] = True
                    reasons[first:last] = [type(error).__name__ + ": " + str(error)]*(last-first)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter()-before
            predictions[first:last] = value
            cost = commit(last)
            print(json.dumps(dict(status="w2_predictions_committed", grid=grid, method=label,
                                  completed=last, population=180)), flush=True)
        raw.attrs["status"] = "complete"
        raw.flush()
    return cost


def measure_predictions(prediction, truth, ids, *, solver_failed=None):
    """Return complete-population short-window errors; explicit failures retain180."""
    prediction, truth, ids = np.asarray(prediction), np.asarray(truth), np.asarray(ids)
    if (prediction.shape != truth.shape or truth.ndim != 4 or truth.shape[1] != 26
            or truth.shape[-1] not in GRIDS or truth.shape[-2] != truth.shape[-1] or len(ids) != len(truth)):
        raise ValueError("W2 predictions/truth require identical [B,26,N,N] axes")
    if not np.isfinite(truth).all():
        raise ValueError("nonfinite reference cannot be used for scoring")
    explicit = np.zeros(len(ids), bool) if solver_failed is None else np.asarray(solver_failed, bool)
    if explicit.shape != (len(ids),):
        raise ValueError("one solver-failure flag required per trajectory")
    finite = np.isfinite(prediction).all(axis=(-2, -1))
    failed = ~finite[:, 1:].all(axis=1) | explicit
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        target = truth.astype(np.float64)
        squared = (prediction.astype(np.float64)-target)**2
        numerator, denominator = squared.sum(axis=(-2, -1)), (target**2).sum(axis=(-2, -1))
        if np.any(denominator <= 0):
            raise ValueError("relative L2 needs nonzero reference energy at every frame")
        relative = np.sqrt(numerator/denominator)
        metrics = dict(future_mean_relative_l2=relative[:, 1:].mean(axis=1), t6_relative_l2=relative[:, -1].copy(),
                       future_mse=squared[:, 1:].mean(axis=(1, 2, 3)),
                       future_path_relative_l2=np.sqrt(numerator[:, 1:].sum(axis=1)/denominator[:, 1:].sum(axis=1)),
                       t0_relative_l2=relative[:, 0].copy())
    for key in metrics:
        if key != "t0_relative_l2":
            metrics[key][failed] = np.inf
    from experiments.formal._shared.common import metric_arrays
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        bands = metric_arrays(prediction, truth)
    for key, values in bands.items():
        if key == "full_relative_l2":
            continue
        for suffix, vector in (("future_mean", values[:, 1:].mean(axis=1)), ("t6", values[:, -1].copy())):
            vector[failed] = np.inf
            metrics[key + "_" + suffix] = vector
    return dict(per_trajectory=dict(trajectory_id=ids.astype(np.int64), **metrics), relative_l2_by_time=relative,
                finite_by_time=finite, failed=failed, solver_failed=explicit)


def score_raw(raw_path, window, metric_path, truth_path, *, batch_size=4):
    """Save numerical arrays in batches after raw prediction completion, then summarize."""
    per_trajectory, relative, finite, failed, explicit = {}, [], [], [], []
    with h5py.File(raw_path, "r") as raw:
        if raw.attrs.get("status") != "complete":
            raise ValueError("raw autonomous predictions must be complete before scoring")
        if truth_path.exists():
            with h5py.File(truth_path, "r") as stored:
                truth_complete = stored.attrs.get("status") == "complete"
                if truth_complete and (not np.array_equal(stored["trajectory_index"][:], window.trajectory_ids)
                        or not np.array_equal(stored["time"][:], QUERY_TIMES)
                        or stored["vorticity"].shape != (180, 26, window.grid, window.grid)):
                    raise ValueError("saved W2 reference IDs/time nodes differ")
            if not truth_complete:
                shared.preserve_interrupted(truth_path)
        create_truth = not truth_path.exists()
        truth_handle = h5py.File(truth_path, "x") if create_truth else None
        try:
            if create_truth:
                truth_handle.attrs["status"] = "incomplete"
                truth_handle.create_dataset("trajectory_index", data=window.trajectory_ids)
                truth_handle.create_dataset("time", data=QUERY_TIMES)
                truth_handle.create_dataset("absolute_time", data=5+QUERY_TIMES)
                truth_handle.create_dataset("vorticity", shape=(180, 26, window.grid, window.grid), dtype="f4",
                                            chunks=(1, 1, window.grid, window.grid), compression="lzf")
            for first in range(0, 180, batch_size):
                last = min(first+batch_size, 180)
                truth = window.truth(first, last)
                if create_truth:
                    truth_handle["vorticity"][first:last] = truth
                measured = measure_predictions(raw["prediction"][first:last], truth, window.trajectory_ids[first:last],
                                               solver_failed=raw["solver_failed"][first:last])
                for key, values in measured["per_trajectory"].items():
                    per_trajectory.setdefault(key, []).append(values)
                relative.append(measured["relative_l2_by_time"])
                finite.append(measured["finite_by_time"])
                failed.append(measured["failed"])
                explicit.append(measured["solver_failed"])
            if create_truth:
                truth_handle.attrs["status"] = "complete"
                truth_handle.flush()
        finally:
            if truth_handle is not None:
                truth_handle.close()
    per_trajectory = {key: np.concatenate(value) for key, value in per_trajectory.items()}
    relative, finite, failed, explicit = map(np.concatenate, (relative, finite, failed, explicit))
    summary = _metric_summary(per_trajectory, relative, failed, window.trajectory_ids)
    with h5py.File(metric_path, "x") as stored:
        stored.attrs["status"] = "incomplete"
        for key, value in per_trajectory.items():
            stored.create_dataset(key, data=value)
        for key, value in (("relative_l2_by_time", relative), ("finite_by_time", finite),
                           ("failed", failed), ("solver_failed", explicit)):
            stored.create_dataset(key, data=value)
        stored.attrs["status"] = "complete"
    return summary


def _metric_summary(per_trajectory, relative, failed, ids):
    """Summarize the saved whole-population arrays without dropping failed members."""
    summary = dict(status="complete" if not failed.any() else "failed_trajectories", population=180,
                   failed_count=int(failed.sum()), failure_rate=float(failed.mean()), failed_ids=ids[failed].tolist(),
                   primary_metric_name="future25_mean_per_frame_relative_l2",
                   metrics={key: shared._summary(value) for key, value in per_trajectory.items() if key != "trajectory_id"},
                   mean_relative_l2_by_time=[shared._summary(relative[:, i])["mean"] for i in range(26)],
                   time_points=26, future_steps=25, query_times=QUERY_TIMES.tolist(), absolute_times=(5+QUERY_TIMES).tolist(),
                   t0_in_primary_score=False, dt=.04)
    summary["primary_metric"] = summary["metrics"]["future_mean_relative_l2"]["mean"]
    return summary


def _failed_training_record(label, grid, failure):
    """Keep a failed fixed training slot in every target-grid population."""
    return dict(method=label, grid=grid, status="training_failed", seed=failure.get("seed"),
                training_failure=failure, summary=dict(population=180, failed_count=180,
                failure_rate=1., primary_metric=None, metrics={}, mean_relative_l2_by_time=None))


def _same_metric_summary(actual, expected):
    """Allow float64 reduction roundoff while requiring identical structure and counts."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_same_metric_summary(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_metric_summary(left, right) for left, right in zip(actual, expected)))
    if isinstance(expected, float):
        return (isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and bool(np.isclose(actual, expected, rtol=1e-12, atol=1e-14)))
    return actual == expected


def _metric_record(record_path, metric_path, raw_path, window, label, binding):
    """Reuse only complete metric arrays tied to this checkpoint, raw output and cohort.

    New records bind metric bytes with SHA-256. Older receipts have no metric
    digest; validate those arrays against their bound raw/reference inputs in
    bounded batches, without rewriting any historical measurement or receipt.
    """
    if not record_path.is_file() or not metric_path.is_file():
        return None
    try:
        candidate = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("saved W2 metric receipt is corrupt") from error
    if (not isinstance(candidate, dict)
            or not isinstance(candidate.get("raw_prediction_file"), str)
            or not isinstance(candidate.get("metric_file"), str)
            or candidate.get("method") != label
            or candidate.get("grid") != window.grid or candidate.get("seed") != binding["seed"]
            or candidate.get("selected_epoch") != binding["selected_epoch"]
            or candidate.get("checkpoint_sha256") != binding["sha256"]
            or Path(candidate.get("raw_prediction_file", "")) != Path("raw") / raw_path.name
            or Path(candidate.get("metric_file", "")) != Path("metrics") / metric_path.name
            or candidate.get("raw_prediction_sha256") != shared.file_sha256(raw_path)):
        raise ValueError("saved W2 metrics bind different weights, raw outputs or grid")
    if "metric_sha256" in candidate and candidate["metric_sha256"] != shared.file_sha256(metric_path):
        raise ValueError("saved W2 metric bytes differ from the recorded measurement")
    with h5py.File(metric_path, "r") as stored:
        if stored.attrs.get("status") != "complete":
            return None
        required = {"trajectory_id", "future_mean_relative_l2", "t6_relative_l2", "future_mse",
                    "future_path_relative_l2", "t0_relative_l2", "relative_l2_by_time",
                    "finite_by_time", "failed", "solver_failed"}
        if not required.issubset(stored):
            raise ValueError("completed W2 metric arrays are missing required measurements")
        arrays = {name: stored[name][:] for name in stored}
    matrix_names = {"relative_l2_by_time", "finite_by_time"}
    if (any(value.shape != ((180, 26) if name in matrix_names else (180,)) for name, value in arrays.items())
            or not np.array_equal(arrays["trajectory_id"], window.trajectory_ids)):
        raise ValueError("completed W2 metric population or time-grid axes differ")
    if any(arrays[name].dtype.kind != "b" for name in ("finite_by_time", "failed", "solver_failed")):
        raise ValueError("saved W2 failure masks must preserve all trajectory outcomes")
    per_trajectory = {name: value for name, value in arrays.items()
                      if name not in matrix_names | {"failed", "solver_failed"}}
    expected = _metric_summary(per_trajectory, arrays["relative_l2_by_time"], arrays["failed"], window.trajectory_ids)
    if (not _same_metric_summary(candidate.get("summary"), shared._json_ready(expected))
            or candidate.get("status") != expected["status"]):
        raise ValueError("saved W2 metric summary differs from the complete-population arrays")
    if "metric_sha256" not in candidate:
        with h5py.File(raw_path, "r") as raw:
            for first in range(0, 180, 4):
                last = min(first+4, 180)
                measured = measure_predictions(raw["prediction"][first:last], window.truth(first, last),
                    window.trajectory_ids[first:last], solver_failed=raw["solver_failed"][first:last])
                values = dict(measured["per_trajectory"], **{name: measured[name]
                    for name in ("relative_l2_by_time", "finite_by_time", "failed", "solver_failed")})
                if set(values) != set(arrays):
                    raise ValueError("legacy W2 metrics do not contain the declared complete metric set")
                for name, value in values.items():
                    # Re-scoring uses float64 reductions; tolerate only backend
                    # roundoff, with identical NaN/Inf and failure-mask locations.
                    if not np.allclose(arrays[name][first:last], value, rtol=1e-12, atol=1e-14, equal_nan=True):
                        raise ValueError("legacy W2 metric arrays differ from their bound raw/reference inputs: " + name)
    return candidate


def _numeric_artifacts_complete(output, report, plan, windows):
    """A terminal JSON needs every declared slot, raw array and metric measurement."""
    if not isinstance(report, dict) or report.get("status") != "numerical_complete":
        return False
    expected = dict(schema="gift.w2-single-native.evaluation.v1", profile=plan["profile"], cohort=plan["cohort"],
                    grids=plan["grids"], trajectory_ids=plan["trajectory_ids"], population_per_grid=180,
                    query_times=plan["query_times"], absolute_times=plan["absolute_times"],
                    future_steps=25, time_points=26, dt=.04, ANIE_included=False, test_statistics_used=False)
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("completed W2 report differs from its evaluation plan/population")
    if not isinstance(report.get("methods"), dict) or set(report["methods"]) != {str(grid) for grid in GRIDS}:
        raise ValueError("completed W2 report omits a target-grid population")
    complete = True
    for grid in GRIDS:
        if (not isinstance(report["methods"][str(grid)], dict)
                or set(report["methods"][str(grid)]) != set(METHODS)):
            raise ValueError("completed W2 report omits a fixed method slot")
        for label in METHODS:
            record = report["methods"][str(grid)][label]
            if not isinstance(record, dict):
                raise ValueError("completed W2 report contains an invalid method record")
            stem = f"N{grid}_{label}"
            record_path = output / "metrics" / (stem+".json")
            if label in plan["failed_training_slots"]:
                expected_failure = _failed_training_record(label, grid, plan["failed_training_slots"][label])
                if record != shared._json_ready(expected_failure):
                    raise ValueError("completed W2 report changes a failed training slot")
                if not record_path.is_file():
                    complete = False
                elif json.loads(record_path.read_text(encoding="utf-8")) != record:
                    raise ValueError("saved failed W2 training outcome differs from its report")
                continue
            raw_path, metric_path = output / "raw" / (stem+".h5"), output / "metrics" / (stem+".h5")
            binding = plan["checkpoint_bindings"][label]
            if (record.get("checkpoint_sha256") != binding["sha256"]
                    or record.get("summary", {}).get("population") != 180):
                raise ValueError("completed W2 report changes checkpoint or metric population")
            if not raw_path.is_file():
                complete = False
                continue
            status, _ = raw_status(raw_path, windows[grid], label, binding["sha256"])
            if shared.file_sha256(raw_path) != record.get("raw_prediction_sha256"):
                raise ValueError("completed W2 report's raw prediction bytes changed")
            if status != "complete":
                complete = False
                continue
            cached = _metric_record(record_path, metric_path, raw_path, windows[grid], label, binding)
            if cached is None:
                complete = False
            elif cached != record:
                raise ValueError("completed W2 report differs from its per-method measurement")
    return complete


def evaluate_checkpoints(checkpoints, data_file, cohort, output, *, base_lock_file, extension_lock_file,
                         n64_data_file=None, failed_runs=None, device="cpu", batch_size=1, resume=False,
                         plots=True, reuse_evaluations=(), job_events=None):
    """Evaluate six explicitly labelled slots and preserve full paired180 on every grid."""
    from experiments.additional import w1_extended_evaluate as extended
    if batch_size < 1:
        raise ValueError("positive inference batch required")
    from experiments.additional.w1_single_evaluate import protocol_lock, job_cost_summary
    base_lock = protocol_lock(base_lock_file)["lock_snapshot"]
    extension_lock = json.loads(Path(extension_lock_file).read_text(encoding="utf-8"))
    if (extension_lock.get("schema") != "gift.w1w2-native.extension-lock.v1"
            or extension_lock.get("profile") != "single_native76_n300"
            or extension_lock.get("configurations_locked") is not True):
        raise ValueError("W2 requires the fixed native extension configurations lock")
    checkpoints = {label: Path(path).resolve(strict=True) for label, path in dict(checkpoints).items()}
    failures = {}
    for label, path in dict(failed_runs or {}).items():
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_seed = 20260820 if label in ("GIFT", "GIFT-N50") else 0
        if not record.get("reason") or record.get("failure_kind") == "infrastructure" or record.get("seed") != expected_seed:
            raise ValueError("failed W2 training slots need their fixed seed and explicit scientific reason")
        failures[label] = record
    if set(checkpoints) & set(failures) or set(checkpoints) | set(failures) != set(METHODS):
        raise ValueError("W2 requires exactly the six fixed method slots, including declared failures; ANIE excluded")
    payloads = {label: extended.checkpoint_metadata(label, path, base_lock, extension_lock) for label, path in checkpoints.items()}
    windows = paired_windows(data_file, n64_data_file, cohort)
    bindings = {label: dict(checkpoint=str(path), sha256=shared.file_sha256(path), seed=int(payloads[label]["seed"]),
                           selected_epoch=payloads[label]["epoch"], config=payloads[label]["config"],
                           model_spec=payloads[label]["model_spec"]) for label, path in checkpoints.items()}
    plan = dict(schema="gift.w2-single-native.evaluation-plan.v1", profile="single_native76_n300", cohort=cohort,
                interpretation="paired extension of previously evaluated confirmation180; not a fresh blind cohort" if cohort == "confirmation"
                else "posthoc exploratory original test180", grids=list(GRIDS), methods=list(METHODS),
                training_grid=64, target_grid_retraining=False, target_grid_fitting=False, test_statistics_used=False,
                ANIE_included=False, query_times=QUERY_TIMES.tolist(), absolute_times=(5+QUERY_TIMES).tolist(),
                future_steps=25, time_points=26, dt=.04, trajectory_ids=windows[64].trajectory_ids.tolist(),
                source_files={str(n): dict(path=str(w.path), group=w.group, sha256=shared.file_sha256(w.path)) for n, w in windows.items()},
                checkpoint_bindings=bindings, failed_training_slots=failures,
                base_lock_sha256=shared.file_sha256(base_lock_file), extension_lock_sha256=shared.file_sha256(extension_lock_file),
                reuse_evaluations=[str(Path(p).resolve()) for p in reuse_evaluations],
                FNO3D_forward_grid="original75 future .04 queries then select first25; no temporal-token resize",
                GIFT_internal_dt=.02, GIFT_internal_steps=50, keyframe_absolute_times=[5.24, 5.48, 5.76, 6.],
                GIFT_N50_scope="same epoch recipe on fixed50 training trajectories; separate sample-budget ablation")
    output = Path(output).resolve()
    if resume:
        if json.loads((output / "evaluation_plan.json").read_text(encoding="utf-8")) != shared._json_ready(plan):
            raise ValueError("W2 resume must preserve fixed weights, population, references and native grid protocol")
        if (output / "numeric_report.json").exists():
            try:
                completed_report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("saved W2 numerical report is corrupt") from error
            else:
                if _numeric_artifacts_complete(output, completed_report, plan, windows):
                    render_report(output)
                    if plots:
                        draw_figures(output)
                    return completed_report
                shared.preserve_interrupted(output / "numeric_report.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "raw").mkdir()
        (output / "metrics").mkdir()
        shared.write_json(output / "evaluation_plan.json", plan)
    records = {}
    for grid in GRIDS:
        window = windows[grid]
        records[str(grid)] = {}
        for label in METHODS:
            stem = f"N{grid}_{label}"
            raw_path, metric_path, record_path = output / "raw" / (stem+".h5"), output / "metrics" / (stem+".h5"), output / "metrics" / (stem+".json")
            if label in failures:
                record = _failed_training_record(label, grid, failures[label])
                if not record_path.exists():
                    shared.write_json(record_path, record)
                elif json.loads(record_path.read_text(encoding="utf-8")) != shared._json_ready(record):
                    raise ValueError("saved failed W2 training slot differs from this scientific outcome")
                records[str(grid)][label] = record
                continue
            checkpoint, payload, binding = checkpoints[label], payloads[label], bindings[label]
            previous = None
            if raw_path.exists():
                status, cost = raw_status(raw_path, window, label, binding["sha256"])
                if status != "complete":
                    previous = shared.preserve_interrupted(raw_path)
            if not raw_path.exists():
                # A copied run may retain old metric receipts but lack their
                # raw inputs. Preserve those files before recovering the same
                # slot; a regenerated raw file needs its own measured receipt.
                if previous is None:
                    for path in (metric_path, record_path):
                        if path.exists():
                            shared.preserve_interrupted(path)
                source = reuse_n64_source(reuse_evaluations, label, binding["sha256"], window) if grid == 64 else None
                cost = predict_checkpoint(label, checkpoint, payload, window, raw_path, device=device,
                                          batch_size=batch_size, previous_raw=previous, reuse_source=source)
            saved = _metric_record(record_path, metric_path, raw_path, window, label, binding)
            if saved is not None:
                records[str(grid)][label] = saved
                continue
            for path in (metric_path, record_path):
                if path.exists():
                    shared.preserve_interrupted(path)
            summary = score_raw(raw_path, window, metric_path, output / "raw" / f"N{grid}_truth.h5")
            record = dict(method=label, display_name=DISPLAY_NAMES[label], grid=grid, seed=int(payload["seed"]),
                          status=summary["status"], summary=summary, selected_epoch=payload["epoch"],
                          checkpoint=str(checkpoint), checkpoint_sha256=binding["sha256"],
                          raw_prediction_file=str(raw_path.relative_to(output)), raw_prediction_sha256=shared.file_sha256(raw_path),
                          metric_file=str(metric_path.relative_to(output)), metric_sha256=shared.file_sha256(metric_path), cost=cost,
                          comparison_scope="sample-budget ablation" if label == "GIFT-N50" else "common N300 native prediction")
            shared.write_json(record_path, record)
            records[str(grid)][label] = record
    report = dict(schema="gift.w2-single-native.evaluation.v1", status="numerical_complete", profile="single_native76_n300",
                  cohort=cohort, interpretation=plan["interpretation"], population_per_grid=180, trajectory_ids=plan["trajectory_ids"],
                  grids=list(GRIDS), methods=records, dt=.04, query_times=QUERY_TIMES.tolist(),
                  absolute_times=(5+QUERY_TIMES).tolist(), future_steps=25, time_points=26, ANIE_included=False,
                  keyframe_absolute_times=plan["keyframe_absolute_times"], native_zero_shot=True,
                  test_statistics_used=False, training_cost_not_counted_again_in_W2=True,
                  actual_process_cost=job_cost_summary(job_events) if job_events else None,
                  limitations=["Single fixed training instance per method; no training-seed SD or robustness claim.",
                               "Confirmation180 was already evaluated for four models before extending W1/W2; not a new blind cohort.",
                               "U-Net has native variable-grid convolution/deconvolution but fixed pixel kernel physical scale changes with resolution.",
                               "FNO3D training statistics are lifted deterministically in space only, without target-grid/test-statistic fitting.",
                               "GIFT-N50 is a separate sample-budget ablation: same epochs, fewer updates and50 rather than300 samples.",
                               "All180 trajectories remain in every metric; failed full-population means are null, not finite-subset replacements."])
    shared.write_json(output / "numeric_report.json", report)
    render_report(output)
    if plots:
        draw_figures(output)
    return report


def render_report(output):
    """Render readable tables exclusively from committed numerical measurements."""
    output = Path(output)
    report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
    report_hash, target, marker = shared.file_sha256(output / "numeric_report.json"), output / "RESULTS.md", output / "REPORT_COMPLETE.json"
    if marker.exists() and target.exists():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = {}
        if saved.get("numeric_report_sha256") == report_hash and saved.get("results_sha256") == shared.file_sha256(target):
            return target
    for path in (target, marker):
        if path.exists():
            shared.preserve_interrupted(path)
    lines = ["# W2：N64训练模型的原生跨分辨率预测", "", f"人口：{report['cohort']}，每个网格180条；{report['interpretation']}。", "",
             "单初态t5输入，观测间隔0.04秒，报告t5.04–6的25帧；主指标为每条轨迹逐帧相对L2的未来均值，再对完整180条取均值。ANIE按用户要求不加入W2。",
             "", "| 网格 | 方法 | future25相对L2 | t6相对L2 | future MSE | 失败/180 | 新增推理秒 |", "|---|---|---:|---:|---:|---:|---:|"]
    def value(entry, key):
        number = entry.get("summary", {}).get("metrics", {}).get(key, {}).get("mean")
        return "—" if number is None else f"{number:.6g}"
    for grid in GRIDS:
        for label in METHODS:
            entry = report["methods"][str(grid)][label]
            seconds = entry.get("cost", {}).get("inference_seconds")
            lines.append(f"| N{grid} | {DISPLAY_NAMES[label]} | {value(entry,'future_mean_relative_l2')} | {value(entry,'t6_relative_l2')} | "
                         f"{value(entry,'future_mse')} | {entry['summary']['failed_count']} | {'—' if seconds is None else f'{seconds:.1f}'} |")
    lines.extend(["", "GIFT-N50单列为样本预算消融；它不满足主比较的共同N300训练人口，不能替代完整GIFT。已保存W1的N64预测前缀若复用，新增模型推理耗时为0，原成本仍留在W1记录。", "",
                  "FNO3D保持原75个未来时间token，完整联合预测后取前25；FNO2D/U-NO/U-Net原生递归25次，GIFT内部RK4仍0.02秒，50步保存26帧。所有模型只在N64训练，目标网格无拟合、无重训。", "",
                  "外频带保留原W2语义：P21、Q21、P22–31、outer P31，以及剔除Nyquist线的reliable outer P31；最后一项只在N96/N128存在。见逐方法metrics JSON/HDF。", "", "局限："])
    lines.extend([f"- {item}" for item in report["limitations"]])
    with target.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    shared.write_json(marker, dict(numeric_report_sha256=report_hash, results_sha256=shared.file_sha256(target)))
    return target


def draw_figures(output, *, force=False):
    """Save new PNG/SVG curves, resolution metrics and fixed-ID fields after numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                               "font.family": "DejaVu Sans", "axes.spines.top": False,
                               "axes.spines.right": False, "legend.frameon": False})
    output = Path(output)
    report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
    numeric_hash, marker = shared.file_sha256(output / "numeric_report.json"), output / "FIGURES_COMPLETE.json"
    if marker.exists():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = {}
        if (not force and saved.get("numeric_report_sha256") == numeric_hash
                and all((output / p).exists() and shared.file_sha256(output / p) == h for p, h in saved.get("figure_sha256", {}).items())):
            return
        shared.preserve_interrupted(marker)
    figures = output / "figures"
    if figures.exists():
        index = 1
        while (output / f"figures_retry_{index:03d}").exists():
            index += 1
        figures = output / f"figures_retry_{index:03d}"
    figures.mkdir()
    created = []
    def save(fig, stem):
        for suffix in ("png", "svg"):
            path = figures / (stem+"."+suffix)
            if path.exists():
                shared.preserve_interrupted(path)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            created.append(path)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, grid in zip(axes, GRIDS):
        for label in METHODS:
            curve = report["methods"][str(grid)][label]["summary"].get("mean_relative_l2_by_time")
            if curve is not None:
                axis.plot(5+QUERY_TIMES[1:], np.asarray(curve[1:], float)*100, label=DISPLAY_NAMES[label], color=COLORS[label],
                          linestyle="--" if label == "GIFT-N50" else "-", linewidth=1.7)
        axis.set_title(f"N64 trained → N{grid}")
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Mean relative L2 (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("W2: native target-grid forecast, one initial field, no target-grid training")
    save(fig, "future25_error")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, key, title in zip(axes, ("future_mean_relative_l2", "t6_relative_l2"), ("Mean future25", "Endpoint t=6")):
        for label in METHODS:
            numbers = [report["methods"][str(n)][label]["summary"].get("metrics", {}).get(key, {}).get("mean") for n in GRIDS]
            axis.plot(GRIDS, np.asarray([np.nan if v is None else v*100 for v in numbers]), marker="o", color=COLORS[label],
                      linestyle="--" if label == "GIFT-N50" else "-", label=DISPLAY_NAMES[label])
        axis.set_xticks(GRIDS)
        axis.set_xlabel("Target spatial grid N")
        axis.set_ylabel("Relative L2 (%)")
        axis.set_title(title)
        axis.grid(alpha=.2)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    save(fig, "cross_resolution_error")
    fig, axis = plt.subplots(figsize=(7, 4.3))
    for label in METHODS:
        numbers = [report["methods"][str(n)][label]["summary"].get("metrics", {}).get("reliable_outer_P31_relative_l2_t6", {}).get("mean") for n in (96, 128)]
        axis.plot((96, 128), [np.nan if x is None else x*100 for x in numbers], marker="o", color=COLORS[label],
                  linestyle="--" if label == "GIFT-N50" else "-", label=DISPLAY_NAMES[label])
    axis.set(xticks=(96, 128), xlabel="Target spatial grid N", ylabel="Relative L2 (%)",
             title="t=6: reliable modes outside P31 (Nyquist lines excluded)")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    save(fig, "outer_P31_t6")
    fixed_id = 100000 if report["cohort"] == "confirmation" else 1040
    row = report["trajectory_ids"].index(fixed_id)
    times = np.asarray(report["keyframe_absolute_times"])
    columns = np.asarray([np.argmin(np.abs((5+QUERY_TIMES)-t)) for t in times])
    for grid in GRIDS:
        truth_path = output / "raw" / f"N{grid}_truth.h5"
        if not truth_path.exists():
            continue
        with h5py.File(truth_path, "r") as source:
            truth = source["vorticity"][row, columns]
        predictions = {}
        for label in METHODS:
            entry = report["methods"][str(grid)][label]
            if "raw_prediction_file" in entry:
                with h5py.File(output / entry["raw_prediction_file"], "r") as source:
                    predictions[label] = source["prediction"][row, columns]
        limit = max([float(np.abs(truth).max()), np.finfo(float).eps] +
                    [float(np.abs(values[np.isfinite(values)]).max()) for values in predictions.values() if np.isfinite(values).any()])
        fig, axes = plt.subplots(len(METHODS)+1, len(times), figsize=(13, 20), squeeze=False)
        for j, (field, absolute) in enumerate(zip(truth, times)):
            axes[0, j].imshow(field, origin="lower", extent=(0, 2*np.pi, 0, 2*np.pi), cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[0, j].set_title(f"t={absolute:g}s")
        axes[0, 0].set_ylabel("Truth")
        for i, label in enumerate(METHODS, 1):
            entry = report["methods"][str(grid)][label]
            if "raw_prediction_file" in entry:
                predicted = predictions[label]
                for j, field in enumerate(predicted):
                    axes[i, j].imshow(np.ma.masked_invalid(field), origin="lower", extent=(0, 2*np.pi, 0, 2*np.pi), cmap="RdBu_r", vmin=-limit, vmax=limit)
                    if not np.isfinite(field).all():
                        axes[i, j].text(.5, .5, "Nonfinite prediction", ha="center", va="center", transform=axes[i, j].transAxes)
            else:
                for axis in axes[i]:
                    axis.text(.5, .5, "Training failed", ha="center", va="center", transform=axis.transAxes)
            axes[i, 0].set_ylabel(DISPLAY_NAMES[label])
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(f"W2 native N{grid}: fixed trajectory {fixed_id}; shared vorticity scale ±{limit:.3g}")
        fig.colorbar(axes[0, 0].images[0], ax=axes.ravel().tolist(), fraction=.015, pad=.01, label="Vorticity")
        save(fig, f"fixed_id_keyframes_N{grid}")
    shared.write_json(marker, dict(numeric_report_sha256=numeric_hash,
                     figure_sha256={str(path.relative_to(output)): shared.file_sha256(path) for path in created},
                     fixed_trajectory_id=fixed_id, ANIE_included=False))


def labelled_paths(values):
    """Parse explicit LABEL=PATH slots, avoiding GIFT/GIFT-N50 identity ambiguity."""
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("checkpoint/failure arguments must be LABEL=PATH")
        label, path = value.split("=", 1)
        if label not in METHODS or label in result or not path:
            raise ValueError("unknown, duplicate or empty W2 method slot")
        result[label] = Path(path)
    return result


def main(argv=None):
    """Evaluate only frozen single-run W2 slots, with no training or test-statistic fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--extension-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--failed-run", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--n64-data-file", type=Path)
    parser.add_argument("--cohort", choices=("confirmation", "old_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reuse-evaluation", action="append", default=[], type=Path)
    parser.add_argument("--job-events", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    return evaluate_checkpoints(labelled_paths(args.checkpoint), args.data_file, args.cohort, args.output,
                               base_lock_file=args.base_lock, extension_lock_file=args.extension_lock,
                               n64_data_file=args.n64_data_file, failed_runs=labelled_paths(args.failed_run),
                               device=args.device, batch_size=args.batch_size, resume=args.resume,
                               plots=not args.skip_plots, reuse_evaluations=args.reuse_evaluation, job_events=args.job_events)


if __name__ == "__main__":
    main()
