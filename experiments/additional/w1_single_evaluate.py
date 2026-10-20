"""Evaluate the four fixed single-training models on the locked N300/76-point task.

This entry point reuses W1-v2's exact time selection, complete-population metrics
and raw-file resume checks. It has one slot per method, actual method-specific
seeds, native terminal/best selection, and no training-seed mean or SD. Model
construction uses the new single-run loader so ANIE's original coordinate profile
cannot silently load with the old aligned profile. All figures read committed
numbers; future observations are never passed to a predictor.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import h5py
import numpy as np

from experiments.additional import w1_v2_evaluate as shared
from scripts.generate_w1_v2_confirmation import locked_configuration

PROFILE = "single_native76_n300"
METHOD_SEEDS = {"anie_ns": 7, "fno2d": 0, "fno3d": 0, "GIFT": 20260820}
QUERY_TIMES = shared.evaluation_grid(dt=.04)
DISPLAY_NAMES = {"anie_ns": "ANIE", "fno2d": "FNO2D", "fno3d": "FNO3D", "GIFT": "GIFT"}


def method_name(value):
    """Canonicalize only the four prescribed method identities."""
    name = "GIFT" if str(value).lower() == "gift" else str(value)
    if name not in METHOD_SEEDS:
        raise ValueError("single-run evaluation accepts only ANIE-NS, FNO2D, FNO3D and GIFT")
    return name


def protocol_lock(path):
    """Validate the fixed recipes without reading any validation or test fields."""
    receipt = locked_configuration(path)
    lock = receipt["lock_snapshot"]
    if lock.get("profile") != PROFILE:
        raise ValueError("single-run evaluation requires its own PROTOCOL_LOCK")
    for method, seed in METHOD_SEEDS.items():
        config = lock["training_configurations"][method]
        if config.get("seed") != seed or config.get("method") != method:
            raise ValueError("locked method/seed differs from the prespecified single run")
        if config.get("profile") != PROFILE or config.get("stage") != "formal":
            raise ValueError("single-run training configurations must use their formal profile")
        if not isinstance(config.get("config"), dict) or not config["config"]:
            raise ValueError("lock requires each job's complete scientific configuration")
    return receipt


def checkpoint_metadata(path, lock):
    """Bind a fresh single-run checkpoint to the recipe and train-only grid.

    A finite, completed terminal FNO/GIFT model remains eligible if its diagnostic
    validation rollout fails: validation is not an extra selection gate. ANIE's
    validation-best checkpoint must have a finite selection MSE.
    """
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_spec", "config", "epoch", "val_metric", "seed", "stage"):
        if key not in payload:
            raise ValueError("single-run checkpoint missing " + key)
    method = method_name(payload["model_spec"]["method"])
    if payload["stage"] != "formal" or int(payload["seed"]) != METHOD_SEEDS[method]:
        raise ValueError("checkpoint stage/seed is outside the fixed single-run population")
    if payload["model_spec"].get("history") != 1 or payload["config"].get("profile") != PROFILE:
        raise ValueError("single-run evaluation requires fresh single-field profile weights")
    job = lock["training_configurations"][method]
    for key, value in job["config"].items():
        if key not in payload["config"] or shared._json_ready(payload["config"][key]) != value:
            raise ValueError("checkpoint recipe differs from PROTOCOL_LOCK at " + key)
    for key, value in job.get("model_spec", {}).items():
        actual = payload["model_spec"].get(key)
        if key == "query_times":
            if actual is None or not np.allclose(actual, value, rtol=0, atol=1e-12):
                raise ValueError("checkpoint model specification differs at query_times")
        elif shared._json_ready(actual) != value:
            raise ValueError("checkpoint model specification differs at " + key)
    if method == "anie_ns":
        spec = payload["model_spec"]
        from training.w1_single_train import NS_SOURCE_PROFILES
        if spec.get("source_profile") not in NS_SOURCE_PROFILES or spec.get("condition_mode") != "native":
            raise ValueError("ANIE must declare a supported NS single-run source profile")
        provenance = payload.get("source_provenance", {}).get("upstream", {})
        if ((spec["source_profile"] == "aligned_ns_single_v1" or "source_profile" in provenance)
                and provenance.get("source_profile") != spec["source_profile"]):
            raise ValueError("ANIE trained source provenance differs from its coordinate profile")
        metric = payload["val_metric"]
        if metric is None or not np.isfinite(float(metric)):
            raise ValueError("ANIE validation-best weights require a finite native selection MSE")
        status_path = Path(path).parent / "STATUS.json"
        if not status_path.is_file():
            raise ValueError("ANIE best weights require the completed training run's STATUS receipt")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("method") != method or status.get("seed") != METHOD_SEEDS[method]:
            raise ValueError("ANIE best weights must come from the completed fixed single run")
        if status.get("selected_checkpoint"):
            selected = Path(status["selected_checkpoint"])
            if not selected.is_absolute():
                selected = status_path.parent / selected
            if selected.resolve() != Path(path).resolve():
                raise ValueError("ANIE checkpoint differs from the run's native best selection")
    else:
        expected_epoch = sum(job["config"]["branch_phases"]) if method == "GIFT" else job["config"]["max_epochs"]
        if payload["epoch"] != expected_epoch:
            raise ValueError("terminal checkpoint has not completed its declared epoch budget")
        if method == "GIFT":
            if payload.get("selection") != "terminal_epoch":
                raise ValueError("GIFT must use native terminal weights")
        elif payload.get("status") != "terminal":
            raise ValueError("FNO must use its terminal checkpoint")
    shared.validate_checkpoint_grid(payload, QUERY_TIMES)
    # Loading never turns an explicitly nonfinite trained state into a successful
    # inference slot. Nonfinite training states belong in declared failed runs.
    def finite_state(value):
        if isinstance(value, dict):
            return all(finite_state(item) for item in value.values())
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            return bool(torch.isfinite(value).all())
        return True
    state = payload.get("model_state", payload.get("model_state_dict"))
    if not isinstance(state, dict) or not finite_state(state):
        raise ValueError("checkpoint requires a finite model state")
    return payload


def load_checkpoint(path, payload, device):
    """Construct the declared source profile, then strictly load its weights."""
    method = method_name(payload["model_spec"]["method"])
    if method == "GIFT":
        from training.w1_v2_gift import load_gift_checkpoint
        low, branch, _ = load_gift_checkpoint(path, device=device)
        return low, branch
    from training.w1_single_train import build_network
    from experiments.model_locations import runtime_model_spec
    model = build_network(runtime_model_spec(payload["model_spec"]), device="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def training_cost(checkpoint, payload):
    """Keep native cost breakdowns plus the completed run's timing receipt."""
    cost = shared.training_cost(checkpoint, payload)
    directory = Path(checkpoint).parent
    receipt_path = directory / "STATUS.json"
    if payload["model_spec"].get("family") == "gift" or str(payload["model_spec"]["method"]).lower() == "gift":
        receipt_path = directory / "COMPLETE.json"
    if receipt_path.is_file():
        cost["terminal_run_receipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))
        cost["terminal_run_receipt_file"] = str(receipt_path)
        cost["terminal_run_receipt_sha256"] = shared.file_sha256(receipt_path)
    return cost


def predict_checkpoint(checkpoint, payload, window, raw_path, *, device="cpu", batch_size=4, previous_raw=None,
                       loader=None, forward=None):
    """Commit autonomous [180,76,64,64] outputs and explicit per-trajectory failures.

    Partial raw files preserve their completed prefix and actual inference cost.
    Programming/resource errors propagate so the orchestrator can resume the same
    attempt; arithmetic prediction failures remain in the scored population.
    """
    import torch
    from training.w1_single_train import forward_model
    loader = load_checkpoint if loader is None else loader
    forward_model = forward_model if forward is None else forward
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup_start = time.perf_counter()
    model = loader(checkpoint, payload, device)
    run_cost = training_cost(checkpoint, payload)
    setup_seconds = time.perf_counter() - setup_start
    spec = payload["model_spec"]
    from experiments.model_locations import runtime_model_spec
    runtime_spec = runtime_model_spec(spec)
    gift = spec.get("family") == "gift" or str(spec["method"]).lower() == "gift"
    population, points = len(window.trajectory_ids), len(QUERY_TIMES)
    failures, reasons = np.zeros(population, bool), [""] * population
    completed, prior_cost = 0, {}
    if previous_raw is not None:
        with h5py.File(previous_raw, "r") as prior:
            completed = int(prior.attrs.get("completed_trajectories", 0))
            prior_cost = json.loads(prior.attrs.get("inference_cost_json", "{}"))
            if completed and not prior_cost:
                raise ValueError("committed predictions require saved cost for resume")
            failures[:completed] = prior["solver_failed"][:completed]
            reasons[:completed] = prior["failure_reason"].asstr()[:completed].tolist()
    if not 0 <= completed < population:
        raise ValueError("partial progress must be inside the declared cohort")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, inference_seconds = time.perf_counter(), 0.
    with h5py.File(raw_path, "x") as raw:
        raw.attrs.update(status="incomplete", checkpoint=str(checkpoint), inference_input="omega(t5) only",
                         cohort=window.cohort, model_spec_json=json.dumps(shared._json_ready(spec), allow_nan=False),
                         profile=PROFILE, time_points=points, future_steps=75, prediction_dt=.04)
        raw.create_dataset("trajectory_index", data=window.trajectory_ids)
        raw.create_dataset("absolute_time", data=5+QUERY_TIMES)
        raw.create_dataset("time", data=QUERY_TIMES)
        predictions = raw.create_dataset("prediction", shape=(population, points, 64, 64), dtype="f4",
                                         chunks=(1, 1, 64, 64), compression="lzf", fillvalue=np.nan)
        stored_failures = raw.create_dataset("solver_failed", data=failures)
        stored_reasons = raw.create_dataset("failure_reason", data=np.asarray(reasons, dtype=h5py.string_dtype("utf-8")))
        if completed:
            with h5py.File(previous_raw, "r") as prior:
                for first in range(0, completed, batch_size):
                    last = min(first+batch_size, completed)
                    predictions[first:last] = prior["prediction"][first:last]

        def commit(last):
            """Flush output/cost before marking its cohort prefix committed."""
            peak = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None
            if peak is not None:
                peak = max(peak, prior_cost.get("peak_cuda_allocated_bytes") or 0)
            cost = {
                "inference_seconds": prior_cost.get("inference_seconds", 0.) + inference_seconds,
                "prediction_and_io_seconds": prior_cost.get("prediction_and_io_seconds", 0.) + time.perf_counter()-started,
                "model_setup_seconds": prior_cost.get("model_setup_seconds", 0.) + setup_seconds,
                "batch_size": batch_size, "device": str(device), "peak_cuda_allocated_bytes": peak,
                "precision": "float32; TF32 disabled",
                "parameter_count": payload.get("parameter_counts", {}).get("total") if gift else sum(p.numel() for p in model.parameters()),
                "parameter_counts": payload.get("parameter_counts") if gift else None,
                "training_cost": run_cost,
                "inference_measurement_resumed": previous_raw is not None,
                "inference_segments": prior_cost.get("inference_segments", []) + [{
                    "first_trajectory": completed, "committed_until": last, "device": str(device),
                    "batch_size": batch_size, "inference_seconds": inference_seconds}],
                "cost_scope": "actual autonomous inference; interrupted native full-path work may include uncommitted outputs",
            }
            cost["seconds_per_trajectory"] = cost["inference_seconds"] / population
            raw.attrs["inference_cost_json"] = json.dumps(shared._json_ready(cost), allow_nan=False)
            stored_failures[:], stored_reasons[:] = failures, reasons
            raw.flush()
            raw.attrs["completed_trajectories"] = last
            raw.flush()
            return cost

        cost = commit(completed)
        if gift:
            from experiments.formal._shared.gift_runtime import rollout_gift
            initial = window.initial(completed, population)
            if not np.isfinite(initial).all():
                raise ValueError("nonfinite initial observations")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            result = rollout_gift(low=model[0], branch=model[1], initial_state=initial,
                                  trajectory_ids=window.trajectory_ids[completed:], duration=3., save_interval=.04,
                                  step=.02, device=torch.device(device), batch_size=batch_size,
                                  correction=True, execution="eager")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter()-before
            gift_prediction = shared._numpy(result.prediction)
            if gift_prediction.shape != (population-completed, points, 64, 64):
                raise ValueError("GIFT returned an incompatible field layout")
            failures[completed:] = np.asarray(result.failure_step) >= 0
            # Native successful trajectories use None; UTF8 HDF strings use
            # an empty reason while preserving every actual failure message.
            reasons[completed:] = ["" if reason is None else reason for reason in result.failure_reason]
            cost = commit(completed)
        for first in range(completed, population, batch_size):
            last = min(first+batch_size, population)
            if gift:
                predictions[first:last] = gift_prediction[first-completed:last-completed]
            else:
                initial = window.initial(first, last)
                if not np.isfinite(initial).all():
                    raise ValueError("nonfinite initial observations")
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                before = time.perf_counter()
                try:
                    with torch.inference_mode():
                        value = shared._numpy(forward_model(model, torch.from_numpy(initial).to(device), runtime_spec,
                                                             payload.get("normalization"), device, training=False))
                    if value.shape != (last-first, points, 64, 64):
                        raise ValueError("predictor returned an incompatible field layout")
                except (ArithmeticError, FloatingPointError) as error:
                    value = np.full((last-first, points, 64, 64), np.nan, np.float32)
                    failures[first:last] = True
                    reasons[first:last] = [type(error).__name__ + ": " + str(error)] * (last-first)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter()-before
                predictions[first:last] = value
            cost = commit(last)
            print(json.dumps({"status": "predictions_committed", "method": spec["method"], "seed": payload["seed"],
                              "completed": last, "population": population}), flush=True)
        raw.attrs["status"] = "complete"
    return cost


def paired_statistics(records, *, bootstrap_replicates=0, seed=2026100304):
    """Pair complete trajectory populations, conditional on these four trained models."""
    by_method = {record["method"]: record for record in records}
    rng, output = np.random.default_rng(seed), []
    for left, right in itertools.combinations(METHOD_SEEDS, 2):
        entry = dict(left=left, right=right, delta_definition="left minus right",
                     bootstrap_scope="trajectory sampling conditional on the single trained model per method; not training-seed uncertainty")
        pair = [by_method[name] for name in (left, right)]
        if any(record["status"] != "complete" for record in pair):
            output.append(dict(entry, status="failed_training_or_predictions"))
            continue
        ids = np.sort(np.asarray(pair[0]["per_trajectory"]["trajectory_id"], np.int64))
        aligned = []
        for record in pair:
            current_ids = np.asarray(record["per_trajectory"]["trajectory_id"], np.int64)
            if len(np.unique(current_ids)) != len(current_ids) or not np.array_equal(np.sort(current_ids), ids):
                raise ValueError("paired populations differ; do not intersect away trajectory IDs")
            lookup = {int(value): index for index, value in enumerate(current_ids)}
            values = np.asarray(record["per_trajectory"]["future_mean_relative_l2"], np.float64)
            aligned.append(values[[lookup[int(value)] for value in ids]])
        delta = aligned[0]-aligned[1]
        if not np.isfinite(delta).all():
            output.append(dict(entry, status="nonfinite_complete_population"))
            continue
        interval = None
        if bootstrap_replicates:
            means = [delta[rng.integers(0, len(ids), len(ids))].mean() for _ in range(bootstrap_replicates)]
            interval = np.quantile(means, [.025, .975]).tolist()
        output.append(dict(entry, status="complete", trajectory_ids=ids.tolist(), delta_values=delta.tolist(),
                           mean_delta=float(delta.mean()), bootstrap_95_percent_interval=interval,
                           bootstrap_replicates=bootstrap_replicates, bootstrap_seed=seed))
    return output


def job_cost_summary(job_events=None, historical_events=None):
    """Keep actual child-process wall times and historical sunk costs separate.

    A process interval includes loading, validation and checkpoint I/O and must
    not be added to its contained epoch timers. Repeated/interrupted attempts are
    retained. Historical totals cover only the explicitly supplied event log.
    """
    def events(path):
        if path is None:
            return [], None
        path = Path(path).resolve(strict=True)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows, {"path": str(path), "sha256": shared.file_sha256(path)}
    rows, source = events(job_events)
    by_method = {method: [] for method in METHOD_SEEDS}
    for row in rows:
        command = row.get("command", [])
        if not any(str(value).startswith("training.w1_single") for value in command):
            continue
        label = str(row.get("label", "")).lower()
        for method in METHOD_SEEDS:
            if method.lower() in label:
                by_method[method].append(row)
                break
    history, history_source = events(historical_events)
    def total(values):
        return sum(float(row["process_seconds"]) for row in values if row.get("process_seconds") is not None)
    unmeasured = [row for row in history if row.get("process_seconds") is None]
    committed_diagnostics = []
    if history_source:
        historical_root = Path(history_source["path"]).parent
        for label in dict.fromkeys(str(row.get("label", "unknown")) for row in unmeasured):
            for phase in ("formal", "development"):
                prefix = phase + "_"
                if not label.startswith(prefix):
                    continue
                status_path = historical_root / phase / label[len(prefix):] / "STATUS.json"
                if status_path.is_file():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    committed_diagnostics.append(dict(label=label, status_file=str(status_path),
                        status_sha256=shared.file_sha256(status_path),
                        values={key: status[key] for key in ("status", "epoch", "updates", "training_seconds", "validation_seconds") if key in status},
                        scope="saved committed-state diagnostic only; excludes uncommitted work and is not an additional process-time measurement"))
    return {
        "new_run_event_source": source,
        "method_training_process_seconds": {name: total(values) if values else None for name, values in by_method.items()},
        "training_attempt_events": by_method, "training_process_seconds_total": total([item for values in by_method.values() for item in values]),
        "all_recorded_new_child_process_seconds": total(rows),
        "historical_sunk_cost": {"event_source": history_source, "recorded_events": len(history),
                                  "recorded_child_process_seconds": total(history) if history_source else None,
                                  "known_recorded_process_seconds": total(history) if history_source else None,
                                  "unclosed_or_unmeasured_labels": list(dict.fromkeys(str(row.get("label", "unknown")) for row in unmeasured)),
                                  "unclosed_or_unmeasured_events": unmeasured,
                                  "committed_state_diagnostics": committed_diagnostics,
                                  "complete_historical_wall_cost_known": False,
                                  "scope": "supplied historical log only; unclosed/unmeasured and absent attempts are not reconstructed; excluded from new method training time and progress"},
        "scope": "JOB_EVENTS actual process wall times including failed/interrupted attempts; epoch breakdowns are contained diagnostics",
    }


def evaluate_checkpoints(checkpoints, data_file, group, cohort, output, *, lock_file, failed_runs=(),
                         device="cpu", batch_size=4, bootstrap_replicates=0, plots=True, resume=False,
                         job_events=None, historical_events=None):
    """Evaluate all four frozen slots, preserving raw-first ordering and exact IDs."""
    if cohort not in ("confirmation", "old_test") or batch_size < 1 or bootstrap_replicates < 0:
        raise ValueError("single-run cohorts/compute options are invalid")
    lock_receipt = protocol_lock(lock_file)
    lock = lock_receipt["lock_snapshot"]
    checkpoints = [Path(path).resolve(strict=True) for path in checkpoints]
    frozen = [checkpoint_metadata(path, lock) for path in checkpoints]
    failures = []
    for source in failed_runs:
        failed = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else dict(source)
        method = method_name(failed.get("method"))
        if failed.get("seed") != METHOD_SEEDS[method] or not failed.get("reason"):
            raise ValueError("failed slot requires its fixed seed and an explicit training failure reason")
        failures.append(dict(failed, method=method, stage="formal", status="training_failed"))
    identities = [method_name(payload["model_spec"]["method"]) for payload in frozen] + [record["method"] for record in failures]
    if len(identities) != 4 or set(identities) != set(METHOD_SEEDS):
        raise ValueError("declare exactly one checkpoint or failed training slot for each of the four methods")
    common = [shared.common_data_identity(payload) for payload in frozen]
    if any(value != common[0] for value in common[1:]):
        raise ValueError("the four models must use identical training/validation observations and IDs")
    if common and "data_binding" in lock:
        locked_identity = shared.common_data_identity(dict(model_spec=dict(method="anie_ns", history=1,
                                                         query_times=QUERY_TIMES.tolist()), data_binding=lock["data_binding"]))
        if common[0] != locked_identity:
            raise ValueError("checkpoint observations differ from the PROTOCOL_LOCK data binding")
    # Cohort opening occurs only after frozen-model/configuration preflight.
    window = shared.observation_window(data_file, group, cohort, query_times=QUERY_TIMES)
    output = Path(output).resolve()
    bindings = [dict(checkpoint=str(path), sha256=shared.file_sha256(path), model_spec=payload["model_spec"],
                     config=payload["config"], seed=payload["seed"], selected_epoch=payload["epoch"], val_metric=payload["val_metric"],
                     source_provenance=payload.get("source_provenance", payload.get("source_metadata")))
                for path, payload in zip(checkpoints, frozen)]
    plan = dict(schema="gift.w1-single-native.evaluation-plan.v1", profile=PROFILE, cohort=cohort,
                interpretation="independent_confirmation" if cohort == "confirmation" else "posthoc_exploratory",
                data_file=str(window.path), group=group, trajectory_ids=window.trajectory_ids,
                query_times=QUERY_TIMES, absolute_times=5+QUERY_TIMES, future_steps=75, time_points=76,
                method_seeds=METHOD_SEEDS, checkpoint_bindings=bindings, failed_training_slots=failures,
                common_training_binding=common[0] if common else None,
                locked_configuration=lock, keyframe_absolute_times=[5.48, 6., 7., 8.], bootstrap_replicates=bootstrap_replicates,
                job_events_file=str(Path(job_events).resolve()) if job_events else None,
                historical_events_file=str(Path(historical_events).resolve()) if historical_events else None)
    if resume:
        if json.loads((output / "evaluation_plan.json").read_text(encoding="utf-8")) != shared._json_ready(plan):
            raise ValueError("resume fixed weights, recipes, grid or complete cohort differ")
        if (output / "numeric_report.json").exists():
            try:
                report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                shared.preserve_interrupted(output / "numeric_report.json")
            else:
                render_report(output)
                if plots:
                    draw_figures(output)
                return report
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "raw").mkdir()
        (output / "metrics").mkdir()
        shared.write_json(output / "evaluation_plan.json", plan)
    records, truth = [], None
    for checkpoint, payload, binding in zip(checkpoints, frozen, bindings):
        method = method_name(payload["model_spec"]["method"])
        raw_path, metric_path = output / "raw" / (method + ".h5"), output / "metrics" / (method + ".h5")
        record_path = output / "metrics" / (method + ".json")
        previous_raw = None
        if raw_path.exists():
            raw_status, cost = shared.raw_checkpoint_status(raw_path, window, checkpoint, payload)
            if raw_status != "complete":
                previous_raw = shared.preserve_interrupted(raw_path)
        if not raw_path.exists():
            cost = predict_checkpoint(checkpoint, payload, window, raw_path, device=device, batch_size=batch_size, previous_raw=previous_raw)
        if metric_path.exists() and record_path.exists():
            with h5py.File(metric_path, "r") as stored:
                complete = stored.attrs.get("status") == "complete"
            if complete:
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    record = None
                if record is not None:
                    if record["checkpoint_sha256"] != binding["sha256"] or record["raw_prediction_sha256"] != shared.file_sha256(raw_path):
                        raise ValueError("saved metrics have different weights or raw predictions")
                    records.append(record)
                    continue
        for incomplete in (metric_path, record_path):
            if incomplete.exists():
                shared.preserve_interrupted(incomplete)
        if truth is None:
            truth_path = output / "raw" / "truth.h5"
            if truth_path.exists():
                with h5py.File(truth_path, "r") as stored:
                    complete = stored.attrs.get("status") == "complete"
                    if complete:
                        if not np.array_equal(stored["trajectory_index"][:], window.trajectory_ids) or not np.array_equal(stored["time"][:], QUERY_TIMES):
                            raise ValueError("saved truth cohort or physical nodes differ")
                        truth = stored["vorticity"][:]
                if not complete:
                    shared.preserve_interrupted(truth_path)
            if truth is None:
                truth = window.truth()
                with h5py.File(truth_path, "x") as stored:
                    stored.attrs["status"] = "incomplete"
                    stored.create_dataset("trajectory_index", data=window.trajectory_ids)
                    stored.create_dataset("time", data=QUERY_TIMES)
                    stored.create_dataset("absolute_time", data=5+QUERY_TIMES)
                    stored.create_dataset("vorticity", data=truth, chunks=(1, 1, 64, 64), compression="lzf")
                    stored.attrs["status"] = "complete"
        measured = shared.score_raw_predictions(raw_path, truth, metric_path)
        record = dict(method=method, seed=int(payload["seed"]), status=measured["summary"]["status"],
                      checkpoint=str(checkpoint), checkpoint_sha256=binding["sha256"],
                      model_spec=payload["model_spec"], config=payload["config"], selected_epoch=payload["epoch"],
                      source_provenance=binding["source_provenance"],
                      validation_metric_diagnostic=payload["val_metric"], summary=measured["summary"],
                      per_trajectory=measured["per_trajectory"], cost=cost, raw_prediction_sha256=shared.file_sha256(raw_path),
                      raw_prediction_file=str(raw_path.relative_to(output)), metric_file=str(metric_path.relative_to(output)))
        shared.write_json(record_path, record)
        records.append(record)
    for failed in failures:
        record = dict(failed, summary=dict(status="training_failed", primary_metric=None, population=180,
                                          failed_count=180, failure_rate=1., failed_ids=window.trajectory_ids.tolist(),
                                          query_times=QUERY_TIMES.tolist(), mean_relative_l2_by_time=None, metrics={}),
                      per_trajectory=dict(trajectory_id=window.trajectory_ids, future_mean_relative_l2=np.full(180, np.inf)))
        failure_path = output / "metrics" / (failed["method"] + ".json")
        if failure_path.exists():
            if json.loads(failure_path.read_text(encoding="utf-8")) != shared._json_ready(record):
                raise ValueError("saved failed training slot differs")
        else:
            shared.write_json(failure_path, record)
        records.append(record)
    methods = {method: next(record for record in records if record["method"] == method) for method in METHOD_SEEDS}
    report = dict(schema="gift.w1-single-native.evaluation.v1", profile=PROFILE, status="numerical_complete",
                  cohort=cohort, interpretation=plan["interpretation"], data_file=str(window.path), group=group,
                  population=180, trajectory_ids=window.trajectory_ids, method_seeds=METHOD_SEEDS,
                  query_times=QUERY_TIMES, absolute_times=5+QUERY_TIMES, time_points=76, future_steps=75, dt=.04,
                  common_training_binding=plan["common_training_binding"], keyframe_absolute_times=plan["keyframe_absolute_times"],
                  methods=methods, paired_comparisons=paired_statistics(records, bootstrap_replicates=bootstrap_replicates),
                  actual_process_cost=job_cost_summary(job_events, historical_events),
                  limitations=["One training instance per method: no training-seed SD or ranking robustness claim.",
                               "Paired trajectory intervals, if requested, condition on these fixed trained models.",
                               "Failures retain the full180 population; finite-subset means are diagnostics only.",
                               "The old test is exploratory after protocol development." if cohort == "old_test" else
                               "The independent confirmation is not used for configuration or checkpoint selection."])
    shared.write_json(output / "numeric_report.json", report)
    render_report(output)
    if plots:
        draw_figures(output)
    return report


def render_report(output):
    """Write the cohort RESULTS.md from saved numbers, without cross-seed statistics."""
    output = Path(output)
    report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
    target = output / "RESULTS.md"
    marker = output / "REPORT_COMPLETE.json"
    report_hash = shared.file_sha256(output / "numeric_report.json")
    if target.exists() and marker.exists():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = None
        if saved == dict(numeric_report_sha256=report_hash, results_sha256=shared.file_sha256(target)):
            return target
    for incomplete in (target, marker):
        if incomplete.exists():
            shared.preserve_interrupted(incomplete)
    def number(value):
        return "—" if value is None else f"{value:.6g}"
    lines = ["# W1 single-training evaluation", "",
             "Independent confirmation (primary)." if report["cohort"] == "confirmation" else "Old test: posthoc exploratory reevaluation.", "",
             "One fixed training instance per method; same180 trajectory IDs and76 nodes at5–8s. The primary score excludes the known t5 field.", "",
             "| Method | Seed | Future75 relative L2 | t8 relative L2 | Future physical MSE | Failed/180 | Training process h | Inference s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    costs = report["actual_process_cost"]["method_training_process_seconds"]
    for method, record in report["methods"].items():
        summary, metrics = record["summary"], record["summary"].get("metrics", {})
        train = costs.get(method)
        lines.append(f"| {DISPLAY_NAMES[method]} | {record['seed']} | {number(summary.get('primary_metric'))} | "
                     f"{number(metrics.get('t8_relative_l2', {}).get('mean'))} | {number(metrics.get('future_mse', {}).get('mean'))} | "
                     f"{summary.get('failed_count', 180)}/180 | {number(train/3600 if train is not None else None)} | "
                     f"{number(record.get('cost', {}).get('inference_seconds'))} |")
    lines += ["", "A failed training slot or any failed prediction invalidates that method's whole-population mean; no successful-trajectory ranking replaces it.", "",
              "Training process time comes from JOB_EVENTS and includes recorded interrupted attempts. Contained epoch/validation timers are available in numeric_report.json and are not added twice.", ""]
    history = report["actual_process_cost"]["historical_sunk_cost"]
    lines.append("Historical known recorded sunk cost (separate from this run): " + number(
        history["recorded_child_process_seconds"]/3600 if history["recorded_child_process_seconds"] is not None else None) + " h.")
    if history["unclosed_or_unmeasured_labels"]:
        lines += ["", "Unclosed/unmeasured historical attempt labels: " + ", ".join(history["unclosed_or_unmeasured_labels"]) +
                  ". Their saved-state diagnostics, if available, are separate and are not added to process times."]
    lines += ["", "ANIE uses its native validation-MSE best checkpoint; FNO/GIFT use terminal weights. Fixed keyframes use ID1040 (old test) or100000 (confirmation), t5.48/6/7/8, each method's actual seed.", "",
              "Raw fields, per-trajectory arrays, failures, model/data/configuration bindings and actual costs are saved before plots.", ""]
    for method, record in report["methods"].items():
        lines.append(f"- {DISPLAY_NAMES[method]}: status={record['status']}; selected/terminal epoch={record.get('selected_epoch', 'unavailable')}" +
                     ("; reason=" + str(record["reason"]) if record.get("reason") else ""))
    lines += ["", *report["limitations"], ""]
    with target.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    shared.write_json(marker, dict(numeric_report_sha256=report_hash, results_sha256=shared.file_sha256(target)))
    return target


def draw_figures(output):
    """Plot only saved numbers, using actual seeds and predeclared ID/time nodes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    try:
        if shared.figures_complete(output):
            return
    except json.JSONDecodeError:
        shared.preserve_interrupted(output / "FIGURES_COMPLETE.json")
    report = json.loads((output / "numeric_report.json").read_text(encoding="utf-8"))
    figures = output / "figures"
    if figures.exists():
        index = 1
        while (output / f"figures_retry_{index:03d}").exists():
            index += 1
        figures = output / f"figures_retry_{index:03d}"
    figures.mkdir()
    times = np.asarray(report["absolute_times"])
    fig, axis = plt.subplots(figsize=(7, 4))
    count = 0
    for method, record in report["methods"].items():
        if record["status"] == "complete":
            axis.plot(times[1:], record["summary"]["mean_relative_l2_by_time"][1:],
                      label=f"{DISPLAY_NAMES[method]} (seed {record['seed']})")
            count += 1
    axis.set(xlabel="Physical time (s)", ylabel="Mean relative L2 error", title="Single initial field at t=5")
    if count:
        axis.legend()
    else:
        axis.text(.5, .5, "No complete prediction population", ha="center", transform=axis.transAxes)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(figures / ("future_error." + extension), dpi=180)
    plt.close(fig)
    keyframes = draw_keyframes(output, figures, report)
    shared.write_json(figures / "provenance.json", dict(numeric_report_sha256=shared.file_sha256(output / "numeric_report.json"),
                                                        training_seed_shading=False, t0_excluded=True, keyframes=keyframes))
    marker = output / "FIGURES_COMPLETE.json"
    if marker.exists():
        shared.preserve_interrupted(marker)
    shared.write_json(marker, dict(numeric_report_sha256=shared.file_sha256(output / "numeric_report.json"),
                                  files=[str(path.relative_to(output)) for path in sorted(figures.iterdir())]))


def draw_keyframes(output, figures, report):
    """Show actual fixed nodes and all four slots, including failures, without interpolation."""
    import matplotlib.pyplot as plt
    fixed_id = 100000 if report["cohort"] == "confirmation" else 1040
    truth_path = output / "raw" / "truth.h5"
    if not truth_path.exists():
        return dict(status="unavailable", reason="all training slots failed; no prediction truth was read", trajectory_id=fixed_id)
    times, columns = np.asarray(report["absolute_times"]), np.array([12, 25, 50, 75])
    if not np.allclose(times[columns], [5.48, 6., 7., 8.], rtol=0, atol=1e-12):
        raise ValueError("fixed keyframes must be the real76-point nodes")
    with h5py.File(truth_path, "r") as saved:
        row = np.flatnonzero(saved["trajectory_index"][:] == fixed_id)
        if len(row) != 1:
            raise ValueError("fixed truth trajectory is absent or repeated")
        truth = saved["vorticity"][int(row[0]), columns]
    predictions = {}
    for method, record in report["methods"].items():
        if record.get("raw_prediction_file"):
            with h5py.File(output / record["raw_prediction_file"], "r") as saved:
                row = np.flatnonzero(saved["trajectory_index"][:] == fixed_id)
                if len(row) != 1:
                    raise ValueError("fixed prediction ID is absent or repeated")
                predictions[method] = saved["prediction"][int(row[0]), columns]
    field_limit, error_limit = max(float(abs(truth).max()), 1e-12), 1e-12
    for value in predictions.values():
        error = abs(value-truth)
        if np.isfinite(error).any():
            error_limit = max(error_limit, float(error[np.isfinite(error)].max()))
    fig, axes = plt.subplots(9, 4, figsize=(11, 18), squeeze=False)
    field_image, error_image = None, None
    for col in range(4):
        field_image = axes[0, col].imshow(truth[col], origin="lower", cmap="RdBu_r", vmin=-field_limit, vmax=field_limit)
        axes[0, col].set_title(f"t={times[columns[col]]:g}")
    axes[0, 0].set_ylabel("Truth")
    for index, (method, record) in enumerate(report["methods"].items()):
        first = 1+2*index
        axes[first, 0].set_ylabel(f"{DISPLAY_NAMES[method]}\nseed {record['seed']}\nprediction")
        axes[first+1, 0].set_ylabel("absolute error")
        for col in range(4):
            if method not in predictions:
                for offset in (0, 1):
                    axes[first+offset, col].text(.5, .5, "Training failed", ha="center", transform=axes[first+offset, col].transAxes)
                continue
            value = predictions[method][col]
            field_image = axes[first, col].imshow(np.ma.masked_invalid(value), origin="lower", cmap="RdBu_r", vmin=-field_limit, vmax=field_limit)
            error_image = axes[first+1, col].imshow(np.ma.masked_invalid(abs(value-truth[col])), origin="lower", cmap="magma", vmin=0, vmax=error_limit)
            if fixed_id in record["summary"].get("failed_ids", []):
                axes[first, col].text(.5, .5, "Failed trajectory", ha="center", transform=axes[first, col].transAxes)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"Prespecified trajectory {fixed_id}; one trained model per method")
    fig.subplots_adjust(left=.18, right=.85, top=.95, bottom=.03, hspace=.18, wspace=.08)
    fig.colorbar(field_image, cax=fig.add_axes([.88, .55, .015, .32]), label="Vorticity (truth-fixed limits)")
    if error_image is not None:
        fig.colorbar(error_image, cax=fig.add_axes([.88, .12, .015, .32]), label="Absolute error (shared limits)")
    for extension in ("png", "svg"):
        fig.savefig(figures / ("fixed_id_keyframes." + extension), dpi=160)
    plt.close(fig)
    return dict(status="complete", trajectory_id=fixed_id, method_seeds=report["method_seeds"],
                absolute_time=times[columns].tolist(), source_time_indices=columns.tolist(),
                selection="prespecified ID/time nodes, never selected by performance", interpolation=False,
                field_limits=[-field_limit, field_limit], error_limits=[0, error_limit],
                truth_sha256=shared.file_sha256(truth_path),
                source_raw_files=[record["raw_prediction_file"] for record in report["methods"].values() if record.get("raw_prediction_file")])


def main(argv=None):
    """Run frozen-model evaluation only; never train or select configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--failed-run", type=Path, action="append", default=[])
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--group", default="N64/test")
    parser.add_argument("--cohort", choices=("confirmation", "old_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--job-events", type=Path)
    parser.add_argument("--historical-events", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    return evaluate_checkpoints(args.checkpoint, args.data_file, args.group, args.cohort, args.output,
                                lock_file=args.lock, failed_runs=args.failed_run, device=args.device,
                                batch_size=args.batch_size, bootstrap_replicates=args.bootstrap_replicates,
                                plots=not args.skip_plots, resume=args.resume, job_events=args.job_events,
                                historical_events=args.historical_events)


if __name__ == "__main__":
    main()
