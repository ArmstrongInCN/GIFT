"""Evaluate a completed ANIE run once over t=5..8; retain all 180 trajectories.

This separate entry only reads the published six-method W1 result. It does not
modify its source, bound results, data, or checkpoints. Without --execute it
prints a plan and creates no output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
for _root in (ROOT, ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import h5py
import numpy as np

from experiments.formal._shared.common import (
    file_record, json_ready, relative_l2, write_csv_new, write_json_new,
)
from experiments.formal._shared.figure_evidence import bind_result, finish_figures
from adapters import public_metadata

TEST_IDS = np.arange(1040, 1220, dtype=np.int64)
TIMES = 5.0 + np.arange(151, dtype=np.float64) * .02
REPORT_STEPS = np.arange(25, 151, 25, dtype=np.int64)
REFERENCE_METHODS = ("GIFT", "GIFT-Lite", "FNO-2D", "FNO-3D", "U-NO", "U-Net")
COHORT = "test_1040_1219"


def configure_inference_precision():
    """Use float32 inference without TF32 kernels."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def digest_file(path):
    """Stream a scientific input/reference digest without loading its full bytes into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_formal_metadata(weights):
    """Budget labels must agree with the terminal export and scientific identity."""
    expected = {
        "schema": "gift.anie-dense-terminal-weights.v1", "status": "complete",
        "method": "anie_dense151", "artifact_role": "fresh_terminal",
        "fresh_training": True, "completed_epochs": 500, "terminal_epoch": 500,
    }
    for key, value in expected.items():
        if weights.get(key) != value or (isinstance(value, bool) and weights.get(key) is not value):
            raise ValueError("ANIE formal terminal metadata differs: " + key)
    configuration = weights.get("formal_configuration", {})
    expected_configuration = {
        "method": "anie_dense151", "protocol": "anie.dense151.v1",
        "epochs": 500, "trajectories": 1000, "validation_trajectories": 20,
        "dt": .02, "window_seconds": 3., "frames": 151,
        "training_time_points": 151, "prediction_time_points": 151,
        "effective_batch": 32, "micro_batch": 4, "accumulation": 8, "drop_last": False,
        "seed": 20260820, "precision": "float32", "amp": False,
        "matmul_tf32": False, "cudnn_tf32": False, "initial_mode": "single_observed_field",
        "initialization": "random_from_seed_no_published_checkpoint",
        "loss": "torch.nn.functional.mse_loss_mean_151_observed_times_including_initial",
        "teacher_forcing": False, "normalization": "none_raw_observed_vorticity",
        "selection": "terminal_epoch_not_validation", "formal": True,
        "fixture_only": False,
    }
    for key, value in expected_configuration.items():
        if configuration.get(key) != value or (
            isinstance(value, bool) and configuration.get(key) is not value
        ):
            raise ValueError("ANIE formal budget/configuration differs: " + key)
    if (configuration.get("training_frame_indices") != np.arange(151, dtype=int).tolist()
            or configuration.get("training_local_times") != (np.arange(151, dtype=int) * .02).tolist()
            or configuration.get("micro_batch", 0) * configuration.get("accumulation", 0) != 32):
        raise ValueError("ANIE training time sampling/effective batch differs")
    provenance = weights.get("run_provenance", {})
    identity = provenance.get("identity", {})
    if (identity.get("kind") != "gift.anie-dense-training.v1"
            or identity.get("configuration") != configuration
            or identity.get("model_configuration") != weights.get("model_configuration")
            or not provenance.get("run_id")):
        raise ValueError("ANIE export has no matching fresh-run identity")
    data = identity.get("data", {})
    if (data.get("cohort") != "four_vortex" or data.get("fixture_only") is not False
            or data.get("training_ids") != list(range(1000))
            or data.get("validation_ids") != list(range(1000, 1020))):
        raise ValueError("ANIE training/validation population differs from W1")
    validate_time_grid_metadata(identity)
    return identity


def validate_time_grid_metadata(identity):
    """Check the serialized physical/normalized 151-point float32 query vectors.

    Physical local time runs from 0 to 3 and normalized time from 0 to 1. Their
    relationship is checked with float32 tolerance, without a current-device or
    source-identity eligibility gate.
    """
    grid = identity.get("time_grid", {})
    if any(grid.get(key) != value for key, value in
           (("frames", 151), ("dtype", "float32"), ("window_seconds", 3.))):
        raise ValueError("Dense ANIE requires its actual 151-point time-grid identity")
    physical = np.asarray(identity["configuration"]["training_local_times"], dtype=np.float32)
    for stem, terminal in (("physical", 3.), ("normalized", 1.)):
        values = np.asarray(grid.get(stem + "_float32_values", []), dtype=np.float32)
        if (values.shape != (151,) or not np.isfinite(values).all() or values[0] != 0.
                or values[-1] != terminal or not (np.diff(values) > 0).all()):
            raise ValueError("Dense ANIE recorded actual time-grid vector differs: " + stem)
    if not np.array_equal(physical, np.asarray(grid["physical_float32_values"], dtype=np.float32)):
        raise ValueError("Dense ANIE physical time-grid bits differ from its training configuration")
    if not np.allclose(np.asarray(grid["normalized_float32_values"], dtype=np.float64),
                       physical.astype(np.float64) / 3., rtol=0, atol=2e-7):
        raise ValueError("Dense ANIE recorded normalized time grid is not the physical grid divided by three")


def actual_time_grid(physical):
    """Record the actual device vectors consumed by the native adapter."""
    normalized = physical / 3.
    return {            "physical_float32_values": physical.detach().cpu().tolist(),
            "normalized_float32_values": normalized.detach().cpu().tolist(),
            "byte_order": "little", "declared_backend": physical.device.type,
            "device": str(physical.device),
            "frames": 151, "dtype": "float32", "window_seconds": 3.}


def load_formal_predictor(checkpoint, data_root, source_root, device):
    """Load a terminal export without the original training directory."""
    import torch
    from adapters.anie import ANIEConfig, ANIEPredictor

    configure_inference_precision()
    checkpoint = Path(checkpoint).resolve(strict=True)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = validate_formal_metadata(weights)
    predictor = ANIEPredictor(config=ANIEConfig(**weights["model_configuration"]),
                              source_root=Path(source_root))
    predictor.load_state_dict(weights["model_state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in predictor.state_dict().values()):
        raise ValueError("ANIE export contains nonfinite weights")
    predictor.to(device).eval()
    return predictor, weights, public_metadata({
        "checkpoint": file_record(checkpoint), "training_identity": identity,
        "upstream": predictor.source_identity(),
        "training_provenance_role": "historical metadata; training journal not required for inference",
    })


def _numpy(value):
    """Detach a predictor tensor onto CPU, or preserve an already numerical array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def batch_metrics(prediction, truth, decoder_initial):
    """Keep failure states and distinguish the decoder from the solver t0 output."""
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if (prediction.shape != truth.shape or prediction.ndim != 4
            or prediction.shape[1] != 151 or prediction.shape[-2:] != (64, 64)):
        raise ValueError("ANIE W1 prediction must be [B,151,64,64]")
    if not np.isfinite(truth).all():
        raise ValueError("W1 truth contains nonfinite observations")
    decoder_initial = np.asarray(decoder_initial, dtype=np.float32)
    if decoder_initial.shape != truth[:, 0].shape:
        raise ValueError("ANIE initial decoder reconstruction shape differs")
    fields_finite = np.isfinite(prediction).all(axis=(-2, -1))
    future_finite = fields_finite[:, 1:]
    first = np.full(len(prediction), -1, dtype=np.int32)
    for index, row in enumerate(future_finite):
        failed = np.flatnonzero(~row)
        if failed.size:
            first[index] = int(failed[0])
    with np.errstate(invalid="ignore", over="ignore"):
        errors = relative_l2(prediction, truth)
        reconstruction = relative_l2(decoder_initial, truth[:, 0])
    return {
        "full_relative_l2": errors, "field_finite": fields_finite,
        "finite_by_trajectory_future_step": future_finite,
        "first_nonfinite_future_step_zero_based": first,
        "all_150_future_fields_finite": future_finite.all(axis=1),
        "initial_decoder_error": reconstruction,
        "solver_t0_error": errors[:, 0],
    }


def predict_and_measure(predictor, initial, truth):
    """The model receives only initial, once; truth is used solely afterwards."""
    prediction = _numpy(predictor(initial))
    decoder = _numpy(predictor.initial_decoder(initial))
    return prediction, decoder, batch_metrics(prediction, truth, decoder)


def population_summary(values):
    """Describe the entire error population and keep finite-subset values explicitly diagnostic.

    If any trajectory fails, whole-population mean/spread are unavailable; original
    population and finite counts remain in the report.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    kept = values[finite]
    complete = bool(finite.all())
    return {
        "population": int(values.size), "finite_count": int(finite.sum()),
        "complete_population": complete,
        "mean": float(values.mean()) if complete else None,
        "sample_sd": float(values.std(ddof=1)) if complete and values.size > 1 else None,
        "median": float(np.median(values)) if complete else None,
        "p95": float(np.quantile(values, .95)) if complete else None,
        "maximum": float(values.max()) if complete else None,
        "finite_subset_mean_diagnostic": float(kept.mean()) if kept.size else None,
    }


def test_layout(handle):
    """Check the independent four-vortex N64 observation layout and return its lazy HDF5 field.

    The 180 IDs are fixed before inference. Native absolute times span t=4.1..8.0
    in dt=0.02 steps; frame 45 is the t=5 initial condition.
    """
    from gift.prediction_cohorts import observation_cohort
    if (observation_cohort(handle) != "four_vortex"
            or handle.attrs.get("trajectory_id_scheme") != "canonical"
            or bool(handle.attrs.get("fixture_only", False))):
        raise ValueError("ANIE W1 requires canonical four-vortex independent observations")
    group = handle["N64/test"]
    ids = group["trajectory_index"][:]
    times = group["time"][:]
    field = group["vorticity"]
    if (ids.dtype != np.dtype("int64") or not np.array_equal(ids, TEST_IDS)
            or times.dtype != np.dtype("float64") or len(times) != 196
            or not np.allclose(times, 4.1 + np.arange(196) * .02, rtol=0, atol=2e-12)
            or field.dtype != np.dtype("float32") or field.shape != (180, 196, 64, 64)):
        raise ValueError("W1 independent identities/shape/time support differ")
    if not np.allclose(times[45:], TIMES, rtol=0, atol=2e-12):
        raise ValueError("W1 evaluation must use observed t=5..8 at dt=.02")
    return field


def reference_rows(directory, test_hash):
    """Read saved six-method W1 means for the same independent observation content.

    This compares recorded population/time/input provenance; it does not rerun
    baselines or perform full artifact-integrity verification.
    """
    directory = Path(directory).resolve(strict=True)
    integrity = {"status": "METADATA_READ", "numerical_metrics_recomputed": False}
    manifest = json.loads((directory / "published.json").read_text(encoding="utf-8"))
    measurement = manifest["measurement"]["source_measurement"]
    original = measurement["inputs"]["fno_test_data"]
    if original["sha256"].lower() != test_hash.lower():
        raise ValueError("Published W1 and new ANIE use different independent observed bytes")
    with (directory / "summary/metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["cohort"] == COHORT]
    output = []
    for method in REFERENCE_METHODS:
        for step in REPORT_STEPS:
            matches = [row for row in rows if row["method"] == method
                       and abs(float(row["absolute_time"]) - TIMES[step]) < 1e-12]
            count = 3 if method in ("GIFT", "GIFT-Lite") else 1
            if (len(matches) != count or any(int(row["population"]) != 180 for row in matches)
                    or any(int(row["finite_count"]) != 180 for row in matches)):
                raise ValueError("Published W1 six-method key-time population differs")
            if count == 3 and {row["seed"] for row in matches} != {"20260820", "20260821", "20260822"}:
                raise ValueError("Published GIFT seed set differs")
            means = np.asarray([float(row["mean"]) for row in matches])
            if not np.isfinite(means).all():
                raise ValueError("Published W1 means are nonfinite")
            output.append({"method": method, "absolute_time": float(TIMES[step]),
                           "mean_relative_l2": float(means.mean()),
                           "training_seed_sd": float(means.std(ddof=1)) if count == 3 else None,
                           "n_training_seeds": count, "population": 180, "finite_count": 180,
                           "complete_population": True, "evidence_kind": "existing_published_measurement"})
    return output, {"integrity": integrity, "published_manifest": file_record(directory / "published.json"),
                    "metrics": file_record(directory / "summary/metrics.csv"),
                    "source_report_sha256": manifest["measurement"]["report_sha256"],
                    "baselines_rerun": False}


def anie10_reference_rows(directory, test_hash):
    """Read an existing measurement with its population, time and data identity."""
    directory = Path(directory).resolve(strict=True)
    published = json.loads((directory / "published.json").read_text(encoding="utf-8-sig"))
    if (published.get("schema") != "gift.additional-published-results.v1"
            or published.get("experiment") != "W1-ANIE" or published.get("status") != "complete"):
        raise ValueError("ANIE-10 reference is not a completed publication")
    numeric_path = directory / "measurement/numeric_report.json"
    numeric = json.loads(numeric_path.read_text(encoding="utf-8-sig"))
    report_hash = digest_file(numeric_path).upper()
    training = numeric["inputs"]["new_anie"]["training_identity"]
    config = training["configuration"]
    if (numeric.get("experiment") != "W1-ANIE" or numeric.get("status") != "numerical_complete"
            or training.get("kind") != "gift.anie-training.v1"
            or config.get("training_time_points") != 10 or config.get("prediction_time_points") != 151
            or config.get("epochs") != 500 or config.get("trajectories") != 1000
            or numeric.get("independent_ids") != TEST_IDS.tolist()
            or numeric.get("times") != TIMES.tolist()
            or numeric["inputs"] != published.get("source_identities")
            or numeric["inputs"]["independent_observations"]["sha256"].lower() != test_hash.lower()):
        raise ValueError("ANIE-10 protocol/data identity differs from its measured reference")
    with (directory / "summary/metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    output = []
    for step in REPORT_STEPS:
        matches = [row for row in source_rows if row["method"] == "ANIE"
                   and abs(float(row["absolute_time"]) - TIMES[step]) < 1e-12]
        if len(matches) != 1:
            raise ValueError("ANIE-10 reference report time is missing/duplicated")
        row = matches[0]
        if int(row["population"]) != 180 or int(row["finite_count"]) != 180 or row["complete_population"] != "True":
            raise ValueError("ANIE-10 reference population differs")
        mean = float(row["mean"])
        if not np.isfinite(mean):
            raise ValueError("ANIE-10 reference mean is not finite")
        output.append({"method": "ANIE-10", "absolute_time": float(TIMES[step]),
                       "mean_relative_l2": mean, "training_seed_sd": None, "n_training_seeds": 1,
                       "population": 180, "finite_count": 180, "complete_population": True,
                       "evidence_kind": "existing_published_anie10_measurement"})
    return output, {"status": "METADATA_READ", "integrity_verified": False,
                    "published_manifest": file_record(directory / "published.json"),
                    "numeric_report": file_record(numeric_path), "metrics": file_record(directory / "summary/metrics.csv"),
                    "source_report_sha256": report_hash, "training_time_points": 10,
                    "prediction_time_points": 151, "reference_recomputed": False}


def draw_comparison(output, figure_output=None):
    """New SVG, preserving the existing W1 typography/palette and measured values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.interpolate import PchipInterpolator
    from experiments.formal.w1_recursive_prediction.plot_mean_error import STYLES
    evidence = bind_result(output, "W1-ANIE-151", ("summary/comparison.csv",))
    with (output / "summary/comparison.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    figures = Path(figure_output) if figure_output is not None else output / "figures"
    figures.mkdir(exist_ok=False)
    write_csv_new(figures / "source_data.csv", rows)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "font.size": 8, "axes.labelsize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.linewidth": .8, "axes.spines.right": False,
                         "axes.spines.top": False, "legend.frameon": False})
    figure, axis = plt.subplots(figsize=(183 / 25.4, 96 / 25.4))
    methods = (*REFERENCE_METHODS, "ANIE-10", "ANIE-151")
    styles = dict(STYLES, **{"ANIE-10": ("#C42D57", "P", "-", 1.8, 7),
                            "ANIE-151": ("#374151", "*", "-", 1.8, 11)})
    handles, labels, maximum = [], [], 0.
    for method in methods:
        selected = sorted([row for row in rows if row["method"] == method],
                          key=lambda row: float(row["absolute_time"]))
        # Never replace a missing full-population point with a finite-subset mean.
        prefix = []
        for row in selected:
            if row["complete_population"] != "True":
                break
            prefix.append(row)
        color, marker, line, width, zorder = styles[method]
        label = method + (" (new)" if method == "ANIE-151" else " (published)")
        if len(prefix) != len(selected):
            label += " [nonfinite futures]"
        handles.append(Line2D([0], [0], color=color, marker=marker, linewidth=width))
        labels.append(label)
        if not prefix:
            continue
        times = np.asarray([float(row["absolute_time"]) for row in prefix])
        values = np.asarray([float(row["mean_relative_l2"]) for row in prefix])
        maximum = max(maximum, float(values.max()))
        if len(prefix) > 1:
            curve = PchipInterpolator(times, values)
            points = np.linspace(times[0], times[-1], 301)
            axis.plot(points, curve(points), color=color, linewidth=width, zorder=zorder)
        axis.plot(times, values, linestyle="none", color=color, marker=marker,
                  markersize=4.2, markeredgecolor="white", markeredgewidth=.55, zorder=zorder+.2)
    axis.set(xlim=(5.46, 8.04), ylim=(0, max(1.28, 1.08 * maximum)),
             xlabel=r"Time, $t$", ylabel=r"Mean relative $L^2$ error")
    axis.set_xticks(TIMES[REPORT_STEPS])
    axis.tick_params(direction="out", length=3., width=.8, pad=2.5)
    axis.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, 1.015),
                ncol=4, handlelength=2.5, columnspacing=1.3, handletextpad=.55)
    figure.subplots_adjust(left=.105, right=.985, bottom=.165, top=.80)
    figure.savefig(figures / "w1_with_anie151_mean_relative_l2.svg", bbox_inches="tight")
    limits = list(axis.get_ylim())
    plt.close(figure)
    finish_figures(figures, evidence, Path(__file__), {
        "methods": list(methods), "metric": "mean per-trajectory full-field relative L2",
        "population": 180, "curve": "PCHIP visual guide through six measured future times",
        "anchor_excluded": "t=5 is a given observation, not evidence of decoder reconstruction",
        "existing_measurements_not_rerun": [*REFERENCE_METHODS, "ANIE-10"], "y_limits": limits,
    })


def evaluate(args):
    """Evaluate the completed 151-point-trained ANIE model on all 180 independent trajectories.

    Only the observed t=5 field enters each autonomous predictor call. Prediction,
    decoder reconstruction, raw solver t0 output and future failures are stored
    separately on the 151-point t=5..8 grid. Existing comparison measurements are
    read without rerunning their models; plotting follows numerical publication.
    """
    import torch
    from training.checkpoints import runtime_identity
    configure_inference_precision()
    data_root = args.data_root.resolve(strict=True)
    from gift.paths import data_path
    test = data_path(data_root, "four_vortex.fno_test_dt0p02", "fno/fno_test_n64_n96_n128_dt0p02.h5")
    test_hash = digest_file(test)
    published, reference = reference_rows(args.reference, test_hash)
    previous_rows, previous_reference = anie10_reference_rows(args.reference_anie10, test_hash)
    published += previous_rows
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor, weights, identity = load_formal_predictor(args.checkpoint, data_root, args.source_root, device)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    query_times = torch.as_tensor(weights["formal_configuration"]["training_local_times"],
                                  dtype=torch.float32, device=device)
    inference_grid = actual_time_grid(query_times)
    output = args.output.resolve(strict=False)
    if output.exists():
        raise FileExistsError("Refusing to overwrite additional ANIE result: " + str(output))
    if output == ROOT:
        raise ValueError("ANIE result must use a new output directory")
    for protected in (data_root, args.source_root.resolve(strict=True)):
        if output == protected or protected in output.parents:
            raise ValueError("ANIE result must be outside dataset and upstream checkout")
    output.mkdir(parents=True)
    for name in ("raw", "summary"):
        (output / name).mkdir()
    all_errors = np.empty((180, 151), dtype=np.float64)
    finite = np.empty((180, 150), dtype=bool)
    first = np.empty(180, dtype=np.int32)
    decoder_errors = np.empty(180, dtype=np.float64)
    trajectory_rows, initial_rows, solver_batches = [], [], []
    started = time.perf_counter()
    with h5py.File(test, "r") as observations, h5py.File(output / "raw/predictions.h5", "x") as raw:
        fields = test_layout(observations)
        raw.attrs.update(schema="gift.additional.W1.ANIE151.raw.v1", method="ANIE-151",
                         source_observations_sha256=test_hash, autonomous_input="omega(t=5) only",
                         initial_output_policy="raw solver output; no observed-anchor substitution")
        raw.create_dataset("trajectory_index", data=TEST_IDS)
        raw.create_dataset("time", data=TIMES)
        predictions = raw.create_dataset("prediction", shape=(180, 151, 64, 64), dtype="float32",
                                         chunks=(1, 1, 64, 64), compression="gzip", compression_opts=4)
        truths = raw.create_dataset("truth", shape=(180, 151, 64, 64), dtype="float32",
                                    chunks=(1, 1, 64, 64), compression="gzip", compression_opts=4)
        decoders = raw.create_dataset("initial_decoder", shape=(180, 64, 64), dtype="float32")
        for start in range(0, 180, args.batch_size):
            end = min(start + args.batch_size, 180)
            # Only frame 45 is read before invoking the predictor. Future truth
            # never enters its signature, interpolation, initialization or solver.
            initial_array = np.ascontiguousarray(fields[start:end, 45], dtype=np.float32)
            if not np.isfinite(initial_array).all():
                raise ValueError("Nonfinite independent initial observation")
            initial = torch.from_numpy(initial_array).to(device)
            with torch.no_grad():
                prediction = _numpy(predictor(initial, times=query_times))
                decoder = _numpy(predictor.initial_decoder(initial))
            solver_batches.append({"first_trajectory_id": int(TEST_IDS[start]),
                                   "last_trajectory_id": int(TEST_IDS[end - 1]), "batch_size": end - start,
                                   "native_solver_iterations": int(predictor.last_solver_iterations)})
            truth = np.asarray(fields[start:end, 45:], dtype=np.float32)
            metrics = batch_metrics(prediction, truth, decoder)
            predictions[start:end] = prediction
            truths[start:end] = truth
            decoders[start:end] = decoder
            all_errors[start:end] = metrics["full_relative_l2"]
            finite[start:end] = metrics["finite_by_trajectory_future_step"]
            first[start:end] = metrics["first_nonfinite_future_step_zero_based"]
            decoder_errors[start:end] = metrics["initial_decoder_error"]
            for index in range(end - start):
                tid = int(TEST_IDS[start + index])
                initial_rows.append({"trajectory_id": tid, "observed_anchor_error": 0.,
                                     "initial_decoder_error": float(decoder_errors[start + index]),
                                     "solver_t0_error": float(all_errors[start + index, 0]),
                                     "all_150_future_fields_finite": bool(finite[start + index].all()),
                                     "first_nonfinite_future_step_zero_based": int(first[start + index]),
                                     "first_nonfinite_absolute_time": None if first[start + index] < 0
                                     else float(TIMES[first[start + index] + 1])})
                for step, absolute_time in enumerate(TIMES):
                    trajectory_rows.append({"trajectory_id": tid, "absolute_time": float(absolute_time),
                                            "full_relative_l2": float(all_errors[start + index, step]),
                                            "field_finite": bool(metrics["field_finite"][index, step])})
            raw.flush()
            print(json.dumps({"completed_test_trajectories": end, "population": 180,
                              "elapsed_prediction_and_write_seconds": time.perf_counter() - started}), flush=True)
        raw.create_dataset("full_relative_l2", data=all_errors)
        raw.create_dataset("finite_by_trajectory_future_step", data=finite)
        raw.create_dataset("first_nonfinite_future_step_zero_based", data=first)
        raw.create_dataset("initial_decoder_error", data=decoder_errors)
    metrics_rows = [{"cohort": COHORT, "experiment": "W1-ANIE-151", "method": "ANIE-151",
                     "seed": weights["formal_configuration"]["seed"], "absolute_time": float(value),
                     "lead_time": float(value - 5), **population_summary(all_errors[:, index])}
                    for index, value in enumerate(TIMES)]
    comparison = list(published)
    for step in REPORT_STEPS:
        summary = population_summary(all_errors[:, step])
        comparison.append({"method": "ANIE-151", "absolute_time": float(TIMES[step]),
                           "mean_relative_l2": summary["mean"], "training_seed_sd": None,
                           "n_training_seeds": 1, "population": 180,
                           "finite_count": summary["finite_count"],
                           "complete_population": summary["complete_population"],
                           "evidence_kind": "new_anie_measurement"})
    write_csv_new(output / "raw/trajectory_metrics.csv", trajectory_rows)
    write_csv_new(output / "raw/initial_metrics.csv", initial_rows)
    write_csv_new(output / "summary/metrics.csv", metrics_rows)
    write_csv_new(output / "summary/comparison.csv", comparison)
    summary = {
        "method": "ANIE-151", "population": 180, "complete_prediction_finite_count": int(finite.all(axis=1).sum()),
        "finite_count_by_future_step": finite.sum(axis=0).astype(int).tolist(),
        "initial_decoder_error": population_summary(decoder_errors),
        "solver_t0_error": population_summary(all_errors[:, 0]),
        "observed_anchor": "input observation; error 0 is not a reconstruction measurement",
        "t8": population_summary(all_errors[:, -1]), "prediction_design": "one full-sequence solver call per batch",
        "training_cost": weights.get("training_cost"), "elapsed_prediction_and_write_seconds": time.perf_counter() - started,
        "inference_batch_size": args.batch_size, "native_solver_batches": solver_batches,
    }
    write_json_new(output / "summary/summary.json", summary)
    numeric_files = [file_record(path, output) for path in sorted(output.rglob("*")) if path.is_file()]
    report = {
        "schema": "gift.additional.W1.ANIE151.v1", "experiment": "W1-ANIE-151", "status": "numerical_complete",
        "inputs": {"new_anie": identity, "independent_observations": file_record(test),
                   "existing_W1": reference, "existing_ANIE10": previous_reference},
        "evaluation_source": file_record(Path(__file__), ROOT), "numeric_files": numeric_files,
        "inference_numerical_profile": runtime_identity(),
        "inference_device": str(next(predictor.parameters()).device),
        "inference_time_grid": inference_grid,
        "inference_time_grid_policy": "151 recorded training times; actual inference device tokens recorded separately",
        "summary": summary, "independent_ids": TEST_IDS.tolist(), "times": TIMES.tolist(),
        "failure_policy": "retain all predictions; no finite-subset mean reported as full population",
        "baseline_measurement_scope": "six original W1 methods and published ANIE-10; integrity checked; not recomputed",
        "prediction_scope": "only omega(5) initial input; no future observations or teacher forcing",
    }
    report = public_metadata(report)
    write_json_new(output / "numeric_report.json", report)
    status = "complete"
    plot_error = None
    if not args.skip_plots:
        try:
            draw_comparison(output)
        except Exception as error:
            status = "numerical_complete_plot_failed"
            plot_error = error
            write_json_new(output / "plot_failure.json", {"type": type(error).__name__, "message": str(error)})
    report["status"] = status
    report["numerical_report_sha256"] = digest_file(output / "numeric_report.json")
    write_json_new(output / "report.json", report)
    write_json_new(output / "manifest.json", {"schema": "gift.additional-results.v1", "experiment": "W1-ANIE-151",
                                               "status": status, "files": [file_record(path, output) for path in sorted(output.rglob("*"))
                                                                            if path.is_file()]})
    print(json.dumps(json_ready({"output": str(output), "status": status, "summary": summary}), ensure_ascii=False))
    if plot_error is not None:
        raise RuntimeError(f"Numerical results saved in {output}; plotting failed") from plot_error
    return report


def main(argv=None):
    """Print the selected ANIE evaluation plan or execute it into a fresh result directory."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/formal/W1_recursive_prediction")
    parser.add_argument("--reference-anie10", type=Path, default=ROOT / "results/additional/W1_ANIE_20261001")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "method": "ANIE-151", "population": 180,
                          "input": "omega(5) only", "frames": 151, "dt": .02, "end_time": 8.,
                          "evaluation_profile": "W1 reference protocol; 500 declared training epochs",
                          "output": str(args.output), "outputs_created": False}, indent=2))
        return
    evaluate(args)


if __name__ == "__main__":
    main()
