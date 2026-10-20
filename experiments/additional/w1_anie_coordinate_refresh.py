"""Replace only ANIE in the immutable seven-method, two-cohort W1 comparison.

New ANIE inference uses its trained coordinate profile and all180 trajectories.
The other six raw predictions and metric arrays are reused byte-for-byte. No
future observation reaches any predictor, and no test score selects a model.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.additional import w1_extended_evaluate as extended
from experiments.additional import w1_single_evaluate as native
from experiments.additional import w1_v2_evaluate as shared


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _evaluation_plan(plan, lock, slot, previous, old_report, cohort):
    """Bind the cache to the current selected model, lock and immutable source."""
    checkpoint = Path(slot["checkpoint"]) if slot["status"] == "complete" else None
    failure = slot.get("failure")
    return dict(schema="gift.anie-coordinate-refresh.evaluation-plan.v1", cohort=cohort,
                source_report=str(previous / "numeric_report.json"),
                source_report_sha256=shared.file_sha256(previous / "numeric_report.json"),
                checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=shared.file_sha256(checkpoint) if checkpoint else None,
                failure=failure, failure_sha256=shared.file_sha256(failure) if failure else None,
                source_profile=plan["source_profile"], lock=lock,
                trajectory_ids=old_report["trajectory_ids"], query_times=native.QUERY_TIMES.tolist())


def complete_evaluation(output):
    """Accept a cache only when bytes and the current queue inputs still match."""
    output = Path(output)
    marker = output / "EVALUATION_COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        complete = read(marker)
    except (OSError, json.JSONDecodeError):
        return False
    if complete.get("status") != "complete":
        return False
    hashes = complete.get("file_sha256")
    if not isinstance(hashes, dict) or not {"numeric_report.json", "evaluation_plan.json", "raw/truth.h5"} <= set(hashes):
        return False
    if not all((output / name).is_file() and shared.file_sha256(output / name) == digest
               for name, digest in hashes.items()):
        return False
    # The queue calls this function before evaluate(), so its early skip must
    # enforce the same identity gate as an explicit resume, without inference.
    root = output.parent.parent
    try:
        plan, lock, slot = (read(root / name) for name in ("RUN_PLAN.json", "PROTOCOL_LOCK.json", "FORMAL.json"))
        previous = Path(plan["old_root"]) / "w1_evaluation" / output.name
        old_report = read(previous / "numeric_report.json")
        return read(output / "evaluation_plan.json") == _evaluation_plan(plan, lock, slot, previous, old_report, output.name)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


def evaluate(root, cohort, *, resume=False):
    """Score the frozen new ANIE once and carry the six old measurements forward."""
    root = Path(root).resolve()
    plan = read(root / "RUN_PLAN.json")
    lock = read(root / "PROTOCOL_LOCK.json")
    slot = read(root / "FORMAL.json")
    previous = Path(plan["old_root"]) / "w1_evaluation" / cohort
    old_report = read(previous / "numeric_report.json")
    old_plan = read(previous / "evaluation_plan.json")
    output = root / "w1_evaluation" / cohort
    window = shared.observation_window(old_plan["data_file"], old_plan["group"], cohort,
                                       query_times=native.QUERY_TIMES)
    if len(window.trajectory_ids) != 180 or old_report["trajectory_ids"] != window.trajectory_ids.tolist():
        raise ValueError("coordinate refresh requires the exact previous180-ID cohort")
    np.testing.assert_allclose(old_report["query_times"], native.QUERY_TIMES, rtol=0, atol=1e-12)
    checkpoint = Path(slot["checkpoint"]) if slot["status"] == "complete" else None
    evaluation_plan = _evaluation_plan(plan, lock, slot, previous, old_report, cohort)
    if output.exists():
        if read(output / "evaluation_plan.json") != evaluation_plan:
            raise ValueError("resume requires the same model, protocol and full cohort")
        if complete_evaluation(output):
            return read(output / "numeric_report.json")
        if not resume:
            raise ValueError("an incomplete evaluation requires resume")
    else:
        (output / "raw").mkdir(parents=True)
        (output / "metrics").mkdir()
        shared.write_json(output / "evaluation_plan.json", evaluation_plan)
    extended._validate_imported_truth(window, previous / "raw/truth.h5")
    extended._link_immutable(previous / "raw/truth.h5", output / "raw/truth.h5")
    records = {}
    for label, original in old_report["methods"].items():
        if label == "anie_ns":
            continue
        record = copy.deepcopy(original)
        for field, digest_field in (("raw_prediction_file", "raw_prediction_sha256"), ("metric_file", "metric_sha256")):
            if field in record:
                source = previous / record[field]
                if shared.file_sha256(source) != record[digest_field]:
                    raise ValueError("a frozen measurement changed: " + label)
                destination = output / record[field]
                extended._link_immutable(source, destination)
        record.update(reused_measurement_from=str(previous), new_inference_performed=False)
        record_path = output / "metrics" / (label + ".json")
        if record_path.exists():
            if read(record_path) != shared._json_ready(record):
                raise ValueError("cached unchanged method differs: " + label)
        else:
            shared.write_json(record_path, record)
        records[label] = record
    if checkpoint is None:
        failure = read(slot["failure"])
        if failure.get("failure_kind") not in ("numerical", "resource"):
            raise ValueError("infrastructure interruption is not a scientific failure result")
        record = dict(method="anie_ns", seed=7, status="training_failed", reason=failure["reason"],
                      comparison_role="common_N300", source_profile=plan["source_profile"],
                      summary=dict(status="training_failed", population=180, failed_count=180,
                                   primary_metric=None, metrics={}, failed_ids=window.trajectory_ids.tolist()))
    else:
        payload = native.checkpoint_metadata(checkpoint, lock)
        if payload["model_spec"]["source_profile"] != plan["source_profile"]:
            raise ValueError("the new ANIE must use the trained aligned coordinate profile")
        raw_path, metric_path = output / "raw/anie_ns.h5", output / "metrics/anie_ns.h5"
        previous_raw = None
        if raw_path.is_file():
            raw_status, cost = shared.raw_checkpoint_status(raw_path, window, checkpoint, payload)
            if raw_status != "complete":
                previous_raw = shared.preserve_interrupted(raw_path)
        if not raw_path.is_file():
            cost = native.predict_checkpoint(checkpoint, payload, window, raw_path, device=plan["device"],
                                             batch_size=plan["evaluation_batch"], previous_raw=previous_raw)
        cached = extended._metric_record(output / "metrics/anie_ns.json", metric_path,
                                         shared.file_sha256(checkpoint), shared.file_sha256(raw_path))
        if cached is not None:
            record = cached
        else:
            for incomplete in (metric_path, output / "metrics/anie_ns.json"):
                if incomplete.exists():
                    shared.preserve_interrupted(incomplete)
            with h5py.File(output / "raw/truth.h5", "r") as f:
                truth = f["vorticity"][:]
            measured = shared.score_raw_predictions(raw_path, truth, metric_path)
            record = dict(method="anie_ns", seed=7, status=measured["summary"]["status"],
                          checkpoint=str(checkpoint), checkpoint_sha256=shared.file_sha256(checkpoint),
                          selected_epoch=payload["epoch"], model_spec=payload["model_spec"], config=payload["config"],
                          data_binding=payload["data_binding"], summary=measured["summary"], per_trajectory=measured["per_trajectory"],
                          cost=cost, new_inference_performed=True, comparison_role="common_N300",
                          raw_prediction_sha256=shared.file_sha256(raw_path), metric_sha256=shared.file_sha256(metric_path),
                          raw_prediction_file="raw/anie_ns.h5", metric_file="metrics/anie_ns.h5",
                          source_provenance=payload["source_provenance"])
    anie_record = output / "metrics/anie_ns.json"
    if anie_record.exists():
        if read(anie_record) != shared._json_ready(record):
            raise ValueError("ANIE metric record differs on resume")
    else:
        shared.write_json(anie_record, record)
    records["anie_ns"] = record
    records = {label: records[label] for label in old_report["methods"]}
    report = copy.deepcopy(old_report)
    # The former extension lock describes the source study, including its old
    # ANIE. Keep that provenance explicitly historical rather than presenting
    # it as the active checkpoint contract for the newly trained model.
    if "extension_lock" in report:
        report["prior_extension_lock"] = report.pop("extension_lock")
    report.update(methods=records, base_lock=lock, job_events_file=plan["job_events_file"],
                  interpretation="posthoc_same_heldout_population_coordinate_corrected_ANIE",
                  refresh_source=str(previous),
                  limitations=["One prescribed training initialization per method; no training-seed SD.",
                               "Only ANIE is freshly trained with aligned space/time tokens; other six measurements are immutable.",
                               "Both held-out cohorts have already been scored; this refresh is not a new blind test.",
                               "Decoder, loss, physical query grid and training control remain the declared original NS configuration.",
                               "GIFT-N50 reduces both data and update budget; training targets and budgets differ across methods."])
    report_path = output / "numeric_report.json"
    if report_path.exists():
        if read(report_path) != shared._json_ready(report):
            shared.preserve_interrupted(report_path)
    if not report_path.exists():
        shared.write_json(report_path, report)
    files = ["numeric_report.json", "evaluation_plan.json", "raw/truth.h5"]
    for label, value in records.items():
        files.append("metrics/" + label + ".json")
        files.extend(value[key] for key in ("raw_prediction_file", "metric_file") if key in value)
    marker = output / "EVALUATION_COMPLETE.json"
    if marker.exists():
        shared.preserve_interrupted(marker)
    shared.write_json(marker, dict(status="complete",
                      file_sha256={name: shared.file_sha256(output / name) for name in files}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", choices=("confirmation", "old_test"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.root, args.cohort, resume=args.resume)
    print(json.dumps(dict(status=report["status"], cohort=report["cohort"],
                         ANIE=report["methods"]["anie_ns"]["summary"]), allow_nan=False))


if __name__ == "__main__":
    main()
