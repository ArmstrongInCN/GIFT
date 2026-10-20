"""Prepare, train and evaluate the current seven-method W1/W2 protocol.

Downloaded observations remain read-only. New runs contain fixed jobs and their
own checkpoints; historical queues, training-cost forecasts and model files are
not needed. Each train command runs one declared method with its native budget.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from gift.paths import data_path
from scripts.run_w1_single import fixed_json, file_hash, queue_lease

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "single_native76_n300"
METHODS = ("GIFT", "fno2d", "fno3d", "anie_ns", "uno", "unet", "GIFT-N50")
SEEDS = {method: 20260820 if method.startswith("GIFT") else 7 if method == "anie_ns" else 0 for method in METHODS}
MODULES = {method: "training.w1_single_gift" if method == "GIFT" else
           "training.w1_extended_gift_n50" if method == "GIFT-N50" else
           "training.w1_extended_baselines" if method in {"uno", "unet"} else
           "training.w1_single_train" for method in METHODS}


def read(path):
    """Read a saved JSON recipe without changing it."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_root(path, *inputs):
    """Keep new artifacts out of this checkout and every supplied input root."""
    path = Path(path).expanduser().resolve()
    configured = [os.environ.get(key) for key in (
        "GIFT_DATA_ROOT", "GIFT_EXTERNAL_ROOT", "GIFT_ANIE_SOURCE_ROOT", "GIFT_EXTERNAL_ADAPTER_ROOT")]
    for protected in (ROOT, *(Path(value).expanduser().resolve() for value in (*inputs, *configured) if value)):
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError("Run/output must be separate from the checkout and input roots")
    return path


def checkpoint_paths(root):
    """Return one fixed terminal/best path per method, including both GIFT pairs."""
    root = Path(root)
    return {method: root / "training" / f"{method}_seed{SEEDS[method]}" /
            ("branch/model.pt" if method.startswith("GIFT") else "best.pt" if method == "anie_ns" else "terminal_weights.pt")
            for method in METHODS}


def prepare(root, dataset_root, *, device="cuda", source_root=None, anie_root=None):
    """Validate both train-only windows and freeze all current jobs before training."""
    from scripts.queue_runtime import configured_source
    from training.w1_v2_data import WindowBank
    from training.w1_single_train import native_training_config as native_config
    from training.w1_extended_baselines import native_training_config as extended_config
    from training.w1_single_gift import native_training_config as gift_config
    source = configured_source(source_root, "GIFT_EXTERNAL_ROOT", required=True)
    anie = configured_source(anie_root, "GIFT_ANIE_SOURCE_ROOT", required=True)
    dataset_root = Path(dataset_root).expanduser().resolve(strict=True)
    root = output_root(root, dataset_root, source, anie)
    windows = {300: data_path(dataset_root, "w1_w2.train300"), 50: data_path(dataset_root, "w1_w2.train50")}
    bindings = {}
    for count, path in windows.items():
        bank = WindowBank(path)
        bindings[count] = bank.binding
        if (bank.future_steps != 75 or bank.time_stride != 2 or len(bank.training_ids) != count
                or bank.binding.get("training_count") != count or bank.binding.get("subset_seed") != 2026100304):
            raise ValueError("Current W1/W2 requires its fixed N300/N50, 76-node windows")
        del bank
    if (bindings[50]["training"]["ids"] != bindings[300]["training"]["ids"][:50]
            or bindings[50]["validation"] != bindings[300]["validation"]):
        raise ValueError("N50 must be the ordered first50 with identical validation observations")
    jobs = {}
    for method in METHODS:
        job = dict(method=method, profile=PROFILE, stage="formal", seed=SEEDS[method],
                   window=str(windows[50 if method == "GIFT-N50" else 300]),
                   output=str(root / "training" / f"{method}_seed{SEEDS[method]}"), device=device)
        if method.startswith("GIFT"):
            job.update(generator_seed=2026072301, config=json.loads(json.dumps(gift_config())))
            if method == "GIFT-N50":
                job["expected_training_count"] = 50
        else:
            spec = dict(method=method, history=1, source_root=anie if method == "anie_ns" else source,
                        query_times=bindings[300]["query_times"], future_steps=75, observation_dt=.04)
            if method == "anie_ns":
                spec.update(condition_mode="native", source_profile="aligned_ns_single_v1")
            if method == "fno2d":
                spec["activation_checkpoint"] = False
            job.update(model_spec=spec, config=extended_config(method) if method in {"uno", "unet"} else native_config(method))
        jobs[method] = job
    plan = dict(schema="gift.current-prediction-reproduction.v1", profile=PROFILE,
                root=str(root), dataset_root=str(dataset_root), jobs=jobs,
                common_data_binding=bindings[300], n50_data_binding=bindings[50],
                methods=list(METHODS), ANIE_source_profile="aligned_ns_single_v1",
                hyperparameter_search=False, test_used_for_selection=False)
    if root.exists() and not (root / "REPRODUCTION_PLAN.json").is_file():
        raise FileExistsError("Prepare a new run; this directory has no matching reproduction plan")
    root.mkdir(parents=True, exist_ok=True)
    fixed_json(root / "REPRODUCTION_PLAN.json", plan)
    lock = dict(schema="gift.w1-single-native.protocol-lock.v1", profile=PROFILE,
                protocol_locked=True, configurations_locked=True,
                protocol_file=str(ROOT / "docs/W1_W2_PROTOCOL.md"), evaluation_dt=.04,
                evaluation_query_times=bindings[300]["query_times"], observation_points=76, future_steps=75,
                training_count=300, training_subset_seed=2026100304,
                training_configurations={method: jobs[method] for method in ("anie_ns", "fno2d", "fno3d", "GIFT")},
                data_binding=bindings[300])
    fixed_json(root / "PROTOCOL_LOCK.json", lock)
    for method, job in jobs.items():
        fixed_json(root / "configs" / f"{method}.json", job)
    return plan


def train(root, method, *, resume=False, execute=False):
    """Run one fixed recipe; a dry command prints the full invocation and budget."""
    from scripts.queue_runtime import child_environment, validate_job_source
    root = output_root(root)
    plan = read(root / "REPRODUCTION_PLAN.json")
    job = plan["jobs"][method]
    if read(root / "configs" / f"{method}.json") != job:
        raise ValueError("Training config differs from the fixed reproduction plan")
    output_root(job["output"], plan["dataset_root"])
    command = [sys.executable, "-B", "-m", MODULES[method], "--run-config", str(root / "configs" / f"{method}.json")]
    if resume:
        command.append("--resume")
    if not execute:
        return dict(status="planned_not_trained", command=command, job=job)
    failure_path = Path(job["output"]) / "FAILURE.json"
    if failure_path.is_file() and read(failure_path).get("failure_kind") != "infrastructure":
        return dict(status="scientific_failure_retained", method=method, failure=str(failure_path))
    validate_job_source(job)
    if Path(job["output"]).exists() and not resume:
        raise FileExistsError("Use --resume for this run's committed state; existing slots are never overwritten")
    with queue_lease(root):
        with (root / f"train_{method}.log").open("a", encoding="utf-8") as log:
            code = subprocess.call(command, cwd=ROOT, env=child_environment(), stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"Training failed/interrupted (exit {code}); preserve the log and failure receipt before resuming")
    return dict(status="command_finished", method=method, log=str(root / f"train_{method}.log"))


def finalize(root):
    """Validate trained slots, retain scientific failures, and freeze evaluation bindings."""
    from experiments.additional import w1_extended_evaluate as evaluation
    root = output_root(root)
    plan, base = read(root / "REPRODUCTION_PLAN.json"), read(root / "PROTOCOL_LOCK.json")
    paths = checkpoint_paths(root)
    reused, checkpoints, failures = {}, {}, {}
    for method, path in paths.items():
        failure = path.parents[1] / "FAILURE.json" if method.startswith("GIFT") else path.parent / "FAILURE.json"
        slot = path.parents[1] if method.startswith("GIFT") else path.parent
        completed = (slot / "STATUS.json").is_file() and read(slot / "STATUS.json").get("status") == "complete"
        if failure.is_file() and not completed:
            record = read(failure)
            if (record.get("failure_kind") not in {"numerical", "resource"} or not record.get("reason")
                    or record.get("method") != method or record.get("stage") != "formal"
                    or record.get("seed") != SEEDS[method]):
                raise ValueError("Resolve the interruption or invalid fixed-slot failure before finalizing " + method)
            failures[method] = str(failure)
        else:
            if not path.is_file():
                raise FileNotFoundError("Train the missing fixed slot before finalize: " + str(path))
            checkpoints[method] = str(path)
            if method in ("anie_ns", "fno2d", "fno3d", "GIFT"):
                reused[method] = dict(checkpoint=str(path), sha256=file_hash(path), source_root=str(root))
    extension = dict(schema="gift.w1w2-native.extension-lock.v1", profile=PROFILE,
                     configurations_locked=True, protocol_locked=True, base_lock=str(root / "PROTOCOL_LOCK.json"),
                     reused_models=reused, training_configurations={key: plan["jobs"][key] for key in ("uno", "unet", "GIFT-N50")},
                     common_data_binding=plan["common_data_binding"], n50_data_binding=plan["n50_data_binding"])
    for method, path in checkpoints.items():
        evaluation.checkpoint_metadata(method, path, base, extension)
    fixed_json(root / "EXTENSION_LOCK.json", extension)
    receipt = dict(schema="gift.current-prediction.models.v1", checkpoints=checkpoints, failed_runs=failures,
                   base_lock=str(root / "PROTOCOL_LOCK.json"), extension_lock=str(root / "EXTENSION_LOCK.json"))
    fixed_json(root / "MODELS.json", receipt)
    return receipt


def evaluate(root, dataset_root, experiment, cohort, output, *, device="cpu", batch_size=1, resume=False, plots=False):
    """Evaluate current frozen slots against explicit complete-population observations."""
    root, dataset_root = Path(root).resolve(strict=True), Path(dataset_root).resolve(strict=True)
    models = read(root / "MODELS.json")
    output = output_root(output, dataset_root, root, models.get("bundle_root"))
    if experiment == "W1":
        from experiments.additional.w1_extended_evaluate import evaluate as run
        path = data_path(dataset_root, "w1_w2.confirmation_n64" if cohort == "confirmation" else "four_vortex.fno_test_dt0p02")
        return run(models["checkpoints"], models["failed_runs"], base_lock=models["base_lock"],
                   extension_lock=models["extension_lock"], data_file=path,
                   group="test" if cohort == "confirmation" else "N64/test", cohort=cohort, output=output,
                   device=device, batch_size=batch_size, resume=resume, plots=plots)
    from experiments.additional.w2_single_evaluate import evaluate_checkpoints
    path = data_path(dataset_root, "w1_w2.confirmation_n96_n128" if cohort == "confirmation" else "four_vortex.fno_test_dt0p02")
    return evaluate_checkpoints({key: value for key, value in models["checkpoints"].items() if key != "anie_ns"},
        path, cohort, output, base_lock_file=models["base_lock"], extension_lock_file=models["extension_lock"],
        n64_data_file=data_path(dataset_root, "w1_w2.confirmation_n64") if cohort == "confirmation" else None,
        failed_runs={key: value for key, value in models["failed_runs"].items() if key != "anie_ns"},
        device=device, batch_size=batch_size, resume=resume, plots=plots)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset-root", type=Path, required=True)
    prep.add_argument("--source-root", type=Path)
    prep.add_argument("--anie-root", type=Path)
    prep.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    fitting = sub.add_parser("train")
    fitting.add_argument("--method", choices=METHODS, required=True)
    fitting.add_argument("--resume", action="store_true")
    fitting.add_argument("--execute", action="store_true")
    sub.add_parser("finalize")
    scoring = sub.add_parser("evaluate")
    scoring.add_argument("--dataset-root", type=Path, required=True)
    scoring.add_argument("--experiment", choices=("W1", "W2"), required=True)
    scoring.add_argument("--cohort", choices=("confirmation", "old_test"), required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    scoring.add_argument("--batch-size", type=int, default=1)
    scoring.add_argument("--resume", action="store_true")
    scoring.add_argument("--plots", action="store_true")
    for command in sub.choices.values():
        command.add_argument("--root", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    function = {"prepare": prepare, "train": train, "finalize": finalize, "evaluate": evaluate}[action]
    print(json.dumps(function(**args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
