"""Run the fixed W3 parameter-identification protocol at 0%, 1% and 5% noise.

Every generator and native PINN instance starts from its declared initialization.
Only this run's journals may resume. Paired observations, sensor designs, costs,
failed slots and parameter measurements remain separate immutable evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from scripts.run_w1_single import atomic_json, fixed_json, file_hash, queue_lease, read_json, status
from scripts.run_training import child_environment

CONDITIONS = {"noise_000": 0.0, "noise_001": 0.01, "noise_005": 0.05}
METHODS = {"GIFT": "gift", "PDE-FIND": "pde_find", "PDE-FIND-KC": "pde_find_kc",
           "PINN-SR": "pinn_sr", "PINN-SR-KC": "pinn_sr_kc"}
PINN_BUDGET = {"pre_nadam": 5000, "rounds": 6, "ado_nadam": 1000,
               "post_nadam": 20000, "lbf_maxiter": 10000, "lbf_maxfun": 10000}


def now():
    """UTC process clock; physical observation times are stored separately."""
    return datetime.now(timezone.utc).isoformat()


def default_plan(directory):
    """Declare the whole scientific workload before generating observations or scores."""
    directory = Path(directory).resolve()
    return dict(schema="gift.w3-parameters.queue.v1", experiment="W3", root=str(directory),
        project_root=str(ROOT), source_data_root=r"D:\GIFT\GIFT-dataset",
        source_clean=r"D:\GIFT\GIFT-dataset\standard_ns_n64_full_spectrum.h5",
        external_source_root=r"D:\GIFT\release_work_20260910\external",
        external_adapter_root=r"D:\2_GIFT\gift-external-runtime",
        torch_python=r"D:\2_GIFT\anie-env\Scripts\python.exe",
        pinn_python=r"D:\GIFT\repro_audit_20260904_from_scratch\envs\pinnsr_py38\python.exe",
        data_root=str(directory / "data"), conditions=CONDITIONS, methods=METHODS,
        training_trajectory_id=0, observation_grid=64, observation_dt=.02,
        observation_times=[0., 10.], observation_points=501,
        generator=dict(seed=2026072301, training_trajectories=1, validation_trajectories=20,
            phase_steps=[6000, 6000], phase_learning_rates=[.01, .003], batch_size=16,
            cutoff=21, rank=8, affine_refit_interval=500, validation_interval=250,
            checkpoint_interval=250, checkpoint_policy="native_validation_best_within_each_fixed_phase_then_final_affine_fit"),
        pinn=dict(seed=1234, budget=PINN_BUDGET, train_rows=24000, physics_rows=84000,
            sensors=500, time_samples=60, lhs_points=60000, full_population=True,
            pre_lr=.001, later_lr=.0001, checkpoint_interval=1000, checkpoint_policy="terminal"),
        noise=dict(seed=0, rng="MT19937", scaling="std(clean_group,ddof=0)",
            groups=["training", "validation"], frame_chunk=16, paired_noise_direction=True),
        fresh_training_slots=9, fresh_measurement_slots=15, parameter_rows=45,
        parameter_search=False, repeated_seed_runs=False, old_weights_reused=False,
        scientific_failures_preserved=True, device="cuda")


def append_event(directory, event):
    """Append and fsync an attempt event; earlier starts and interruptions are never replaced."""
    with (Path(directory) / "JOB_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_inventory():
    """Read Windows command lines without signalling, stopping or changing any process."""
    if os.name != "nt":
        return []
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(?:w)?\\.exe$' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"],
        capture_output=True, text=True, encoding="utf-8", check=True)
    records = json.loads(result.stdout or "[]")
    return [records] if isinstance(records, dict) else records


def matching_children(records, directory, old=None):
    """Match the actual run trainer and its wrapper chain, excluding unrelated Python tasks."""
    root_text = str(Path(directory).resolve()).casefold()
    old = old or {}
    pid = old.get("child_pid")
    expected = old.get("command", [])
    selected = []
    for item in records:
        command = (item.get("CommandLine") or "").casefold()
        same_wrapper = item.get("ProcessId") == pid and expected and all(
            str(token).casefold() in command for token in expected[2:])
        same_trainer = root_text in command and any(token in command for token in (
            "training.train_pinn", "experiments.formal.train_low_generator",
            "scripts.generate_noise", "scripts.generate_sampling", "experiments.formal.w3_equation_identification"))
        if item.get("ProcessId") != os.getpid() and (same_wrapper or same_trainer):
            selected.append(item)
    return selected


def wait_surviving_children(directory):
    """Adopt a verified surviving wrapper/trainer before any same-run continuation starts."""
    previous = read_json(Path(directory) / "ORCHESTRATOR.json", {})
    selected = matching_children(process_inventory(), directory, previous)
    if not selected:
        return
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    status(directory, "running", active_job=previous.get("active_job"),
        child_pid=previous.get("child_pid"), command=previous.get("command"),
        log=previous.get("log"), verified_process_chain=selected, adopted_surviving_child=True)
    for item in selected:
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, item["ProcessId"])
        if handle:
            try:
                while kernel.WaitForSingleObject(handle, 30000) == 0x102:
                    pass
            finally:
                kernel.CloseHandle(handle)


def close_interrupted_events(directory):
    """Account for unclosed parent attempts after verified children have exited.

    Their precise exit time is unknown. Record an observed duration upper bound
    separately from completed process measurements instead of inventing a cost.
    """
    path = Path(directory) / "JOB_EVENTS.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    closed = {item["attempt_id"] for item in events if item["event"] in ("finish", "interrupted")}
    for item in events:
        if item["event"] == "start" and item["attempt_id"] not in closed:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(item["started_utc"])).total_seconds()
            append_event(directory, dict(event="interrupted", attempt_id=item["attempt_id"],
                label=item["label"], command=item["command"], observed_utc=now(),
                elapsed_upper_bound_seconds=max(0., elapsed), process_seconds=None,
                reason="parent event did not close; no matching child survives; exact exit time unavailable"))


def run_module(directory, plan, module, arguments, label, *, pinn=False, cpu=False):
    """Run one isolated command and record every attempt without a shell or visible window."""
    directory = Path(directory)
    python = plan["pinn_python"] if pinn else plan["torch_python"]
    environment = child_environment("pinn" if pinn else "gift_low", python, "cpu" if cpu else "cuda")
    environment.update(GIFT_DATA_ROOT=plan["data_root"],
        GIFT_EXTERNAL_ROOT=plan["external_source_root"], GIFT_EXTERNAL_ADAPTER_ROOT=plan["external_adapter_root"],
        PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [python, "-u", "-B", "-m", module, *map(str, arguments)]
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / (label + ".log")
    attempt = uuid.uuid4().hex
    began, started = now(), time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        append_event(directory, dict(event="start", attempt_id=attempt, label=label, command=command,
            started_utc=began, parent_pid=os.getpid(), child_pid=child.pid, cpu_only=cpu))
        status(directory, "running", active_job=label, child_pid=child.pid, command=command,
            log=str(log_path), started_utc=began, attempt_id=attempt, cpu_only=cpu)
        code = child.wait()
    event = dict(event="finish", attempt_id=attempt, label=label, command=command,
        started_utc=began, finished_utc=now(), exit_code=code,
        process_seconds=time.perf_counter() - started, cpu_only=cpu)
    append_event(directory, event)
    status(directory, "between_jobs", active_job=None, child_pid=None, last_job=label, last_exit_code=code)
    if code:
        text = log_path.read_text(encoding="utf-8", errors="replace")[-18000:].casefold()
        scientific = any(value in text for value in ("floatingpointerror", "nonfinite pinn state", "metric became nonfinite",
            "terminal arrays must be safe finite", "invalid sparse regression output", "requires exactly three finite reported parameters",
            "out of range float values", "out of memory", "resourceexhaustederror"))
        return dict(status="failed" if scientific else "interrupted", failure_kind="scientific" if scientific else "infrastructure",
            event=event, log=str(log_path))
    return dict(status="complete", event=event, log=str(log_path))


def require_command(result):
    """Stop on a specific preparation/infrastructure error; retain its log and attempt cost."""
    if result["status"] != "complete":
        raise RuntimeError("command did not finish; inspect preserved evidence: " + result["log"])


def copy_bound(source, destination):
    """Copy observations into a new package, verifying bytes without changing the source."""
    source, destination = Path(source).resolve(strict=True), Path(destination)
    checksum = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != checksum:
            raise ValueError("existing copied input differs: " + str(destination))
    else:
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".copying")
        shutil.copyfile(source, temporary)
        if file_hash(temporary) != checksum:
            raise ValueError("copied input checksum differs; preserve partial evidence")
        temporary.rename(destination)
    return dict(path=str(destination), bytes=destination.stat().st_size, sha256=checksum)


def prepare_data(directory, plan):
    """Generate full paired 1%/5% fields and three identical seeded sensor designs on CPU."""
    directory = Path(directory)
    receipt_path = directory / "DATA_RECEIPT.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for item in receipt["files"].values():
            if file_hash(item["path"]) != item["sha256"]:
                raise ValueError("prepared input bytes changed")
        return receipt
    data = Path(plan["data_root"])
    files = {"noise_000": copy_bound(plan["source_clean"], data / "standard_ns_n64_full_spectrum.h5")}
    noise = directory / "preparation" / "paired_noise"
    if not (noise / "COMPLETE.json").exists():
        arguments = ["--clean", plan["source_clean"], "--output", noise, "--execute"]
        if (noise / "run.json").exists():
            arguments.append("--resume")
        require_command(run_module(directory, plan, "scripts.generate_noise", arguments, "prepare_paired_noise", cpu=True))
    for condition in ("noise_001", "noise_005"):
        files[condition] = copy_bound(noise / (condition + ".h5"), data / "w3_parameter_identification" / (condition + ".h5"))
    for condition in CONDITIONS:
        attempt = directory / "preparation" / "sampling" / condition
        if not (attempt / "COMPLETE.json").exists():
            arguments = ["--input", files[condition]["path"], "--output", attempt, "--execute"]
            if (attempt / "run.json").exists():
                arguments.append("--resume")
            require_command(run_module(directory, plan, "scripts.generate_sampling", arguments, "prepare_sampling_" + condition, cpu=True))
        files[condition + "_sampling"] = copy_bound(attempt / "sampling.npz",
            data / "auxiliary" / "pinn_sampling" / (condition + "_seed1234.npz"))
    verification = verify_paired_data(plan, files, noise)
    fixed_json(directory / "DATA_EQUIVALENCE.json", verification)
    receipt = dict(schema="gift.w3-parameters.data-receipt.v1", conditions=CONDITIONS, files=files,
        clean_reintegrated=False, source_modified=False, paired_noise_complete=str(noise / "COMPLETE.json"),
        verification_sha256=file_hash(directory / "DATA_EQUIVALENCE.json"), observations_only=True)
    fixed_json(receipt_path, receipt)
    return receipt


def verify_paired_data(plan, files, noise):
    """Check every generated noise value and pair the original 1% fields exactly.

    Both new conditions use clean observations plus the same float64 Gaussian
    draw before one float32 cast. PINN sensor/time/split/LHS arrays are shared;
    original clean/1% targets are checked against the read-only released caches.
    """
    from contextlib import ExitStack
    import h5py
    import numpy as np
    from scripts.generate_sampling import sampling_design
    state = read_json(Path(noise) / "run.json")
    rng = np.random.RandomState(0)
    count = 0
    with ExitStack() as stack:
        clean = stack.enter_context(h5py.File(plan["source_clean"], "r"))
        sampling_times = np.asarray(clean["training/time"], dtype=np.float32)
        candidates = list(Path(plan["source_data_root"]).glob("*/noise_001.h5"))
        if len(candidates) != 1:
            raise ValueError("read-only reference package needs one existing 1% observation file")
        original = stack.enter_context(h5py.File(candidates[0], "r"))
        outputs = {key: stack.enter_context(h5py.File(files[key]["path"], "r")) for key in ("noise_001", "noise_005")}
        for group in ("training", "validation"):
            field = clean[group + "/vorticity"]
            sigma = state["statistics"][group]["std_ddof0"]
            for row in range(field.shape[0]):
                for start in range(0, field.shape[1], 16):
                    values = np.asarray(field[row, start:start + 16], dtype=np.float32)
                    z = rng.standard_normal(values.shape)
                    for key, target in outputs.items():
                        expected = (values.astype(np.float64) + CONDITIONS[key] * sigma * z).astype(np.float32)
                        actual = np.asarray(target[group + "/vorticity"][row, start:start + 16])
                        if not np.array_equal(expected, actual):
                            raise ValueError("paired noise values differ: " + key)
                        if key == "noise_001" and not np.array_equal(actual, original[group + "/vorticity"][row, start:start + 16]):
                            raise ValueError("fresh 1% observations differ from original values")
                    count += values.size
    design_fields = ["measurement_coordinates", "train_indices", "validation_indices", "lhs_coordinates",
                     "sensor_flat", "time_indices", "lower", "upper", "trajectory_id"]
    reference = None
    for condition in CONDITIONS:
        with np.load(files[condition + "_sampling"]["path"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if reference is None:
            reference = arrays
            expected = sampling_design(sampling_times, 64, np.random.RandomState(1234))
            if any(not np.array_equal(arrays[key], expected[key]) for key in design_fields):
                raise ValueError("native seed1234 sampling design differs")
        elif any(not np.array_equal(arrays[key], reference[key]) for key in design_fields):
            raise ValueError("noise conditions do not share the same sensor design")
        if condition != "noise_005":
            original_path = Path(plan["source_data_root"]) / "auxiliary/pinn_sampling" / (condition + "_seed1234.npz")
            with np.load(original_path, allow_pickle=False) as original:
                if set(arrays) != set(original.files) or any(not np.array_equal(arrays[key], original[key]) for key in arrays):
                    raise ValueError("fresh clean/1% sampling values differ from original cache")
    return dict(status="passed", noise_formula="clean + fraction * std(clean_group,ddof=0) * shared_Z",
        paired_conditions=["noise_001", "noise_005"], all_generated_noise_values_checked=True,
        clean_scalar_values_per_condition=count, original_1percent_all_arrays_bitwise_equal=True,
        sampling_design_same_across_conditions=True, original_clean_1percent_sampling_bitwise_equal=True,
        five_percent_derived_from_clean=True, labels_alone_changed=False, gpu_used=False)


def prepare_run(directory, plan):
    """Create one locked fixed-budget experiment; preparation and training remain explicit actions."""
    directory = Path(directory)
    fixed_json(directory / "RUN_PLAN.json", plan)
    fixed_json(directory / "COST_FORECAST.json", dict(schema="gift.w3-parameters.cost-forecast.v1",
        full_delivery_hours=[8., 11.], estimates_are_not_promises=True,
        historical_generator_seconds=[102.656307, 104.023722, 103.273881],
        historical_pinn_seconds={"known": [3666.94193, 5542.48930, 3525.99037],
                                 "open": [5043.26710, 5645.00780, 4959.81270]},
        historical_pde_seconds={"open": [244.70659, 242.83827, 243.38909], "known": [3.39963, 3.62569, 3.86100]},
        timing_scope="native committed scientific phases; final process ledger records setup/IO and interrupted attempts separately",
        basis="same original W3 shapes, seeds and full budgets on this machine; no added GPU probe"))
    lock = dict(schema="gift.w3-parameters.protocol-lock.v1", experiment="W3", protocol_locked=True,
        configurations_locked=True, plan=plan, input_weights=[], complete_training_from_scratch=True,
        data_file_names={condition: "standard_ns_n64_full_spectrum.h5" if condition == "noise_000" else
            "w3_parameter_identification/" + condition + ".h5" for condition in CONDITIONS},
        readout=dict(GIFT="frozen P21 generator component projection on ID0, t5..6/.1 (11 states)",
            PINN_SR="terminal learned coefficients; known 4/open90 library readout",
            PDE_FIND="245760 observation-derived rows; native open90/known4 STRidge"),
        noise_005_fraction=.05, no_test_parameter_selection=True)
    fixed_json(directory / "PROTOCOL_LOCK.json", lock)
    source_registry = read_json(ROOT / "external_sources.json")["sources"]
    bindings = {}
    for family in ("pde_find", "pinn_sr", "tensorflow115"):
        source = source_registry[family]
        files = {}
        for relative in source["files"]:
            path = Path(plan["external_source_root"]) / source["directory"] / relative
            files[relative] = dict(path=str(path), sha256=file_hash(path))
        bindings[family] = dict(repository=source["repository"], commit=source["commit"], files=files)
    source_record = directory / "SOURCE_BINDINGS.json"
    if not source_record.exists():
        fixed_json(source_record, dict(schema="gift.w3-parameters.sources.v1", external_sources=bindings,
            repository_sources={name: file_hash(ROOT / name) for name in (
                "scripts/generate_noise.py", "scripts/generate_sampling.py", "training/train_pinn.py",
                "training/gift_data.py", "experiments/formal/train_low_generator.py",
                "experiments/formal/_shared/gift_generator_training.py",
                "experiments/formal/w3_equation_identification/run.py",
                "experiments/formal/w3_equation_identification/aggregate.py",
                "experiments/formal/w3_equation_identification/plot_parameters.py")},
            external_interface_sources={name: file_hash(Path(plan["external_adapter_root"]) / "adapters" / name)
                for name in ("pde_find.py", "pinn_runtime.py", "pinn_reference.py", "pinn_stridge.py")},
            source_hashes_are_provenance=True, official_sources_modified=False))
    protocol = directory / "PROTOCOL_LOCKED.md"
    if not protocol.exists():
        protocol.write_text("""# W3 方程参数识别协议

五配置 GIFT、PDE-FIND、PDE-FIND-KC、PINN-SR、PINN-SR-KC，三噪声 0%、1%、5%，共15个独立参数识别槽位，每槽位读取三个参数。
从同一未改变的干净N64观测生成1%/5%数据，RandomState(0)、整组std(ddof=0)、共享Gaussian方向；每组最后转换float32。
共同训练轨迹为本地ID0，t0–10/.02/501点。PINN用固定seed1234的500传感器、60时刻、24000训练观测与84000物理点；三条件采样设计相同。
GIFT三个生成元从头初始化seed2026072301，6000+6000更新、batch16、lr.01/.003、原相位cosine与affine refit；保留每固定相位原validation-best再最终affine-fit，不引入候选或超参数搜索。
PINN六实例均原Xavier seed1234、完整人口；5000 NAdam(lr.001)+原L-BFGS上限10000+六次STRidge各1000 NAdam(lr.0001)+20000后训练；冻结原精确非零mask，终态识别，无最终L-BFGS。
PDE-FIND保留原90/4列字典与固定ridge参数；噪声预处理保持固定26/20/20阶SVD、原时间/空间多项式窗口，不按5%结果重新选秩或阈值。
GIFT读出在同ID0的t5–6/.1十一状态上投影；PINN读取原终态物理系数。三参数nu=.01、beta=1、gamma=1仅用于计算读出误差，不用于选参或训练GIFT。
所有9个训练实例和15个参数测量均使用新输出目录，不复用旧权重，不增加seed。失败保留唯一槽位；基础设施中断只允许从本槽位全状态恢复，不改变科学配方。
外部算法源码和接口仍在GIFT仓库外；本次仅外部条件标识支持noise_005，官方数值源码保持不变。图保持原三参数面板、颜色、线型、字体与比例，只替换观测噪声节点和新测量值。
""", encoding="utf-8")
    return lock


def scientific_failure(directory, output, label, result):
    """Retain a numerical/resource outcome under its unique slot without another initialization."""
    record = dict(status="failed", failure_kind="scientific", label=label, evidence=result,
        retry_permitted=False, seed_unchanged=True)
    fixed_json(Path(output) / "FAILURE.json", record)
    return dict(status="failed", label=label, failure=str(Path(output) / "FAILURE.json"))


def train_slot(directory, plan, condition, method):
    """Resume only a slot's own full journal and verify the final declared fixed budget."""
    directory = Path(directory)
    key = "gift" if method == "GIFT" else "known" if method == "PINN-SR-KC" else "open"
    output = directory / "training" / condition / key
    label = "train_" + condition + "_" + key
    failure = read_json(output / "FAILURE.json")
    if failure:
        return dict(status="failed", condition=condition, method=method, label=label,
                    seed=2026072301 if method == "GIFT" else 1234, failure=str(output / "FAILURE.json"))
    if method == "GIFT":
        model = output / ("gift_main.pt" if condition == "noise_000" else "gift_" + condition + ".pt")
        complete = output / "manifest.json"
        module = "experiments.formal.train_low_generator"
        arguments = ["--condition", condition, "--data-profile", "regenerated", "--training-trajectories", "1",
            "--output", output, "--device", "cuda", "--seed", "2026072301", "--phase-steps", "6000", "6000",
            "--phase-learning-rates", ".01", ".003", "--batch-size", "16", "--affine-refit-interval", "500",
            "--validation-interval", "250", "--checkpoint-interval", "250"]
        recoverable = output / "checkpoints/LATEST.json"
        arguments.append("--resume" if recoverable.exists() else "--run-training")
    else:
        model, complete = output / "result.json", output / "COMPLETED.json"
        module = "training.train_pinn"
        arguments = ["--execute", "--mode", key, "--device", "gpu", "--condition", condition,
            "--data-profile", "regenerated", "--output", output, "--nadam-interval", "1000"]
        recoverable = output / "LATEST.json"
        if recoverable.exists():
            arguments.append("--resume")
    if not complete.exists():
        if output.exists() and not recoverable.exists():
            raise RuntimeError("existing slot lacks its initial full-state journal: " + str(output))
        result = run_module(directory, plan, module, arguments, label, pinn=method != "GIFT")
        if result["status"] == "failed":
            failed = scientific_failure(directory, output, label, result)
            return dict(failed, condition=condition, method=method, seed=2026072301 if method == "GIFT" else 1234)
        require_command(result)
    if not complete.exists() or not model.exists():
        raise RuntimeError("child returned without a committed complete slot: " + str(output))
    if method == "GIFT":
        report = read_json(output / "training_report.json")
        cost = report["training_cost"]
        if cost["phase_updates"] != [6000, 6000] or report["optimization"]["seed"] != 2026072301:
            raise ValueError("completed generator budget/seed differs")
        seed = 2026072301
    else:
        report = read_json(model)
        cost = report["training_cost"]
        if report["requested_budget"] != PINN_BUDGET or cost["nadam_updates"] != 31000 or cost["stridge_calls"] != 6:
            raise ValueError("completed PINN budget differs")
        if report["diagnostic_test_only"]:
            raise ValueError("diagnostic PINN cannot fill a W3 slot")
        profile = report["terminal_snapshot"]["identity"]["backend"]["profile"]
        if profile["initialization"] != "native_Xavier_seed1234" or profile["tensorflow_tf32"]:
            raise ValueError("completed PINN initialization/precision differs")
        seed = 1234
    return dict(status="complete", condition=condition, method=method, label=label, seed=seed,
        checkpoint=str(model), checkpoint_sha256=file_hash(model), completion_sha256=file_hash(complete),
        output=str(output), training_cost=cost, fresh_initialization=True)


def execute(directory, plan):
    """Finish nine fresh training slots, fifteen readouts, numerical aggregation and original-style SVG."""
    directory = Path(directory)
    prepare_data(directory, plan)
    slots = []
    # Serial GPU ownership keeps native TensorFlow and Torch memory independent.
    for condition in CONDITIONS:
        for method in ("GIFT", "PINN-SR-KC", "PINN-SR"):
            slots.append(train_slot(directory, plan, condition, method))
            atomic_json(directory / "FORMAL.json", slots)
    previous_measurements = {(item["condition"], item["method"]): item
        for item in read_json(directory / "MEASUREMENTS.json", [])}
    measurements = []
    for condition in CONDITIONS:
        for method in METHODS:
            previous = previous_measurements.get((condition, method))
            if previous and previous["status"] == "failed":
                measurements.append(previous)
                continue
            selected = next((item for item in slots if item["condition"] == condition and item["method"] == method), None)
            if selected and selected["status"] != "complete":
                measurements.append(dict(condition=condition, method=method, status="failed", failure=selected["failure"]))
                continue
            path = directory / "jobs" / condition / METHODS[method]
            if not (path / "COMPLETE.json").exists():
                if path.with_name(path.name + ".partial").exists():
                    raise RuntimeError("partial parameter measurement preserved; resolve concrete interruption before retry: " + str(path))
                arguments = ["--execute", "--method", method, "--condition", condition,
                    "--output", directory / "jobs", "--device", "cpu"]
                if selected:
                    arguments.extend(["--checkpoint", selected["checkpoint"], "--checkpoint-sha256", selected["checkpoint_sha256"]])
                result = run_module(directory, plan, "experiments.formal.w3_equation_identification.run", arguments,
                    "measure_" + condition + "_" + METHODS[method], cpu=True)
                if result["status"] == "failed":
                    measurements.append(dict(condition=condition, method=method, status="failed", evidence=result))
                    continue
                require_command(result)
            from experiments.formal.w3_equation_identification.run import read_completed
            read_completed(path, method, condition)
            measurements.append(dict(condition=condition, method=method, status="complete", job=str(path),
                result_sha256=file_hash(path / "result.json"), completion_sha256=file_hash(path / "COMPLETE.json")))
            atomic_json(directory / "MEASUREMENTS.json", measurements)
    atomic_json(directory / "MEASUREMENTS.json", measurements)
    failed = [item for item in measurements if item["status"] != "complete"]
    if failed:
        status(directory, "scientific_failures_documents_pending", child_pid=None, active_job=None,
            training_slots=9, measurement_slots=15, failures=failed,
            reason="all unique slots retained; a finite 45-row table cannot replace failed coefficients")
        return
    aggregate = directory / "aggregate"
    if not (aggregate / "numeric_report.json").exists():
        require_command(run_module(directory, plan, "experiments.formal.w3_equation_identification.aggregate",
            ["--jobs", directory / "jobs", "--output", aggregate], "aggregate_and_plot", cpu=True))
    if not (aggregate / "figures/figure_manifest.json").exists():
        require_command(run_module(directory, plan, "experiments.formal.w3_equation_identification.plot_parameters",
            ["--result-dir", aggregate, "--output-dir", aggregate / "figures"], "plot_parameters", cpu=True))
    report = read_json(aggregate / "numeric_report.json")
    if report["completed_jobs"] != 15 or report["parameter_rows"] != 45:
        raise ValueError("W3 complete numerical population differs")
    fixed_json(directory / "TRAINING_AUDIT.json", dict(status="passed", unique_training_slots=9,
        methods_per_condition=["GIFT", "PINN-SR-KC", "PINN-SR"], conditions=CONDITIONS,
        fresh_initialization=True, parameter_search=False, repeated_seed_runs=False, checkpoints=slots,
        data_receipt_sha256=file_hash(directory / "DATA_RECEIPT.json")))
    status(directory, "documents_pending", active_job=None, child_pid=None, training_slots=9,
        measurement_slots=15, parameter_rows=45, numeric_report=str(aggregate / "numeric_report.json"),
        figure=str(aggregate / "figures/parameter_identification_ape_vs_noise.svg"),
        final_publication_and_document_audit_pending=True)


def main(argv=None):
    """Plan without training; --prepare is CPU-only and --run executes the locked full queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    directory = args.root.resolve()
    if directory == ROOT or ROOT in directory.parents:
        raise ValueError("W3 runs and observations must stay outside the GIFT repository")
    directory.mkdir(parents=True, exist_ok=True)
    plan = default_plan(directory)
    for protected in (Path(plan["source_data_root"]), Path(plan["external_source_root"]),
                      Path(plan["external_adapter_root"])):
        protected = protected.resolve()
        if directory == protected or protected in directory.parents or directory in protected.parents:
            raise ValueError("W3 output must not contain or modify an external input directory")
    with queue_lease(directory):
        wait_surviving_children(directory)
        close_interrupted_events(directory)
        prepare_run(directory, plan)
        try:
            if args.run:
                execute(directory, plan)
            elif args.prepare:
                prepare_data(directory, plan)
                status(directory, "prepared_not_started", child_pid=None, active_job=None, gpu_training_started=False)
            else:
                status(directory, "planned_not_started", child_pid=None, active_job=None, gpu_training_started=False)
            print(json.dumps(read_json(directory / "ORCHESTRATOR.json"), ensure_ascii=False, indent=2))
        except Exception as error:
            status(directory, "needs_attention", reason=repr(error), child_pid=None, active_job=None)
            raise


if __name__ == "__main__":
    main()
