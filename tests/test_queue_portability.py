"""Queue source selection and child launch contracts without training or GPU use."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import queue_runtime, run_anie_coordinate_refresh, run_w1_single, run_w1_v2, run_w1_w2_single


@pytest.mark.parametrize("module", (run_w1_single, run_w1_v2))
def test_queue_children_preserve_user_source_device_and_thread_configuration(tmp_path, monkeypatch, module):
    """The real launch wrapper must pass configured paths and device restrictions to Popen."""
    configured = {"GIFT_EXTERNAL_ROOT": str(tmp_path / "upstreams"),
                  "GIFT_ANIE_SOURCE_ROOT": str(tmp_path / "anie"),
                  "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2",
                  "PYTHONPATH": str(tmp_path / "user_modules")}
    for key, value in configured.items():
        monkeypatch.setenv(key, value)
    launches = []
    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(pid=123, wait=lambda: 0)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    assert module.call_module(tmp_path, "fixture.no_execution", ["--flag", "a b"], "capture") == 0
    command, options = launches[0]
    assert command[-2:] == ["--flag", "a b"]
    for key, value in configured.items():
        if key != "PYTHONPATH":
            assert options["env"][key] == value
    assert options["env"]["PYTHONPATH"].split(os.pathsep) == [str(module.ROOT / "src"), configured["PYTHONPATH"]]


@pytest.mark.parametrize("module", (run_w1_single, run_w1_v2, run_w1_w2_single))
def test_new_plan_source_options_override_environment_without_machine_defaults(tmp_path, monkeypatch, module):
    """New immutable plans record actual user choices; absent inputs remain unset during planning."""
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", str(tmp_path / "environment"))
    monkeypatch.setenv("GIFT_ANIE_SOURCE_ROOT", str(tmp_path / "environment_anie"))
    explicit = tmp_path / "explicit"
    plan = module.default_plan(tmp_path / "run", fno_source=explicit)
    assert plan["fno_source"] == str(explicit)
    if module is not run_w1_w2_single:
        assert plan["anie_source"] == str(tmp_path / "environment_anie")
        assert module.default_plan(tmp_path / "run", anie_source=explicit)["anie_source"] == str(explicit)
    else:
        job = module.job_configuration(plan, "unet", [0., .04])
        assert job["model_spec"]["source_root"] == str(explicit)
    monkeypatch.delenv("GIFT_EXTERNAL_ROOT")
    assert module.default_plan(tmp_path / "run")["fno_source"] is None


def test_execution_reports_missing_sources_and_rejects_vendored_algorithm_paths(tmp_path, monkeypatch):
    """Missing sources fail before a child starts; algorithms cannot be stored in GIFT."""
    monkeypatch.delenv("GIFT_EXTERNAL_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="GIFT_EXTERNAL_ROOT"):
        queue_runtime.validate_job_source({"model_spec": {"method": "unet", "source_root": None}})
    with pytest.raises(FileNotFoundError, match="does not exist"):
        queue_runtime.validate_job_source({"model_spec": {"method": "unet", "source_root": str(tmp_path / "missing")}})
    with pytest.raises(ValueError, match="outside GIFT"):
        queue_runtime.configured_source(queue_runtime.ROOT / "upstreams", "GIFT_EXTERNAL_ROOT")


def test_queue_lease_is_exclusive_and_released_after_context_exit(tmp_path):
    """A second queue cannot own a live run; the persistent file does not block later resume."""
    with queue_runtime.queue_lease(tmp_path):
        with pytest.raises(OSError):
            with queue_runtime.queue_lease(tmp_path):
                pytest.fail("second lock must not be granted")
    with queue_runtime.queue_lease(tmp_path):
        assert (tmp_path / "queue.lock").is_file()


def test_aligned_anie_new_job_relocates_source_without_mutating_historical_lock(tmp_path, monkeypatch):
    """The aligned profile retains the fixed recipe while an explicit new source path wins."""
    old = dict(profile="single_native76_n300", data_binding={"observations": "fixture"},
               training_configurations={"anie_ns": dict(method="anie_ns", window="fixed.h5",
                   output="historical", model_spec=dict(method="anie_ns", source_root="historical_source",
                                                        source_profile="native_ns_single_v1"))})
    original = copy.deepcopy(old)
    base = tmp_path / "historical"
    run_w1_single.atomic_json(base / "PROTOCOL_LOCK.json", old)
    historical_bytes = (base / "PROTOCOL_LOCK.json").read_bytes()
    monkeypatch.setattr(run_anie_coordinate_refresh, "BASE", base)
    root = tmp_path / "new_run"
    run_w1_single.atomic_json(root / "COST_FORECAST.json", {"fixture": True})
    monkeypatch.setenv("GIFT_ANIE_SOURCE_ROOT", str(tmp_path / "environment_anie"))
    source = tmp_path / "explicit_anie"
    plan = run_anie_coordinate_refresh.prepare(root, anie_source=source)
    assert plan["training_job"]["model_spec"]["source_root"] == str(source)
    assert plan["training_job"]["model_spec"]["source_profile"] == "aligned_ns_single_v1"
    assert plan["old_native_job"] == original["training_configurations"]["anie_ns"]
    assert (base / "PROTOCOL_LOCK.json").read_bytes() == historical_bytes


@pytest.mark.parametrize("module", (run_w1_single, run_w1_v2, run_w1_w2_single))
def test_historical_cli_uses_its_own_locked_plan_without_new_default_fields(tmp_path, monkeypatch, module):
    """New runtime source settings cannot rewrite an existing scientific plan on resume."""
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", os.environ.get("GIFT_EXTERNAL_ROOT", ""))
    plan = module.default_plan(tmp_path)
    plan["fno_source"] = "historical_external_source"
    if module is run_w1_w2_single:
        plan.pop("fno_source")  # This field did not exist in extension queues.
    run_w1_single.atomic_json(tmp_path / "RUN_PLAN.json", plan)
    original = (tmp_path / "RUN_PLAN.json").read_bytes()
    received = []
    if module is not run_w1_v2:
        monkeypatch.setattr(module, "wait_surviving_child", lambda root: None)
        def prepared(root, current):
            received.append(current)
            return ({}, {}) if module is run_w1_w2_single else {}
        monkeypatch.setattr(module, "prepare_run", prepared)
    monkeypatch.setattr(module, "default_plan", lambda *args, **kwargs: pytest.fail("stored plan must be used"))
    selected = tmp_path / "new_machine_source"
    monkeypatch.setattr(module.sys, "argv", [module.__name__, "--root", str(tmp_path), "--fno-source", str(selected)])
    module.main()
    assert (tmp_path / "RUN_PLAN.json").read_bytes() == original
    assert os.environ["GIFT_EXTERNAL_ROOT"] == str(selected)
    if received:
        assert received == [plan]


def test_existing_v2_plan_rejects_explicit_scientific_profile_change(tmp_path, monkeypatch):
    """Runtime relocation is allowed, but a different observation grid needs a new run."""
    plan = run_w1_v2.default_plan(tmp_path, "sparse76_n300")
    run_w1_single.atomic_json(tmp_path / "RUN_PLAN.json", plan)
    monkeypatch.setattr(run_w1_v2.sys, "argv", ["queue", "--root", str(tmp_path), "--profile", "dense151"])
    with pytest.raises(ValueError, match="scientific plan differs"):
        run_w1_v2.main()


def test_existing_anie_plan_and_old_extension_source_configs_remain_immutable(tmp_path, monkeypatch):
    """Archived source roots remain in locked inputs while current locations reach execution."""
    monkeypatch.setenv("GIFT_ANIE_SOURCE_ROOT", os.environ.get("GIFT_ANIE_SOURCE_ROOT", ""))
    run = tmp_path / "anie"
    plan = dict(schema="gift.anie-coordinate-refresh.plan.v1", source_profile="aligned_ns_single_v1",
                training_job={"model_spec": {"source_root": "historical"}})
    run_w1_single.atomic_json(run / "RUN_PLAN.json", plan)
    original = (run / "RUN_PLAN.json").read_bytes()
    assert run_anie_coordinate_refresh.prepare(run, anie_source=tmp_path / "relocated") == plan
    assert (run / "RUN_PLAN.json").read_bytes() == original
    legacy = run_w1_w2_single.default_plan(tmp_path / "extension")
    legacy.pop("fno_source")
    source = "historical_fixed_source"
    config = Path(legacy["root"]) / "configs/unet_seed0.json"
    run_w1_single.atomic_json(config, {"model_spec": {"source_root": source}})
    assert run_w1_w2_single.job_configuration(legacy, "unet", [0., .04])["model_spec"]["source_root"] == source
    current = tmp_path / "current_upstreams"
    current.mkdir()
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", str(current))
    queue_runtime.validate_job_source({"model_spec": {"method": "unet", "source_root": source}})
