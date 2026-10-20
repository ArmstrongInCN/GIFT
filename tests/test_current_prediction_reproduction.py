"""Current-protocol preparation must need no historical run or cost archive."""
import json
import os
from pathlib import Path

import pytest

from scripts import reproduce_prediction as current


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    """Use synthetic routing/bindings; numerical window validation has separate tests."""
    if not os.environ.get("GIFT_EXTERNAL_ADAPTER_ROOT"):
        pytest.skip("current seven-method recipe planning needs the prepared external interface")
    from training import w1_v2_data
    source, anie, data = (tmp_path / name for name in ("upstreams", "anie", "inputs"))
    for path in (source, anie, data):
        path.mkdir()
    roles = {"w1_w2.train300": "300.h5", "w1_w2.train50": "50.h5"}
    (data / "dataset.json").write_text(json.dumps({"schema": "gift.dataset.v1", "roles": roles}))
    for name in roles.values():
        (data / name).write_bytes(b"routing fixture, not numerical observations")
    class Bank:
        def __init__(self, path):
            count = 300 if Path(path).name == "300.h5" else 50
            self.future_steps, self.time_stride = 75, 2
            self.training_ids = list(range(count))
            self.binding = dict(training_count=count, subset_seed=2026100304,
                training={"ids": self.training_ids, "observations_sha256": "a" * 64},
                validation={"ids": list(range(1000, 1020)), "observations_sha256": "b" * 64},
                query_times=[i * .04 for i in range(76)])
    monkeypatch.setattr(w1_v2_data, "WindowBank", Bank)
    return tmp_path / "fresh", data, source, anie


def test_prepare_locks_all_current_recipes_and_preserves_inputs(preparation):
    """N300/N50 and aligned ANIE are prepared without any old source registry."""
    root, data, source, anie = preparation
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    plan = current.prepare(root, data, source_root=source, anie_root=anie, device="cpu")
    assert plan["ANIE_source_profile"] == "aligned_ns_single_v1"
    assert set(plan["jobs"]) == set(current.METHODS)
    assert plan["jobs"]["GIFT-N50"]["expected_training_count"] == 50
    assert plan["jobs"]["uno"]["config"]["training_future_steps"] == 20
    assert plan["jobs"]["unet"]["config"]["training_future_steps"] == 4
    assert plan["jobs"]["anie_ns"]["config"]["selection"] == "validation_mse_best"
    assert not (root / "training").exists()
    from experiments.additional.w1_single_evaluate import protocol_lock
    assert protocol_lock(root / "PROTOCOL_LOCK.json")["lock_snapshot"]["training_count"] == 300
    assert before == {p.name: p.read_bytes() for p in data.iterdir()}
    assert current.prepare(root, data, source_root=source, anie_root=anie, device="cpu") == plan
    with pytest.raises(ValueError, match="configuration differs"):
        current.prepare(root, data, source_root=source, anie_root=anie, device="cuda")


def test_training_requires_explicit_execution_and_never_overwrites_a_slot(preparation):
    """A complete dry plan neither creates a trainer output nor invokes optimization."""
    root, data, source, anie = preparation
    plan = current.prepare(root, data, source_root=source, anie_root=anie, device="cpu")
    result = current.train(root, "unet")
    assert result["status"] == "planned_not_trained"
    assert "training.w1_extended_baselines" in result["command"]
    output = Path(plan["jobs"]["unet"]["output"])
    assert not output.exists()
    output.mkdir(parents=True)
    (output / "saved.pt").write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        current.train(root, "unet", execute=True)
    assert (output / "saved.pt").read_bytes() == b"preserved"


def test_preparation_rejects_repository_and_input_output_overlap(preparation):
    """No source-only command may create runtime state in an observation package."""
    _, data, source, anie = preparation
    with pytest.raises(ValueError, match="separate"):
        current.prepare(data / "run", data, source_root=source, anie_root=anie)
