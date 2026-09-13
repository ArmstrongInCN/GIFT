"""CPU receipt/control tests, not full-population experimental acceptance."""

import json

import numpy as np
import pytest

from experiments.formal._shared.gift_runtime import RolloutResult
from experiments.formal._shared.resume import ExperimentSession


def test_completed_call_round_trip_and_new_aggregation(tmp_path):
    """A resumed unit must not invoke the original expensive function again."""
    directory = tmp_path / "run"
    identity = {"source": "test-source", "input": "test-input", "batch": 3}
    result = RolloutResult(
        prediction=np.arange(12, dtype=np.float32).reshape(3, 4),
        finite_by_time=np.ones((3, 4), dtype=bool),
        failure_step=np.full(3, -1, dtype=np.int32),
        failure_reason=(None, None, None),
        correction_trigger_count=np.zeros((3, 4), dtype=np.int64),
        correction_step_count=np.zeros(3, dtype=np.int64),
        correction_triggered_trajectory_ids=[],
        correction={"components": {}, "nan_example": float("nan")},
        runtime={"device": "cpu"}, branch_audit={"enabled": True},
    )
    first = ExperimentSession(directory, identity)
    original = first.call("N64_seed_1", lambda: (result, {2: np.float64(1.25)}))
    first.finish()
    original_record = (directory / "continuation.json").read_bytes()
    resumed = ExperimentSession(directory, identity, resume=True)
    def must_not_run():
        raise AssertionError("completed inference was recomputed")
    actual = resumed.call("N64_seed_1", must_not_run)
    assert isinstance(actual[0], RolloutResult)
    np.testing.assert_array_equal(actual[0].prediction, original[0].prediction)
    assert actual[0].prediction.dtype == np.float32
    assert actual[0].failure_reason == (None, None, None)
    assert actual[1] == {2: 1.25}
    assert np.isnan(actual[0].correction["nan_example"])
    assert resumed.output.parent == directory / "resumed_attempts"
    resumed.finish()
    assert (directory / "continuation.json").read_bytes() == original_record
    assert json.loads((resumed.output / "continuation.json").read_text())["calls"][0]["action"] == "reused_completed_same_run_call"


def test_interrupted_uncommitted_call_does_not_invalidate_completed_work(tmp_path):
    directory = tmp_path / "run"
    first = ExperimentSession(directory, {"source": "v1"})
    first.call("first", lambda: np.arange(3))
    # Crash debris has no committed '<key>.json' receipt and is never executed.
    (first.cache / "second.deadbeef.receipt.json").write_text("unfinished")
    first.close()
    resumed = ExperimentSession(directory, {"source": "v1"}, resume=True)
    np.testing.assert_array_equal(resumed.call("first", lambda: pytest.fail("recomputed")), np.arange(3))
    np.testing.assert_array_equal(resumed.call("second", lambda: np.arange(4)), np.arange(4))
    assert (resumed.cache / "second.deadbeef.receipt.json").read_text() == "unfinished"
    resumed.finish()


def test_changed_identity_or_corrupt_payload_is_rejected(tmp_path):
    directory = tmp_path / "run"
    first = ExperimentSession(directory, {"source": "v1", "input_sha": "a"})
    first.call("unit", lambda: np.arange(3))
    first.close()
    with pytest.raises(ValueError, match="identity differs"):
        ExperimentSession(directory, {"source": "v2", "input_sha": "a"}, resume=True)
    receipt = json.loads((directory / ".resume" / "unit.json").read_text())
    array = directory / ".resume" / receipt["arrays"]["file"]
    with array.open("ab") as stream:
        stream.write(b"corruption")
    resumed = ExperimentSession(directory, {"source": "v1", "input_sha": "a"}, resume=True)
    with pytest.raises(ValueError, match="path/hash mismatch"):
        resumed.call("unit", lambda: pytest.fail("corruption silently replaced"))
    resumed.close()


def test_concurrent_owner_and_bad_keys_rejected(tmp_path):
    directory = tmp_path / "run"
    first = ExperimentSession(directory, {})
    with pytest.raises(RuntimeError, match="Another process owns"):
        ExperimentSession(directory, {}, resume=True)
    with pytest.raises(ValueError, match="Invalid or repeated"):
        first.call("../escape", lambda: 1)
    first.call("one", lambda: 1)
    with pytest.raises(ValueError, match="Invalid or repeated"):
        first.call("one", lambda: 2)
    first.close()


def test_object_arrays_cannot_create_completed_receipt(tmp_path):
    session = ExperimentSession(tmp_path / "run", {})
    with pytest.raises(TypeError, match="object arrays"):
        session.call("bad", lambda: np.asarray([object()], dtype=object))
    assert not (session.cache / "bad.json").exists()
    session.close()


def test_equal_configuration_is_not_permission_to_borrow_another_run(tmp_path):
    first = ExperimentSession(tmp_path / "first", {"config": "equal"})
    first.call("unit", lambda: 1)
    receipt = json.loads((first.cache / "unit.json").read_text())
    second = ExperimentSession(tmp_path / "second", {"config": "equal"})
    # Deliberately tamper only with the test-owned receipt to exercise run ID
    # checking before any external payload path can be followed.
    (second.cache / "unit.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity mismatch"):
        second.call("unit", lambda: pytest.fail("borrowed another run"))
    first.close()
    second.close()
