"""Identity-based test selection must preserve every independent observation."""

import h5py
import numpy as np
import pytest

from experiments.formal._shared.prediction_data import _observations, _positions


def write_container(path, *, canonical=False, duplicate=False, missing=False):
    ids = np.arange(1040, 1220) if canonical else np.arange(1000, 1200)
    if duplicate:
        ids[-1] = ids[-2]
    if missing:
        ids = ids[:-1]
    with h5py.File(path, "x") as handle:
        handle.attrs["fixture_only"] = True
        if canonical:
            handle.attrs["trajectory_id_scheme"] = "canonical"
        group = handle.create_group("test")
        group.create_dataset("trajectory_index", data=ids.astype(np.int64))
        group.create_dataset("time", data=np.asarray([5.0, 5.5, 6.0], dtype=np.float64))
        values = np.broadcast_to(ids[:, None, None, None], (len(ids), 3, 4, 4)).astype(np.float32)
        group.create_dataset("vorticity", data=values)


def test_storage_identity_selection_is_not_value_based(tmp_path):
    path = tmp_path / "observations.h5"
    write_container(path)
    ids, times, fields = _observations(path, "test", 4)
    assert np.array_equal(ids, np.arange(1040, 1220))
    assert fields.shape == (180, 3, 4, 4)
    assert fields[0, 0, 0, 0] == 1020
    assert fields[-1, -1, -1, -1] == 1199
    assert times.tolist() == [5.0, 5.5, 6.0]


def test_canonical_ids_are_not_mapped_a_second_time(tmp_path):
    path = tmp_path / "canonical.h5"
    write_container(path, canonical=True)
    ids, _, fields = _observations(path, "test", 4)
    assert np.array_equal(ids, np.arange(1040, 1220))
    assert fields[0, 0, 0, 0] == 1040


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("invalid", ["duplicate", "missing"])
def test_missing_or_duplicate_test_id_is_not_silently_accepted(tmp_path, canonical, invalid):
    path = tmp_path / "bad.h5"
    write_container(path, canonical=canonical, **{invalid: True})
    with pytest.raises(ValueError):
        _observations(path, "test", 4)


def test_only_saved_times_are_returned():
    times = np.asarray([5.0, 5.5, 6.0])
    assert _positions(times, [5.0, 6.0]).tolist() == [0, 2]
    with pytest.raises(ValueError):
        _positions(times, [5.25])
    with pytest.raises(ValueError):
        _positions(times, [5.0, 5.0])
