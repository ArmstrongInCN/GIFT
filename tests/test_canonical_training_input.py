"""Canonical HDF schema tests with sparse synthetic fields."""
import h5py
import numpy as np
import pytest
from gift.canonical_package import training_input


@pytest.fixture
def package(tmp_path):
    """Create sparse canonical-layout fields with NaN fill, avoiding allocation of full trajectories."""
    path = tmp_path / "observations.h5"
    with h5py.File(path, "x") as handle:
        handle.attrs["trajectory_id_scheme"] = "canonical"
        handle.attrs["fixture_only"] = True
        for name, ids in (("training", range(1000)), ("validation", range(1000, 1020))):
            handle.create_dataset(f"{name}/trajectory_index", data=np.asarray(list(ids), dtype="i8"))
            handle.create_dataset(f"{name}/time", data=np.arange(501) * .02)
            handle.create_dataset(f"{name}/vorticity", shape=(len(ids), 501, 64, 64),
                                  dtype="f4", fillvalue=np.nan, chunks=(1, 1, 64, 64))
    return tmp_path, path


def test_metadata_reader_needs_no_manifest(package):
    """Metadata-only inspection must report canonical IDs without requiring a collection manifest."""
    root, path = package
    result = training_input(root, path, full=False)
    assert result["profile"] == "canonical"
    assert result["fixture_only"]
    assert result["trajectory_ids"] == list(range(1000))
    assert not result["full_input_checks_performed"]


def test_full_check_rejects_unwritten_observations(package):
    """A requested value scan must detect NaN-filled sparse fields rather than certify unseen data."""
    root, path = package
    with pytest.raises(ValueError, match="nonfinite"):
        training_input(root, path, full=True)


@pytest.mark.parametrize("change", ["ids", "time", "scheme"])
def test_actual_observation_contract_rejected(package, change):
    """Reject wrong training IDs, time spacing or an incorrect declared ID scheme."""
    root, path = package
    with h5py.File(path, "r+") as handle:
        if change == "ids":
            handle["training/trajectory_index"][0] = 1040
        elif change == "time":
            handle["validation/time"][1] = .03
        else:
            handle.attrs["trajectory_id_scheme"] = "storage"
    with pytest.raises(ValueError):
        training_input(root, path, full=False)
