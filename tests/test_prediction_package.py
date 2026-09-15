"""Lossless canonical export and M1 source immutability checks on tiny fixtures."""

import json

import h5py
import numpy as np
import pytest

from scripts.prepare_prediction_package import canonical_hdf5
from training.checkpoints import digest_file


def container(path, test_ids=(1000, 1020, 1042)):
    with h5py.File(path, "x") as handle:
        # A metadata range is not an explicit list of trajectory identities.
        handle.attrs["metadata_json"] = json.dumps({"training_trajectory_ids": [[0, 49], [1200, 2149]],
                                                    "validation_trajectory_ids": [50, 69]})
        for name, ids in (("training", (0, 1200)), ("validation", (50,)), ("test", test_ids)):
            group = handle.create_group(name)
            group.create_dataset("trajectory_index", data=np.asarray(ids, dtype=np.int64))
            group.create_dataset("time", data=np.asarray([5.0, 5.5, 6.0]))
            shape = (len(ids), 3, 4, 4)
            group.create_dataset("vorticity", data=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                                 compression="gzip")
            group.create_dataset("initial_parameters", data=np.arange(len(ids) * 16, dtype=np.float64).reshape(len(ids), 4, 4))


def test_reindex_preserves_observations_and_source_file(tmp_path):
    source, target = tmp_path / "source.h5", tmp_path / "target.h5"
    container(source)
    before = digest_file(source)
    record = canonical_hdf5(source, target)
    assert digest_file(source) == before
    with h5py.File(source, "r") as old, h5py.File(target, "r") as new:
        assert new.attrs["trajectory_id_scheme"] == "canonical"
        assert new["training/vorticity"].chunks[0] == 1
        assert new["training/trajectory_index"][:].tolist() == [0, 50]
        assert new["validation/trajectory_index"][:].tolist() == [1000]
        assert new["validation_aux/trajectory_index"][:].tolist() == [1020]
        assert new["test/trajectory_index"][:].tolist() == [1040, 1062]
        metadata = json.loads(new.attrs["metadata_json"])
        assert metadata["training_trajectory_ids"] == [0, 50]
        assert metadata["validation_trajectory_ids"] == [1000, 1020]
        assert metadata["test_trajectory_ids"] == [1040, 1062]
        for name in ("vorticity", "initial_parameters"):
            assert np.array_equal(new[f"test/{name}"][:], old[f"test/{name}"][1:])
            assert np.array_equal(new[f"validation_aux/{name}"][:], old[f"test/{name}"][:1])
            assert np.array_equal(new[f"training/{name}"][:], old[f"training/{name}"][:])
        assert len(record["/test/vorticity"]["decoded_sha256"]) == 64
    with pytest.raises(FileExistsError):
        canonical_hdf5(source, target)


def test_training_rows_cannot_be_exported_as_test(tmp_path):
    source = tmp_path / "bad.h5"
    container(source, test_ids=(1200,))
    with pytest.raises(ValueError, match="training observations"):
        canonical_hdf5(source, tmp_path / "rejected.h5")
