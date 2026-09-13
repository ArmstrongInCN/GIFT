"""CPU-only checks of noise and measurement preparation, without model imports."""
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import generate_noise as noise  # noqa: E402 - standalone checkout test
from scripts import generate_sampling as sampling  # noqa: E402 - standalone checkout test


def test_noise_resume_crosses_group_boundary(tmp_path):
    source = tmp_path / "clean.h5"
    with h5py.File(source, "x") as handle:
        for name, ids in (("training", [0]), ("validation", [50])):
            group = handle.create_group(name)
            group.create_dataset("trajectory_index", data=np.asarray(ids, dtype=np.int64))
            group.create_dataset("time", data=np.arange(17)*.02)
            data = np.random.RandomState(10 if name == "training" else 11).normal(size=(1, 17, 64, 64)).astype(np.float32)
            group.create_dataset("vorticity", data=data)
    straight, paused = tmp_path / "straight", tmp_path / "paused"
    common = ["--clean", str(source), "--pilot", "--execute", "--checkpoint-every", "1"]
    noise.main(common + ["--output", str(straight)])
    noise.main(common + ["--output", str(paused), "--stop-after-chunks", "2"])
    assert not (paused / "COMPLETE.json").exists()
    noise.main(common + ["--output", str(paused), "--resume"])
    for condition in noise.FRACTIONS:
        with h5py.File(straight / (condition + ".h5"), "r") as a, h5py.File(paused / (condition + ".h5"), "r") as b:
            for group in noise.GROUPS:
                np.testing.assert_array_equal(a[group + "/vorticity"][:], b[group + "/vorticity"][:])
    with pytest.raises(ValueError, match="already complete"):
        noise.main(common + ["--output", str(paused), "--resume"])


def test_rng_state_includes_cached_gaussian():
    rng = np.random.RandomState(0)
    rng.standard_normal(3)
    saved = noise.rng_arrays(rng)
    assert saved["rng_has_gauss"] == 1
    restored = noise.restore_rng(saved)
    np.testing.assert_array_equal(rng.standard_normal(101), restored.standard_normal(101))


def test_original_noise_first_two_blocks(tmp_path):
    configured = os.environ.get("GIFT_TEST_REFERENCE_DATA_ROOT")
    if not configured:
        pytest.skip("Optional read-only reference data not configured")
    data = Path(configured)
    output = tmp_path / "first_two_noise_blocks"
    common = ["--clean", str(data / "standard_ns_n64_full_spectrum.h5"),
              "--output", str(output), "--stop-after-chunks", "1", "--execute"]
    noise.main(common)
    noise.main(common + ["--resume"])
    with (output / "run.json").open(encoding="utf-8") as stream:
        run = json.load(stream)
    assert run["binding"]["scaling"] == "std(clean_group,ddof=0)"
    assert not (output / "COMPLETE.json").exists()
    for condition in noise.FRACTIONS:
        with h5py.File(output / (condition + ".h5"), "r") as actual, h5py.File(data / "m1_parameter_identification" / (condition + ".h5"), "r") as reference:
            np.testing.assert_array_equal(actual["training/vorticity"][0, :32], reference["training/vorticity"][0, :32])
    print("NOISE_ORACLE_EXACT=2 conditions x 32 frames x 64 x 64; remainder not generated")


@pytest.mark.parametrize("condition", ["noise_000", "noise_001", "noise_010"])
def test_sampling_against_original_and_own_resume(tmp_path, condition):
    configured = os.environ.get("GIFT_TEST_REFERENCE_DATA_ROOT")
    if not configured:
        pytest.skip("Optional read-only reference data not configured")
    data = Path(configured)
    source = data / "standard_ns_n64_full_spectrum.h5" if condition == "noise_000" else data / "m1_parameter_identification" / (condition + ".h5")
    output = tmp_path / condition
    common = ["--input", str(source), "--output", str(output), "--execute"]
    sampling.main(common + ["--stop-after-design"])
    assert not (output / "sampling.npz").exists()
    sampling.main(common + ["--resume"])
    with np.load(output / "sampling.npz", allow_pickle=False) as actual, np.load(data / "auxiliary/pinn_sampling" / (condition + "_seed1234.npz"), allow_pickle=False) as reference:
        assert set(actual.files) == set(reference.files)
        for name in reference.files:
            assert actual[name].dtype == reference[name].dtype
            np.testing.assert_array_equal(actual[name], reference[name], err_msg=condition + ":" + name)
    print("SAMPLING_ORACLE_EXACT=" + condition + ":10 arrays, design-resume checked, no network")
