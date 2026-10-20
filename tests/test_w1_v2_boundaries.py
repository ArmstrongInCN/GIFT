"""Check scientific resume bindings and the predeclared rollout curriculum."""
import copy
import hashlib

import h5py
import numpy as np
import pytest

from training.w1_v2_data import (LOCAL_TIMES, SCHEMA, SPARSE_SCHEMA, SUBSET_SEED,
                                 WindowBank, prepare_window, window_manifest)
from training.w1_v2_train import scientific_identity, training_horizon, _bind_window_grid


def _window(path):
    """A fresh small fixture carries the same copy receipt as the real window."""
    with h5py.File(path, "x") as output:
        output.attrs.update(schema=SCHEMA, status="complete", fixture_only=True,
                            source_file="fixture-original.h5")
        for name, ident, amplitude in (("training", 0, 1.), ("validation", 1, 2.)):
            group = output.create_group(name)
            values = np.full((1, 151, 64, 64), amplitude, dtype=np.float32)
            group["vorticity"] = values
            group["trajectory_index"] = np.array([ident], dtype=np.int64)
            group["time"] = LOCAL_TIMES
            group["absolute_time"] = 5. + LOCAL_TIMES
            group.attrs["observations_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()


def test_window_binding_checks_consumed_fields_instead_of_trusting_receipt(tmp_path):
    """A finite field edit with an unchanged attribute must fail before training."""
    path = tmp_path / "window.h5"
    _window(path)
    original = WindowBank(path, fixture=True)
    assert original.binding["validation"]["observations_sha256"] == hashlib.sha256(
        original.validation.tobytes()).hexdigest()
    with h5py.File(path, "r+") as output:
        output["validation/vorticity"][0, 100, 0, 0] = 3.
    with pytest.raises(ValueError, match="observations differ"):
        WindowBank(path, fixture=True)


def test_scientific_identity_allows_paths_but_binds_data_method_and_training(tmp_path):
    """Relocation is provenance; a changed free term, LR or decoded data is semantic."""
    path = tmp_path / "window.h5"
    _window(path)
    identity = dict(schema="gift.w1-v2.training.v2",
                    model_spec=dict(method="anie_ns", history=1, condition_mode="native", source_root="old-source"),
                    config=dict(learning_rate=1e-4, micro_batch=4, effective_batch=32,
                                max_epochs=1500, sample_seed=2026100301),
                    data=WindowBank(path, fixture=True).binding)
    relocated = copy.deepcopy(identity)
    relocated["model_spec"]["source_root"] = "new-source"
    relocated["data"].update(file="new-window", source_file="new-original")
    assert scientific_identity(relocated) == scientific_identity(identity)
    for section, key, value in (("config", "learning_rate", 3e-4),
                                ("config", "micro_batch", 8),
                                ("model_spec", "condition_mode", "fixed_f")):
        changed = copy.deepcopy(identity)
        changed[section][key] = value
        assert scientific_identity(changed) != scientific_identity(identity)
    changed = copy.deepcopy(identity)
    changed["data"]["training"]["observations_sha256"] = "different-decoded-fields"
    assert scientific_identity(changed) != scientific_identity(identity)


def test_curriculum_boundaries_are_fixed_and_other_methods_keep_full_horizon():
    """Six declared twenty-epoch stages end at epoch120, then every stage is150."""
    config = {"closed_loop_curriculum": True}
    for first, last, horizon in ((1, 20, 1), (21, 40, 5), (41, 60, 10),
                                 (61, 80, 20), (81, 100, 50), (101, 120, 100),
                                 (121, 1500, 150)):
        assert training_horizon("fno2d", first, config) == horizon
        assert training_horizon("fno2d", last, config) == horizon
    assert training_horizon("fno2d", 1, {"closed_loop_curriculum": False}) == 150
    for method in ("anie_ns", "anie_f", "fno3d"):
        assert training_horizon(method, 1, config) == 150


def test_sparse_manifest_is_seeded_before_model_rng_and_preserves_physical_window():
    """N200 and stride5 select real source observations, with a separate subset RNG."""
    first = window_manifest(200, 5, SUBSET_SEED)
    np.random.seed(97)
    second = window_manifest(200, 5, SUBSET_SEED)
    assert first == second
    assert first["training_ids"] == np.random.default_rng(2026100304).permutation(1000)[:200].tolist()
    assert first["validation_ids"] == list(range(1000, 1020))
    assert first["source_frame_indices"] == list(range(250, 401, 5))
    assert first["points"] == 31 and first["future_steps"] == 30
    assert np.allclose(first["query_times"], np.arange(31) * .1, rtol=0, atol=2e-12)
    for kwargs in ({"training_count": 0}, {"training_count": True}, {"time_stride": 4},
                   {"time_stride": 0}, {"subset_seed": -1}):
        with pytest.raises(ValueError):
            window_manifest(**kwargs)


@pytest.mark.parametrize("training_count,time_stride", [(200, 5), (300, 2)])
def test_formal_sparse_copy_binding_and_training_only_statistics(tmp_path, training_count, time_stride):
    """Exercise the legal formal schema, real ID subset and hash; never enable fixture mode.

    A virtual fill dataset gives the source's full canonical population without
    allocating every original field. Only the selected observed frames are
    written, and a deliberately unusable test group must never be accessed.
    """
    original, derived = tmp_path / "canonical.h5", tmp_path / "sparse.h5"
    manifest = window_manifest(training_count, time_stride, SUBSET_SEED)
    points = manifest["points"]
    frames = manifest["source_frame_indices"]
    with h5py.File(original, "x") as source:
        for split, ids in (("training", np.arange(1000)), ("validation", np.arange(1000, 1020))):
            node = source.create_group(split)
            node["trajectory_index"] = ids
            node["time"] = np.arange(501) * .02
            field = node.create_dataset("vorticity", shape=(len(ids), 501, 64, 64), dtype="f4",
                                        chunks=(1, 1, 64, 64), fillvalue=1., compression="lzf")
            selected = manifest["training_ids"] if split == "training" else ids
            for ident in selected:
                row = int(ident) if split == "training" else int(ident) - 1000
                # All future positions have nonzero between-training SD; val is
                # deliberately far away so leakage would change the mean.
                amplitude = 1. + int(ident) / 1000 if split == "training" else 100.
                field[row, frames] = np.full((points, 64, 64), amplitude, dtype="f4")
        source["test"] = "must not be consumed"
    assert prepare_window(original, derived, training_count=training_count, time_stride=time_stride) == derived.resolve()
    assert prepare_window(original, derived, training_count=training_count, time_stride=time_stride) == derived.resolve()
    bank = WindowBank(derived)
    assert bank.binding["schema"] == SPARSE_SCHEMA and not bank.binding["fixture_only"]
    assert bank.training.shape == (training_count, points, 64, 64) and bank.validation.shape == (20, points, 64, 64)
    assert bank.training_ids.tolist() == manifest["training_ids"]
    assert bank.validation_ids.tolist() == list(range(1000, 1020))
    assert bank.binding["query_times"] == manifest["query_times"]
    assert bank.future_steps == points - 1 and bank.time_stride == time_stride
    assert np.allclose(bank.training[:, :, 0, 0],
                       (1 + bank.training_ids / 1000).astype("f4")[:, None])
    norm = bank.normalization("fno3d")
    assert norm["output_mean"].shape == (64, 64, points - 1)
    assert norm["output_std"].shape == (64, 64, points - 1)
    assert float(norm["output_mean"].max()) < 2 and not norm["validation_or_test_used"]
    with h5py.File(derived) as source:
        assert set(source.keys()) == {"training", "validation"}
    with pytest.raises(ValueError, match="configuration differs"):
        prepare_window(original, derived, training_count=training_count, time_stride=time_stride, subset_seed=SUBSET_SEED + 1)
    with pytest.raises(ValueError, match="fixture/formal"):
        WindowBank(derived, fixture=True)
    with h5py.File(derived, "r+") as source:
        source["training/time"][1] = .11
    with pytest.raises(ValueError, match="local time grid differs"):
        WindowBank(derived)


def test_sparse_grid_changes_resume_identity_and_requires_explicit_training_contract():
    """New query times/count cannot restore a dense checkpoint or silently use 150 steps."""
    manifest = window_manifest(200, 5)
    binding = {k: v for k, v in manifest.items() if k not in ("training_ids", "validation_ids")}
    spec = dict(method="fno2d", history=1, source_root="fixture")
    config = dict(future_steps=30, closed_loop_curriculum=True,
                  closed_loop_horizons=[1, 1, 2, 4, 10, 20, 30])
    actual = _bind_window_grid(spec, config, binding)
    assert actual["query_times"] == manifest["query_times"] and actual["future_steps"] == 30
    assert "query_times" not in spec
    rounded = _bind_window_grid(dict(spec, query_times=(np.arange(31) * .1).tolist()), config, binding)
    assert rounded == actual
    with pytest.raises(ValueError, match="explicit training future_steps"):
        _bind_window_grid(spec, {}, binding)
    with pytest.raises(ValueError, match="query_times differ"):
        _bind_window_grid(dict(spec, query_times=LOCAL_TIMES.tolist()), config, binding)
    for epoch, horizon in ((1, 1), (20, 1), (21, 1), (40, 1), (41, 2), (60, 2),
                           (61, 4), (80, 4), (81, 10), (100, 10), (101, 20),
                           (120, 20), (121, 30), (1500, 30)):
        assert training_horizon("fno2d", epoch, config) == horizon
    assert training_horizon("anie_ns", 1, config) == 30
    with pytest.raises(ValueError, match="closed_loop_horizons"):
        training_horizon("fno2d", 1, dict(future_steps=30, closed_loop_curriculum=True))
    dense = dict(schema="gift.w1-v2.training.v2", model_spec=spec, config={}, data=dict(points=151))
    sparse = dict(schema="gift.w1-v2.training.v2", model_spec=actual, config=config, data=binding)
    assert scientific_identity(dense) != scientific_identity(sparse)


def test_sparse76_is_nested_and_uses_the_predeclared_nearest_earlier_curriculum():
    """The0.1-second tie becomes0.08, while the first stage needs at least one0.04 step."""
    previous = window_manifest(200, 5)
    manifest = window_manifest(300, 2)
    assert manifest["training_ids"][:200] == previous["training_ids"]
    assert manifest["training_ids"] == np.random.default_rng(2026100304).permutation(1000)[:300].tolist()
    assert manifest["source_frame_indices"] == list(range(250, 401, 2))
    assert manifest["query_times"] == LOCAL_TIMES[::2].tolist()
    assert manifest["points"] == 76 and manifest["future_steps"] == 75
    config = dict(future_steps=75, closed_loop_curriculum=True,
                  closed_loop_horizons=[1, 2, 5, 10, 25, 50, 75])
    binding = {k: v for k, v in manifest.items() if k not in ("training_ids", "validation_ids")}
    spec = _bind_window_grid(dict(method="fno2d", query_times=(np.arange(76) * .04).tolist(),
                                 observation_dt=.04), config, binding)
    assert spec["query_times"] == manifest["query_times"] and spec["future_steps"] == 75
    for stage, horizon in enumerate(config["closed_loop_horizons"]):
        assert training_horizon("fno2d", stage * 20 + 1, config) == horizon
        assert training_horizon("fno2d", stage * 20 + 20, config) == horizon
    assert training_horizon("fno2d", 1500, config) == 75
    assert np.allclose(np.asarray(config["closed_loop_horizons"]) * .04,
                       [.04, .08, .2, .4, 1., 2., 3.], atol=2e-12, rtol=0)
    for method in ("anie_ns", "anie_f", "fno3d"):
        assert training_horizon(method, 1, config) == 75
    with pytest.raises(ValueError, match="training future_steps differ"):
        _bind_window_grid(dict(method="fno2d"), dict(future_steps=30), binding)
    with pytest.raises(ValueError, match="query_times differ"):
        _bind_window_grid(dict(method="fno2d", query_times=previous["query_times"]), config, binding)
