"""Ensure renumbering cannot hide split overlap or silently omit trajectories."""

import pytest

from gift.data_splits import (
    CHECKPOINT_VALIDATION_IDS, LITE_TRAINING_IDS, TEST_IDS, TRAINING_IDS,
    VALIDATION_IDS, canonical_ids, partition_rows, prediction_split_manifest,
    split_name, storage_ids,
)


def test_partition_is_contiguous_disjoint_and_complete():
    combined = TRAINING_IDS + VALIDATION_IDS + TEST_IDS
    assert combined == tuple(range(1220))
    assert len(set(combined)) == len(combined)
    assert LITE_TRAINING_IDS == tuple(range(50))
    assert set(CHECKPOINT_VALIDATION_IDS) <= set(VALIDATION_IDS)
    assert tuple(map(split_name, TRAINING_IDS)) == ("training",) * 1000
    assert tuple(map(split_name, VALIDATION_IDS)) == ("validation",) * 40
    assert tuple(map(split_name, TEST_IDS)) == ("test",) * 180


def test_all_observation_identities_round_trip_without_field_changes():
    public = tuple(range(1220))
    stored = storage_ids(public)
    assert len(stored) == len(set(stored)) == 1220
    assert canonical_ids(stored) == public
    assert canonical_ids([2149, 0, 50, 1000, 1019, 1020, 1199]) == (
        999, 0, 1000, 1020, 1039, 1040, 1219,
    )


def test_independent_test_does_not_include_validation_or_training():
    original_evaluation = tuple(range(1000, 1200))
    rows = partition_rows(original_evaluation)
    assert rows == {"training": (), "validation": tuple(range(20)), "test": tuple(range(20, 200))}
    assert set(storage_ids(TEST_IDS)).isdisjoint(storage_ids(VALIDATION_IDS))
    assert set(storage_ids(TEST_IDS)).isdisjoint(storage_ids(TRAINING_IDS))


def test_correction_second_cohort_is_training_not_test():
    rows = partition_rows(range(1200, 1400))
    assert rows == {"training": tuple(range(200)), "validation": (), "test": ()}


@pytest.mark.parametrize("values", [[-1], [70], [999], [2150]])
def test_unknown_storage_ids_are_rejected(values):
    with pytest.raises(ValueError, match="unknown"):
        canonical_ids(values)


@pytest.mark.parametrize("values", [[-1], [1220]])
def test_unknown_public_ids_are_rejected(values):
    with pytest.raises(ValueError, match="unknown"):
        storage_ids(values)


@pytest.mark.parametrize("value", [1.0, True, "1", None])
def test_coercion_cannot_silently_change_identity(value):
    for operation in (canonical_ids, storage_ids):
        with pytest.raises(TypeError):
            operation([value])


def test_duplicate_rows_are_not_silently_accepted():
    for operation in (canonical_ids, storage_ids):
        with pytest.raises(ValueError, match="duplicate"):
            operation([0, 0])


def test_manifest_does_not_call_lite_training_a_new_split():
    value = prediction_split_manifest()
    assert set(value["gift_lite_training"]) < set(value["training"])
    assert set(value["training"]).isdisjoint(value["validation"])
    assert set(value["test"]).isdisjoint(value["validation"])
    assert len(value["test"]) == 180
