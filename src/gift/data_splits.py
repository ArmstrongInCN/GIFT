"""Canonical identities for the prediction benchmark's disjoint data splits.

Training IDs come first, followed by validation and independent test IDs.
Different grids and observation conditions of one trajectory always share one
identity. The conversion is an explicit storage-ID lookup, never error-based
sample selection. M1 keeps its separately defined identification data protocol.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable


TRAINING_IDS = tuple(range(1000))
LITE_TRAINING_IDS = TRAINING_IDS[:50]
VALIDATION_IDS = tuple(range(1000, 1040))
# These twenty trajectories contain the full 0--10 training/validation time range.
CHECKPOINT_VALIDATION_IDS = VALIDATION_IDS[:20]
TEST_IDS = tuple(range(1040, 1220))

# Existing observation containers store these identities. Their physical fields
# must be preserved when materializing a package with canonical IDs. This table
# is needed to read those containers; it does not define an additional split.
_STORAGE_IDS = (
    *range(50), *range(1200, 2150), *range(50, 70), *range(1000, 1200)
)
_TO_CANONICAL = {stored: canonical for canonical, stored in enumerate(_STORAGE_IDS)}


def _integer(value: object) -> int:
    """Reject silently truncated floats and boolean pseudo-identifiers."""
    if isinstance(value, bool):
        raise TypeError("trajectory IDs must be integers, not booleans")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError("trajectory IDs must be integers") from exc


def canonical_ids(storage_ids: Iterable[int]) -> tuple[int, ...]:
    """Map observation identities without sorting, dropping or duplicating rows."""
    result = []
    for raw in storage_ids:
        stored = _integer(raw)
        if stored not in _TO_CANONICAL:
            raise ValueError(f"unknown observation trajectory identity: {stored}")
        result.append(_TO_CANONICAL[stored])
    if len(set(result)) != len(result):
        raise ValueError("duplicate trajectory identities in one array")
    return tuple(result)


def storage_ids(public_ids: Iterable[int]) -> tuple[int, ...]:
    """Resolve canonical IDs for read-only access to observation containers."""
    result = []
    for raw in public_ids:
        identifier = _integer(raw)
        if not 0 <= identifier < len(_STORAGE_IDS):
            raise ValueError(f"unknown canonical trajectory identity: {identifier}")
        result.append(_STORAGE_IDS[identifier])
    if len(set(result)) != len(result):
        raise ValueError("duplicate trajectory identities in one array")
    return tuple(result)


def split_name(identifier: int) -> str:
    """Return exactly one of the three benchmark partitions."""
    identifier = _integer(identifier)
    if 0 <= identifier < 1000:
        return "training"
    if 1000 <= identifier < 1040:
        return "validation"
    if 1040 <= identifier < 1220:
        return "test"
    raise ValueError(f"unknown canonical trajectory identity: {identifier}")


def partition_rows(storage_identifiers: Iterable[int]) -> dict[str, tuple[int, ...]]:
    """Return row indices for each split, preserving the input's row order."""
    rows: dict[str, list[int]] = {name: [] for name in ("training", "validation", "test")}
    for row, identifier in enumerate(canonical_ids(storage_identifiers)):
        rows[split_name(identifier)].append(row)
    return {name: tuple(indices) for name, indices in rows.items()}


def prediction_split_manifest() -> dict:
    """Machine-readable partitions, with no validation/test overlap."""
    return {
        "schema": "gift.prediction-splits.v1",
        "identity_scope": "same initial condition across grids and noise conditions",
        "training": list(TRAINING_IDS),
        "validation": list(VALIDATION_IDS),
        "test": list(TEST_IDS),
        "gift_lite_training": list(LITE_TRAINING_IDS),
        "checkpoint_validation": list(CHECKPOINT_VALIDATION_IDS),
        "test_selection": "fixed identity partition, not selected by model outcomes",
        "m1_protocol": "separate equation-identification dataset; unchanged",
    }
