"""Disjoint, prespecified prediction populations; IDs never depend on errors."""

from .data_splits import prediction_split_manifest

FOUR_VORTEX = "four_vortex"
GAUSSIAN = "gaussian_s4"
COHORTS = (FOUR_VORTEX, GAUSSIAN)


def partitions(cohort=FOUR_VORTEX):
    """Keep the original IDs intact and append the Gaussian population."""
    if cohort not in COHORTS:
        raise ValueError("Unknown prediction initial-condition population")
    if cohort == FOUR_VORTEX:
        return prediction_split_manifest()
    return {
        "schema": "gift.prediction-splits.v1", "cohort": GAUSSIAN,
        "training": list(range(1220, 2220)),
        "validation": list(range(2220, 2260)),
        "test": list(range(2260, 2440)),
        "gift_lite_training": list(range(1220, 1270)),
        "checkpoint_validation": list(range(2220, 2240)),
        "test_selection": "fixed before generation and model training",
    }


def observation_cohort(handle):
    """Missing tags retain the existing four-vortex reader contract."""
    cohort = handle.attrs.get("prediction_cohort", FOUR_VORTEX)
    if cohort not in COHORTS:
        raise ValueError("Unknown observation initial-condition population")
    return cohort
