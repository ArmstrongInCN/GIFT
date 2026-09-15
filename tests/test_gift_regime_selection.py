"""Checkpoint-metadata fixtures test regime selection, not model computation."""

from argparse import Namespace

import pytest
import torch

from experiments.formal._shared.common import ProjectPaths, SEEDS
from experiments.formal._shared.gift_regimes import resolve_regimes, extra_regime_files
from training.checkpoints import digest_file


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root, data = tmp_path / "project", tmp_path / "data"
    root.mkdir()
    data.mkdir()
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    monkeypatch.delenv("GIFT_CHECKPOINT_ROOT", raising=False)
    paths = ProjectPaths.from_root(root)
    args = Namespace(low_model=None, gift_model=[], lite_low_model=None, lite_gift_model=[])
    return root, paths, args


def weights(root, *, lite=False):
    low = (root / "artifacts/fixed_k21_n64_unmasked/gift_main.pt" if lite
           else root / "artifacts/gift_full/generator.pt")
    low.parent.mkdir(parents=True)
    payload = ({"training_data": {"trajectory_ids": list(range(50))}} if lite else
               {"training_regime": "full_data_prediction", "training_cost": {"epochs": 500, "optimizer_updates": 31500},
                "training_data": {"training_ids": list(range(1000))}})
    torch.save(payload, low)
    for seed in SEEDS:
        directory = root / ("artifacts/gift" if lite else "artifacts/gift_full")
        directory.mkdir(exist_ok=True)
        torch.save({"seed": seed, "phase": "terminal_epoch", "frozen_generator_sha256": digest_file(low),
                    "training_cost": {"epochs": 500, "optimizer_updates": 45200},
                    "training_data": {"training_ids": list(range(1000))}}, directory / f"gift_seed_{seed}.pt")


@pytest.mark.parametrize("regime", ["GIFT", "GIFT-Lite"])
def test_single_regime_does_not_require_unrelated_weights(setup, regime):
    root, paths, args = setup
    weights(root, lite=regime == "GIFT-Lite")
    args.gift_regime = regime
    assert list(resolve_regimes(args, paths)) == [regime]


def test_two_regimes_and_canonical_prediction_path(setup):
    root, paths, args = setup
    weights(root)
    weights(root, lite=True)
    prediction = paths.standard_n64.parent / "prediction" / paths.standard_n64.name
    prediction.parent.mkdir()
    prediction.touch()  # Path selection only; not a scientific HDF5 fixture.
    result = resolve_regimes(args, paths)
    assert list(result) == ["GIFT", "GIFT-Lite"]
    assert all(value[0].standard_n64 == prediction for value in result.values())


@pytest.mark.parametrize("regime", ["GIFT", "GIFT-Lite"])
def test_prediction_subset_binds_only_selected_completed_models(setup, regime):
    root, paths, args = setup
    weights(root, lite=regime == "GIFT-Lite")
    args.gift_regimes = [regime]
    result = resolve_regimes(args, paths)
    assert list(result) == [regime]
    assert extra_regime_files(result) == {}


def test_duplicate_regime_is_rejected(setup):
    _, paths, args = setup
    args.gift_regimes = ["GIFT-Lite", "GIFT-Lite"]
    with pytest.raises(ValueError, match="at most once"):
        resolve_regimes(args, paths)


@pytest.mark.parametrize("change", ["seed", "generator", "budget"])
def test_mislabeled_full_branch_is_rejected(setup, change):
    root, paths, args = setup
    weights(root)
    args.gift_regime = "GIFT"
    path = root / "artifacts/gift_full/gift_seed_20260820.pt"
    payload = torch.load(path, weights_only=True)
    if change == "seed":
        payload["seed"] = 20260821
    elif change == "generator":
        payload["frozen_generator_sha256"] = "0" * 64
    else:
        payload["training_cost"]["optimizer_updates"] -= 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="matching seed and generator"):
        resolve_regimes(args, paths)
