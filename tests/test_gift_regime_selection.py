"""Checkpoint-metadata fixtures test regime selection, not model computation."""

from argparse import Namespace
from dataclasses import replace

import pytest
import torch

from experiments.formal._shared.common import ProjectPaths, SEEDS
from experiments.formal._shared.gift_regimes import resolve_regimes, extra_regime_files
from training.checkpoints import digest_file, state_digest
from gift.prediction_cohorts import GAUSSIAN, partitions


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Construct temporary path/config objects; no real observations or neural models are loaded."""
    root, data = tmp_path / "project", tmp_path / "data"
    root.mkdir()
    data.mkdir()
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    monkeypatch.delenv("GIFT_CHECKPOINT_ROOT", raising=False)
    paths = ProjectPaths.from_root(root)
    args = Namespace(low_model=None, gift_model=[], lite_low_model=None, lite_gift_model=[])
    return root, paths, args


def weights(root, *, lite=False):
    """Write metadata-only generator/branch fixtures for routing tests, not scientific model exports."""
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
    """Selecting one regime must not require the other regime's absent exports."""
    root, paths, args = setup
    weights(root, lite=regime == "GIFT-Lite")
    args.gift_regime = regime
    assert list(resolve_regimes(args, paths)) == [regime]


def test_two_regimes_and_canonical_prediction_path(setup):
    """Both regimes must share the independent prediction input while retaining their own models."""
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
    """A single selected regime contributes no unused second-regime resume files."""
    root, paths, args = setup
    weights(root, lite=regime == "GIFT-Lite")
    args.gift_regimes = [regime]
    result = resolve_regimes(args, paths)
    assert list(result) == [regime]
    assert extra_regime_files(result) == {}


def test_duplicate_regime_is_rejected(setup):
    """Repeated regime labels must fail before a misleading comparison can be assembled."""
    _, paths, args = setup
    args.gift_regimes = ["GIFT-Lite", "GIFT-Lite"]
    with pytest.raises(ValueError, match="at most once"):
        resolve_regimes(args, paths)


@pytest.mark.parametrize("change", ["seed", "generator"])
def test_mislabeled_full_branch_is_rejected(setup, change):
    """Seed, generator and population labels must describe the branch actually selected."""
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


def test_custom_budget_remains_descriptive_metadata(setup):
    """An otherwise compatible export stays selectable when its recorded training budget is custom."""
    root, paths, args = setup
    weights(root)
    args.gift_regime = "GIFT"
    low = root / "artifacts/gift_full/generator.pt"
    payload = torch.load(low, weights_only=True)
    payload["training_cost"] = {"epochs": 7, "optimizer_updates": 14}
    torch.save(payload, low)
    for seed in SEEDS:
        path = root / f"artifacts/gift_full/gift_seed_{seed}.pt"
        branch = torch.load(path, weights_only=True)
        branch["frozen_generator_sha256"] = digest_file(low)
        branch["training_cost"] = {"epochs": 9, "optimizer_updates": 18}
        torch.save(branch, path)
    assert list(resolve_regimes(args, paths)) == ["GIFT"]


def gaussian_state_weights(root, args, regime):
    """Small state/metadata fixtures; no trained model or real data are loaded."""
    split = partitions(GAUSSIAN)
    low = root / f"{regime}_generator.pt"
    state = {"fixture_weight": torch.arange(3, dtype=torch.float32)}
    data = {"cohort": GAUSSIAN, "sha256": None}
    if regime == "GIFT":
        data["training_ids"] = split["training"]
        args.low_model = low
    else:
        data["trajectory_ids"] = split["gift_lite_training"]
        args.lite_low_model = low
    torch.save({"model_state_dict": state, "training_regime": "full_data_prediction",
                "training_data": data}, low)
    specs = []
    for seed in SEEDS:
        branch = root / f"{regime}_{seed}.pt"
        torch.save({"seed": seed, "phase": "terminal_epoch", "training_data": data,
                    "generator_weights_sha256": state_digest(state),
                    # A new state fingerprint takes precedence over legacy file metadata.
                    "frozen_generator_sha256": "0" * 64}, branch)
        specs.append(f"{seed}={branch}")
    if regime == "GIFT":
        args.gift_model = specs
    else:
        args.lite_gift_model = specs
    args.gift_regime = regime
    return low, root / f"{regime}_{SEEDS[0]}.pt"


@pytest.mark.parametrize("regime", ["GIFT", "GIFT-Lite"])
def test_gaussian_ordinary_weights_use_state_binding_without_scanning_input(setup, monkeypatch, regime):
    """New Gaussian exports match loaded generator tensors once and need no unused data-file scan."""
    import experiments.formal._shared.gift_regimes as module
    root, paths, args = setup
    gaussian_state_weights(root, args, regime)
    paths = replace(paths, prediction_cohort=GAUSSIAN)
    monkeypatch.setattr(module, "digest_file", lambda _: pytest.fail("ordinary weights scanned a file"))
    calls = []
    actual_digest = module.state_digest
    def measured_state(state):
        """Count loaded-state digest calls while delegating the actual tensor digest."""
        calls.append(state)
        return actual_digest(state)
    monkeypatch.setattr(module, "state_digest", measured_state)
    assert list(resolve_regimes(args, paths)) == [regime]
    assert len(calls) == 1  # The three branch pairings share the loaded generator fingerprint.


@pytest.mark.parametrize("regime", ["GIFT", "GIFT-Lite"])
@pytest.mark.parametrize("change", ["state", "population", "cohort"])
def test_gaussian_scientific_binding_still_rejects_wrong_weights(setup, regime, change):
    """A different frozen generator must be rejected even when cohort labels match."""
    root, paths, args = setup
    low, branch = gaussian_state_weights(root, args, regime)
    path = low if change == "population" else branch
    payload = torch.load(path, weights_only=True)
    if change == "state":
        payload["generator_weights_sha256"] = "0" * 64
    elif change == "cohort":
        payload["training_data"]["cohort"] = "four_vortex"
    else:
        key = "training_ids" if regime == "GIFT" else "trajectory_ids"
        payload["training_data"][key][0] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError):
        resolve_regimes(args, replace(paths, prediction_cohort=GAUSSIAN))


@pytest.mark.parametrize("regime", ["GIFT", "GIFT-Lite"])
def test_gaussian_declared_legacy_input_hash_is_checked_once(setup, monkeypatch, regime):
    """Recorded legacy Gaussian file bindings are checked once and a wrong observation digest is rejected."""
    import experiments.formal._shared.gift_regimes as module
    from gift.gaussian_package import DENSE_FILE, LITE_FILE
    root, paths, args = setup
    low, _ = gaussian_state_weights(root, args, regime)
    dataset = root.parent / "data" / (DENSE_FILE if regime == "GIFT" else LITE_FILE)
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"synthetic legacy hash fixture, not observation data")
    checksum = digest_file(dataset)
    for path in [low, *(root / f"{regime}_{seed}.pt" for seed in SEEDS)]:
        payload = torch.load(path, weights_only=True)
        payload["training_data"]["sha256"] = checksum
        torch.save(payload, path)
    calls = []
    def measured_file(path):
        """Record the selected observation path hashed by the legacy compatibility route."""
        calls.append(path)
        return digest_file(path)
    monkeypatch.setattr(module, "digest_file", measured_file)
    resolve_regimes(args, replace(paths, prediction_cohort=GAUSSIAN))
    assert calls == [dataset]
    dataset.write_bytes(b"changed legacy scientific-input fixture")
    with pytest.raises(ValueError, match="training population"):
        resolve_regimes(args, replace(paths, prediction_cohort=GAUSSIAN))
