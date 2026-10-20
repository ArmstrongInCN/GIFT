"""Round-trip real generator/branch exports through the inference loaders."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gift import GIFTGenerator, FixedBandwidthGridGenerator
from experiments.formal.train_gift_branches import _write_checkpoint
from experiments.formal._shared.gift_runtime import load_gift_models
from experiments.formal._shared.high_frequency import HighFrequencyBranch, HighFrequencyConfig
from training.checkpoints import digest_file, state_digest


@pytest.mark.parametrize("binding", ["generator_weights_sha256", "frozen_generator_sha256"])
def test_real_exports_load_with_new_and_legacy_generator_binding(tmp_path, binding):
    """Read both binding formats and reject changed numerical generator weights.

    The new fingerprint permits metadata-only edits to the weight archive.
    The branch is newly initialised test state, not a trained accuracy result.
    """
    torch.manual_seed(74)
    generator = GIFTGenerator(cutoff=21, rank=1, state_scale=1.)
    payload = {"format_version": 3, "model_configuration": generator.configuration(),
               "model_state_dict": generator.state_dict()}
    low_path, branch_path = tmp_path / "generator.pt", tmp_path / "branch.pt"
    torch.save(payload, low_path)
    frozen = FixedBandwidthGridGenerator(generator, 64, reference_grid=64)
    branch = HighFrequencyBranch(HighFrequencyConfig(), state_scale=1., high_scale=1.,
                                low_rhs_scale=1., high_rhs_scale=1., frozen_generator=frozen)
    fingerprint = (state_digest(generator.state_dict()) if binding == "generator_weights_sha256"
                   else digest_file(low_path))
    _write_checkpoint(branch_path, branch, seed=74, phase="terminal_epoch", epoch=1,
                      selection_metric=1., **{binding: fingerprint})
    paths = SimpleNamespace(root=Path(__file__).resolve().parents[1], low_model=low_path,
                            require=lambda *args: None)
    low, loaded, _ = load_gift_models(paths, branch_path, 64, torch.device("cpu"))
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in branch.state_dict().items())
    assert all(torch.equal(value, low.model.state_dict()[key]) for key, value in generator.state_dict().items())
    if binding == "generator_weights_sha256":
        payload["note"] = "Metadata edits do not change the generator used by the branch."
        torch.save(payload, low_path)
        load_gift_models(paths, branch_path, 64, torch.device("cpu"))
    first = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"][first] = payload["model_state_dict"][first].clone() + .01
    torch.save(payload, low_path)
    with pytest.raises(ValueError, match="different P21 generator"):
        load_gift_models(paths, branch_path, 64, torch.device("cpu"))
