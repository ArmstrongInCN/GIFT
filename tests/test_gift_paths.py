"""Input location is independent of output overwrite protection."""
import h5py
import pytest

from gift.paths import checkpoint_root, data_root, external_root
from training.configuration import output_path


def test_project_local_inputs_can_be_read_while_outputs_are_protected(tmp_path, monkeypatch):
    """Allow explicit local inputs while preventing outputs from overwriting them."""
    project = tmp_path / "project"
    data = project / "data"
    external = project / "vendor"
    data.mkdir(parents=True)
    external.mkdir()
    monkeypatch.setenv("GIFT_DATA_ROOT", str(data))
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", str(external))
    with h5py.File(data / "observations.h5", "x") as handle:
        handle.create_dataset("values", data=[1., 2.])
    with h5py.File(data_root(project) / "observations.h5", "r") as handle:
        assert handle["values"][:].tolist() == [1., 2.]
    assert external_root(project) == external
    for destination in (project / "src" / "run", data / "run", external / "run"):
        with pytest.raises(ValueError, match="overwrite"):
            output_path(destination, project)
    assert output_path(project / "runs" / "new", project) == project / "runs" / "new"


def test_plan_paths_need_no_existing_directory(tmp_path, monkeypatch):
    """Resolve configured paths during planning and report only missing configuration."""
    monkeypatch.setenv("GIFT_DATA_ROOT", str(tmp_path / "missing_data"))
    monkeypatch.setenv("GIFT_EXTERNAL_ROOT", str(tmp_path / "missing_sources"))
    assert data_root(tmp_path) == tmp_path / "missing_data"
    assert external_root(tmp_path) == tmp_path / "missing_sources"
    monkeypatch.delenv("GIFT_CHECKPOINT_ROOT", raising=False)
    assert checkpoint_root(tmp_path) == tmp_path / "artifacts"
    monkeypatch.setenv("GIFT_CHECKPOINT_ROOT", str(tmp_path / "models"))
    assert checkpoint_root(tmp_path) == tmp_path / "models"
    monkeypatch.delenv("GIFT_DATA_ROOT")
    with pytest.raises(FileNotFoundError, match="Set GIFT_DATA_ROOT"):
        data_root(tmp_path)
