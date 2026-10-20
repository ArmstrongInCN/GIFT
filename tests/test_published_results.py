"""Compact-package fixtures certify integrity, not measured scientific results."""
import json

import numpy as np
import pytest

from experiments.formal._shared.figure_evidence import file_record, finish_figures
from scripts import audit_repository as audit
from scripts.verify_published_results import verify_package
from scripts.verify_results import verify_result


def package(root, figures=True):
    """Build a compact W2 summary/figure fixture with no measured experiment values."""
    root.mkdir(parents=True)
    (root / "summary").mkdir()
    metric = root / "summary/metrics.csv"
    metric.write_text("fixture_only\nnot_measured\n", encoding="utf-8")
    manifest = dict(schema="gift.published-results.v1", experiment="W2",
        scope="summaries_and_selected_figures", full_raw_outputs_included=False,
        resumable_experiment=False, measurement={"report_sha256": "a" * 64},
        figure_directories=[])
    if figures:
        output = root / "figures"
        output.mkdir()
        (output / "test.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>fixture</text></svg>')
        np.savez(output / "source_fields.npz", field=np.ones((4, 4)))
        finish_figures(output, {"experiment": "W2", "report_sha256": "A" * 64},
            metric, {"fixture_only": True})
        manifest["figure_directories"] = ["figures"]
    commit(root, manifest)
    return manifest


def commit(root, manifest):
    """Refresh fixture file records and write its published-package manifest."""
    manifest["files"] = [file_record(p, root) for p in sorted(root.rglob("*"))
                         if p.is_file() and p.name != "published.json"]
    (root / "published.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("figures", [False, True])
def test_explicit_scope_and_full_run_separation(tmp_path, figures):
    """Accept compact-package integrity while rejecting it as a complete experiment run."""
    root = tmp_path / "package"
    package(root, figures)
    check = verify_package(root, "W2")
    assert check["status"] == "PASS_PUBLISHED_INTEGRITY_ONLY"
    assert not check["full_raw_outputs_verified"] and not check["training_performed"]
    with pytest.raises(FileNotFoundError):
        verify_result(root, "W2")


@pytest.mark.parametrize("change", ["raw_claim", "resumable", "experiment", "hash", "extra",
    "corrupt", "duplicate", "unsafe", "unbound_figure", "arbitrary_archive"])
def test_invalid_package_rejected(tmp_path, change):
    """Reject false scope claims, corrupted inventories and undeclared figure/data files."""
    root = tmp_path / "package"
    manifest = package(root)
    if change == "raw_claim":
        manifest["full_raw_outputs_included"] = True
    elif change == "resumable":
        manifest["resumable_experiment"] = True
    elif change == "experiment":
        manifest["experiment"] = "W1"
    elif change == "hash":
        manifest["measurement"]["report_sha256"] = "invalid"
    elif change == "extra":
        (root / "unlisted.txt").write_text("extra")
    elif change == "corrupt":
        (root / "summary/metrics.csv").write_text("changed")
    elif change == "duplicate":
        manifest["files"] *= 2
    elif change == "unsafe":
        manifest["figure_directories"] = ["../figures"]
    elif change == "unbound_figure":
        manifest["figure_directories"] = []
    else:
        np.savez(root / "summary/training.npz", x=np.ones(8))
        commit(root, manifest)
    (root / "published.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_package(root, "W2")


def test_auditor_allows_only_bound_figure_arrays(tmp_path, monkeypatch):
    """Select only the compact package files associated with its declared figure manifest."""
    relative = "results/formal/W2_cross_resolution"
    package(tmp_path / relative)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    allowed = set(audit.figure_source_allowlist())
    assert relative + "/figures/source_fields.npz" in allowed
    assert relative + "/summary/metrics.csv" in allowed


@pytest.mark.parametrize("value", [np.array([1j]),
                                  np.array(["string"]), np.array([{}], dtype=object)])
def test_nonreal_or_nonnumeric_field_archive_rejected(tmp_path, value):
    """Reject complex, string and object payloads in the compact real-field archive."""
    from scripts.verify_published_results import verify_field_source
    path = tmp_path / "source_fields.npz"
    np.savez(path, x=value)
    with pytest.raises(ValueError):
        verify_field_source(path)


def test_invalid_zip_is_a_validation_failure(tmp_path):
    """Report malformed NPZ bytes as an ordinary figure validation failure."""
    from scripts.verify_published_results import verify_field_source
    path = tmp_path / "source_fields.npz"
    path.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="Invalid figure"):
        verify_field_source(path)


def test_failed_numeric_field_source_is_preserved(tmp_path):
    """Allow numeric NaN/Inf predictions with failure flags so failed panels remain reproducible."""
    from scripts.verify_published_results import verify_field_source
    path = tmp_path / "source_fields.npz"
    np.savez_compressed(path, prediction=np.asarray([np.nan, np.inf], dtype=np.float32),
                        field_finite=np.asarray([False, False]))
    assert verify_field_source(path) is None
