"""Repository audit uses Git's file selection, including worktree-style .git files."""
import shutil
import subprocess

import pytest

from scripts.audit_repository import audit_files, selected_files

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="Git is required")


def git(root, *args):
    """Run Git against the disposable test checkout and capture its output."""
    return subprocess.run([GIT, "-C", str(root), *args], check=True, capture_output=True)


def test_selection_respects_ignore_and_tracked_mode(tmp_path):
    """Select tracked files alone or add unignored local files without changing the index."""
    git(tmp_path, "init", "--quiet")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "model.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("notes\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("local\n", encoding="utf-8")
    git(tmp_path, "add", "--", ".gitignore", "model.py")
    assert selected_files(tmp_path, tracked=True) == [".gitignore", "model.py"]
    assert selected_files(tmp_path) == [".gitignore", "model.py", "notes.md"]


def test_git_directory_file_is_supported(tmp_path):
    """Support a checkout whose .git file points to a separate metadata directory."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    gitdir = tmp_path / "metadata"
    git(checkout, "init", "--quiet", "--separate-git-dir", str(gitdir))
    assert (checkout / ".git").is_file()
    (checkout / "module.py").write_text("pass\n", encoding="utf-8")
    assert selected_files(checkout) == ["module.py"]


def test_syntax_and_data_selection(tmp_path):
    """Report invalid Python and uncatalogued HDF files while accepting ordinary source."""
    (tmp_path / "good.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("if:\n", encoding="utf-8")
    (tmp_path / "observations.h5").write_bytes(b"data")
    failures = audit_files(tmp_path, ["good.py", "bad.py", "observations.h5"], {})
    assert len(failures) == 2
    assert any("syntax" in error for error in failures)
    assert any("Uncatalogued" in error for error in failures)
