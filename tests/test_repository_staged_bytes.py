"""Real temporary Git indexes; no project Git operations, commits or deletion.

Each fixture starts in a new pytest directory and is retained. The checker is
read-only: its call must leave both index and existing work-file bytes intact.
"""
import shutil
import subprocess

import pytest

from scripts.audit_repository import staged_byte_failures


GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="Git is required for staged-byte fixture tests")


@pytest.fixture
def git_environment(monkeypatch, tmp_path):
    # Ignore machine/global Git attributes/config and inherited repository targets.
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_COUNT"):
        monkeypatch.delenv(name, raising=False)
    empty_config = tmp_path / "empty-global-config"
    empty_config.write_bytes(b"")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))


def git(repo, *arguments, check=True):
    return subprocess.run(
        [GIT, "-C", str(repo), "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
    )


def new_repo(tmp_path, object_format="sha1"):
    repo = tmp_path / ("repo-" + object_format)
    repo.mkdir(exist_ok=False)
    result = git(repo, "init", "--quiet", "--object-format=" + object_format, check=False)
    if result.returncode and object_format == "sha256":
        error = result.stderr.decode("utf-8", errors="replace").lower()
        if any(marker in error for marker in ("unknown option", "unknown hash", "unsupported", "not supported")):
            pytest.skip("Installed Git does not support SHA256 repositories")
    result.check_returncode()
    return repo


def work_bytes(repo):
    return {str(path.relative_to(repo)): path.read_bytes() for path in repo.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(repo).parts}


def checked(repo, files):
    index = repo / ".git/index"
    before_index = index.read_bytes()
    before_work = work_bytes(repo)
    result = staged_byte_failures(repo, files)
    assert index.read_bytes() == before_index
    assert work_bytes(repo) == before_work
    return result


@pytest.mark.parametrize("name,raw", [
    ("metrics.csv", b"name,value\r\nnu,0.03\r\n"),
    ("manifest.json", b'{\r\n  "complete": false\r\n}\r\n'),
])
def test_crlf_normalized_staging_is_detected(tmp_path, git_environment, name, raw):
    repo = new_repo(tmp_path)
    (repo / ".gitattributes").write_bytes(b"* text=auto\n*.csv text eol=lf\n*.json text eol=lf\n")
    (repo / name).write_bytes(raw)
    git(repo, "add", "--", ".gitattributes", name)
    assert (repo / name).read_bytes() == raw
    assert git(repo, "show", ":" + name).stdout == raw.replace(b"\r\n", b"\n")
    assert checked(repo, [".gitattributes", name]) == [
        "Staged bytes differ from audited working file: " + name]


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
def test_minus_text_preserves_crlf_and_binary_bytes(tmp_path, git_environment, object_format):
    repo = new_repo(tmp_path, object_format)
    payloads = {".gitattributes": b"* -text\n", "metrics.csv": b"a,b\r\n1,2\r\n",
                "manifest.json": b'{\r\n"ok": true\r\n}\r\n',
                "reference.md": b"unchanged reference\r\n", "numeric.bin": bytes(range(256))}
    for name, raw in payloads.items():
        (repo / name).write_bytes(raw)
    git(repo, "add", "--", *payloads)
    assert git(repo, "rev-parse", "--show-object-format").stdout.strip() == object_format.encode()
    for name, raw in payloads.items():
        assert git(repo, "show", ":" + name).stdout == raw
    assert checked(repo, list(payloads)) == []


def test_dirty_working_bytes_are_detected(tmp_path, git_environment):
    repo = new_repo(tmp_path)
    (repo / "metrics.csv").write_bytes(b"old,value\n")
    git(repo, "add", "--", "metrics.csv")
    (repo / "metrics.csv").write_bytes(b"new,value\n")
    assert checked(repo, ["metrics.csv"]) == [
        "Staged bytes differ from audited working file: metrics.csv"]
    assert git(repo, "show", ":metrics.csv").stdout == b"old,value\n"


def test_missing_working_file_is_detected_and_retained(tmp_path, git_environment):
    repo = new_repo(tmp_path)
    source = repo / "metrics.csv"
    source.write_bytes(b"retained,value\n")
    git(repo, "add", "--", "metrics.csv")
    # Move only this newly created fixture file; preserve it outside the repo.
    retained = tmp_path / "retained-metrics.csv"
    assert not retained.exists() and source.parent == repo
    source.rename(retained)
    assert checked(repo, ["metrics.csv"]) == [
        "Cannot verify staged bytes for unsafe/missing file: metrics.csv"]
    assert retained.read_bytes() == b"retained,value\n"


@pytest.mark.parametrize("mismatch", ["audited_subset", "unstaged_extra"])
def test_staged_inventory_mismatch_is_detected(tmp_path, git_environment, mismatch):
    repo = new_repo(tmp_path)
    for name in ("a.csv", "b.json"):
        (repo / name).write_bytes(b"fixture\n")
    if mismatch == "audited_subset":
        git(repo, "add", "--", "a.csv", "b.json")
        failures = checked(repo, ["a.csv"])
        assert failures == ["Staged file inventory differs from audited selection"]
    else:
        git(repo, "add", "--", "a.csv")
        failures = checked(repo, ["a.csv", "b.json"])
        assert failures == ["Staged file inventory differs from audited selection",
                            "Staged bytes differ from audited working file: b.json"]
