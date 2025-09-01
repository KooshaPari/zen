import pytest

from utils.scope_utils import WorkDirError, get_repo_root, validate_and_normalize_work_dir


def test_validate_valid_subdir(tmp_path, monkeypatch):
    # Simulate repo root as tmp_path
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "frontend" / "ui"
    sub.mkdir(parents=True)

    rel, abs_path = validate_and_normalize_work_dir("frontend/ui")
    assert rel == "frontend/ui"
    assert abs_path == str(sub.resolve())


def test_reject_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkDirError):
        validate_and_normalize_work_dir("/etc")


def test_reject_traversal(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkDirError):
        validate_and_normalize_work_dir("../outside")


def test_warn_only_defaults(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    # No directory created to trigger error in server logic (will use utils separately here)
    # The warn-only behavior is applied in server code, not in utils; utils always raises.
    # So this test ensures utils raises when path missing
    with pytest.raises(WorkDirError):
        validate_and_normalize_work_dir("missing_dir")


def test_get_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert get_repo_root() == str(tmp_path)

