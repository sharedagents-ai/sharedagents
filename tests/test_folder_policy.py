"""Approved-root validation for per-user folder grants."""

from pathlib import Path

import pytest

from sharedagents_core.access_grants import FolderGrant
from sharedagents_core.folder_policy import FolderPolicyError, normalize_folder_grants


def test_normalize_accepts_non_overlapping_roots_under_approved(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_FOLDER_GRANT_ROOTS", str(tmp_path))
    a = tmp_path / "alice"
    b = tmp_path / "bob"
    a.mkdir()
    b.mkdir()
    got = normalize_folder_grants(
        [
            FolderGrant(str(a), "rw"),
            FolderGrant(str(b), "ro"),
        ]
    )
    assert {(g.path, g.mode) for g in got} == {(str(a.resolve()), "rw"), (str(b.resolve()), "ro")}


def test_normalize_rejects_home_system_overlap_and_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_FOLDER_GRANT_ROOTS", str(tmp_path))
    nested = tmp_path / "proj"
    nested.mkdir()
    with pytest.raises(FolderPolicyError):
        normalize_folder_grants([FolderGrant("/", "rw")])
    with pytest.raises(FolderPolicyError):
        normalize_folder_grants([FolderGrant(str(Path.home()), "ro")])
    with pytest.raises(FolderPolicyError):
        normalize_folder_grants([FolderGrant("relative/path", "rw")])
    with pytest.raises(FolderPolicyError):
        normalize_folder_grants(
            [
                FolderGrant(str(tmp_path), "rw"),
                FolderGrant(str(nested), "ro"),
            ]
        )
    with pytest.raises(FolderPolicyError):
        normalize_folder_grants([FolderGrant("/etc", "ro")])
