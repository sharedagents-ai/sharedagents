"""apply_patch must honor a path-scoped agent's write roots the same way
write/edit do, and a delegated agent's tool calls must resolve those roots
against the folder it was dispatched to.

Regression for the 2026-10-07 "Redis cluster configuration discovery" session:
a ``build`` agent dispatched to a git worktree had ``apply_patch`` refused on a
file inside that worktree, while ``edit`` on the same file was allowed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sharedagents_agent.tools.permissions import (
    apply_patch_target_denied,
    write_scope_check,
)
from sharedagents_gateway.tool_execution import _session_workspace_cwd

BUILD_ROOTS = (".",)


def _worktree(tmp_path: Path) -> Path:
    """A linked worktree: ``.git`` is a file, as ``git worktree add`` leaves it."""
    repo = tmp_path / "repo" / ".worktrees" / "fix_topic"
    (repo / "plan").mkdir(parents=True)
    (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/fix_topic\n")
    return repo


def test_apply_patch_follows_the_targets_own_repository(tmp_path):
    repo = _worktree(tmp_path)
    target = str(repo / "plan" / "runbook.md")
    elsewhere = tmp_path / "service-cwd"
    elsewhere.mkdir()

    for base in (str(elsewhere), ""):
        allowed, _ = write_scope_check(
            "edit",
            {"filePath": target},
            write_roots=BUILD_ROOTS,
            base_dir=base,
        )
        denied, reason = apply_patch_target_denied(
            target,
            write_roots=BUILD_ROOTS,
            base_dir=base,
        )
        assert allowed
        assert not denied, reason


def test_apply_patch_still_refuses_a_target_outside_every_root(tmp_path, monkeypatch):
    from sharedagents_agent.tools import permissions

    # This case models a directory outside any repository, even when pytest's
    # temporary directory is itself nested under a development checkout.
    monkeypatch.setattr(permissions, "_git_repo_root", lambda target: "")
    base = tmp_path / "workspace"
    base.mkdir()
    loose = tmp_path / "not-a-repo"
    loose.mkdir()

    denied, reason = apply_patch_target_denied(
        str(loose / "notes.md"),
        write_roots=BUILD_ROOTS,
        base_dir=str(base),
    )
    assert denied
    assert "write roots" in reason


def test_apply_patch_does_not_anchor_on_a_repository_nested_in_the_workspace(tmp_path):
    base = tmp_path / "workspace"
    nested = base / "vendor" / "nested"
    (nested / "plan").mkdir(parents=True)
    (nested / ".git").mkdir()

    denied, _ = apply_patch_target_denied(
        str(nested / "plan" / "x.md"),
        write_roots=("plan/",),
        base_dir=str(base),
    )
    assert denied


def test_empty_write_roots_refuse_even_inside_a_repository(tmp_path):
    repo = _worktree(tmp_path)

    denied, _ = apply_patch_target_denied(
        str(repo / "plan" / "runbook.md"),
        write_roots=(),
        base_dir=str(repo),
    )
    assert denied


def test_workspace_cwd_comes_from_the_live_session_table():
    sess = SimpleNamespace(workspace_cwd="/work/repo")
    backend = SimpleNamespace(_active_sessions={"s1": sess})

    assert _session_workspace_cwd(backend, "s1") == "/work/repo"
    assert _session_workspace_cwd(backend, "missing") == ""


def test_workspace_cwd_is_found_for_a_delegated_agent():
    """A delegated agent runs outside the bridge: its backend only exposes
    ``_session_for``. The dispatch folder must still reach the tool runner."""
    sess = SimpleNamespace(workspace_cwd="/work/repo/.worktrees/fix_topic")

    class _Shim:
        def _session_for(self, sid):
            return sess if sid == "child" else None

    assert _session_workspace_cwd(_Shim(), "child") == "/work/repo/.worktrees/fix_topic"
    assert _session_workspace_cwd(_Shim(), "other") == ""
    assert _session_workspace_cwd(object(), "child") == ""
