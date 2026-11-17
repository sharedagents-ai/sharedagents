"""Bubblewrap confinement for restricted AccessStamps."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sharedagents_agent.folder_sandbox import (
    SandboxRefused,
    narrow_env,
    needs_sandbox,
    runtime_ro_paths,
    sandbox_argv,
)
from sharedagents_core.access_grants import AccessStamp, FolderGrant, Grants


def _stamp(folders):
    grants = Grants.from_obj({"agents": ["jarvis"], "folders": [{"path": f.path, "mode": f.mode} for f in folders]})
    return AccessStamp("alice@example.com", grants)


def test_unrestricted_stamp_is_not_sandboxed():
    assert needs_sandbox(None) is False
    assert needs_sandbox(AccessStamp("a@x", Grants.all())) is False
    argv, env = sandbox_argv(["python", "-m", "sharedagents_agent.tool_runner"], {"K": "v"}, None)
    assert argv == ["python", "-m", "sharedagents_agent.tool_runner"]
    assert env["K"] == "v"


def test_narrow_env_drops_secrets_and_mcp(monkeypatch):
    env = narrow_env(
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "sk-secret",
            "SHAREDAGENTS_SESSION_ID": "abc",
            "SHAREDAGENTS_ACP_MCP_SERVERS": "{}",
            "AWS_SECRET_ACCESS_KEY": "nope",
        }
    )
    assert env["SHAREDAGENTS_SESSION_ID"] == "abc"
    assert "OPENAI_API_KEY" not in env
    assert "SHAREDAGENTS_ACP_MCP_SERVERS" not in env
    assert env["SHAREDAGENTS_FOLDER_SANDBOX"] == "1"


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not installed")
def test_two_users_same_uid_different_folder_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_FOLDER_GRANT_ROOTS", str(tmp_path))
    alice_dir = tmp_path / "alice"
    bob_dir = tmp_path / "bob"
    alice_dir.mkdir()
    bob_dir.mkdir()
    script = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')"
    stamp = _stamp([FolderGrant(str(alice_dir.resolve()), "rw")])
    argv, env = sandbox_argv(
        [sys.executable, "-c", script, str(alice_dir / "out.txt")],
        {"PATH": os.environ.get("PATH", "/usr/bin")},
        stamp,
    )
    subprocess.run(argv, env=env, check=True, timeout=20)
    assert (alice_dir / "out.txt").read_text() == "ok"

    argv_deny, env_deny = sandbox_argv(
        [sys.executable, "-c", script, str(bob_dir / "out.txt")],
        {"PATH": os.environ.get("PATH", "/usr/bin")},
        stamp,
    )
    denied = subprocess.run(argv_deny, env=env_deny, timeout=20)
    assert denied.returncode != 0
    assert not (bob_dir / "out.txt").exists()


def test_restricted_without_bwrap_refuses(monkeypatch):
    monkeypatch.setattr("sharedagents_agent.folder_sandbox.shutil.which", lambda name: None)
    stamp = AccessStamp("a@x", Grants.none())
    with pytest.raises(SandboxRefused):
        sandbox_argv([sys.executable, "-c", "pass"], {"PATH": "/usr/bin"}, stamp)


def test_runtime_ro_paths_bind_only_existing_directories():
    paths = runtime_ro_paths({"PYTHONPATH": ".:src:/usr"})
    assert "." not in paths
    assert "src" not in paths
    assert sys.executable not in paths
    for path in paths:
        assert Path(path).is_absolute()
        assert Path(path).is_dir()


# -- finishing review: what the sandbox confines -------------------


def test_all_folders_is_no_sandbox_and_sharedagents_tools_stay_outside():
    every_folder = AccessStamp(
        "a@x",
        Grants.from_obj(
            {"agents": ["jarvis"], "tools": ["bash"], "folders": "*"},
        ),
    )
    assert needs_sandbox(every_folder, "bash") is False
    no_folders = AccessStamp(
        "a@x",
        Grants.from_obj(
            {"agents": ["jarvis"], "tools": ["bash", "todowrite", "Skill", "AgentSync"]},
        ),
    )
    for tool in (
        "bash",
        "read",
        "write",
        "edit",
        "apply_patch",
        "glob",
        "grep",
        "lsp",
        "mcp__files__write_file",
        "some_future_tool",
    ):
        assert needs_sandbox(no_folders, tool) is True, tool  # host tools, and unknown ones
    for tool in (
        "todowrite",
        "Skill",
        "AgentSync",
        "AgentAsync",
        "compact_context",
        "ExitPlanMode",
        "websearch",
        "webfetch",
    ):
        assert needs_sandbox(no_folders, tool) is False, tool  # sharedagents's own state / network
    argv, env = sandbox_argv(
        ["python", "-m", "sharedagents_agent.tool_runner"], {"K": "v"}, no_folders, tool="todowrite"
    )
    assert argv == ["python", "-m", "sharedagents_agent.tool_runner"] and env["K"] == "v"


def test_confined_tools_have_no_network():
    # A shell with the LAN would reach unauthenticated services (and local MCP
    # servers over HTTP) that act outside the granted folders.
    stamp = AccessStamp("a@x", Grants.from_obj({"agents": ["jarvis"], "tools": ["bash"], "folders": []}))
    if not shutil.which("bwrap"):
        with pytest.raises(SandboxRefused):
            sandbox_argv(["python", "-c", "0"], {}, stamp, tool="bash")
        return
    argv, _env = sandbox_argv(["python", "-c", "0"], {}, stamp, tool="bash")
    assert "--unshare-net" in argv


def test_the_sandbox_mounts_only_the_runtime_it_needs(tmp_path):
    # Not all of /opt, and no PYTHONPATH entry that is not sharedagents's own code.
    import sharedagents_agent

    other = tmp_path / "someone-elses-code"
    other.mkdir()
    own_src = str(Path(sharedagents_agent.__file__).resolve().parents[1])
    paths = runtime_ro_paths({"PYTHONPATH": f"{other}{os.pathsep}{own_src}"})
    assert "/opt" not in paths
    assert str(other) not in paths
    assert own_src in paths


def test_runtime_ro_paths_never_bind_root_or_the_home(monkeypatch):
    # /bin/python3 -> python3.12 on merged-/usr hosts once yielded "/" here,
    # which became --ro-bind / / and exposed the whole host.
    import sharedagents_agent.folder_sandbox as fs

    home = Path.home().resolve()
    monkeypatch.setattr(fs, "_symlink_hop_dirs", lambda _p: ["/", str(home), str(home.parent)])
    paths = runtime_ro_paths({})
    assert "/" not in paths
    assert str(home) not in paths
    assert str(home.parent) not in paths


def test_symlink_hops_stop_below_the_filesystem_root(tmp_path):
    from sharedagents_agent.folder_sandbox import _symlink_hop_dirs

    root_link = tmp_path / "python3"
    root_link.symlink_to(Path(tmp_path.anchor) / "python3.12")
    assert tmp_path.anchor not in _symlink_hop_dirs(str(root_link))
