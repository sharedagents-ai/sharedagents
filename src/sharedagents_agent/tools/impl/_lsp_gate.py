"""Post-write LSP diagnostics feedback — opencode edit.ts/write.ts/
apply_patch.ts parity (soft gate), backed by a persistent per-workspace
daemon (opencode lsp.ts singleton + lazy spawn).

Also owns the production path for the read ``lsp`` tool: every definition /
references / hover / … call rides the same warm daemon instead of spawning
a one-shot basedpyright per request.

Architecture: the gate runs inside the per-call tool_runner subprocess,
which dies after each tool call — so the language server cannot live here.
:mod:`_lsp_daemon` owns the persistent LspClient; this module ensures the
daemon is up (first-write cost = one index) and does cheap socket
round-trips afterwards.

Server resolution (fixes the inert-gate PATH bug): ``SHAREDAGENTS_LSP_SERVER``
→ ``shutil.which`` with ``PATH`` extended by ``~/.local/bin`` → documented
absolute pipx fallbacks. Soft-gate when nothing resolves.

Session wall-clock deadline (``_WRITE_LSP_TIMEOUT_S``, default 10s) covers
cold daemon spawn + server index + every diagnostics poll for every file.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat
from sharedagents_core.cwdutil import safe_getcwd

from ._lsp_daemon import is_valid_workspace, paths_for, socket_dir
from ._utils import MAX_OUTPUT_BYTES, _truncate

# opencode lsp/diagnostic.ts caps: severity-1 (ERROR) only, 20 per file.
_MAX_DIAGNOSTICS_PER_FILE = 20
# SESSION-wide wall-clock budget for the WHOLE write gate (daemon ensure +
# every diagnostics round-trip for every file in the call).
_WRITE_LSP_TIMEOUT_S = 10.0
# Match LspClient defaults / OpenCode wait windows (was 2×0.2 — too short
# for a warm basedpyright re-analyse after didChange).
_DIAGNOSTICS_ROUNDS = 8
_DIAGNOSTICS_DELAY_S = 0.35

_HEADER = "\n\nLSP errors detected in this file, please fix:\n"

# Documented absolute pipx / user-local fallbacks when PATH is stripped
# (tool_runner often gets ``/usr/local/sbin:...`` without ``~/.local/bin``).
_ABSOLUTE_SERVER_FALLBACKS = (
    (os.path.expanduser("~/.local/bin/basedpyright-langserver"), ["--stdio"]),
    (os.path.expanduser("~/.local/bin/pyright-langserver"), ["--stdio"]),
    (os.path.expanduser("~/.local/bin/pylsp"), []),
)


def _is_python(path: str) -> bool:
    return Path(path).suffix == ".py"


def _lsp_root() -> str:
    # Same base as the dispatch write-scope gate (runners.py:221):
    # SHAREDAGENTS_WORKSPACE_CWD, else the process cwd.
    return os.environ.get("SHAREDAGENTS_WORKSPACE_CWD") or safe_getcwd()


def resolve_server_cmd_extended(
    explicit: list[str] | None = None,
) -> list[str] | None:
    """Resolve a language-server argv with user-local PATH + abs fallbacks.

    Calls ``lsp_client.resolve_server_cmd`` under a PATH that includes
    ``~/.local/bin`` so pipx installs are visible even when the tool_runner
    env stripped them. Absolute path fallbacks cover the remaining case.
    PATH mutation is scoped to this call (restored in ``finally``).
    """
    from sharedagents_agent import lsp_client as _lsp

    local_bin = os.path.expanduser("~/.local/bin")
    old_path = os.environ.get("PATH", "")
    parts = old_path.split(os.pathsep) if old_path else []
    if local_bin not in parts:
        os.environ["PATH"] = local_bin + (os.pathsep + old_path if old_path else "")
    try:
        cmd = _lsp.resolve_server_cmd(explicit)
        if cmd:
            return cmd
    finally:
        os.environ["PATH"] = old_path

    for exe, args in _ABSOLUTE_SERVER_FALLBACKS:
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return [exe, *args]
    return None


def _socket_peer_ok(sock_path: Path) -> bool:
    """True when the socket path is a socket owned by this uid (mode ≤ 0770)."""
    try:
        st = sock_path.lstat()
    except OSError:
        return False
    if not stat.S_ISSOCK(st.st_mode):
        return False
    if st.st_uid != os.getuid():
        return False
    # Reject world-writable sockets.
    return not st.st_mode & 2


def _socket_alive(sock_path: Path) -> bool:
    """True when a daemon is accepting on ``sock_path``.

    Connect-only (no ping): the daemon serves connections inline on its
    accept loop, so a ping can time out while a diagnostics request is
    in flight and falsely look "dead" under concurrent ensure/collect.

    Only ``ECONNREFUSED`` unlinks a stale socket path. Timeout / backlog
    errors are treated as "alive" so we never spawn a replacement over a
    live daemon.
    """
    if not sock_path.exists():
        return False
    if not _socket_peer_ok(sock_path):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            s.connect(str(sock_path))
            return True
        finally:
            s.close()
    except ConnectionRefusedError:
        try:
            sock_path.unlink()
        except OSError:
            pass
        return False
    except OSError as exc:
        err = getattr(exc, "errno", None)
        if err == errno.ECONNREFUSED:
            try:
                sock_path.unlink()
            except OSError:
                pass
            return False
        # Overloaded / timed out / permission — do NOT unlink; treat as live
        # when the path still looks like our socket, else dead.
        return err in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ETIMEDOUT, errno.EINTR)


def _daemon_python() -> str:
    return sys.executable


def _spawn_daemon(
    workspace_root: str,
    *,
    sock_path: Path,
    server_cmd: list[str] | None,
    lock_fd: int,
) -> None:
    """Start the daemon process; transfer the locked FD for lifetime ownership."""
    env = os.environ.copy()
    local_bin = os.path.expanduser("~/.local/bin")
    path = env.get("PATH", "")
    if local_bin not in path.split(os.pathsep):
        env["PATH"] = local_bin + os.pathsep + path
    src_root = str(Path(__file__).resolve().parents[3])
    # parents: impl -> tools -> sharedagents_agent -> src
    pp = env.get("PYTHONPATH", "")
    if src_root not in pp.split(os.pathsep):
        env["PYTHONPATH"] = src_root + (os.pathsep + pp if pp else "")
    env["SHAREDAGENTS_LSP_LOCK_FD"] = str(lock_fd)

    argv = [
        _daemon_python(),
        "-m",
        "sharedagents_agent.tools.impl._lsp_daemon",
        "--workspace-root",
        os.path.abspath(workspace_root),
        "--sock",
        str(sock_path),
    ]
    if server_cmd:
        argv.extend(["--server-cmd", json.dumps(list(server_cmd))])
    subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        pass_fds=(lock_fd,),
    )


def ensure_daemon(
    workspace_root: str,
    *,
    server_cmd: list[str] | None = None,
    deadline: float | None = None,
) -> Path | None:
    """Return the live socket path, spawning the daemon if needed.

    Race-safe: flock on the lockfile → re-check socket → spawn, transferring
    the locked FD to the child for lifetime ownership. Returns None if the
    server cannot be resolved, the workspace root is invalid, or the
    deadline expires before the socket accepts.
    """
    if oscompat.IS_WINDOWS:
        return None  # Unix-socket LSP remains outside the Windows port.
    import fcntl

    if not is_valid_workspace(workspace_root):
        return None
    root = os.path.abspath(workspace_root)
    sock_path, lock_path = paths_for(root)
    socket_dir()  # ensure dir exists

    if _socket_alive(sock_path):
        return sock_path

    cmd = server_cmd or resolve_server_cmd_extended()
    if not cmd:
        return None

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    transferred = False
    try:
        # Non-blocking flock with deadline (Codex I1) — never sit forever.
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                time.sleep(0.05)
        if _socket_alive(sock_path):
            return sock_path
        if deadline is not None and time.monotonic() >= deadline:
            return None
        _spawn_daemon(
            root,
            sock_path=sock_path,
            server_cmd=cmd,
            lock_fd=fd,
        )
        # Child owns the lock via the inherited FD duplicate. Closing our
        # copy does not unlock while the child's FD remains open.
        transferred = True
        while True:
            if _socket_alive(sock_path):
                return sock_path
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.05)
    finally:
        if not transferred:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass


def _rpc(
    sock_path: Path,
    request: dict,
    *,
    deadline: float | None = None,
) -> dict | None:
    """One request/response over the Unix socket. None on any failure."""
    try:
        remaining = 5.0
        if deadline is not None:
            remaining = max(0.01, deadline - time.monotonic())
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(remaining)
            s.connect(str(sock_path))
            s.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    return None
                buf += chunk
            line = buf.split(b"\n", 1)[0]
            resp = json.loads(line.decode("utf-8"))
            return resp if isinstance(resp, dict) else None
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        return None


def daemon_lsp_op(
    root: str,
    op: str,
    file: str,
    *,
    line: int | None = None,
    character: int | None = None,
    query: str | None = None,
    timeout: float = 20.0,
    server_cmd: list[str] | None = None,
) -> str:
    """Run an OpenCode ``lsp`` tool op on the warm daemon; return formatted text.

    Raises ``LspUnavailable`` on hard failure so ``integrations.run_lsp`` can
    surface ``ok=False`` the same way a missing server does.
    """
    from sharedagents_agent.lsp_client import LspUnavailable

    if not is_valid_workspace(root):
        raise LspUnavailable(
            f"invalid workspace root for LSP: {root!r} (set SHAREDAGENTS_WORKSPACE_CWD to the project directory)"
        )
    deadline = time.monotonic() + timeout
    sock = ensure_daemon(root, server_cmd=server_cmd, deadline=deadline)
    if sock is None:
        raise LspUnavailable(
            "no language server found (install basedpyright/pyright/pylsp or set SHAREDAGENTS_LSP_SERVER)"
        )
    req: dict[str, Any] = {
        "op": op,
        "file": str(Path(file).resolve()) if not os.path.isabs(file) else file,
        "workspace_root": os.path.abspath(root),
    }
    if op == "diagnostics":
        req["as_tool"] = True
    if line is not None:
        req["line"] = int(line)
    if character is not None:
        req["character"] = int(character)
    if query is not None:
        req["query"] = query
    resp = _rpc(sock, req, deadline=deadline)
    if not resp:
        raise LspUnavailable("LSP daemon RPC failed")
    if not resp.get("ok"):
        raise LspUnavailable(str(resp.get("error") or "LSP daemon error"))
    return str(resp.get("result") or "")


def _collect_diagnostics(
    paths: list[str],
    *,
    root: str,
    server_cmd: list[str] | None,
    timeout: float,
) -> dict[str, Any]:
    """Fetch diagnostics for ``paths`` via the daemon.

    Returns ``{path: list|None}``. Soft failures omit the path. Extracted
    so unit tests can monkeypatch without standing up a real daemon.
    """
    if not is_valid_workspace(root):
        return {}
    deadline = time.monotonic() + timeout
    sock = ensure_daemon(root, server_cmd=server_cmd, deadline=deadline)
    if sock is None:
        return {}
    out: dict[str, Any] = {}
    for p in paths:
        if time.monotonic() >= deadline:
            break
        resp = _rpc(
            sock,
            {
                "op": "diagnostics",
                "file": p,
                "workspace_root": os.path.abspath(root),
                "rounds": _DIAGNOSTICS_ROUNDS,
                "delay": _DIAGNOSTICS_DELAY_S,
            },
            deadline=deadline,
        )
        if not resp or not resp.get("ok"):
            continue
        out[p] = resp.get("diagnostics")
    return out


def _format_block(path: str, diags: Any) -> str:
    if not diags:
        return ""
    errors = [d for d in diags if (d.get("severity") or 1) == 1]
    lines = []
    for d in errors[:_MAX_DIAGNOSTICS_PER_FILE]:
        rng = (d.get("range") or {}).get("start", {})
        ln = rng.get("line", 0) + 1
        col = rng.get("character", 0) + 1
        lines.append(f'<diagnostics file="{path}">ERROR [{ln}:{col}] {d.get("message", "")}</diagnostics>')
    if not lines:
        return ""
    return _HEADER + "\n".join(lines)


def lsp_errors_block(
    paths: list[str],
    *,
    root: str | None = None,
    server_cmd: list[str] | None = None,
    timeout: float = _WRITE_LSP_TIMEOUT_S,
) -> str:
    """Return the opencode diagnostics block for the written ``paths``, or "".

    Only ``.py`` paths are checked. Severity-1 (ERROR) only — a missing
    ``severity`` counts as ERROR (LSP default) — max 20 per file, 1-based
    ``line:col``. Soft gate: never raises; any failure → "".
    """
    try:
        py_paths = [p for p in paths if _is_python(p)]
        if not py_paths:
            return ""
        ws = root or _lsp_root()
        if not is_valid_workspace(ws):
            return ""
        # Fast degrade when nothing is resolvable (no spawn attempt).
        if server_cmd is None and resolve_server_cmd_extended() is None:
            return ""
        by_path = _collect_diagnostics(
            py_paths,
            root=ws,
            server_cmd=server_cmd,
            timeout=timeout,
        )
        blocks: list[str] = []
        for p in py_paths:
            try:
                block = _format_block(p, by_path.get(p))
                if block:
                    blocks.append(block)
            except Exception:  # noqa: BLE001 — one bad file must not wipe others
                continue
        block = "".join(blocks)
        if len(block.encode("utf-8")) > MAX_OUTPUT_BYTES:
            block, _ = _truncate(block)
        return block
    except Exception:  # noqa: BLE001 — soft gate: never fail the write
        return ""


def lsp_touch_async(path: str, *, root: str | None = None) -> None:
    """Fire-and-forget read pre-warm (opencode read.ts:119 touchFile parity).

    Spawns a detached helper process so the warm survives the tool_runner's
    ``os._exit(0)`` (a daemon thread inside the runner is killed with it).
    """
    if oscompat.IS_WINDOWS:
        return
    if not _is_python(path):
        return
    ws = root or _lsp_root()
    if not is_valid_workspace(ws):
        return
    if resolve_server_cmd_extended() is None:
        return
    src_root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    if src_root not in pp.split(os.pathsep):
        env["PYTHONPATH"] = src_root + (os.pathsep + pp if pp else "")
    local_bin = os.path.expanduser("~/.local/bin")
    path_env = env.get("PATH", "")
    if local_bin not in path_env.split(os.pathsep):
        env["PATH"] = local_bin + os.pathsep + path_env
    argv = [
        _daemon_python(),
        "-m",
        "sharedagents_agent.tools.impl._lsp_gate",
        "--touch",
        str(Path(path).resolve()),
        "--workspace-root",
        os.path.abspath(ws),
    ]
    try:
        subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:  # noqa: BLE001
        return


def _touch_main(path: str, workspace_root: str) -> int:
    """Helper entry for ``--touch`` (detached pre-warm)."""
    try:
        if not is_valid_workspace(workspace_root):
            return 0
        deadline = time.monotonic() + _WRITE_LSP_TIMEOUT_S
        sock = ensure_daemon(workspace_root, deadline=deadline)
        if sock is None:
            return 0
        _rpc(
            sock,
            {"op": "touch", "file": str(Path(path).resolve())},
            deadline=deadline,
        )
    except Exception:  # noqa: BLE001
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="sharedagents LSP gate helpers")
    p.add_argument("--touch", default=None, help="pre-warm path (detached)")
    p.add_argument("--workspace-root", default=None)
    args = p.parse_args(argv)
    if args.touch:
        return _touch_main(args.touch, args.workspace_root or _lsp_root())
    p.error("nothing to do (pass --touch)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
