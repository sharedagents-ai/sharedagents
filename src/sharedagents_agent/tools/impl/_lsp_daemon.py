"""Persistent per-workspace LSP daemon (opencode lsp.ts singleton parity).

One daemon process per workspace root owns ONE basedpyright/pyright/pylsp
session via :class:`sharedagents_agent.lsp_client.LspClient`. Tool-runner
subprocesses (which die after each call) talk to it over a Unix socket so
index cost is paid once, not on every write / ``lsp`` tool call.

Wire protocol — line-delimited JSON over a Unix domain socket
--------------------------------------------------------------
Request (client → daemon), one JSON object per line::

    {"op": "diagnostics", "file": "/abs/path.py", "workspace_root": "/ws"}
    {"op": "touch", "file": "/abs/path.py"}
    {"op": "goToDefinition", "file": "...", "line": 1, "character": 1}
    {"op": "ping"}

Response (daemon → client), one JSON object per line::

    {"ok": true, "diagnostics": [ ... LSP Diagnostic dicts ... ]}
    {"ok": true, "result": "formatted text"}   # tool ops
    {"ok": true}                               # touch / ping
    {"ok": false, "error": "message"}

``diagnostics`` returns the raw publishDiagnostics list (or ``null`` when
no *fresh* publish arrived — same semantics as ``LspClient.diagnostics``).
Filtering/formatting for the write soft-gate stays in ``_lsp_gate``.

Paths / env
-----------
* Socket dir: ``$SHAREDAGENTS_LSP_SOCKET_DIR`` or ``$XDG_RUNTIME_DIR/sharedagents-lsp``
  or ``/tmp/sharedagents-lsp-<uid>`` (never bare shared ``/tmp``)
* Socket: ``lsp-<sha1(workspace_root)[:16]>.sock`` (mode 0600)
* Lock:   same dir, ``lsp-<hash>.lock`` — held by the daemon for its lifetime
  (parent passes the locked FD via ``SHAREDAGENTS_LSP_LOCK_FD``)
* Idle:   ``SHAREDAGENTS_LSP_IDLE_S`` (default 300) — no requests → graceful exit
* Server: ``SHAREDAGENTS_LSP_SERVER`` / ``--server-cmd`` (JSON argv preferred)

Workspace root ``/`` (or empty) is rejected — a root-wide daemon would share
one index across every session.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import socket
import sys
import threading
import traceback
from hashlib import sha1
from pathlib import Path
from typing import Any

DEFAULT_IDLE_S = 300.0
_DIAG_ROUNDS = 8
_DIAG_DELAY_S = 0.35
_MAX_ROUNDS = 10
_MAX_DELAY_S = 1.0
_MAX_REQ_BYTES = 1 << 20  # 1 MiB request frame cap

# Tool ops that return formatted ``result`` text (OpenCode tool/lsp.ts vocabulary
# plus diagnostics for the unused-but-present tool body).
_TOOL_OPS = frozenset(
    {
        "goToDefinition",
        "findReferences",
        "hover",
        "documentSymbol",
        "workspaceSymbol",
        "goToImplementation",
        "prepareCallHierarchy",
        "incomingCalls",
        "outgoingCalls",
        "diagnostics",
    }
)


def is_valid_workspace(workspace_root: str) -> bool:
    """Reject filesystem-root / empty workspaces."""
    if not workspace_root or not isinstance(workspace_root, str):
        return False
    abs_root = os.path.abspath(workspace_root)
    if abs_root in ("/", ""):
        return False
    return os.path.isdir(abs_root)


def socket_dir() -> Path:
    raw = os.environ.get("SHAREDAGENTS_LSP_SOCKET_DIR")
    if raw:
        p = Path(raw)
    elif os.environ.get("XDG_RUNTIME_DIR"):
        p = Path(os.environ["XDG_RUNTIME_DIR"]) / "sharedagents-lsp"
    else:
        p = Path(f"/tmp/sharedagents-lsp-{os.getuid()}")
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p


def workspace_hash(workspace_root: str) -> str:
    return sha1(os.path.abspath(workspace_root).encode("utf-8")).hexdigest()[:16]


def paths_for(workspace_root: str) -> tuple[Path, Path]:
    h = workspace_hash(workspace_root)
    d = socket_dir()
    return d / f"lsp-{h}.sock", d / f"lsp-{h}.lock"


def idle_timeout_s() -> float:
    raw = os.environ.get("SHAREDAGENTS_LSP_IDLE_S")
    if raw is None or raw == "":
        return DEFAULT_IDLE_S
    try:
        return max(0.05, float(raw))
    except ValueError:
        return DEFAULT_IDLE_S


class LspDaemon:
    """Serve diagnostics/touch/tool ops over a Unix socket; own one LspClient."""

    def __init__(
        self,
        workspace_root: str,
        *,
        server_cmd: list[str] | None = None,
        idle_s: float | None = None,
        sock_path: Path | None = None,
        lock_file: Any = None,
    ):
        if not is_valid_workspace(workspace_root):
            raise ValueError(f"invalid workspace root for LSP daemon: {workspace_root!r}")
        self.workspace_root = os.path.abspath(workspace_root)
        self.server_cmd = server_cmd
        self.idle_s = idle_s if idle_s is not None else idle_timeout_s()
        self.sock_path = sock_path or paths_for(self.workspace_root)[0]
        self._lock_file = lock_file  # held for lifetime when inherited
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._stop = threading.Event()
        self.ready = threading.Event()
        self._listen: socket.socket | None = None
        self._sock_identity: tuple[int, int] | None = None  # (st_dev, st_ino)

    # -- LSP client lifecycle ------------------------------------------- #
    def _ensure_client(self):
        from sharedagents_agent import lsp_client as _lsp

        with self._client_lock:
            c = self._client
            if c is not None and getattr(c, "_proc", None) is not None:
                proc = c._proc
                if proc.poll() is None:
                    return c
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
                self._client = None

            c = _lsp.LspClient(
                self.workspace_root,
                server_cmd=self.server_cmd,
                timeout=20.0,
            )
            c.start()
            self._client = c
            return c

    def _close_client(self) -> None:
        with self._client_lock:
            c = self._client
            self._client = None
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass

    def _clamp_poll(self, rounds: Any, delay: Any) -> tuple[int, float]:
        try:
            r = int(rounds) if rounds is not None else _DIAG_ROUNDS
        except (TypeError, ValueError):
            r = _DIAG_ROUNDS
        try:
            d = float(delay) if delay is not None else _DIAG_DELAY_S
        except (TypeError, ValueError):
            d = _DIAG_DELAY_S
        return max(1, min(r, _MAX_ROUNDS)), max(0.0, min(d, _MAX_DELAY_S))

    def _tool_result(self, op: str, req: dict) -> dict:
        from sharedagents_agent import lsp_client as L

        f = req.get("file")
        if not isinstance(f, str) or not f:
            return {"ok": False, "error": "file required"}
        abs_f = os.path.abspath(f)
        # Keep requests inside the workspace (prefix + separator).
        root_prefix = self.workspace_root.rstrip(os.sep) + os.sep
        if abs_f != self.workspace_root and not abs_f.startswith(root_prefix):
            return {"ok": False, "error": "file outside workspace_root"}

        try:
            c = self._ensure_client()
            line = req.get("line")
            character = req.get("character")
            if op in (
                "goToDefinition",
                "findReferences",
                "hover",
                "goToImplementation",
                "prepareCallHierarchy",
                "incomingCalls",
                "outgoingCalls",
            ):
                if line is None or character is None:
                    return {"ok": False, "error": "line and character required"}
                # Wire is 1-based; LspClient methods take 0-based.
                ln = int(line) - 1
                ch = int(character) - 1

            if op == "goToDefinition":
                text = L.format_definition(c.definition(abs_f, ln, ch))
            elif op == "findReferences":
                text = L.format_references(c.references(abs_f, ln, ch))
            elif op == "hover":
                text = c.hover(abs_f, ln, ch) or "no hover info"
            elif op == "documentSymbol":
                text = L.format_symbols(c.document_symbols(abs_f))
            elif op == "workspaceSymbol":
                text = L.format_workspace_symbols(c.workspace_symbols(abs_f, str(req.get("query") or "")))
            elif op == "goToImplementation":
                locs = c.implementation(abs_f, ln, ch)
                text = "no implementation found" if not locs else "\n".join(loc.human() for loc in locs)
            elif op == "prepareCallHierarchy":
                text = L.format_call_hierarchy_items(c.prepare_call_hierarchy(abs_f, ln, ch))
            elif op == "incomingCalls":
                text = L.format_incoming_calls(c.incoming_calls(abs_f, ln, ch))
            elif op == "outgoingCalls":
                text = L.format_outgoing_calls(c.outgoing_calls(abs_f, ln, ch))
            elif op == "diagnostics":
                rounds, delay = self._clamp_poll(
                    req.get("rounds"),
                    req.get("delay"),
                )
                diags = c.diagnostics(abs_f, rounds=rounds, delay=delay)
                text = L.format_diagnostics(abs_f, diags)
            else:
                return {"ok": False, "error": f"unknown op: {op!r}"}
            return {"ok": True, "result": text}
        except Exception as e:  # noqa: BLE001
            self._close_client()
            return {"ok": False, "error": str(e)}

    # -- request handlers ----------------------------------------------- #
    def _handle(self, req: dict) -> dict:
        try:
            op = req.get("op")
            if op == "ping":
                return {"ok": True}
            if op == "touch":
                f = req.get("file")
                if not isinstance(f, str) or not f:
                    return {"ok": False, "error": "file required"}
                try:
                    c = self._ensure_client()
                    c.sync_document(f)
                    return {"ok": True}
                except Exception as e:  # noqa: BLE001
                    self._close_client()
                    return {"ok": False, "error": str(e)}
            if op == "diagnostics" and not req.get("as_tool"):
                # Write-gate path: raw diagnostic list.
                f = req.get("file")
                if not isinstance(f, str) or not f:
                    return {"ok": False, "error": "file required"}
                root = req.get("workspace_root")
                if isinstance(root, str) and root and os.path.abspath(root) != self.workspace_root:
                    return {
                        "ok": False,
                        "error": "workspace_root mismatch",
                    }
                rounds, delay = self._clamp_poll(
                    req.get("rounds"),
                    req.get("delay"),
                )
                try:
                    c = self._ensure_client()
                    diags = c.diagnostics(f, rounds=rounds, delay=delay)
                    return {"ok": True, "diagnostics": diags}
                except Exception as e:  # noqa: BLE001
                    self._close_client()
                    return {"ok": False, "error": str(e)}
            if op in _TOOL_OPS:
                return self._tool_result(op, req)
            return {"ok": False, "error": f"unknown op: {op!r}"}
        except Exception as e:  # noqa: BLE001 — never crash the accept loop
            return {"ok": False, "error": str(e)}

    def _serve_conn(self, conn: socket.socket) -> None:
        buf = b""
        try:
            conn.settimeout(30.0)
            while not self._stop.is_set():
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > _MAX_REQ_BYTES:
                    conn.sendall(b'{"ok":false,"error":"request too large"}\n')
                    break
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        req = json.loads(line.decode("utf-8"))
                    except Exception:  # noqa: BLE001
                        resp = {"ok": False, "error": "invalid json"}
                    else:
                        if not isinstance(req, dict):
                            resp = {"ok": False, "error": "request must be object"}
                        else:
                            resp = self._handle(req)
                    conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:  # noqa: BLE001 — connection-level; don't kill daemon
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # -- main loop ------------------------------------------------------ #
    def _bind(self) -> socket.socket:
        # Only remove a leftover socket if nothing is accepting on it.
        if self.sock_path.exists():
            try:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.settimeout(0.2)
                    probe.connect(str(self.sock_path))
                    # Live daemon already owns this path — refuse to steal.
                    raise RuntimeError(f"LSP socket already live: {self.sock_path}")
                except ConnectionRefusedError:
                    try:
                        self.sock_path.unlink()
                    except OSError:
                        pass
                except OSError:
                    # Not a clean refuse — leave path alone and try bind.
                    pass
                finally:
                    probe.close()
            except RuntimeError:
                raise
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.sock_path))
        try:
            os.chmod(self.sock_path, 0o600)
        except OSError:
            pass
        try:
            st = self.sock_path.stat()
            self._sock_identity = (st.st_dev, st.st_ino)
        except OSError:
            self._sock_identity = None
        sock.listen(16)
        sock.setblocking(False)
        return sock

    def _unlink_own_socket(self) -> None:
        """Unlink only if the path still refers to *our* inode."""
        if not self.sock_path.exists():
            return
        try:
            st = self.sock_path.stat()
            if self._sock_identity is not None and (st.st_dev, st.st_ino) != self._sock_identity:
                return  # replaced by another daemon — leave it
            self.sock_path.unlink()
        except OSError:
            pass

    def run(self) -> int:
        def _stop_handler(_signum, _frame):
            self._stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, _stop_handler)
            except (ValueError, OSError):
                pass

        self._listen = self._bind()
        self.ready.set()
        last_activity = __import__("time").monotonic()

        try:
            while not self._stop.is_set():
                timeout = max(0.05, min(1.0, self.idle_s))
                try:
                    ready, _, _ = select.select([self._listen], [], [], timeout)
                except (InterruptedError, ValueError):
                    continue
                now = __import__("time").monotonic()
                if ready:
                    try:
                        conn, _ = self._listen.accept()
                    except BlockingIOError:
                        continue
                    last_activity = now
                    # Handle inline — diagnostics are short; avoids thread
                    # storms under apply_patch multi-file. Crash isolation
                    # is per-connection try/except in _serve_conn.
                    self._serve_conn(conn)
                    last_activity = __import__("time").monotonic()
                elif (now - last_activity) >= self.idle_s:
                    break
        finally:
            self._close_client()
            if self._listen is not None:
                try:
                    self._listen.close()
                except OSError:
                    pass
            self._unlink_own_socket()
            if self._lock_file is not None:
                try:
                    self._lock_file.close()
                except Exception:  # noqa: BLE001
                    pass
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="sharedagents per-workspace LSP daemon")
    p.add_argument("--workspace-root", required=True)
    p.add_argument(
        "--server-cmd",
        default=None,
        help="language-server argv as JSON list, or space-separated fallback",
    )
    p.add_argument(
        "--idle-s",
        type=float,
        default=None,
        help="idle reap seconds (default SHAREDAGENTS_LSP_IDLE_S or 300)",
    )
    p.add_argument(
        "--sock",
        default=None,
        help="override Unix socket path (tests)",
    )
    args = p.parse_args(argv)

    server_cmd: list[str] | None = None
    if args.server_cmd:
        try:
            parsed = json.loads(args.server_cmd)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                server_cmd = parsed
            else:
                server_cmd = args.server_cmd.split()
        except json.JSONDecodeError:
            server_cmd = args.server_cmd.split()

    # Inherit the flock FD from the spawner (lifetime ownership).
    lock_file = None
    lock_fd_raw = os.environ.get("SHAREDAGENTS_LSP_LOCK_FD")
    if lock_fd_raw:
        try:
            lock_file = os.fdopen(int(lock_fd_raw), "a+")
        except Exception:  # noqa: BLE001
            lock_file = None

    local_bin = os.path.expanduser("~/.local/bin")
    path = os.environ.get("PATH", "")
    if local_bin not in path.split(os.pathsep):
        os.environ["PATH"] = local_bin + os.pathsep + path
    try:
        return LspDaemon(
            args.workspace_root,
            server_cmd=server_cmd,
            idle_s=args.idle_s,
            sock_path=Path(args.sock) if args.sock else None,
            lock_file=lock_file,
        ).run()
    except Exception:  # noqa: BLE001 — surface to stderr for journal/debug
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
