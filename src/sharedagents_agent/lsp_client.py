"""Minimal LSP (Language Server Protocol) client — code intelligence for the
``build`` / ``explore`` agents.

sharedagents's code-touching agents advertised "LSP-powered code intelligence" but
had none (only ACP's JSON-RPC transport existed).
This gives them real read-only intelligence: go-to-definition, find-references,
hover, document symbols, and diagnostics — so ``build`` can locate a symbol and
check for type errors instead of grepping and guessing.

Transport is stdio JSON-RPC with LSP's ``Content-Length`` framing. The server
binary (``basedpyright-langserver --stdio`` by default, ``pyright-langserver``
or ``pylsp`` if configured) is owned by the per-workspace daemon
(``tools.impl._lsp_daemon``) for production calls — OpenCode lsp.ts singleton
parity. Tool bodies fall back to a one-shot session only when a test
``transport`` / ``server_cmd`` is injected. When no server is installed the
tools degrade gracefully with a clear message rather than crash (honours the
no-auto-install policy).

The framing codec and message pump are pure/injectable so the whole client is
unit-testable against an in-memory fake server, with no language server binary
present.

NOTE: the server runs as a subprocess; the code graph cannot see across that
boundary, so guards live on this (parent) side.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".jsx": "javascriptreact",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".json": "json",
    ".md": "markdown",
}


def language_id_for_path(path: str | Path) -> str:
    """LSP languageId for *path*. Empty means unknown — do not guess Python."""
    return _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower(), "")


# --------------------------------------------------------------------------- #
# Framing codec (pure — directly testable)
# --------------------------------------------------------------------------- #
def encode_message(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def read_message(stream: BinaryIO) -> dict | None:
    """Read one LSP message (headers + JSON body) from ``stream``.

    Returns None on clean EOF. Raises on a malformed header block.
    """
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.decode("ascii", "replace").strip()
        if line == "":
            break
        if ":" in line:
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return {}
    body = stream.read(length)
    return json.loads(body.decode("utf-8"))


def path_to_uri(p: str | Path) -> str:
    return Path(p).resolve().as_uri()


DEFAULT_SERVERS = (
    ("basedpyright-langserver", ["--stdio"]),
    ("pyright-langserver", ["--stdio"]),
    ("pylsp", []),
)


def resolve_server_cmd(explicit: list[str] | None = None) -> list[str] | None:
    """Return the argv for an available language server, or None if none found.

    System services commonly inherit a minimal PATH even though pipx/user
    installs expose executables under ``~/.local/bin``.  Probe that conventional
    location explicitly after PATH lookup so the tool runner sees the same
    installed server as an interactive shell, without mutating process-global
    PATH or installing anything automatically.
    """
    if explicit:
        exe = explicit[0]
        return explicit if (os.path.isabs(exe) or shutil.which(exe)) else None
    env = os.environ.get("SHAREDAGENTS_LSP_SERVER")
    if env:
        parts = env.split()
        return parts if shutil.which(parts[0]) else None
    for exe, args in DEFAULT_SERVERS:
        if shutil.which(exe):
            return [exe, *args]
        user_exe = str(Path.home() / ".local" / "bin" / exe)
        if shutil.which(user_exe):
            return [user_exe, *args]
    return None


class LspUnavailable(RuntimeError):
    """No language server binary is installed / configured."""


@dataclass
class Location:
    uri: str
    line: int  # 0-based (LSP native)
    character: int

    @property
    def path(self) -> str:
        from urllib.parse import unquote, urlparse

        return unquote(urlparse(self.uri).path)

    def human(self) -> str:
        # 1-based for display.
        return f"{self.path}:{self.line + 1}:{self.character + 1}"


class LspClient:
    """A short-lived LSP session over stdio JSON-RPC.

    Spawn (``start``), one or more requests, then ``close``. A ``transport``
    (object with ``.stdin`` / ``.stdout`` binary streams and ``.terminate``)
    may be injected for tests; otherwise a subprocess is spawned from
    ``server_cmd``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        server_cmd: list[str] | None = None,
        transport: Any | None = None,
        timeout: float = 20.0,
    ):
        self.root = Path(root)
        self.server_cmd = server_cmd
        self._transport = transport
        self._proc = None
        self._id = 0
        self._timeout = timeout
        self._diagnostics: dict[str, list[dict]] = {}
        self._diag_published: set[str] = set()  # URIs the server actually published for
        # OpenCode client.ts: track open docs + last publish freshness so a
        # long-lived client does not return stale diagnostics after edits.
        self._files: dict[str, dict[str, Any]] = {}  # uri -> {version, text}
        self._diag_meta: dict[str, dict[str, Any]] = {}  # uri -> {at, version}
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> LspClient:
        if self._transport is None:
            cmd = self.server_cmd or resolve_server_cmd()
            if not cmd:
                raise LspUnavailable(
                    "no language server found (install basedpyright/pyright/pylsp or set SHAREDAGENTS_LSP_SERVER)"
                )
            import subprocess

            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._transport = self._proc
        self._initialize()
        return self

    def _write(self, payload: dict) -> None:
        self._transport.stdin.write(encode_message(payload))
        self._transport.stdin.flush()

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _has_buffered_bytes(self) -> bool:
        """True when the transport's own read buffer already holds unread bytes.

        ``select()`` only sees the raw file descriptor, but we parse frames off a
        buffered stream. A server that writes a notification and the matching
        response back-to-back (every real server does: ``publishDiagnostics``
        right before a query reply) lands BOTH frames in a single raw read; once
        the notification is parsed the response sits in user space and the fd
        looks idle. Without this probe the client would wait out the full
        deadline and kill a perfectly healthy server. Never blocks: the fd is
        flipped to non-blocking for the peek and restored afterwards.
        """
        stream = getattr(self._transport, "stdout", None)
        peek = getattr(stream, "peek", None)
        if peek is None:  # raw/unbuffered stream: select is authoritative
            return False
        try:
            fd = stream.fileno()
            import fcntl

            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except Exception:  # noqa: BLE001 — unprobeable transport: fall back to select
            return False
        try:
            return bool(peek(1))
        except Exception:  # noqa: BLE001 — BlockingIOError == nothing buffered
            return False
        finally:
            try:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)
            except OSError:
                pass

    def _wait_readable(self, deadline: float | None) -> None:
        """Block until the transport has data or the deadline passes. Only a real
        subprocess pipe is selectable; injected (test) transports have data ready
        synchronously, so they skip the wait. Raises LspUnavailable on timeout and
        terminates the (hung) server so a bad server can't wedge the caller."""
        if self._proc is None or deadline is None:
            return
        import select
        import time

        # A complete frame may already be buffered and thus invisible to select.
        # Checked before the deadline test so an answer we already hold is never
        # thrown away as a "timeout".
        if self._has_buffered_bytes():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._kill_hung()
            raise LspUnavailable("language server request timed out")
        try:
            fd = self._transport.stdout.fileno()
        except Exception:  # noqa: BLE001 — unselectable transport: don't block on select
            return
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            self._kill_hung()
            raise LspUnavailable("language server request timed out")

    def _kill_hung(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass

    def _pump_until(self, want_id: int, deadline: float | None = None) -> Any:
        """Read messages until the response with ``want_id`` arrives, banking
        any ``publishDiagnostics`` notifications (and which URIs were published)
        seen along the way. Bounded by ``deadline`` for a real server."""
        while True:
            self._wait_readable(deadline)
            msg = read_message(self._transport.stdout)
            if msg is None:
                raise LspUnavailable("language server closed the connection")
            if msg.get("method") == "textDocument/publishDiagnostics":
                import time as _time

                params = msg.get("params") or {}
                uri = params.get("uri", "")
                self._diagnostics[uri] = params.get("diagnostics") or []
                self._diag_published.add(uri)
                meta: dict[str, Any] = {"at": _time.monotonic()}
                if isinstance(params.get("version"), int):
                    meta["version"] = params["version"]
                self._diag_meta[uri] = meta
                continue
            if "id" in msg and msg.get("id") == want_id:
                if "error" in msg:
                    raise RuntimeError(f"lsp error: {msg['error']}")
                return msg.get("result")
            # server->client requests (e.g. workspace/configuration): reply empty.
            if msg.get("method") and "id" in msg:
                self._write({"jsonrpc": "2.0", "id": msg["id"], "result": None})

    def request(self, method: str, params: dict) -> Any:
        import time

        rid = self._next_id()
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = (time.monotonic() + self._timeout) if self._proc is not None else None
        return self._pump_until(rid, deadline)

    def notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _initialize(self) -> None:
        self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": path_to_uri(self.root),
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {"relatedInformation": True},
                        "hover": {},
                        "definition": {},
                        "references": {},
                        "implementation": {},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "callHierarchy": {"dynamicRegistration": False},
                    },
                    "workspace": {"symbol": {}},
                },
            },
        )
        self.notify("initialized", {})

    def sync_document(self, file: str | Path) -> tuple[str, int]:
        """Open or refresh a document (OpenCode ``notify.open`` parity).

        First sighting → ``didOpen`` at version 0 and clear cached diagnostics
        for the URI. Later sightings → ``didChange`` with a bumped version and
        full-text content change (do **not** wipe diagnostics — some servers
        only re-emit when content actually changes; freshness is decided by
        publish timestamp/version in :meth:`diagnostics`).
        """
        uri = path_to_uri(file)
        text = Path(file).read_text(errors="replace")
        doc = self._files.get(uri)
        if doc is None:
            self._diagnostics.pop(uri, None)
            self._diag_published.discard(uri)
            self._diag_meta.pop(uri, None)
            version = 0
            lang = language_id_for_path(file)
            if lang != "python":
                # This process hosts basedpyright. Opening HTML/JS as Python
                # produces false parser errors and must not happen.
                self._files[uri] = {"version": 0, "text": text, "skipped": True}
                return uri, 0
            self.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang,
                        "version": version,
                        "text": text,
                    }
                },
            )
            self._files[uri] = {"version": version, "text": text}
            return uri, version

        if doc.get("skipped"):
            return uri, int(doc.get("version") or 0)
        version = int(doc["version"]) + 1
        self.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            },
        )
        self._files[uri] = {"version": version, "text": text}
        return uri, version

    def did_open(self, file: str | Path) -> str:
        """Backward-compatible alias for :meth:`sync_document` (returns URI)."""
        uri, _version = self.sync_document(file)
        return uri

    def close(self) -> None:
        """Best-effort teardown that never raises.

        Sends shutdown/exit, then terminates the server subprocess and waits
        up to 2s for it to exit; a server that ignores SIGTERM (e.g. a
        busy-spinning node process) is escalated to kill() with another
        bounded wait, so a wedged server can never be orphaned under PID 1.
        Total worst-case block is ~4s; no behavior change for servers that
        exit promptly. Callers rely on this in ``finally`` blocks, so every
        step is individually guarded.
        """
        import subprocess

        try:
            self.request("shutdown", {})
            self.notify("exit", {})
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._kill_hung()  # SIGTERM ignored -> SIGKILL + bounded wait
            except Exception:  # noqa: BLE001 — already-exited/unsupported proc
                pass

    # -- high-level queries --------------------------------------------- #
    @staticmethod
    def _as_locations(result: Any) -> list[Location]:
        if result is None:
            return []
        items = result if isinstance(result, list) else [result]
        out: list[Location] = []
        for it in items:
            uri = it.get("uri") or it.get("targetUri")
            rng = it.get("range") or it.get("targetSelectionRange") or {}
            start = rng.get("start") or {}
            if uri:
                out.append(Location(uri, int(start.get("line", 0)), int(start.get("character", 0))))
        return out

    def definition(self, file: str | Path, line: int, character: int) -> list[Location]:
        uri, _ = self.sync_document(file)
        res = self.request(
            "textDocument/definition",
            {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}},
        )
        return self._as_locations(res)

    def references(
        self, file: str | Path, line: int, character: int, include_declaration: bool = True
    ) -> list[Location]:
        uri, _ = self.sync_document(file)
        res = self.request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": include_declaration},
            },
        )
        return self._as_locations(res)

    def hover(self, file: str | Path, line: int, character: int) -> str:
        uri, _ = self.sync_document(file)
        res = self.request(
            "textDocument/hover", {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}}
        )
        if not res:
            return ""
        contents = res.get("contents")
        if isinstance(contents, dict):
            return str(contents.get("value", ""))
        if isinstance(contents, list):
            return "\n".join(c.get("value", "") if isinstance(c, dict) else str(c) for c in contents)
        return str(contents or "")

    def document_symbols(self, file: str | Path) -> list[dict]:
        uri, _ = self.sync_document(file)
        res = self.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        return res or []

    def implementation(self, file: str | Path, line: int, character: int) -> list[Location]:
        uri, _ = self.sync_document(file)
        res = self.request(
            "textDocument/implementation",
            {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}},
        )
        return self._as_locations(res)

    def workspace_symbols(self, file: str | Path, query: str) -> list[dict]:
        # Opening the file ensures the server has started/loaded the workspace
        # so ``workspace/symbol`` has something to search.
        self.sync_document(file)
        res = self.request("workspace/symbol", {"query": query})
        return res or []

    def prepare_call_hierarchy(self, file: str | Path, line: int, character: int) -> list[dict]:
        uri, _ = self.sync_document(file)
        res = self.request(
            "textDocument/prepareCallHierarchy",
            {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}},
        )
        return res or []

    def incoming_calls(self, file: str | Path, line: int, character: int) -> list[dict]:
        items = self.prepare_call_hierarchy(file, line, character)
        if not items:
            return []
        res = self.request("callHierarchy/incomingCalls", {"item": items[0]})
        return res or []

    def outgoing_calls(self, file: str | Path, line: int, character: int) -> list[dict]:
        items = self.prepare_call_hierarchy(file, line, character)
        if not items:
            return []
        res = self.request("callHierarchy/outgoingCalls", {"item": items[0]})
        return res or []

    def diagnostics(self, file: str | Path, *, rounds: int = 8, delay: float = 0.35) -> list[dict] | None:
        """Sync the file and return diagnostics published *after* that sync.

        OpenCode waitForFreshPush parity: a long-lived client must not return
        a prior non-empty publish just because the URI is already in the cache.
        We record ``after`` at sync time and accept a publish only when its
        banked timestamp is ``>= after`` (and version matches when the server
        includes one). Returns None when no fresh publish arrives.
        """
        import time

        after = time.monotonic()
        uri, version = self.sync_document(file)
        rounds = max(1, min(int(rounds), 10))
        delay = max(0.0, min(float(delay), 1.0))
        for i in range(rounds):
            if delay and i and self._proc is not None:
                time.sleep(delay)
            try:
                self.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
            except LspUnavailable:
                break
            except Exception:  # noqa: BLE001
                break
            meta = self._diag_meta.get(uri)
            if not meta or meta.get("at", 0) < after:
                continue
            pub_ver = meta.get("version")
            if isinstance(pub_ver, int) and pub_ver != version:
                continue
            # Fresh publish (empty or non-empty) — stop polling.
            break
        meta = self._diag_meta.get(uri)
        if not meta or meta.get("at", 0) < after:
            return None
        pub_ver = meta.get("version")
        if isinstance(pub_ver, int) and pub_ver != version:
            return None
        if uri not in self._diag_published:
            return None
        return self._diagnostics.get(uri, [])


# --------------------------------------------------------------------------- #
# Tool bodies — formatted text results for the code/explore agents.
# `line`/`character` are 1-based on the wire (what a human/editor reports);
# converted to LSP's 0-based here. Read-only: definition/references/hover/
# symbols/diagnostics never mutate the workspace.
#
# Production calls (no transport/server_cmd) ride the per-workspace daemon —
# OpenCode's persistent LSP.service. Tests inject transport/server_cmd for a
# one-shot session.
# --------------------------------------------------------------------------- #
def _session(root, server_cmd=None, transport=None) -> LspClient:
    return LspClient(root, server_cmd=server_cmd, transport=transport).start()


def _via_daemon_or_session(
    root,
    *,
    transport,
    server_cmd,
    op: str,
    file: str | Path,
    line: int | None = None,
    character: int | None = None,
    query: str | None = None,
    body: Callable[[LspClient], str],
) -> str:
    if transport is not None or server_cmd is not None:
        c = _session(root, server_cmd, transport)
        try:
            return body(c)
        finally:
            c.close()
    # Lazy import: gate already imports lsp_client lazily; avoid cycle at import.
    from sharedagents_agent.tools.impl._lsp_gate import daemon_lsp_op

    return daemon_lsp_op(
        str(root),
        op,
        str(file),
        line=line,
        character=character,
        query=query,
    )


def _severity(n: Any) -> str:
    return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(n, "diag")


def _flatten_symbols(syms: list[dict], depth: int = 0) -> list[str]:
    out: list[str] = []
    for s in syms:
        name = s.get("name", "?")
        kind = s.get("kind", "")
        rng = (s.get("selectionRange") or s.get("range") or {}).get("start", {})
        ln = rng.get("line", 0) + 1
        out.append(f"{'  ' * depth}{name} (kind={kind}) :{ln}")
        out.extend(_flatten_symbols(s.get("children") or [], depth + 1))
    return out


def _symbol_location_line(sym: dict) -> str:
    """Render a SymbolInformation/WorkspaceSymbol as ``name (kind) uri:line``."""
    name = sym.get("name", "?")
    kind = sym.get("kind", "")
    loc = sym.get("location") or {}
    uri = loc.get("uri") or ""
    rng = (loc.get("range") or {}).get("start", {})
    ln = rng.get("line", 0) + 1
    where = f"{uri}:{ln}" if uri else ""
    return f"{name} (kind={kind}) {where}".rstrip()


def _call_hierarchy_item_line(item: dict) -> str:
    name = item.get("name", "?")
    kind = item.get("kind", "")
    uri = item.get("uri") or ""
    rng = (item.get("selectionRange") or item.get("range") or {}).get("start", {})
    ln = rng.get("line", 0) + 1
    where = f"{uri}:{ln}" if uri else ""
    return f"{name} (kind={kind}) {where}".rstrip()


def format_definition(locs: list[Location]) -> str:
    if not locs:
        return "no definition found"
    return "\n".join(loc.human() for loc in locs)


def format_references(locs: list[Location]) -> str:
    if not locs:
        return "no references found"
    return f"{len(locs)} reference(s):\n" + "\n".join(loc.human() for loc in locs)


def format_symbols(syms: list[dict]) -> str:
    lines = _flatten_symbols(syms)
    return "\n".join(lines) if lines else "no symbols found"


def format_workspace_symbols(syms: list[dict]) -> str:
    if not syms:
        return "no symbols found"
    return "\n".join(_symbol_location_line(s) for s in syms if isinstance(s, dict))


def format_call_hierarchy_items(items: list[dict]) -> str:
    if not items:
        return "no call-hierarchy item at position"
    return "\n".join(_call_hierarchy_item_line(it) for it in items if isinstance(it, dict))


def format_incoming_calls(calls: list[dict]) -> str:
    if not calls:
        return "no incoming calls found"
    out = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        out.append(_call_hierarchy_item_line(call.get("from") or {}))
    return "\n".join(out) if out else "no incoming calls found"


def format_outgoing_calls(calls: list[dict]) -> str:
    if not calls:
        return "no outgoing calls found"
    out = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        out.append(_call_hierarchy_item_line(call.get("to") or {}))
    return "\n".join(out) if out else "no outgoing calls found"


def format_diagnostics(file, diags: list[dict] | None) -> str:
    if diags is None:
        return "diagnostics unavailable (language server did not report)"
    if not diags:
        return "no diagnostics (clean)"
    out = []
    for d in diags:
        rng = (d.get("range") or {}).get("start", {})
        ln = rng.get("line", 0) + 1
        col = rng.get("character", 0) + 1
        out.append(f"{_severity(d.get('severity'))} {file}:{ln}:{col}: {d.get('message', '')}")
    return "\n".join(out)


def lsp_definition(root, file, line, character, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="goToDefinition",
        file=file,
        line=int(line),
        character=int(character),
        body=lambda c: format_definition(c.definition(file, int(line) - 1, int(character) - 1)),
    )


def lsp_references(root, file, line, character, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="findReferences",
        file=file,
        line=int(line),
        character=int(character),
        body=lambda c: format_references(c.references(file, int(line) - 1, int(character) - 1)),
    )


def lsp_hover(root, file, line, character, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="hover",
        file=file,
        line=int(line),
        character=int(character),
        body=lambda c: c.hover(file, int(line) - 1, int(character) - 1) or "no hover info",
    )


def lsp_symbols(root, file, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="documentSymbol",
        file=file,
        body=lambda c: format_symbols(c.document_symbols(file)),
    )


def lsp_implementation(root, file, line, character, *, transport=None, server_cmd=None) -> str:
    def _body(c: LspClient) -> str:
        locs = c.implementation(file, int(line) - 1, int(character) - 1)
        if not locs:
            return "no implementation found"
        return "\n".join(loc.human() for loc in locs)

    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="goToImplementation",
        file=file,
        line=int(line),
        character=int(character),
        body=_body,
    )


def lsp_workspace_symbols(root, file, query, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="workspaceSymbol",
        file=file,
        query=str(query or ""),
        body=lambda c: format_workspace_symbols(c.workspace_symbols(file, str(query or ""))),
    )


def lsp_prepare_call_hierarchy(root, file, line, character, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="prepareCallHierarchy",
        file=file,
        line=int(line),
        character=int(character),
        body=lambda c: format_call_hierarchy_items(c.prepare_call_hierarchy(file, int(line) - 1, int(character) - 1)),
    )


def lsp_incoming_calls(root, file, line, character, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="incomingCalls",
        file=file,
        line=int(line),
        character=int(character),
        body=lambda c: format_incoming_calls(c.incoming_calls(file, int(line) - 1, int(character) - 1)),
    )


def lsp_outgoing_calls(root, file, line, character, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="outgoingCalls",
        file=file,
        line=int(line),
        character=int(character),
        body=lambda c: format_outgoing_calls(c.outgoing_calls(file, int(line) - 1, int(character) - 1)),
    )


def lsp_diagnostics(root, file, *, transport=None, server_cmd=None) -> str:
    return _via_daemon_or_session(
        root,
        transport=transport,
        server_cmd=server_cmd,
        op="diagnostics",
        file=file,
        body=lambda c: format_diagnostics(file, c.diagnostics(file)),
    )
