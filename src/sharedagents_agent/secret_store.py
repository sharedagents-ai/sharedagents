"""Named secrets in the runtime's dotenv files — the web "Add secret" store.

Values live where the runtime already reads them:

* ``<home>/secrets.env`` — shared by every agent.
* ``<home>/agents/<name>/secrets.env`` — one agent only (wins over shared).

``sharedagents_agent.agent._load_agent_secrets`` layers both into ``cfg.api_keys``
(external CLIs get them as spawn env) and the gateway lifts the shared file
into ``os.environ`` (local tool subprocesses). The orchestrator's mtime config
watcher re-reads both after a write, so no restart is needed.

This module never logs, returns or echoes a value: listing yields names only.
Writes are atomic (0600 temp file in the same directory + ``os.replace``),
serialised by a lock file, and keep every other line of the file intact.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

#: Environment-variable names only; the name is shown in the UI and the chat.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
#: Same rule as the Agent Builder (``agent_editor._VALID_NAME_RE``).
AGENT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
MAX_VALUE_BYTES = 16 * 1024
SHARED = ""

#: Every separator ``str.splitlines`` (and so ``_parse_dotenv``) breaks on,
#: plus NUL: one of these in a value would split it into extra assignments.
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029\x00")

#: Names that steer the runtime or a child process rather than carry a
#: credential. Refusing them also keeps every stored name outside the folder
#: sandbox's env allowlist (``folder_sandbox.narrow_env`` keeps only these
#: exact names plus ``LC_*`` / ``SHAREDAGENTS_*``), so restricted turns never see
#: a stored secret.
_RESERVED_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TZ",
        "TMPDIR",
        "LANG",
        "PWD",
        "OLDPWD",
        "IFS",
        "ENV",
        "BASH_ENV",
        "PS4",
        "VIRTUAL_ENV",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PERL5OPT",
        "PERL5LIB",
        "RUBYOPT",
        "RUBYLIB",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_EXEC_PATH",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "EDITOR",
        "VISUAL",
        "PAGER",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    }
)
_RESERVED_PREFIXES = ("SHAREDAGENTS_", "LC_", "LD_", "DYLD_", "PYTHON", "XDG_")

_LINE_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


class SecretStoreError(Exception):
    """A rejected request; ``status`` is the HTTP code, ``code`` a stable slug."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SecretEntry:
    name: str
    agent: str  # "" = shared by every agent

    def to_dict(self) -> dict:
        return {"name": self.name, "agent": self.agent or None}


def _home(home: Path | None) -> Path:
    if home is not None:
        return Path(home)
    from sharedagents_agent.agent import sharedagents_root

    return sharedagents_root()


def _reserved(name: str) -> bool:
    from sharedagents_agent.folder_sandbox import _KEEP_EXACT

    upper = name.upper()
    return upper in _RESERVED_EXACT or upper in _KEEP_EXACT or upper.startswith(_RESERVED_PREFIXES)


def validate_name(name: object) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise SecretStoreError(
            400,
            "invalid_name",
            "Use letters, digits and underscores, starting with a letter or underscore (an environment variable name).",
        )
    if _reserved(name):
        raise SecretStoreError(
            400,
            "reserved_name",
            f"{name} controls the runtime, not a credential; pick another name.",
        )
    return name


def validate_value(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SecretStoreError(400, "empty_value", "The value is empty.")
    if any(ch in _LINE_BREAKS for ch in value):
        raise SecretStoreError(
            400,
            "invalid_value",
            "The value must be a single line.",
        )
    if len(value.encode("utf-8")) > MAX_VALUE_BYTES:
        raise SecretStoreError(400, "value_too_large", "The value is too long.")
    return value


def secrets_path(agent: str = SHARED, *, home: Path | None = None) -> Path:
    """The dotenv file for ``agent`` (``""`` = shared). The agent must exist."""
    root = _home(home)
    if not agent:
        return root / "secrets.env"
    if not isinstance(agent, str) or not AGENT_RE.fullmatch(agent):
        raise SecretStoreError(400, "invalid_agent", "Unknown agent.")
    agent_dir = root / "agents" / agent
    if not agent_dir.is_dir():
        raise SecretStoreError(404, "unknown_agent", "Unknown agent.")
    return agent_dir / "secrets.env"


def _quote(value: str) -> str:
    # ``_parse_dotenv`` strips exactly one pair of matching outer quotes, so
    # wrapping always round-trips — even a value that is itself quoted.
    return f'"{value}"' if "'" in value else f"'{value}'"


def _line_key(line: str) -> str | None:
    if line.lstrip().startswith("#"):
        return None
    m = _LINE_KEY_RE.match(line)
    return m.group(1) if m else None


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    from sharedagents_core.oscompat import file_lock

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path.with_name(path.name + ".lock")), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with file_lock(fd):
            yield
    finally:
        os.close(fd)


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def _write_lines(path: Path, lines: list[str]) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        try:
            data = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("secret file write made no progress")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    """Make the rename durable (POSIX; directories cannot be opened on Windows)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def set_secret(name: str, value: str, agent: str = SHARED, *, home: Path | None = None) -> SecretEntry:
    """Create or replace ``name``; returns the entry (never the value)."""
    validate_name(name)
    validate_value(value)
    path = secrets_path(agent, home=home)
    line = f"{name}={_quote(value)}"
    with _locked(path):
        out: list[str] = []
        placed = False
        for raw in _read_lines(path):
            if _line_key(raw) == name:
                if not placed:
                    out.append(line)
                    placed = True
                continue  # drop duplicates; the parser keeps the last one
            out.append(raw)
        if not placed:
            out.append(line)
        _write_lines(path, out)
    return SecretEntry(name=name, agent=agent)


def delete_secret(name: str, agent: str = SHARED, *, home: Path | None = None) -> bool:
    """Remove ``name``; ``False`` when it was not there."""
    validate_name(name)
    path = secrets_path(agent, home=home)
    with _locked(path):
        lines = _read_lines(path)
        kept = [raw for raw in lines if _line_key(raw) != name]
        if len(kept) == len(lines):
            return False
        _write_lines(path, kept)
    return True


def _names(path: Path) -> list[str]:
    from sharedagents_agent.agent import _parse_dotenv

    try:
        return sorted(_parse_dotenv(path.read_text(encoding="utf-8")))
    except OSError:
        return []


def secret_agents(*, home: Path | None = None) -> list[str]:
    """Agents a secret can be scoped to: every configured agent directory."""
    agents_dir = _home(home) / "agents"
    if not agents_dir.is_dir():
        return []
    return sorted(d.name for d in agents_dir.iterdir() if AGENT_RE.fullmatch(d.name) and (d / "config.json").is_file())


def list_secrets(*, home: Path | None = None) -> list[SecretEntry]:
    """Every stored name, shared first then per agent. No values."""
    root = _home(home)
    out = [SecretEntry(n, SHARED) for n in _names(root / "secrets.env")]
    agents_dir = root / "agents"
    if agents_dir.is_dir():
        for agent_dir in sorted(agents_dir.iterdir()):
            if not AGENT_RE.fullmatch(agent_dir.name):
                continue
            path = agent_dir / "secrets.env"
            if path.is_file():
                out.extend(SecretEntry(n, agent_dir.name) for n in _names(path))
    return out


__all__ = [
    "MAX_VALUE_BYTES",
    "SecretEntry",
    "SecretStoreError",
    "delete_secret",
    "list_secrets",
    "secret_agents",
    "secrets_path",
    "set_secret",
    "validate_name",
    "validate_value",
]
