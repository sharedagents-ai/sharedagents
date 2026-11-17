"""The codex models sharedagents may pass to ``codex --model``.

The authority is the INSTALLED codex CLI. Its model list comes from the
ChatGPT backend, gated by client version and login, or, for an API-key
login, from the catalog bundled with the binary.
``$CODEX_HOME/models_cache.json`` is shared by every codex on the host (IDE
extensions, long-running app-servers, older installs, other logins), and each
of them rewrites it in place. So, for the installed version and the current
codex login:

* ACCEPT a model found in the NEWER of two lists: the shared cache written
  by the installed version (ignored for an API-key login, which codex serves
  from its bundled catalog), and the list saved for this version and login
  under ``<sharedagents root>/cache/codex-models/<version>/<login>.json``. The
  newer list wins by file mtime, so a fresh live fetch beats an older cache
  and a refreshed cache retires models an older saved list still has. Only
  lists that can be attributed to this login are saved: the picker's live
  fetch made as the installed version, and what the installed codex reports
  when asked. The shared cache's owner is opaque to sharedagents, so it is never
  saved.
* REFUSE only what the installed codex, asked at that moment
  (``codex debug models``, which refreshes for its own version and login),
  does not list either, and only when codex could VOUCH for that answer: an
  API-key login's printed catalog, or a cache for the installed version left
  behind by the refresh. Codex prints its bundled catalog, exit 0, when the
  refresh fails, and may print a catalog enforced by its
  ``model_catalog_json`` setting, so what it prints always counts for
  accepting. When codex cannot vouch, callers let codex itself judge the
  model. A vouch holds until codex's own cache would expire (codex answers
  from a fresh cache without fetching), 5 min at most; codex is not re-asked
  while a recent ask is failing. A model the backend added inside codex's
  cache TTL is unknown to codex too until that TTL ends.
* With nothing known (fresh install, or codex's version unreadable) codex is
  asked once. If it cannot answer, callers let codex itself be the authority.

There is deliberately no hardcoded model list: a new model works without a
source change.

Incident 2026-09-23: a codex 0.154.0 app-server left running after the npm
upgrade to 0.155.1 rewrote the shared cache every ~2.5 min without gpt-6-sol,
and sharedagents refused gpt-6-sol although the codex it spawns serves it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("sharedagents.discovery.codex_catalogue")

_VERSION_RE = re.compile(r"codex-cli\s+(\S+)")
_SAFE_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$")
# Codex reuses its fetched list for 5 minutes without refetching
# (OnlineIfUncached); sharedagents re-asks on that clock.
_CODEX_CACHE_TTL_S = 300.0
_ASK_RETRY_S = 300.0
_CONFIRM_TTL_S = 300.0

_State = tuple[int, int, int] | None

# (real path, inode, mtime_ns, size) of the codex executable -> its version.
_version_memo: dict[tuple[str, int, int, int], str] = {}
# models_cache.json path -> ((inode, mtime_ns, size), parsed or None).
_shared_memo: dict[str, tuple[tuple[int, int, int], tuple[str, dict[str, dict[str, Any]]] | None]] = {}
# (real codex path, version, login) -> time of the last failed ask.
_ask_failed: dict[tuple[str, str, str], float] = {}
# (real codex path, version, login, shared-cache state, saved-list state) ->
# when codex's vouch for its list there stops holding (``_now()`` clock).
_confirmed: dict[tuple[str, str, str, _State, _State], float] = {}


@dataclass(frozen=True)
class CodexCatalogue:
    """Models the installed codex CLI may be asked to run."""

    version: str
    source: str  # e.g. "cache", "saved", "cache+saved", "codex"; "" = unknown
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return bool(self.source)

    def ids(self) -> dict[str, str]:
        """Lowercased slug -> the backend's own spelling."""
        return {key: str(entry.get("slug") or key) for key, entry in self.models.items()}

    def context_windows(self) -> dict[str, int]:
        """Lowercased slug -> the larger of ``context_window`` and
        ``max_context_window`` (the picker badge and context meter want the max)."""
        out: dict[str, int] = {}
        for key, entry in self.models.items():
            window = max(_as_int(entry.get("context_window")), _as_int(entry.get("max_context_window")))
            if window > 0:
                out[key] = window
        return out


def installed_codex_version() -> str:
    """Version reported by the codex CLI sharedagents spawns (``codex --version``),
    e.g. ``0.155.1``; ``""`` when codex is missing or does not answer.

    Remembered per executable identity (real path, inode, mtime, size), so an
    upgrade in place is noticed on the next call without restarting sharedagents.
    """
    exe = _codex_executable()
    try:
        real = os.path.realpath(exe)
        st = os.stat(real)
    except OSError:
        return ""
    key = (real, st.st_ino, st.st_mtime_ns, st.st_size)
    remembered = _version_memo.get(key)
    if remembered is not None:
        return remembered
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        _LOG.info("codex --version failed: %s", exc)
        return ""
    match = _VERSION_RE.search(out.stdout or "")
    version = match.group(1) if match else ""
    if version:
        _version_memo[key] = version
    return version


def load_codex_catalogue(*, ask_codex: bool = False) -> CodexCatalogue:
    """Models the installed codex may be asked to run: the newer of the
    shared cache written by the installed version (not for an API-key login)
    and the list saved for this version and login. With nothing known and
    ``ask_codex``, the installed codex is asked once. ``source == ""`` when
    nothing is known. Never raises."""
    version = installed_codex_version()
    if not version:
        # codex is missing, broken, or `--version` timed out: no list can be
        # vouched for (the shared cache may be any version's), so callers
        # defer to codex itself.
        return CodexCatalogue("", "")
    login = _codex_login_key()
    shared = _read_shared_cache() if login != "apikey" else None
    if shared is not None and shared[0] != version:
        shared = None
    saved, saved_mtime = _read_saved(version, login)
    if saved and (shared is None or saved_mtime >= shared[2]):
        return CodexCatalogue(version, "saved", saved)
    if shared is not None:
        return CodexCatalogue(version, "cache", shared[1])
    if ask_codex:
        asked, _vouched = _ask_codex(version, login)
        if asked:
            return CodexCatalogue(version, "codex", asked)
    return CodexCatalogue(version, "")


def confirm_codex_catalogue() -> tuple[CodexCatalogue, bool]:
    """``(catalogue, vouched)`` to check a would-be REFUSAL against.

    A same-version shared cache can hold another login's list, and a model
    can be newer than a cache still inside codex's TTL, so the installed
    codex is asked now (it refreshes for its own version AND login) and what
    it prints is added to the catalogue. ``vouched`` is True only while a
    vouch from codex still holds (see :func:`_vouch_expiry`); only then may a
    model missing from the catalogue be refused. Codex is asked at most once
    per shared-cache state while its vouch holds, and not while a recent ask
    is failing. Never raises.
    """
    version = installed_codex_version()
    if not version:
        return CodexCatalogue("", ""), False
    login = _codex_login_key()
    expires_at = _confirmed.get(_confirm_key(version, login))
    if expires_at is not None and _now() < expires_at:
        return load_codex_catalogue(), True
    asked, vouched = _ask_codex(version, login)
    catalogue = load_codex_catalogue()
    if asked:
        source = "+".join(part for part in (catalogue.source, "codex") if part)
        catalogue = CodexCatalogue(version, source, {**asked, **catalogue.models})
    return catalogue, vouched


def remember_live_models(client_version: str, payload: Any) -> None:
    """Save a backend list fetched as ``client_version`` for the current
    login when that is the installed codex's version, so the picker's live
    fetch also answers the spawn gate in every sharedagents process."""
    version = (client_version or "").strip()
    if not version or version != installed_codex_version():
        return
    models = _models_from_entries(payload.get("models") if isinstance(payload, dict) else None)
    if models:
        _save(version, _codex_login_key(), models, "live")


def _ask_codex(version: str, login: str) -> tuple[dict[str, dict[str, Any]], bool]:
    """``(models, vouched)`` from the installed codex (``codex debug models``).

    Codex refreshes its list for its own version and login first. What it
    prints always counts for accepting: a catalog enforced with its
    ``model_catalog_json`` setting can list models no cache has. It VOUCHES
    for its list (``vouched``, and the list is saved for ``version`` and
    ``login``) when the login is an API key, whose bundled catalog is what it
    prints, or when a cache for ``version`` is present after the refresh,
    whose models are added. Otherwise the refresh failed and codex printed
    its BUNDLED catalog. That can accept a model but never refuse one, and it
    is not saved, since that would lock out a model the backend added later.
    An ask that fails or cannot vouch is not repeated for five minutes for
    the same codex and login (a login made meanwhile is asked at once).
    Never raises.
    """
    exe = _codex_executable()
    key = (os.path.realpath(exe), version, login)
    failed_at = _ask_failed.get(key)
    if failed_at is not None and _now() - failed_at < _ASK_RETRY_S:
        return {}, False
    try:
        payload = _fetch_codex_models_payload(exe)
    except Exception as exc:  # noqa: BLE001 - missing binary / timeout / bad output
        _LOG.info("codex debug models failed: %s", exc)
        _ask_failed[key] = _now()
        return {}, False
    models = _models_from_entries(payload.get("models"))
    if login == "apikey":
        vouched = bool(models)
    else:
        shared = _read_shared_cache()
        vouched = shared is not None and shared[0] == version
        if shared is not None and vouched:
            models = {**models, **shared[1]}
    if not vouched:
        _ask_failed[key] = _now()
        return models, False
    _save(version, login, models, "codex")
    # Codex just vouched for its list in the cache state it left behind.
    _remember_confirmed(_confirm_key(version, login), _vouch_expiry(login))
    return models, True


def _vouch_expiry(login: str) -> float:
    """When a vouch made now stops holding: after ``_CONFIRM_TTL_S`` at most,
    and no later than codex's own cache expiry (for a non-API-key login).
    Codex answers from a fresh cache without fetching, so a model the backend
    adds meanwhile only appears once that cache expires, and sharedagents must not
    hold a refusal beyond it (a stale cache whose refresh keeps failing
    gives no hold, so each refusal re-asks codex; refusals are rare)."""
    hold = _CONFIRM_TTL_S
    if login != "apikey":
        try:
            age = time.time() - (_codex_home() / "models_cache.json").stat().st_mtime
            hold = min(hold, max(0.0, _CODEX_CACHE_TTL_S - age))
        except OSError:
            pass
    return _now() + hold


def _fetch_codex_models_payload(exe: str) -> dict[str, Any]:
    """``codex debug models`` output as a dict (normally ~0.4 s; bounded so a
    stalled refresh cannot hold a spawn or the picker for long). Raises on
    any failure."""
    out = subprocess.run(
        [exe, "debug", "models"],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )
    if out.returncode != 0:
        raise RuntimeError((out.stderr or "").strip()[:300] or f"exit status {out.returncode}")
    data = json.loads(out.stdout or "{}")
    if not isinstance(data, dict):
        raise ValueError("codex debug models printed no catalog object")
    return data


def _confirm_key(version: str, login: str) -> tuple[str, str, str, _State, _State]:
    """A vouch covers exactly the shared-cache AND saved-list files it was
    made against: rewriting either (a codex refresh, a live picker fetch)
    means codex must be asked again before a refusal."""
    return (
        os.path.realpath(_codex_executable()),
        version,
        login,
        _file_state(_codex_home() / "models_cache.json"),
        _file_state(_saved_path(version, login)),
    )


def _file_state(path: Path) -> _State:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _remember_confirmed(key: tuple[str, str, str, _State, _State], expires_at: float) -> None:
    if len(_confirmed) > 256:
        _confirmed.clear()
    _confirmed[key] = expires_at


def _now() -> float:
    return time.monotonic()


def _codex_login_key() -> str:
    """Which codex login a list belongs to (lists differ by login):
    ``apikey``, ``chatgpt-<first 12 hex of sha256(account id)>``, or
    ``unknown`` when codex's ``auth.json`` is unreadable (keyring store, not
    logged in). The account id is hashed, never written out."""
    data = _codex_auth()
    if not data:
        return "unknown"
    mode = str(data.get("auth_mode") or "").strip().lower()
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    if ("api" in mode) if mode else (bool(data.get("OPENAI_API_KEY")) and not tokens.get("access_token")):
        return "apikey"
    account = str(tokens.get("account_id") or data.get("account_id") or "")
    if not account:
        return "unknown"
    return "chatgpt-" + hashlib.sha256(account.encode("utf-8")).hexdigest()[:12]


def _codex_auth() -> dict[str, Any]:
    try:
        data = json.loads((_codex_home() / "auth.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _codex_executable() -> str:
    """The codex sharedagents spawns (same resolver as the spawn path)."""
    from sharedagents_agent.tools.impl._utils import _resolve_cli_executable

    return _resolve_cli_executable("codex")


def _codex_home() -> Path:
    env = (os.environ.get("CODEX_HOME") or "").strip()
    return Path(env).expanduser() if env else Path.home() / ".codex"


def _read_shared_cache() -> tuple[str, dict[str, dict[str, Any]], int] | None:
    """``(client_version, models, mtime_ns)`` from ``models_cache.json``;
    ``None`` when absent, unparsable (codex rewrites it in place, so a read
    can land mid-write) or empty. Re-parsed only when the file changes."""
    path = _codex_home() / "models_cache.json"
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    hit = _shared_memo.get(str(path))
    if hit is not None and hit[0] == key:
        return None if hit[1] is None else (*hit[1], st.st_mtime_ns)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    parsed: tuple[str, dict[str, dict[str, Any]]] | None = None
    if isinstance(data, dict):
        models = _models_from_entries(data.get("models"))
        if models:
            parsed = (str(data.get("client_version") or "").strip(), models)
    elif isinstance(data, list):
        # Legacy flat list of ids: written by an unknown codex version.
        models = _models_from_entries(data)
        if models:
            parsed = ("", models)
    _shared_memo[str(path)] = (key, parsed)
    return None if parsed is None else (*parsed, st.st_mtime_ns)


def _saved_path(version: str, login: str) -> Path:
    from sharedagents_agent.agent import sharedagents_root

    return sharedagents_root() / "cache" / "codex-models" / version / f"{login}.json"


def _read_saved(version: str, login: str) -> tuple[dict[str, dict[str, Any]], int]:
    """``(models, mtime_ns)`` of the list saved for ``version`` and ``login``;
    ``({}, 0)`` when there is none."""
    if not _SAFE_VERSION_RE.match(version):
        return {}, 0
    try:
        with open(_saved_path(version, login), "rb") as fh:
            mtime_ns = os.fstat(fh.fileno()).st_mtime_ns
            data = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError):
        return {}, 0
    if not isinstance(data, dict) or data.get("client_version") != version or data.get("login") != login:
        return {}, 0
    return _models_from_entries(data.get("models")), mtime_ns


def _save(version: str, login: str, models: dict[str, dict[str, Any]], source: str) -> None:
    """Persist ``models`` as the list for ``version`` under ``login`` with an
    atomic replace. ALWAYS rewritten, even when unchanged: the file's mtime
    records when this login's list was last answered, which decides whether
    it or the shared cache is newer. Saves are rare (a live picker fetch, or
    an answer from codex)."""
    if not models or not _SAFE_VERSION_RE.match(version):
        return
    entries = [models[key] for key in sorted(models)]
    path = _saved_path(version, login)
    tmp_name = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{login}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "client_version": version,
                    "login": login,
                    "source": source,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "models": entries,
                },
                fh,
            )
        os.replace(tmp_name, path)
        tmp_name = ""
    except OSError as exc:
        _LOG.info("could not save the codex %s model list: %s", version, exc)
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _models_from_entries(entries: Any) -> dict[str, dict[str, Any]]:
    """Lowercased slug -> entry, from a backend ``models`` array (entries
    carry ``slug``; ``id`` / ``name`` tolerated; bare strings are legacy)."""
    if not isinstance(entries, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, str):
            entry = {"slug": entry}
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug") or entry.get("id") or entry.get("name") or "").strip()
        if slug:
            out[slug.lower()] = {**entry, "slug": slug}
    return out


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "CodexCatalogue",
    "confirm_codex_catalogue",
    "installed_codex_version",
    "load_codex_catalogue",
    "remember_live_models",
]
