"""Remembered reasoning effort per provider/model.

When the operator applies an effort together with a model (``/model …
effort=``, ``/effort``, the picker's "Set default", a new chat's
``initial_effort``), the level is remembered for that model. The next time
that model is selected without an explicit level, and when a new session
starts on it, the remembered level is applied. Picker rows carry it as
``effort_pref`` so frontends can render it without deciding anything.

The store is ``<sharedagents home>/model_efforts.json``, shared by every agent::

    {"version": 1, "efforts": {"<provider>/<model id>": "<level>"}}

The key's provider is the row's ``provider``, else its ``backend``, else the
engine's provider; key parts are lowercased. ``default`` (or an empty level)
removes the entry. Writes hold a sidecar lock for the whole
read-modify-write and replace the file atomically (tmp + ``os.replace``), so
concurrent processes (gateway, orchestrator, web) never lose an update or see
a torn file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("sharedagents.model.effort_prefs")

STORE_NAME = "model_efforts.json"
_VERSION = 1
_CLEAR = frozenset({"", "default", "auto", "reset", "clear"})

_cache_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}


def store_path() -> Path:
    """``<sharedagents home>/model_efforts.json`` (honours ``SHAREDAGENTS_HOME``)."""
    from sharedagents_core.kernel.layout import sharedagents_root

    return sharedagents_root() / STORE_NAME


def _part(value: Any) -> str:
    return str(value or "").strip().lower()


def pref_key(provider: Any, model_id: Any) -> str:
    """``<provider>/<model id>``, lowercased; ``""`` without a model id."""
    model = _part(model_id)
    if not model:
        return ""
    return f"{_part(provider) or 'unknown'}/{model}"


def row_provider(row: Mapping[str, Any], engine_provider: str = "") -> str:
    """Provider part of a row's key: provider, else backend, else engine's."""
    return _part(row.get("provider")) or _part(row.get("backend")) or _part(engine_provider)


def row_model_id(row: Mapping[str, Any]) -> str:
    return str(row.get("wire_id") or row.get("model_id") or row.get("id") or row.get("alias") or "").strip()


def row_key(row: Mapping[str, Any] | None, engine_provider: str = "") -> str:
    """The store key for a model row (``""`` when the row has no id)."""
    if not isinstance(row, Mapping):
        return ""
    return pref_key(row_provider(row, engine_provider), row_model_id(row))


def _configured_engine(agent: str) -> str:
    """The ``engine`` named in ``agents/<agent>/config.json`` (cheap raw read)."""
    if not agent:
        return ""
    try:
        from sharedagents_core.kernel.layout import agent_config_dir

        raw = json.loads((agent_config_dir(agent) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(raw.get("engine") or "").strip() if isinstance(raw, dict) else ""


def engine_provider(cfg: Any = None, agent: str = "") -> str:
    """The provider an agent's engine speaks for, used when a row names none.

    External engines carry it in the registry (``anthropic``, ``openai`` …);
    a native (local) engine is ``local``. ``cfg`` wins; without it the agent's
    own config file names the engine, then the agent's name.
    """
    from sharedagents_agent import external_registry as reg

    engine = str(getattr(cfg, "engine", "") or "").strip() if cfg is not None else ""
    if not engine:
        engine = _configured_engine(agent)
    if engine:
        if not reg.is_external_engine_kind(engine):
            return "local"
        try:
            return _part(reg.spec_for(engine).provider) or _part(engine)
        except Exception:  # noqa: BLE001 — generic ACP has no registry spec
            return _part(engine)
    try:
        return _part(reg.spec_for_short_name(agent).provider) or "local"
    except Exception:  # noqa: BLE001
        return "local"


# ── storage ──────────────────────────────────────────────────────────────


def _stat_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        _LOG.warning("model effort store unreadable (%s); treating as empty", type(exc).__name__)
        return {}
    efforts = raw.get("efforts") if isinstance(raw, dict) else None
    if not isinstance(efforts, dict):
        return {}
    out: dict[str, str] = {}
    for key, level in efforts.items():
        k, v = _part(key), str(level or "").strip()
        if k and v and v.lower() not in _CLEAR:
            out[k] = v
    return out


def load_prefs() -> dict[str, str]:
    """All remembered levels ``{key: level}`` (cached by file mtime/size)."""
    path = store_path()
    if _stat_sig(path) is None:
        return {}
    # Readers share the sidecar lock too: Windows cannot replace a file
    # while another process has it open through the normal Python reader.
    try:
        with _write_lock(path):
            sig = _stat_sig(path)
            cache_key = str(path)
            with _cache_lock:
                hit = _cache.get(cache_key)
                if hit is not None and hit[0] == sig:
                    return dict(hit[1])
            data = _read(path)
            with _cache_lock:
                if sig is not None:
                    _cache[cache_key] = (sig, data)
            return dict(data)
    except OSError:
        return {}


def get_pref(key: str) -> str:
    """The remembered level for ``key``, or ``""``."""
    if not key:
        return ""
    return load_prefs().get(_part(key), "")


def get_row_pref(row: Mapping[str, Any] | None, engine_provider: str = "") -> str:
    return get_pref(row_key(row, engine_provider))


@contextmanager
def _write_lock(path: Path) -> Iterator[None]:
    from sharedagents_core import oscompat

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as handle, oscompat.file_lock(handle):
        yield


def _atomic_write(path: Path, efforts: dict[str, str]) -> None:
    blob = (
        json.dumps(
            {"version": _VERSION, "efforts": dict(sorted(efforts.items()))},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def set_pref(key: str, level: Any) -> bool:
    """Remember ``level`` for ``key``; ``default``/empty removes the entry.

    Returns True when the file changed. Never raises: a store that cannot be
    written must not fail the effort change the operator asked for.
    """
    key = _part(key)
    if not key:
        return False
    value = str(level or "").strip()
    path = store_path()
    try:
        with _write_lock(path):
            efforts = _read(path)
            if value.lower() in _CLEAR:
                if key not in efforts:
                    return False
                efforts.pop(key)
            else:
                if efforts.get(key) == value:
                    return False
                efforts[key] = value
            _atomic_write(path, efforts)
            with _cache_lock:
                _cache.pop(str(path), None)
        return True
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("model effort store write failed (%s)", type(exc).__name__)
        return False


def set_row_pref(
    row: Mapping[str, Any] | None,
    level: Any,
    engine_provider: str = "",
) -> bool:
    return set_pref(row_key(row, engine_provider), level)


def stamp_rows(
    rows: Iterable[Any] | None,
    engine_provider: str = "",
) -> Iterable[Any] | None:
    """Set ``effort_pref`` (the remembered level or ``""``) on every dict row."""
    if not rows:
        return rows
    prefs = load_prefs()
    for row in rows:
        if isinstance(row, dict):
            row["effort_pref"] = prefs.get(row_key(row, engine_provider), "")
    return rows


def remembered_level_for(
    row: Mapping[str, Any] | None,
    values: Iterable[str],
    engine_provider: str = "",
) -> str:
    """The remembered level for ``row`` when ``values`` still lists it, else ``""``."""
    level = get_row_pref(row, engine_provider)
    return level if level and level in set(values or ()) else ""


def selected_level_for(
    row: Mapping[str, Any],
    agent_effort: Any = "",
    *,
    engine_provider: str = "",
    session_model: str = "",
    session_effort: str = "",
    mode: str = "auto",
) -> str:
    """Resolve the picker's effort on the server; clients only render it."""
    from sharedagents_model.models import derive_effort_capability

    cap = derive_effort_capability(row, mode=mode)
    values = cap.get("values") or []
    names = {_part(row.get(k)) for k in ("id", "alias", "wire_id", "model_id")}
    if _part(session_model) and _part(session_model) in names and _part(session_effort) not in _CLEAR:
        level = str(session_effort).strip()
        return level if level in values else ""
    remembered = get_row_pref(row, engine_provider)
    if remembered:
        return remembered if remembered in values else ""
    level = str(agent_effort or "").strip()
    if level and level.lower() not in _CLEAR:
        return level if level in values else ""
    return str(cap.get("default") or "") if cap.get("source") != "advertised" else ""
