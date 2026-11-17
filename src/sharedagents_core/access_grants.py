"""Per-user access grants: which agents, models, tools and skills a person may use.

An admin grants each non-admin Web account a :class:`Grants` value (stored on
the ``users`` row). The Web enforces agents and models itself; tools, skills
and delegation are enforced inside the gateway turn runtime, so the grants of
whoever submitted a prompt ride that prompt as an :class:`AccessStamp` in the
first content block's ``_meta`` slot — the same channel the turn ``actor``
uses (``sharedagents_session.actors``).

Encoding, per dimension: ``"*"`` = All (including anything added later), a
list = only these, anything else (missing, malformed, ``[]``) = none. Decoding
fails closed. Stdlib-only leaf so web, gateway, agent and session layers can
all import it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

ALL = "*"
DIMENSIONS = ("agents", "models", "tools", "skills")
ACCESS_META_KEY = "sharedagents_access"
STAMP_VERSION = 1
# The tool that loads skill bodies; it is useless (and removed) without skills.
SKILL_TOOL = "Skill"
_FOLDER_MODES = frozenset({"ro", "rw"})

_Names = frozenset | None  # None = All


@dataclass(frozen=True)
class FolderGrant:
    """One approved host directory a restricted turn may see."""

    path: str
    mode: str  # ``ro`` or ``rw``


def _clean_folders(raw: Any) -> tuple[FolderGrant, ...] | None:
    """``None`` = unrestricted host FS (admin All); ``()`` = no folders."""
    if raw == ALL:
        return None
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[FolderGrant] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, Mapping):
            path = str(item.get("path") or "").strip()
            mode = str(item.get("mode") or "ro").strip().lower()
        elif isinstance(item, str):
            path, mode = item.strip(), "rw"
        else:
            continue
        if not path or mode not in _FOLDER_MODES or path in seen:
            continue
        seen.add(path)
        out.append(FolderGrant(path, mode))
    return tuple(out)


def _folder_obj(folders: tuple[FolderGrant, ...] | None) -> Any:
    if folders is None:
        return ALL
    return [{"path": f.path, "mode": f.mode} for f in folders]


def _intersect_folders(
    mine: tuple[FolderGrant, ...] | None,
    theirs: tuple[FolderGrant, ...] | None,
) -> tuple[FolderGrant, ...] | None:
    if mine is None:
        return theirs
    if theirs is None:
        return mine
    other = {f.path: f.mode for f in theirs}
    out = []
    for grant in mine:
        mode = other.get(grant.path)
        if mode is None:
            continue
        out.append(FolderGrant(grant.path, "ro" if "ro" in (grant.mode, mode) else "rw"))
    return tuple(out)


def _clean_names(dim: str, raw: Any) -> _Names:
    if raw == ALL:
        return None
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    out = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if dim == "agents":
            name = name.lower()
        if name:
            out.add(name)
    return frozenset(out)


@dataclass(frozen=True)
class Grants:
    """What one person may use. ``None`` in a dimension means All.

    ``folders`` is separate from the four catalogue dimensions: ``None`` is
    unrestricted host filesystem (an admin's All), ``()`` is none, and a
    tuple is the granted roots. A user payload that omits ``folders`` is
    none, never All.
    """

    agents: _Names = frozenset()
    models: _Names = frozenset()
    tools: _Names = frozenset()
    skills: _Names = frozenset()
    folders: tuple[FolderGrant, ...] | None = ()

    @classmethod
    def all(cls) -> Grants:
        return cls(None, None, None, None, None)

    @classmethod
    def none(cls) -> Grants:
        return cls()

    @classmethod
    def from_obj(cls, obj: Any) -> Grants:
        if not isinstance(obj, Mapping):
            return cls.none()
        fields = {dim: _clean_names(dim, obj.get(dim)) for dim in DIMENSIONS}
        fields["folders"] = _clean_folders(obj.get("folders"))
        return cls(**fields)

    @classmethod
    def from_json(cls, raw: Any) -> Grants:
        try:
            return cls.from_obj(json.loads(raw)) if isinstance(raw, str) else cls.from_obj(raw)
        except (TypeError, ValueError):
            return cls.none()

    def to_obj(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for dim in DIMENSIONS:
            names = getattr(self, dim)
            out[dim] = ALL if names is None else sorted(names)
        out["folders"] = _folder_obj(self.folders)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_obj(), sort_keys=True, separators=(",", ":"))

    def names(self, dim: str) -> _Names:
        return getattr(self, dim)

    def is_all(self, dim: str) -> bool:
        return getattr(self, dim) is None

    def is_unrestricted(self) -> bool:
        return all(self.is_all(dim) for dim in DIMENSIONS) and self.folders is None

    def allows(self, dim: str, *candidates: str) -> bool:
        """True when any non-empty candidate is granted in ``dim``.

        Tools accept glob patterns (``mcp__<server>__*``); every other
        dimension compares exactly (agents case-insensitively).
        """
        names = getattr(self, dim)
        wanted = [str(c).strip() for c in candidates if c is not None and str(c).strip()]
        if names is None:
            return bool(wanted)
        for cand in wanted:
            if dim == "agents":
                cand = cand.lower()
            if cand in names:
                return True
            if dim == "tools" and any(fnmatchcase(cand, pat) for pat in names):
                return True
        return False

    def intersect(self, other: Grants) -> Grants:
        merged = {}
        for dim in DIMENSIONS:
            mine, theirs = getattr(self, dim), getattr(other, dim)
            if mine is None:
                merged[dim] = theirs
            elif theirs is None:
                merged[dim] = mine
            else:
                merged[dim] = frozenset(
                    {n for n in mine if other.allows(dim, n)} | {n for n in theirs if self.allows(dim, n)}
                )
        merged["folders"] = _intersect_folders(self.folders, other.folders)
        return Grants(**merged)

    def is_subset_of(self, other: Grants) -> bool:
        for dim in DIMENSIONS:
            mine, theirs = getattr(self, dim), getattr(other, dim)
            if theirs is None:
                continue
            if mine is None:
                return False
            if not all(other.allows(dim, n) for n in mine):
                return False
        if other.folders is None:
            return True
        if self.folders is None:
            return False
        allowed = {f.path: f.mode for f in other.folders}
        for grant in self.folders:
            mode = allowed.get(grant.path)
            if mode is None:
                return False
            if grant.mode == "rw" and mode != "rw":
                return False
        return True


@dataclass(frozen=True)
class AccessStamp:
    """The grants a submitted prompt runs under, and whose they are."""

    email: str
    grants: Grants

    def to_obj(self) -> dict[str, Any]:
        return {"v": STAMP_VERSION, "email": self.email, "grants": self.grants.to_obj()}

    @classmethod
    def from_obj(cls, obj: Any) -> AccessStamp | None:
        if not isinstance(obj, Mapping) or not isinstance(obj.get("grants"), Mapping):
            return None
        email = str(obj.get("email") or "").strip().lower()
        return cls(email, Grants.from_obj(obj.get("grants")))

    @property
    def restricted(self) -> bool:
        return not self.grants.is_unrestricted()


def strip_access_meta(blocks: Any) -> list[dict[str, Any]]:
    """Copies of ``blocks`` with every access stamp removed from ``_meta``."""
    out: list[dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        copy = dict(block)
        meta = copy.get("_meta")
        if isinstance(meta, Mapping) and ACCESS_META_KEY in meta:
            meta = dict(meta)
            meta.pop(ACCESS_META_KEY, None)
            copy["_meta"] = meta
        out.append(copy)
    return out


def stamp_blocks(blocks: Any, stamp: AccessStamp) -> list[dict[str, Any]]:
    """Strip foreign stamps, then stamp block 0 (the only block read back)."""
    out = strip_access_meta(blocks)
    if out:
        meta = dict(out[0]["_meta"]) if isinstance(out[0].get("_meta"), Mapping) else {}
        meta[ACCESS_META_KEY] = stamp.to_obj()
        out[0]["_meta"] = meta
    return out


def stamp_from_blocks(blocks: Any) -> AccessStamp | None:
    """The stamp on block 0, else ``None`` (no stamp: keep the bound policy)."""
    for block in blocks or []:
        if not isinstance(block, Mapping):
            return None
        meta = block.get("_meta")
        if isinstance(meta, Mapping):
            return AccessStamp.from_obj(meta.get(ACCESS_META_KEY))
        return None
    return None


def merge_stamps(stamps: Iterable[AccessStamp | None]) -> AccessStamp | None:
    """One stamp for a turn that answers several people's queued messages.

    Unstamped items do not vote. If anyone restricted wrote one of the items,
    the turn runs under the intersection of the restricted grants.
    """
    present = [s for s in stamps if isinstance(s, AccessStamp)]
    if not present:
        return None
    restricted = [s for s in present if s.restricted]
    if not restricted:
        return present[0]
    grants = restricted[0].grants
    for other in restricted[1:]:
        grants = grants.intersect(other.grants)
    return AccessStamp(restricted[0].email, grants)


# Tools that act through a separate process the folder sandbox cannot confine
# (an MCP server, an ACP agent): a folder-limited turn never sees them.
UNSANDBOXABLE_TOOL_PREFIXES = ("mcp__",)
UNSANDBOXABLE_TOOLS = frozenset({"acp_call"})


def filter_tool_names(names: Iterable[str], grants: Grants | None) -> set[str]:
    """The subset of ``names`` a turn under ``grants`` may see and call."""
    names = set(names)
    if grants is None:
        return names
    kept = {n for n in names if grants.allows("tools", n)}
    if grants.skills is not None and not grants.skills:
        kept.discard(SKILL_TOOL)
    if grants.folders is not None:
        kept = {n for n in kept if n not in UNSANDBOXABLE_TOOLS and not n.startswith(UNSANDBOXABLE_TOOL_PREFIXES)}
    return kept


def delegation_refusal(
    grants: Grants | None,
    agent: str,
    *,
    external: bool,
    model: str = "",
) -> str | None:
    """Why a turn under ``grants`` may not delegate to ``agent``, else ``None``.

    External-engine agents (Claude Code, Codex, Cursor, …) bring their own
    tools and skills and run outside the folder sandbox, which sharedagents cannot
    narrow, so they need All tools, All skills and All folders.
    """
    if grants is None:
        return None
    name = str(agent or "").strip()
    if not grants.allows("agents", name):
        return f"[access] agent {name!r} is not granted to this user"
    if external and not (grants.is_all("tools") and grants.is_all("skills") and grants.folders is None):
        return (
            f"[access] agent {name!r} runs its own tools outside the folder "
            "sandbox; it needs All tools, All skills and All folders granted"
        )
    model = str(model or "").strip()
    if model and not grants.allows("models", model):
        return f"[access] model {model!r} is not granted to this user"
    return None
