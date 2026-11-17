"""Embedding-backed skill auto-invoke index.

`SkillIndex` keeps an in-memory vector for each skill the agent can
see, computed from `<name>: <description>` so the model's user-message
embedding lines up against the same surface the model would read in
the system prompt's `# SKILLS` block. ANN over a few dozen skills is
trivial enough that we skip sqlite-vec / Milvus and do a plain dot
product (vectors are L2-normalized by `memory_embed`, so cosine ==
inner product).

Lifecycle:

  - Built lazily on first `search()`. If the embedder isn't
    configured, build returns an empty index and search returns []
    forever — caller treats that as "feature disabled".
  - mtime-tracked: each `search()` calls `ensure_built()` which
    re-walks the discovered skill set. If any source's mtime moved,
    a skill was added, or a skill was removed, we rebuild. So
    editing a `SKILL.md` and re-saving picks up on the next user
    turn without an agent restart.

Concurrency:

  - One `asyncio.Lock` guards rebuild so two concurrent turns can't
    fight each other into the embedder. `search()` waits if a
    rebuild is in flight, then proceeds against the fresh index.

Failure modes — index never raises into the dispatch loop:
    embedder errors, batch failures, missing files all collapse to
    "no auto-invoke this turn." A failed build sets `_built=True`
    so we don't retry on every search; `clear()` forces a rebuild.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger("sharedagents.skill_index")


@dataclass
class _IndexedSkill:
    """One indexed skill — vector + the body we'll splice into a
    user preamble if the skill scores high enough on a query."""

    name: str
    description: str
    body: str
    source: Path
    mtime: float
    vector: list[float]


class SkillIndex:
    """Vector index over the agent's discoverable skills.

    Owned by `AgentApp`; survives across turns. The embedder is
    passed in (so a future test can inject a deterministic stub),
    but in production it comes from `memory_embed.resolve_embedder()`.
    """

    # Minimum gap between rebuild RETRIES after a failed embed. If the
    # last build attempt hit a transient embedder error (Ollama down,
    # OpenAI 5xx), we don't want every subsequent `search()` to hammer
    # the same broken endpoint. Cooldown re-uses the previously-built
    # `_items` (if any) until the gap elapses, then tries once more.
    _REBUILD_RETRY_COOLDOWN_S: float = 60.0

    def __init__(
        self,
        embedder: Any,
        agent_home: Path | None,
    ) -> None:
        self._embedder = embedder
        self._agent_home = Path(agent_home) if agent_home else None
        self._items: list[_IndexedSkill] = []
        self._built: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        # Monotonic timestamp of the last failed `_rebuild`. `0.0` means
        # "no failure recorded" or "the failure has been cleared by a
        # successful rebuild." Reads MUST use `time.monotonic()` to
        # match. Drives the retry cooldown above.
        self._last_failed_at: float = 0.0

    @property
    def is_available(self) -> bool:
        """True iff an embedder was supplied; False = feature disabled."""
        return self._embedder is not None

    def clear(self) -> None:
        """Drop the cached index. Next `search` will rebuild."""
        self._items = []
        self._built = False

    # ---- lifecycle ----

    async def ensure_built(self) -> None:
        """Rebuild if stale (added/removed skills, edited SKILL.md).

        Cheap when nothing changed: walks the disk listing once and
        compares names + mtimes against the cached set. The actual
        embed call only fires when something actually moved.

        Failure recovery: if the last `_rebuild` raised, we don't
        retry every `search()` — that would hammer a broken embedder
        and freeze a useful failure mode (Ollama restart) into a
        useless one ("every turn waits 30s on a doomed connect").
        Instead we hold off retries for `_REBUILD_RETRY_COOLDOWN_S`
        and keep serving the previously-built `_items` (which may be
        slightly stale but is better than empty). After the cooldown
        elapses, the next `search()` retries.
        """
        from sharedagents_skill.skills import discover_skills, get_active_skill_allowlist

        # Honour the active skill permission so AUTO-INVOKE never surfaces a skill the
        # agent isn't allowed to use (the index feeds suggestions to the model mid-turn).
        _allow, _strict = get_active_skill_allowlist()
        skills = discover_skills(self._agent_home, allowlist=_allow, allowlist_strict=_strict)
        cur_names = {s.name for s in skills}
        cur_mtimes: dict[str, float] = {}
        for s in skills:
            try:
                cur_mtimes[s.name] = s.source.stat().st_mtime
            except OSError:
                cur_mtimes[s.name] = 0.0

        if self._built and self._items:
            cached_names = {it.name for it in self._items}
            cached_mtimes = {it.name: it.mtime for it in self._items}
            if cur_names == cached_names and cur_mtimes == cached_mtimes:
                return  # no drift
        elif self._built and not skills:
            return  # was built empty, still empty

        # Drift detected (or first build). If the last rebuild
        # failed and the cooldown hasn't elapsed, skip the retry
        # and keep serving stale items rather than burning 30s on
        # a still-broken embedder. `_built=False` after a failure
        # means a search WILL try again once the cooldown lapses.
        if self._last_failed_at:
            elapsed = time.monotonic() - self._last_failed_at
            if elapsed < self._REBUILD_RETRY_COOLDOWN_S:
                return

        async with self._lock:
            # Re-check after acquiring the lock — another coroutine
            # may have rebuilt while we were waiting.
            cached_names_now = {it.name for it in self._items}
            cached_mtimes_now = {it.name: it.mtime for it in self._items}
            if self._built and cur_names == cached_names_now and cur_mtimes == cached_mtimes_now:
                return
            # Re-check cooldown under the lock: a concurrent coroutine
            # might have just failed while we were waiting.
            if self._last_failed_at:
                elapsed = time.monotonic() - self._last_failed_at
                if elapsed < self._REBUILD_RETRY_COOLDOWN_S:
                    return
            await self._rebuild(skills, cur_mtimes)

    async def _rebuild(
        self,
        skills: list,
        mtimes: dict[str, float],
    ) -> None:
        if not self._embedder:
            self._items = []
            self._built = True
            self._last_failed_at = 0.0
            return
        if not skills:
            self._items = []
            self._built = True
            self._last_failed_at = 0.0
            return
        # Anchor each embedding on `name: description` so the
        # similarity geometry matches what the user typically writes
        # ("how do I use brave-search?", "search the web for X").
        texts: list[str] = []
        for s in skills:
            desc = (s.description or s.short_description or s.name).strip()
            texts.append(f"{s.name}: {desc}")
        try:
            vectors = await self._embedder.embed_batch(texts)
        except Exception as e:  # noqa: BLE001
            # Transient embedder failure — keep the previously-built
            # `_items` intact (search continues against the old set)
            # and arm the cooldown so we don't retry on every turn.
            # `_built=False` means the cooldown-gated `ensure_built`
            # retry path will give it another shot once enough time
            # has elapsed.
            _log.warning(
                "skill index rebuild failed during embed (keeping %d stale item(s); retry in %.0fs): %s",
                len(self._items),
                self._REBUILD_RETRY_COOLDOWN_S,
                e,
            )
            self._built = False
            self._last_failed_at = time.monotonic()
            return
        items: list[_IndexedSkill] = []
        for s, v in zip(skills, vectors):
            if not v:
                # A failed item in a batch (None vector); skip.
                continue
            items.append(
                _IndexedSkill(
                    name=s.name,
                    description=s.description or s.short_description or "",
                    body=s.body or "",
                    source=s.source,
                    mtime=mtimes.get(s.name, 0.0),
                    vector=list(v),
                )
            )
        self._items = items
        self._built = True
        # Successful rebuild clears the failure stamp so subsequent
        # drift triggers an immediate retry (no cooldown after a
        # green build).
        self._last_failed_at = 0.0
        _log.info(
            "skill index built: %d skill(s) embedded (model=%s)",
            len(items),
            getattr(self._embedder, "model", "?"),
        )

    # ---- query ----

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        threshold: float = 0.6,
    ) -> list[tuple[str, float, str]]:
        """Return up to `top_k` `(name, score, body)` matches above `threshold`.

        Score is cosine similarity; vectors are pre-normalized so this
        is a plain inner product. Empty query / no embedder / build
        failure → []. The body included in each tuple is the full
        SKILL.md body (post-frontmatter) so the caller can inject it
        verbatim into the prompt without a second filesystem hit.
        """
        if not self._embedder:
            return []
        q = (query or "").strip()
        if not q:
            return []
        await self.ensure_built()
        if not self._items:
            return []
        try:
            qvec = await self._embedder.embed_query(q)
        except Exception as e:  # noqa: BLE001
            _log.warning("skill index query embed failed: %s", e)
            return []
        if not qvec:
            return []
        scored: list[tuple[float, _IndexedSkill]] = []
        for it in self._items:
            score = _dot(qvec, it.vector)
            scored.append((score, it))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[tuple[str, float, str]] = []
        for score, it in scored[: max(1, int(top_k))]:
            if score < threshold:
                break  # remainder are even lower (sorted desc)
            out.append((it.name, float(score), it.body))
        return out


def _dot(a: list[float], b: list[float]) -> float:
    """Inner product. Both vectors are assumed L2-normalized (the
    embedder does this in `memory_embed._normalize`); for a normalized
    pair this equals cosine similarity."""
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


def extract_query_text(user_input: Any) -> str:
    """Pull plain-text from the user input for similarity search.

    `user_input` may be a string (text-only turn) or a list of
    OpenAI-style multi-part content parts (vision turn). For lists
    we concatenate every `type=="text"` part; image parts contribute
    nothing to the search query.
    """
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for p in user_input:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                t = p.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        return "\n".join(parts)
    return ""


def render_preamble(matches: list[tuple[str, float, str]]) -> str:
    """Format auto-invoked skill matches as a user-role preamble.

    Returns the joined block (or `""` when there are no matches).
    Caller prepends this to the live user message.
    """
    if not matches:
        return ""
    chunks: list[str] = []
    for name, score, body in matches:
        chunks.append(f"[Auto-invoked skill: {name}] (relevance: {score:.2f})\n\n{(body or '').strip()}\n\n---\n\n")
    return "".join(chunks)
