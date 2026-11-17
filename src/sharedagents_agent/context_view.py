"""A session's context as an ordered, paged list of items.

Backs the WebUI context sheet (the composer's context meter opens it). The
backend decides what "the context" is; the page only renders what this module
returns.

Native engine (``local_llm``)
    The exact request the next round would send: the engine's own message
    builder (``_build_messages_simple`` — the same call manual ``/compact``
    uses to read the running context) with the system prompt and rolling
    preamble the next turn would compose, passed through the worker-request
    builder so provider-specific sanitising and the ``tools[]`` payload are
    the real ones. Token counts use the same counter as the pre-send budget
    (the model's own tokenizer when its row has one, else the character
    estimate), so the total agrees with the context meter.

    Not shown: memory recall and auto-invoked skills, which are chosen per
    prompt from the text of the next message.

External engines (Claude / Codex / Antigravity / Cursor / other CLIs)
    The engine keeps its own conversation; sharedagents cannot read it. The list
    holds what sharedagents has: the instructions it sent and the conversation it
    recorded. The total shown is what the engine last reported.

Paging is by cursor (``<snapshot>:<offset>``). A snapshot is cached for a few
minutes so every page of one scroll comes from the same list even while the
session keeps growing; an evicted snapshot is rebuilt and, if the context
changed meanwhile, the caller is told to start again from the top.

Image bytes never leave this module: image parts become placeholders.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PAGE_LIMIT_DEFAULT = 50
PAGE_LIMIT_MAX = 200
#: Rough byte budget of one page's item bodies (a page always carries one item).
PAGE_BYTES_DEFAULT = 256 * 1024
PAGE_BYTES_MAX = 1024 * 1024
#: Characters of one item's content carried inline in a page; the rest is
#: fetched with :func:`context_item_range`.
PAGE_ITEM_CHARS = 4_000
PREVIEW_CHARS = 160
RANGE_CHARS_DEFAULT = 256 * 1024
RANGE_CHARS_MAX = 1024 * 1024

_SNAPSHOT_TTL_S = 600.0
_SNAPSHOT_MAX = 8

WITHHELD_TEXT = "Only the conversation owner can read the agent's instructions."

_TURN_MARKER_RE = re.compile(r"\[Turn (\d+)(?:[^\]\n]*)\]\n")
_PREAMBLE_LABELS = (
    ("[PINNED FACTS", "preamble", "Pinned facts"),
    ("[ROLLING COMPACTION SUMMARY", "summary", "Compaction summary"),
    ("[CURRENT TODO LIST", "preamble", "Todo list"),
    ("[Auto-invoked skill", "preamble", "Skill"),
    ("<sharedagents-canonical-handoff>", "preamble", "Handoff"),
    ("[Active Agent Context", "preamble", "Agent switch note"),
    ("[sharedagents live budget]", "preamble", "Live budget"),
)

#: Short chip text per item kind.
KIND_LABELS = {
    "system": "System",
    "tools": "Tools",
    "preamble": "Context",
    "summary": "Summary",
    "user": "User",
    "assistant": "Assistant",
    "reasoning": "Reasoning",
    "tool_call": "Tool call",
    "tool_result": "Tool result",
    "image": "Image",
    "notice": "Notice",
    "marker": "Next turn",
}

#: Kinds whose content is the agent's instructions rather than conversation.
_INSTRUCTION_KINDS = frozenset({"system", "preamble", "tools"})


@dataclass
class ContextItem:
    kind: str
    role: str
    label: str
    content: str
    tokens: int = 0
    images: int = 0
    turn: int | None = None
    # Text pieces the model's tokenizer sees for this item and the fixed
    # overhead (chat-template framing, image tokens) charged to it.
    _texts: list = field(default_factory=list, repr=False)
    _overhead: int = 0


@dataclass
class ContextSnapshot:
    snapshot_id: str
    session_id: str
    agent: str
    engine: str
    scope: str  # "model_input" | "sharedagents_record"
    notice: str
    items: list
    totals: dict
    notes: list
    created_at: float = field(default_factory=time.monotonic)


# ── formatting (the page renders these strings as-is) ────────────────────


def format_tokens(n: Any) -> str:
    """Same units as the composer's context meter (``79K``, ``1.0M``)."""
    try:
        v = int(n or 0)
    except (TypeError, ValueError):
        v = 0
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 10_000:
        return f"{v / 1000:.0f}K"
    if v >= 1_000:
        return f"{v / 1000:.1f}K"
    return str(v)


def format_chars(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M chars"
    if n >= 1_000:
        return f"{n / 1000:.1f}K chars"
    return f"{n} chars"


def _preview(text: str) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= PREVIEW_CHARS:
        return flat
    return flat[: PREVIEW_CHARS - 1] + "…"


def _agents_md_body(agent: Any) -> str:
    """The AGENTS.md text this agent's prompt was composed from.

    The context sheet receives one joined system string. Matching this body
    is how that string is split back into a row the operator can see.
    """
    cfg = getattr(agent, "cfg", None)
    home = getattr(cfg, "home", None) if cfg is not None else None
    raw = ""
    if home:
        path = Path(home) / "AGENTS.md"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            raw = ""
    if not str(raw).strip() and cfg is not None:
        raw = str(getattr(cfg, "_agent_profile_system", "") or "")
    if not str(raw).strip():
        return ""
    from sharedagents_agent.context import strip_legacy_artifact_prompt_section

    return strip_legacy_artifact_prompt_section(str(raw)).strip()


def split_system_prompt_sections(
    system_prompt: str,
    agents_md: str,
    *,
    head_label: str,
) -> list[tuple[str, str]]:
    """Pull the AGENTS.md section out of a joined system prompt.

    The composer glues sections with a blank line. When the on-disk body
    sits on those boundaries, the sheet gets three rows: the instructions
    before it, ``AGENTS.md``, and whatever follows. Any other shape stays
    one row so a stale or partial file cannot slice the prompt mid-sentence.
    """
    prompt = system_prompt or ""
    body = (agents_md or "").strip()
    if not prompt:
        return []
    if not body:
        return [(head_label, prompt)]
    idx = prompt.find(body)
    end = idx + len(body) if idx >= 0 else -1
    bounded = (
        idx >= 0
        and (idx == 0 or prompt[:idx].endswith("\n\n"))
        and (end == len(prompt) or prompt[end:].startswith("\n"))
    )
    if not bounded:
        return [(head_label, prompt)]
    before = prompt[:idx].strip()
    after = prompt[end:].strip()
    out: list[tuple[str, str]] = []
    if before:
        out.append((head_label, before))
    out.append(("AGENTS.md", prompt[idx:end]))
    if after:
        out.append(("After AGENTS.md", after))
    return out


def _skip_separators(prompt: str, pos: int) -> int:
    marker = "<!-- SHAREDAGENTS_CACHE_BOUNDARY -->"
    while True:
        if prompt.startswith("\n\n", pos):
            pos += 2
            continue
        if prompt.startswith("\n" + marker + "\n", pos):
            pos += len(marker) + 2
            continue
        if prompt.startswith(marker + "\n", pos):
            pos += len(marker) + 1
            continue
        return pos


def _needle_at(prompt: str, pos: int, body: str) -> str:
    if body and prompt.startswith(body, pos):
        return body
    stripped = body.strip()
    if stripped and prompt.startswith(stripped, pos):
        return stripped
    return ""


def _at_section_boundary(prompt: str, idx: int) -> bool:
    if idx <= 0:
        return True
    if prompt[idx - 2 : idx] == "\n\n":
        return True
    return prompt[idx - 1 : idx] == "\n" and "CACHE_BOUNDARY" in prompt[max(0, idx - 40) : idx]


def _gap_label(gap: str) -> str:
    head = gap.lstrip()
    if head.startswith("# WORKSPACE-CONTEXT"):
        return "WORKSPACE-CONTEXT"
    if head.startswith("# MEMORY.md"):
        return "MEMORY.md"
    return "Also in the prompt"


def attribute_prompt_sections(
    prompt: str,
    parts: list[tuple[str, str]],
) -> list[tuple[str, str]] | None:
    """Label every section of *prompt* that was joined from *parts*.

    Parts are ``(title, text)`` in composer order. The next row is the first
    remaining part that starts at the cursor, so a section the stored prompt
    does not contain (a mode block added later, for example) does not swallow
    ``MEMORY.md`` and the workspace block that follow it. A gap that is itself
    one of those blocks keeps that name. Returns ``None`` when nothing matched.
    """
    if not prompt or not parts:
        return None
    pos = 0
    rows: list[tuple[str, str]] = []
    pending = [(title, text) for title, text in parts if (text or "").strip()]
    while pending:
        cursor = _skip_separators(prompt, pos)
        found: tuple[int, str, str] | None = None
        for i, (title, body) in enumerate(pending):
            needle = _needle_at(prompt, cursor, body)
            if needle:
                found = (i, title, needle)
                break
        if found is None:
            best: tuple[int, int, str, str] | None = None
            for i, (title, body) in enumerate(pending):
                for candidate in (body, body.strip()):
                    if not candidate:
                        continue
                    idx = prompt.find(candidate, cursor)
                    if idx < 0 or not _at_section_boundary(prompt, idx):
                        continue
                    if best is None or idx < best[0]:
                        best = (idx, i, title, candidate)
                    break
            if best is None:
                break
            idx, i, title, needle = best
            gap = prompt[cursor:idx].strip()
            if gap:
                rows.append((_gap_label(gap), gap))
            rows.append((title, prompt[idx : idx + len(needle)]))
            pos = idx + len(needle)
            pending = pending[i + 1 :]
            continue
        i, title, needle = found
        rows.append((title, prompt[cursor : cursor + len(needle)]))
        pos = cursor + len(needle)
        pending = pending[i + 1 :]
    if not rows:
        return None
    tail = prompt[_skip_separators(prompt, pos) :].strip()
    if tail:
        rows.append((_gap_label(tail), tail))
    return rows


def _prompt_sections(agent: Any, session: Any) -> list[tuple[str, str]]:
    """Composer sections for this agent, without leaving the skill allowlist changed."""
    cfg = getattr(agent, "cfg", None)
    if cfg is None or not getattr(cfg, "home", None):
        return []
    try:
        mode = agent._effective_mode(session)
    except Exception:  # noqa: BLE001
        mode = str(getattr(session, "system_mode", "") or "auto")
    try:
        from sharedagents_agent.prompt_build import composed_prompt_sections
        from sharedagents_skill import skills as skills_mod

        var = skills_mod._ACTIVE_SKILL_ALLOWLIST
        previous = var.get()
        try:
            parts = composed_prompt_sections(
                cfg,
                str(mode or "auto"),
                workspace_cwd=str(getattr(session, "workspace_cwd", "") or ""),
                workspace_origin=str(getattr(session, "workspace_origin", "") or ""),
                record_breakdown=False,
            )
        finally:
            var.set(previous)
    except Exception:  # noqa: BLE001 — a context read must not fail the page
        return []
    return [
        (str(p.get("title") or "Section"), str(p.get("text") or "")) for p in parts if str(p.get("text") or "").strip()
    ]


def expand_system_sections(
    items: list[ContextItem],
    parts: list[tuple[str, str]],
    agents_md: str,
) -> list[ContextItem]:
    """Replace each system row with the sections actually inside it.

    Composer titles win (``AGENTS.md``, ``MEMORY.md``, each hardcoded
    block). A prompt the composer cannot attribute still gets ``AGENTS.md``
    pulled out when that file is a whole section.
    """
    out: list[ContextItem] = []
    for item in items:
        if item.kind != "system" or not item.content:
            out.append(item)
            continue
        sections = attribute_prompt_sections(item.content, parts)
        if sections is None:
            sections = split_system_prompt_sections(
                item.content,
                agents_md,
                head_label=item.label or "System prompt",
            )
        # A one-section row keeps its item only when the title did not
        # change. The native dynamic suffix is often just MEMORY.md, and
        # that row must be named MEMORY.md rather than the piece's
        # generic "workspace context" label.
        if len(sections) == 1 and sections[0][1] == item.content and sections[0][0] == (item.label or "System prompt"):
            out.append(item)
            continue
        for i, (label, text) in enumerate(sections):
            out.append(
                ContextItem(
                    kind="system",
                    role="system",
                    label=label,
                    content=text,
                    turn=item.turn,
                    _texts=[text],
                    _overhead=item._overhead if i == 0 else 0,
                    images=item.images if i == 0 else 0,
                )
            )
    return out


# ── message → items ──────────────────────────────────────────────────────


def _image_placeholder(part: Mapping[str, Any], n: int) -> str:
    url = ""
    raw = part.get("image_url")
    if isinstance(raw, Mapping):
        url = str(raw.get("url") or "")
    elif isinstance(raw, str):
        url = raw
    if url.startswith("data:"):
        mime = url[5:].split(";", 1)[0] or "image"
        size = int(len(url.split(",", 1)[-1]) * 3 / 4)
        return f"[image {n}: {mime}, {format_chars(size).replace(' chars', 'B')}]"
    return f"[image {n}]"


def _content_parts(content: Any) -> tuple[str, list[str], int]:
    """``(display text, tokenized texts, image count)`` for one content."""
    from sharedagents_model.token_count import _texts_of

    texts, images = _texts_of({"content": content})
    if isinstance(content, list):
        shown: list[str] = []
        n = 0
        for part in content:
            if isinstance(part, Mapping):
                kind = str(part.get("type") or "")
                if kind == "text":
                    shown.append(str(part.get("text") or ""))
                elif kind in ("image_url", "image", "input_image"):
                    n += 1
                    shown.append(_image_placeholder(part, n))
                else:
                    shown.append(json.dumps(part, ensure_ascii=False))
            elif part:
                shown.append(str(part))
        return "\n".join(s for s in shown if s), texts, images
    return ("" if content is None else str(content)), texts, images


def _pretty_args(args: Any) -> str:
    if isinstance(args, (dict, list)):
        return json.dumps(args, ensure_ascii=False, indent=2)
    text = "" if args is None else str(args)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return text


def _label_for_block(text: str) -> tuple[str, str]:
    head = text[:120]
    for marker, kind, label in _PREAMBLE_LABELS:
        if marker in head:
            return kind, label
    if "source=memory:" in head or "[RECALLED CONTEXT" in head:
        return "preamble", "Recalled memory"
    return "preamble", "Rolling context"


def _split_live_message(
    text: str,
    pieces: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]] | None:
    """Split the next turn's rolling head back into its blocks.

    The engine joins the preamble blocks, the system prompt's dynamic suffix
    and the turn marker into the head of the live user message. ``pieces`` is
    that known sequence ``(text, kind, label)``; returns the ordered
    ``(kind, label, text)`` parts, or ``None`` when the text does not have the
    expected shape (then it stays one item).
    """
    pos = 0
    out: list[tuple[str, str, str]] = []
    for piece, kind, label in pieces:
        if not piece:
            continue
        if not text.startswith(piece, pos):
            return None
        out.append((kind, label, piece))
        pos += len(piece)
        if text.startswith("\n\n", pos):
            pos += 2
    rest = text[pos:]
    marker = None
    for m in _TURN_MARKER_RE.finditer(rest):
        marker = m
    if marker is None:
        return None
    between = rest[: marker.start()].rstrip("\n")
    if between:
        kind, label = _label_for_block(between)
        out.append((kind, label, between))
    tail = rest[marker.start() :]
    out.append(("marker", f"Turn {marker.group(1)} · next message goes here", tail))
    return out


def items_from_messages(
    messages: list[Mapping[str, Any]],
    *,
    tools: list[Mapping[str, Any]] | None = None,
    live_pieces: list[tuple[str, str, str]] | None = None,
    drop_live: bool = False,
) -> list[ContextItem]:
    """Ordered items for an OpenAI-shaped message list (+ ``tools[]``).

    The order is the order the model reads: the system message, the tool
    definitions (chat templates render them right after the system head),
    then every message as sent. An assistant message with tool calls becomes
    one item for its reasoning, its text and each call. ``live_pieces``
    splits the rolling head of the final user message (see
    :func:`_split_live_message`); ``drop_live`` omits that message.
    """
    from sharedagents_core.attachments import is_tool_image_message
    from sharedagents_model.token_count import (
        IMAGE_TOKENS,
        TEMPLATE_TOKENS_PER_MESSAGE,
        TOOLS_FRAMING_TOKENS,
    )

    items: list[ContextItem] = []
    msgs = [m for m in (messages or []) if isinstance(m, Mapping)]
    last = len(msgs) - 1
    turn: int | None = None
    call_names: dict[str, str] = {}

    def add(item: ContextItem) -> ContextItem:
        items.append(item)
        return item

    tools_added = False

    def add_tools() -> None:
        nonlocal tools_added
        if tools_added or not tools:
            return
        tools_added = True
        names = [str(((t.get("function") or {}) if isinstance(t, Mapping) else {}).get("name") or "") for t in tools]
        add(
            ContextItem(
                kind="tools",
                role="tools",
                label=f"Tool definitions · {len(tools)} tools",
                content=(
                    ", ".join(n for n in names if n) + "\n\n" + json.dumps(list(tools), ensure_ascii=False, indent=1)
                ),
                _texts=[json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))],
                _overhead=TOOLS_FRAMING_TOKENS,
            )
        )

    for i, m in enumerate(msgs):
        role = str(m.get("role") or "")
        if role != "system":
            add_tools()
        frame = TEMPLATE_TOKENS_PER_MESSAGE
        content = m.get("content")
        if role == "system":
            shown, texts, images = _content_parts(content)
            add(
                ContextItem(
                    kind="system",
                    role=role,
                    label="System prompt",
                    content=shown,
                    _texts=texts,
                    _overhead=frame + images * IMAGE_TOKENS,
                    images=images,
                )
            )
            continue
        if role == "user":
            if i == last and drop_live:
                continue
            shown, texts, images = _content_parts(content)
            if i == last and live_pieces is not None and isinstance(content, str):
                parts = _split_live_message(content, live_pieces)
                if parts:
                    for k, (kind, label, text) in enumerate(parts):
                        add(
                            ContextItem(
                                kind=kind,
                                role=role,
                                label=label,
                                content=text,
                                turn=turn,
                                _texts=[text],
                                _overhead=frame if k == 0 else 0,
                            )
                        )
                    continue
            marker = _TURN_MARKER_RE.match(shown)
            if marker:
                turn = int(marker.group(1))
                label = f"Turn {turn} · prompt"
                kind = "user"
            elif is_tool_image_message(dict(m)):
                kind, label = "image", (f"Turn {turn} · tool image" if turn is not None else "Tool image")
            elif shown.startswith("<sharedagents-canonical-handoff>"):
                kind, label = "preamble", "Handoff"
            elif shown.startswith("…[truncated:"):
                kind, label = "notice", "Older context omitted"
            else:
                kind = "user"
                label = f"Turn {turn} · message" if turn is not None else "Message"
            add(
                ContextItem(
                    kind=kind,
                    role=role,
                    label=label,
                    content=shown,
                    turn=turn,
                    images=images,
                    _texts=texts,
                    _overhead=frame + images * IMAGE_TOKENS,
                )
            )
            continue
        if role == "assistant":
            prefix = f"Turn {turn} · " if turn is not None else ""
            first = True
            reasoning = m.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                add(
                    ContextItem(
                        kind="reasoning",
                        role=role,
                        label=f"{prefix}reasoning",
                        content=reasoning,
                        turn=turn,
                        _texts=[reasoning],
                        _overhead=frame,
                    )
                )
                first = False
            shown, texts, images = _content_parts(content)
            if shown or not m.get("tool_calls"):
                add(
                    ContextItem(
                        kind="assistant",
                        role=role,
                        label=f"{prefix}{'aborted' if 'aborted by user]' in shown[:80] else 'reply'}",
                        content=shown,
                        turn=turn,
                        images=images,
                        _texts=texts,
                        _overhead=(frame if first else 0) + images * IMAGE_TOKENS,
                    )
                )
                first = False
            for call in m.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                name = str(fn.get("name") or call.get("name") or "tool")
                args = fn.get("arguments")
                raw_args = (
                    args
                    if isinstance(args, str)
                    else json.dumps(
                        args if args is not None else {},
                        ensure_ascii=False,
                    )
                )
                cid = str(call.get("id") or "")
                if cid:
                    call_names[cid] = name
                add(
                    ContextItem(
                        kind="tool_call",
                        role=role,
                        label=f"{prefix}{name}",
                        content=_pretty_args(args),
                        turn=turn,
                        _texts=[name, raw_args],
                        _overhead=(frame if first else 0) + 8,
                    )
                )
                first = False
            continue
        if role == "tool":
            name = call_names.get(str(m.get("tool_call_id") or ""), "tool")
            shown, texts, images = _content_parts(content)
            prefix = f"Turn {turn} · " if turn is not None else ""
            add(
                ContextItem(
                    kind="tool_result",
                    role=role,
                    label=f"{prefix}{name} result",
                    content=shown,
                    turn=turn,
                    images=images,
                    _texts=texts,
                    _overhead=frame + images * IMAGE_TOKENS,
                )
            )
            continue
        shown, texts, images = _content_parts(content)
        add(
            ContextItem(
                kind=role or "message",
                role=role,
                label=role or "message",
                content=shown,
                turn=turn,
                images=images,
                _texts=texts,
                _overhead=frame + images * IMAGE_TOKENS,
            )
        )
    add_tools()
    return items


def count_item_tokens(items: list[ContextItem], tokenizer: Any) -> None:
    """Fill ``item.tokens`` with the pre-send counter's per-item share.

    With a tokenizer the shares add up exactly to
    :func:`sharedagents_model.token_count.count_payload_tokens` (minus the fixed
    generation prompt); without one each share is the character estimate.
    """
    from sharedagents_model.token_count import FALLBACK_CHARS_PER_TOKEN

    flat: list[tuple[int, str]] = []
    for idx, item in enumerate(items):
        for text in item._texts:
            if text:
                flat.append((idx, text))
    counts = [0] * len(items)
    encoded = False
    if tokenizer is not None and flat:
        try:
            encs = tokenizer.encode_batch([t for _, t in flat], add_special_tokens=False)
            for (idx, _), enc in zip(flat, encs):
                counts[idx] += len(enc.ids)
            encoded = True
        except Exception:  # noqa: BLE001 — fall back to the estimate
            counts = [0] * len(items)
    if not encoded:
        chars = [0] * len(items)
        for idx, text in flat:
            chars[idx] += len(text)
        counts = [int(c / FALLBACK_CHARS_PER_TOKEN) for c in chars]
    for idx, item in enumerate(items):
        item.tokens = int(counts[idx] + item._overhead)


# ── session inputs ───────────────────────────────────────────────────────


def _native_system_prompt(agent: Any, session: Any) -> str:
    """The system prompt the next turn uses (``Agent.run_turn``'s rule)."""
    sid = str(getattr(session, "hash", "") or "")
    try:
        mode = agent._effective_mode(session)
    except Exception:  # noqa: BLE001
        mode = str(getattr(session, "system_mode", "") or "auto")
    cwd = str(getattr(session, "workspace_cwd", "") or "")
    origin = str(getattr(session, "workspace_origin", "") or "")
    cache = getattr(agent, "_system_prompt_cache", None) or {}
    cached = cache.get(sid) if isinstance(cache, Mapping) else None
    if cached is not None and cached[0] == (mode, cwd, origin):
        return str(cached[1] or "")
    compose = getattr(agent, "compose_system_prompt", None)
    if callable(compose):
        try:
            return str(
                compose(
                    mode=mode,
                    workspace_cwd=cwd,
                    workspace_origin=origin,
                )
                or ""
            )
        except Exception:  # noqa: BLE001 — the persisted copy is the fallback
            pass
    return str(getattr(session, "system_prompt", "") or "")


def _native_preamble(agent: Any, session: Any) -> list[dict[str, Any]]:
    """Pinned facts, compaction summary, todos and the live budget block."""
    out: list[dict[str, Any]] = []
    compose = getattr(agent, "compose_round_preamble", None)
    if callable(compose):
        try:
            out = list(compose(session, "") or [])
        except Exception:  # noqa: BLE001
            out = []
    try:
        from sharedagents_goal.goal_service import live_budget_preamble_block

        block = live_budget_preamble_block(session)
    except Exception:  # noqa: BLE001 — display only
        block = None
    if block:
        out.append(block)
    return out


def _per_prompt_notes(cfg: Any) -> list[str]:
    notes: list[str] = []
    recall = str(getattr(cfg, "memory_recall_mode", "") or "").lower()
    skills = getattr(cfg, "skill_auto_invoke", None)
    skills_on = bool(getattr(skills, "enabled", False) or (isinstance(skills, Mapping) and skills.get("enabled")))
    if recall in ("context", "hybrid") or skills_on:
        notes.append(
            "Recalled memory and auto-invoked skills are picked from your next message, so they are not listed."
        )
    return notes


def _model_route(engine: Any, session: Any) -> tuple[str, str]:
    cfg = getattr(engine, "cfg", None)
    endpoint = str(getattr(session, "endpoint", "") or getattr(cfg, "upstream_endpoint", "") or "")
    model_id = str(getattr(session, "model", "") or "")
    if not model_id or model_id.lower() in ("auto", "ask", "plan"):
        model_id = str(getattr(cfg, "upstream_model", "") or model_id)
    return endpoint, model_id


def _tokenizer_for(engine: Any, session: Any) -> Any:
    from sharedagents_model import token_count

    endpoint, model_id = _model_route(engine, session)
    try:
        row = engine.cfg.model_row_for_endpoint(endpoint, model_id) or {}
    except Exception:  # noqa: BLE001
        row = {}
    return token_count.tokenizer_for_row(row)


def _wire_payload(
    engine: Any,
    session: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Messages and ``tools[]`` exactly as the worker request carries them."""
    try:
        from sharedagents_gateway.worker_request import _build_worker_request

        allowed = None
        allowed_fn = getattr(engine, "_round_allowed_tools", None)
        if callable(allowed_fn):
            allowed = allowed_fn(session)
        payload = _build_worker_request(
            SimpleNamespace(cfg=engine.cfg),
            session,
            list(messages),
            allowed_tool_names=allowed,
        )
        return list(payload.get("messages") or messages), (payload.get("tools") or None)
    except Exception:  # noqa: BLE001 — show the engine's list without tools
        return list(messages), None


def _calibrated_prompt(engine: Any, session: Any, raw: int, tokenizer: Any) -> int:
    """The pre-send budget's ``prompt`` figure for a raw count.

    Mirrors ``CompactionMixin._presend_budget``: a tokenizer count gets the
    learned template offset, a character estimate the learned ratio.
    """
    sid = str(getattr(session, "hash", "") or "")
    if tokenizer is None:
        ratio = (getattr(engine, "_presend_ratio", None) or {}).get(sid, 1.0)
        return int(raw * ratio)
    return int(raw + (getattr(engine, "_presend_offset", None) or {}).get(sid, 0))


def _window(engine: Any, session: Any) -> int:
    fn = getattr(engine, "_effective_context_window", None)
    if callable(fn):
        try:
            value = int(fn(session) or 0)
            if value > 0:
                return value
        except Exception:  # noqa: BLE001
            pass
    return int(getattr(session, "context_window", 0) or 0)


def _snapshot_id(session_id: str, items: list[ContextItem]) -> str:
    h = hashlib.sha1(session_id.encode("utf-8"))
    for item in items:
        h.update(b"\x00")
        h.update(item.kind.encode())
        h.update(b"\x01")
        h.update(item.label.encode("utf-8", "replace"))
        h.update(b"\x01")
        h.update(item.content.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _totals(
    items: list[ContextItem],
    *,
    tokens: int,
    meter: int,
    window: int,
    tokens_exact: bool,
    source: str,
) -> dict[str, Any]:
    pct = round(tokens * 100.0 / window, 1) if window > 0 else None
    chars = sum(len(i.content) for i in items)
    used = format_tokens(tokens)
    label = f"{used} / {format_tokens(window)}" if window > 0 else used
    if pct is not None:
        label += f" · {pct:g}%"
    return {
        "items": len(items),
        "tokens": int(tokens),
        "items_tokens": int(sum(i.tokens for i in items)),
        "meter_tokens": int(meter),
        "window": int(window),
        "percent": pct,
        "chars": chars,
        "tokens_exact": bool(tokens_exact),
        "tokens_source": source,
        "label": label,
        "meter_label": (f"Last request {format_tokens(meter)}" if meter > 0 and source != "engine" else ""),
        "chars_label": format_chars(chars),
    }


def build_native_snapshot(agent: Any, session: Any, *, session_id: str) -> ContextSnapshot:
    """The next round's request for a native (local_llm) engine."""
    from sharedagents_model.token_count import count_payload_tokens

    engine = agent.engine
    system_prompt = _native_system_prompt(agent, session)
    preamble = _native_preamble(agent, session)
    messages = engine._build_messages_simple(session, "", system_prompt, preamble)
    wire, tools = _wire_payload(engine, session, messages)

    dyn_suffix = ""
    try:
        from sharedagents_agent.system_prompt import split_system_prompt_cache_boundary

        split = split_system_prompt_cache_boundary(system_prompt or "")
        if split is not None:
            dyn_suffix = split["dynamic_suffix"]
    except Exception:  # noqa: BLE001
        dyn_suffix = ""
    pieces: list[tuple[str, str, str]] = []
    for block in preamble:
        text = str((block or {}).get("content") or "").strip()
        if text:
            kind, label = _label_for_block(text)
            pieces.append((text, kind, label))
    if dyn_suffix:
        pieces.append((dyn_suffix, "system", "System prompt · workspace context"))

    items = expand_system_sections(
        items_from_messages(wire, tools=tools, live_pieces=pieces),
        _prompt_sections(agent, session),
        _agents_md_body(agent),
    )
    tokenizer = _tokenizer_for(engine, session)
    count_item_tokens(items, tokenizer)
    # The total is the pre-send counter's own figure over the same payload
    # (per-item shares are rounded separately when estimating).
    raw = count_payload_tokens(wire, tools, tokenizer)
    tokens = _calibrated_prompt(engine, session, raw, tokenizer)
    totals = _totals(
        items,
        tokens=tokens,
        meter=int(getattr(session, "last_prompt_tokens", 0) or 0),
        window=_window(engine, session),
        tokens_exact=tokenizer is not None,
        source="request",
    )
    return ContextSnapshot(
        snapshot_id=_snapshot_id(session_id, items),
        session_id=session_id,
        agent=str(getattr(agent, "name", "") or ""),
        engine=str(getattr(getattr(engine, "kind", ""), "value", "") or "local_llm"),
        scope="model_input",
        notice="",
        items=items,
        totals=totals,
        notes=_per_prompt_notes(getattr(agent, "cfg", None)),
    )


def _engine_display(kind: Any) -> str:
    try:
        from sharedagents_agent.external_registry import spec_for

        return str(spec_for(kind).display_name or "")
    except Exception:  # noqa: BLE001
        raw = str(getattr(kind, "value", kind) or "").replace("_", " ").strip()
        return raw or "This engine"


def _last_reported(session: Any) -> tuple[int, int]:
    """``(prompt_tokens, context_window)`` the engine last reported."""
    pt = int(getattr(session, "last_prompt_tokens", 0) or 0)
    cw = int(getattr(session, "context_window", 0) or 0)
    for turn in reversed(list(getattr(session, "turns", None) or [])):
        metrics = turn.get("metrics") if isinstance(turn, Mapping) else None
        if not isinstance(metrics, Mapping):
            continue
        if not pt:
            try:
                pt = int(metrics.get("prompt_tokens") or 0)
            except (TypeError, ValueError):
                pt = 0
        if not cw:
            try:
                cw = int(metrics.get("context_window") or 0)
            except (TypeError, ValueError):
                cw = 0
        if pt and cw:
            break
    return pt, cw


def build_external_snapshot(agent: Any, session: Any, *, session_id: str) -> ContextSnapshot:
    """What sharedagents holds for an engine that keeps its own context."""
    from sharedagents_agent.agents.local_llm.injection import InjectionMixin

    engine = getattr(agent, "engine", None)
    kind = getattr(engine, "kind", "")
    display = _engine_display(kind)
    sid = str(getattr(session, "hash", "") or session_id)
    cache = getattr(agent, "_system_prompt_cache", None) or {}
    cached = cache.get(sid) if isinstance(cache, Mapping) else None
    system_prompt = str((cached[1] if cached is not None else "") or getattr(session, "system_prompt", "") or "")
    # The transcript through the same renderer the native engine replays with,
    # without a system message (shown whole below) or a next turn.
    view = copy.copy(session)
    try:
        view.system_prompt = ""
    except Exception:  # noqa: BLE001
        view = SimpleNamespace(**{**vars(session), "system_prompt": ""})
    builder = InjectionMixin()
    builder.cfg = getattr(agent, "cfg", None)
    builder._injected_messages = {}
    messages = builder._build_messages_simple(view, "", None, [])
    items: list[ContextItem] = []
    if system_prompt:
        items.extend(
            expand_system_sections(
                [
                    ContextItem(
                        kind="system",
                        role="system",
                        label="Instructions sharedagents sent",
                        content=system_prompt,
                        _texts=[system_prompt],
                    )
                ],
                _prompt_sections(agent, session),
                _agents_md_body(agent),
            )
        )
    items.extend(items_from_messages(messages, drop_live=True))
    count_item_tokens(items, None)
    reported, window = _last_reported(session)
    tokens = reported if reported > 0 else sum(i.tokens for i in items)
    totals = _totals(
        items,
        tokens=tokens,
        meter=reported,
        window=window,
        tokens_exact=False,
        source="engine" if reported > 0 else "estimate",
    )
    notice = (
        f"{display} keeps its own context, which sharedagents cannot read. This list "
        "shows the instructions sharedagents sent and the conversation it recorded."
    )
    if reported > 0:
        notice += f" The size above is what {display} last reported."
    return ContextSnapshot(
        snapshot_id=_snapshot_id(session_id, items),
        session_id=session_id,
        agent=str(getattr(agent, "name", "") or ""),
        engine=str(getattr(kind, "value", kind) or ""),
        scope="sharedagents_record",
        notice=notice,
        items=items,
        totals=totals,
        notes=[],
    )


# ── snapshot cache + pages ───────────────────────────────────────────────


class _SnapshotCache:
    def __init__(self, max_entries: int = _SNAPSHOT_MAX, ttl_s: float = _SNAPSHOT_TTL_S):
        self._max = max_entries
        self._ttl = ttl_s
        self._data: OrderedDict[tuple[str, str], ContextSnapshot] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str, snapshot_id: str) -> ContextSnapshot | None:
        key = (session_id, snapshot_id)
        with self._lock:
            snap = self._data.get(key)
            if snap is None:
                return None
            if time.monotonic() - snap.created_at > self._ttl:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return snap

    def put(self, snap: ContextSnapshot) -> None:
        key = (snap.session_id, snap.snapshot_id)
        with self._lock:
            self._data[key] = snap
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


SNAPSHOTS = _SnapshotCache()
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


def _resolve_session(backend: Any, session_id: str) -> Any:
    """Load the next turn's session through the same path as Agent.run_turn.

    The core's attachment cache can predate both completed turns and
    compaction: turn execution loads a separate Session from the durable log.
    Do not use that cache or register a read as attached on the core.
    """
    agent = getattr(backend, "agent", None)
    getter = getattr(agent, "get_session", None)
    return getter(session_id) if callable(getter) else None


def build_snapshot(backend: Any, session_id: str) -> ContextSnapshot | None:
    """Build (and cache) the context snapshot for one session."""
    session = _resolve_session(backend, session_id)
    if session is None:
        return None
    agent = backend.agent
    native = hasattr(getattr(agent, "engine", None), "_build_messages_simple")
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.setdefault(session_id, threading.Lock())
    with lock:
        last_exc: BaseException | None = None
        for _attempt in range(2):
            try:
                snap = (
                    build_native_snapshot(agent, session, session_id=session_id)
                    if native
                    else build_external_snapshot(agent, session, session_id=session_id)
                )
                break
            except RuntimeError as exc:  # a live turn mutated a list mid-read
                last_exc = exc
        else:
            raise last_exc  # type: ignore[misc]
    SNAPSHOTS.put(snap)
    return snap


def _parse_cursor(cursor: str) -> tuple[str, int]:
    raw = str(cursor or "").strip()
    if not raw:
        return "", 0
    snap, _, off = raw.partition(":")
    try:
        offset = max(0, int(off))
    except ValueError:
        offset = 0
    return snap, offset


def _clamp(value: Any, default: int, low: int, high: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, v))


def _item_json(item: ContextItem, index: int, *, instructions: bool) -> dict[str, Any]:
    withheld = item.kind in _INSTRUCTION_KINDS and not instructions
    chars = len(item.content)
    content = "" if withheld else item.content[:PAGE_ITEM_CHARS]
    return {
        "index": index,
        "kind": item.kind,
        "kind_label": KIND_LABELS.get(item.kind, item.kind.replace("_", " ").title()),
        "role": item.role,
        "label": item.label,
        "turn": item.turn,
        "tokens": item.tokens,
        "tokens_label": format_tokens(item.tokens),
        "chars": chars,
        "chars_label": format_chars(chars),
        "images": item.images,
        "preview": WITHHELD_TEXT if withheld else _preview(item.content),
        "content": content,
        "truncated": (not withheld) and chars > PAGE_ITEM_CHARS,
        "withheld": withheld,
    }


def _page_json(
    snap: ContextSnapshot,
    offset: int,
    limit: int,
    max_bytes: int,
    *,
    instructions: bool,
) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    used = 0
    end = offset
    for index in range(offset, min(len(snap.items), offset + limit)):
        row = _item_json(snap.items[index], index, instructions=instructions)
        cost = len(row["content"]) + len(row["preview"]) + len(row["label"]) + 200
        if out and used + cost > max_bytes:
            break
        out.append(row)
        used += cost
        end = index + 1
    return {
        "ok": True,
        "session_id": snap.session_id,
        "snapshot": snap.snapshot_id,
        "agent": snap.agent,
        "engine": snap.engine,
        "scope": snap.scope,
        "notice": snap.notice,
        "notes": list(snap.notes),
        "totals": dict(snap.totals),
        "offset": offset,
        "items": out,
        "next_cursor": f"{snap.snapshot_id}:{end}" if end < len(snap.items) else None,
        "empty_text": "" if snap.items else "Nothing has been sent to the model yet.",
    }


def context_page(
    backend: Any,
    session_id: str,
    *,
    cursor: str = "",
    limit: Any = None,
    max_bytes: Any = None,
    instructions: bool = True,
) -> tuple[int, dict[str, Any]]:
    """One page of a session's context. Returns ``(http status, payload)``."""
    limit_n = _clamp(limit, PAGE_LIMIT_DEFAULT, 1, PAGE_LIMIT_MAX)
    budget = _clamp(max_bytes, PAGE_BYTES_DEFAULT, 4096, PAGE_BYTES_MAX)
    snap_id, offset = _parse_cursor(cursor)
    snap = SNAPSHOTS.get(session_id, snap_id) if snap_id else None
    if snap is None:
        snap = build_snapshot(backend, session_id)
        if snap is None:
            return 404, {"ok": False, "code": "not_found", "message": "Session not found"}
        if snap_id and snap.snapshot_id != snap_id:
            return 409, {
                "ok": False,
                "code": "context_changed",
                "message": "The context changed. Reload it from the top.",
            }
    return 200, _page_json(snap, offset, limit_n, budget, instructions=instructions)


def context_item_range(
    backend: Any,
    session_id: str,
    index: Any,
    *,
    snapshot: str,
    offset: Any = 0,
    length: Any = None,
    instructions: bool = True,
) -> tuple[int, dict[str, Any]]:
    """A character range of one item's full content."""
    snap = SNAPSHOTS.get(session_id, str(snapshot or "")) if snapshot else None
    if snap is None:
        snap = build_snapshot(backend, session_id)
        if snap is None:
            return 404, {"ok": False, "code": "not_found", "message": "Session not found"}
        if snapshot and snap.snapshot_id != snapshot:
            return 409, {
                "ok": False,
                "code": "context_changed",
                "message": "The context changed. Reload it from the top.",
            }
    try:
        idx = int(index)
    except (TypeError, ValueError):
        idx = -1
    if idx < 0 or idx >= len(snap.items):
        return 404, {"ok": False, "code": "not_found", "message": "Item not found"}
    item = snap.items[idx]
    if item.kind in _INSTRUCTION_KINDS and not instructions:
        return 403, {"ok": False, "code": "withheld", "message": WITHHELD_TEXT}
    start = _clamp(offset, 0, 0, len(item.content))
    size = _clamp(length, RANGE_CHARS_DEFAULT, 1, RANGE_CHARS_MAX)
    chunk = item.content[start : start + size]
    end = start + len(chunk)
    return 200, {
        "ok": True,
        "snapshot": snap.snapshot_id,
        "index": idx,
        "offset": start,
        "content": chunk,
        "total_chars": len(item.content),
        "next_offset": end if end < len(item.content) else None,
    }


__all__ = [
    "ContextItem",
    "ContextSnapshot",
    "SNAPSHOTS",
    "build_snapshot",
    "build_native_snapshot",
    "build_external_snapshot",
    "context_item_range",
    "context_page",
    "count_item_tokens",
    "format_tokens",
    "items_from_messages",
]
