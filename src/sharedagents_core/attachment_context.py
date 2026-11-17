"""Saved attachments: one model-facing context, kept out of the transcript.

A chat upload is saved on disk and travels as a file *reference*
(``{path, name, mime, size}``). The model needs to know where the original
lives and how to sample it; the person who sent it only needs to see the
words they typed plus a chip per file. This module owns both halves:

* :func:`context_text` / :func:`model_text` / :func:`model_user` build the
  text the model receives. The format is the one the Web UI used to append to
  the prompt itself, so an agent sees exactly what it saw before (now with
  the real size read from disk).
* :func:`split_legacy` and :func:`display_user` take that block back out of a
  recorded prompt for display. Older sessions stored the block inside the
  user text; a block is removed only when the parsed references rebuild it
  byte for byte, so text a person typed that merely looks similar stays.

References ride an ACP content block's ``_meta`` slot (:data:`META_KEY`),
which every prompt flattener ignores and which survives gateway hops, the
same way the turn author does.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
from collections.abc import Mapping
from typing import Any

META_KEY = "sharedagents/attachments"
META_KEYS_READ = (META_KEY,)
MAX_FILES = 64
_MAX_NAME = 255
_MAX_PATH = 4096

GUIDANCE = (
    "[Saved attachments]\n"
    "For tabular data, use Python to inspect the schema and a small row sample "
    "before choosing an analysis. Keep full datasets on disk. For CSV/TSV, run "
    "python -m sharedagents_core.attachment_sampling with the saved paths; inspect "
    "any metadata rows before selecting column headers."
)

_BLOCK_RE = re.compile(
    r"\[ATTACHED FILE: (?P<name>[^\n]*)\]\n"
    r"Original file saved at: (?P<path>\"(?:[^\"\\\n]|\\.)*\")\n"
    r"Size: (?P<size>\d+) bytes\. Read the saved original as needed\.\n---"
)


def _guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or ""


def normalize_files(raw: Any, *, stat: bool = True, limit: int | None = MAX_FILES) -> list[dict[str, Any]]:
    """Clean file references: absolute path, display name, MIME type, size.

    Entries without an absolute path are dropped. With ``stat`` the size is
    the file's current size on disk (falling back to the given size when the
    file cannot be read).
    """
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if limit is not None and len(out) >= limit:
            break
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or "").strip()
        if not path or len(path) > _MAX_PATH or "\n" in path or not os.path.isabs(path):
            continue
        name = str(item.get("name") or "").strip().replace("\n", " ")
        if not name:
            name = os.path.basename(path) or "attachment"
        name = name[:_MAX_NAME]
        try:
            size = max(0, int(item.get("size") or 0))
        except (TypeError, ValueError, OverflowError):
            size = 0
        if stat:
            try:
                size = os.stat(path).st_size
            except OSError:
                pass
        mime = str(item.get("mime") or "").strip()[:128] or _guess_mime(name)
        out.append({"path": path, "name": name, "mime": mime, "size": size})
    return out


def _file_block(file: Mapping[str, Any]) -> str:
    return (
        f"[ATTACHED FILE: {file['name']}]\n"
        f"Original file saved at: {json.dumps(str(file['path']), ensure_ascii=False)}\n"
        f"Size: {int(file.get('size') or 0)} bytes. Read the saved original as needed.\n"
        "---"
    )


def context_text(files: Any) -> str:
    """The model-facing attachment context, or ``""`` without references."""
    refs = normalize_files(files, stat=False, limit=None)
    if not refs:
        return ""
    return GUIDANCE + "\n\n" + "\n\n".join(_file_block(f) for f in refs)


def model_text(text: Any, files: Any) -> str:
    """The prompt text the model receives: context first, typed text after."""
    prompt = str(text or "")
    context = context_text(files)
    if not context:
        return prompt
    return context + ("\n\n" + prompt if prompt else "")


def model_user(user: Any, files: Any) -> Any:
    """A recorded ``user`` value (string or content parts) with its context.

    Used wherever a stored turn is replayed to a model. Without references the
    value is returned unchanged.
    """
    context = context_text(files)
    if not context:
        return user
    if isinstance(user, list):
        parts = [dict(p) if isinstance(p, Mapping) else p for p in user]
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = model_text(part.get("text"), files)
                return parts
        return [{"type": "text", "text": context}, *parts]
    return model_text(user if isinstance(user, str) else "", files)


def model_blocks(blocks: Any) -> list[dict[str, Any]]:
    """ACP blocks with the attachment context folded into the text.

    For paths that hand blocks straight to a model (no turn record).
    """
    items = [dict(b) for b in blocks or [] if isinstance(b, Mapping)]
    files = files_from_blocks(items)
    if not files:
        return items
    for block in items:
        if block.get("type") == "text":
            block["text"] = model_text(block.get("text"), files)
            return items
    return [{"type": "text", "text": context_text(files)}, *items]


def with_files_meta(block: Mapping[str, Any], files: Any) -> dict[str, Any]:
    """Copy of an ACP block carrying ``files`` in its ``_meta`` slot."""
    out = dict(block)
    meta = dict(out["_meta"]) if isinstance(out.get("_meta"), Mapping) else {}
    refs = normalize_files(files, stat=False, limit=None)
    if refs:
        meta[META_KEY] = refs
    else:
        meta.pop(META_KEY, None)
    if meta:
        out["_meta"] = meta
    else:
        out.pop("_meta", None)
    return out


def strip_files_meta(blocks: Any) -> list[dict[str, Any]]:
    """Copies of ``blocks`` with every attachment reference removed."""
    out: list[dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        copy = dict(block)
        meta = copy.get("_meta")
        if isinstance(meta, Mapping) and any(k in meta for k in META_KEYS_READ):
            meta = {k: v for k, v in meta.items() if k not in META_KEYS_READ}
            if meta:
                copy["_meta"] = meta
            else:
                copy.pop("_meta", None)
        out.append(copy)
    return out


def files_from_blocks(blocks: Any) -> list[dict[str, Any]]:
    """Every attachment reference carried in the blocks' ``_meta`` slots."""
    found: list[Any] = []
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        meta = block.get("_meta")
        if isinstance(meta, Mapping):
            for key in META_KEYS_READ:
                if isinstance(meta.get(key), list):
                    found.extend(meta[key])
    return normalize_files(found, stat=False)


def split_legacy(text: Any) -> tuple[str, list[dict[str, Any]]]:
    """Split a recorded prompt into ``(typed text, file references)``.

    Matches only the exact saved-attachment block at the start of the text:
    the guidance paragraph, one block per saved file, then optionally a blank
    line and the typed text. The references must rebuild the block byte for
    byte; anything else returns the text unchanged with no references.
    """
    raw = text if isinstance(text, str) else ""
    head = GUIDANCE + "\n\n"
    if not raw.startswith(head):
        return raw, []
    pos = len(head)
    files: list[dict[str, Any]] = []
    while True:
        m = _BLOCK_RE.match(raw, pos)
        if m is None:
            return raw, []
        try:
            path = json.loads(m.group("path"))
        except ValueError:
            return raw, []
        if not isinstance(path, str):
            return raw, []
        files.append(
            {
                "path": path,
                "name": m.group("name"),
                "mime": _guess_mime(m.group("name")),
                "size": int(m.group("size")),
            }
        )
        pos = m.end()
        if raw.startswith("\n\n[ATTACHED FILE: ", pos):
            pos += 2
            continue
        break
    if pos == len(raw):
        rest = ""
    elif raw.startswith("\n\n", pos):
        rest = raw[pos + 2 :]
    else:
        return raw, []
    if context_text(files) != raw[:pos]:
        return raw, []
    return rest, files


def has_image_parts(user: Any) -> bool:
    if not isinstance(user, list):
        return False
    for part in user:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") in ("image", "image_url") or part.get("image_url"):
            return True
    return False


def display_files(files: Any, user: Any = None) -> list[dict[str, Any]]:
    """Chips for a transcript: name, size and type, never the saved path.

    Images that already ride the prompt as picture parts show as thumbnails,
    so they get no chip.
    """
    refs = normalize_files(files, stat=False, limit=None)
    inline = has_image_parts(user)
    out = []
    for ref in refs:
        if inline and str(ref.get("mime") or "").startswith("image/"):
            continue
        out.append({"name": ref["name"], "size": ref["size"], "mime": ref["mime"]})
    return out


def display_user(user: Any, files: Any = None) -> tuple[Any, list[dict[str, Any]]]:
    """``(user, references)`` for display: a legacy block split out.

    With ``files`` (a turn recorded with separate references) the user value
    is already the typed text and is returned unchanged.
    """
    refs = normalize_files(files, stat=False, limit=None)
    if refs:
        return user, refs
    if isinstance(user, str):
        clean, found = split_legacy(user)
        return (clean, found) if found else (user, [])
    if isinstance(user, list):
        for idx, part in enumerate(user):
            if isinstance(part, Mapping) and part.get("type") == "text":
                clean, found = split_legacy(part.get("text"))
                if not found:
                    return user, []
                parts = list(user)
                if clean:
                    parts[idx] = {**part, "text": clean}
                else:
                    parts.pop(idx)
                return parts, found
    return user, []


def project_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """A delivered log event in display form (same object when unchanged).

    ``turn_started`` / ``turn_ended`` get the typed text in ``user`` and chips
    in ``attached_files``; an injected follow-up gets the same for ``text``.
    Saved paths never leave in ``attached_files``.
    """
    etype = event.get("type")
    if etype in ("turn_started", "turn_ended"):
        user, refs = display_user(event.get("user"), event.get("attachments"))
        if not refs:
            return event
        out = {k: v for k, v in event.items() if k != "attachments"}
        out["user"] = user
        out["attached_files"] = display_files(refs, user)
        out["attachment_names"] = [ref["name"] for ref in refs]
        return out
    if etype == "queue_msg_injected":
        clean, refs = split_legacy(event.get("text"))
        if not refs:
            return event
        return {**event, "text": clean, "attached_files": display_files(refs)}
    if etype in ("queue_msg_added", "queue_msg_updated"):
        clean, refs = split_legacy(event.get("text"))
        if not refs:
            return event
        return {**event, "display_text": clean, "attached_files": display_files(refs)}
    return event


def project_queue_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """A pending follow-up with ``display_text`` + chips when it carries files.

    The queue stores model-facing text (``text`` is what a steer or an edit
    sends on); ``display_text`` is what a person typed.
    """
    out = dict(item)
    clean, refs = split_legacy(out.get("text"))
    if refs:
        out["display_text"] = clean
        out["attached_files"] = display_files(refs)
    return out


def keep_files_on_edit(old_text: Any, new_text: Any) -> str:
    """The edited text of a queued follow-up, keeping its saved files.

    An edit box shows only the typed text; the references of the original
    item are folded back unless the new text already carries a block.
    """
    edited = str(new_text or "")
    _, refs = split_legacy(old_text)
    if not refs or split_legacy(edited)[1]:
        return edited
    return model_text(edited, refs)


def merge_queue_text(items: Any) -> str:
    """Merge known queue entries, collecting their exact attachment prefixes.

    Parse each entry separately: searching inside arbitrary typed text would
    also hide examples of the old format that a person intentionally quoted.
    """
    texts = []
    files = []
    for item in items:
        clean, refs = split_legacy(item.get("text"))
        if clean.strip():
            texts.append(clean.strip())
        files.extend(refs)
    # Keep each original block, including queues larger than one upload's
    # admission limit. No files may disappear during aggregation.
    context = GUIDANCE + "\n\n" + "\n\n".join(_file_block(f) for f in files) if files else ""
    return "\n\n".join(part for part in (context, "\n\n".join(texts)) if part)


def project_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Split a folded turn's legacy block into ``user`` + ``attachments``.

    In place; also cleans injected-follow-up render segments. Replay to a
    model goes through :func:`model_user`, which restores the block.
    """
    if not turn.get("attachments"):
        user, refs = display_user(turn.get("user"))
        if refs:
            turn["user"] = user
            turn["attachments"] = refs
    if turn.get("attachments"):
        turn["attachment_names"] = [ref["name"] for ref in turn["attachments"]]
    for seg in turn.get("render_segments") or []:
        if isinstance(seg, dict) and seg.get("kind") == "user":
            clean, refs = split_legacy(seg.get("text"))
            if refs:
                seg["text"] = clean
                seg["attached_files"] = display_files(refs)
    return turn


__all__ = [
    "GUIDANCE",
    "META_KEY",
    "context_text",
    "display_files",
    "display_user",
    "files_from_blocks",
    "model_blocks",
    "model_text",
    "model_user",
    "keep_files_on_edit",
    "merge_queue_text",
    "normalize_files",
    "project_event",
    "project_queue_item",
    "project_turn",
    "split_legacy",
    "strip_files_meta",
    "with_files_meta",
]
