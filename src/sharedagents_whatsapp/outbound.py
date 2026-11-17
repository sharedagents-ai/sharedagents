"""Outbound queue for the running WhatsApp channel.

A second neonize client cannot share ``neonize.sqlite3`` with ``sharedagents
whatsapp run`` (the store locks; decrypt fails). Agents enqueue a line — and
optionally files — here; the live channel sends it to self-chat or a
send-allowlisted dest.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_whatsapp.config import WhatsAppConfig
from sharedagents_whatsapp.neonize_transport import linked_account, mask_user

_OTHER_DEST = re.compile(
    r"^(?:\+|00)\d{6,}|^\d{8,}@s\.whatsapp\.net$|^\d{8,}@lid$",
    re.I,
)
# Deliverable types an agent may queue as files. An allowlist, not a denylist:
# keys, stores, env files and archives have no business leaving by chat.
SEND_FILE_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".pdf",
        ".md",
        ".markdown",
        ".txt",
        ".csv",
    }
)


@dataclass(frozen=True)
class SendRequest:
    """Parsed ``send`` Skill args: the text, the dest and any ``--file`` paths."""

    text: str
    dest: str
    files: tuple[str, ...] = ()


def _take_path(raw: str) -> tuple[str, str]:
    """Split a leading path (``<…>``, quoted, or up to whitespace) from ``raw``."""
    closers = {"<": ">", '"': '"', "'": "'"}
    if raw[:1] in closers:
        end = raw.find(closers[raw[0]], 1)
        if end > 0:
            return raw[1:end].strip(), raw[end + 1 :]
    path, _, rest = raw.partition(" ")
    return path.strip(), rest


def outbound_path(home: Path) -> Path:
    return Path(home) / "outbound.jsonl"


def retry_path(home: Path) -> Path:
    """Jobs the channel could not send yet. Read before ``outbound.jsonl`` so
    they keep their place ahead of anything the skill queued meanwhile."""
    return Path(home) / "outbound.retry.jsonl"


def channel_running(home: Path) -> bool:
    pid_file = Path(home) / "whatsapp.pid"
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text().strip() or "0")
    except ValueError:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _split_dest(raw: str) -> tuple[str, str]:
    """``(text, dest)`` for option-free args that may open with a positional dest."""
    if not raw:
        return "", "self"
    first, _, rest = raw.partition(" ")
    if (
        _OTHER_DEST.match(first)
        or first.lower() not in {"self", "me", "you"}
        and (first.startswith("+") or "@" in first)
    ):
        return rest.strip(), first
    if first.lower() in {"self", "me", "you"}:
        return rest.strip(), "self"
    return raw, "self"


def _allowlist_labels(cfg: WhatsAppConfig | None) -> frozenset[str]:
    """Lowercased allowlist labels (full and pre-``/``) for multi-word ``--to``."""
    if cfg is None:
        return frozenset()
    labels: set[str] = set()
    for rows in (cfg.allowed_people, cfg.allowed_groups):
        for row in rows:
            label = str(row.get("label") or "").strip().lower()
            if not label:
                continue
            labels.add(label)
            labels.add(label.split("/", 1)[0].strip())
    return frozenset(labels)


def _take_named_dest(raw: str, labels: frozenset[str]) -> tuple[str, str]:
    """``(dest, rest)`` for a ``--to`` value.

    With ``labels`` (the channel allowlist), the longest prefix of the
    remaining words that exactly matches an allowlist label is the dest, so
    multi-word labels work; otherwise the first word is the dest.
    """
    if raw:
        words = raw.split(" ")
        if labels:
            for end in range(len(words), 0, -1):
                candidate = " ".join(words[:end])
                if candidate.lower() in labels:
                    return candidate, " ".join(words[end:]).strip()
    named, _, rest = raw.partition(" ")
    return named.strip(), rest.strip()


def parse_send_request(args: str, *, cfg: WhatsAppConfig | None = None) -> SendRequest:
    """Skill args ``send [--to <dest>] [--file <path>]… [text]``, options in any order.

    A path with spaces goes in ``<…>`` or quotes. With ``cfg`` (the channel
    allowlist), ``--to`` may be a multi-word label; without it, ``--to`` takes
    its first word as before.
    """
    raw = (args or "").strip()
    if raw.lower().startswith("send"):
        raw = raw[4:].strip()
    labels = _allowlist_labels(cfg)
    dest: str | None = None
    files: list[str] = []
    positional_tried = False
    while True:
        if raw.startswith("--to "):
            named, raw = _take_named_dest(raw[5:].strip(), labels)
            dest = named or "self"
        elif raw.startswith("--file "):
            path, raw = _take_path(raw[7:].strip())
            if path:
                files.append(path)
        elif dest is None and not positional_tried and raw:
            # ``send +55… --file …`` / ``send me --file …``: a positional dest
            # may lead, and options may still follow it.
            positional_tried = True
            rest, named = _split_dest(raw)
            if rest == raw:
                break
            dest, raw = named, rest
        else:
            break
        raw = raw.strip()
    return SendRequest(raw, dest or "self", tuple(files))


def parse_send_args(args: str) -> tuple[str, str]:
    """``(text, dest)`` from Skill args. ``dest`` is ``self`` unless another target was named."""
    request = parse_send_request(args)
    return request.text, request.dest


def dest_is_self(dest: str) -> bool:
    d = (dest or "self").strip().lower()
    return d in {"", "self", "me", "you"}


def resolve_send_dest(
    dest: str,
    cfg: WhatsAppConfig,
) -> tuple[str, dict | None]:
    """Map a skill dest to ``self`` or an allowlist row id.

    Accepts a JID, a phone, or an allowlist label (e.g. ``lia``).
    """
    raw = (dest or "self").strip()
    if dest_is_self(raw):
        return "self", None
    for group in (False, True):
        row = cfg.entry(raw, group=group)
        if row and str(row.get("id") or "").strip():
            return str(row["id"]).strip(), row
    needle = raw.lower().lstrip("@")
    if needle.endswith("@s.whatsapp.net") or needle.endswith("@g.us") or needle.endswith("@lid"):
        needle = needle.split("@", 1)[0]
    for rows in (cfg.allowed_people, cfg.allowed_groups):
        for row in rows:
            label = str(row.get("label") or "").strip().lower()
            ident = str(row.get("id") or "").strip()
            ident_user = ident.split("@", 1)[0].lower()
            label_first = label.split("/", 1)[0].strip()
            if needle in {label, label_first, ident.lower(), ident_user}:
                return ident, row
    return raw, None


def account_snapshot(*, home: Path | None = None) -> str:
    """Masked linked-account status. Never lists WhatsApp contacts."""
    cfg = WhatsAppConfig.load(home)
    account = linked_account(cfg.store_path)
    lines = ["## WhatsApp account"]
    if not account:
        lines.append("Not linked. Load this skill with args `qr` and scan the live QR.")
        return "\n".join(lines) + "\n"
    lines.append(f"Linked as **{mask_user(account['user'])}**.")
    lines.append("No contacts directory — use the Channels allowlist labels.")
    lines.append("Do not query `neonize.sqlite3` / `whatsmeow_contacts`.")
    if channel_running(cfg.home):
        lines.append(
            "Channel is running. `Skill` args `send <text>` (self-chat) or `send --to <label> <text>` "
            "for a send-allowlisted person/group; add `--file </abs/path>` (repeatable) to attach "
            "an image, PDF, Markdown/text or CSV file."
        )
    else:
        lines.append("Channel is not running. Start it with `sharedagents whatsapp run` before sending.")
    return "\n".join(lines) + "\n"


def enqueue_self_send(text: str, *, home: Path | None = None) -> str:
    """Queue ``text`` for the live channel to send to self-chat."""
    return enqueue_send(text, dest="self", home=home)


def validate_send_file(path: str, max_bytes: int) -> tuple[Path | None, str]:
    """``(resolved path, "")`` for a sendable file, else ``(None, reason)``.

    Absolute path to an existing regular file, a deliverable suffix
    (:data:`SEND_FILE_SUFFIXES`) and at most ``max_bytes``.
    """
    raw = Path(str(path or "")).expanduser()
    if not raw.is_absolute():
        return None, f"`{path}` is not an absolute path"
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError):
        return None, f"`{path}` does not exist"
    if not resolved.is_file():
        return None, f"`{path}` is not a regular file"
    if resolved.suffix.lower() not in SEND_FILE_SUFFIXES:
        kinds = " ".join(sorted(SEND_FILE_SUFFIXES))
        return None, f"`{resolved.name}` is not a sendable file type ({kinds})"
    size = resolved.stat().st_size
    if size > max_bytes:
        return None, (f"`{resolved.name}` is {size / 1_048_576:.1f} MB (limit {max_bytes / 1_048_576:.1f} MB)")
    return resolved, ""


def enqueue_send(
    text: str,
    dest: str = "self",
    *,
    home: Path | None = None,
    files: Iterable[str] = (),
) -> str:
    """Queue ``text`` and ``files`` for the live channel: self-chat or a send-allowlisted dest."""
    cfg = WhatsAppConfig.load(home)
    body = (text or "").strip()
    wanted = [str(item) for item in files or () if str(item).strip()]
    if not body and not wanted:
        return (
            "## WhatsApp send\nNo text. Use args `send hi`, `send --to <label> hi` "
            "or `send --file </abs/report.pdf>`.\n"
        )
    if not linked_account(cfg.store_path):
        return "## WhatsApp send\nNot linked. Load this skill with args `qr` first.\n"
    if not channel_running(cfg.home):
        return (
            "## WhatsApp send\n"
            "Channel is not running. Start `sharedagents whatsapp run`, then reload "
            "with args `send …`.\n"
            "Do not start a second neonize client and do not kill a running channel.\n"
        )
    ident, row = resolve_send_dest(dest, cfg)
    payload: dict[str, Any] = {"text": body}
    if ident != "self":
        group = bool(row) and str(row.get("id") or "").endswith("@g.us")
        if row is None or not cfg.allows(ident, group=group, access="send"):
            return refuse_other_dest(dest)
        payload["dest"] = ident
        who = str(row.get("label") or "allowlisted chat").split("/", 1)[0].strip()
        target = f"allowlisted chat **{who}**"
    else:
        target = "self-chat (**Mensagem para você mesmo**)"
    specs: list[dict[str, str]] = []
    for item in wanted:
        resolved, reason = validate_send_file(item, cfg.max_media_bytes)
        if resolved is None:
            return f"## WhatsApp send\nRefused: {reason}. Nothing was queued.\n"
        specs.append({"path": str(resolved), "name": resolved.name})
    if specs:
        payload["files"] = specs
    cfg.ensure_home()
    path = outbound_path(cfg.home)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    what = [f"{len(body)} chars"] if body else []
    if specs:
        what.append(", ".join(spec["name"] for spec in specs))
    return f"## WhatsApp send\nQueued for {target} ({'; '.join(what)}). The running channel delivers it.\n"


def _job_from(data: Any) -> dict[str, Any] | None:
    """One queue row → ``{text, dest[, files]}``; ``None`` when it carries nothing."""
    dest: str | None = None
    files: list[dict[str, str]] = []
    if isinstance(data, dict):
        text = str(data.get("text") or "").strip()
        dest = str(data.get("dest") or "").strip() or None
        for spec in data.get("files") or []:
            if isinstance(spec, dict) and str(spec.get("path") or "").strip():
                files.append(
                    {
                        "path": str(spec["path"]),
                        "name": str(spec.get("name") or Path(str(spec["path"])).name),
                    }
                )
    elif isinstance(data, str):
        text = data.strip()
    else:
        return None
    if not text and not files:
        return None
    job: dict[str, Any] = {"text": text, "dest": dest}
    if files:
        job["files"] = files
    return job


def _take_file(path: Path) -> list[dict[str, Any]]:
    """Atomically read and clear one queue file."""
    if not path.is_file():
        return []
    tmp = path.with_name(path.name + ".taking")
    try:
        path.replace(tmp)
    except FileNotFoundError:
        return []
    jobs: list[dict[str, Any]] = []
    try:
        for line in tmp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data: Any = json.loads(line)
            except ValueError:
                data = line
            job = _job_from(data)
            if job is not None:
                jobs.append(job)
    finally:
        tmp.unlink(missing_ok=True)
    return jobs


def take_outbound(home: Path) -> list[dict[str, Any]]:
    """Atomically read and clear queued jobs ``{text, dest[, files]}``, oldest first:
    jobs put back by :func:`requeue_outbound`, then newly queued ones."""
    return _take_file(retry_path(home)) + _take_file(outbound_path(home))


def requeue_outbound(home: Path, jobs: list[dict[str, Any]]) -> None:
    """Put unsent jobs back, ahead of anything queued since they were taken."""
    if not jobs:
        return
    with retry_path(home).open("a", encoding="utf-8") as fh:
        for job in jobs:
            fh.write(json.dumps(job, ensure_ascii=False) + "\n")


def refuse_other_dest(dest: str) -> str:
    del dest
    return (
        "## WhatsApp send\n"
        "Refused. Send only to the linked self-chat or a send-allowlisted "
        "person/group (use the Channels label, not a raw number).\n"
    )
