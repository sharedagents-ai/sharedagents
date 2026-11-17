"""Read side of the session log (append-only JSONL).

The on-disk session log is ``<agent_home>/sessions/<sid>.jsonl`` — one
event per LF-terminated line, each line prefixed with a CRC32 of the
JSON payload:

    <crc8> <json>\\n

where ``<crc8>`` is the 8-char lowercase hex of ``zlib.crc32`` over the
UTF-8 JSON bytes (everything right of the first space, excluding the
trailing newline) and ``<json>`` is
``json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))``.

The CRC lives OUTSIDE the JSON object (as a line prefix) on purpose:
an in-object ``_crc`` field is not byte-deterministic across json
serialisers, so a reader recomputing it would spuriously fail. The
prefix form lets the reader CRC exactly the on-disk bytes with no
re-serialisation. See ``session_writer.encode_line``.

Three readers consume this file:

* :func:`tail_jsonl` — incremental event stream from a byte offset,
  tolerant of a writer mid-append (partial trailing line) and of a
  corrupt line (CRC mismatch). Never raises on a malformed tail.
* :func:`fold_jsonl_into_turns` — reconstruct the legacy per-turn dict
  shape (the shape ``_render_one_saved_turn`` / telegram ``/history`` /
  memory ``on_session_end`` / the fine-tune harvester consume).
* The indexer's ``tail`` / ``rebuild`` (see ``session_index.py``) reuse
  :func:`tail_jsonl` to project events into ``sessions.db``.

Readers NEVER take the per-session lock. The writer's flock plus the
``\\n``-terminator-before-parse rule is the entire synchronisation
story; a reader that wants to stay consistent with ``sessions.db``
reads ``sessions_meta.live_last_event_offset`` first and bounds its
fold there (see the spec, §6).
"""

from __future__ import annotations

import codecs
import json
import logging
import re
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sharedagents_session.actors import normalize_actor, normalize_actors

_log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

# Process-wide count of dropped (CRC-mismatch / malformed) JSONL lines.
# A silently-dropped line is a hole in the fine-tune corpus, so every
# drop is logged AND counted. Readable for a Prometheus/metrics export.
SESSION_JSONL_CORRUPTION_TOTAL = 0


# ---- line codec -----------------------------------------------------


def _crc8(json_bytes: bytes) -> str:
    """8-char lowercase hex CRC32 of the JSON payload bytes."""
    return f"{zlib.crc32(json_bytes) & 0xFFFFFFFF:08x}"


def encode_line(event: dict[str, Any]) -> bytes:
    """Serialise one event to its on-disk line bytes (incl. trailing LF).

    The canonical wire form. ``session_writer`` calls this; tests assert
    against it. Pinned ``json.dumps`` flags make the bytes deterministic
    so the CRC round-trips.
    """
    from sharedagents_session.jsonl_protocol import CODEC_DUMP_KWARGS

    payload = json.dumps(event, **CODEC_DUMP_KWARGS).encode("utf-8")
    return _crc8(payload).encode("ascii") + b" " + payload + b"\n"


def decode_line(raw: bytes) -> dict[str, Any] | None:
    """Parse one on-disk line (without trailing LF) back to an event.

    Returns ``None`` when the line is malformed OR the CRC does not
    match — the caller treats both as "skip this line." Never raises.
    """
    if not raw:
        return None
    sp = raw.find(b" ")
    if sp != 8:
        # Prefix must be exactly 8 hex chars + a space. Anything else
        # is a torn/partial/foreign line.
        return None
    prefix = raw[:8]
    payload = raw[9:]
    if not payload:
        return None
    if _crc8(payload).encode("ascii") != prefix:
        return None
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


# ---- tail reader ----------------------------------------------------


def tail_jsonl(path: Path, start_offset: int = 0) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield ``(event, end_offset)`` for every complete, CRC-valid line
    at or after ``start_offset``.

    ``end_offset`` is the byte position immediately after the line's
    terminating ``\\n`` — feed it back as the next ``start_offset`` to
    resume without re-reading. A partial trailing line (writer mid-
    append, no ``\\n`` yet) is left unconsumed; the next call retries
    from the last complete line's end. A CRC-mismatched / malformed
    line is skipped (the caller may bump a corruption counter) and the
    scan advances past it so it is never re-read in a loop.

    Tolerant of a missing file (yields nothing) — a freshly created
    session may not have flushed its first line yet.
    """
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return
    try:
        f.seek(start_offset)
        buf = b""
        pos = start_offset  # byte offset of the start of ``buf``
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = buf[:nl]
                end_offset = pos + nl + 1
                event = decode_line(line)
                # Advance regardless of validity so a bad line is never
                # re-scanned on the next pass.
                buf = buf[nl + 1 :]
                if event is not None:
                    pos = end_offset
                    yield event, end_offset
                elif line:
                    # A non-empty line that failed CRC/parse is a
                    # corrupt event — count + log it (a silent drop is a
                    # hole in the fine-tune corpus). Empty lines (a bare
                    # \n) are ignored without noise.
                    global SESSION_JSONL_CORRUPTION_TOTAL
                    SESSION_JSONL_CORRUPTION_TOTAL += 1
                    _log.warning(
                        "session jsonl: dropped corrupt line at offset %d in %s",
                        pos,
                        path,
                    )
                    pos = end_offset
                else:
                    pos = end_offset
    finally:
        f.close()


#: First window for :func:`last_event` — one oversized ``tool_result`` line
#: without reading a multi-hundred-MB session backwards.
_LAST_EVENT_WINDOW = 64 << 10
#: Grown ONCE for a final record bigger than the first window; beyond this the
#: forward-scan fallback takes over.
_LAST_EVENT_WINDOW_MAX = 1 << 20


def _last_complete_line(buf: bytes, start: int) -> tuple[dict[str, Any], int] | None:
    """Newest decodable line in ``buf`` (file bytes from ``start``), or None.

    Walked BACKWARDS from the end so a torn trailing append (writer mid-flush),
    a window that began mid-line, and a corrupt record are each skipped without
    another read. Returns ``(event, end_offset)`` with ``end_offset`` the byte
    position just past the line's terminating ``\\n``.
    """
    pos = len(buf)
    while pos > 0:
        nl = buf.rfind(b"\n", 0, pos)
        if nl < 0:
            return None
        line_start = buf.rfind(b"\n", 0, nl) + 1
        line = buf[line_start:nl]
        if line:
            event = decode_line(line)
            if event is not None:
                return event, start + nl + 1
        pos = nl
    return None


def last_event(path: Path) -> tuple[dict[str, Any], int] | None:
    """``(event, end_offset)`` for the LAST complete, CRC-valid record.

    Bounded: reads at most :data:`_LAST_EVENT_WINDOW` bytes from the end of the
    log, growing once to :data:`_LAST_EVENT_WINDOW_MAX` when that window holds
    no complete line (one record larger than the window), then falling back to
    a forward :func:`tail_jsonl` scan for a record larger still.

    ``end_offset`` is the byte position immediately after the line's
    terminating ``\\n`` — the same cursor :func:`tail_jsonl` yields, so a caller
    can compare it against a marker it captured earlier (e.g. the log size when
    it published a turn) and learn whether the record was written AFTER that
    point.

    Unlike a slice of :func:`tail_jsonl` starting at a raw byte offset, this
    never mis-treats a mid-record window start as corruption: a healthy log
    yields an event and no corruption warning. A missing, empty or unreadable
    file yields ``None``.
    """
    try:
        size = path.stat().st_size
    except (OSError, ValueError):
        return None
    if size <= 0:
        return None

    for window in (_LAST_EVENT_WINDOW, _LAST_EVENT_WINDOW_MAX):
        start = max(0, size - window)
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                buf = fh.read(size - start)
        except OSError:
            return None
        found = _last_complete_line(buf, start)
        if found is not None or start == 0:
            return found

    # Final record larger than every window — walk forward (still bounded by
    # the file, and the only path that can report genuinely corrupt lines).
    last: tuple[dict[str, Any], int] | None = None
    for event, end_offset in tail_jsonl(path, 0):
        last = (event, end_offset)
    return last


_TRUNCATED_CONTENT = "[content truncated]"
_MISSING = object()

# A "plain content" run of a JSON string ends at the closing quote, at a
# backslash escape, or at a raw control byte (illegal unescaped in JSON).
_RE_STRING_SPECIAL = re.compile(rb'["\\\x00-\x1f]')
# A complete JSON escape: the one-character forms plus \uXXXX (<= 6 bytes).
_RE_STRING_ESCAPE = re.compile(rb'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})')


class _StreamingJson:
    """Bounded-memory JSON validator with top-level scalar projection.

    The parser consumes one payload from a buffered file range. It validates
    the complete JSON grammar while retaining only requested top-level scalar
    values (each capped at 64 KiB). Large content strings and nested payloads
    are streamed and discarded.
    """

    _WANTED = {
        "type",
        "turn_idx",
        "user",
        "source",
        "kind",
        "turn_seq",
        "agent_name",
        "effective_mode_id",
        "state",
        "engine",
    }

    def __init__(self, stream: Any, start: int, end: int) -> None:
        self._stream = stream
        self._start = start
        self._end = end
        self._pos = start
        self._buf = b""
        self._idx = 0
        self._pushback: int | None = None
        self.crc = 0
        self.bytes_read = 0
        self.scalars: dict[str, Any] = {}

    def _fill(self) -> bool:
        if self._pos >= self._end:
            return False
        self._stream.seek(self._pos)
        self._buf = self._stream.read(min(1 << 16, self._end - self._pos))
        self._pos += len(self._buf)
        self._idx = 0
        self.crc = zlib.crc32(self._buf, self.crc)
        self.bytes_read += len(self._buf)
        return bool(self._buf)

    def byte(self) -> int | None:
        """Read one byte."""
        if self._pushback is not None:
            value = self._pushback
            self._pushback = None
            return value
        if self._idx >= len(self._buf) and not self._fill():
            return None
        value = self._buf[self._idx]
        self._idx += 1
        return value

    def unread(self, value: int) -> None:
        """Push one byte back."""
        if self._pushback is not None:
            raise ValueError("double JSON pushback")
        self._pushback = value

    def nonspace(self) -> int | None:
        """Read the next non-space byte."""
        while True:
            value = self.byte()
            if value is None or value not in b" \t\r\n":
                return value

    def string(self, *, capture: bool) -> Any:
        """Scan one JSON string."""
        if not capture:
            self._skip_string()
            return _MISSING
        raw = bytearray(b'"')
        utf8_left = 0
        utf8_low = 0x80
        utf8_high = 0xBF
        while True:
            value = self.byte()
            if value is None:
                raise ValueError("unterminated JSON string")
            if utf8_left:
                if not utf8_low <= value <= utf8_high:
                    raise ValueError("invalid UTF-8 in JSON string")
                utf8_left -= 1
                utf8_low, utf8_high = 0x80, 0xBF
                if raw is not None:
                    raw.append(value)
                continue
            if value == 0x22:
                if raw is None:
                    return _MISSING
                raw.append(value)
                try:
                    return json.loads(bytes(raw).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("invalid JSON string") from exc
            if value == 0x5C:
                escaped = self.byte()
                if escaped is None or escaped not in b'"\\/bfnrtu':
                    raise ValueError("invalid JSON escape")
                if raw is not None:
                    raw.extend((value, escaped))
                if escaped == ord("u"):
                    for _ in range(4):
                        digit = self.byte()
                        if digit is None or chr(digit) not in "0123456789abcdefABCDEF":
                            raise ValueError("invalid JSON unicode escape")
                        if raw is not None:
                            raw.append(digit)
                continue
            if value < 0x20:
                raise ValueError("control byte in JSON string")
            if value >= 0x80:
                if 0xC2 <= value <= 0xDF:
                    utf8_left = 1
                elif value == 0xE0:
                    utf8_left, utf8_low = 2, 0xA0
                elif 0xE1 <= value <= 0xEC or 0xEE <= value <= 0xEF:
                    utf8_left = 2
                elif value == 0xED:
                    utf8_left, utf8_high = 2, 0x9F
                elif value == 0xF0:
                    utf8_left, utf8_low = 3, 0x90
                elif 0xF1 <= value <= 0xF3:
                    utf8_left = 3
                elif value == 0xF4:
                    utf8_left, utf8_high = 3, 0x8F
                else:
                    raise ValueError("invalid UTF-8 in JSON string")
            if raw is not None:
                if len(raw) < (1 << 16):
                    raw.append(value)
                else:
                    raw = None

    # ---- non-captured string skipping (oversized records) -------------
    #
    # The byte-wise loop above costs ~3 Python calls per character. Session
    # records can be multi-megabyte: ``turn_ended`` persists the FULL
    # ``effective_system_prompt`` (~1.9 MB) into every turn, and 55 such lines
    # were 77% of a measured 95 MB session. Parsing one took 150-300 ms
    # (~10-12 MB/s), so ``scan_history_page`` on that session cost ~650 ms and
    # ``load_session_dict`` ~890 ms — while a 406 MB session with no oversized
    # record loaded in ~60 ms.
    #
    # Skipping non-captured content in C keeps the accepted/rejected language
    # identical: plain runs are matched by one regex search and validated by
    # the C UTF-8 incremental decoder; escapes, control bytes and unterminated
    # strings still raise exactly as before. The CRC still covers every byte
    # (``_fill`` stays the only reader) and ``bytes_read`` accounting is
    # unchanged, because the same buffers are still fed through it.

    def _skip_string(self) -> None:
        """Skip one non-captured JSON string body at chunk speed."""
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        while True:
            if self._idx >= len(self._buf) and not self._fill():
                raise ValueError("unterminated JSON string")
            buf = self._buf
            special = _RE_STRING_SPECIAL.search(buf, self._idx)
            if special is None:
                self._decode_utf8(decoder, buf[self._idx :])
                self._idx = len(buf)
                continue
            index = special.start()
            if index > self._idx:
                self._decode_utf8(decoder, buf[self._idx : index])
                self._idx = index
            value = buf[index]
            if value == 0x22:
                try:
                    decoder.decode(b"", True)
                except UnicodeDecodeError as exc:
                    raise ValueError("invalid UTF-8 in JSON string") from exc
                self._idx = index + 1
                return
            if value != 0x5C:
                # Only a raw control byte can stop the search inside a string.
                raise ValueError("control byte in JSON string")
            escape = _RE_STRING_ESCAPE.match(buf, index)
            if escape is not None:
                self._idx = escape.end()
                continue
            # Not a complete escape with this chunk: it is either malformed or
            # straddling the 64 KiB boundary (max escape length is 6 bytes).
            if len(buf) - index >= 7:
                raise ValueError("invalid JSON escape")
            pending, _flag = decoder.getstate()
            if pending:
                raise ValueError("invalid UTF-8 in JSON string")
            # ``_skip_escape`` reads the escaped CHARACTER, so the backslash
            # itself is consumed first (it may sit in the previous chunk).
            self._idx = index + 1
            self._skip_escape()

    @staticmethod
    def _decode_utf8(decoder: Any, segment: bytes) -> None:
        """Feed one plain run to the incremental decoder (C-speed UTF-8 check)."""
        if not segment:
            return
        try:
            decoder.decode(segment)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 in JSON string") from exc

    def _skip_escape(self) -> None:
        """Validate one JSON escape byte-wise (chunk-boundary fallback only)."""
        escaped = self.byte()
        if escaped is None or escaped not in b'"\\/bfnrtu':
            raise ValueError("invalid JSON escape")
        if escaped == ord("u"):
            for _ in range(4):
                digit = self.byte()
                if digit is None or chr(digit) not in "0123456789abcdefABCDEF":
                    raise ValueError("invalid JSON unicode escape")

    def scalar(self, first: int) -> Any:
        """Scan one bounded JSON scalar."""
        if first == 0x22:
            return self.string(capture=True)
        token = bytearray((first,))
        while True:
            value = self.byte()
            if value is None or value in b" \t\r\n,]}":
                if value is not None:
                    self.unread(value)
                break
            if len(token) >= 128:
                raise ValueError("oversized JSON scalar")
            token.append(value)
        try:
            return json.loads(bytes(token).decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON scalar") from exc

    def value(self, first: int | None = None, *, capture: bool = False) -> Any:
        """Scan one JSON value."""
        value = self.nonspace() if first is None else first
        if value is None:
            raise ValueError("missing JSON value")
        if value == 0x7B:
            self.object(top=False)
            return _MISSING
        if value == 0x5B:
            self.array()
            return _MISSING
        if capture:
            return self.scalar(value)
        if value == 0x22:
            self.string(capture=False)
            return _MISSING
        self.scalar(value)
        return _MISSING

    def object(self, *, top: bool) -> None:
        """Scan one JSON object."""
        value = self.nonspace()
        if value == 0x7D:
            return
        while True:
            if value != 0x22:
                raise ValueError("object key is not a string")
            key = self.string(capture=top)
            if self.nonspace() != 0x3A:
                raise ValueError("missing object colon")
            first = self.nonspace()
            capture = top and isinstance(key, str) and key in self._WANTED and first not in (0x7B, 0x5B)
            projected = self.value(first, capture=capture)
            if capture and projected is not _MISSING:
                self.scalars[key] = projected
            value = self.nonspace()
            if value == 0x7D:
                return
            if value != 0x2C:
                raise ValueError("missing object comma")
            value = self.nonspace()

    def array(self) -> None:
        """Scan one JSON array."""
        value = self.nonspace()
        if value == 0x5D:
            return
        while True:
            self.value(value)
            value = self.nonspace()
            if value == 0x5D:
                return
            if value != 0x2C:
                raise ValueError("missing array comma")
            value = self.nonspace()

    def parse(self) -> dict[str, Any]:
        """Project the top-level event."""
        if self.nonspace() != 0x7B:
            raise ValueError("session event is not a JSON object")
        self.object(top=True)
        if self.nonspace() is not None:
            raise ValueError("trailing bytes after JSON object")
        return self.scalars


def _record_reverse_corruption(
    path: Path,
    line_start: int,
    stats: dict[str, int],
) -> None:
    global SESSION_JSONL_CORRUPTION_TOTAL
    SESSION_JSONL_CORRUPTION_TOTAL += 1
    stats["corrupt_lines"] = stats.get("corrupt_lines", 0) + 1
    _log.warning(
        "session jsonl: dropped corrupt line at offset %d in %s",
        line_start,
        path,
    )


def _oversized_event_projection(
    stream: Any,
    path: Path,
    line_start: int,
    line_end: int,
    stats: dict[str, int],
) -> dict[str, Any] | None:
    """CRC-verify and project one oversized record without materialising it.

    ``line_end`` is immediately after the LF terminator. Validation bytes are
    reported separately from the bounded reverse-page scan because a CRC
    necessarily covers the complete line. Only genuine top-level scalar
    identity/boundary fields survive projection; nested payload fields cannot
    become turn boundaries.
    """
    try:
        stream.seek(line_start)
        prefix = stream.read(9)
        stream.seek(line_end - 1)
        newline = stream.read(1)
        stats["validation_bytes_read"] = stats.get("validation_bytes_read", 0) + len(prefix) + len(newline)
        if len(prefix) != 9 or prefix[8:9] != b" " or newline != b"\n" or line_end <= line_start + 10:
            raise ValueError("malformed oversized line framing")
        parser = _StreamingJson(stream, line_start + 9, line_end - 1)
        try:
            fields = parser.parse()
        finally:
            stats["validation_bytes_read"] += parser.bytes_read
        expected_crc = f"{parser.crc & 0xFFFFFFFF:08x}".encode("ascii")
        if expected_crc != prefix[:8]:
            raise ValueError("oversized line CRC mismatch")
    except (OSError, ValueError, RecursionError):
        _record_reverse_corruption(path, line_start, stats)
        return None

    event_type = fields.get("type")
    turn_idx = fields.get("turn_idx")
    if isinstance(event_type, str):
        event: dict[str, Any] = {
            "type": event_type,
            "turn_idx": turn_idx if isinstance(turn_idx, int) else 0,
            "_content_truncated": True,
        }
        if isinstance(fields.get("source"), str):
            event["source"] = fields["source"]
        # A turn_started carrying an inline image easily exceeds the line
        # cap; Stop needs its engine to close the turn as the right engine.
        if isinstance(fields.get("engine"), str):
            event["engine"] = fields["engine"]
        if event_type == "turn_ended":
            event["user"] = fields.get("user") or _TRUNCATED_CONTENT
            event["final_content"] = _TRUNCATED_CONTENT
        elif event_type == "turn_started":
            event["user"] = _TRUNCATED_CONTENT
        else:
            event["delta"] = _TRUNCATED_CONTENT
            event["content"] = _TRUNCATED_CONTENT
        return event

    kind = fields.get("kind")
    turn_seq = fields.get("turn_seq")
    if not isinstance(kind, str):
        return None
    event = {
        "kind": kind,
        "turn_seq": turn_seq if isinstance(turn_seq, int) else None,
        "_content_truncated": True,
    }
    for key in ("agent_name", "effective_mode_id", "state"):
        value = fields.get(key)
        if value is not None:
            event[key] = value
    if kind in {"user_message_committed", "assistant_message_committed"}:
        event["content"] = _TRUNCATED_CONTENT
    return event


def reverse_jsonl(
    path: Path,
    end_offset: int | None = None,
    *,
    stats: dict[str, int] | None = None,
    max_bytes: int | None = None,
    max_events: int | None = None,
    max_line_bytes: int = 2 << 20,
) -> Iterator[tuple[dict[str, Any], int, int]]:
    """Yield valid complete JSONL records backwards as ``(event, start, end)``.

    ``end_offset`` is an exclusive byte cursor returned by this reader.  Only
    blocks needed to reach the requested history page are read, so opening a
    long session is O(page bytes), not O(session bytes).
    """
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return
    try:
        size = f.seek(0, 2)
        line_end = min(
            size,
            max(0, int(end_offset if end_offset is not None else size)),
        )
        if stats is None:
            stats = {}

        def _chunk_before(end: int) -> tuple[int, bytes]:
            remaining = end if max_bytes is None else max(0, max_bytes - stats.get("bytes_read", 0))
            if remaining <= 0:
                stats["budget_exhausted"] = 1
                stats["resume_cursor"] = end
                return end, b""
            count = min(1 << 16, end, remaining)
            start = end - count
            f.seek(start)
            chunk = f.read(count)
            stats["bytes_read"] = stats.get("bytes_read", 0) + len(chunk)
            return start, chunk

        if line_end:
            f.seek(line_end - 1)
            if f.read(1) != b"\n":
                # Ignore a writer's partial tail (or a foreign mid-line
                # cursor) without accumulating the partial line in memory.
                while line_end:
                    start, chunk = _chunk_before(line_end)
                    if not chunk:
                        return
                    nl = chunk.rfind(b"\n")
                    if nl >= 0:
                        line_end = start + nl + 1
                        break
                    line_end = start

        # Exclude the final newline. ``suffix`` holds only the bounded tail of
        # a line crossing a block boundary; oversized lines are skipped until
        # their preceding newline instead of being accumulated into RSS.
        cursor = max(0, line_end - 1)
        current_end = line_end
        suffix = b""
        oversized = False
        while cursor > 0:
            if max_events is not None and stats.get("events_decoded", 0) >= max_events:
                stats["budget_exhausted"] = 1
                stats["resume_cursor"] = cursor
                return
            start, chunk = _chunk_before(cursor)
            if not chunk:
                return
            segment_end = len(chunk)
            while True:
                nl = chunk.rfind(b"\n", 0, segment_end)
                if nl < 0:
                    piece = chunk[:segment_end]
                    if not oversized:
                        if len(piece) + len(suffix) <= max_line_bytes:
                            suffix = piece + suffix
                        else:
                            suffix = b""
                            oversized = True
                    break
                line_start = start + nl + 1
                piece = chunk[nl + 1 : segment_end]
                if oversized or len(piece) + len(suffix) > max_line_bytes:
                    stats["oversized_lines"] = stats.get("oversized_lines", 0) + 1
                    event = _oversized_event_projection(
                        f,
                        path,
                        line_start,
                        current_end,
                        stats,
                    )
                    if event is not None:
                        stats["events_decoded"] = stats.get("events_decoded", 0) + 1
                        stats["history_truncated"] = 1
                        yield event, line_start, current_end
                else:
                    if max_events is not None and stats.get("events_decoded", 0) >= max_events:
                        stats["budget_exhausted"] = 1
                        stats["resume_cursor"] = line_start
                        return
                    raw = piece + suffix
                    event = decode_line(raw)
                    if event is not None:
                        stats["events_decoded"] = stats.get("events_decoded", 0) + 1
                        yield event, line_start, current_end
                    elif raw:
                        _record_reverse_corruption(path, line_start, stats)
                current_end = line_start
                suffix = b""
                oversized = False
                segment_end = nl
            cursor = start

        if current_end > 0 and (oversized or len(suffix) > max_line_bytes):
            stats["oversized_lines"] = stats.get("oversized_lines", 0) + 1
            event = _oversized_event_projection(
                f,
                path,
                0,
                current_end,
                stats,
            )
            if event is not None:
                stats["events_decoded"] = stats.get("events_decoded", 0) + 1
                stats["history_truncated"] = 1
                yield event, 0, current_end
        elif current_end > 0:
            event = decode_line(suffix)
            if event is not None:
                stats["events_decoded"] = stats.get("events_decoded", 0) + 1
                yield event, 0, current_end
            elif suffix:
                _record_reverse_corruption(path, 0, stats)
    finally:
        f.close()


def turn_engine(sessions_dir: Any, sid: str, turn_idx: int) -> str:
    """Engine that ran ``turn_idx``, from that turn's ``turn_started`` event.

    ``sessions_meta.engine`` (and the session header) keep the agent's
    original runtime, e.g. ``local_llm``, after ``/agent`` moves the session
    to a CLI engine, while every turn event carries the engine that ran it.
    A Stop that closes a turn with the header engine is dropped by the web
    renderer as another engine's event, so the stopped turn stays RUNNING.

    Scans back only to the newest turn boundary. Returns "" when that is not
    this turn's start (already closed, or no start on record), so callers
    fall back to the row's engine.
    """
    try:
        for ev, _start, _end in reverse_jsonl(Path(sessions_dir) / f"{sid}.jsonl"):
            etype = ev.get("type")
            if etype not in ("turn_started", "turn_ended"):
                continue
            if etype == "turn_started" and ev.get("turn_idx") == int(turn_idx):
                return str(ev.get("engine") or "")
            return ""
    except Exception:  # noqa: BLE001 — best-effort; a Stop must still close
        return ""
    return ""


def merge_reverse_scan_stats(
    target: dict[str, int],
    phase: str,
    source: dict[str, int],
) -> None:
    """Merge one independently-budgeted reverse scan into public diagnostics."""
    for key, value in source.items():
        target[f"{phase}_{key}"] = value
        if key in {
            "bytes_read",
            "events_decoded",
            "oversized_lines",
            "validation_bytes_read",
            "corrupt_lines",
        }:
            target[key] = target.get(key, 0) + value
        elif key in {
            "budget_exhausted",
            "history_truncated",
            "tail_continuation",
            "compact_terminal_fallback",
        }:
            target[key] = max(target.get(key, 0), value)
    if "resume_cursor" in source and (phase == "page" or "resume_cursor" not in target):
        target["resume_cursor"] = source["resume_cursor"]


# A saved turn must retain its complete chronological card sequence.  The Web
# renderer mounts Activity payloads lazily, while the per-card limits below keep
# large reasoning/tool bodies bounded.  The former default of 12 silently
# discarded earlier assistant prose and activity because no intra-turn pager
# exists; a newly opened tab therefore disagreed with a tab that followed the
# same turn live.  ``None`` keeps every card.  The optional function argument is
# retained for callers that intentionally need an explicit emergency cap.
HISTORY_SEGMENTS_MAX: int | None = None
HISTORY_THINKING_MAX = 2048
HISTORY_TOOL_CONTENT_MAX = 2048
HISTORY_TOOL_ARGS_PREVIEW = 157
_KEEP_FULL_TOOL_NAMES = {
    "ExitPlanMode",
    "TaskCreate",
    "TaskUpdate",
    "TaskStop",
    "delegate_claude",
    "delegate_codex",
    "delegate_antigravity",
    "compact_context",
}


def _is_ask_user_name(name: Any) -> bool:
    """True for any engine's spelling of the ask-user tool.

    ``sharedagents_agent.tools.widget_adapt`` owns the alias table; imported lazily
    because this package sits below ``sharedagents_agent`` in the import graph (same
    pattern as the plan-tool helpers below).
    """
    from sharedagents_agent.tools.widget_adapt import is_ask_user_tool

    return is_ask_user_tool(name)


# Newest REST page is last-N turns. An AskUserQuestion on an older turn
# vanishes on refresh once SSE tails instead of replaying from 0. Extend
# that first page through the most recent ask turn, then stop.
HISTORY_ASK_LOOKBACK = 16
_KEEP_FULL_TOOL_KEYS = {
    "todowrite",
    "todo",
    "todos",
    "updatetodos",
}
_KEEP_FULL_DELEGATE_NAMES = {
    "delegate_agent",
    "agent",
    "agentsync",
    "agentasync",
    "task",
    "general-purpose",
}


def _card_locator(value: Any) -> int | None:
    """Non-negative JSONL byte offset, or None when the caller has none."""
    if value is None or value is False:
        return None
    try:
        loc = int(value)
    except (TypeError, ValueError):
        return None
    return loc if loc >= 0 else None


def parse_card_id(call_id: str) -> tuple[str, int | None]:
    """Split ``<id>@<byte-offset>`` from a lazy card locator."""
    needle = str(call_id or "")
    if "@" not in needle:
        return needle, None
    base, loc = needle.rsplit("@", 1)
    parsed = _card_locator(loc)
    if parsed is None:
        return needle, None
    return (base or "reason"), parsed


def thinking_card_id_for(_text: str = "", locator: Any = None) -> str:
    """Activity id is the JSONL start offset. The log is append-only."""
    loc = _card_locator(locator)
    if loc is None:
        return "reason"
    return f"reason@{loc}"


def _history_keeps_full_tool(name: Any) -> bool:
    """True for widgets that remount from args/body, not the lazy activity card."""
    raw = str(name or "")
    if raw in _KEEP_FULL_TOOL_NAMES:
        return True
    if _is_ask_user_name(raw):
        # The ask card remounts from its own args/body, so its payload is never
        # replaced by the lazy activity card — for every engine's spelling.
        return True
    key = "".join(ch for ch in raw.lower() if ch.isalnum())
    if key in _KEEP_FULL_TOOL_KEYS:
        return True
    # Every plan-publishing tool carries the turn's DELIVERABLE in its
    # arguments — Cursor's ``Create Plan`` as much as ``ExitPlanMode``. The
    # 157-char one-line preview turns ``{"plan": …}`` into an unterminated
    # JSON fragment, which the replay renderer cannot parse: it then opens
    # neither the 📐 card nor a tool row, so history showed narration about a
    # plan the operator could not read (live: session
    # ``<session-id>`` turn 19). Reuse the module's own
    # plan predicate rather than a third alias list.
    if _is_plan_tool_name(raw):
        return True
    low = raw.lower()
    return low in _KEEP_FULL_DELEGATE_NAMES or low.startswith("delegate_")


def project_history_segments(
    segments: list[dict[str, Any]] | None,
    *,
    max_segments: int | None = HISTORY_SEGMENTS_MAX,
    max_thinking: int = HISTORY_THINKING_MAX,
    max_tool_content: int = HISTORY_TOOL_CONTENT_MAX,
) -> list[dict[str, Any]]:
    """Bound card payloads without dropping chronological history.

    ``max_segments`` is opt-in.  The Web history path leaves it as ``None`` so
    reload/reopen sees the same ordered cards as the live event stream.
    """
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        item = dict(seg)
        if item.get("kind") == "tool":
            content = item.get("content")
            # Keep-full widgets remount from their args or body and carry NO
            # lazy card id, so the ``[truncated N chars]`` placeholder would be
            # PERMANENT for them — there is nothing to fetch the body back
            # from. The replay renderer feeds a plan tool's result body
            # straight into ``_setPlan``, so the placeholder overwrote the 📐
            # card's markdown with "[truncated 4481 chars]" and the operator
            # saw no plan at all (live: session
            # ``<session-id>`` turn 19). Plumbing bodies
            # whose card CAN be lazily restored still take the placeholder.
            if (
                isinstance(content, str)
                and len(content) > max_tool_content
                and not _history_keeps_full_tool(item.get("name"))
            ):
                try:
                    n = int(item.get("content_chars") or len(content))
                except (TypeError, ValueError):
                    n = len(content)
                item["content_chars"] = n
                item["content"] = f"[truncated {n} chars]"
                item["content_truncated"] = True
            raw = item.get("raw_arguments")
            if isinstance(raw, str) and len(raw) > max_tool_content:
                item["raw_arguments"] = raw[:max_tool_content]
                item["raw_arguments_truncated"] = True
        out.append(item)
    if max_segments is not None:
        cap = max(1, int(max_segments))
        if len(out) > cap:
            out = out[-cap:]
    return out


def trim_history_messages(
    messages: list[dict[str, Any]] | None,
    *,
    max_segments: int | None = HISTORY_SEGMENTS_MAX,
) -> list[dict[str, Any]]:
    """Apply :func:`project_history_segments` to every assistant row."""
    out: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        row = dict(message)
        segs = row.get("segments")
        if isinstance(segs, list) and segs:
            row["segments"] = project_history_segments(
                segs,
                max_segments=max_segments,
            )
        out.append(row)
    return out


def _preview_tool_arguments(value: Any) -> Any:
    """Keep a one-liner args shell; drop large remember/Shell blobs."""
    if value is None:
        return {}
    if isinstance(value, str):
        if len(value) <= HISTORY_TOOL_ARGS_PREVIEW:
            return value
        return value[:HISTORY_TOOL_ARGS_PREVIEW]
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) <= HISTORY_TOOL_ARGS_PREVIEW:
        return value
    return text[:HISTORY_TOOL_ARGS_PREVIEW]


def _colliding_tool_locators(
    messages: list[dict[str, Any]] | None,
) -> set[tuple[int, str]]:
    """``(card_start, call_id)`` pairs reused across a folded turn."""
    counts: dict[tuple[int, str], int] = {}
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        for seg in message.get("segments") or []:
            if not isinstance(seg, dict) or seg.get("kind") != "tool":
                continue
            loc = _card_locator(seg.get("card_start"))
            call_id = str(seg.get("call_id") or "")
            if loc is None or not call_id:
                continue
            key = (loc, call_id)
            counts[key] = counts.get(key, 0) + 1
    return {key for key, n in counts.items() if n > 1}


def summarize_history_segments(
    segments: list[dict[str, Any]] | None,
    *,
    colliding: set[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """One-liner tool shells for reload; assistant reasoning stays inline.

    Tool bodies still load on card expand (``/cards/<id>``). Thinking stays on
    the history / live message payload — same content the SSE stream already
    delivered — so expand must not round-trip ``reason@…`` for reasoning.
    """
    out: list[dict[str, Any]] = []
    blocked = colliding or set()
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        item = dict(seg)
        kind = item.get("kind")
        if kind == "tool":
            if _history_keeps_full_tool(item.get("name")):
                out.append(item)
                continue
            loc = _card_locator(item.get("card_start"))
            call_id = str(item.get("call_id") or "")
            if loc is None or not call_id.strip() or _is_reason_card(call_id) or (loc, call_id) in blocked:
                out.append(item)
                continue
            item.pop("content", None)
            item["lazy"] = True
            item["card_id"] = f"{call_id}@{loc}"
            if "arguments" in item:
                item["arguments"] = _preview_tool_arguments(item.get("arguments"))
            raw = item.get("raw_arguments")
            if isinstance(raw, str) and len(raw) > HISTORY_TOOL_ARGS_PREVIEW:
                item["raw_arguments"] = raw[:HISTORY_TOOL_ARGS_PREVIEW]
        # assistant: keep ``thinking`` inline (SSE / live parity). Do not
        # emit lazy_thinking — that forced a /cards fetch on expand.
        out.append(item)
    return out


def lazy_history_messages(
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Strip tool card bodies after stamping start offsets; expand seeks those."""
    return trim_history_messages(summarize_history_messages(messages))


def summarize_history_messages(
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Apply :func:`summarize_history_segments` to every assistant row."""
    out: list[dict[str, Any]] = []
    colliding = _colliding_tool_locators(messages)
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        row = dict(message)
        segs = row.get("segments")
        if isinstance(segs, list) and segs:
            row["segments"] = summarize_history_segments(segs, colliding=colliding)
        out.append(row)
    return out


def _unwrap_card_event(event: dict[str, Any], *, logical: bool) -> dict[str, Any]:
    if logical and event.get("kind") == "physical_event_relayed":
        inner = event.get("event") or {}
        if isinstance(inner, dict):
            return inner
    return event


def _is_reason_card(needle: str) -> bool:
    return needle == "reason" or needle.startswith("reason-")


def _collect_card_events_from(
    path: Path,
    needle: str,
    locator: int,
    *,
    logical: bool,
) -> list[dict[str, Any]]:
    """Seek to the stamped start and read that one activity."""
    is_reason = _is_reason_card(needle)
    out: list[dict[str, Any]] = []
    started = False
    for event, _end in tail_jsonl(path, locator):
        raw = event
        event = _unwrap_card_event(event, logical=logical)
        etype = event.get("type")
        if logical and raw.get("kind") == "turn_terminal_committed" and started:
            break
        if etype == "turn_started":
            if started:
                break
            started = True
            continue
        started = True
        if is_reason:
            if etype == "thinking_delta" or (etype == "turn_ended" and event.get("final_thinking")):
                out.append(event)
            continue
        if str(event.get("call_id") or "") == needle:
            out.append(event)
    return out


def collect_card_events(
    path: Path,
    call_id: str,
    *,
    logical: bool = False,
) -> list[dict[str, Any]]:
    """Return JSONL events for one tool card or reasoning trace.

    Expand requires a stamped start offset. No offset means no card.
    """
    needle, locator = parse_card_id(call_id)
    if locator is None:
        return []
    return _collect_card_events_from(
        Path(path),
        needle or "reason",
        locator,
        logical=logical,
    )


def fold_card_segment(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Rebuild one lazy history tool or reasoning segment from raw log events."""
    if not events:
        return None
    if any(
        ev.get("type") == "thinking_delta"
        or (ev.get("type") == "turn_ended" and ev.get("final_thinking") not in (None, ""))
        for ev in events
    ):
        final = ""
        parts: list[str] = []
        for ev in events:
            if ev.get("type") == "thinking_delta":
                parts.append(str(ev.get("delta") or ""))
            elif ev.get("type") == "turn_ended":
                raw_final = ev.get("final_thinking")
                if raw_final not in (None, ""):
                    final = str(raw_final)
        content = final or "".join(parts)
        if not content.strip():
            return None
        return {
            "kind": "thinking",
            "thinking_card_id": "reason",
            "content": content,
        }
    name = ""
    args: Any = None
    streamed_parts: list[str] = []
    final_content: str | None = None
    ok: bool | None = None
    outcome = None
    member_outcomes = ()
    for ev in events:
        etype = ev.get("type")
        if etype == "tool_call":
            name = str(ev.get("name") or name or "tool")
            args = ev.get("arguments") if ev.get("arguments") is not None else args
        elif etype == "tool_result_meta":
            outcome = ev.get("outcome")
            member_outcomes = ev.get("member_outcomes") or ()
            ok = bool(ev.get("ok")) if ev.get("ok") is not None else ok
        elif etype == "tool_delta":
            delta = ev.get("delta") or ev.get("chunk") or ""
            if delta not in (None, ""):
                streamed_parts.append(str(delta))
        elif etype == "tool_result":
            if ev.get("ok") is not None:
                ok = bool(ev.get("ok"))
            body = ev.get("content") if ev.get("content") is not None else ev.get("result")
            if body not in (None, ""):
                final_content = str(body)
    content = final_content if final_content is not None else "".join(streamed_parts)
    if not name and not content and ok is None:
        return None
    seg: dict[str, Any] = {
        "kind": "tool",
        "name": name or "tool",
        "call_id": str(events[0].get("call_id") or ""),
        "arguments": args if args is not None else {},
    }
    if outcome:
        seg["outcome"] = outcome
    if member_outcomes:
        seg["member_outcomes"] = list(member_outcomes)
    if ok is not None:
        seg["ok"] = ok
    if content:
        seg["content"] = content
    return seg


def open_turn_belongs_in_snapshot(
    turn: dict[str, Any],
    *,
    snapshot_stats: dict[str, int] | None = None,
) -> bool:
    """True if an in-progress turn should fold into the newest REST page.

    Assistant content, thinking, and cards already belong so a refresh can
    pin to the live bubble. A prompting-only turn (user sent, no tokens yet)
    must also belong: otherwise the snapshot omits the user bubble, the
    browser drops ``pendingEcho`` on reload, and warming attaches to the
    previous closer.

    Prompting-only inclusion is refused when the open-turn snapshot dropped
    a corrupt/oversized record or exhausted its budget — those leftovers
    look like a live prompt but are actually an unreadable terminal.
    """
    if not turn.get("in_progress"):
        return False
    if str(turn.get("content") or "").strip():
        return True
    if str(turn.get("thinking") or "").strip():
        return True
    if turn.get("render_segments"):
        return True
    raw_user = turn.get("user")
    has_user = (isinstance(raw_user, str) and raw_user.strip()) or (isinstance(raw_user, list) and bool(raw_user))
    if not has_user:
        return False
    stats = snapshot_stats or {}
    return not (
        int(stats.get("corrupt_lines") or 0)
        or int(stats.get("oversized_lines") or 0)
        or int(stats.get("budget_exhausted") or 0)
    )


def snapshot_phase_quality(stats: dict[str, int] | None) -> dict[str, int]:
    """Normalize diagnostics from one open-turn reverse-scan phase."""
    stats = stats or {}
    return {
        "corrupt_lines": int(stats.get("corrupt_lines") or 0),
        "oversized_lines": int(stats.get("oversized_lines") or 0),
        "budget_exhausted": int(stats.get("budget_exhausted") or 0),
    }


def snapshot_open_turn_events(
    path: Path,
    *,
    open_start: int,
    eof: int,
    stats: dict[str, int] | None = None,
    max_scan_bytes: int = 8 << 20,
    max_scan_events: int = 20_000,
    logical: bool = False,
) -> list[dict[str, Any]]:
    """Collect the open turn's events as a bounded newest-page snapshot.

    Reverse-reads from ``eof`` down to ``open_start`` (or the scan budget)
    so a long in-progress turn becomes one folded REST page instead of an
    SSE replay from the first ``turn_started``. Oversized records stay
    projected/truncated by ``reverse_jsonl``.
    """
    if open_start >= eof:
        return []
    snap_stats: dict[str, int] = {}
    collected: list[dict[str, Any]] = []
    for event, start, _end in reverse_jsonl(
        path,
        eof,
        stats=snap_stats,
        max_bytes=max_scan_bytes,
        max_events=max_scan_events,
    ):
        if start < open_start:
            break
        if logical:
            if event.get("kind") == "physical_event_relayed":
                inner = event.get("event") or {}
                if isinstance(inner, dict):
                    collected.append(inner)
        elif isinstance(event.get("type"), str):
            collected.append(event)
    if stats is not None:
        stats["_snapshot_stats"] = snap_stats
        stats["open_turn_snapshot"] = 1
    collected.reverse()
    return collected


def _events_have_ask_user(events: list[dict[str, Any]]) -> bool:
    """True when this turn group carries an AskUserQuestion-family tool_call."""
    for event in events:
        if event.get("type") != "tool_call":
            continue
        if _is_ask_user_name(event.get("name")):
            return True
    return False


def scan_history_page(
    path: Path,
    *,
    before_offset: int | None = None,
    limit: int = 2,
    stats: dict[str, int] | None = None,
    max_scan_bytes: int = 8 << 20,
    max_scan_events: int = 20_000,
) -> tuple[list[dict[str, Any]], int, bool, int | None]:
    """Fold one reverse-read page of physical turns.

    The page cursor is a byte boundary immediately after the preceding
    completed turn. Feeding it back as ``before_offset`` tiles older pages
    without gaps or duplicates. The newest page snapshots a readable open turn
    and advances its live cursor to EOF; if that snapshot cannot be folded, the
    cursor stays at the open turn so SSE can replay it without loss.

    The newest page (``before_offset is None``) also keeps the most recent
    AskUserQuestion turn even when it sits older than ``limit``, so refresh
    remounts the clarification card from REST. Extra lookback is capped.
    """
    path = Path(path)
    try:
        eof = path.stat().st_size
    except OSError:
        return [], 0, False, None
    lim = max(1, int(limit or 2))

    tail_offset = eof
    active_starts: list[int] = []
    latest_terminal = False
    if stats is None:
        stats = {}
    tail_stats: dict[str, int] = {}
    for event, start, _end in reverse_jsonl(
        path,
        eof,
        stats=tail_stats,
        max_bytes=max_scan_bytes,
        max_events=max_scan_events,
    ):
        etype = event.get("type")
        if etype == "turn_started":
            active_starts.append(start)
        elif etype == "turn_ended":
            latest_terminal = True
            break
    if active_starts:
        tail_offset = max(active_starts)
    active = bool(active_starts)
    if not active and not latest_terminal and tail_stats.get("budget_exhausted"):
        # The active turn itself exceeds the snapshot budget. Continue from
        # the earliest bounded cursor reached; event_frames resynchronises a
        # mid-line cursor forward, skipping at most the oversized record while
        # retaining every later live frame.
        active = True
        tail_offset = int(tail_stats.get("resume_cursor") or max(0, eof - max_scan_bytes))
        tail_stats["tail_continuation"] = 1
    merge_reverse_scan_stats(stats, "tail", tail_stats)

    page_end = max(0, min(int(before_offset), eof)) if before_offset is not None else tail_offset
    groups: list[tuple[int | None, list[dict[str, Any]]]] = []
    extras: list[tuple[int | None, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] = []
    saw_terminal = False
    limit_reached = False
    page_stats: dict[str, int] = {}
    for event, start, _end in reverse_jsonl(
        path,
        page_end,
        stats=page_stats,
        max_bytes=max_scan_bytes,
        max_events=max_scan_events,
    ):
        if not saw_terminal:
            if event.get("type") != "turn_ended":
                continue
            saw_terminal = True
        current.append(event)
        if event.get("type") == "turn_started":
            group = (start, list(reversed(current)))
            current = []
            saw_terminal = False
            if len(groups) < lim:
                groups.append(group)
                if len(groups) < lim:
                    continue
                limit_reached = True
                if before_offset is not None or any(_events_have_ask_user(events) for _start, events in groups):
                    break
                continue
            extras.append(group)
            if len(extras) >= HISTORY_ASK_LOOKBACK:
                break
    if extras:
        last_ask = None
        for idx, (_start, events) in enumerate(extras):
            if _events_have_ask_user(events):
                last_ask = idx
        if last_ask is not None:
            groups.extend(extras[: last_ask + 1])

    if current and saw_terminal:
        # A giant newest turn may exhaust the cap before turn_started. Its
        # terminal record still carries the canonical user/final answer and
        # metrics, so project the bounded suffix instead of returning no page.
        compact_events = list(reversed(current))
        terminal = next(
            (event for event in reversed(compact_events) if event.get("type") == "turn_ended"),
            {},
        )
        if terminal:
            synthesized = {
                "type": "turn_started",
                "turn_idx": terminal.get("turn_idx", 0),
                "user": terminal.get("user", ""),
                "ts": terminal.get("started_at", ""),
            }
            # turn_ended repeats the author(s)/source.
            if terminal.get("actors"):
                synthesized["actors"] = terminal["actors"]
            if terminal.get("attachments"):
                synthesized["attachments"] = terminal["attachments"]
            if terminal.get("source"):
                synthesized["source"] = terminal["source"]
            compact_events.insert(0, synthesized)
        groups.append((None, compact_events))
        page_stats["compact_terminal_fallback"] = 1

    merge_reverse_scan_stats(stats, "page", page_stats)

    turns: list[dict[str, Any]] = []
    for start, events in reversed(groups):
        for turn in fold_events_into_turns(iter(events)):
            if turn.get("in_progress"):
                continue
            if start is not None:
                turn["_card_start"] = start
            turns.append(turn)
    before_cursor = None
    last_start = groups[-1][0] if groups else None
    if limit_reached and last_start is not None and last_start > 0:
        # ``limit_reached`` alone is not enough: a file with *exactly* ``limit``
        # turns still stops with groups[-1][0] > 0 because the session header
        # occupies offset 0 and ``reverse_jsonl`` never yields that record
        # (``while cursor > 0``). Confirm a prior completed turn exists before
        # advertising another page.
        older_end = last_start
        peek_stats: dict[str, int] = {}
        for event, _start, _end in reverse_jsonl(
            path,
            older_end,
            stats=peek_stats,
            max_bytes=max_scan_bytes,
            max_events=max_scan_events,
        ):
            if event.get("type") == "turn_ended":
                before_cursor = older_end
                break
    if before_cursor is None and page_stats.get("budget_exhausted"):
        continuation = int(page_stats.get("resume_cursor") or 0)
        if 0 < continuation < page_end:
            before_cursor = continuation
    # Fold the open turn into the REST page (lazy one-liner shells) and tail
    # SSE from EOF so reload never replays hundreds of historical events.
    if active and before_offset is None and tail_offset < eof:
        snap_events = snapshot_open_turn_events(
            path,
            open_start=tail_offset,
            eof=eof,
            stats=stats,
            max_scan_bytes=max_scan_bytes,
            max_scan_events=max_scan_events,
        )
        snap_stats = stats.get("_snapshot_stats") or {}
        if snap_events:
            snap_appended = False
            open_start = tail_offset
            for turn in fold_events_into_turns(iter(snap_events)):
                if open_turn_belongs_in_snapshot(
                    turn,
                    snapshot_stats=snapshot_phase_quality(snap_stats),
                ):
                    if not tail_stats.get("tail_continuation"):
                        turn["_card_start"] = open_start
                    turns.append(turn)
                    snap_appended = True
            if snap_appended:
                if snap_stats:
                    merge_reverse_scan_stats(stats, "snapshot", snap_stats)
                stats.pop("_snapshot_stats", None)
                tail_offset = eof
            else:
                if snap_stats:
                    merge_reverse_scan_stats(stats, "snapshot", snap_stats)
                stats.pop("_snapshot_stats", None)
                stats.pop("open_turn_snapshot", None)
        else:
            # No foldable event means REST did not take ownership of the open
            # turn.  Keep the SSE cursor at its prior boundary and do not leak
            # private phase stats / a false active-snapshot marker to callers.
            if snap_stats:
                merge_reverse_scan_stats(stats, "snapshot", snap_stats)
            stats.pop("_snapshot_stats", None)
            stats.pop("open_turn_snapshot", None)
    return turns, tail_offset, active, before_cursor


def _resync_and_base_seq(path: Path, start_offset: int) -> tuple[int, int]:
    """Resolve ``start_offset`` to a clean line boundary + its seq base.

    Returns ``(resume_offset, base_seq)``:

    * ``resume_offset`` — the first line boundary AT or AFTER ``start_offset``.
      An exact-boundary cursor resumes right there; a cursor that fell
      mid-line is resynced *forward* to the next ``\\n`` (§9.2 "resync a
      mid-line ``from`` to the next ``\\n``"), so the containing partial line
      is skipped and never replayed — the reader never emits bytes before the
      requested offset, and the partial bytes never reach ``tail_jsonl``'s
      corrupt-line path.
    * ``base_seq`` — the count of complete events up to and including the line
      that ends at ``resume_offset``; the 0-based line index the first emitted
      frame should carry (seq is "derived at read-time from line index",
      §3.3). For a mid-line cursor the skipped containing line is counted, so
      a deliberate mid-line start simply omits that one seq.

    ``start_offset <= 0`` is the fast path (replay from the top): no scan.
    Otherwise this is a single O(prefix) scan done ONCE at subscribe time;
    the live tail afterwards seeks straight to ``resume_offset``.
    """
    if start_offset <= 0:
        return 0, 0
    resume = 0
    base = 0
    for _ev, end in tail_jsonl(path, 0):
        resume = end
        base += 1
        if end >= start_offset:
            # First boundary at/after the cursor — resume here. A mid-line
            # cursor (end > start_offset for the containing line) lands on the
            # NEXT newline, dropping that partial line; an exact boundary
            # (end == start_offset) resumes immediately after it.
            break
    return resume, base


def event_frames(path: Path, start_offset: int = 0) -> Iterator[tuple[dict[str, Any], int, int]]:
    """Yield ``(event, next_offset, seq)`` for the canonical event stream.

    The read side of spec §3.3 / §9.2's ``mode=events``:

    * ``next_offset`` is the byte position after the line's ``\\n`` — the
      AUTHORITATIVE resume cursor (feed it back as ``start_offset``).
    * ``seq`` is the 0-based line index, advisory, derived here at read-time
      and never stored in the JSONL; a consumer asserts "no gap" by checking
      it increments by exactly 1, but never seeks by it.

    A mid-line ``start_offset`` is resynced *forward* to the next ``\\n`` (§9.2;
    the containing partial line is skipped, never replayed — see
    :func:`_resync_and_base_seq`); only whole, CRC-valid lines are ever
    yielded.  This is a finite generator over the bytes present *now* — the
    live tail loop re-invokes it from the last ``next_offset`` on each wake.
    """
    resume, seq = _resync_and_base_seq(path, start_offset)
    for ev, end in tail_jsonl(path, resume):
        yield ev, end, seq
        seq += 1


# ---- turn materialisation -------------------------------------------


def _closed_by_successful_goal_met(turn: dict[str, Any]) -> bool:
    """True when the last tool of *turn* was ``goal action=met``.

    A goal auto-turn that marks the goal met used to cancel its own
    supervisor task, so the writer closed the turn as
    ``turn_interrupted``. That is a successful terminal, not a failure.
    """
    calls = turn.get("tool_calls") or []
    if not calls:
        return False
    last = calls[-1]
    if str(last.get("name") or "") != "goal":
        return False
    args = last.get("arguments") or {}
    if not isinstance(args, dict):
        return False
    return str(args.get("action") or "").lower() == "met"


def _empty_turn(turn_idx: int) -> dict[str, Any]:
    """The legacy per-turn dict skeleton every consumer expects.

    Mirrors the shape ``Session.add_turn`` produced plus the fine-
    tune fields. Consumers (``_render_one_saved_turn``, telegram
    ``/history``, memory ``on_session_end``) read a subset; the fold
    fills every field it can so no consumer regresses.
    """
    return {
        "user": "",
        "actors": [],  # WebUI author(s) of the prompt
        "source": "",  # non-user origin: queue_drain / loop_supervisor / …
        "thinking": "",
        "content": "",
        "started_at": "",
        "ended_at": "",
        "aborted": False,
        "error": None,
        "metrics": {},
        "in_progress": False,
        "engine": "",
        "agent_name": "",
        "model": "",
        "permission_mode": "",
        "cwd": "",
        "tool_calls": [],
        "tool_results": [],
        "round_metrics": [],
        # Ordered, per-generation render blocks reconstructed from event
        # ORDER (not round_idx, which is unreliable — it can stay 0 across
        # rounds). Lets a replay/render reproduce the live interleaving
        # exactly: assistant generation (thinking + content) → its tool
        # calls/results → next generation. Each block is either
        #   {"kind":"assistant","thinking","content","reasoning_chars",
        #    "thinking_ms"}
        # or
        #   {"kind":"tool","call_id","name","arguments","raw_arguments",
        #    "ok","content","content_chars","was_streaming"}.
        # Aggregate thinking/content/tool_calls/tool_results stay for
        # legacy consumers. (turn-replication: exact cross-frontend parity)
        "render_segments": [],
        # Durable semantic record of user-role messages injected after the
        # turn began. ``render_segments`` is a UI projection; local-LLM prior
        # turn replay intentionally does not consume it. Keeping injected
        # messages in their own semantic field prevents a background result or
        # type-ahead follow-up from disappearing from later model context while
        # preserving the canonical opening prompt in ``user``.
        "injected_messages": [],
        # replay-fidelity (from round_started / turn_ended events)
        "model_input_messages": [],
        "effective_system_prompt": "",
        "effective_system_prompt_sha256": "",
        "effective_preamble": [],
        "upstream_endpoint": "",
        # engine's own resume/session token for the turn's FINAL round — a
        # standalone training-row field. Folded from the
        # terminal ``turn_ended`` (authoritative) or a ``round_started``.
        "engine_session_id": "",
        "finish_reason": "",
        # lifecycle audit lists (from lifecycle events)
        "hook_outcomes": [],
        "approval_decisions": [],
        "artifact_references": [],
        "physical_turn_id": "",
        "compaction_events": [],
        "distill_events": [],
        "nudge_events": [],
        "pause_resume_log": [],
        "_turn_idx": turn_idx,
    }


_ENVELOPE_KEYS = frozenset({"v", "ts", "type", "engine", "turn_idx", "round_idx"})


def _payload_only(ev: dict[str, Any]) -> dict[str, Any]:
    """Strip the event envelope, returning just the payload fields."""
    return {k: v for k, v in ev.items() if k not in _ENVELOPE_KEYS}


def _ts_span_ms(start: str | None, end: str | None) -> float | None:
    """Milliseconds between two ISO-ms timestamps (``_now_iso_ms``
    output). Used as the per-generation thinking-duration source when no
    persisted ``thinking_display_ms`` exists — it measures the same
    streamed span the live TUI clocks, so it tracks closely. Returns
    ``None`` on any parse failure (renderer then omits the duration).
    """
    if not start or not end:
        return None
    try:
        from datetime import datetime

        a = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        ms = (b - a).total_seconds() * 1000.0
        return ms if ms >= 0 else None
    except (ValueError, TypeError):
        return None


def _finalize_segments(t: dict[str, Any], s: dict[str, Any]) -> None:
    """Close the trailing assistant generation, map per-round metrics
    (``reasoning_chars`` + thinking duration) onto generations IN ORDER,
    strip transient fields, and stamp ``t["render_segments"]``.
    """
    if s["cur"] is not None:
        s["gens"].append(s["cur"])
        s["cur"] = None
    # Flush notices buffered after the last assistant generation (a
    # trailing warning/compaction with no following content). Live's
    # eager bubble for that round stayed empty and is hidden, so the
    # notice is the only visible artifact — emit it at the tail.
    if s.get("pending_notices"):
        s["blocks"].extend(s["pending_notices"])
        s["pending_notices"] = []
    rms = s["round_metrics_seq"]
    for i, g in enumerate(s["gens"]):
        # Map by the generation's own round index (set at creation), not
        # positional ``rms[i]`` — a tool-only round contributes a round
        # metric but no generation, so positional mapping skews.
        ri = g.get("_round")
        if isinstance(ri, int) and 0 <= ri < len(rms):
            m = rms[ri]
        elif i < len(rms):
            m = rms[i]  # legacy fallback (older blocks without _round)
        else:
            m = {}
        # Per-generation round metrics so a renderer can stamp each
        # assistant bubble's stats line — matching the live driver, which
        # stamps every round's bubble (not just the final one). KEEP
        # ``model_alias`` so the stats line renders ``local/qwen3.6-27b-fp8``
        # (alias/model) — the operator wants to see the bucket they picked,
        # not the bare wire id.
        if m:
            g["metrics"] = dict(m)
        else:
            g["metrics"] = None
        rc = m.get("reasoning_chars")
        if isinstance(rc, int):
            g["reasoning_chars"] = rc
        # Duration: a persisted display value wins (exact parity with the
        # live title); else the streamed thinking ts-span; else nothing.
        tms = m.get("thinking_display_ms")
        if tms is None:
            tms = m.get("assistant_thinking_ms")
        if tms is None:
            tms = _ts_span_ms(g.get("_th_start"), g.get("_th_end"))
        g["thinking_ms"] = tms
    for b in s["blocks"]:
        b.pop("_th_start", None)
        b.pop("_th_end", None)
        b.pop("_round", None)
    t["render_segments"] = s["blocks"]
    _fill_recovered_plan(t)


def fold_events_into_turns(
    events: Iterator[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fold an event stream into the legacy per-turn dict list.

    One dict per ``turn_idx``, in turn order. ``text_delta`` /
    ``thinking_delta`` chunks concatenate into ``content`` / ``thinking``;
    ``tool_call`` / ``tool_result_meta`` / ``tool_result`` accumulate
    into ``tool_calls`` / ``tool_results``; ``round_ended`` metrics
    append to ``round_metrics``; ``turn_ended`` stamps the terminal
    fields (``final_content`` wins over accumulated deltas, since the
    engine may have rewritten the final text). ``turn_started`` stamps
    ``user`` / ``permission_mode`` / ``cwd``.

    The fold is lossless for the legacy shape: a consumer that read
    ``data["turns"]`` from the old ``<sid>.json`` sees the same keys.
    """
    # Turns are keyed by a monotonic FOLD KEY that advances on every
    # ``turn_started`` boundary — NOT by the raw ``turn_idx`` value. A log
    # whose writer reused a ``turn_idx`` would
    # otherwise collapse several real turns into one giant blob here; keying
    # on the boundary keeps them distinct. For a well-formed log (one
    # ``turn_started`` per monotonic ``turn_idx``) the fold key tracks the
    # ``turn_idx`` exactly, so output is byte-identical. The displayed
    # ``_turn_idx`` still carries the event's own value.
    turns: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    key_display: dict[int, int] = {}

    def _turn(key: int) -> dict[str, Any]:
        t = turns.get(key)
        if t is None:
            t = _empty_turn(key_display.get(key, key))
            turns[key] = t
            order.append(key)
        return t

    # Transient per-turn builder state for `render_segments` (ordered,
    # per-generation blocks). Segmentation follows event ORDER, matching
    # the live TUI's bubble boundaries: a `tool_call` closes the current
    # assistant generation; the first assistant delta AFTER a tool result
    # opens a new one. No empty assistant bubbles (consecutive tool calls
    # stay in the tool phase). (turn-replication)
    seg_state: dict[int, dict[str, Any]] = {}

    def _seg(idx: int) -> dict[str, Any]:
        s = seg_state.get(idx)
        if s is None:
            s = {
                "blocks": [],
                "cur": None,
                "after_tr": False,
                "gens": [],
                "calls": {},
                "round_metrics_seq": [],
                "current_round": 0,
                # Notices (runtime warnings / compaction) that arrived with
                # NO assistant generation open — buffered so they land
                # AFTER the next assistant bubble (or before the next tool
                # block), matching the live eager-mounted bubble that takes
                # the surrounding content. (turn-replication: notice order)
                "pending_notices": [],
            }
            seg_state[idx] = s
        return s

    def _emit_notice(s: dict[str, Any], notice: dict[str, Any]) -> None:
        """Place a notice block to match the LIVE transcript order.

        If an assistant generation is open, the live bubble is already
        mounted and the widget mounts right after it → append now. If NO
        generation is open (pre-content, or just after a tool_result), the
        live TUI has an EAGER bubble mounted that the upcoming content will
        fill, with the widget mounted after it → buffer the notice and
        flush it once that generation block exists (or before the next
        tool block). (turn-replication: notice order)
        """
        if s["cur"] is not None:
            s["blocks"].append(notice)
        else:
            s["pending_notices"].append(notice)

    def _flush_pending_notices(s: dict[str, Any]) -> None:
        if s["pending_notices"]:
            s["blocks"].extend(s["pending_notices"])
            s["pending_notices"] = []

    def _seg_apply(idx: int, etype: str, ev: dict[str, Any]) -> None:
        s = _seg(idx)
        # Runtime-warning chunk (loop / nudge / distill / upstream-retry /
        # force-terminate). The engine emits these as a TextDelta whose
        # body is ``\n\n{⚠|↻} runtime: <event> · …\n\n``; the live TUI
        # sniffs the glyph prefix and mounts a RuntimeWarningNotice widget
        # INSTEAD of appending the text to the content bubble. Mirror that
        # here: emit an ordered ``notice`` block and DON'T add the warning
        # to assistant content, so a replayed/rendered turn renders the
        # same widget rather than inlining the warning prose.
        # (turn-replication: runtime-warning parity)
        if etype == "text_delta":
            _wd = ev.get("delta", "") or ""
            _ws = _wd.lstrip()
            if _ws[:1] in ("⚠", "↻") and "runtime:" in _ws[:40]:
                notice = {
                    "kind": "notice",
                    "subtype": "runtime_warning",
                    "text": _wd,
                }
                if "runtime: nudge" in _ws[:40].lower():
                    if s["cur"] is not None:
                        s["gens"].append(s["cur"])
                        s["cur"] = None
                    s["blocks"].append(notice)
                    s["after_tr"] = True
                else:
                    _emit_notice(s, notice)
                return
        if etype == "compaction":
            # Mid-turn prompt compaction (proactive or post-overflow, and
            # the ``/compact`` command). Persisted as a lifecycle event;
            # the live TUI mounts a CompactionNotice. Emit an ordered
            # notice block so replay reproduces it. (loop/nudge/distill
            # lifecycle events are NOT rendered here — they already surface
            # via the runtime-warning TextDelta path above; rendering them
            # again would double up.) (turn-replication: compaction parity)
            _emit_notice(
                s,
                {
                    "kind": "notice",
                    "subtype": "compaction",
                    "payload": _payload_only(ev),
                },
            )
            return
        if etype in ("text_delta", "thinking_delta"):
            if s["cur"] is None or s["after_tr"]:
                s["cur"] = {
                    "kind": "assistant",
                    "thinking": "",
                    "content": "",
                    "reasoning_chars": None,
                    "thinking_ms": None,
                    "metrics": None,
                    "_th_start": None,
                    "_th_end": None,
                    # Index of the round this generation belongs to =
                    # rounds completed so far (the round currently
                    # in-flight). Lets `_finalize_segments` map the RIGHT
                    # round metrics onto the generation even when an
                    # earlier round was tool-only (no generation) — a
                    # positional ``rms[i]`` would skew by one per
                    # tool-only round. (turn-replication: per-round parity)
                    "_round": int(s.get("current_round") or 0),
                }
                s["blocks"].append(s["cur"])
                s["after_tr"] = False
                # The eager bubble now exists — flush any notice that
                # arrived before it so the widget lands AFTER the bubble
                # (it keeps growing as more deltas land in ``cur``).
                _flush_pending_notices(s)
            delta = ev.get("delta", "") or ""
            if etype == "text_delta":
                s["cur"]["content"] += delta
            else:
                s["cur"]["thinking"] += delta
                ts = ev.get("ts")
                if ts:
                    if s["cur"]["_th_start"] is None:
                        s["cur"]["_th_start"] = ts
                    s["cur"]["_th_end"] = ts
        elif etype == "tool_call":
            if s["cur"] is not None:
                s["gens"].append(s["cur"])
                s["cur"] = None
            # A notice buffered before any content of this round (the live
            # eager bubble stayed empty → hidden) lands BEFORE the tool
            # block, matching the live ``[warning widget][tool widget]``.
            _flush_pending_notices(s)
            cid = ev.get("call_id", "")
            tb = {
                "kind": "tool",
                "call_id": cid,
                "name": ev.get("name", ""),
                "arguments": ev.get("arguments", {}),
                "raw_arguments": ev.get("raw_arguments", ""),
                "ok": None,
                "content": None,
                "content_chars": 0,
                "was_streaming": False,
            }
            s["blocks"].append(tb)
            s["calls"][cid] = tb
            s["after_tr"] = False
        elif etype == "queue_msg_injected":
            # A queued follow-up was injected mid-turn — render it as an inline
            # user bubble at this point in the turn (closing any open assistant
            # generation first, like a tool_call does).
            if s["cur"] is not None:
                s["gens"].append(s["cur"])
                s["cur"] = None
            _flush_pending_notices(s)
            seg = {
                "kind": "user",
                "text": ev.get("text", ""),
                "queue_id": ev.get("queue_id", ""),
                "source": str(ev.get("source") or ""),
            }
            who = normalize_actor(ev.get("actor"))
            if who is not None:
                seg["actor"] = who  # Injected steer keeps its author
            s["blocks"].append(seg)
            s["after_tr"] = False
        elif etype == "tool_result_meta":
            tb = s["calls"].get(ev.get("call_id", ""))
            if tb is not None:
                tb.update({k: ev[k] for k in ("outcome", "member_outcomes") if k in ev})
                tb["ok"] = bool(ev.get("ok"))
                tb["content_chars"] = int(ev.get("content_chars", 0) or 0)
            s["after_tr"] = True
        elif etype == "tool_delta":
            tb = s["calls"].get(ev.get("call_id", ""))
            if tb is not None:
                tb["was_streaming"] = True
                tb["content"] = (tb["content"] or "") + (ev.get("delta", "") or "")
        elif etype == "tool_result":
            tb = s["calls"].get(ev.get("call_id", ""))
            if tb is not None:
                # Final authoritative body replaces any streamed text —
                # same semantics as the live finalize_streaming path.
                tb["content"] = ev.get("content", "")
            s["after_tr"] = True
        elif etype == "round_started":
            if s["cur"] is not None:
                s["gens"].append(s["cur"])
                s["cur"] = None
            s["current_round"] = int(ev.get("round_idx") or 0)
            s["after_tr"] = True
        elif etype == "round_ended":
            if s["cur"] is not None:
                s["gens"].append(s["cur"])
                s["cur"] = None
            s["round_metrics_seq"].append(ev.get("metrics") or {})
            s["after_tr"] = True

    cur_key = -1
    current_agent = ""
    delegation_calls: dict[str, dict[str, Any]] = {}
    delegation_children: dict[str, dict[str, Any]] = {}
    for ev in events:
        etype = ev.get("type")
        ti = ev.get("turn_idx")
        if etype == "subagent_linked":
            cid = str(ev.get("call_id") or "")
            block = delegation_calls.get(cid)
            if block is None and "." in cid:
                block = delegation_calls.get(cid.rsplit(".", 1)[0])
            child_sid = str(ev.get("child_session_id") or "")
            if block is not None and child_sid:
                children = block.setdefault("delegations", [])
                if not any(c["call_id"] == cid for c in children):
                    child = {
                        "call_id": cid,
                        "task_id": child_sid,
                        "agent": str(ev.get("child_agent_name") or ""),
                    }
                    children.append(child)
                    delegation_children[child_sid] = child
            continue
        if etype == "subagent_result":
            # Terminal records are not scoped to the dispatching parent turn.
            # Keep their outcome on the original card even after job cleanup.
            child = delegation_children.get(str(ev.get("child_sid") or ""))
            if child is not None:
                ok = ev.get("ok") is not False and not ev.get("error")
                child.update(status=ev.get("status") or ("done" if ok else "failed"), ok=bool(ok))
                if ev.get("job_id"):
                    child["job_id"] = str(ev["job_id"])
            continue
        if etype == "subagent_done":
            # Foreground (fan-out) member terminal edge: the batch returns one
            # aggregate tool result, so this is the only durable record that
            # THIS member finished while its siblings kept running.
            child = delegation_children.get(str(ev.get("child_session_id") or ev.get("child_sid") or ""))
            if child is not None:
                ok = ev.get("ok") is not False
                child.update(
                    status=str(ev.get("status") or ("done" if ok else "failed")),
                    ok=bool(ok),
                )
            continue
        if etype == "session_started":
            current_agent = str(ev.get("agent") or ev.get("agent_name") or "")
            continue
        if etype == "agent_switched":
            current_agent = str(ev.get("to_agent") or "").strip()
            continue
        if ti is None:
            # header_update / session_closed — not turn-scoped.
            continue
        ti_i = int(ti)
        # Advance the fold key on a turn boundary; also open a key for the
        # very first turn-scoped event if a paged read starts mid-turn (no
        # leading ``turn_started`` in this slice). As a fallback for
        # older/partial/malformed logs that omit ``turn_started`` for a later
        # turn, also advance when the raw ``turn_idx`` changes — this matches
        # the legacy raw-``turn_idx`` grouping so such turns don't merge into
        # the previous one. (fold boundary fallback.)
        if etype == "turn_started":
            prev = turns.get(cur_key) if cur_key >= 0 else None
            if prev is not None and not prev.get("_terminal"):
                prev["aborted"] = True
                prev["_terminal"] = True
            cur_key += 1
            key_display[cur_key] = ti_i
        elif cur_key < 0:
            cur_key = 0
            key_display[cur_key] = ti_i
        elif ti_i != key_display.get(cur_key):
            prev = turns.get(cur_key)
            if prev is not None and not prev.get("_terminal"):
                # Logical relay can omit abort's turn_ended and the retry's
                # turn_started when they share physical_turn_id. A turn_idx
                # bump is still a new attempt — close the previous card as
                # aborted and keep the operator prompt on the retry.
                prev["aborted"] = True
                prev["_terminal"] = True
            cur_key += 1
            key_display[cur_key] = ti_i
        t = _turn(cur_key)
        if (
            etype != "turn_started"
            and not t.get("user")
            and cur_key > 0
            and turns.get(cur_key - 1)
            and turns[cur_key - 1].get("aborted")
            and turns[cur_key - 1].get("user")
            and key_display.get(cur_key) != key_display.get(cur_key - 1)
        ):
            t["user"] = turns[cur_key - 1]["user"]
        if ev.get("physical_turn_id") and not t.get("physical_turn_id"):
            t["physical_turn_id"] = str(ev["physical_turn_id"])
        _seg_apply(cur_key, etype or "", ev)
        if etype == "tool_call" and str(ev.get("name") or "").lower() in _KEEP_FULL_DELEGATE_NAMES:
            cid = str(ev.get("call_id") or "")
            delegation_calls[cid] = _seg(cur_key)["calls"][cid]
        engine = ev.get("engine") or ""
        if engine and not t["engine"]:
            t["engine"] = engine
        if etype == "turn_started":
            t["user"] = ev.get("user", "")
            t["permission_mode"] = ev.get("permission_mode", "") or ""
            t["cwd"] = ev.get("cwd", "") or ""
            t["started_at"] = ev.get("ts", "") or t["started_at"]
            t["agent_name"] = str(ev.get("agent_name") or ev.get("agent") or current_agent or "")
            t["actors"] = normalize_actors(ev.get("actors"))
            t["source"] = str(ev.get("source") or "")
            if ev.get("attachments"):
                t["attachments"] = ev["attachments"]
        elif etype == "queue_msg_injected":
            # This is model-visible semantic input, not merely an inline UI
            # decoration. Store it separately from ``user`` so later-turn
            # replay can restore every injected message exactly once.
            t["injected_messages"].append(
                {
                    "text": ev.get("text", "") or "",
                    "queue_id": ev.get("queue_id", "") or "",
                    "source": ev.get("source", "") or "",
                    "author": ev.get("author", "") or "",
                    "round_idx": ev.get("round_idx"),
                    "ts": ev.get("ts", "") or "",
                }
            )
        elif etype == "round_started":
            if ev.get("model") and not t["model"]:
                t["model"] = ev["model"]
            if ev.get("model_input"):
                t["model_input_messages"] = ev["model_input"]
            if ev.get("effective_system_prompt"):
                t["effective_system_prompt"] = ev["effective_system_prompt"]
            if ev.get("effective_system_prompt_sha256"):
                t["effective_system_prompt_sha256"] = ev["effective_system_prompt_sha256"]
            if ev.get("effective_preamble"):
                t["effective_preamble"] = ev["effective_preamble"]
            if ev.get("upstream_endpoint"):
                t["upstream_endpoint"] = ev["upstream_endpoint"]
            if ev.get("engine_session_id"):
                t["engine_session_id"] = ev["engine_session_id"]
        elif etype == "text_delta":
            t["content"] += ev.get("delta", "") or ""
        elif etype == "thinking_delta":
            t["thinking"] += ev.get("delta", "") or ""
        elif etype == "tool_call":
            t["tool_calls"].append(
                {
                    "call_id": ev.get("call_id", ""),
                    "name": ev.get("name", ""),
                    "arguments": ev.get("arguments", {}),
                    "raw_arguments": ev.get("raw_arguments", ""),
                    "started_at": ev.get("started_at", ""),
                }
            )
            # A plan published via ExitPlanMode IS the turn's deliverable, so it
            # belongs in the answer — not only in the tool-call record, which no
            # surface renders on reopen. The live block stream already paints it
            # as a card; the fold dropped it, so reopening a Plan session showed
            # narration about a plan the operator could not read. When the
            # plan/<topic>.md write is also refused, this payload is the plan's
            # only copy. Same "📐 Plan" marker the live renderer uses, so the two
            # surfaces cannot drift.
            if ev.get("name") == "ExitPlanMode" or _is_plan_tool_name(ev.get("name")):
                _args = ev.get("arguments") or {}
                _plan = _plan_from_tool_args(_args)
                if _plan:
                    t.setdefault("_plans", []).append(_plan)
        elif etype == "tool_result_meta":
            t["tool_results"].append(
                {
                    "call_id": ev.get("call_id", ""),
                    "ok": bool(ev.get("ok")),
                    "ended_at": ev.get("ended_at", ""),
                    "content_chars": ev.get("content_chars", 0),
                    **{k: ev[k] for k in ("outcome", "member_outcomes") if k in ev},
                }
            )
        elif etype == "tool_result":
            # Body event (gated). Attach to the matching meta row if
            # present, else stash a body-only row so harvesters see it.
            cid = ev.get("call_id", "")
            _images = ev.get("images") if isinstance(ev.get("images"), list) else None
            for r in reversed(t["tool_results"]):
                if r.get("call_id") == cid and "content" not in r:
                    r["content"] = ev.get("content", "")
                    if _images:
                        r["images"] = _images
                    break
            else:
                t["tool_results"].append(
                    {
                        "call_id": cid,
                        "content": ev.get("content", ""),
                        **({"images": _images} if _images else {}),
                    }
                )
            body = str(ev.get("content") or "").strip()
            if body and not _is_plan_status_or_fence(body) and not t.get("_plans"):
                for tc in reversed(t.get("tool_calls") or []):
                    if tc.get("call_id") == cid and _is_plan_tool_name(tc.get("name")):
                        t.setdefault("_plans", []).append(body)
                        break
        elif etype == "round_ended":
            m = ev.get("metrics") or {}
            if ev.get("tokenizer_id"):
                m = {**m, "tokenizer_id": ev["tokenizer_id"]}
            t["round_metrics"].append(m)
            if ev.get("finish_reason"):
                t["finish_reason"] = ev["finish_reason"]
        elif etype == "hook_outcome":
            t["hook_outcomes"].append(_payload_only(ev))
        elif etype == "approval_decision":
            t["approval_decisions"].append(_payload_only(ev))
        elif etype == "artifact_reference":
            t["artifact_references"].append(_payload_only(ev))
        elif etype == "compaction":
            t["compaction_events"].append(_payload_only(ev))
        elif etype == "distill":
            t["distill_events"].append(_payload_only(ev))
        elif etype == "nudge":
            t["nudge_events"].append(_payload_only(ev))
        elif etype == "pause_resume":
            t["pause_resume_log"].append(_payload_only(ev))
        elif etype == "loop_detected":
            t.setdefault("loop_events", []).append(_payload_only(ev))
        elif etype == "turn_ended":
            # turn_ended repeats the author(s) (fresh folds / truncated heads).
            if not t.get("actors") and ev.get("actors"):
                t["actors"] = normalize_actors(ev.get("actors"))
            if not t.get("source") and ev.get("source"):
                t["source"] = str(ev.get("source") or "")
            if not t.get("attachments") and ev.get("attachments"):
                t["attachments"] = ev["attachments"]
            fc = ev.get("final_content")
            if fc is not None:
                t["content"] = fc
            # AFTER final_content: turn_ended REPLACES the accumulated content,
            # so a plan appended when the tool call was seen is discarded. The
            # plan is the turn's deliverable and must outlive that replacement.
            t["content"] = plan_augmented_content(
                t["content"],
                [{"name": "ExitPlanMode", "arguments": {"plan": _p}} for _p in (t.get("_plans") or [])],
            )
            ft = ev.get("final_thinking")
            if ft is not None:
                t["thinking"] = ft
            t["_terminal"] = True
            if ev.get("model_input_messages"):
                t["model_input_messages"] = ev["model_input_messages"]
            if ev.get("effective_system_prompt"):
                t["effective_system_prompt"] = ev["effective_system_prompt"]
            if ev.get("effective_system_prompt_sha256"):
                t["effective_system_prompt_sha256"] = ev["effective_system_prompt_sha256"]
            if ev.get("effective_preamble"):
                t["effective_preamble"] = ev["effective_preamble"]
            if ev.get("upstream_endpoint"):
                t["upstream_endpoint"] = ev["upstream_endpoint"]
            if ev.get("engine_session_id"):
                t["engine_session_id"] = ev["engine_session_id"]
            t["aborted"] = bool(ev.get("aborted"))
            t["error"] = ev.get("error")
            if ev.get("error_classification"):
                t["error_classification"] = ev["error_classification"]
            if (
                t.get("aborted")
                and str(t.get("error_classification") or "") == "turn_interrupted"
                and _closed_by_successful_goal_met(t)
            ):
                t["aborted"] = False
                t["error"] = None
                t["error_classification"] = ""
            t["metrics"] = ev.get("metrics") or t["metrics"]
            # ``turn_ended`` metrics omit session-stable fields that the
            # engine only stamps on per-round records — notably
            # ``context_window`` (the metrics-line renderer appends
            # ``/<window>`` to the ``in <pt>`` segment from it). Without
            # this, a replayed/rendered turn drops the ``/256000`` suffix
            # the live driver shows, because the driver renders from a
            # round metric while replay renders from the turn metric.
            # Backfill context_window AND model_alias from the last round that
            # carries them, so the turn's closing stats line renders the full
            # ``local/qwen3.6-27b-fp8 · … /256000`` label (the turn_ended event
            # omits both session-stable fields that only the per-round records
            # stamp). (turn-replication: exact metrics-line parity)
            if isinstance(t["metrics"], dict):
                _tm = t["metrics"]
                for _field in ("context_window", "model_alias"):
                    if _tm.get(_field) in (None, ""):
                        for _rm in reversed(t.get("round_metrics") or []):
                            if isinstance(_rm, dict) and _rm.get(_field) not in (None, ""):
                                _tm = {**_tm, _field: _rm[_field]}
                                break
                t["metrics"] = _tm
            t["ended_at"] = ev.get("ts", "") or t["ended_at"]
            t["in_progress"] = False
            # The terminal record now also stamps the permission mode;
            # honor it if turn_started was missing / lost (don't clobber
            # an already-folded non-empty value with an empty one).
            if ev.get("permission_mode") and not t.get("permission_mode"):
                t["permission_mode"] = ev["permission_mode"]

    # For an IN-FLIGHT turn (no turn_ended seen), content/thinking were
    # assembled from per-chunk deltas that were each scrubbed
    # independently — a secret split across a debounce boundary could
    # reassemble in the clear here. Re-scrub the folded text for any turn
    # that didn't reach turn_ended (terminal turns already carry the
    # scrubbed final_content).
    out = []
    for i in order:
        t = turns[i]
        s = seg_state.get(i)
        if s is not None:
            _finalize_segments(t, s)
        if not t.pop("_terminal", False):
            from sharedagents_core.secrets_scrubber import scrub_text

            if t.get("content"):
                t["content"] = scrub_text(t["content"])
            if t.get("thinking"):
                t["thinking"] = scrub_text(t["thinking"])
            # An in-flight turn's render blocks were assembled from
            # per-chunk deltas (each scrubbed independently — a secret
            # split across a debounce boundary could reassemble in the
            # clear). Re-scrub block text, same policy as the aggregate.
            for b in t.get("render_segments") or []:
                if b.get("kind") == "assistant":
                    if b.get("thinking"):
                        b["thinking"] = scrub_text(b["thinking"])
                    if b.get("content"):
                        b["content"] = scrub_text(b["content"])
                elif b.get("kind") == "tool" and b.get("content"):
                    b["content"] = scrub_text(b["content"])
            t["in_progress"] = True
        # Older logs stored the saved-attachment block inside ``user``: show
        # the typed text and keep the references beside it (replay restores
        # the block through attachment_context.model_user).
        from sharedagents_core.attachment_context import project_turn

        project_turn(t)
        out.append(t)
    return out


def fold_queue_state(events: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the queue events into the ordered list of PENDING follow-up items.

    `queue_msg_added` inserts a pending item; `queue_msg_updated` rewrites its
    text iff still pending; `queue_msg_removed` / `queue_msg_injected` drop it.
    Returns pending items in submission order.
    """
    pending: dict[str, dict[str, Any]] = {}
    for ev in events:
        et = ev.get("type")
        qid = ev.get("queue_id")
        if not qid:
            continue
        if et == "queue_msg_added":
            pending[qid] = {
                "queue_id": qid,
                "text": ev.get("text", ""),
                "source": ev.get("source", ""),
                "author": ev.get("author", ""),
                "actor": ev.get("actor") if isinstance(ev.get("actor"), dict) else None,
                # The author's grants; None = not stamped.
                "access": ev.get("access") if isinstance(ev.get("access"), dict) else None,
                "added_ts": ev.get("ts", ""),
            }
        elif et == "queue_msg_updated":
            if qid in pending:
                pending[qid]["text"] = ev.get("text", pending[qid]["text"])
                if isinstance(ev.get("access"), dict):  # re-stamped by an edit
                    pending[qid]["access"] = ev.get("access")
        elif et in ("queue_msg_removed", "queue_msg_injected"):
            pending.pop(qid, None)
    return list(pending.values())


PLAN_MARKER = "📐 Plan"


def _is_plan_status_or_fence(text: str) -> bool:
    """True for ExitPlanMode status wrappers, not the operator-facing plan."""
    t = str(text or "")
    if "UNTRUSTED-DATA" in t:
        return True
    return "Plan ready for operator review" in t


def _is_plan_tool_name(name: str) -> bool:
    try:
        from sharedagents_agent.tools.widget_adapt import is_plan_tool_name
    except Exception:
        key = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
        return key in {"exitplanmode", "createplan"}
    return is_plan_tool_name(name)


def _fill_recovered_plan(t: dict[str, Any]) -> None:
    """Attach Cursor ``*.plan.md`` onto empty Create Plan calls.

    ACP often persists ``arguments: {}`` and ``content_chars: 0``. The
    markdown still lands on disk; history fold must surface it so the
    📐 card is what the operator approves.
    """
    try:
        from sharedagents_agent.tools.widget_adapt import recover_cursor_plan_markdown
    except Exception:
        return
    hint = str(t.get("engine_session_id") or "")
    recovered = ""
    for tc in t.get("tool_calls") or []:
        if not _is_plan_tool_name(tc.get("name")):
            continue
        if _plan_from_tool_args(tc.get("arguments")):
            continue
        recovered = recovered or recover_cursor_plan_markdown(session_hint=hint)
        if not recovered:
            return
        args = tc.get("arguments")
        args = dict(args) if isinstance(args, dict) else {}
        args["plan"] = recovered
        tc["arguments"] = args
        t.setdefault("_plans", []).append(recovered)
    for b in t.get("render_segments") or []:
        if b.get("kind") != "tool" or not _is_plan_tool_name(b.get("name")):
            continue
        if _plan_from_tool_args(b.get("arguments")):
            continue
        recovered = recovered or recover_cursor_plan_markdown(session_hint=hint)
        if not recovered:
            return
        args = b.get("arguments")
        args = dict(args) if isinstance(args, dict) else {}
        args["plan"] = recovered
        b["arguments"] = args
        if not str(b.get("content") or "").strip():
            b["content"] = recovered
    if recovered and t.get("_plans"):
        t["content"] = plan_augmented_content(
            t.get("content") or "",
            [{"name": "ExitPlanMode", "arguments": {"plan": p}} for p in t["_plans"]],
        )


def _plan_from_tool_args(args) -> str:
    try:
        from sharedagents_agent.tools.widget_adapt import extract_plan_markdown
    except Exception:
        extract_plan_markdown = None
    if extract_plan_markdown is not None:
        return extract_plan_markdown(args)
    if isinstance(args, str):
        try:
            import json as _json

            args = _json.loads(args)
        except (ValueError, TypeError):
            args = {}
    if isinstance(args, dict):
        return str(args.get("plan") or "").strip()
    return ""


def plan_augmented_content(content: str, tool_calls) -> str:
    """Append any ExitPlanMode plan bodies to a turn's assistant content.

    A plan published through ExitPlanMode IS the turn's deliverable, but it
    lives in the tool call's ``arguments``, not in the assistant text — so every
    surface that renders only the text showed narration about a plan the reader
    could not see.

    ONE implementation deliberately shared by both history paths: the physical
    fold (:func:`fold_events_into_turns`) and the logical conversation
    projection (``project_completed_turn``'s caller). They are separate stores
    with separate writers, and a second copy of this rule is how they would
    drift into showing the same plan two different ways.

    Idempotent: a plan already present in ``content`` is not appended twice, so
    a re-fold or a re-projection cannot duplicate it.
    """
    out = str(content or "")
    for call in tool_calls or []:
        if isinstance(call, dict):
            name = call.get("name") or ""
            args = call.get("arguments")
        else:
            name = getattr(call, "name", "") or ""
            args = getattr(call, "arguments", None)
        if name != "ExitPlanMode" and not _is_plan_tool_name(name):
            continue
        plan = _plan_from_tool_args(args)
        if plan and not _is_plan_status_or_fence(plan) and plan not in out:
            out += ("\n\n" if out else "") + PLAN_MARKER + "\n\n" + plan
    return out


def scan_history_with_offsets(path: Path) -> tuple[list[dict[str, Any]], int, bool]:
    """Fold COMPLETED turns and report the live-tail resume offset.

    Returns ``(turns, tail_offset, active)``:

    * ``turns`` — folded turns for completed turns only. An in-flight turn
      (a ``turn_started`` with no matching ``turn_ended``) is EXCLUDED so the
      caller renders a clean page and the live stream replays the in-flight
      turn from its ``turn_started`` (no gap, no overlap).
    * ``tail_offset`` — byte offset to resume the live stream from: the
      in-flight turn's ``turn_started`` START offset if a turn is in flight,
      else the consumed EOF.
    * ``active`` — ``True`` iff a turn is in flight.
    """
    open_starts: list[int] = []  # line-start offsets of turn_starts not yet closed
    events: list[tuple[dict[str, Any], int]] = []  # (event, line_start_offset)
    line_start = 0
    eof = 0
    last_ended = -1  # line-start offset of the last turn_ended seen
    for ev, end in tail_jsonl(path, 0):
        etype = ev.get("type")
        if etype == "turn_started":
            open_starts.append(line_start)
        elif etype == "turn_ended":
            last_ended = line_start
            if open_starts:
                open_starts.pop()
        events.append((ev, line_start))
        line_start = end
        eof = end
    # An unmatched turn_started BEFORE the last completed turn was ABANDONED —
    # an abort, a kill, a crash — not a turn in flight. Only one that no
    # turn_ended follows can still be running.
    #
    # Cutting at the earliest unmatched start (the previous behaviour) meant a
    # single stale orphan truncated the page for the rest of the session's life:
    # every load rendered only the turns before the orphan and the live stream
    # replayed the remainder, so the operator watched a stale tail get replaced
    # seconds later — "shows the last message then flickers to some other turn".
    # Measured on session <session-id> (16 turn_started / 15
    # turn_ended): 22 messages painted, then 32 with a different last message
    # 2.3s later. The three sessions probed with balanced counts never did it.
    live_starts = [ls for ls in open_starts if ls > last_ended]
    if live_starts:
        # A turn really is in flight. Cut at the EARLIEST such start so the
        # completed page excludes it and the live stream replays it from its
        # turn_started. (A stray turn_ended with no open turn is ignored — it
        # can't make a session look complete when it isn't.)
        tail_offset = live_starts[0]
        active = True
        kept = [ev for ev, ls in events if ls < tail_offset]
    else:
        tail_offset = eof
        active = False
        kept = [ev for ev, _ in events]
    return fold_events_into_turns(iter(kept)), tail_offset, active


def fold_jsonl_into_turns(path: Path, start_offset: int = 0) -> list[dict[str, Any]]:
    """Convenience: tail a JSONL file from disk and fold to turns."""
    return fold_events_into_turns(ev for ev, _off in tail_jsonl(path, start_offset))


def session_log_path(sessions_dir: Path, sid: str) -> Path | None:
    """Return the durable JSONL for a session or conversation hash.

    Physical CoreBackend logs live at ``<sessions_dir>/<sid>.jsonl``.
    Coordinator conversations live at
    ``<sessions_dir>/conversations/<sid>.jsonl``. Founding same-hash ACP
    lanes often keep only the conversation log after a gateway restart
    (no ``sessions_meta`` row, no top-level JSONL). Callers that 404 on
    the physical path alone then fail ``/model`` and prompt submit.
    """
    sessions_dir = Path(sessions_dir)
    if not sid:
        return None
    path = sessions_dir / f"{sid}.jsonl"
    return path if path.is_file() else None


def load_session_dict(
    sessions_dir: Path,
    sid: str,
    *,
    expected_agent: str | None = None,
    fold: bool = True,
) -> dict[str, Any] | None:
    """Return a legacy ``<sid>.json``-shaped dict for one session: the
    materialised header (sessions.db, refreshed via tail) plus the turns
    folded from ``<sid>.jsonl``. ``None`` when the session is unknown.

    The single read helper for consumers that used to
    ``json.loads(<sid>.json)`` (telegram /status·/history·/context, etc.).

    ``fold`` controls the (O(file)) whole-log fold into ``d["turns"]``. Pass
    ``fold=False`` when the caller only needs the cheap sessions.db header and
    folds the log itself by another path — e.g. the web / agent-monitor
    history route already runs :func:`scan_history_with_offsets`, so folding
    the whole file a SECOND time here (only to read the last turn's model)
    doubled the cost of opening a large conversation. With ``fold=False``,
    ``d["turns"]`` is ``[]``. Default ``True`` keeps every existing caller
    unchanged.

    ``expected_agent`` is the OPTIONAL cross-agent scope check. When
    set + the loaded session's ``agent_name`` is also populated +
    they disagree, the function returns ``None`` instead of leaking
    another agent's session into the caller. Empty session_agent
    bypasses the check so legacy / unattributed sessions stay
    accessible to whichever consumer finds them. Existing callers that don't pass
    ``expected_agent`` behave exactly as before.
    """
    sessions_dir = Path(sessions_dir)
    jsonl = sessions_dir / f"{sid}.jsonl"
    db = sessions_dir / "sessions.db"
    if not jsonl.exists() and not db.exists():
        return None
    from sharedagents_session.session_index import (  # local import avoids a cycle
        SessionIndex,
        connect_sessions_db,
    )

    d: dict[str, Any] = {}
    try:
        SessionIndex(sessions_dir).tail(sid)
        con = connect_sessions_db(db)
        try:
            row = con.execute("SELECT * FROM sessions_meta WHERE session_hash=?", (sid,)).fetchone()
        finally:
            con.close()
        if row is not None:
            d = dict(row)
            d["hash"] = d.get("session_hash")
            pf = d.get("pinned_facts")
            if isinstance(pf, str):
                try:
                    d["pinned_facts"] = json.loads(pf)
                except (ValueError, TypeError):
                    d["pinned_facts"] = []
    except Exception:  # noqa: BLE001
        d = {"hash": sid}
    sess_agent = str(d.get("agent_name") or "")
    if expected_agent and sess_agent and sess_agent != expected_agent:
        return None
    if fold:
        try:
            d["turns"] = fold_jsonl_into_turns(jsonl)
        except Exception:  # noqa: BLE001
            d["turns"] = []
    else:
        d["turns"] = []
    if not d.get("hash"):
        d["hash"] = sid
    # ACP/CLI engines historically left last_prompt_tokens=0 on the header
    # even when turn_ended.metrics carried a real occupancy (cursor WebUI
    # sat at 0/200K). Heal the returned dict from the newest turn so a
    # joining interface sees the meter without waiting for a new turn.
    # Also heal ESTIMATED occupancy collapse: after a process restart the
    # newest turn can be far below the session peak (580 after 15889).
    _backfill_context_meter_from_turns(d, jsonl)
    return d


def _backfill_context_meter_from_turns(
    d: dict[str, Any],
    jsonl: Path,
) -> None:
    """Fill last_prompt_tokens / context_window from turn_ended metrics.

    Read-only: does not rewrite the header. Cheap reverse scan.
    When the newest estimated occupancy collapsed below the session peak
    (ACP in-memory char total reset on restart), prefer the peak so the
    footer does not jump to 580/200K. Real (non-estimated) newest usage
    stays authoritative so a compact can shrink the meter.
    """
    have_pt = int(d.get("last_prompt_tokens") or 0)
    have_cw = int(d.get("context_window") or 0) > 0
    if not jsonl.exists():
        return
    (newest_pt, newest_cw, newest_estimated), (peak_pt, peak_cw) = _context_meter_walk(jsonl)
    pt = max(have_pt, newest_pt, peak_pt) if newest_estimated else newest_pt or have_pt
    cw = newest_cw or peak_cw
    if pt > have_pt:
        d["last_prompt_tokens"] = pt
    if (not have_cw) and cw > 0:
        d["context_window"] = cw


def _context_meter_int(value: Any) -> int:
    """``int(value)`` or 0 — the meter fields are best-effort diagnostics."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _context_meter_walk(
    jsonl: Path,
) -> tuple[tuple[int, int, bool], tuple[int, int]]:
    """Read BOTH context-meter readings from ONE bounded reverse walk.

    Returns ``((newest_pt, newest_cw, newest_estimated), (peak_pt, peak_cw))``
    — the newest ``turn_ended`` carrying occupancy plus the high-water values
    across the same window.

    The two readings used to cost two independent 80-event reverse walks of
    the very same bytes. On a session whose recent records are oversized that
    redundancy was ~890 ms of every ``load_session_dict`` (operator store,
    2026-09-09/10: ``reverse_jsonl`` alone measured 423-722 ms per walk on the
    three affected sessions) — paid twice to read the same turn metrics.
    """
    newest: tuple[int, int, bool] | None = None
    peak_pt = 0
    peak_cw = 0
    try:
        for event, _start, _end in reverse_jsonl(jsonl, max_events=80):
            if event.get("type") != "turn_ended":
                continue
            metrics = event.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            pt = _context_meter_int(metrics.get("prompt_tokens"))
            cw = _context_meter_int(metrics.get("context_window"))
            if newest is None and (pt > 0 or cw > 0):
                newest = (pt, cw, bool(metrics.get("tokens_estimated")))
            if pt > peak_pt:
                peak_pt = pt
            if cw > peak_cw:
                peak_cw = cw
    except Exception:  # noqa: BLE001 — meter reads are best-effort
        pass
    return newest or (0, 0, False), (peak_pt, peak_cw)


def last_turn_context_meter(jsonl: Path) -> tuple[int, int]:
    """Newest ``turn_ended`` occupancy as ``(prompt_tokens, context_window)``.

    ``(0, 0)`` when the log has no usable metrics. Scans backwards so a
    long session does not pay O(file).
    """
    newest, _peak = _context_meter_walk(jsonl)
    pt, cw, _estimated = newest
    return pt, cw


def last_turn_context_meter_detail(jsonl: Path) -> tuple[int, int, bool]:
    """Newest ``turn_ended`` occupancy plus ``tokens_estimated``.

    ``(0, 0, False)`` when the log has no usable metrics.
    """
    newest, _peak = _context_meter_walk(jsonl)
    return newest


def peak_turn_context_meter(jsonl: Path) -> tuple[int, int]:
    """High-water ``prompt_tokens`` / ``context_window`` across recent turns.

    Used to restore ACP occupancy after a process restart resets the
    in-memory char total. ``(0, 0)`` when the log has no usable metrics.
    """
    _newest, peak = _context_meter_walk(jsonl)
    return peak


def latest_engine_model_for_agent(
    sessions_dir: Path,
    agent_name: str,
) -> str:
    """Most-recent NON-empty ``engine_model`` recorded for ``agent_name``
    in ``sessions.db``, or ``""`` when none is known.

    External CLI/ACP engines (claude/codex/cursor/antigravity) only learn
    their concrete model once a turn lands — it's folded into the session's
    ``engine_model`` header at TurnDone. Consumers that need the id WITHOUT
    a live session in hand (e.g. the Telegram ``/new`` banner after a
    gateway restart marked the current session stale, so ``get_session_id``
    returns ``None``) can read the last-known value here instead of
    mislabelling the agent with its bare engine name. Best-effort: any DB
    error or missing file yields ``""``.
    """
    agent_name = str(agent_name or "")
    if not agent_name:
        return ""
    db = Path(sessions_dir) / "sessions.db"
    if not db.exists():
        return ""
    from sharedagents_session.session_index import (  # local import avoids a cycle
        connect_sessions_db,
    )

    try:
        con = connect_sessions_db(db)
        try:
            row = con.execute(
                "SELECT engine_model FROM sessions_meta "
                "WHERE agent_name=? AND engine_model<>'' "
                "ORDER BY updated_at DESC LIMIT 1",
                (agent_name,),
            ).fetchone()
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return ""
    if not row:
        return ""
    return str(row[0] or "")


def tail_jsonl_rows(path: Path | str, from_row_idx: int = 0) -> Iterator[tuple[int, dict[str, Any], int]]:
    """Yield ``(row_idx, event_dict, end_byte_offset)`` for lines from ``from_row_idx``."""
    p = Path(path)
    if not p.is_file():
        return
    with p.open("rb") as f:
        row_idx = 0
        while True:
            line = f.readline()
            if not line:
                break
            if row_idx >= from_row_idx:
                space = line.find(b" ")
                if space > 0:
                    payload = line[space + 1 :].rstrip(b"\r\n")
                    try:
                        event = json.loads(payload.decode("utf-8"))
                        yield row_idx, event, f.tell()
                    except Exception:
                        pass
            row_idx += 1
