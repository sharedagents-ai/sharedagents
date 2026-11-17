"""Extracted tool-runner family: read (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path
from typing import Any

from ._utils import (
    _GREP_IGNORE_DIRS,
    _GREP_MAX_DEPTH,
    _GREP_MAX_FILE_BYTES,
    DEFAULT_READ_LIMIT,
    READ_FILE_MAX_BYTES,
    READ_MAX_BYTES_LABEL,
    READ_MAX_LINE_LENGTH,
    READ_MAX_LINE_SUFFIX,
    READ_MAX_OUTPUT_BYTES,
    ToolResult,
    _err,
    _truncate,
)

# ---- read tools ----

# File extensions that markitdown can convert to markdown. The Read tool
# auto-detects these and runs them through markitdown instead of trying
# to decode as UTF-8 (which would produce garbage for binary formats).
_OFFICE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".docx",
        ".doc",  # Word
        ".pptx",
        ".ppt",  # PowerPoint
        ".xlsx",
        ".xls",  # Excel
        ".pdf",  # PDF
    }
)


def _convert_office_to_markdown(path: Path) -> ToolResult | None:
    """Try to convert an office document to markdown via markitdown.

    Returns a ``ToolResult`` on success/failure, or ``None`` when
    markitdown is not installed (caller should fall through to the
    raw-binary read path).
    """
    try:
        from markitdown import MarkItDown as _MarkItDown
    except ImportError:
        return None
    try:
        result = _MarkItDown().convert_local(str(path))
        md = result.markdown if result else ""
    except Exception as e:  # noqa: BLE001
        return _err(f"markitdown conversion failed: {e}")
    if not md:
        return _err("markitdown returned empty content — file may be unreadable")
    body, truncated = _truncate(md)
    return ToolResult(content=body, ok=True, truncated=truncated)


def _coerce_line_arg(value: Any, name: str) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None, f"{name} must be an integer"
    if n < 1:
        return None, f"{name} must be >= 1"
    return n, None


# Binary-file detection (mirror opencode ``read.ts::isBinaryFile``): a set
# of known-binary extensions plus a non-printable-byte-ratio heuristic on a
# leading sample. Office formats are handled separately (markitdown), so they
# are intentionally NOT listed here.
_READ_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".zip",
        ".tar",
        ".gz",
        ".exe",
        ".dll",
        ".so",
        ".class",
        ".jar",
        ".war",
        ".7z",
        ".bin",
        ".dat",
        ".obj",
        ".o",
        ".a",
        ".lib",
        ".wasm",
        ".pyc",
        ".pyo",
    }
)
_READ_SAMPLE_BYTES = 4096


def _read_is_binary(path: Path, sample: bytes) -> bool:
    """True if ``path`` looks binary (opencode ``isBinaryFile`` parity)."""
    if path.suffix.lower() in _READ_BINARY_EXTENSIONS:
        return True
    if not sample:
        return False
    non_printable = 0
    for b in sample:
        if b == 0:
            return True
        if b < 9 or (13 < b < 32):
            non_printable += 1
    return (non_printable / len(sample)) > 0.3


# Raster formats ``read`` cannot show the model (the chat transports accept
# PNG, JPEG, GIF and WebP only). They keep the binary refusal, with a hint.
_READ_UNVIEWABLE_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".bmp",
        ".tif",
        ".tiff",
        ".ico",
        ".heic",
        ".heif",
        ".avif",
        ".psd",
    }
)


def _read_image_too_large(p: Path, size: int) -> ToolResult:
    from sharedagents_core.attachments import TOOL_IMAGE_MAX_BYTES

    cap_mib = TOOL_IMAGE_MAX_BYTES // (1024 * 1024)
    return _err(
        f"image too large to view: {p} is at least {size} bytes; the limit for "
        f"images is {cap_mib} MiB ({TOOL_IMAGE_MAX_BYTES} bytes). Make a "
        "smaller copy (downscale or crop, save as PNG or JPEG) and read "
        "that file instead."
    )


def _read_image(p: Path, blob: bytes, mime: str) -> ToolResult:
    """An image file: a short text description plus the image itself.

    The image rides ``ToolResult.images`` as an ``image_url`` part; the
    engine shows it to the model the way a user attachment is shown.
    """
    from sharedagents_core.attachments import (
        TOOL_IMAGE_MAX_BYTES,
        image_dimensions,
        image_url_part,
    )

    size = len(blob)
    if size > TOOL_IMAGE_MAX_BYTES:
        return _read_image_too_large(p, size)
    dims = image_dimensions(blob, mime)
    attrs = f'mime="{mime}" bytes="{size}"'
    if dims:
        attrs += f' width="{dims[0]}" height="{dims[1]}"'
    content = (
        f"<path>{p}</path>\n<type>image</type>\n<image {attrs}/>\n(The image is shown to you in the next message.)"
    )
    return ToolResult(content=content, ok=True, images=(image_url_part(blob, mime),))


def _read_miss(p: Path) -> ToolResult:
    """opencode ``ReadTool.miss``: ``File not found`` + fuzzy suggestions."""
    base = p.name.lower()
    suggestions: list[str] = []
    try:
        for name in sorted(os.listdir(p.parent)):
            nl = name.lower()
            if nl in base or base in nl:
                suggestions.append(str(p.parent / name))
            if len(suggestions) >= 3:
                break
    except OSError:
        suggestions = []
    if suggestions:
        return _err(f"File not found: {p}\n\nDid you mean one of these?\n" + "\n".join(suggestions))
    return _err(f"File not found: {p}")


def _read_directory(p: Path, offset: int, limit: int) -> ToolResult:
    """opencode ``ReadTool.list``: entries one-per-line, dirs get ``/``."""
    try:
        raw_entries = os.listdir(p)
    except OSError as e:
        return _err(f"{e}")
    entries: list[str] = []
    for name in raw_entries:
        try:
            if (p / name).is_dir():
                entries.append(name + "/")
            else:
                entries.append(name)
        except OSError:
            entries.append(name)
    entries.sort(key=str.lower)
    start = offset - 1
    sliced = entries[start : start + limit]
    truncated = start + len(sliced) < len(entries)
    tail = (
        f"\n(Showing {len(sliced)} of {len(entries)} entries. Use 'offset' "
        f"parameter to read beyond entry {offset + len(sliced)})"
        if truncated
        else f"\n({len(entries)} entries)"
    )
    output = "\n".join(
        [
            f"<path>{p}</path>",
            "<type>directory</type>",
            "<entries>",
            "\n".join(sliced),
            tail,
            "</entries>",
        ]
    )
    body, was_trunc = _truncate(output)
    return ToolResult(content=body, ok=True, truncated=was_trunc or truncated)


def run_read_file(args: dict[str, Any]) -> ToolResult:
    # OpenCode ``read`` schema: ``filePath`` + ``offset`` (1-based start) +
    # ``limit`` (lines, default 2000). We keep tolerant aliases for the
    # legacy sharedagents / Claude args (``file_path`` / ``path`` /
    # ``start_line`` / ``end_line``) so old call sites don't hard-break,
    # but the model-facing schema is opencode's camelCase.
    path = args.get("filePath") or args.get("file_path") or args.get("path")
    if not isinstance(path, str) or not path:
        return _err("filePath required")
    raw_start = args.get("offset")
    if raw_start is None:
        raw_start = args.get("start_line")
    start_line, err = _coerce_line_arg(raw_start, "offset")
    if err:
        return _err(err)
    offset = start_line if start_line is not None else 1

    raw_limit = args.get("limit")
    if raw_limit is not None:
        limit_int, err = _coerce_line_arg(raw_limit, "limit")
        if err:
            return _err(err)
        limit = limit_int or DEFAULT_READ_LIMIT
    else:
        # Legacy ``end_line`` (inclusive) → limit relative to offset.
        end_line, err = _coerce_line_arg(args.get("end_line"), "end_line")
        if err:
            return _err(err)
        if end_line is not None:
            if end_line < offset:
                return _err("end_line/limit must be >= offset")
            limit = end_line - offset + 1
        else:
            limit = DEFAULT_READ_LIMIT

    p = Path(path).expanduser()
    try:
        st = p.stat()
    except FileNotFoundError:
        return _read_miss(p)
    except OSError as e:
        return _err(f"{e}")

    # Directory read (opencode ``read`` lists a directory).
    if p.is_dir():
        return _read_directory(p, offset, limit)

    # Read the body EXACTLY ONCE (the binary sniff samples the leading bytes
    # of the same blob). Opening the path a second time would block forever on
    # a FIFO / named pipe, whose single writer has already closed.
    from sharedagents_core.attachments import TOOL_IMAGE_MAX_BYTES, sniff_image_mime

    try:
        with open(p, "rb") as fh:
            # Sniff before materialising the body, including files with no
            # extension. Keep one open stream so FIFO reads still work.
            head = fh.read(16)
            image_mime = sniff_image_mime(head)
            if image_mime and st.st_size > TOOL_IMAGE_MAX_BYTES:
                return _read_image_too_large(p, st.st_size)
            if not image_mime and st.st_size > READ_FILE_MAX_BYTES:
                return _err(
                    f"file too large: {st.st_size} bytes exceeds "
                    f"READ_FILE_MAX_BYTES={READ_FILE_MAX_BYTES}. "
                    "Use `grep` to extract a portion, or read line ranges via "
                    "`mcp__memory__get` (for memory files) / a slice tool."
                )
            # Image bytes take precedence over an office-document suffix.
            # Actual office documents retain the existing markdown conversion.
            if not image_mime and p.suffix.lower() in _OFFICE_EXTENSIONS:
                result = _convert_office_to_markdown(p)
                if result is not None:
                    return result
            cap = TOOL_IMAGE_MAX_BYTES if image_mime else READ_FILE_MAX_BYTES
            blob = head + fh.read(max(0, cap + 1 - len(head)))
    except (OSError, IsADirectoryError) as e:
        return _err(f"{e}")
    if len(blob) > READ_FILE_MAX_BYTES:
        return _err(f"file grew past READ_FILE_MAX_BYTES={READ_FILE_MAX_BYTES} during read; refusing to materialise.")
    # Images are recognised by their bytes, never by the file name. SVG is
    # XML text and is read as text below.
    if image_mime:
        return _read_image(p, blob, image_mime)
    if _read_is_binary(p, blob[:_READ_SAMPLE_BYTES]):
        if p.suffix.lower() in _READ_UNVIEWABLE_IMAGE_EXTENSIONS:
            return _err(
                f"Cannot read binary file: {p} ({p.suffix.lower()} images "
                "cannot be shown to the model; only PNG, JPEG, GIF and WebP "
                "can. Convert it to PNG and read the copy.)"
            )
        return _err(f"Cannot read binary file: {p}")
    text = blob.decode("utf-8", errors="replace")

    # Split into logical lines (drop the single trailing empty element a
    # terminating newline produces, so ``count`` matches opencode).
    all_lines = text.split("\n")
    if all_lines and all_lines[-1] == "":
        all_lines.pop()
    count = len(all_lines)

    # Out-of-range offset (opencode errors; an empty file at offset 1 is OK).
    if count < offset and not (count == 0 and offset == 1):
        return _err(f"Offset {offset} is out of range for this file ({count} lines)")

    start = offset - 1
    raw: list[str] = []
    used_bytes = 0
    cut = False
    more = False
    for line in all_lines[start:]:
        if len(raw) >= limit:
            more = True
            break
        clipped = line[:READ_MAX_LINE_LENGTH] + READ_MAX_LINE_SUFFIX if len(line) > READ_MAX_LINE_LENGTH else line
        size = len(clipped.encode("utf-8")) + (1 if raw else 0)
        if used_bytes + size > READ_MAX_OUTPUT_BYTES:
            cut = True
            more = True
            break
        raw.append(clipped)
        used_bytes += size

    numbered = "\n".join(f"{i + offset}: {ln}" for i, ln in enumerate(raw))
    last = offset + len(raw) - 1
    nxt = last + 1
    if cut:
        footer = (
            f"(Output capped at {READ_MAX_BYTES_LABEL}. Showing lines {offset}-{last}. Use offset={nxt} to continue.)"
        )
    elif more:
        footer = f"(Showing lines {offset}-{last} of {count}. Use offset={nxt} to continue.)"
    else:
        footer = f"(End of file - total {count} lines)"

    output = f"<path>{p}</path>\n<type>file</type>\n<content>\n{numbered}\n\n{footer}\n</content>"
    body, was_trunc = _truncate(output)
    # opencode read.ts:119 — fire-and-forget LSP touchFile pre-warm.
    # Trivial with the persistent daemon: best-effort socket send on a
    # daemon thread; errors swallowed; never blocks the read result.
    if p.suffix == ".py":
        try:
            from ._lsp_gate import lsp_touch_async

            lsp_touch_async(str(p.resolve()))
        except Exception:  # noqa: BLE001
            pass
    return ToolResult(content=body, ok=True, truncated=was_trunc or more or cut)


def run_list_dir(args: dict[str, Any]) -> ToolResult:
    # Accept OpenCode-style ``filePath`` as well as the registered ``path`` —
    # otherwise a model that sends only ``filePath`` silently lists cwd.
    path = args.get("path") or args.get("filePath") or args.get("file_path") or "."
    if not isinstance(path, str):
        return _err("path must be a string")
    try:
        entries = sorted(os.listdir(Path(path).expanduser()))
    except OSError as e:
        return _err(f"{e}")
    body = "\n".join(entries) if entries else "(empty)"
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)


def _glob_match(path: Path, root: Path, glob_pat: str) -> bool:
    """Match ``path`` under ``root`` against a glob pattern (supports ``**``)."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    if rel.match(glob_pat):
        return True
    # ``Path.match('**/…')`` does not match entries at the walk root (e.g.
    # ``a.py`` vs ``**/*.py``, ``data`` vs ``**/data``).
    if glob_pat.startswith("**/"):
        return rel.match(glob_pat[3:])
    return False


def run_glob(args: dict[str, Any]) -> ToolResult:
    """Find files matching a glob pattern. Returns absolute paths
    sorted by mtime (most recent first), one per line. Mirrors
    Claude Code's ``Glob`` tool.

    Uses the same bounded walk as ``run_grep`` (ignore dirs, depth cap,
    file cap, wall-clock deadline) so a recursive ``**`` from ``/`` or
    ``~`` cannot spin forever."""
    # Resolve the caps through the ``runners`` re-export so tests that do
    # ``monkeypatch.setattr(runners, "_GREP_MAX_FILES"/"_GREP_TIMEOUT_S", …)``
    # still bound this walk after the de-monolith split (local shadow keeps the
    # existing reference sites unchanged; falls back to the module copy).
    from sharedagents_agent.tools import runners as _r

    _GREP_TIMEOUT_S = getattr(_r, "_GREP_TIMEOUT_S", globals()["_GREP_TIMEOUT_S"])
    _GREP_MAX_FILES = getattr(_r, "_GREP_MAX_FILES", globals()["_GREP_MAX_FILES"])
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return _err("pattern required")
    root_raw = args.get("path")
    root = Path(root_raw).expanduser() if isinstance(root_raw, str) and root_raw else Path(".").expanduser()
    if not root.exists():
        return _err(f"path not found: {root_raw or '.'}")
    root_resolved = root.resolve()
    glob_pat = pattern.lstrip("/") if pattern.startswith("/") else pattern

    deadline = time.monotonic() + _GREP_TIMEOUT_S
    rows: list[tuple[float, str]] = []
    entries_examined = 0
    stop_reason: str | None = None
    # opencode caps glob at 100 results and appends this exact truncation
    # note when the cap is hit.
    max_output = 100
    _GLOB_TRUNC_MSG = (
        f"(Results are truncated: showing first {max_output} results. Consider using a more specific path or pattern.)"
    )

    def _maybe_add(p: Path) -> bool:
        if not _glob_match(p, root_resolved, glob_pat):
            return False
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return False
        rows.append((mtime, str(p.resolve())))
        return len(rows) >= max_output

    if root.is_file():
        _maybe_add(root_resolved)
    else:
        root_str = str(root_resolved)
        for dirpath, dirnames, filenames in os.walk(root_str):
            rel_dir = os.path.relpath(dirpath, root_str)
            depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
            if depth >= _GREP_MAX_DEPTH:
                dirnames[:] = []
            else:
                dirnames[:] = [d for d in dirnames if d not in _GREP_IGNORE_DIRS]
            if time.monotonic() >= deadline:
                stop_reason = f"(glob stopped after {_GREP_TIMEOUT_S}s, consider narrowing path)"
                break
            for dn in dirnames:
                if _maybe_add(Path(dirpath) / dn):
                    stop_reason = _GLOB_TRUNC_MSG
                    break
            if stop_reason:
                break
            for fn in filenames:
                if time.monotonic() >= deadline:
                    stop_reason = f"(glob stopped after {_GREP_TIMEOUT_S}s, consider narrowing path)"
                    break
                if entries_examined >= _GREP_MAX_FILES:
                    stop_reason = (
                        f"(glob stopped: examined {_GREP_MAX_FILES} entries, consider narrowing path or pattern)"
                    )
                    break
                entries_examined += 1
                if _maybe_add(Path(dirpath) / fn):
                    stop_reason = _GLOB_TRUNC_MSG
                    break
            if stop_reason:
                break

    rows.sort(key=lambda r: -r[0])
    if not rows:
        body = "No files found"
        if stop_reason:
            body = f"{body}\n{stop_reason}"
        return ToolResult(content=body, ok=True, truncated=False)
    body = "\n".join(p for _, p in rows)
    if stop_reason:
        # opencode separates the 100-result truncation note with a blank
        # line; the sharedagents bounded-walk stop reasons keep a single newline.
        sep = "\n\n" if stop_reason == _GLOB_TRUNC_MSG else "\n"
        body = f"{body}{sep}{stop_reason}"
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)


def run_exit_plan_mode(args: dict[str, Any]) -> ToolResult:
    """Return a completed plan for readiness-only operator review.

    This read-only runner does not change the session mode, dispatch Build,
    or authorize implementation. Any implementation requires a separate
    operator request, approval, and authority.
    """
    plan = args.get("plan") or ""
    if not isinstance(plan, str) or not plan.strip():
        return _err("plan required")
    body = (
        "Plan ready for operator review. Implementation requires a separate "
        "operator request, approval, and authority:\n\n" + plan.strip()
    )
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)


_GREP_MATCH_LIMIT = 100  # opencode caps grep at 100 matches


def _grep_with_ripgrep(
    pattern: str,
    cwd: str,
    include: str | None,
    limit: int,
    target: str = ".",
) -> list[tuple[str, int, str]] | None:
    """Run ripgrep (``rg``) and return ``(abs_path, line_no, text)`` rows.

    ``target`` is the search target relative to ``cwd`` — ``"."`` for a directory
    path, or the file's basename for a FILE path. Searching the specific file (not
    ``"."`` in its parent) is a SECURITY requirement: a file-valued ``path`` is
    validated one-file by the memory-privacy gate, so ``rg`` must not fan out to
    the parent's siblings (e.g. a grep of ``<home>/AGENTS.md`` must not read
    ``<home>/MEMORY.md`` / ``memory/``).

    Returns ``None`` when ``rg`` is unavailable or errors out (so the caller
    falls back to the pure-Python scanner). A clean "no matches" is an empty
    list, not ``None``."""
    import shutil
    import subprocess

    # See run_glob: honour monkeypatch of ``runners._GREP_TIMEOUT_S`` post-split.
    from sharedagents_agent.tools import runners as _r

    _GREP_TIMEOUT_S = getattr(_r, "_GREP_TIMEOUT_S", globals()["_GREP_TIMEOUT_S"])
    rg = shutil.which("rg")
    if not rg:
        return None
    cmd = [rg, "--line-number", "--no-heading", "--color", "never"]
    if include:
        cmd += ["--glob", include]
    cmd += ["--regexp", pattern, target]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GREP_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # rg exit 1 == no matches (not an error); >1 == real error.
    if proc.returncode not in (0, 1):
        return None
    rows: list[tuple[str, int, str]] = []
    for line in proc.stdout.splitlines():
        # ``<relpath>:<lineno>:<text>``
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        rel, lineno_s, text = parts
        try:
            lineno = int(lineno_s)
        except ValueError:
            continue
        rows.append((str((Path(cwd) / rel).resolve()), lineno, text))
        if len(rows) >= limit:
            break
    return rows


def _grep_with_python(
    pattern: str,
    root: Path,
    include: str | None,
    limit: int,
) -> tuple[list[tuple[str, int, str]] | None, str | None]:
    """Bounded pure-Python fallback for :func:`run_grep`.

    Returns ``(rows, error)``. ``error`` is a bad-regex message (rows None);
    otherwise rows is a possibly-empty list. Uses Python ``re`` — semantics
    differ from ripgrep's Rust regex on lookbehind / backrefs / Unicode
    classes (documented divergence)."""
    # See run_glob: honour monkeypatch of the ``runners._GREP_*`` caps post-split.
    from sharedagents_agent.tools import runners as _r

    _GREP_TIMEOUT_S = getattr(_r, "_GREP_TIMEOUT_S", globals()["_GREP_TIMEOUT_S"])
    _GREP_MAX_FILES = getattr(_r, "_GREP_MAX_FILES", globals()["_GREP_MAX_FILES"])
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return None, f"bad regex: {e}"
    deadline = time.monotonic() + _GREP_TIMEOUT_S
    rows: list[tuple[str, int, str]] = []
    files_examined = 0

    def _scan_one(p: Path) -> bool:
        try:
            st = p.stat()
        except OSError:
            return False
        if st.st_size > _GREP_MAX_FILE_BYTES:
            return False
        try:
            with open(p, "rb") as fh:
                blob = fh.read()
        except OSError:
            return False
        if b"\x00" in blob[:8192]:
            return False
        text = blob.decode("utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                rows.append((str(p.resolve()), i, line))
                if len(rows) >= limit:
                    return True
        return False

    if root.is_file():
        _scan_one(root)
    else:
        root_str = str(root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root_str)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= _GREP_MAX_DEPTH:
                dirnames[:] = []
            else:
                dirnames[:] = [d for d in dirnames if d not in _GREP_IGNORE_DIRS]
            for fn in filenames:
                if include and not fnmatch.fnmatch(fn, include):
                    continue
                if time.monotonic() >= deadline:
                    return rows, None
                if files_examined >= _GREP_MAX_FILES:
                    return rows, None
                files_examined += 1
                if _scan_one(Path(dirpath) / fn):
                    return rows, None
    return rows, None


def run_grep(args: dict[str, Any]) -> ToolResult:
    # OpenCode ``grep`` schema: ``pattern`` + ``path`` + ``include``. Uses
    # ripgrep when available (Rust regex, .gitignore-aware), falling back to
    # a bounded pure-Python scanner. Output is opencode's grouped shape:
    #   Found N matches[ (more matches available)]
    #
    #   <abs-path>:
    #     Line n: <text>
    # Tolerant of the legacy ``glob`` alias for ``include`` and Zed's
    # ``query`` alias for ``pattern``.
    pattern = args.get("pattern") or args.get("query")
    if not isinstance(pattern, str) or not pattern:
        return _err("pattern required")
    path = args.get("path") or "."
    include_raw = args.get("include") or args.get("glob")
    include = include_raw if isinstance(include_raw, str) and include_raw else None

    root = Path(path).expanduser()
    if not root.exists():
        return _err(f"path not found: {path}")
    cwd = str(root) if root.is_dir() else str(root.parent)

    limit = _GREP_MATCH_LIMIT
    # For a FILE path, search only that file (its basename relative to cwd=parent),
    # never "." in the parent — else rg would expose sibling files the memory
    # privacy gate never cleared.
    _rg_target = "." if root.is_dir() else root.name
    rows = _grep_with_ripgrep(pattern, cwd, include, limit, target=_rg_target)
    if rows is None:
        rows, err = _grep_with_python(pattern, root, include, limit)
        if err is not None:
            return _err(err)
    assert rows is not None

    if not rows:
        return ToolResult(content="No files found", ok=True)

    total = len(rows)
    truncated = total >= limit
    header = f"Found {total} matches" + (" (more matches available)" if truncated else "")
    out_lines = [header]
    current = ""
    for abs_path, lineno, text in rows:
        if current != abs_path:
            if current != "":
                out_lines.append("")
            current = abs_path
            out_lines.append(f"{abs_path}:")
        out_lines.append(f"  Line {lineno}: {text}")
    if truncated:
        out_lines.append("")
        out_lines.append("(Results truncated. Consider using a more specific path or pattern.)")
    body, was_trunc = _truncate("\n".join(out_lines))
    return ToolResult(content=body, ok=True, truncated=was_trunc)


__all__ = [
    "_OFFICE_EXTENSIONS",
    "_convert_office_to_markdown",
    "_coerce_line_arg",
    "_READ_BINARY_EXTENSIONS",
    "_READ_SAMPLE_BYTES",
    "_read_is_binary",
    "_read_miss",
    "_read_directory",
    "run_read_file",
    "run_list_dir",
    "_glob_match",
    "run_glob",
    "run_exit_plan_mode",
    "_GREP_MATCH_LIMIT",
    "_grep_with_ripgrep",
    "_grep_with_python",
    "run_grep",
]
