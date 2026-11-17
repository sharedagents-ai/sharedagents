"""Extracted tool-runner family: write (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sharedagents_core.cwdutil import safe_getcwd

from ._lsp_gate import lsp_errors_block
from ._utils import (
    ToolResult,
    _err,
)

# ---- write tools ----


def run_write_file(args: dict[str, Any]) -> ToolResult:
    # OpenCode ``write`` schema: ``content`` + ``filePath``. Tolerant of the
    # legacy ``file_path`` / ``path`` aliases so old callers keep working.
    path = args.get("filePath") or args.get("file_path") or args.get("path")
    content = args.get("content")
    if not isinstance(path, str) or not path:
        return _err("filePath required")
    if not isinstance(content, str):
        return _err("content must be a string")
    p = Path(path).expanduser()
    # Preserve/derive BOM exactly as opencode does: keep a BOM the existing
    # file already had, or one the new content carries (``source.bom ||
    # next.bom``).
    existing_bom = False
    if p.exists():
        try:
            existing_bom, _ = _ap_bom_split(p.read_bytes().decode("utf-8", errors="replace"))
        except OSError:
            existing_bom = False
    next_bom, next_text = _ap_bom_split(content)
    desired_bom = existing_bom or next_bom
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_ap_bom_join(next_text, desired_bom).encode("utf-8"))
    except OSError as e:
        return _err(f"{e}")
    feedback = lsp_errors_block([str(p)])
    return ToolResult(content="Wrote file successfully." + feedback, ok=True)


# ---- edit: OpenCode multi-strategy fuzzy replacer cascade ----
#
# Faithful port of opencode ``tool/edit.ts`` (branch ``dev``). ``edit`` is NOT
# a naive substring replace: it runs a cascade of nine replacer strategies in
# order until one yields a candidate span that occurs uniquely (or, with
# ``replaceAll``, at all) in the file. Sourced (as opencode notes) from cline
# and gemini-cli diff-apply implementations. A disproportionate-match guard
# refuses a span much larger than ``oldString``. Line-endings + BOM are
# preserved. Empty ``oldString`` creates a new file.

_EDIT_SINGLE_SIMILARITY = 0.65
_EDIT_MULTI_SIMILARITY = 0.65


def _levenshtein(a: str, b: str) -> int:
    if a == "" or b == "":
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def _edit_simple(content: str, find: str):
    yield find


def _edit_line_trimmed(content: str, find: str):
    original = content.split("\n")
    search = find.split("\n")
    if search and search[-1] == "":
        search.pop()
    if not search:
        return
    for i in range(0, len(original) - len(search) + 1):
        if all(original[i + j].strip() == search[j].strip() for j in range(len(search))):
            start = sum(len(original[k]) + 1 for k in range(i))
            end = start
            for k in range(len(search)):
                end += len(original[i + k])
                if k < len(search) - 1:
                    end += 1
            yield content[start:end]


def _edit_block_anchor(content: str, find: str):
    original = content.split("\n")
    search = find.split("\n")
    if len(search) < 3:
        return
    if search and search[-1] == "":
        search.pop()
    first = search[0].strip()
    last = search[-1].strip()
    block_size = len(search)
    max_delta = max(1, block_size // 4)
    candidates: list[tuple[int, int]] = []
    for i in range(len(original)):
        if original[i].strip() != first:
            continue
        for j in range(i + 2, len(original)):
            if original[j].strip() == last:
                actual = j - i + 1
                if abs(actual - block_size) <= max_delta:
                    candidates.append((i, j))
                break
    if not candidates:
        return

    def _span(start_line: int, end_line: int) -> str:
        s = sum(len(original[k]) + 1 for k in range(start_line))
        e = s
        for k in range(start_line, end_line + 1):
            e += len(original[k])
            if k < end_line:
                e += 1
        return content[s:e]

    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        actual = end_line - start_line + 1
        lines_to_check = min(block_size - 2, actual - 2)
        if lines_to_check > 0:
            similarity = 0.0
            j = 1
            while j < block_size - 1 and j < actual - 1:
                o = original[start_line + j].strip()
                s = search[j].strip()
                max_len = max(len(o), len(s))
                if max_len:
                    similarity += (1 - _levenshtein(o, s) / max_len) / lines_to_check
                    if similarity >= _EDIT_SINGLE_SIMILARITY:
                        break
                j += 1
        else:
            similarity = 1.0
        if similarity >= _EDIT_SINGLE_SIMILARITY:
            yield _span(start_line, end_line)
        return

    best = None
    max_sim = -1.0
    for start_line, end_line in candidates:
        actual = end_line - start_line + 1
        lines_to_check = min(block_size - 2, actual - 2)
        if lines_to_check > 0:
            similarity = 0.0
            j = 1
            while j < block_size - 1 and j < actual - 1:
                o = original[start_line + j].strip()
                s = search[j].strip()
                max_len = max(len(o), len(s))
                if max_len:
                    similarity += 1 - _levenshtein(o, s) / max_len
                j += 1
            similarity /= lines_to_check
        else:
            similarity = 1.0
        if similarity > max_sim:
            max_sim = similarity
            best = (start_line, end_line)
    if max_sim >= _EDIT_MULTI_SIMILARITY and best is not None:
        yield _span(*best)


def _edit_whitespace_normalized(content: str, find: str):
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    normalized_find = norm(find)
    lines = content.split("\n")
    for line in lines:
        if norm(line) == normalized_find:
            yield line
        elif normalized_find in norm(line):
            words = [w for w in re.split(r"\s+", find.strip()) if w]
            if words:
                pattern = r"\s+".join(re.escape(w) for w in words)
                try:
                    m = re.search(pattern, line)
                except re.error:
                    m = None
                if m:
                    yield m.group(0)
    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(0, len(lines) - len(find_lines) + 1):
            block = lines[i : i + len(find_lines)]
            if norm("\n".join(block)) == normalized_find:
                yield "\n".join(block)


def _edit_indentation_flexible(content: str, find: str):
    def remove_indent(text: str) -> str:
        lines = text.split("\n")
        non_empty = [ln for ln in lines if ln.strip()]
        if not non_empty:
            return text
        min_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
        return "\n".join(ln if not ln.strip() else ln[min_indent:] for ln in lines)

    normalized_find = remove_indent(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(0, len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i : i + len(find_lines)])
        if remove_indent(block) == normalized_find:
            yield block


def _edit_unescape(s: str) -> str:
    return re.sub(
        r"\\(n|t|r|'|\"|`|\\|\n|\$)",
        lambda m: {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "'": "'",
            '"': '"',
            "`": "`",
            "\\": "\\",
            "\n": "\n",
            "$": "$",
        }.get(m.group(1), m.group(0)),
        s,
    )


def _edit_escape_normalized(content: str, find: str):
    unescaped_find = _edit_unescape(find)
    if unescaped_find in content:
        yield unescaped_find
    lines = content.split("\n")
    find_lines = unescaped_find.split("\n")
    for i in range(0, len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if _edit_unescape(block) == unescaped_find:
            yield block


def _edit_trimmed_boundary(content: str, find: str):
    trimmed = find.strip()
    if trimmed == find:
        return
    if trimmed in content:
        yield trimmed
    lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(0, len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if block.strip() == trimmed:
            yield block


def _edit_context_aware(content: str, find: str):
    find_lines = find.split("\n")
    if len(find_lines) < 3:
        return
    if find_lines and find_lines[-1] == "":
        find_lines.pop()
    content_lines = content.split("\n")
    first = find_lines[0].strip()
    last = find_lines[-1].strip()
    for i in range(len(content_lines)):
        if content_lines[i].strip() != first:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last:
                block_lines = content_lines[i : j + 1]
                block = "\n".join(block_lines)
                if len(block_lines) == len(find_lines):
                    matching = 0
                    total = 0
                    for k in range(1, len(block_lines) - 1):
                        bl = block_lines[k].strip()
                        fl = find_lines[k].strip()
                        if bl or fl:
                            total += 1
                            if bl == fl:
                                matching += 1
                    if total == 0 or matching / total >= 0.5:
                        yield block
                        return
                break


def _edit_multi_occurrence(content: str, find: str):
    start = 0
    while True:
        idx = content.find(find, start)
        if idx == -1:
            break
        yield find
        start = idx + len(find)


_EDIT_REPLACERS = (
    _edit_simple,
    _edit_line_trimmed,
    _edit_block_anchor,
    _edit_whitespace_normalized,
    _edit_indentation_flexible,
    _edit_escape_normalized,
    _edit_trimmed_boundary,
    _edit_context_aware,
    _edit_multi_occurrence,
)


def _edit_disproportionate(search: str, old_string: str) -> bool:
    old_lines = len(old_string.split("\n"))
    search_lines = len(search.split("\n"))
    if search_lines >= max(old_lines + 3, old_lines * 2):
        return True
    if old_lines == 1:
        return False
    return len(search.strip()) > max(len(old_string.strip()) + 500, len(old_string.strip()) * 4)


class _EditError(Exception):
    pass


def _edit_replace(content: str, old_string: str, new_string: str, replace_all: bool) -> str:
    """opencode ``edit.ts::replace`` — the 9-strategy cascade + guards."""
    if old_string == new_string:
        raise _EditError("No changes to apply: oldString and newString are identical.")
    if old_string == "":
        raise _EditError(
            "oldString cannot be empty when editing an existing file. "
            "Provide the exact text to replace, or use write for an "
            "intentional full-file replacement."
        )
    not_found = True
    for replacer in _EDIT_REPLACERS:
        for search in replacer(content, old_string):
            idx = content.find(search)
            if idx == -1:
                continue
            not_found = False
            if _edit_disproportionate(search, old_string):
                raise _EditError(
                    "Refusing replacement because the matched span is much "
                    "larger than oldString. Re-read the file and provide the "
                    "full exact oldString for the intended replacement."
                )
            if replace_all:
                return content.replace(search, new_string)
            if idx != content.rfind(search):
                continue
            return content[:idx] + new_string + content[idx + len(search) :]
    if not_found:
        raise _EditError(
            "Could not find oldString in the file. It must match exactly, "
            "including whitespace, indentation, and line endings."
        )
    raise _EditError("Found multiple matches for oldString. Provide more surrounding context to make the match unique.")


def run_edit_file(args: dict[str, Any]) -> ToolResult:
    # OpenCode ``edit`` schema: ``filePath / oldString / newString /
    # replaceAll``. Tolerant of legacy sharedagents / Claude args
    # (``file_path / old_string / new_string / replace_all`` and
    # ``path / old / new``) so old callers keep working, but the
    # model-facing schema is opencode's camelCase.
    path = args.get("filePath") or args.get("file_path") or args.get("path")

    def _pick(*keys: str) -> Any:
        for k in keys:
            v = args.get(k)
            if v is not None:
                return v
        return None

    old = _pick("oldString", "old_string", "old")
    new = _pick("newString", "new_string", "new")
    replace_all = bool(args.get("replaceAll", args.get("replace_all", False)))
    if not isinstance(path, str) or not path:
        return _err("filePath required")
    if old is None:
        old = ""
    if not isinstance(old, str):
        return _err("oldString must be a string")
    if not isinstance(new, str):
        return _err("newString must be a string")
    if old == new:
        return _err("No changes to apply: oldString and newString are identical.")

    p = Path(path).expanduser()

    # Empty oldString → create a new file from newString (opencode).
    if old == "":
        if p.exists():
            return _err(
                "oldString cannot be empty when editing an existing file. "
                "Provide the exact text to replace, or use write for an "
                "intentional full-file replacement."
            )
        next_bom, next_text = _ap_bom_split(new)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_ap_bom_join(next_text, next_bom).encode("utf-8"))
        except OSError as e:
            return _err(f"{e}")
        return ToolResult(
            content="Edit applied successfully." + lsp_errors_block([str(p)]),
            ok=True,
        )

    try:
        original_text = p.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return _err(f"File {p} not found")
    except IsADirectoryError:
        return _err(f"Path is a directory, not a file: {p}")
    except OSError as e:
        return _err(f"{e}")

    # Preserve BOM + original line endings exactly as opencode does.
    bom, content_old = _ap_bom_split(original_text)
    ending = "\r\n" if "\r\n" in content_old else "\n"
    old_conv = old.replace("\r\n", "\n")
    new_conv = new.replace("\r\n", "\n")
    if ending == "\r\n":
        old_conv = old_conv.replace("\n", "\r\n")
        new_conv = new_conv.replace("\n", "\r\n")

    try:
        result_text = _edit_replace(content_old, old_conv, new_conv, replace_all)
    except _EditError as e:
        return _err(str(e))

    next_bom, next_text = _ap_bom_split(result_text)
    desired_bom = bom or next_bom
    try:
        p.write_bytes(_ap_bom_join(next_text, desired_bom).encode("utf-8"))
    except OSError as e:
        return _err(f"{e}")
    feedback = lsp_errors_block([str(p)])
    return ToolResult(content="Edit applied successfully." + feedback, ok=True)


def run_update_file(args: dict[str, Any]) -> ToolResult:
    """``Update`` — Claude Code's canonical file-edit tool.

    Identical contract to ``Edit`` (``old_string`` → ``new_string``);
    the name matches the "Update" label Claude Code shows for file
    edits. The sharedagents TUI renders an ``Update`` call as a colored
    code-diff card — but that diff is computed *client-side* from the
    call's arguments (see ``UpdateMessage`` in ``sharedagents_tui.app``),
    so the ``ToolResult.content`` here stays a one-line summary. That
    keeps the model's context lean AND means the Telegram frontend —
    which only ever echoes this summary, never the arguments — shows
    no code diff (TUI-only widget, by construction).
    """
    res = run_edit_file(args)
    if res.ok and res.content.startswith("edited "):
        return ToolResult(
            content="updated " + res.content[len("edited ") :],
            ok=res.ok,
            truncated=res.truncated,
        )
    return res


# ---- apply_patch (OpenCode-compatible multi-file patch envelope) ----
#
# Faithful Python port of opencode's ``tool/apply_patch.ts`` +
# ``patch/index.ts`` (branch ``dev``). The grammar is a stripped-down,
# file-oriented diff envelope:
#
#     *** Begin Patch
#     *** Add File: <path>        (each following ``+`` line = new contents)
#     *** Delete File: <path>     (nothing follows)
#     *** Update File: <path>     (optional ``*** Move to: <newpath>`` rename,
#     @@ <context>                 then ``@@`` hunk headers with ` `/`+`/`-` rows)
#     *** End Patch
#
# All Add/Delete/Update(+Move) operations across MULTIPLE files apply in ONE
# call. The runner is ATOMIC by construction: every change is parsed, the
# target files are read, and the new contents are derived UP-FRONT (phase 1)
# before a single byte is written (phase 2). A malformed patch, a missing
# update/delete target, or a hunk whose context/old-lines don't match aborts
# in phase 1 with a clear error and touches nothing on disk. BOM is preserved
# exactly as opencode does (``util/bom.ts``); the multi-pass fuzzy line seek
# (exact -> rstrip -> trim -> unicode-normalized, with an EOF anchor) mirrors
# opencode's ``seekSequence``. Path resolution + the write permission gate are
# sharedagents's own: this tool is registered ``ToolKind.WRITE`` so every call goes
# through the same PermissionEvaluator floor as ``Write``/``Edit``, and it
# resolves paths with ``Path(...).expanduser()`` like ``run_write_file``.

_AP_BOM = "﻿"


class _ApPatchError(Exception):
    """Parse/apply failure in an apply_patch envelope (aborts before writes)."""


def _ap_bom_split(text: str) -> tuple[bool, str]:
    """Mirror opencode ``util/bom.split``: strip a leading BOM, report presence."""
    if text[:1] == _AP_BOM:
        return True, text[1:]
    return False, text


def _ap_bom_join(text: str, bom: bool) -> str:
    """Mirror opencode ``util/bom.join``: re-attach a BOM iff ``bom`` is set."""
    _, stripped = _ap_bom_split(text)
    return (_AP_BOM + stripped) if bom else stripped


def _ap_strip_heredoc(inp: str) -> str:
    """Mirror opencode ``stripHeredoc``: unwrap a ``cat <<'EOF' … EOF`` wrapper."""
    m = re.match(r"^(?:cat\s+)?<<['\"]?(\w+)['\"]?\s*\n([\s\S]*?)\n\1\s*$", inp)
    return m.group(2) if m else inp


def _ap_parse_header(lines: list[str], i: int) -> tuple[str, str | None, int] | None:
    """Mirror opencode ``parsePatchHeader``: (file_path, move_path, next_idx)."""
    line = lines[i]
    if line.startswith("*** Add File:"):
        fp = line[len("*** Add File:") :].strip()
        return (fp, None, i + 1) if fp else None
    if line.startswith("*** Delete File:"):
        fp = line[len("*** Delete File:") :].strip()
        return (fp, None, i + 1) if fp else None
    if line.startswith("*** Update File:"):
        fp = line[len("*** Update File:") :].strip()
        move: str | None = None
        nxt = i + 1
        if nxt < len(lines) and lines[nxt].startswith("*** Move to:"):
            move = lines[nxt][len("*** Move to:") :].strip()
            nxt += 1
        return (fp, move, nxt) if fp else None
    return None


def _ap_parse_update_chunks(lines: list[str], start: int) -> tuple[list[dict[str, Any]], int]:
    """Mirror opencode ``parseUpdateFileChunks``: ``@@``-delimited hunks."""
    chunks: list[dict[str, Any]] = []
    i = start
    while i < len(lines) and not lines[i].startswith("***"):
        if lines[i].startswith("@@"):
            context = lines[i][2:].strip()
            i += 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            is_eof = False
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("***"):
                cl = lines[i]
                if cl == "*** End of File":
                    is_eof = True
                    i += 1
                    break
                if cl.startswith(" "):
                    c = cl[1:]
                    old_lines.append(c)
                    new_lines.append(c)
                elif cl.startswith("-"):
                    old_lines.append(cl[1:])
                elif cl.startswith("+"):
                    new_lines.append(cl[1:])
                i += 1
            chunks.append(
                {
                    "old_lines": old_lines,
                    "new_lines": new_lines,
                    "change_context": context or None,
                    "is_end_of_file": is_eof or None,
                }
            )
        else:
            i += 1
    return chunks, i


def _ap_parse_add_content(lines: list[str], start: int) -> tuple[str, int]:
    """Mirror opencode ``parseAddFileContent``: join ``+`` lines, drop last \\n."""
    content = ""
    i = start
    while i < len(lines) and not lines[i].startswith("***"):
        if lines[i].startswith("+"):
            content += lines[i][1:] + "\n"
        i += 1
    if content.endswith("\n"):
        content = content[:-1]
    return content, i


def _ap_parse_patch(patch_text: str) -> list[dict[str, Any]]:
    """Mirror opencode ``parsePatch``: envelope -> list of typed hunks."""
    cleaned = _ap_strip_heredoc(patch_text.strip())
    lines = cleaned.split("\n")
    hunks: list[dict[str, Any]] = []

    begin_marker = "*** Begin Patch"
    end_marker = "*** End Patch"
    begin_idx = next((k for k, ln in enumerate(lines) if ln.strip() == begin_marker), -1)
    end_idx = next((k for k, ln in enumerate(lines) if ln.strip() == end_marker), -1)
    if begin_idx == -1 or end_idx == -1 or begin_idx >= end_idx:
        raise _ApPatchError("Invalid patch format: missing Begin/End markers")

    i = begin_idx + 1
    while i < end_idx:
        header = _ap_parse_header(lines, i)
        if not header:
            i += 1
            continue
        fp, move, nxt = header
        if lines[i].startswith("*** Add File:"):
            content, nxt2 = _ap_parse_add_content(lines, nxt)
            hunks.append({"type": "add", "path": fp, "contents": content})
            i = nxt2
        elif lines[i].startswith("*** Delete File:"):
            hunks.append({"type": "delete", "path": fp})
            i = nxt
        elif lines[i].startswith("*** Update File:"):
            chunks, nxt2 = _ap_parse_update_chunks(lines, nxt)
            hunks.append(
                {
                    "type": "update",
                    "path": fp,
                    "move_path": move,
                    "chunks": chunks,
                }
            )
            i = nxt2
        else:
            i += 1
    return hunks


def _ap_normalize_unicode(s: str) -> str:
    """Mirror opencode ``normalizeUnicode``: smart punctuation -> ASCII."""
    s = re.sub(r"[‘’‚‛]", "'", s)
    s = re.sub(r"[“”„‟]", '"', s)
    s = re.sub(r"[‐‑‒–—―]", "-", s)
    s = s.replace("…", "...")
    s = s.replace(" ", " ")
    return s


def _ap_try_match(
    lines: list[str],
    pattern: list[str],
    start_index: int,
    compare: Callable[[str, str], bool],
    eof: bool,
) -> int:
    """Mirror opencode ``tryMatch``: EOF-anchored then forward window match."""
    plen = len(pattern)
    if eof:
        from_end = len(lines) - plen
        if from_end >= start_index and all(compare(lines[from_end + j], pattern[j]) for j in range(plen)):
            return from_end
    for i in range(start_index, len(lines) - plen + 1):
        if all(compare(lines[i + j], pattern[j]) for j in range(plen)):
            return i
    return -1


def _ap_seek_sequence(
    lines: list[str],
    pattern: list[str],
    start_index: int,
    eof: bool = False,
) -> int:
    """Mirror opencode ``seekSequence``: exact -> rstrip -> trim -> normalized."""
    if len(pattern) == 0:
        return -1
    r = _ap_try_match(lines, pattern, start_index, lambda a, b: a == b, eof)
    if r != -1:
        return r
    r = _ap_try_match(lines, pattern, start_index, lambda a, b: a.rstrip() == b.rstrip(), eof)
    if r != -1:
        return r
    r = _ap_try_match(lines, pattern, start_index, lambda a, b: a.strip() == b.strip(), eof)
    if r != -1:
        return r
    return _ap_try_match(
        lines,
        pattern,
        start_index,
        lambda a, b: _ap_normalize_unicode(a.strip()) == _ap_normalize_unicode(b.strip()),
        eof,
    )


def _ap_compute_replacements(
    original_lines: list[str],
    file_path: str,
    chunks: list[dict[str, Any]],
) -> list[tuple[int, int, list[str]]]:
    """Mirror opencode ``computeReplacements``: (start, old_len, new_lines)[]."""
    replacements: list[tuple[int, int, list[str]]] = []
    line_index = 0
    for chunk in chunks:
        ctx = chunk["change_context"]
        if ctx:
            ctx_idx = _ap_seek_sequence(original_lines, [ctx], line_index)
            if ctx_idx == -1:
                raise _ApPatchError(f"Failed to find context '{ctx}' in {file_path}")
            line_index = ctx_idx + 1

        old_lines = chunk["old_lines"]
        if len(old_lines) == 0:
            if len(original_lines) > 0 and original_lines[-1] == "":
                insertion_idx = len(original_lines) - 1
            else:
                insertion_idx = len(original_lines)
            replacements.append((insertion_idx, 0, list(chunk["new_lines"])))
            continue

        pattern = list(old_lines)
        new_slice = list(chunk["new_lines"])
        eof = bool(chunk["is_end_of_file"])
        found = _ap_seek_sequence(original_lines, pattern, line_index, eof)
        if found == -1 and len(pattern) > 0 and pattern[-1] == "":
            pattern = pattern[:-1]
            if len(new_slice) > 0 and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = _ap_seek_sequence(original_lines, pattern, line_index, eof)

        if found != -1:
            replacements.append((found, len(pattern), new_slice))
            line_index = found + len(pattern)
        else:
            raise _ApPatchError("Failed to find expected lines in " + file_path + ":\n" + "\n".join(chunk["old_lines"]))

    replacements.sort(key=lambda r: r[0])
    return replacements


def _ap_apply_replacements(
    lines: list[str],
    replacements: list[tuple[int, int, list[str]]],
) -> list[str]:
    """Mirror opencode ``applyReplacements``: splice in reverse order."""
    result = list(lines)
    for i in range(len(replacements) - 1, -1, -1):
        start_idx, old_len, new_seg = replacements[i]
        del result[start_idx : start_idx + old_len]
        for j in range(len(new_seg)):
            result.insert(start_idx + j, new_seg[j])
    return result


def _ap_derive_new_contents(
    file_path: str,
    chunks: list[dict[str, Any]],
    original_text: str,
) -> tuple[str, bool]:
    """Mirror opencode ``deriveNewContentsFromChunks``: (new_text, bom)."""
    orig_bom, orig_text = _ap_bom_split(original_text)
    original_lines = orig_text.split("\n")
    if len(original_lines) > 0 and original_lines[-1] == "":
        original_lines.pop()
    replacements = _ap_compute_replacements(original_lines, file_path, chunks)
    new_lines = _ap_apply_replacements(original_lines, replacements)
    if len(new_lines) == 0 or new_lines[-1] != "":
        new_lines.append("")
    next_bom, next_text = _ap_bom_split("\n".join(new_lines))
    return next_text, (orig_bom or next_bom)


def _apply_patch_target_paths(patch_text: str) -> list[str]:
    # Update source + Move destination) so the dispatch write-scope + memory-privacy
    # gates can enforce containment on each — the patch's paths are inside patchText,
    # not a single tool arg.
    import re as _re

    out: list[str] = []
    for _m in _re.finditer(
        r"^\*\*\*\s+(?:Add|Delete|Update)\s+File:\s*(.+?)\s*$",
        patch_text or "",
        _re.M,
    ):
        out.append(_m.group(1))
    for _m in _re.finditer(r"^\*\*\*\s+Move to:\s*(.+?)\s*$", patch_text or "", _re.M):
        out.append(_m.group(1))
    return out


def run_apply_patch(args: dict[str, Any]) -> ToolResult:
    """``apply_patch`` — apply an OpenCode-style multi-file patch envelope.

    Single ``patchText`` param. Add/Delete/Update(+Move) file ops apply
    atomically: all changes are parsed, target files read, and new contents
    derived BEFORE any write, so a bad patch makes NO on-disk changes. BOM is
    preserved. Routes through the WRITE permission gate (registered
    ``ToolKind.WRITE``) exactly like ``Write``/``Edit`` — this runner is the
    child side of that gate and does not bypass it.

    Port of opencode ``tool/apply_patch.ts`` + ``patch/index.ts``.
    """
    patch_text = args.get("patchText")
    if patch_text is None:
        patch_text = args.get("patch_text")  # tolerant alias
    if not isinstance(patch_text, str) or not patch_text:
        return _err("patchText is required")

    # 1. Parse the envelope into hunks.
    try:
        hunks = _ap_parse_patch(patch_text)
    except _ApPatchError as e:
        return _err(f"apply_patch verification failed: {e}")
    except Exception as e:  # defensive: any parse crash -> verification error
        return _err(f"apply_patch verification failed: {e}")

    if len(hunks) == 0:
        normalized = patch_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if normalized == "*** Begin Patch\n*** End Patch":
            return _err("patch rejected: empty patch")
        return _err("apply_patch verification failed: no hunks found")

    # 2. Compute every change WITHOUT touching disk (atomic pre-flight). Any
    #    failure here returns before a single byte is written.
    # Resolve relative targets against the WORKSPACE base — the SAME base the
    # dispatch write-scope + privacy gates validate against — so the bytes land
    # exactly where the gates checked, not under this subprocess's inherited cwd
    # (acp_bridge leaves it at the gateway cwd).
    # REALPATH the base so a symlinked workspace is resolved ONCE, consistently
    # with the guard's ``base_real``: otherwise ``Delete "."`` would unlink the
    # workspace symlink itself and a later ``Add`` would recreate it as a real
    # directory outside the validated real workspace.
    _ws_base = os.path.realpath(os.environ.get("SHAREDAGENTS_WORKSPACE_CWD") or safe_getcwd())

    def _resolve_target(p: str) -> Path:
        pe = Path(p).expanduser()
        return pe if pe.is_absolute() else (Path(_ws_base) / pe)

    planned: list[dict[str, Any]] = []
    for hunk in hunks:
        htype = hunk["type"]
        fp = _resolve_target(hunk["path"])
        if htype == "add":
            contents = hunk["contents"]
            new_content = contents if (len(contents) == 0 or contents.endswith("\n")) else contents + "\n"
            bom, text = _ap_bom_split(new_content)
            planned.append(
                {
                    "op": "add",
                    "path": fp,
                    "text": text,
                    "bom": bom,
                    "display": hunk["path"],
                }
            )
        elif htype == "update":
            if not fp.exists() or fp.is_dir():
                return _err(f"apply_patch verification failed: Failed to read file to update: {fp}")
            try:
                # Raw bytes decode (NOT read_text) so CRLF is preserved
                # exactly like opencode's Bom.readFile — read_text applies
                # universal-newline translation (\r\n -> \n) which would
                # silently rewrite every line ending on the file.
                original_text = fp.read_bytes().decode("utf-8")
            except OSError as e:
                return _err(f"apply_patch verification failed: Failed to read file to update: {fp}: {e}")
            try:
                new_text, bom = _ap_derive_new_contents(str(fp), hunk["chunks"], original_text)
            except _ApPatchError as e:
                return _err(f"apply_patch verification failed: {e}")
            move_raw = hunk.get("move_path")
            move_path = _resolve_target(move_raw) if move_raw else None
            planned.append(
                {
                    "op": "move" if move_path else "update",
                    "path": fp,
                    "move_path": move_path,
                    "text": new_text,
                    "bom": bom,
                    "display": move_raw if move_raw else hunk["path"],
                }
            )
        elif htype == "delete":
            if not fp.exists():
                return _err(f"apply_patch verification failed: Failed to read file to delete: {fp}")
            planned.append({"op": "delete", "path": fp, "display": hunk["path"]})

    # 2b. Reject CONFLICTING operations that touch the same path. Each update
    #     derives from the ORIGINAL file content, so two ops on one file would
    #     silently overwrite one another; two deletes / a delete+update would
    #     leave a NON-ATOMIC partial apply (first op writes, second fails). One
    #     op per file per patch — validate before any byte is written.
    _seen_paths: set[str] = set()
    for p in planned:
        for _pk in (p.get("path"), p.get("move_path")):
            if _pk is None:
                continue
            # Normalize to a canonical absolute realpath so equivalent spellings
            # (relative vs absolute, ``dir/../file``, symlinks) collapse to one key
            # and can't slip a second op on the same file past the dedup.
            _rk = os.path.realpath(str(_pk))
            if _rk in _seen_paths:
                return _err(
                    "apply_patch verification failed: multiple operations target "
                    f"the same path ({_pk}) — one op per file per patch."
                )
            _seen_paths.add(_rk)

    # 3. Apply. Everything above is pre-validated, so only raw OS errors are
    #    possible now (opencode likewise does not roll back a mid-write OS
    #    failure). Parse/verification failures never reach this point.
    try:
        for p in planned:
            op = p["op"]
            # write_bytes (NOT write_text) so no newline translation happens
            # on write either — the derived content ships byte-for-byte,
            # matching opencode's Bom.join + writeWithDirs.
            if op in ("add", "update"):
                p["path"].parent.mkdir(parents=True, exist_ok=True)
                p["path"].write_bytes(_ap_bom_join(p["text"], p["bom"]).encode("utf-8"))
            elif op == "move":
                p["move_path"].parent.mkdir(parents=True, exist_ok=True)
                p["move_path"].write_bytes(_ap_bom_join(p["text"], p["bom"]).encode("utf-8"))
                p["path"].unlink()
            elif op == "delete":
                p["path"].unlink()
    except OSError as e:
        return _err(f"apply_patch failed while writing: {e}")

    # 4. Success summary — opencode's shape: A/D/M <path> per file.
    summary_lines: list[str] = []
    for p in planned:
        if p["op"] == "add":
            summary_lines.append(f"A {p['display']}")
        elif p["op"] == "delete":
            summary_lines.append(f"D {p['display']}")
        else:  # update / move
            summary_lines.append(f"M {p['display']}")
    output = "Success. Updated the following files:\n" + "\n".join(summary_lines)
    # opencode apply_patch.ts parity: per changed file, Deletes skipped.
    _written = [str(p.get("move_path") or p["path"]) for p in planned if p["op"] != "delete"]
    output += lsp_errors_block(_written)
    return ToolResult(content=output, ok=True)


__all__ = [
    "run_write_file",
    "_EDIT_SINGLE_SIMILARITY",
    "_EDIT_MULTI_SIMILARITY",
    "_levenshtein",
    "_edit_simple",
    "_edit_line_trimmed",
    "_edit_block_anchor",
    "_edit_whitespace_normalized",
    "_edit_indentation_flexible",
    "_edit_unescape",
    "_edit_escape_normalized",
    "_edit_trimmed_boundary",
    "_edit_context_aware",
    "_edit_multi_occurrence",
    "_EDIT_REPLACERS",
    "_edit_disproportionate",
    "_EditError",
    "_edit_replace",
    "run_edit_file",
    "run_update_file",
    "_AP_BOM",
    "_ApPatchError",
    "_ap_bom_split",
    "_ap_bom_join",
    "_ap_strip_heredoc",
    "_ap_parse_header",
    "_ap_parse_update_chunks",
    "_ap_parse_add_content",
    "_ap_parse_patch",
    "_ap_normalize_unicode",
    "_ap_try_match",
    "_ap_seek_sequence",
    "_ap_compute_replacements",
    "_ap_apply_replacements",
    "_ap_derive_new_contents",
    "_apply_patch_target_paths",
    "run_apply_patch",
]
