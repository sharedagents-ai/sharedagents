#!/usr/bin/env python3
"""Fail on private identities, internal hosts, RFC1918 addresses, or forge URLs.

Run on a working tree or extracted distribution directory. Paths and contents
are scanned, including binary assets and interpreter caches. Archives (zip,
wheel, egg, jar, tar, and gzip/bzip2/xz streams such as .tar.gz and .tgz) are
opened recursively and every member's name and payload is scanned too; a
compressed payload is otherwise opaque to a byte search. An archive that cannot
be read completely (corrupt, encrypted, nested too deep or expanding past the
size limit) fails the scan: nothing is skipped silently. Only Git internals
are excluded. Identity strings (names, handles, hosts) are read from a denylist
file kept outside the repository, one per line, given with ``--denylist FILE``
or ``SHAREDAGENTS_SCRUB_DENYLIST``; the repository never carries a copy of what
it must reject. Matching is case-insensitive and includes substrings.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import re
import sys
import tarfile
import unicodedata
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

OCTET = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
RULES = [
    re.compile(r"192\.168\.|10\.0\.0\.|172\.17\.", re.I),
    re.compile(
        r"(?<![\d.])(?:10\."
        + OCTET
        + r"\."
        + OCTET
        + r"\."
        + OCTET
        + r"|192\.168\."
        + OCTET
        + r"\."
        + OCTET
        + r"|172\.(?:1[6-9]|2\d|3[01])\."
        + OCTET
        + r"\."
        + OCTET
        + r")(?![\d.])"
    ),
    # A forge URL with a literal host; ``https://{host}/gitea`` templates are not hosts.
    re.compile(r"https?://[^/\s<>\"']*gitea[^\s<>\"']*|https?://[a-z0-9.-]+(?::\d+)?/gitea(?:/|\b)", re.I),
    re.compile(r"(?:ssh|git)://[^\s<>\"']*gitea[^\s<>\"']*", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I),
]
DENYLIST: list[str] = []
# URLs on RFC 2606 / RFC 6761 reserved names cannot point at a private host.
RESERVED_URL = re.compile(
    r"(?:https?|ssh|git)://(?:[^/\s@<>\"']*@)?(?:[a-z0-9-]+\.)*"
    r"(?:example(?:\.(?:com|net|org))?|test|invalid|localhost)(?::\d+)?(?=[/\s<>\"']|$)[^\s<>\"']*",
    re.I,
)


ZIP_SUFFIXES = (".zip", ".whl", ".egg", ".jar")
TAR_SUFFIXES = (".tar",)
STREAMS = {
    ".tgz": (gzip, ".tar"),
    ".gz": (gzip, ""),
    ".tbz2": (bz2, ".tar"),
    ".bz2": (bz2, ""),
    ".txz": (lzma, ".tar"),
    ".xz": (lzma, ""),
}
MAX_DEPTH = 8
MAX_EXPANDED = 1 << 30  # bytes unpacked from one scanned file, all levels together
UNREADABLE = "UNREADABLE ARCHIVE: "


class ArchiveError(Exception):
    """An archive could not be inspected completely, so the scan must fail."""


def _stream_suffix(name: str) -> str | None:
    lower = name.lower()
    return next((suffix for suffix in STREAMS if lower.endswith(suffix)), None)


def _stream_info(name: str, data: bytes) -> tuple[type, str] | None:
    """Determine compression module and inner suffix by magic bytes or extension."""
    suffix = _stream_suffix(name)
    if suffix:
        return STREAMS[suffix]
    if data[:2] == b"\x1f\x8b":
        return gzip, ""
    if data[:3] == b"BZh":
        return bz2, ""
    if data[:6] == b"\xfd7zXZ\x00":
        return lzma, ""
    return None


def _archive_kind(name: str, data: bytes) -> str | None:
    """Classify by suffix or by content, so a renamed archive is still opened."""
    lower = name.lower()
    if lower.endswith(ZIP_SUFFIXES) or data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if _stream_info(name, data) is not None:
        return "stream"
    if lower.endswith(TAR_SUFFIXES) or data[257:262] == b"ustar":
        return "tar"
    return None


def _bounded_read(stream, budget: list[int], name: str) -> bytes:
    data = stream.read(budget[0] + 1)
    budget[0] -= len(data)
    if budget[0] < 0:
        raise ArchiveError(f"{name}: expands past {MAX_EXPANDED} bytes")
    return data


def _members(name: str, data: bytes, kind: str, budget: list[int]) -> Iterator[tuple[str, bytes]]:
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.flag_bits & 0x1:
                    raise ArchiveError(f"{name}: encrypted member {info.filename}")
                if info.is_dir():
                    yield info.filename, b""
                    continue
                with archive.open(info) as stream:
                    yield info.filename, _bounded_read(stream, budget, name)
    elif kind == "tar":
        # Raw tar bytes (owner names, link targets, PAX headers) are scanned by
        # the caller; here every member is visited by name, payload and link target.
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member)
                    yield member.name, _bounded_read(stream, budget, name) if stream else b""
                elif member.issym() or member.islnk():
                    yield member.name, member.linkname.encode("utf-8", "surrogateescape")
                else:
                    yield member.name, b""
    else:
        info = _stream_info(name, data)
        if info is None:
            raise ArchiveError(f"{name}: cannot determine stream compression")
        module, inner_suffix = info
        base = Path(name).name
        suffix = _stream_suffix(name)
        inner = base[: -len(suffix)] + inner_suffix if suffix else base + ".decompressed"
        opener = {gzip: gzip.GzipFile, bz2: bz2.BZ2File, lzma: lzma.LZMAFile}[module]
        with opener(fileobj=io.BytesIO(data)) as stream:
            yield inner, _bounded_read(stream, budget, name)


def expand(name: str, data: bytes, depth: int = 0, budget: list[int] | None = None) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` for a file and, recursively, every archive member.

    Member paths are joined to their archive with ``!``. Raises ArchiveError
    when any level cannot be read completely; callers must treat that as a
    failure, never as a skip.
    """
    budget = [MAX_EXPANDED] if budget is None else budget
    yield name, data
    kind = _archive_kind(name, data)
    if kind is None:
        return
    if depth >= MAX_DEPTH:
        raise ArchiveError(f"{name}: archives nested deeper than {MAX_DEPTH} levels")
    try:
        members = list(_members(name, data, kind, budget))
    except ArchiveError:
        raise
    except (
        zipfile.BadZipFile,
        tarfile.TarError,
        OSError,
        EOFError,
        lzma.LZMAError,
        NotImplementedError,
        RuntimeError,
        ValueError,
        zlib.error,
    ) as exc:
        raise ArchiveError(f"{name}: cannot open {kind} archive ({type(exc).__name__}: {exc})") from exc
    for member, payload in members:
        yield from expand(f"{name}!{member}", payload, depth + 1, budget)


def load_denylist(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        unicodedata.normalize("NFC", line.strip()).lower()
        for line in lines
        if line.strip() and not line.startswith("#")
    ]


def violations(data: bytes) -> bool:
    text = RESERVED_URL.sub("", data.decode("latin-1").lower())
    if any(name in text for name in DENYLIST) or any(rule.search(text) for rule in RULES):
        return True
    # The latin-1 view never matches a non-ASCII term (an accented name) in
    # UTF-8 text, so those terms are also checked against the UTF-8 reading.
    accented = [name for name in DENYLIST if not name.isascii()]
    if not accented:
        return False
    utext = RESERVED_URL.sub("", unicodedata.normalize("NFC", data.decode("utf-8", "replace")).lower())
    return any(name in utext for name in accented)


def scan_file(rel: str, data: bytes) -> list[str]:
    """Hits for one file: its own name and bytes, then every archive member."""
    hits = []
    try:
        for name, payload in expand(rel, data):
            try:
                name_bytes = name.encode("utf-8", "surrogateescape")
            except Exception:
                name_bytes = name.encode("latin-1", "replace")
            if violations(name_bytes) or violations(payload):
                hits.append(name)
    except ArchiveError as exc:
        hits.append(f"{UNREADABLE}{exc}")
    except Exception as exc:
        hits.append(f"{UNREADABLE}{rel}: unhandled scan error ({type(exc).__name__}: {exc})")
    return hits


def scan(root: Path) -> tuple[int, list[str]]:
    hits = []
    files = 0
    candidates = sorted(root.rglob("*")) if root.is_dir() else [root]
    for path in candidates:
        rel = path.relative_to(root) if root.is_dir() else Path(path.name)
        if ".git" in rel.parts or not path.is_file():
            continue
        files += 1
        hits.extend(scan_file(str(rel), path.read_bytes()))
    return files, hits


def main(args: list[str]) -> int:
    denylist = os.environ.get("SHAREDAGENTS_SCRUB_DENYLIST", "")
    if args[:1] == ["--denylist"] and len(args) > 1:
        denylist, args = args[1], args[2:]
    if not args:
        print("usage: check_forbidden_patterns.py [--denylist FILE] DIRECTORY [DIRECTORY ...]")
        return 2
    if denylist:
        DENYLIST[:] = load_denylist(denylist)
    else:
        print("warning: no denylist given; only the generic rules apply", file=sys.stderr)
    failures = 0
    for arg in args:
        root = Path(arg)
        if not root.exists():
            print(f"missing input: {root}")
            return 2
        count, hits = scan(root)
        for hit in hits:
            print(hit if hit.startswith(UNREADABLE) else f"FORBIDDEN: {hit}")
        print(f"{root.name}: scanned {count} files: {len(hits)} forbidden-pattern hits")
        failures += len(hits)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
