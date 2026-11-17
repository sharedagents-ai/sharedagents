#!/usr/bin/env python3
"""Deterministic existing-coverage check for the research-wiki skill.

Replaces the "read the index and eyeball it" step with an actual search
across the whole wiki. Run before starting any research loop:

    python3 check_coverage.py <keyword> [<keyword> ...]
    python3 check_coverage.py --wiki ~/wiki "samsung health" "wearable api"

Prints one machine-parseable summary line (COVERAGE_CHECK: ...) plus the
matched files/lines beneath it. Never guess this output — paste it verbatim.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Only these carry durable claims worth deduping against. sources/ is
# intentionally excluded — a source being cited before doesn't mean the
# topic is covered, only that a URL was read once.
SEARCH_DIRS = ("entities", "concepts")
SEARCH_FILES = ("index.md",)
MAX_SNIPPET_LEN = 200
MAX_MATCHES_PER_KEYWORD = 20


def iter_wiki_files(wiki_root: Path):
    for name in SEARCH_FILES:
        p = wiki_root / name
        if p.is_file():
            yield p
    for d in SEARCH_DIRS:
        dir_path = wiki_root / d
        if dir_path.is_dir():
            yield from sorted(dir_path.rglob("*.md"))


def search(wiki_root: Path, keywords: list[str]) -> dict[str, list[tuple[Path, int, str]]]:
    patterns = [(kw, re.compile(re.escape(kw), re.IGNORECASE)) for kw in keywords]
    hits: dict[str, list[tuple[Path, int, str]]] = {kw: [] for kw in keywords}
    files = list(iter_wiki_files(wiki_root))
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for kw, pattern in patterns:
                if len(hits[kw]) >= MAX_MATCHES_PER_KEYWORD:
                    continue
                if pattern.search(line):
                    snippet = line.strip()
                    if len(snippet) > MAX_SNIPPET_LEN:
                        snippet = snippet[:MAX_SNIPPET_LEN] + "…"
                    hits[kw].append((path, lineno, snippet))
    return hits, len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("keywords", nargs="+", help="topic keywords/synonyms to check")
    ap.add_argument("--wiki", default="~/wiki", help="wiki root (default: ~/wiki)")
    args = ap.parse_args()

    wiki_root = Path(args.wiki).expanduser()
    if not wiki_root.is_dir():
        print(f"COVERAGE_CHECK: WIKI_MISSING root={wiki_root} — scaffold it first (see SKILL.md), then re-run.")
        return 1

    hits, files_scanned = search(wiki_root, args.keywords)
    total_matches = sum(len(v) for v in hits.values())

    matched_files = sorted({str(path.relative_to(wiki_root)) for kw_hits in hits.values() for path, _, _ in kw_hits})

    if total_matches == 0:
        print(f"COVERAGE_CHECK: NO_MATCHES keywords={args.keywords!r} files_scanned={files_scanned} root={wiki_root}")
        return 0

    print(
        f"COVERAGE_CHECK: MATCHES_FOUND keywords={args.keywords!r} "
        f"total_matches={total_matches} matched_files={matched_files} "
        f"files_scanned={files_scanned} root={wiki_root}"
    )
    for kw, kw_hits in hits.items():
        if not kw_hits:
            continue
        print(f"\n-- '{kw}' ({len(kw_hits)} match(es)) --")
        for path, lineno, snippet in kw_hits:
            print(f"  {path.relative_to(wiki_root)}:{lineno}: {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
