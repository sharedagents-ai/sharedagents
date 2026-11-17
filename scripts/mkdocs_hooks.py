"""MkDocs hook: serve README.md as the site's index page.

Links into ``docs/`` become site-relative; links to other repo files point
at GitHub, since only ``docs/`` is published.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from mkdocs.structure.files import File

REPO = Path(__file__).resolve().parents[1]
BLOB = "https://github.com/sharedagents-ai/sharedagents/blob/main/"
_LOCAL_LINK = re.compile(r"\]\((?!https?://|#|<)([^)\s]+)\)")


def _rewrite(match: re.Match) -> str:
    target = match.group(1)
    if target.startswith("docs/"):
        return f"]({target[len('docs/') :]})"
    return f"]({BLOB}{target})"


def on_files(files, config):
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    files.append(File.generated(config, "index.md", content=_LOCAL_LINK.sub(_rewrite, readme)))
    return files


def on_page_markdown(markdown, page, config, files):
    """Keep links to repo-root files working from nested documentation pages."""
    docs_dir = Path(config["docs_dir"]).resolve()
    source_dir = docs_dir / Path(page.file.src_uri).parent

    def rewrite(match: re.Match) -> str:
        target = match.group(1)
        url = urlsplit(target)
        if url.scheme or not url.path:
            return match.group(0)
        resolved = (source_dir / url.path).resolve()
        if not resolved.is_relative_to(REPO) or resolved.is_relative_to(docs_dir) or not resolved.is_file():
            return match.group(0)
        suffix = ("?" + url.query if url.query else "") + ("#" + url.fragment if url.fragment else "")
        return f"]({BLOB}{resolved.relative_to(REPO).as_posix()}{suffix})"

    return _LOCAL_LINK.sub(rewrite, markdown)
