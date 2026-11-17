"""MkDocs hook: serve README.md as the site's index page.

Links into ``docs/`` become site-relative; links to other repo files point
at GitHub, since only ``docs/`` is published.
"""

from __future__ import annotations

import re
from pathlib import Path

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
