"""Documentation links work on GitHub and in the generated site."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mkdocs")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("docs_hooks", ROOT / "scripts" / "mkdocs_hooks.py")
HOOKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOOKS)


def test_nested_page_links_preserve_site_pages_and_anchors():
    page = SimpleNamespace(file=SimpleNamespace(src_uri="development/setup.md"))
    source = "[policy](../../SECURITY.md#reporting-a-vulnerability) [tests](testing.md#default-checks)"
    result = HOOKS.on_page_markdown(source, page, {"docs_dir": str(ROOT / "docs")}, None)
    assert result == (
        "[policy](https://github.com/sharedagents-ai/sharedagents/blob/main/SECURITY.md#reporting-a-vulnerability) "
        "[tests](testing.md#default-checks)"
    )


def test_missing_repository_links_are_left_for_strict_validation():
    page = SimpleNamespace(file=SimpleNamespace(src_uri="development/setup.md"))
    source = "[missing](../../does-not-exist.md) [external](https://example.com) [anchor](#here)"
    assert HOOKS.on_page_markdown(source, page, {"docs_dir": str(ROOT / "docs")}, None) == source
