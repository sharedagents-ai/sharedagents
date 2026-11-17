"""Publication checks reject broken downloads, documentation, and error routing."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("website_checks", ROOT / "scripts/check_website.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\nprintf 'example installer'\n")
    (source / "mkdocs.yml").write_text("site_url: https://example.com/docs/\n")
    monkeypatch.setattr(module, "ROOT", source)
    return module, source


@pytest.fixture
def site(tmp_path, checker):
    _, source = checker
    site = tmp_path / "site"
    for name, content in {
        "install.sh": (source / "install.sh").read_text(),
        "404.html": '<h1>Page not found</h1><a href="/">Home</a>',
        "docs/index.html": "<h1>Documentation</h1>",
        "docs/search/search_index.json": '{"docs": [{"location": "index.html", "title": "Docs"}]}',
        "docs/sitemap.xml": (
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://example.com/docs/</loc></url></urlset>"
        ),
        "robots.txt": "User-agent: *\nAllow: /\nSitemap: https://example.com/docs/sitemap.xml\n",
    }.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return site


def test_valid_publication(checker, site):
    checker[0].check_publication(site)


@pytest.mark.parametrize(
    "missing",
    ["install.sh", "404.html", "docs/index.html", "docs/search/search_index.json", "docs/sitemap.xml", "robots.txt"],
)
def test_missing_publication_file_fails(checker, site, missing):
    (site / missing).unlink()
    with pytest.raises(RuntimeError, match=missing):
        checker[0].check_publication(site)


def test_installer_changed_bytes_fail(checker, site):
    (site / "install.sh").write_text("<!doctype html><h1>Home</h1>")
    with pytest.raises(RuntimeError, match="installer"):
        checker[0].check_publication(site)


def test_missing_installer_source_fails(checker, site):
    (checker[1] / "install.sh").unlink()
    with pytest.raises(RuntimeError, match="install.sh"):
        checker[0].check_publication(site)


def test_empty_search_index_fails(checker, site):
    (site / "docs/search/search_index.json").write_text('{"docs": []}')
    with pytest.raises(RuntimeError, match="search index"):
        checker[0].check_publication(site)


@pytest.mark.parametrize(
    "xml",
    [
        "<broken>",
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://example.net/docs/</loc></url></urlset>",
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://example.com/docs/../private/</loc></url></urlset>",
    ],
)
def test_invalid_sitemap_fails(checker, site, xml):
    (site / "docs/sitemap.xml").write_text(xml)
    with pytest.raises(RuntimeError, match="sitemap"):
        checker[0].check_publication(site)


def test_robots_wrong_sitemap_fails(checker, site):
    (site / "robots.txt").write_text("Sitemap: https://example.net/sitemap.xml\n")
    with pytest.raises(RuntimeError, match="robots"):
        checker[0].check_publication(site)


def test_build_includes_installer_and_root_404(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("website_build", ROOT / "website/build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    (tmp_path / "website/assets").mkdir(parents=True)
    (tmp_path / "docs/assets/demo").mkdir(parents=True)
    for name in ["index.html", "styles.css", "app.js", "404.html"]:
        (tmp_path / "website" / name).write_text(name)
    (tmp_path / "install.sh").write_text("#!/bin/sh\nprintf example\n")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "HERE", tmp_path / "website")
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["build.py"])
    build.main()
    output = tmp_path / "build/website"
    assert (output / "install.sh").read_bytes() == (tmp_path / "install.sh").read_bytes()
    assert (output / "404.html").read_text() == "404.html"
