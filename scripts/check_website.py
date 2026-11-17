"""Check a built website's local links, responsive layout, and interactions."""

from __future__ import annotations

import argparse
import functools
import json
import threading
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"src", "href"} and value:
                self.targets.append(value)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def check(output: Path, report_dir: Path) -> dict:
    output = output.resolve()
    missing = []
    for page in output.rglob("*.html"):
        links = Links()
        links.feed(page.read_text(encoding="utf-8"))
        for target in links.targets:
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            path = unquote(url.path)
            resolved = (output / path.lstrip("/") if path.startswith("/") else page.parent / path).resolve()
            if resolved.is_dir():
                resolved /= "index.html"
            if not resolved.is_relative_to(output) or not resolved.is_file():
                missing.append({"page": str(page.relative_to(output)), "target": target})
    if missing:
        raise RuntimeError(json.dumps({"missing_links": missing}, indent=2))

    report_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(output)))
    helper = threading.Thread(target=server.serve_forever)
    helper.start()
    errors, failures = [], []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
                try:
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.on(
                        "response",
                        lambda response: failures.append(response.status) if response.status >= 400 else None,
                    )
                    origin = f"http://127.0.0.1:{server.server_port}"
                    for width in (360, 390, 768, 1440):
                        page.set_viewport_size({"width": width, "height": 960})
                        page.goto(origin)
                        page.wait_for_load_state("networkidle")
                        assert page.title() == "SharedAgents — Your agents. Your workspace."
                        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
                        assert page.locator("main h1").count() == 1
                        for view in ("tasks", "activity", "development", "welcome"):
                            page.locator(f"[data-view='{view}']").click()
                            page.wait_for_function(
                                "document.querySelector('#tour-image').complete && document.querySelector('#tour-image').naturalWidth > 0"
                            )
                            assert page.locator(f"[data-view='{view}']").get_attribute("aria-pressed") == "true"
                        page.locator("#expand-image").click()
                        assert page.locator("#image-dialog").evaluate("el => el.open")
                        page.keyboard.press("Escape")
                        assert not page.locator("#image-dialog").evaluate("el => el.open")
                        page.locator("#copy-install").click()
                        assert page.evaluate("navigator.clipboard.readText()") == 'pip install "sharedagents[web]"'
                        page.evaluate("scrollTo(0,0)")
                        if width in (390, 1440):
                            name = "mobile" if width == 390 else "desktop"
                            page.screenshot(path=report_dir / f"website-{name}.png", animations="disabled")
                            page.screenshot(
                                path=report_dir / f"website-{name}-full.png", full_page=True, animations="disabled"
                            )
                    response = page.goto(origin + "/docs/getting-started/interface-tour/")
                    assert response and response.status == 200
                    assert page.locator("h1").inner_text().startswith("Interface tour")
                    assert page.locator("article img").count() == 7
                finally:
                    context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        helper.join()
        server.server_close()
    report = {
        "local_links": "PASS",
        "viewport_widths": [360, 390, 768, 1440],
        "interactions": "PASS",
        "page_errors": errors,
        "http_failures": failures,
    }
    (report_dir / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors or failures:
        raise RuntimeError(json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "build/website")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "build/website-qa")
    args = parser.parse_args()
    print(json.dumps(check(args.site, args.report_dir), indent=2))


if __name__ == "__main__":
    main()
