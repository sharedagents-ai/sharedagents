"""The context meter opens a full-screen Context sheet with an infinite,
windowed list: a sheet over the session (like the agent picker) with a compact
action, an infinite scroll over the context in use, and an X to close it.

Drives the REAL sheet markup (cut from index.html), style.css and
context_viewer.js in Chromium against a routed fake of the paged context API:
open / × / Escape / backdrop, pages fetched as the list scrolls, a bounded DOM
with 5,000 items, expanding an item and loading the rest of a truncated one,
and the Compact button running the shared confirm-first compact flow.
"""

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

_playwright = pytest.importorskip("playwright.sync_api")
sync_playwright = _playwright.sync_playwright

ROOT = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web"
STATIC = ROOT / "static"
TEMPLATE = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

N_ITEMS = 5_000
PAGE = 50
BIG = 600_000  # chars of item #3, a huge tool result


def _sheet_markup() -> str:
    start = TEMPLATE.index('<div id="ctx-backdrop"')
    marker = '<div id="ctx-status" class="ctx-status"></div>'
    end = TEMPLATE.index(marker, start) + len(marker)
    end = TEMPLATE.index("</div>", end) + len("</div>")  # .ctx-scroll
    end = TEMPLATE.index("</div>", end) + len("</div>")  # #ctx-sheet
    return TEMPLATE[start:end]


def _page_html() -> str:
    return (
        '<!doctype html><body style="margin:0;height:100vh">'
        '<div id="messages">chat</div>'
        '<div id="status-segments"><button id="compact-context-btn">79K/262K</button></div>'
        '<div id="agent-model-pop" hidden></div>' + _sheet_markup() + "</body>"
    )


def _content(i: int) -> str:
    if i == 3:
        return "R" * BIG
    return f"item {i} body " + ("lorem " * 20)


def _item(i: int, total_tokens_scale: int) -> dict:
    content = _content(i)
    return {
        "index": i,
        "kind": "tool_result" if i % 3 == 0 else "user",
        "kind_label": "Tool result" if i % 3 == 0 else "User",
        "role": "tool",
        "label": f"Turn {i // 3} · item {i}",
        "turn": i // 3,
        "tokens": 100 * total_tokens_scale,
        "tokens_label": f"{total_tokens_scale}00",
        "chars": len(content),
        "chars_label": f"{len(content)} chars",
        "images": 0,
        "preview": content[:80],
        "content": content[:4000],
        "truncated": len(content) > 4000,
        "withheld": False,
    }


class FakeApi:
    def __init__(self):
        self.page_requests = []
        self.item_requests = []
        self.compacted = False
        self.commands = []

    def totals(self):
        tokens = 40_000 if self.compacted else 800_000
        return {
            "items": N_ITEMS,
            "tokens": tokens,
            "window": 1_000_000,
            "label": f"{tokens // 1000}K / 1.0M · {tokens / 10_000:g}%",
            "meter_label": "Last request 790K",
        }

    def handle(self, route):
        url = urlparse(route.request.url)
        q = parse_qs(url.query)
        if url.path.endswith("/context"):
            cursor = (q.get("cursor") or [""])[0]
            offset = int(cursor.split(":")[1]) if cursor else 0
            self.page_requests.append(offset)
            end = min(N_ITEMS, offset + PAGE)
            body = {
                "ok": True,
                "snapshot": "snapB" if self.compacted else "snapA",
                "totals": self.totals(),
                "notice": "",
                "notes": [],
                "items": [_item(i, 2 if self.compacted else 1) for i in range(offset, end)],
                "next_cursor": f"snap:{end}" if end < N_ITEMS else None,
                "can_compact": True,
            }
        elif "/context/items/" in url.path:
            index = int(url.path.rsplit("/", 1)[1])
            offset = int((q.get("offset") or ["0"])[0])
            self.item_requests.append((index, offset))
            content = _content(index)
            chunk = content[offset : offset + 256 * 1024]
            nxt = offset + len(chunk)
            body = {
                "ok": True,
                "index": index,
                "offset": offset,
                "content": chunk,
                "total_chars": len(content),
                "next_offset": nxt if nxt < len(content) else None,
            }
        else:
            return route.fulfill(status=404, body="{}")
        route.fulfill(content_type="application/json", body=json.dumps(body))


STUBS = """
var BASE = "";
var state = { currentSessionId: "S1" };
var _cw = { state: "idle" };
window.__cmds = [];
function compactWidgetState() { return _cw; }
function _set(w) { _cw = w; if (window.SaContextSheet) window.SaContextSheet.compactChanged(w); }
function compactContextStart() { window.__cmds.push("start");
  _set({ state: "confirm", prompt: "Compacting the conversation context cannot be undone. Continue?" }); }
function compactContextConfirm() { window.__cmds.push("confirm"); _set({ state: "compacting" }); }
function compactContextCancel() { window.__cmds.push("cancel"); _set({ state: "idle" }); }
function compactContextDismiss() { _set({ state: "idle" }); }
function closeModelSheet() { document.getElementById("agent-model-pop").hidden = true; }
"""


@pytest.fixture(params=[(1440, 900), (390, 844)], ids=["desktop", "phone"])
def sheet(request):
    width, height = request.param
    api = FakeApi()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("http://ctx.sheet/api/**", api.handle)
        page.route("http://ctx.sheet/", lambda route: route.fulfill(content_type="text/html", body=_page_html()))
        page.goto("http://ctx.sheet/")
        page.add_style_tag(content=(STATIC / "style.css").read_text())
        page.add_script_tag(content=STUBS)
        page.add_script_tag(content=(STATIC / "context_viewer.js").read_text())
        try:
            yield page, api, (width, height), errors
        finally:
            browser.close()


def _stats(page):
    return page.evaluate("() => window.SaContextSheet.stats()")


def _open(page):
    page.evaluate("() => window.openContextSheet()")
    page.wait_for_selector("#ctx-list .ctx-row")


def test_open_close_escape_and_full_screen(sheet):
    page, api, (width, height), errors = sheet
    page.evaluate("() => document.getElementById('agent-model-pop').hidden = false")
    _open(page)
    assert page.is_hidden("#agent-model-pop"), "one sheet at a time"
    assert page.is_visible("#ctx-backdrop")
    box = page.locator("#ctx-sheet").bounding_box()
    assert page.evaluate("getComputedStyle(document.getElementById('ctx-sheet')).position") == "fixed"
    if width <= 520:
        assert box["width"] == width and box["height"] == height, "phones get the whole screen"
    else:
        assert box["width"] >= width - 24 and box["height"] >= height - 24
    assert "800K / 1.0M" in page.inner_text("#ctx-total")
    assert page.evaluate("document.activeElement.id") == "ctx-close"

    page.keyboard.press("Escape")
    assert page.is_hidden("#ctx-sheet") and page.is_hidden("#ctx-backdrop")
    assert page.evaluate("document.activeElement.id") == "compact-context-btn", "focus returns"

    _open(page)
    page.click("#ctx-close")
    assert page.is_hidden("#ctx-sheet")
    assert errors == []


def test_infinite_scroll_fetches_pages_and_keeps_the_dom_bounded(sheet):
    page, api, _vp, errors = sheet
    _open(page)
    assert api.page_requests[:1] == [0]
    start_nodes = _stats(page)["nodes"]
    for _ in range(400):
        stats = _stats(page)
        if stats["loaded"] >= N_ITEMS:
            break
        page.evaluate("() => { const s = document.getElementById('ctx-scroll'); s.scrollTop = s.scrollHeight; }")
        page.wait_for_timeout(15)
    stats = _stats(page)
    assert stats["loaded"] == N_ITEMS and stats["nextCursor"] is None
    offsets = api.page_requests
    assert offsets == sorted(set(offsets)), "each page requested once, in order"
    assert offsets == list(range(0, N_ITEMS, PAGE))
    # Windowed: only the rows near the viewport exist, whatever was loaded.
    assert stats["rendered"] < 60, stats
    assert stats["nodes"] < start_nodes + 400, stats
    page.evaluate("() => { const s = document.getElementById('ctx-scroll'); s.scrollTop = s.scrollHeight; }")
    page.wait_for_function(f"() => !!document.querySelector('#ctx-list .ctx-row[data-index=\"{N_ITEMS - 1}\"]')")
    assert _stats(page)["rendered"] < 60
    # Scrolling back to the top re-renders the first rows.
    page.evaluate("() => { document.getElementById('ctx-scroll').scrollTop = 0; }")
    page.wait_for_function("() => !!document.querySelector('#ctx-list .ctx-row[data-index=\"0\"]')")
    assert errors == []


def test_expand_one_item_and_load_the_rest_of_a_huge_one(sheet):
    page, api, _vp, errors = sheet
    _open(page)
    page.click('#ctx-list .ctx-row[data-index="1"] .ctx-row-head')
    assert page.get_attribute('#ctx-list .ctx-row[data-index="1"]', "data-expanded") == "true"
    page.click('#ctx-list .ctx-row[data-index="3"] .ctx-row-head')
    assert page.get_attribute('#ctx-list .ctx-row[data-index="1"]', "data-expanded") == "false"
    body = page.locator('#ctx-list .ctx-row[data-index="3"] .ctx-body')
    assert len(body.inner_text()) == 4000
    # Rows below the expanded one move down by its height (no overlap).
    exp_box = page.locator('#ctx-list .ctx-row[data-index="3"]').bounding_box()
    next_box = page.locator('#ctx-list .ctx-row[data-index="4"]').bounding_box()
    assert next_box["y"] >= exp_box["y"] + exp_box["height"] - 1
    for _ in range(4):
        more = page.locator('#ctx-list .ctx-row[data-index="3"] .ctx-more')
        if not more.count():
            break
        more.click()
        page.wait_for_timeout(50)
    assert (
        page.evaluate(
            "() => document.querySelector('#ctx-list .ctx-row[data-index=\"3\"] .ctx-body').textContent.length"
        )
        == BIG
    )
    assert [o for _, o in api.item_requests] == [0, 262144, 524288]
    assert errors == []


def test_compact_confirms_first_then_reloads_with_lower_totals(sheet):
    page, api, _vp, errors = sheet
    _open(page)
    page.click("#ctx-compact")
    assert page.is_visible("#ctx-compact-ok"), "confirm before compacting"
    assert "cannot be undone" in page.inner_text("#ctx-confirm")
    page.click("#ctx-compact-cancel")
    assert page.is_hidden("#ctx-confirm")
    page.click("#ctx-compact")
    page.click("#ctx-compact-ok")
    assert page.evaluate("window.__cmds") == ["start", "cancel", "start", "confirm"]
    assert "Compacting" in page.inner_text("#ctx-confirm")
    api.compacted = True
    page.evaluate("() => _set({ state: 'done' })")
    page.wait_for_function("() => document.getElementById('ctx-total').textContent.startsWith('40K')")
    assert api.page_requests.count(0) == 2, "the view reloads from the top"
    assert errors == []
