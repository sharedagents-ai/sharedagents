"""The Usage history charts in a browser: period selector, autozoom, legend.

The page is the shipped template + static files; ``/api/usage/pace-history``
is answered by the real ``pace_history_window`` over a synthetic store.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from jinja2 import Environment, FileSystemLoader

from sharedagents_web import subscription_limits as sl

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/sharedagents_web/static"
TYPES = {".css": "text/css", ".js": "application/javascript", ".svg": "image/svg+xml"}
NEWEST = datetime(2026, 10, 27, 19, 0, tzinfo=timezone.utc)
GAP = (NEWEST - timedelta(hours=10), NEWEST - timedelta(hours=7))  # recorder quiet 3 h


def _point(when: datetime, *, blank: bool = False) -> dict:
    hours = (NEWEST - when).total_seconds() / 3600
    claude = 95 + 35 * math.sin(hours / 5)  # 60..130
    zcode = 410 * math.exp(-(hours - 0.5) * 1.5) if hours >= 0.5 else 20.0
    subs = [
        {"id": "claude", "name": "Claude", "pace_used_percent": None if blank else round(claude, 1)},
        {"id": "codex", "name": "Codex", "pace_used_percent": None if blank else 70.0},
        {"id": "zcode", "name": "ZCode (Z.AI)", "pace_used_percent": None if blank else round(max(zcode, 20.0), 1)},
    ]
    api = [{"id": "xai", "name": "xAI", "remaining": round(18.19 - hours * 0.01, 2), "used": None, "plan": "API"}]
    return {"timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "subscriptions": subs, "api": api}


def _store() -> list[dict]:
    rows = []
    t = NEWEST - timedelta(days=10)
    while t <= NEWEST:
        recent = t >= NEWEST - timedelta(days=1)
        if not (GAP[0] < t < GAP[1]):
            rows.append(_point(t))
        if t == NEWEST - timedelta(hours=2):
            # A refresh that saved no pace at all must not cut the lines.
            rows.append(_point(t + timedelta(seconds=30), blank=True))
        t += timedelta(minutes=10 if recent else 30)
    return rows


@pytest.fixture
def usage_page(monkeypatch, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    store = tmp_path / "pace_history.json"
    store.write_text(json.dumps({"history": _store()}))
    monkeypatch.setenv("SHAREDAGENTS_PACE_HISTORY_PATH", str(store))
    monkeypatch.delenv("SHAREDAGENTS_SUBSCRIPTION_FIXTURE", raising=False)
    env = Environment(loader=FileSystemLoader(ROOT / "src/sharedagents_web/templates"), autoescape=True)
    env.globals["asset_rev"] = lambda name: "fixture"
    html = env.get_template("usage.html").render(base="/sharedagentsweb", site_title="sharedagents", is_admin=True)
    asked: list[str] = []

    def route(request):
        url = request.request.url
        path = urlparse(url).path
        if "/static/" in path:
            file = STATIC / path.split("/static/", 1)[1]
            if file.is_file():
                request.fulfill(path=file, content_type=TYPES.get(file.suffix, "application/octet-stream"))
            else:
                request.fulfill(status=404, body="")
        elif path.endswith("/api/usage/pace-history"):
            period = (parse_qs(urlparse(url).query).get("period") or [""])[0]
            asked.append(period)
            request.fulfill(json=sl.pace_history_window(period) if period else sl.get_pace_history())
        elif "/api/" in path:
            request.fulfill(json={})
        else:
            request.fulfill(body=html, content_type="text/html")

    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except playwright.Error as error:
            if "Executable doesn't exist" in str(error):
                pytest.skip("Playwright Chromium is not installed")
            raise
        # A 24-hour locale; the axis follows the browser's own clock format.
        errors: list[str] = []

        def open_at(
            width: int = 1280, tab: str = "subscriptions", *, history=None, timezone_id="UTC", storage_blocked=False
        ):
            if history is not None:
                store.write_text(json.dumps({"history": history}))
            context = browser.new_context(locale="en-GB", timezone_id=timezone_id)
            if storage_blocked:
                context.add_init_script("""Object.defineProperty(window, 'localStorage', {
                  get() { throw new DOMException('Storage blocked', 'SecurityError'); }
                });""")
            page = context.new_page()
            page.set_viewport_size({"width": width, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", route)
            page.goto(f"https://usage.example/sharedagentsweb/usage#{tab}")
            chart = "#usage-sub-chart" if tab == "subscriptions" else "#usage-api-chart"
            page.wait_for_selector(f"{chart} svg circle")
            return page

        yield open_at, asked
        browser.close()
        assert errors == [], errors


def _chart(page, chart="#usage-sub-chart"):
    return page.evaluate(
        """(sel) => {
          const root = document.querySelector(sel);
          const texts = (cls) => [...root.querySelectorAll('text.' + cls)].map(t => t.textContent);
          const paths = Object.fromEntries([...root.querySelectorAll('path[data-series]')]
            .map(p => [p.dataset.series, (p.getAttribute('d').match(/M/g) || []).length]));
          const bridges = Object.fromEntries([...root.querySelectorAll('path[data-bridge]')]
            .map(p => [p.dataset.bridge, (p.getAttribute('d').match(/M/g) || []).length]));
          const dots = root.querySelectorAll('circle').length;
          return {ylabels: texts('usage-chart-ylabel'), times: texts('usage-chart-time'), paths, bridges, dots};
        }""",
        chart,
    )


def _pressed(page, group):
    return page.locator(f'[data-history-chart="{group}"] button[aria-pressed="true"]').all_text_contents()


def test_default_window_fits_the_axis_and_breaks_at_the_gap(usage_page):
    open_at, asked = usage_page
    page = open_at()
    assert _pressed(page, "sub") == ["24h"]
    assert "24h" in asked
    chart = _chart(page)
    # ZCode's 410% burst sets the top: 0..500 in steps of 100.
    assert chart["ylabels"] == ["0%", "100%", "200%", "300%", "400%", "500%"]
    # Round local times every 6 hours; midnight shows the date.
    assert chart["times"] == ["27 Oct", "06:00", "12:00", "18:00"]
    # The 3-hour gap is one dashed bridge per line; the blank refresh is no gap.
    assert chart["paths"] == {"claude": 2, "codex": 2, "zcode": 2}
    assert chart["bridges"] == {"claude": 1, "codex": 1, "zcode": 1}


def test_hiding_a_line_zooms_to_the_rest(usage_page):
    open_at, _ = usage_page
    page = open_at()
    key = page.locator('#usage-sub-chart-legend [data-series="zcode"]')
    key.click()
    assert key.get_attribute("aria-pressed") == "false"
    chart = _chart(page)
    assert "zcode" not in chart["paths"]
    assert chart["ylabels"] == ["0%", "50%", "100%", "150%"]  # Claude peaks at 130%
    page.locator('#usage-sub-chart-legend [data-series="codex"]').click()
    page.locator('#usage-sub-chart-legend [data-series="claude"]').click()  # the last line stays
    assert list(_chart(page)["paths"]) == ["claude"]
    key.click()
    assert _chart(page)["ylabels"][-1] == "500%"


def test_period_selector_asks_the_server_and_is_remembered(usage_page):
    open_at, asked = usage_page
    page = open_at()
    page.locator('[data-history-chart="sub"] [data-history-period="7d"]').click()
    page.wait_for_function(
        "() => document.querySelector('#usage-sub-chart text.usage-chart-time').textContent === '22 Oct'"
    )
    assert asked[-1] == "7d" and _pressed(page, "sub") == ["7d"]
    chart = _chart(page)
    assert chart["times"] == ["22 Oct", "24 Oct", "26 Oct"]
    assert chart["dots"] <= 3 * sl.PACE_HISTORY_MAX_POINTS
    page.locator('[data-history-chart="sub"] [data-history-period="6h"]').click()
    page.wait_for_function(
        "() => document.querySelector('#usage-sub-chart text.usage-chart-time').textContent === '14:00'"
    )
    assert _chart(page)["ylabels"][-1] == "500%"
    page.reload()
    page.wait_for_selector("#usage-sub-chart svg path")
    assert _pressed(page, "sub") == ["6h"]
    assert _pressed(page, "api") == ["24h"]  # each chart keeps its own period


def test_balance_chart_has_its_own_selector(usage_page):
    open_at, asked = usage_page
    page = open_at(tab="api")
    page.locator('[data-history-chart="api"] [data-history-period="30d"]').click()
    page.wait_for_function(
        "() => document.querySelector('#usage-api-chart text.usage-chart-time').textContent === '19 Oct'"
    )
    assert asked[-1] == "30d" and _pressed(page, "api") == ["30d"]
    assert _chart(page, "#usage-api-chart")["ylabels"][-1] == "$20.0"


@pytest.mark.parametrize("width", [390, 760])
def test_narrow_layouts_fit(usage_page, width):
    open_at, _ = usage_page
    page = open_at(width)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    head = page.locator("#usage-sub-chart-section .usage-pace-chart-header").bounding_box()
    toggle = page.locator('[data-history-chart="sub"]').bounding_box()
    assert toggle["x"] >= head["x"] and toggle["x"] + toggle["width"] <= head["x"] + head["width"] + 1
    times = _chart(page)["times"]
    assert (2 <= len(times) <= 3) if width < 600 else len(times) == 4
    boxes = page.locator("#usage-sub-chart text.usage-chart-time").evaluate_all(
        "els => els.map(el => { const r = el.getBoundingClientRect(); return [r.left, r.right]; })"
    )
    assert all(a[1] + 4 <= b[0] for a, b in zip(boxes, boxes[1:]))


def test_a_new_period_cannot_leave_only_hidden_series(usage_page):
    open_at, _ = usage_page
    page = open_at()
    page.locator('#usage-sub-chart-legend [data-series="claude"]').click()
    page.locator('#usage-sub-chart-legend [data-series="codex"]').click()
    row = _point(NEWEST)
    row["subscriptions"] = row["subscriptions"][:2]
    page.route("**/api/usage/pace-history?period=6h", lambda request: request.fulfill(json={"history": [row]}))
    page.locator('[data-history-chart="sub"] [data-history-period="6h"]').click()
    page.wait_for_function("() => !document.querySelector('#usage-sub-chart-legend [data-series=zcode]')")
    assert _chart(page)["paths"]  # restore a visible series in the new window
    assert page.locator('#usage-sub-chart-legend [aria-pressed="true"]').count() >= 1


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2026-03-07T17:00:00+00:00", "2026-03-09T16:00:00+00:00", ["12:00", "8 Mar", "12:00", "9 Mar", "12:00"]),
        ("2026-10-31T16:00:00+00:00", "2026-11-02T17:00:00+00:00", ["12:00", "1 Nov", "12:00", "2 Nov", "12:00"]),
    ],
)
def test_time_ticks_stay_on_round_local_times_across_dst(usage_page, start, end, expected):
    open_at, _ = usage_page
    rows = [_point(datetime.fromisoformat(t)) for t in [start, end]]
    page = open_at(history=rows, timezone_id="America/New_York")
    page.locator('[data-history-chart="sub"] [data-history-period="7d"]').click()
    page.wait_for_function("() => document.querySelectorAll('#usage-sub-chart circle').length === 6")
    assert _chart(page)["times"] == expected


def test_late_period_response_cannot_replace_the_current_window(usage_page):
    open_at, _ = usage_page
    page = open_at()
    pending = []
    page.route("**/api/usage/pace-history?period=7d", lambda request: pending.append(request))
    page.locator('[data-history-chart="sub"] [data-history-period="7d"]').click()
    page.wait_for_timeout(50)
    assert len(pending) == 1
    page.locator('[data-history-chart="sub"] [data-history-period="6h"]').click()
    page.wait_for_function(
        "() => document.querySelector('#usage-sub-chart text.usage-chart-time').textContent === '14:00'"
    )
    current = _chart(page)
    pending[0].fulfill(json=sl.pace_history_window("7d"))
    page.wait_for_timeout(100)
    assert _chart(page) == current
    assert _pressed(page, "sub") == ["6h"]


def test_periods_work_with_storage_blocked_and_keyboard_legend_keeps_focus(usage_page):
    open_at, asked = usage_page
    page = open_at(storage_blocked=True)
    group = page.get_by_role("group", name="Pace history period")
    assert group.locator('[aria-pressed="false"]').count() == 3
    group.get_by_role("button", name="6h", exact=True).focus()
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => document.querySelector('#usage-sub-chart text.usage-chart-time').textContent === '14:00'"
    )
    assert asked[-1] == "6h" and _pressed(page, "sub") == ["6h"]
    key = page.locator('#usage-sub-chart-legend [data-series="zcode"]')
    key.focus()
    page.keyboard.press("Space")
    assert key.get_attribute("aria-pressed") == "false"
    assert key.evaluate("el => el === document.activeElement")
    page.keyboard.press("Enter")
    assert key.get_attribute("aria-pressed") == "true"
    assert key.evaluate("el => el === document.activeElement")


@pytest.mark.parametrize("value,count", [(0, 1), (0, 3), (0.39, 3), (18.19, 3)])
def test_balance_degenerate_data_and_small_dollar_axis(usage_page, value, count):
    open_at, _ = usage_page
    rows = [_point(NEWEST - timedelta(hours=i)) for i in reversed(range(count))]
    for row in rows:
        row["api"][0]["remaining"] = value
    page = open_at(390, tab="api", history=rows)
    chart = _chart(page, "#usage-api-chart")
    assert chart["dots"] == count
    assert len(chart["ylabels"]) == len(set(chart["ylabels"]))
    assert float(chart["ylabels"][-1][1:]) > value
    assert "NaN" not in page.locator("#usage-api-chart").inner_html()
    assert "Infinity" not in page.locator("#usage-api-chart").inner_html()
    if value == 0.39:
        assert chart["ylabels"] == ["$0.00", "$0.10", "$0.20", "$0.30", "$0.40", "$0.50"]
    boxes = page.locator("#usage-api-chart text.usage-chart-time").evaluate_all(
        "els => els.map(el => { const r = el.getBoundingClientRect(); return [r.left, r.right]; })"
    )
    assert all(a[1] + 4 <= b[0] for a, b in zip(boxes, boxes[1:]))
