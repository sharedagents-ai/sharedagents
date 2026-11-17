"""Render compact subscription cards and their expanded pace and limit rows."""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from sharedagents_web import subscription_limits as sl

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/sharedagents_web/static"
TYPES = {".css": "text/css", ".js": "application/javascript", ".svg": "image/svg+xml"}

BOXES = """() => [...document.querySelectorAll('.usage-sub-card')]
  .filter(card => card.querySelector('.usage-sub-pace-row'))
  .map(card => {
    const box = el => {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height};
    };
    const pace = card.querySelector('.usage-sub-pace-row');
    const vendor = card.querySelector('.usage-sub-windows > .usage-sub-window');
    const secondVendor = vendor && vendor.nextElementSibling;
    const name = card.querySelector('.usage-sub-name');
    return {
      id: card.dataset.provider,
      first: card.querySelector('.usage-sub-windows').firstElementChild === pace,
      inHead: !!card.querySelector('.usage-sub-head .usage-sub-pace-row'),
      head: box(card.querySelector('.usage-sub-head')),
      name: box(name), nameScroll: name.scrollWidth, nameClient: name.clientWidth,
      pace: box(pace), paceBar: box(pace.querySelector('.usage-sub-bar')),
      paceFont: getComputedStyle(pace.querySelector('.usage-sub-row')).fontSize,
      vendor: box(vendor), vendorBar: box(vendor && vendor.querySelector('.usage-sub-bar')),
      secondVendor: box(secondVendor),
      vendorFont: vendor && getComputedStyle(vendor.querySelector('.usage-sub-row')).fontSize,
      label: box(pace.querySelector('.usage-sub-pace-label')),
      value: box(pace.querySelector('.usage-sub-pace-value')),
      next: box(pace.querySelector('.usage-sub-use-next')),
    };
  })"""


def _edge_snapshot(scenario):
    stamp = "2026-10-16T12:00:00Z"
    future = "2026-10-20T00:00:00Z"  # halfway through a week
    win = sl._window
    if scenario == "stale_pin":
        subject = sl._provider(
            "agy",
            updated_at="2026-10-16T11:49:00Z",
            windows=[
                win(
                    "pinned",
                    "Gemini pinned",
                    100,
                    future,
                    window_seconds=604800,
                    model_slug="gemini-3.7-flash-medium",
                    pool="Pinned",
                ),
                win("open", "Gemini general", 1, future, window_seconds=604800, pool="General"),
            ],
        )
    elif scenario == "unpaced_pool":
        subject = sl._provider(
            "cursor",
            updated_at=stamp,
            windows=[
                win("hit", "A weekly", 100, future, window_seconds=604800, pool="A"),
                win("open", "B weekly", 50, None, window_seconds=604800, pool="B"),
            ],
        )
    else:
        subject = sl._provider(
            "cursor",
            updated_at=stamp,
            windows=[
                win("a5", "A 5-hour", 100, "2026-10-16T13:00:00Z", window_seconds=18000, pool="A"),
                win("a7", "A weekly", 20, future, window_seconds=604800, pool="A"),
                win("b", "B weekly", 5, future, window_seconds=604800, pool="B"),
            ],
        )
        if scenario == "reset":
            # Reuse the object so stale tags from the previous calculation
            # would persist if the reset path did not clear them.
            sl._attach_pace_to_card(subject, now=sl._parse_iso_epoch(stamp))
            stamp = subject["updated_at"] = "2026-10-16T13:00:00Z"
    competitor = sl._provider(
        "codex",
        updated_at=stamp,
        windows=[
            win("weekly", "Weekly", 40, future, window_seconds=604800),
        ],
    )
    return sl._attach_pace_to_snapshot(
        {"updated_at": stamp, "providers": [subject, competitor]}, now=sl._parse_iso_epoch(stamp)
    )


@pytest.fixture
def subscriptions_page(monkeypatch, tmp_path, request):
    playwright = pytest.importorskip("playwright.sync_api")
    monkeypatch.setenv("SHAREDAGENTS_SUBSCRIPTION_FIXTURE", "1")
    monkeypatch.setenv("SHAREDAGENTS_SUBSCRIPTION_CACHE", str(tmp_path / "subscriptions.json"))
    scenario = getattr(request, "param", None)
    data = _edge_snapshot(scenario) if scenario and scenario not in ("eight", "eight-api") else sl.snapshot()
    if scenario == "eight":
        for number in range(6, 9):
            card = _simple_card(f"subscription-{number}", number * 10)
            card.update(name=f"Subscription {number}", plan="Standard", updated_at=data["updated_at"])
            data["providers"].append(card)
    if scenario == "eight-api":
        for number in range(5, 9):
            card = _balance_card(f"api-{number}", number * 5, 100)
            card.update(name=f"API {number}", plan="Prepaid", updated_at=data["updated_at"])
            data["providers"].append(card)
    # The longest plan badge seen live; it used to wrap beside the pill.
    data["providers"][0]["plan"] = "Standard · estimate"
    cards = {card["id"]: card for card in data["providers"]}
    env = Environment(loader=FileSystemLoader(ROOT / "src/sharedagents_web/templates"), autoescape=True)
    env.globals["asset_rev"] = lambda name: "fixture"
    html = env.get_template("usage.html").render(base="/sharedagentsweb", site_title="sharedagents", is_admin=True)

    def route(request):
        url = request.request.url.split("?", 1)[0]
        if "/static/" in url:
            file = STATIC / url.split("/static/", 1)[1]
            if file.is_file():
                if file.name == "usage.js":
                    # Exercise the actual private update functions in the real DOM.
                    script = file.read_text().replace(
                        "\n})();",
                        "\nwindow.usageGate = {upsertProviderCard, applyPaceRows: "
                        "typeof applyPaceRows === 'function' ? applyPaceRows : null};\n})();",
                    )
                    request.fulfill(body=script, content_type="application/javascript")
                else:
                    request.fulfill(path=file, content_type=TYPES.get(file.suffix, "application/octet-stream"))
            else:
                request.fulfill(status=404, body="")
        elif url.endswith("/api/usage/subscriptions/catalog"):
            request.fulfill(
                json={
                    "source": "fixture",
                    "providers": [
                        {
                            "id": card["id"],
                            "name": card.get("name") or card["id"],
                            "kind": card.get("kind")
                            or ("subscription" if card["id"] in sl.PROVIDER_IDS else "balance"),
                        }
                        for card in data["providers"]
                    ],
                }
            )
        elif "/api/usage/subscriptions/" in url:
            card = cards[url.rsplit("/", 1)[1]]
            request.fulfill(json={"provider": card, "updated_at": data["updated_at"], "source": "fixture"})
        elif "/api/" in url:
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
        errors = []

        def open_at(width, height=900, tab="subscriptions"):
            # Phones draw overlay scrollbars; emulate one so widths are true.
            phone = width < 600
            page = browser.new_page(viewport={"width": width, "height": height}, is_mobile=phone, has_touch=phone)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", route)
            page.goto(f"https://usage.example/sharedagentsweb/usage#{tab}")
            page.wait_for_selector(".usage-sub-card[data-use-next='true'] .usage-sub-use-next", state="attached")
            page.wait_for_function("!document.querySelector('.usage-sub-card[data-loading=true]')")
            return page

        yield open_at
        browser.close()
        assert errors == [], errors


# 740-900 px is two cards per row: the narrowest desktop cards, where the
# header pill used to cover the plan name.
@pytest.mark.parametrize("width", [320, 390, 720, 740, 900, 1280])
def test_pace_is_the_first_limit_row(subscriptions_page, width):
    page = subscriptions_page(width)
    page.locator("#usage-subscriptions-grid > details").evaluate_all(
        "cards => cards.forEach(card => { card.open = true; })"
    )
    rows = page.evaluate(BOXES)
    assert len(rows) >= 5
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    for row in rows:
        head, pace, vendor, label, value = row["head"], row["pace"], row["vendor"], row["label"], row["value"]
        assert row["first"] and not row["inHead"], row
        assert pace["top"] >= head["bottom"], row
        # The plan name has the whole title row again: nothing covers or clips it.
        assert row["name"]["right"] <= head["right"] + 1, row
        assert row["nameScroll"] <= row["nameClient"] + 1, row
        # Same looks as the vendor rows under it: width, bar and type.
        assert vendor, row
        assert abs(pace["left"] - vendor["left"]) <= 1 and abs(pace["right"] - vendor["right"]) <= 1, row
        assert abs(row["paceBar"]["height"] - row["vendorBar"]["height"]) <= 0.5, row
        assert row["paceFont"] == row["vendorFont"], row
        assert pace["bottom"] <= vendor["top"], row
        if row["secondVendor"]:
            assert abs((vendor["top"] - pace["bottom"]) - (row["secondVendor"]["top"] - vendor["bottom"])) <= 0.5, row
        # Label at the left, value at the right, on one line.
        assert abs(label["left"] - pace["left"]) <= 1 and abs(value["right"] - pace["right"]) <= 1, row
        assert label["right"] <= value["left"], row
        if row["next"]:
            assert row["next"]["right"] <= value["left"], row
            assert row["next"]["bottom"] <= row["paceBar"]["top"], row
    assert sum(1 for row in rows if row["next"]) == 1


@pytest.mark.parametrize(
    "subscriptions_page,tab", [("eight", "subscriptions"), ("eight-api", "api")], indirect=["subscriptions_page"]
)
@pytest.mark.parametrize("width,height", [(320, 568), (360, 640), (390, 844), (430, 932), (1280, 900)])
def test_eight_collapsed_cards_fit_without_scrolling(subscriptions_page, tab, width, height):
    page = subscriptions_page(width, height, tab=tab)
    grid = "#usage-subscriptions-grid" if tab == "subscriptions" else "#usage-api-grid"
    chart = "#usage-sub-chart-section" if tab == "subscriptions" else "#usage-api-chart-section"
    cards = page.locator(grid + " > details.usage-sub-card")
    assert cards.count() == 8
    assert page.locator(grid + " > details[open]").count() == 0
    boxes = cards.evaluate_all(
        "cards => cards.map(card => { const r = card.getBoundingClientRect(); "
        "return {top: r.top, bottom: r.bottom, height: r.height}; })"
    )
    assert all(box["top"] >= 0 and box["bottom"] <= height for box in boxes), boxes
    assert all(box["height"] >= 44 for box in boxes), boxes
    assert page.locator("#usage-app").evaluate("app => app.scrollTop") == 0
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.wait_for_function("""() => [...document.querySelectorAll('.usage-sub-summary img')]
      .every(img => img.complete && img.naturalWidth > 0)""")
    for card in cards.all():
        logo = card.locator(".usage-sub-summary .usage-sub-logo")
        assert logo.is_visible()
        assert card.locator(".usage-sub-summary-name").is_visible()
        assert card.locator(".usage-sub-summary-pace").is_visible()
        assert card.locator(".usage-sub-head").is_hidden()
        assert card.locator(".usage-sub-windows").is_hidden()
        assert card.locator(".usage-sub-updated").is_hidden()
        if card.get_attribute("data-provider") in sl.PROVIDER_IDS:
            assert card.locator(".usage-sub-summary-name").evaluate("name => name.scrollWidth <= name.clientWidth + 1")
    assert cards.last.bounding_box()["y"] < page.locator(chart).bounding_box()["y"]


@pytest.mark.parametrize(
    "used,loading,label,fill",
    [
        (0, False, "0%", 0),
        (42.4, False, "42%", 42.4),
        (90, False, "90%", 90),
        (150, False, "150%", 100),
        (-5, False, "-5%", 0),
        (None, False, "—", 0),
        (50, True, "…", 0),
    ],
)
def test_collapsed_card_shows_pace_number_and_fill(subscriptions_page, used, loading, label, fill):
    page = subscriptions_page(390)
    _upsert(page, _simple_card("codex", used), loading)
    summary = page.locator(".usage-sub-card[data-provider=codex] > summary")
    assert summary.locator("strong").text_content() == label
    background = summary.evaluate("""summary => {
      const style = getComputedStyle(summary);
      return {fill: parseFloat(style.getPropertyValue('--pace-fill')),
              image: style.backgroundImage, color: style.getPropertyValue('--pace-color').trim()};
    }""")
    assert background["fill"] == fill
    assert "linear-gradient" in background["image"] and f"{fill}%" in background["image"]
    number_color = summary.locator("strong").evaluate("number => getComputedStyle(number).color")
    if used == 90:
        assert background["color"] == "#fab219" and number_color == "rgb(250, 178, 25)"
    elif used == 150:
        assert background["color"] == "#f0524f" and number_color == "rgb(240, 82, 79)"
    # Hover must retain the proportional fill.
    summary.hover()
    assert summary.evaluate("summary => getComputedStyle(summary).backgroundImage") == background["image"]


@pytest.mark.parametrize(
    "tab,provider,windows",
    [
        ("subscriptions", "claude", ["five_hour", "seven_day"]),
        ("api", "openrouter", ["credits"]),
    ],
)
def test_provider_expansion_survives_updates_and_supports_keyboard(subscriptions_page, tab, provider, windows):
    page = subscriptions_page(390, 844, tab=tab)
    card = page.locator(f'.usage-sub-card[data-provider="{provider}"]')
    summary = card.locator("summary")
    summary.click()
    assert card.get_attribute("open") is not None
    assert card.locator(".usage-sub-plan").is_visible()
    for window in windows:
        assert card.locator(f'.usage-sub-window[data-window="{window}"]').is_visible()
    if tab == "subscriptions":
        assert card.locator(".usage-sub-reset").first.is_visible()
    else:
        assert card.locator(".usage-sub-hero-value").is_visible()
    original_windows = card.locator(".usage-sub-windows").inner_text()
    # The refresh replaces the card with loading and completed responses.
    update = "#usage-subscriptions-update" if tab == "subscriptions" else "#usage-api-update"
    page.locator(update).click()
    page.wait_for_selector(update + ":not([disabled])")
    assert card.get_attribute("open") is not None
    assert card.locator(".usage-sub-windows").inner_text() == original_windows
    assert page.locator(".usage-subscriptions-grid > details[open]").count() == 1
    summary.focus()
    summary.press("Enter")
    assert card.get_attribute("open") is None
    assert card.locator(".usage-sub-windows").is_hidden()
    summary.press("Space")
    assert card.get_attribute("open") is not None


@pytest.mark.parametrize(
    "subscriptions_page,provider,pace,hits,winner",
    [
        ("stale_pin", "agy", "—", ["pinned"], "codex"),
        ("unpaced_pool", "cursor", "—", ["hit"], "codex"),
        ("sibling", "cursor", "10%", ["a5", "a7"], "cursor"),
        ("reset", "cursor", "40%", [], "cursor"),
    ],
    indirect=["subscriptions_page"],
)
def test_pool_availability_and_reset_in_the_rendered_card(
    subscriptions_page,
    provider,
    pace,
    hits,
    winner,
):
    page = subscriptions_page(1280)
    card = page.locator(f'.usage-sub-card[data-provider="{provider}"]')
    assert card.locator(".usage-sub-pace-value").text_content().strip() == pace
    assert (
        card.locator(".usage-sub-window").evaluate_all(
            "rows => rows.filter(row => row.querySelector('.usage-sub-hit')).map(row => row.dataset.window)"
        )
        == hits
    )
    assert page.locator(".usage-sub-card").evaluate_all(
        "cards => cards.filter(card => card.querySelector('.usage-sub-use-next')).map(card => card.dataset.provider)"
    ) == [winner]


def _upsert(page, provider, loading=False):
    page.evaluate("([card, loading]) => usageGate.upsertProviderCard(card, loading)", [provider, loading])


def _simple_card(provider_id, used=None):
    card = {
        "id": provider_id,
        "kind": "subscription",
        "ok": True,
        "windows": [{"id": "weekly", "label": "Weekly", "used_percent": 25, "remaining_percent": 75}],
    }
    if used is not None:
        card.update(pace_status="ok", pace_used_percent=used)
    return card


def _clear_pace(page):
    page.evaluate("""() => {
      for (const card of document.querySelectorAll('.usage-sub-card')) {
        usageGate.upsertProviderCard({id: card.dataset.provider, kind: card.dataset.kind, ok: true}, false);
      }
    }""")


def _winners(page):
    tagged = page.locator(".usage-sub-use-next").evaluate_all(
        "tags => tags.map(tag => tag.closest('.usage-sub-card').dataset.provider)"
    )
    flagged = page.locator(".usage-sub-card[data-use-next=true]").evaluate_all(
        "cards => cards.map(card => card.dataset.provider)"
    )
    assert tagged == flagged
    return tagged


def test_incremental_pace_rows_and_loading_transitions(subscriptions_page):
    page = subscriptions_page(740)
    _clear_pace(page)
    codex = page.locator(".usage-sub-card[data-provider=codex]")
    claude = page.locator(".usage-sub-card[data-provider=claude]")
    _upsert(page, _simple_card("codex", 30), loading=True)
    assert codex.locator(".usage-sub-summary-pace").text_content() == "Pace …"
    assert codex.locator(".usage-sub-hero-value").text_content() == "…"
    assert codex.locator(".usage-sub-skel").count() == 2
    assert codex.locator(".usage-sub-pace-row").count() == 0
    _upsert(page, _simple_card("codex", 30))
    assert codex.locator(".usage-sub-summary-pace").text_content() == "Pace 30%"
    assert _winners(page) == ["codex"]
    assert codex.locator(".usage-sub-hero").count() == 0
    _upsert(page, _simple_card("claude", 10))
    assert _winners(page) == ["claude"]
    assert codex.locator(".usage-sub-pace-value").text_content() == "30%"
    # A loaded winner returning to a skeleton cannot keep the recommendation.
    _upsert(page, _simple_card("claude", 10), loading=True)
    assert _winners(page) == ["codex"]
    _upsert(page, _simple_card("claude", 10))
    assert _winners(page) == ["claude"]
    # Losing pace must discard the cached winner, preserve the old hero, and
    # update every other card even though this response has no pace_status.
    _upsert(page, _simple_card("claude"))
    assert claude.locator(".usage-sub-summary-pace").text_content() == "Pace —"
    assert _winners(page) == ["codex"]
    assert claude.locator(".usage-sub-pace-row").count() == 0
    assert claude.locator(".usage-sub-hero-value").text_content() == "75%"
    _upsert(page, _simple_card("codex"))
    assert _winners(page) == []
    assert page.locator("#usage-subscriptions-pace-scale").is_hidden()


def test_pace_reconciliation_restores_missing_and_removes_obsolete_rows(subscriptions_page):
    page = subscriptions_page(740)
    _clear_pace(page)
    _upsert(page, _simple_card("codex", 30))
    _upsert(page, _simple_card("claude", 10))
    codex = page.locator(".usage-sub-card[data-provider=codex]")
    claude = page.locator(".usage-sub-card[data-provider=claude]")
    codex.locator(".usage-sub-pace-row").evaluate("row => row.remove()")
    _upsert(page, _simple_card("claude", 40))
    assert _winners(page) == ["codex"]
    assert codex.locator(".usage-sub-windows > .usage-sub-pace-row:first-child").count() == 1
    _upsert(page, _simple_card("claude"))
    claude.locator(".usage-sub-windows").evaluate("""windows => {
      windows.prepend(document.querySelector('.usage-sub-pace-row').cloneNode(true));
    }""")
    page.evaluate("usageGate.applyPaceRows()")
    assert claude.locator(".usage-sub-pace-row").count() == 0
    assert claude.locator(".usage-sub-hero-value").text_content() == "75%"
    assert _winners(page) == ["codex"]


def _balance_card(provider_id, remaining, limit=None, **fields):
    return {
        "id": provider_id,
        "kind": "balance",
        "ok": True,
        "windows": [
            {"id": "balance", "label": "Balance", "unit": "usd", "remaining": remaining, "limit": limit, **fields}
        ],
    }


@pytest.mark.parametrize(
    "remaining,limit,fields,loading,label,fill",
    [
        (12.34, None, {}, False, "$12.34", 0),
        (25, 100, {}, False, "$25.00", 25),
        (25, 100, {"remaining_percent": 90, "used_percent": 10}, False, "$25.00", 25),
        (10, None, {"used_percent": 80}, False, "$10.00", 20),
        (10, None, {"remaining_percent": 40}, False, "$10.00", 40),
        (0, 100, {}, False, "$0.00", 0),
        (-5, 100, {}, False, "$-5.00", 0),
        (150, 100, {}, False, "$150.00", 100),
        (None, None, {}, False, "—", 0),
        (25, 100, {}, True, "…", 0),
    ],
)
def test_balance_card_shows_amount_and_reported_capacity(
    subscriptions_page,
    remaining,
    limit,
    fields,
    loading,
    label,
    fill,
):
    page = subscriptions_page(390, tab="api")
    _upsert(
        page,
        _balance_card("openrouter", remaining, limit, **fields),
        loading,
    )
    card = page.locator(".usage-sub-card[data-provider=openrouter]")
    assert card.get_attribute("open") is None
    assert card.locator("summary strong").text_content() == label
    assert (
        card.locator("summary").evaluate(
            "summary => parseFloat(getComputedStyle(summary).getPropertyValue('--pace-fill'))"
        )
        == fill
    )
    assert card.locator(".usage-sub-hero-value").text_content() == label
    assert card.locator(".usage-sub-pace-row").count() == 0


def test_usage_tab_order_and_keyboard_navigation(subscriptions_page):
    page = subscriptions_page(390)
    tabs = page.locator(".usage-tabs [role=tab]")
    assert tabs.all_text_contents() == ["Usage", "API", "Subscriptions", "Prices", "Logs"]
    tabs.first.focus()
    for index in range(1, 5):
        page.keyboard.press("Tab")
        assert tabs.nth(index).evaluate("tab => tab === document.activeElement")
    tabs.nth(1).click()
    assert page.locator("#usage-pane-api").is_visible()
    assert tabs.nth(1).get_attribute("aria-selected") == "true"
    assert page.url.endswith("#api")


def test_pace_label_note_and_attributes_are_escaped(subscriptions_page):
    page = subscriptions_page(740)
    payload = '"><img src=x onerror="window.paceXss=1"> & <script>window.paceXss=2</script>'
    card = _simple_card("codex", 120)
    card.update(pace_status="exhausted", pace_limit_hit=True, pace_label=payload, pace_note=payload)
    _upsert(page, card)
    row = page.locator(".usage-sub-card[data-provider=codex] .usage-sub-pace-row")
    assert row.locator(".usage-sub-pace-value").text_content() == payload
    assert row.get_attribute("title") == payload + "\n\n" + page.evaluate("saUsagePace.legendText()")
    assert row.locator(".usage-sub-bar").get_attribute("aria-label") == "Pace " + payload
    assert row.locator("img, script, [onerror]").count() == 0
    summary = page.locator(".usage-sub-card[data-provider=codex] summary")
    assert summary.locator("strong").text_content() == payload
    assert summary.locator("img").count() == 1  # The subscription's existing logo.
    assert summary.locator("script, [onerror]").count() == 0
    assert page.evaluate("window.paceXss === undefined")
