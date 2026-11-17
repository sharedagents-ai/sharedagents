"""Exercise the shipped Prices UI and one-row API using an isolated browser fixture."""

import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from sharedagents_core.layout import pricing_path
from sharedagents_model.usage_pricing import (
    PricingTable,
    _pricing_auto_path_for,
    serialize_pricing_table,
    update_pricing_rates,
    upsert_pricing_entry,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/sharedagents_web/static"


@pytest.fixture
def price_page(pricing_home):
    playwright = pytest.importorskip("playwright.sync_api")
    automatic = _pricing_auto_path_for(pricing_path())
    automatic.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "match": {"id": f"fixture-model-{index:02}", "alias": f"fixture-alias-{index:02}"},
                        "input_per_mtok": 2 if index == 0 else 10 if index == 1 else index + 1,
                        "output_per_mtok": 12 if index == 0 else 20 if index == 1 else (index + 1) * 2,
                        "cached_read_per_mtok": None,
                        "thinking_per_mtok": (index + 1) * 2,
                        "auto": {
                            "upstream_provider": "fixture-provider",
                            "upstream_model_id": f"fixture-model-{index:02}",
                        },
                    }
                    for index in range(80)
                ]
            }
        )
    )
    env = Environment(loader=FileSystemLoader(ROOT / "src/sharedagents_web/templates"), autoescape=True)
    env.globals["asset_rev"] = lambda name: "fixture"
    html = env.get_template("usage.html").render(base="/sharedagentsweb", site_title="sharedagents", is_admin=True)
    writes, errors, fail = [], [], {"save": False}

    def route(request):
        url = request.request.url.split("?", 1)[0]
        if "/static/" in url:
            file = STATIC / url.rsplit("/", 1)[1]
            if file.is_file():
                request.fulfill(
                    path=file, content_type="text/css" if file.suffix == ".css" else "application/javascript"
                )
            else:
                request.fulfill(status=404, body="")
        elif url.endswith("/api/usage/pricing/entries") and request.request.method == "PATCH":
            payload = request.request.post_data_json
            writes.append(payload)
            if fail["save"]:
                request.fulfill(status=500, json={"error": "Fixture save failed"})
            else:
                request.fulfill(json=update_pricing_rates(payload, path=pricing_path()))
        elif url.endswith("/api/usage/pricing"):
            request.fulfill(json=serialize_pricing_table(PricingTable.load(pricing_path())))
        elif url.endswith("/api/usage/pricing/entries") and request.request.method == "POST":
            payload = request.request.post_data_json
            writes.append(payload)
            request.fulfill(status=201, json=upsert_pricing_entry(payload, path=pricing_path()))
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
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/*", route)
        page.goto("https://prices.example/sharedagentsweb/usage#prices")
        page.wait_for_selector("#usage-prices-body tr[data-index]")
        yield page, writes, fail, automatic
        browser.close()
        assert errors == [], errors


def test_filters_sort_numerically_and_paginate(price_page):
    page, writes, fail, automatic = price_page
    expect = pytest.importorskip("playwright.sync_api").expect
    expect(page.locator("#usage-prices-body .usage-price-input")).to_have_count(0)
    expect(page.locator("#usage-prices-body tr[data-index]")).to_have_count(50)
    page.locator("#usage-prices-next").click()
    expect(page.locator("#usage-prices-page")).to_contain_text("Page 2")
    page.locator("#usage-prices-provider").select_option("fixture-provider")
    expect(page.locator("#usage-prices-count")).to_contain_text("of 80 prices")
    page.locator('[data-price-sort="input_per_mtok"]').click()
    cells = page.locator("#usage-prices-body td:nth-child(4)")
    expect(cells.first).to_have_text("$2")
    expect(cells.nth(1)).to_have_text("$3")
    page.locator('[data-price-sort="input_per_mtok"]').click()
    expect(cells.first).to_have_text("$80")
    expect(page.locator('[data-price-sort="input_per_mtok"]').locator("..")).to_have_attribute(
        "aria-sort", "descending"
    )
    page.locator("#usage-prices-search").fill("fixture-alias-00")
    expect(page.locator("#usage-prices-body tr[data-index]")).to_have_count(1)
    expect(page.locator(".usage-price-model")).to_contain_text("fixture-model-00")
    page.locator("#usage-prices-source").select_option("operator")
    expect(page.locator("#usage-prices-body")).to_contain_text("No models match")
    page.locator("#usage-prices-clear").click()
    expect(page.locator("#usage-prices-body tr[data-index]")).to_have_count(50)
    assert writes == []


def test_single_row_edit_keeps_draft_and_saves_only_that_model(price_page):
    page, writes, fail, automatic = price_page
    expect = pytest.importorskip("playwright.sync_api").expect
    before_auto = automatic.read_bytes()
    page.locator("#usage-prices-provider").select_option("fixture-provider")
    page.locator('[data-price-action="edit"]').first.click()
    expect(page.locator(".usage-price-input")).to_have_count(4)
    expect(page.locator("#usage-prices-search")).to_be_disabled()
    expect(page.locator("#usage-prices-add")).to_be_disabled()
    expect(page.locator('[data-price-action="edit"]').first).to_be_disabled()
    inp = page.locator('.is-editing [data-field="input_per_mtok"]')
    inp.fill("2.75")
    page.locator("#usage-refresh").click()
    expect(inp).to_have_value("2.75")
    page.locator('[data-price-action="cancel"]').click()
    assert writes == [] and not pricing_path().exists()
    page.locator('[data-price-action="edit"]').first.click()
    inp.fill("2.75")
    fail["save"] = True
    page.locator('[data-price-action="save"]').click()
    expect(page.locator("#usage-prices-message")).to_have_text("Fixture save failed")
    expect(inp).to_have_value("2.75")
    expect(page.locator('[data-price-action="cancel"]')).to_be_enabled()
    fail["save"] = False
    page.locator('[data-price-action="save"]').click()
    expect(page.locator("#usage-prices-message")).to_have_text("Saved prices for fixture-model-00.")
    expect(page.locator(".usage-price-input")).to_have_count(0)
    assert all(set(payload) == {"key", "rates"} for payload in writes)
    assert writes[-1]["key"]["match"]["id"] == "fixture-model-00"
    assert automatic.read_bytes() == before_auto
    saved = json.loads(pricing_path().read_text())["entries"]
    assert len(saved) == 1 and saved[0]["input_per_mtok"] == 2.75
    expect(page.locator("#usage-prices-provider")).to_have_value("fixture-provider")
    page.locator("#usage-prices-source").select_option("operator")
    expect(page.locator(".usage-price-source")).to_have_text("Custom")


def test_default_is_editable_and_mobile_scroll_stays_inside_the_table(price_page):
    page, writes, fail, automatic = price_page
    expect = pytest.importorskip("playwright.sync_api").expect
    page.locator('tr.is-default [data-price-action="edit"]').click()
    page.locator('.is-editing [data-field="output_per_mtok"]').fill("1.75")
    page.locator('[data-price-action="save"]').click()
    expect(page.locator("#usage-prices-message")).to_have_text("Saved prices for Unpriced models.")
    assert writes[-1]["key"]["kind"] == "default"
    assert not json.loads(pricing_path().read_text()).get("entries")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert page.locator(".usage-prices-scroll").evaluate("el => el.scrollWidth > el.clientWidth")
    page.locator("#usage-prices-add").click()
    expect(page.locator("#price-entry-editor")).to_be_visible()
    expect(page.locator('[data-price-action="edit"]').first).to_be_disabled()
    page.locator("#price-entry-cancel").click()
    expect(page.locator('[data-price-action="edit"]').first).to_be_enabled()


def test_add_price_locks_edits_and_creates_one_custom_row(price_page):
    page, writes, fail, automatic = price_page
    expect = pytest.importorskip("playwright.sync_api").expect
    before_auto = automatic.read_bytes()
    page.locator("#usage-prices-add").click()
    expect(page.locator('[data-price-action="edit"]').first).to_be_disabled()
    page.locator("#price-entry-provider").fill("fixture-provider")
    page.locator("#price-entry-id").fill("new-custom-model")
    page.locator("#price-entry-input").fill("1.25")
    page.locator("#price-entry-output").fill("4.5")
    page.locator("#price-entry-save").click()
    expect(page.locator("#usage-prices-message")).to_have_text("Added price for new-custom-model.")
    expect(page.locator("#price-entry-editor")).to_be_hidden()
    expect(page.locator("#usage-prices-search")).to_be_enabled()
    page.locator("#usage-prices-search").fill("new-custom-model")
    expect(page.locator("#usage-prices-body tr[data-index]")).to_have_count(1)
    expect(page.locator(".usage-price-source")).to_have_text("Custom")
    assert len(writes) == 1
    assert len(json.loads(pricing_path().read_text())["entries"]) == 1
    assert automatic.read_bytes() == before_auto
