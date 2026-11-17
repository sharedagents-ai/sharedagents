"""Usage > Subscriptions keeps sign-in alerts, versions, and Check/Update
actions on the subscription cards against the shipped usage.js.

The external-agent inventory is not shown in the Subscriptions tab.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/sharedagents_web/static"


def _health_rows():
    def row(agent, provider, version, state, *, needs=False, hint="", update="hermes update", auto=False, spend=""):
        return {
            "agent": agent,
            "kind": agent + "_cli",
            "display_name": agent.title(),
            "cli": agent,
            "path": "/usr/bin/" + agent,
            "installed": True,
            "adapter": "cli",
            "provider": provider,
            "version": version,
            "authenticated": state == "ok",
            "auth_state": state,
            "needs_auth": needs,
            "auth_hint": hint,
            "auto_auth": auto,
            "install_cmd": "npm install -g demo",
            "update_cmd": update,
            "last_spend": spend,
            "checked_at": 1,
        }

    return [
        row("antigravity", "agy", "1.5.7", "missing", needs=True, hint="antigravity", update="antigravity update"),
        row("claude", "claude", "2.1.322", "ok"),
        row("codex", "codex", "0.167.1", "ok", update="npm install -g @openai/codex@latest"),
        row("cursor", "cursor", "2026.11.11-b912975", "ok", auto=True),
        row("devin", "devin", "3000.12.32", "ok"),
        row("hermes", "", "0.22.5", "ok"),
        row("kimi", "", "0.40.1", "ok", auto=True),
        row("mimo", "mimo", "", "ok", update=""),
        row("opencode", "opencode", "2.2.0", "ok"),
        row("pi", "", "1.5.0", "ok"),
        row("qwen", "", "0.27.0", "ok"),
        row("zcode", "zcode", "3.17.2-45", "ok"),
    ]


CATALOG = {
    "providers": [
        {"id": "codex", "name": "Codex", "kind": "subscription", "version": "0.167.1"},
        {
            "id": "agy",
            "name": "Antigravity",
            "kind": "subscription",
            "version": "1.5.7",
            "auth_required": True,
            "auth_hint": "antigravity",
        },
    ]
}

AGY_CARD = {
    "id": "agy",
    "name": "Antigravity",
    "kind": "subscription",
    "plan": "authentication required",
    "ok": False,
    "error": "Not signed in — run `antigravity` in a terminal to authenticate",
    "windows": [],
    "auth_required": True,
    "auth_hint": "antigravity",
    "version": "1.5.7",
    "updated_at": "2026-11-12T21:00:00Z",
}
CODEX_CARD = {
    "id": "codex",
    "name": "Codex",
    "kind": "subscription",
    "plan": "Plus",
    "ok": True,
    "windows": [{"id": "five_hour", "label": "5-hour", "used_percent": 37.0, "remaining_percent": 63.0}],
    "updated_at": "2026-11-12T21:00:00Z",
}


@pytest.fixture
def health_page(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    playwright = pytest.importorskip("playwright.sync_api")
    env = Environment(loader=FileSystemLoader(ROOT / "src/sharedagents_web/templates"), autoescape=True)
    env.globals["asset_rev"] = lambda name: "fixture"
    html = env.get_template("usage.html").render(base="/sharedagentsweb", site_title="sharedagents", is_admin=True)
    calls = {"auth_check": [], "auth": [], "update": []}
    state = {"signed_in": False}
    errors = []

    def route(request):
        url = request.request.url.split("?", 1)[0]
        method = request.request.method
        if "/static/" in url:
            file = STATIC / url.split("/static/", 1)[1]
            if file.is_file():
                request.fulfill(
                    body=file.read_bytes(),
                    content_type=(mimetypes.guess_type(file.name)[0] or "application/octet-stream") + "; charset=utf-8",
                )
            else:
                request.fulfill(status=404, body="")
        elif url.endswith("/api/usage/subscriptions/catalog"):
            request.fulfill(json=CATALOG)
        elif url.endswith("/api/usage/subscriptions/codex"):
            request.fulfill(json={"updated_at": CODEX_CARD["updated_at"], "source": "live", "provider": CODEX_CARD})
        elif url.endswith("/api/usage/subscriptions/agy"):
            card = dict(AGY_CARD)
            if state["signed_in"]:
                card.update(
                    {
                        "ok": True,
                        "plan": "Ultra",
                        "error": None,
                        "auth_required": False,
                        "windows": [
                            {"id": "weekly", "label": "Weekly", "used_percent": 12.0, "remaining_percent": 88.0}
                        ],
                    }
                )
            calls.setdefault("providers", []).append({"id": "agy", "method": method})
            request.fulfill(json={"updated_at": card["updated_at"], "source": "live", "provider": card})
        elif url.endswith("/api/usage/agents/health"):
            rows = _health_rows()
            if state["signed_in"]:
                for item in rows:
                    if item["agent"] == "antigravity":
                        item.update({"auth_state": "ok", "authenticated": True, "needs_auth": False, "auth_hint": ""})
            request.fulfill(json={"updated_at": "2026-11-12T21:00:00Z", "agents": rows, "alerts": []})
        elif url.endswith("/auth-check") and "/api/usage/agents/" in url:
            agent = url.rsplit("/", 2)[1]
            calls["auth_check"].append(agent)
            row = next(r for r in _health_rows() if r["agent"] == agent)
            if agent == "antigravity" and state["signed_in"]:
                row = dict(row, auth_state="ok", authenticated=True, needs_auth=False, auth_hint="")
            request.fulfill(json=row)
        elif url.endswith("/update") and "/api/usage/agents/" in url and method == "POST":
            agent = url.rsplit("/", 2)[1]
            calls["update"].append(agent)
            request.fulfill(
                status=202,
                json={
                    "job_id": "job1",
                    "agent": agent,
                    "command": "npm install -g @openai/codex@latest",
                    "status": "running",
                    "output": "",
                    "exit_code": None,
                },
            )
        elif "/api/usage/agents/update/" in url:
            request.fulfill(
                json={
                    "job_id": "job1",
                    "agent": "codex",
                    "command": "npm install -g @openai/codex@latest",
                    "status": "ok",
                    "output": "changed 1 package",
                    "exit_code": 0,
                }
            )
        elif url.endswith("/api/usage/agents/auth") or url.endswith("/auth"):
            calls["auth"].append(url.rsplit("/", 2)[1])
            request.fulfill(json={"attempted": True, "ok": True, "detail": "signed in"})
        elif "/api/" in url:
            request.fulfill(json={"history": []} if "pace-history" in url else {})
        else:
            request.fulfill(body=html.encode("utf-8"), content_type="text/html; charset=utf-8")

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
        page.goto("https://usage.example/sharedagentsweb/usage#subscriptions")
        yield page, calls, state, errors
        browser.close()
        assert errors == [], errors


def test_subscriptions_exclude_external_agents_group(health_page):
    page, _calls, _state, _errors = health_page
    expect = pytest.importorskip("playwright.sync_api").expect
    expect(page.locator("#usage-group-subscriptions")).to_be_visible()
    expect(page.locator("#usage-subscriptions-grid .usage-sub-card")).to_have_count(2)
    expect(page.locator("#usage-group-agents")).to_have_count(0)
    expect(page.locator("#usage-agents-grid")).to_have_count(0)
    expect(page.get_by_role("button", name="Re-scan", exact=True)).to_have_count(0)


def test_signed_out_subscription_card_carries_auth_alert_and_version(health_page):
    page, _calls, _state, _errors = health_page
    expect = pytest.importorskip("playwright.sync_api").expect
    agy = page.locator('.usage-sub-card[data-provider="agy"]')
    expect(agy.locator(".usage-sub-auth")).to_contain_text("Authentication required")
    expect(agy.locator(".usage-sub-auth-cmd")).to_have_text("antigravity")
    expect(agy.locator(".usage-sub-version")).to_have_text("v1.5.7")
    codex = page.locator('.usage-sub-card[data-provider="codex"]')
    expect(codex.locator(".usage-sub-version")).to_have_text("v0.167.1")
    expect(codex.locator(".usage-sub-auth")).to_have_count(0)


def test_update_button_posts_job_and_reports_result(health_page):
    page, calls, _state, _errors = health_page
    expect = pytest.importorskip("playwright.sync_api").expect
    card = page.locator('.usage-sub-card[data-provider="codex"]')
    card.locator(".usage-sub-summary").click()  # subscription cards start collapsed
    card.locator("[data-agent-update]").click()
    expect(card.locator("[data-agent-update-status]")).to_have_text("Updated", timeout=15000)
    assert calls["update"] == ["codex"]


def test_check_again_after_sign_in_clears_the_alert(health_page):
    page, calls, state, _errors = health_page
    expect = pytest.importorskip("playwright.sync_api").expect
    card = page.locator('.usage-sub-card[data-provider="agy"]')
    expect(card.locator(".usage-sub-auth")).to_have_count(1)
    # The operator signs in in a terminal; the Check button re-probes and the
    # alert clears in the subscription card.
    state["signed_in"] = True
    card.locator(".usage-sub-summary").click()  # subscription cards start collapsed
    card.locator("[data-agent-recheck]").click()
    expect(card.locator(".usage-sub-auth")).to_have_count(0, timeout=5000)
    expect(card.locator(".usage-sub-plan")).to_have_text("Ultra")
    expect(card.locator(".usage-sub-error")).to_have_count(0)
    expect(card.locator(".usage-sub-window")).to_contain_text("Weekly")
    assert calls["auth_check"] == ["antigravity"]
    assert {"id": "agy", "method": "POST"} in calls["providers"]


PICKER_HTML = """<!doctype html><html><head>
<meta name="sharedagents-base" content="/sharedagentsweb"></head><body>
<div id="host"></div>
<script src="/sharedagentsweb/static/agent_model_catalog.js"></script>
<script>
window.AgentModelCatalogView.renderAgents({
  host: document.getElementById("host"),
  current: "",
  agents: [
    { name: "antigravity", origin: "native", external: true, version: "1.5.7",
      auth_state: "missing", auth_hint: "antigravity", installed: true },
    { name: "claude", origin: "native", external: true, version: "2.1.322",
      auth_state: "ok", installed: true }
  ]
});
</script></body></html>"""


def test_chat_picker_rows_show_version_and_auth_alert():
    """The chat agent selector (shared catalog renderer) instructs auth on
    the card itself: version chip + a needs-authentication alert."""
    playwright = pytest.importorskip("playwright.sync_api")
    errors = []
    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except playwright.Error as error:
            if "Executable doesn't exist" in str(error):
                pytest.skip("Playwright Chromium is not installed")
            raise
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route(request):
            url = request.request.url
            if "/static/" in url:
                # Keep the sub-directory (agent-logos/, provider-logos/) so
                # logo <img> requests resolve instead of crashing the handler.
                file = STATIC / url.split("?", 1)[0].split("/static/", 1)[1]
                if not file.is_file():
                    request.fulfill(status=404, body="")
                    return
                # body+charset: fulfill(path=) omits the charset and the
                # browser would decode the module's UTF-8 glyphs as latin-1.
                ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                request.fulfill(body=file.read_bytes(), content_type=f"{ctype}; charset=utf-8")
            else:
                request.fulfill(body=PICKER_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")

        page.route("**/*", route)
        page.goto("https://picker.example/sharedagentsweb/chat")
        expect = playwright.expect
        agy = page.locator('.agent-row[data-agent="antigravity"]')
        expect(agy.locator(".agent-row-auth")).to_contain_text("needs authentication")
        expect(agy.locator(".agent-row-auth")).to_have_attribute("title", "Authentication required — run: antigravity")
        expect(agy.locator(".agent-row-version")).to_have_text("v1.5.7")
        claude = page.locator('.agent-row[data-agent="claude"]')
        expect(claude.locator(".agent-row-auth")).to_have_count(0)
        expect(claude.locator(".agent-row-version")).to_have_text("v2.1.322")
        browser.close()
        assert errors == [], errors
