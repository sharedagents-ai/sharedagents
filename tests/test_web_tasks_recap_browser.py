"""Exercise the shipped Tasks JS/CSS in Chromium without production side calls."""

import json
from pathlib import Path

import pytest

_playwright = pytest.importorskip("playwright.sync_api")
sync_playwright = _playwright.sync_playwright


def test_recap_collapses_persists_on_poll_and_escapes_model_text():
    root = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web" / "static"
    hostile = '<img src=x onerror="window.injected=true"> & "model text"'
    item = {
        "id": "row",
        "session_id": "parent",
        "title": "Operator task",
        "agent": "build",
        "state": "idle",
        "card": "needs_input",
        "summary": hostile,
        "summary_follow_up": hostile,
        "summary_verdict": "follow_up",
        "summary_status": "done",
        "activity": {"display": {"label": "Ready"}},
    }
    html = """<meta name="sharedagents-base" content="">
      <main id="tasks-board"></main><div id="tasks-empty"></div>
      <span id="tasks-updated"></span><input id="tasks-topic-filter">
      <datalist id="tasks-topic-options"></datalist><button id="tasks-new-toggle"></button>
      <form id="tasks-new-form" hidden><input id="tasks-new-title">
      <select id="tasks-new-topic"></select><button id="tasks-new-cancel"></button></form>
      <input type="checkbox" id="tasks-show-archived">"""
    polls = []

    def route(request):
        if "/api/tasks" in request.request.url:
            polls.append(1)
            request.fulfill(content_type="application/json", body=json.dumps({"items": [item]}))
        else:
            request.fulfill(content_type="text/html", body=html)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.route("http://tasks.review/**", route)
            page.goto("http://tasks.review/tasks")
            page.add_style_tag(content=(root / "tasks.css").read_text())
            page.add_script_tag(content=(root / "share.js").read_text())
            page.add_script_tag(content=(root / "tasks.js").read_text())
            toggle = page.locator(".task-pill-recap")
            toggle.wait_for()
            panel = page.locator(".task-summary-panel")
            assert panel.is_hidden()
            page.wait_for_timeout(5200)
            assert len(polls) >= 2 and panel.is_hidden()
            assert toggle.get_attribute("aria-expanded") == "false"
            toggle.click()
            assert panel.is_visible()
            assert page.url == "http://tasks.review/tasks"
            assert hostile in panel.inner_text()
            assert page.locator(".task-row img").count() == 0
            assert not page.evaluate("window.injected || false")
            before = len(polls)
            page.wait_for_timeout(5200)
            assert len(polls) > before and panel.is_visible()
            assert toggle.get_attribute("aria-expanded") == "true"
            assert page.locator('.task-card[data-card="needs_input"] .task-row').count() == 1
            assert page.locator('.task-card[data-card="completed"] .task-row').count() == 0
            assert page.locator(".task-pill-status").inner_text() == "Needs input"
            toggle.click()
            assert panel.is_hidden()
        finally:
            browser.close()
