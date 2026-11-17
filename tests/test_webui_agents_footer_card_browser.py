"""Agents footer opens a card like the agent selector instead of expanding
inline in the chat ("when i click the agents footer in the
chat session i need u to open a card like the agent selector instead of
expanding in the chat").

Drives the REAL footer markup (cut from index.html), style.css and
agent_mon_mini.js in Chromium: clicking the footer opens #agent-mon-pop as a
fixed sheet, the chat transcript and the composer keep their height, and
Escape / × / backdrop close it.
"""

from pathlib import Path

import pytest

_playwright = pytest.importorskip("playwright.sync_api")
sync_playwright = _playwright.sync_playwright

ROOT = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web"
STATIC = ROOT / "static"
TEMPLATE = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


def _footer_markup() -> str:
    start = TEMPLATE.index('<div id="agent-mon-acc"')
    end = TEMPLATE.index('<div id="ama-body"', start)
    end = TEMPLATE.index("</div>\n        </div>\n", end) + len("</div>\n        </div>\n")
    return TEMPLATE[start:end].replace("{{ base }}", "")


def _page_html() -> str:
    return (
        '<!doctype html><meta name="sharedagents-base" content="">'
        '<body style="margin:0;height:100vh;display:flex;flex-direction:column">'
        '<div id="messages" style="flex:1 1 auto;min-height:0">chat</div>'
        '<div class="composer-card" style="flex:0 0 auto">'
        '<textarea id="message-input"></textarea>' + _footer_markup() + "</div></body>"
    )


def _geom(page):
    return page.evaluate(
        """() => {
          const h = (s) => document.querySelector(s).getBoundingClientRect().height;
          const pop = document.getElementById('agent-mon-pop');
          const r = pop.getBoundingClientRect();
          return {
            messages: h('#messages'), composer: h('.composer-card'),
            popHidden: pop.hidden,
            backdropHidden: document.getElementById('ama-backdrop').hidden,
            expanded: document.querySelector('#agent-mon-acc .ama-summary')
              .getAttribute('aria-expanded'),
            position: getComputedStyle(pop).position,
            pop: {left: r.left, right: r.right, bottom: r.bottom, width: r.width},
            rows: document.querySelectorAll('#agent-mon-pop #ama-body .ama-row').length,
            focus: document.activeElement && document.activeElement.className,
          };
        }"""
    )


@pytest.mark.parametrize("viewport", [(1440, 900), (390, 844)])
def test_footer_click_opens_card_without_moving_chat_and_escape_closes(viewport):
    width, height = viewport
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.route(
                "http://footer.card/**", lambda route: route.fulfill(content_type="text/html", body=_page_html())
            )
            page.goto("http://footer.card/")
            page.add_style_tag(content=(STATIC / "style.css").read_text())
            page.add_script_tag(content=(STATIC / "state.js").read_text())
            page.add_script_tag(content=(STATIC / "agent_mon_mini.js").read_text())
            page.evaluate(
                "() => window.SaAgentMon.update({active: true, hash: 'S1', "
                "agent: 'jarvis', title: 'chat', live: false})"
            )
            before = _geom(page)
            assert before["popHidden"] and before["backdropHidden"]
            assert before["expanded"] == "false"

            page.click("#agent-mon-acc .ama-summary .ama-title")
            opened = _geom(page)
            assert not opened["popHidden"] and not opened["backdropHidden"]
            assert opened["expanded"] == "true"
            assert opened["position"] == "fixed", "the card floats like the agent selector"
            assert opened["rows"] >= 1, "rows paint inside the card"
            assert opened["messages"] == before["messages"], "the chat must not shrink"
            assert opened["composer"] == before["composer"], "the footer must not expand"
            assert opened["pop"]["bottom"] <= height + 0.5
            if width <= 520:
                assert opened["pop"]["width"] == width, "phones get a bottom sheet"
            assert page.evaluate("() => window.SaAgentMon.isOpen()") is True

            page.keyboard.press("Escape")
            closed = _geom(page)
            assert closed["popHidden"] and closed["backdropHidden"]
            assert closed["expanded"] == "false"
            assert "ama-summary" in (closed["focus"] or ""), "focus returns to the footer"

            page.click("#agent-mon-acc .ama-summary .ama-title")
            page.click("#ama-pop-close")
            assert _geom(page)["popHidden"]

            # The Active filter lives in the card header, next to the list
            # it filters: toggling it keeps the card open.
            page.click("#agent-mon-acc .ama-summary .ama-title")
            page.click("#ama-filter")
            assert not _geom(page)["popHidden"], "the filter must not close the card"
            assert page.get_attribute("#ama-filter", "aria-pressed") == "true"
            page.click("#ama-filter")
            assert page.get_attribute("#ama-filter", "aria-pressed") == "false"
            assert errors == []
        finally:
            browser.close()


@pytest.fixture
def card_page():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.route("http://footer.card/**", lambda route: route.fulfill(content_type="text/html", body=_page_html()))
        page.goto("http://footer.card/")
        page.add_style_tag(content=(STATIC / "style.css").read_text())
        page.add_script_tag(content=(STATIC / "state.js").read_text())
        try:
            yield page
        finally:
            browser.close()


def test_escape_dismisses_card_without_cancelling_stream(card_page):
    page = card_page
    # Bind the real application shortcuts alongside the real card handler.
    # Only unrelated controls/callbacks are stubbed; app bootstrap is not run.
    page.evaluate("""() => {
      const controls = document.createElement('div');
      controls.hidden = true;
      for (const id of ['send-btn', 'new-chat-btn', 'toggle-sidebar',
                        'sidebar-backdrop', 'agent-select', 'session-list',
                        'model-select', 'settings-btn', 'close-settings',
                        'settings-modal']) {
        const node = document.createElement('button');
        node.id = id;
        controls.appendChild(node);
      }
      document.body.appendChild(controls);
      window.toggleSidebar = () => {};
      window.closeSidebarMobile = () => {};
      window.cancelCalls = 0;
      window.cancelStreaming = () => { window.cancelCalls++; };
    }""")
    page.add_script_tag(content=(STATIC / "app.js").read_text())
    page.evaluate("setupEventListeners(); state.isStreaming = true;")
    page.add_script_tag(content=(STATIC / "agent_mon_mini.js").read_text())
    page.click(".ama-summary .ama-title")
    page.keyboard.press("Escape")
    assert page.locator("#agent-mon-pop").is_hidden()
    assert page.evaluate("window.cancelCalls") == 0, "dismiss must not abort a turn"
    page.keyboard.press("Escape")
    assert page.evaluate("window.cancelCalls") == 1, "closed card preserves the shortcut"


def test_tab_stays_in_card_when_snapshot_removes_focused_stop(card_page):
    page = card_page
    page.add_script_tag(content=(STATIC / "agent_mon_mini.js").read_text())
    page.evaluate("""() => {
      window.reviewRows = [
        {hash: 'P1', agent: 'jarvis', role: 'current', state: 'running', live: true},
        {hash: 'C1', parent_hash: 'P1', agent: 'build', role: 'sub',
         state: 'running', live: true, depth: 1},
      ];
      SaAgentMon.update({active: true, hash: 'P1', agent: 'jarvis',
                          live: true, stateLabel: 'running'});
      SaAgentMon.applySnapshot({rows: reviewRows});
    }""")
    page.click(".ama-summary .ama-title")
    page.wait_for_function("document.activeElement.id === 'ama-pop-close'")
    page.locator(".ama-stop").focus()
    page.evaluate("SaAgentMon.applySnapshot({rows: reviewRows.slice(0, 1)})")
    page.keyboard.press("Tab")
    assert page.evaluate("document.getElementById('agent-mon-pop').contains(document.activeElement)"), (
        "snapshot repaint must not let Tab reach the background composer"
    )


def test_footer_label_renders_backend_active_and_total_counts(card_page):
    """ "i miss showing the number of active subagents in
    the agents footer". The label renders the payload's backend ``counts``
    verbatim — it shows the active number even when the rows list fewer
    entries, proving the page does not count rows itself."""
    page = card_page
    page.add_script_tag(content=(STATIC / "agent_mon_mini.js").read_text())

    def label():
        return page.evaluate("""() => ({
          footer: document.getElementById('ama-count').textContent,
          card: document.getElementById('ama-pop-count').textContent,
          active: [...document.querySelectorAll('#ama-count .ama-count-active')]
            .map((n) => [n.textContent, getComputedStyle(n).fontWeight]),
        })""")

    page.evaluate("""() => {
      SaAgentMon.update({active: true, hash: 'P1', agent: 'jarvis',
                          live: true, stateLabel: 'waiting'});
      SaAgentMon.applySnapshot({footer: {
        rows: [
          {hash: 'P1', agent: 'jarvis', role: 'current', state: 'waiting', live: true},
          {hash: 'C1', parent_hash: 'P1', agent: 'build', role: 'sub',
           state: 'tool_running', live: true, depth: 1},
        ],
        subs: 17, counts: {active: 3, total: 17, agents: 1},
      }});
    }""")
    got = label()
    assert got["footer"] == "3 active · 17 subs", got
    assert got["card"] == got["footer"], "card header mirrors the footer"
    assert got["active"] == [["3 active", "700"]], "active part is accent styled"

    # Children finished: the next state frame says 0 active.
    page.evaluate("""() => SaAgentMon.applySnapshot({footer: {
      rows: [
        {hash: 'P1', agent: 'jarvis', role: 'current', state: 'idle', live: false},
        {hash: 'C1', parent_hash: 'P1', agent: 'build', role: 'sub',
         state: 'idle', live: false, depth: 1},
      ],
      subs: 17, counts: {active: 0, total: 17, agents: 1},
    }})""")
    got = label()
    assert got["footer"] == "17 subs", got
    assert got["active"] == []

    # Child view: ancestors are agent lanes.
    page.evaluate("""() => SaAgentMon.applySnapshot({
      rows: [{hash: 'P1', agent: 'jarvis', role: 'parent', state: 'idle'}],
      subs: 1, counts: {active: 1, total: 1, agents: 2},
    })""")
    assert label()["footer"] == "2 agents · 1 active · 1 sub"
