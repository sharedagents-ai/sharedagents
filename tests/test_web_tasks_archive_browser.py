"""Exercise selection, archiving and scroll retention in the shipped Tasks UI."""

import json
import re
from pathlib import Path

import pytest

_playwright = pytest.importorskip("playwright.sync_api")
expect, sync_playwright = _playwright.expect, _playwright.sync_playwright


@pytest.fixture()
def board(request):
    viewport = getattr(request, "param", {"width": 1440, "height": 800})
    state = {
        "items": [],
        "archives": [],
        "polls": 0,
        "fail_archive": False,
        "revision": 0,
        "selected_model": "",
        "shares": [],
        "fail_model": False,
    }
    for card in ("needs_input", "working", "completed"):
        for i in range(32):
            complete = card == "completed"
            state["items"].append(
                {
                    "id": f"{card}-{i}",
                    "session_id": f"session-{card}-{i}",
                    "title": f"Task {card} {i:02d}",
                    "card": card,
                    "state": "streaming" if card == "working" else "idle",
                    "topic": "alpha" if i < 16 else "beta",
                    "summary": "Finished task" if complete else "More work remains",
                    "summary_verdict": "complete" if complete else "follow_up",
                    "summary_status": "done",
                    "recap_archivable": complete,
                }
            )
    html = """<meta name="sharedagents-base" content="">
      <body class="monitor-body"><div id="tasks-app">
      <header class="mon-header"><input id="tasks-topic-filter">
      <datalist id="tasks-topic-options"></datalist><button id="tasks-new-toggle"></button>
      <span id="tasks-updated"></span><input type="checkbox" id="tasks-show-archived"></header>
      <button id="tasks-summary-model">Default model</button>
      <span id="tasks-summary-model-status"></span><div id="share-pop" class="share-pop tasks-share-pop" hidden></div>
      <form id="tasks-new-form" hidden><input id="tasks-new-title">
      <select id="tasks-new-topic"></select><button id="tasks-new-cancel"></button></form>
      <main id="tasks-board" class="tasks-board"></main><div id="tasks-empty"></div>
      </div></body>"""

    def route(route):
        req = route.request
        if req.url.endswith("/api/tasks/summary-model"):
            if req.method == "PUT":
                if state["fail_model"]:
                    route.fulfill(status=503, content_type="application/json", body="{}")
                    return
                state["selected_model"] = req.post_data_json["model"]
            route.fulfill(
                content_type="application/json",
                body=json.dumps(
                    {
                        "selected": state["selected_model"],
                        "models": [{"id": "careful", "label": "Careful model"}],
                    }
                ),
            )
        elif req.url.endswith("/api/models"):
            route.fulfill(json={"models": [{"name": "jarvis"}, {"name": "codex"}]})
        elif req.url.endswith("/api/models/discover"):
            assert req.post_data_json["agent"] == "jarvis"
            route.fulfill(
                json={
                    "models": [
                        {"id": "wire-careful", "alias": "careful", "label": "Careful model", "provider": "vllm"},
                        {"id": "unsupported", "provider": "codex"},
                    ]
                }
            )
        elif req.url.endswith("/sharing"):
            if req.method == "PUT":
                state["shares"].append(req.post_data_json)
                # The server stamps the session's sharing on its task rows.
                sid = req.url.rsplit("/", 2)[-2]
                for it in state["items"]:
                    if it["session_id"] == sid:
                        it["sharing"] = {
                            **req.post_data_json,
                            "members": [{"email": m, "name": "Member"} for m in req.post_data_json.get("members", [])],
                        }
            body = state["shares"][-1] if state["shares"] else {"visibility": "private", "members": []}
            route.fulfill(
                content_type="application/json",
                body=json.dumps(
                    {
                        **body,
                        "can_manage": True,
                        "owner_email": "owner@sharedagents.test",
                    }
                ),
            )
        elif req.url.endswith("/api/users/directory"):
            route.fulfill(
                content_type="application/json",
                body=json.dumps(
                    {
                        "users": [
                            {"email": "member@sharedagents.test", "name": "Member"},
                        ]
                    }
                ),
            )
        elif req.url.endswith("/api/tasks/archive"):
            ids = req.post_data_json["ids"]
            state["archives"].append(ids)
            if state["fail_archive"]:
                route.fulfill(status=503, content_type="application/json", body="{}")
                return
            state["items"][:] = [it for it in state["items"] if it["id"] not in ids]
            route.fulfill(
                content_type="application/json", body=json.dumps({"ok": True, "archived_ids": ids, "skipped_ids": []})
            )
        elif "/api/tasks" in req.url:
            state["polls"] += 1
            route.fulfill(content_type="application/json", body=json.dumps({"items": state["items"]}))
        else:
            route.fulfill(content_type="text/html", body=html)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport=viewport)
            page.route("http://tasks.archive/**", route)
            page.goto("http://tasks.archive/tasks")
            root = Path(__file__).resolve().parents[1] / "src/sharedagents_web/static"
            for name in ("style.css", "agents.css", "tasks.css"):
                page.add_style_tag(content=(root / name).read_text())
            page.evaluate("window.confirmCard = async () => true")
            page.add_script_tag(content=(root / "share.js").read_text())
            for name in ("agent_model_catalog.js", "agent_model_picker.js", "agent_model_dialog.js"):
                page.add_script_tag(content=(root / name).read_text())
            page.add_script_tag(content=(root / "tasks.js").read_text())
            expect(page.locator(".task-row")).to_have_count(96)
            settle(page)
            yield page, state
        finally:
            browser.close()


def settle(page):
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


def refresh(page, state):
    state["revision"] += 1
    title = f"Task refreshed {state['revision']}"
    state["items"][0]["title"] = title
    page.evaluate("document.dispatchEvent(new CustomEvent('sharedagents:tasks-refresh'))")
    expect(page.locator('[data-id="needs_input-0"] .task-title')).to_have_text(title)
    settle(page)


@pytest.mark.parametrize(
    "board",
    [
        {"width": 1440, "height": 800},
        {"width": 600, "height": 800},
        {"width": 390, "height": 844},
    ],
    indirect=True,
    ids=["desktop-columns", "tablet-page", "phone-page"],
)
def test_scroll_position_stays_fixed_across_updates_and_poll(board):
    page, state = board
    page.evaluate("""() => {
        if (window.innerWidth >= 900) {
            document.querySelectorAll('.task-card-body').forEach((body, i) => body.scrollTop = 160 + i * 40);
        } else window.scrollTo(0, 520);
    }""")
    settle(page)
    measure = """() => ({page: window.scrollY, board: document.querySelector('#tasks-board').scrollTop,
      cards: Array.from(document.querySelectorAll('.task-card-body'), body => body.scrollTop)})"""
    before = page.evaluate(measure)
    assert before["page"] > 100 or all(value > 100 for value in before["cards"])
    refresh(page, state)
    assert page.evaluate(measure) == before
    polls = state["polls"]
    page.wait_for_timeout(5250)
    assert state["polls"] > polls
    settle(page)
    assert page.evaluate(measure) == before


def test_select_many_survives_refresh_then_archive_and_select_all(board):
    page, state = board
    expect(page.locator(".task-select")).to_have_count(32)
    page.locator(".task-select").nth(0).check()
    page.locator(".task-select").nth(1).check()
    assert page.locator(".task-select-all").evaluate("el => el.indeterminate")
    refresh(page, state)
    expect(page.locator(".task-select:checked")).to_have_count(2)
    page.locator(".task-archive-selected").click()
    expect(page.locator(".task-select")).to_have_count(30)
    assert state["archives"] == [["completed-0", "completed-1"]]
    assert page.url == "http://tasks.archive/tasks"
    page.locator(".task-select-all").check()
    expect(page.locator(".task-select:checked")).to_have_count(30)
    page.locator(".task-archive-selected").click()
    expect(page.locator(".task-select")).to_have_count(0)
    assert len(state["archives"][1]) == 30
    expect(page.locator('.task-card[data-card="needs_input"] .task-row')).to_have_count(32)
    expect(page.locator('.task-card[data-card="working"] .task-row')).to_have_count(32)


def test_select_all_respects_topic_filter_and_failed_archive_keeps_selection(board):
    page, state = board
    page.locator("#tasks-topic-filter").fill("alpha")
    page.locator(".task-select-all").check()
    expect(page.locator(".task-select:checked")).to_have_count(16)
    state["fail_archive"] = True
    page.locator(".task-archive-selected").click()
    expect(page.locator(".task-archive-message")).to_have_text("Archiving failed. Try again.")
    expect(page.locator(".task-select:checked")).to_have_count(16)
    state["fail_archive"] = False
    page.locator(".task-archive-selected").click()
    expect(page.locator(".task-select")).to_have_count(0)
    assert all(int(row_id.rsplit("-", 1)[1]) < 16 for row_id in state["archives"][-1])
    page.locator("#tasks-topic-filter").fill("")
    expect(page.locator(".task-select")).to_have_count(16)


def test_current_turn_hides_old_recap_then_shows_recapping_until_new_result(board):
    page, state = board
    item = next(it for it in state["items"] if it["id"] == "completed-0")
    row = page.locator('[data-id="completed-0"]')
    expect(row.locator(".task-summary-toggle")).to_have_count(0)  # no separate expand pill
    pill = row.locator(".task-pill-recap")
    expect(pill).to_have_attribute("aria-expanded", "false")
    pill.click()
    expect(row.locator(".task-summary-panel")).to_be_visible()
    expect(pill).to_have_attribute("aria-expanded", "true")
    pill.press("Enter")
    expect(row.locator(".task-summary-panel")).to_be_hidden()
    pill.press(" ")
    expect(row.locator(".task-summary-panel")).to_be_visible()
    item.update(
        card="working",
        state="streaming",
        recap_archivable=False,
        summary_verdict="follow_up",
        summary_follow_up="Old next step",
        activity={"display": {"label": "Working"}},
    )
    refresh(page, state)
    expect(row.locator(".task-pill-recap")).to_have_count(0)
    expect(row.locator(".task-pill-status")).to_have_text("Working")
    item.update(card="completed", state="idle", summary_pending=True, activity={"display": {"label": "Ready"}})
    for status in ("done", "queued", "running"):
        item["summary_status"] = status
        refresh(page, state)
        expect(row.locator(".task-pill-status")).to_have_text("Recapping…")
        expect(row.locator(".task-summary-panel")).to_have_count(0)
        expect(row).not_to_contain_text("Old next step")
        expect(row).not_to_contain_text("needs follow-up")
    item.update(
        card="needs_input",
        summary_pending=False,
        summary_status="done",
        summary="New recap",
        summary_follow_up="New next step",
    )
    refresh(page, state)
    expect(row.locator(".task-pill-status")).to_have_text("Needs input")
    expect(row.locator(".task-summary-panel")).to_be_hidden()
    row.locator(".task-pill-recap").click()
    expect(row.locator(".task-summary-panel")).to_be_visible()
    expect(row).to_contain_text("New next step")


def test_working_task_shows_its_last_recap_without_the_old_verdict(board):
    page, state = board
    item = next(it for it in state["items"] if it["id"] == "completed-0")
    item.update(
        card="working",
        state="streaming",
        recap_archivable=False,
        summary_previous=True,
        summary_status="",
        summary="Merged the fix.",
        summary_verdict="follow_up",
        summary_follow_up="Approve the restart",
        activity={"display": {"label": "Working"}},
    )
    refresh(page, state)
    row = page.locator('[data-id="completed-0"]')
    expect(row.locator(".task-pill-status")).to_have_text("Working")
    pill = row.locator(".task-pill-recap")
    expect(pill).to_have_text("last recap")
    expect(row).not_to_contain_text("needs follow-up")
    expect(row.locator(".task-summary-panel")).to_be_hidden()
    pill.click()
    panel = row.locator(".task-summary-panel")
    expect(panel).to_be_visible()
    expect(panel).to_contain_text("Merged the fix.")
    expect(panel).to_contain_text("Approve the restart")
    refresh(page, state)  # stays open across the poll rebuild
    expect(row.locator(".task-summary-panel")).to_be_visible()


def test_model_selection_card_expansion_and_title_navigation(board):
    page, state = board
    page.evaluate(
        "localStorage.setItem('sharedagents.web.pickerAgent', 'codex'); localStorage.setItem('sharedagents.web.pickerModels', JSON.stringify({jarvis:'different', codex:'saved'}))"
    )
    preferences = page.evaluate("JSON.stringify(localStorage)")
    page.locator("#tasks-summary-model").click()
    picker = page.locator(".shared-agent-model-dialog")
    expect(picker).to_be_visible()
    expect(picker.locator(".amp-tabs-nav")).to_be_hidden()
    expect(picker.locator(".amp-provider-tab")).to_contain_text(["vllm"])
    expect(picker.locator('.amp-model-card[data-model="unsupported"]')).to_have_count(0)
    picker.get_by_role("searchbox").fill("careful")
    picker.locator('[data-model="wire-careful"]').click()
    picker.locator("[data-save]").click()
    expect(picker).to_be_hidden()
    expect(page.locator("#tasks-summary-model-status")).to_have_text("Saved for the next recap.")
    assert state["selected_model"] == "careful"
    assert page.evaluate("JSON.stringify(localStorage)") == preferences
    row = page.locator('[data-id="completed-0"]')
    row.locator(".task-pill-recap").click()
    expect(row.locator(".task-summary-panel")).to_be_visible()
    assert page.url == "http://tasks.archive/tasks"
    row.locator(".task-title").click()
    expect(page).to_have_url("http://tasks.archive/?session=session-completed-0")


def test_share_button_uses_existing_permission_editor_without_opening_the_session(board):
    page, state = board
    row = page.locator('[data-id="completed-0"]')
    share = row.locator(".task-share")
    # The lock sits beside the archive ×, on the title line, not in the pills.
    expect(row.locator(".task-actions > .task-share + .task-remove")).to_have_count(1)
    expect(row.locator(".task-meta .task-share")).to_have_count(0)
    title_box, share_box = row.locator(".task-title").bounding_box(), share.bounding_box()
    assert share_box["y"] < title_box["y"] + title_box["height"]
    assert share_box["x"] > title_box["x"] + title_box["width"] - 1
    # Same control as the chat header: a lock while the session is private.
    expect(share).to_have_class("task-share icon-btn share-btn")
    expect(share).to_have_attribute("aria-label", "Sharing: Private: only you and admins")
    expect(share.locator("svg rect")).to_have_count(1)
    expect(share).to_have_text("")
    share.click()
    pop = page.locator("#share-pop")
    expect(pop).to_be_visible()
    expect(pop.locator('#share-add option[value="member@sharedagents.test"]')).to_have_count(1)
    pop.locator('input[name="share-vis"][value="public"]').check()
    pop.locator("#share-add").select_option("member@sharedagents.test")
    pop.locator('[data-share="save"]').click()
    expect(pop).to_be_hidden()
    assert state["shares"] == [{"visibility": "public", "share_all": False, "members": ["member@sharedagents.test"]}]
    assert page.url == "http://tasks.archive/tasks"
    expect(row.locator(".task-summary-panel")).to_be_hidden()
    # The saved session's card turns to the people icon without a reload,
    # and the board's re-read from the server keeps it.
    expect(share).to_have_class("task-share icon-btn share-btn is-public")
    expect(share).to_have_attribute("aria-label", "Sharing: Shared with Member")
    expect(share.locator("svg circle")).to_have_count(1)
    refresh(page, state)
    expect(share).to_have_class("task-share icon-btn share-btn is-public")
    expect(page.locator('[data-id="completed-1"] .task-share')).not_to_have_class(re.compile(r"\bis-public\b"))


def test_shared_session_task_shows_the_header_globe_icon(board):
    page, state = board
    state["items"][0]["sharing"] = {"visibility": "public", "share_all": True, "members": []}
    refresh(page, state)
    share = page.locator(f'[data-id="{state["items"][0]["id"]}"] .task-share')
    expect(share).to_have_class("task-share icon-btn share-btn is-public")
    expect(share).to_have_attribute("title", "Public: everyone with an account")
    expect(share.locator("svg circle")).to_have_count(1)
    expect(share.locator("svg path")).to_have_count(2)


@pytest.mark.parametrize(
    "board",
    [
        {"width": 1440, "height": 800},
        {"width": 390, "height": 844},
    ],
    indirect=True,
    ids=["desktop", "phone"],
)
def test_shared_model_picker_discard_retry_and_default(board):
    page, state = board
    state["selected_model"] = "careful"
    page.locator("#tasks-summary-model").click()
    picker = page.locator(".shared-agent-model-dialog")
    expect(picker.locator('[data-model="wire-careful"]')).to_have_attribute("aria-checked", "true")
    assert picker.evaluate(
        "el => {const r=el.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth && r.bottom <= innerHeight;}"
    )
    picker.locator("[data-dismiss]").last.click()
    expect(picker).to_be_hidden()
    assert state["selected_model"] == "careful"
    page.locator("#tasks-summary-model").click()
    state["fail_model"] = True
    picker.locator("[data-default]").click()
    expect(picker.locator("[data-error]")).to_have_text("Could not save recap model.")
    expect(picker).to_be_visible()
    assert state["selected_model"] == "careful"
    expect(page.locator("#tasks-summary-model")).to_have_text("Careful model")
    state["fail_model"] = False
    picker.locator("[data-default]").click()
    expect(picker).to_be_hidden()
    expect(page.locator("#tasks-summary-model")).to_have_text("Default model")
    assert state["selected_model"] == ""


def test_shared_dialog_keeps_the_agent_and_model_workflow_for_other_pages(board):
    page, _state = board
    page.evaluate("""async () => {
        window.selection = null;
        window.existingDialog = new AgentModelDialog({onSave: selected => window.selection = selected});
        await existingDialog.open({agent: 'jarvis', model: 'careful'});
    }""")
    picker = page.locator(".shared-agent-model-dialog[open]")
    expect(picker.locator(".amp-tabs-nav")).to_be_visible()
    expect(picker.locator('[data-agent="codex"]')).to_be_visible()
    expect(picker.locator("[data-default]")).to_be_hidden()
    picker.locator('[data-tab="model"]').click()
    expect(picker.locator('[data-model="wire-careful"]')).to_have_attribute("aria-checked", "true")
    picker.locator("[data-save]").click()
    expect(picker).to_be_hidden()
    assert page.evaluate("window.selection.model") == "wire-careful"
    assert page.evaluate("JSON.parse(localStorage.getItem('sharedagents.web.pickerModels')).jarvis") == "wire-careful"


@pytest.mark.parametrize("archived", [False, True], ids=["completed", "archived"])
def test_readable_recap_without_a_verdict_has_a_keyboard_expander(board, archived):
    page, state = board
    item = next(it for it in state["items"] if it["id"] == "completed-0")
    item.update(summary="Readable fallback recap", summary_verdict="", summary_status="done")
    if archived:
        item.update(card="archived", archived=True)
        page.locator("#tasks-show-archived").check()
    refresh(page, state)
    row = page.locator('[data-id="completed-0"]')
    pill = row.get_by_role("button", name="recap", exact=True)
    expect(pill).to_have_attribute("tabindex", "0")
    expect(pill).to_have_attribute("aria-expanded", "false")
    pill.focus()
    before = page.evaluate("window.scrollY")
    pill.press(" ")
    expect(pill).to_be_focused()
    expect(pill).to_have_attribute("aria-expanded", "true")
    expect(row.locator(".task-summary-panel")).to_be_visible()
    expect(row.locator(".task-summary-panel")).to_have_text("Readable fallback recap")
    assert page.evaluate("window.scrollY") == before
    pill.press("Enter")
    expect(pill).to_be_focused()
    expect(row.locator(".task-summary-panel")).to_be_hidden()


@pytest.mark.parametrize("filtered", [False, True], ids=["visible", "filtered-out"])
def test_last_recap_open_state_resets_when_pending(board, filtered):
    page, state = board
    item = next(it for it in state["items"] if it["id"] == "completed-16")
    item.update(card="working", state="streaming", summary_previous=True, summary_status="", recap_archivable=False)
    refresh(page, state)
    row = page.locator('[data-id="completed-16"]')
    row.locator(".task-pill-recap").click()
    expect(row.locator(".task-summary-panel")).to_be_visible()
    if filtered:
        page.locator("#tasks-topic-filter").fill("alpha")
        expect(row).to_have_count(0)
    item.update(card="completed", state="idle", summary_previous=False, summary_pending=True, summary_status="running")
    refresh(page, state)
    if not filtered:
        expect(row.locator(".task-summary-panel")).to_have_count(0)
        expect(row).not_to_contain_text("Finished task")
    item.update(summary_pending=False, summary_status="done", summary="New turn recap", summary_verdict="complete")
    refresh(page, state)
    page.locator("#tasks-topic-filter").fill("")
    expect(row.locator(".task-summary-panel")).to_be_hidden()
    expect(row.locator(".task-pill-recap")).to_have_attribute("aria-expanded", "false")
    row.locator(".task-pill-recap").click()
    expect(row.locator(".task-summary-panel")).to_have_text("New turn recap")


def test_row_actions_do_not_toggle_or_swallow_navigation(board):
    page, state = board
    item = next(it for it in state["items"] if it["id"] == "completed-0")
    item["sdlc_run_id"] = "run-123"
    refresh(page, state)
    row = page.locator('[data-id="completed-0"]')
    panel = row.locator(".task-summary-panel")
    row.locator(".task-select").check()
    expect(row.locator(".task-select")).to_be_checked()
    expect(panel).to_be_hidden()
    # Cancellation leaves the recap collapsed; accepting still untracks.
    page.evaluate("window.confirmCard = async () => false")
    row.locator(".task-remove").click()
    expect(panel).to_be_hidden()
    # Let anchor events reach document before preventing navigation so both
    # the Dev and title links can be checked against the same live board.
    page.evaluate("""() => {
        window.links = [];
        document.addEventListener('click', e => {
            const link = e.target.closest('a');
            if (link) {
                window.links.push({href: link.getAttribute('href'), prevented: e.defaultPrevented});
                e.preventDefault();
            }
        });
    }""")
    row.locator(".task-pill-dev").click()
    expect(panel).to_be_hidden()
    row.locator(".task-title").click()
    expect(panel).to_be_hidden()
    assert page.evaluate("window.links") == [
        {"href": "/dev?run=run-123", "prevented": False},
        {"href": "/?session=session-completed-0", "prevented": False},
    ]
    row.locator(".task-meta").click(position={"x": 2, "y": 2})
    expect(panel).to_be_visible()
    row.locator(".task-pill-recap").click()
    expect(panel).to_be_hidden()

    deletes = []

    def untrack(route):
        assert route.request.method == "DELETE"
        deletes.append(item["id"])
        state["items"].remove(item)
        route.fulfill(json={"ok": True})

    page.route("http://tasks.archive/api/tasks/completed-0", untrack)
    page.evaluate("window.confirmCard = async () => true")
    row.locator(".task-remove").click()
    expect(row).to_have_count(0)
    assert deletes == ["completed-0"]
    assert page.url == "http://tasks.archive/tasks"


def test_pending_recap_without_panel_is_not_a_button(board):
    page, state = board
    item = next(it for it in state["items"] if it["id"] == "completed-0")
    item.update(summary_pending=True, summary_status="failed", summary="", summary_verdict="")
    refresh(page, state)
    row = page.locator('[data-id="completed-0"]')
    pill = row.locator(".task-pill-recap")
    expect(pill).to_have_text("recap retry")
    assert pill.get_attribute("role") is None
    assert pill.get_attribute("tabindex") is None
    assert pill.get_attribute("aria-expanded") is None
    expect(row.locator(".task-summary-panel")).to_have_count(0)
