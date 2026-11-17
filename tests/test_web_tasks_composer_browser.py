"""Exercise the rendered Tasks page and shared composer with bounded API fixtures."""

import base64
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

_playwright = pytest.importorskip("playwright.sync_api")
expect, sync_playwright = _playwright.expect, _playwright.sync_playwright

ROOT = Path(__file__).resolve().parents[1] / "src/sharedagents_web"
BASE = "/mounted"
ORIGIN = "https://tasks.example"


@pytest.fixture
def board(request):
    config = getattr(request, "param", {"width": 1440, "height": 900})
    viewport = {key: config[key] for key in ("width", "height")}
    can_manage = config.get("can_manage", True)
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape())
    env.globals["asset_rev"] = lambda name: "test"
    html = env.get_template("tasks.html").render(
        base=BASE,
        site_title="SharedAgents",
        is_admin=can_manage,
        can_manage_tasks=can_manage,
        asset_rev=lambda name: "test",
    )
    state = {
        "creates": [],
        "names": [],
        "prompts": [],
        "uploads": 0,
        "transcriptions": 0,
        "fail_create": False,
        "fail_name": False,
        "fail_prompt": False,
    }
    items = [
        {
            "id": f"row-{i}",
            "session_id": f"session-{i}",
            "title": f"Example task {i}",
            "card": "needs_input",
            "state": "unstarted",
            "topic": "Research",
        }
        for i in range(40)
    ]

    def route(route):
        req = route.request
        path = urlsplit(req.url).path.removeprefix(BASE)
        if path == "/tasks":
            route.fulfill(content_type="text/html", body=html)
        elif path.startswith("/static/"):
            route.fulfill(path=ROOT / path.lstrip("/"))
        elif path == "/api/models":
            route.fulfill(json={"models": [{"name": "build"}, {"name": "review"}]})
        elif path == "/api/models/discover":
            route.fulfill(
                json={
                    "models": [{"id": "wire-careful", "alias": "careful", "label": "Careful model", "provider": "vllm"}]
                }
            )
        elif path == "/api/tasks/summary-model":
            route.fulfill(json={"models": [], "selected": ""})
        elif path == "/api/tasks" and req.method == "POST":
            state["creates"].append(req.post_data_json)
            if state["fail_create"]:
                route.fulfill(status=503, json={"error": "Gateway unavailable"})
            else:
                route.fulfill(status=201, json={"id": "new-task", "session": {"id": "new-session"}})
        elif path == "/api/tasks/new-task/describe":
            state["names"].append(req.post_data_json)
            if state["fail_name"]:
                route.fulfill(status=503, json={"error": "Naming unavailable"})
            else:
                route.fulfill(json={"title": "Fix login failures", "topic": "Authentication"})
        elif path == "/api/sessions/new-session/prompt":
            state["prompts"].append(req.post_data_json)
            route.fulfill(
                status=503 if state["fail_prompt"] else 202,
                json={"error": "Submit unavailable"} if state["fail_prompt"] else {"accepted": True},
            )
        elif path == "/api/tasks":
            route.fulfill(
                json={"items": items, "snapshot": {"topics": [{"topic": "Research"}], "sessions": [], "approvals": []}}
            )
        elif path == "/api/topics":
            route.fulfill(json={"topics": [{"topic": "Research"}]})
        elif path == "/api/attachments":
            state["uploads"] += 1
            is_image = b"image/png" in (req.post_data_buffer or b"")
            route.fulfill(
                json={
                    "path": "/workspace/reference.png" if is_image else "/workspace/notes.txt",
                    "name": "reference.png" if is_image else "notes.txt",
                    "size": 68,
                    **({"image_mime": "image/png"} if is_image else {}),
                }
            )
        elif path == "/api/transcribe":
            state["transcriptions"] += 1
            route.fulfill(json={"text": "Fix login failures"})
        elif path == "/":
            route.fulfill(content_type="text/html", body="Task chat")
        else:
            route.fulfill(json={"sessions": [], "topics": [], "approvals": []})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True, args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
        )
        try:
            context = browser.new_context(viewport=viewport, permissions=["microphone"])
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route(ORIGIN + "/**", route)
            page.goto(ORIGIN + BASE + "/tasks")
            expect(page.locator(".task-row")).to_have_count(40)
            if can_manage:
                expect(page.locator("#tasks-agent-model")).to_be_enabled()
            yield page, state, errors
        finally:
            browser.close()


@pytest.mark.parametrize(
    "board", [{"width": 1440, "height": 900}, {"width": 390, "height": 844}], indirect=True, ids=["desktop", "phone"]
)
def test_composer_visible_and_manual_creation_unchanged(board):
    page, state, errors = board
    composer = page.locator("#tasks-composer")
    expect(composer).to_be_in_viewport()
    assert composer.bounding_box()["y"] + composer.bounding_box()["height"] <= page.viewport_size["height"]
    expect(page.locator("#tasks-new-form")).to_be_hidden()
    page.locator("#tasks-new-toggle").click()
    page.locator("#tasks-new-title").fill("Manual title")
    page.locator("#tasks-new-topic").select_option("Research")
    page.locator('#tasks-new-form [type="submit"]').click()
    expect(page.locator("#tasks-new-form")).to_be_hidden()
    assert state["creates"] == [{"title": "Manual title", "topic": "Research"}]
    assert not state["names"] and not state["prompts"]
    page.locator("#message-input").fill("Fix login failures")
    page.locator("#message-input").press("Shift+Enter")
    assert page.locator("#message-input").input_value().endswith("\n")
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    expect(composer).to_be_in_viewport()
    if directory := os.environ.get("SHAREDAGENTS_SHOT_DIR"):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / f"tasks-composer-{page.viewport_size['width']}.png"))
    assert not errors


def test_description_uses_model_picker_and_opens_started_task(board):
    page, state, errors = board
    page.locator("#tasks-agent-model").click()
    dialog = page.locator(".shared-agent-model-dialog[open]")
    dialog.locator('[data-agent="review"]').click()
    dialog.locator('[data-model="wire-careful"]').click()
    dialog.locator("[data-save]").click()
    page.locator("#message-input").fill("Fix login failures")
    page.locator("#message-input").press("Enter")
    expect(page).to_have_url(ORIGIN + BASE + "/?session=new-session")
    assert state["creates"] == [{"agent": "review", "model": "wire-careful"}]
    assert state["names"] == [{"description": "Fix login failures"}]
    assert state["prompts"][0]["prompt"] == "Fix login failures"
    assert state["prompts"][0]["initial_model"] == "wire-careful"
    assert not errors


@pytest.mark.parametrize("failure", ["fail_name", "fail_prompt"])
def test_retry_preserves_task_and_request(board, failure):
    page, state, errors = board
    state[failure] = True
    page.locator("#message-input").fill("Fix login failures")
    page.locator("#send-btn").click()
    expect(page.locator("#status-text")).to_contain_text("unavailable")
    expect(page.locator("#tasks-created-chat")).to_have_attribute("href", BASE + "/?session=new-session")
    expect(page.locator("#message-input")).to_have_value("Fix login failures")
    state[failure] = False
    page.locator("#send-btn").click()
    expect(page).to_have_url(ORIGIN + BASE + "/?session=new-session")
    assert len(state["creates"]) == 1
    if failure == "fail_prompt":
        assert state["prompts"][0] == state["prompts"][1]
    assert not errors


def test_creation_failure_restores_editable_draft(board):
    page, state, errors = board
    state["fail_create"] = True
    page.locator("#message-input").fill("Fix login failures")
    page.locator("#send-btn").click()
    expect(page.locator("#status-text")).to_have_text("Gateway unavailable")
    expect(page.locator("#message-input")).to_be_editable()
    expect(page.locator("#send-btn")).to_be_enabled()
    assert not state["names"] and not state["prompts"] and not errors


@pytest.mark.parametrize("board", [{"width": 390, "height": 844, "can_manage": False}], indirect=True)
def test_viewer_has_no_composer_or_reserved_footer_space(board):
    page, _, errors = board
    expect(page.locator("#tasks-describe-form")).to_have_count(0)
    expect(page.locator("#tasks-new-toggle")).to_be_disabled()
    assert page.locator("#tasks-app").evaluate("el => getComputedStyle(el).paddingBottom") == "0px"
    assert not errors


def test_shared_attachments_wait_and_forward_originals_and_vision(board):
    page, state, errors = board
    page.locator("#attach-input").set_input_files(
        [
            {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"Context for the task"},
            {
                "name": "reference.png",
                "mimeType": "image/png",
                "buffer": base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1sAAAAASUVORK5CYII="
                ),
            },
        ]
    )
    page.locator("#message-input").fill("Fix login failures")
    page.locator("#send-btn").click()
    expect(page).to_have_url(ORIGIN + BASE + "/?session=new-session")
    assert state["uploads"] == 2
    prompt = state["prompts"][0]
    assert len(prompt["files"]) == 2
    assert {f["path"] for f in prompt["files"]} == {"/workspace/notes.txt", "/workspace/reference.png"}
    assert prompt["prompt"][0] == {"type": "text", "text": "Fix login failures"}
    assert prompt["prompt"][1]["type"] == "image" and prompt["prompt"][1]["data"]
    assert not errors


def test_spoken_task_uses_shared_recording_and_transcription(board):
    page, state, errors = board
    page.locator("#mic-btn").click()
    expect(page.locator("#mic-btn")).to_have_attribute("aria-label", "Stop recording")
    page.wait_for_function("state.recordingSamples >= 8000")
    # Send while recording stops the mic and waits for its final transcript.
    page.locator("#send-btn").click()
    expect(page).to_have_url(ORIGIN + BASE + "/?session=new-session")
    assert state["transcriptions"] > 0
    assert state["names"] == [{"description": "Fix login failures"}]
    assert state["prompts"][0]["prompt"] == "Fix login failures"
    assert not errors


@pytest.mark.parametrize(
    "board", [{"width": 1440, "height": 900}, {"width": 390, "height": 844}], indirect=True, ids=["desktop", "phone"]
)
def test_description_can_select_existing_topic_and_keeps_it_during_refresh(board):
    page, state, errors = board
    topic = page.get_by_role("combobox", name="New task topic", exact=True)
    expect(topic.locator("option")).to_have_text(["Automatic topic", "General", "Research"])
    topic.select_option("Research")
    page.evaluate("document.dispatchEvent(new CustomEvent('sharedagents:tasks-refresh'))")
    expect(topic).to_have_value("Research")
    expect(page.locator("#tasks-new-topic")).to_have_value("")
    page.locator("#message-input").fill("Fix login failures")
    if directory := os.environ.get("SHAREDAGENTS_SHOT_DIR"):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / f"tasks-topic-choice-{page.viewport_size['width']}.png"))
    state["fail_name"] = True
    page.locator("#send-btn").click()
    expect(page.locator("#status-text")).to_contain_text("unavailable")
    expect(topic).to_be_enabled()
    expect(topic).to_have_value("Research")
    state["fail_name"] = False
    page.locator("#send-btn").click()
    expect(page).to_have_url(ORIGIN + BASE + "/?session=new-session")
    assert len(state["creates"]) == 1
    assert state["creates"][0]["topic"] == "Research"
    assert state["names"] == [{"description": "Fix login failures", "topic": "Research"}] * 2
    assert state["prompts"][0]["prompt"] == "Fix login failures"
    assert not errors


def test_failed_automatic_naming_can_retry_with_manual_topic(board):
    page, state, errors = board
    state["fail_name"] = True
    page.locator("#message-input").fill("Fix login failures")
    page.locator("#send-btn").click()
    expect(page.locator("#status-text")).to_contain_text("unavailable")
    page.get_by_role("combobox", name="New task topic", exact=True).select_option("Research")
    state["fail_name"] = False
    page.locator("#send-btn").click()
    expect(page).to_have_url(ORIGIN + BASE + "/?session=new-session")
    assert len(state["creates"]) == 1
    assert state["names"] == [
        {"description": "Fix login failures"},
        {"description": "Fix login failures", "topic": "Research"},
    ]
    assert not errors
