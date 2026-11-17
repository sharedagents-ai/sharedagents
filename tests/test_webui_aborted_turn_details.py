"""An aborted turn keeps details on its own card, without a command bubble."""

from pathlib import Path

import pytest

from test_webui_goal_reset_new_chat_js import _js, needs_node

STATIC = Path(__file__).resolve().parents[1] / "src/sharedagents_web/static"


@needs_node
@pytest.mark.parametrize("replay", [False, True])
def test_aborted_turn_keeps_available_metrics(replay):
    setup = (
        "const bs = blockStreamFromSavedSegments([{kind:'assistant',content:'Partial answer',"
        "metrics:{model:'example-model',completion_tokens:5}}], '', state.vis, {aborted:true});"
        if replay
        else "const bs = new WebBlockStream(); bs.textDelta('Partial answer');"
        "bs.turnEnded({model:'example-model',completion_tokens:5},{aborted:true});"
    )
    result = _js(
        setup + "({aborted:bs.aborted, ended:bs.ended,"
        "stats:bs.blocks.map(block=>block.stats).filter(Boolean),"
        "text:bs.blocks.map(block=>WebBlockStream._text(block)).join('')})",
        extra_files=("streaming.js",),
    )
    assert result["aborted"] and result["ended"]
    assert result["text"] == "Partial answer"
    assert len(result["stats"]) == 1 and "example-model" in result["stats"][0]
    assert "out 5 tok" in result["stats"][0]


@needs_node
@pytest.mark.parametrize("reason", ["owner_aborted", "stale_live_reconcile"])
def test_abort_terminal_does_not_add_command_output(reason):
    result = _js(
        "const notices=[]; appendSystemMessage=(text)=>notices.push(text);"
        "renderBlockTurn=()=>{};"
        "_stopStreaming=()=>{state.isStreaming=false;};"
        "for(const name of ['updateSendState','compactContextTurnEnded',"
        "'_finalizePendingAskUserCards','_finalizePendingPermissionCards',"
        "'applyTurnGoalMetrics','applyLiveContextMetrics','syncContextMeterFromHeader',"
        "'updateMicButton','syncTeacherState','refreshSessionWidgets','refreshTreeLive',"
        "'renderGoalWidget']) globalThis[name]=()=>{};"
        "const bs=new WebBlockStream(); bs.textDelta('Partial answer');"
        "Object.assign(state,{currentSessionId:'S1',isStreaming:true,"
        "liveTurn:{bs,el:{}},msgCache:[],livePhysicalTurnId:'P1'});"
        "renderCanonicalEvent({type:'turn_ended',physical_turn_id:'P1',aborted:true,"
        f"metrics:{{abort_reason:'{reason}'}}}});"
        "({notices,aborted:bs.aborted,streaming:state.isStreaming})",
        extra_files=("streaming.js",),
    )
    assert result == {"notices": [], "aborted": True, "streaming": False}


@pytest.fixture
def aborted_page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 400})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content(
            '<!doctype html><meta name="viewport" content="width=device-width">'
            '<body class="monitor-body"><main id="messages"></main></body>'
        )
        page.add_style_tag(path=STATIC / "style.css")
        page.add_script_tag(
            content="const state = {msgCache:[]};function formatMessageTime(value) { return value ? '12:00:03' : ''; }"
        )
        source = (STATIC / "virtual_scroll.js").read_text()
        page.add_script_tag(
            content=source[
                source.index("function _historyTurnStatus(") : source.index("function _remountAskUserFromCache(")
            ]
        )
        page.add_script_tag(path=STATIC / "output_artifacts.js")
        yield page
        browser.close()
        assert errors == [], errors


def mount_aborted_turn(page, metrics=False):
    page.evaluate(
        """metrics => {
      document.querySelector('#messages').innerHTML = `
        <div class="message message-assistant message-aborted" id="aborted-turn">
          <div class="message-header"><div class="message-avatar assistant">A</div>
            <span>Assistant</span></div>
          <div class="message-content">Partial answer</div>
          <div class="message-telemetry"></div>
        </div>`;
      const el = document.querySelector('#aborted-turn');
      state.msgCache = [{el,role:'assistant',status:'aborted',aborted:true,
        timestamp:'2026-11-17T12:00:00Z',endedAt:'2026-11-17T12:00:03Z'}];
      if (metrics) {
        const row=document.createElement('div');
        row.className='assist-metrics block-metrics';
        row.textContent='example-model · in 40 · out 5 tok';
        el.querySelector('.message-content').appendChild(row);
      }
      renderMessageTelemetry(el);
    }""",
        metrics,
    )


@pytest.mark.parametrize("metrics", [False, True])
def test_aborted_details_survive_footer_refresh(aborted_page, metrics):
    page = aborted_page
    mount_aborted_turn(page, metrics)
    card = page.locator("#aborted-turn")
    assert card.locator(".aborted-pill").is_visible()
    assert card.locator(".turn-elapsed").text_content() == "3s"
    details = card.locator("details.turn-details")
    details.locator("summary").click()
    assert details.get_attribute("open") is not None
    for _ in range(3):
        page.evaluate("renderMessageTelemetry(document.querySelector('#aborted-turn'))")
    assert details.get_attribute("open") is not None
    assert card.locator("details.turn-details").count() == 1
    if metrics:
        assert details.locator(".block-metrics").text_content() == "example-model · in 40 · out 5 tok"
        assert details.locator(".turn-details-empty").count() == 0
    else:
        assert "before usage metrics were reported" in details.inner_text()
    assert "Partial answer" in card.locator(".message-content").text_content()
