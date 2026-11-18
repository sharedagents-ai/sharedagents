"""Compaction stays in its child session across live and replayed chat turns."""

import json
from pathlib import Path

import pytest

from test_webui_goal_reset_new_chat_js import _js, needs_node

STATIC = Path(__file__).resolve().parents[1] / "src/sharedagents_web/static"
SUMMARY = "compacted: ~72000 tokens removed · prompt now ~51000 tokens\n\n## Goal\nKeep the current task."
NOTICE = {
    "kind": "notice",
    "subtype": "compaction",
    "payload": {
        "child_session_id": "compact-child",
        "phase": "done",
        "trigger": "tool",
        "before_tokens": 123000,
        "after_tokens": 51000,
        "summary": SUMMARY,
    },
}


def _segments():
    return [
        {"kind": "tool", "call_id": "compact-1", "name": "compact_context", "ok": True, "content": SUMMARY},
        NOTICE,
    ]


@needs_node
@pytest.mark.parametrize("tools", [False, True])
def test_live_compaction_does_not_split_or_replace_the_answer(tools):
    result = _js(
        "const bs=new WebBlockStream({toolShowHeader:" + json.dumps(tools) + "}); bs.textDelta('Before. ');"
        "bs.toolCall('compact-1','compact_context',{});"
        f"bs.toolDelta('compact-1',{json.dumps(SUMMARY)});"
        f"bs.toolResult('compact-1',{json.dumps(SUMMARY)},true);"
        "bs.textDelta('After.'); bs.turnEnded({});"
        "({text:foldedRawFromBlocks(bs),kinds:bs.visibleBlocks().map(b=>b.kind)})",
        extra_files=("streaming.js",),
    )
    assert result == {"text": "Before. After.", "kinds": ["answer"]}


@needs_node
@pytest.mark.parametrize("duplicate", [False, True])
def test_replay_omits_compaction_even_when_committed_answer_repeats_it(duplicate):
    segments = _segments()
    if duplicate:
        segments.append({"kind": "assistant", "content": SUMMARY})
    result = _js(
        f"const bs=blockStreamFromSavedSegments({json.dumps(segments)},"
        f"{json.dumps(SUMMARY)},state.vis);"
        "({text:foldedRawFromBlocks(bs),hidden:bs.isCompactionOnly()})",
        extra_files=("streaming.js",),
    )
    assert result == {"text": "", "hidden": True}


@needs_node
def test_replay_keeps_task_output_and_unrelated_compacted_prose():
    answer = "compacted files are ready for review."
    segments = _segments() + [{"kind": "assistant", "content": answer}]
    result = _js(
        f"const bs=blockStreamFromSavedSegments({json.dumps(segments)},"
        f"{json.dumps(answer)},state.vis);"
        "({text:foldedRawFromBlocks(bs),hidden:bs.isCompactionOnly()})",
        extra_files=("streaming.js",),
    )
    assert result == {"text": answer, "hidden": False}


@needs_node
def test_child_summary_and_other_direct_dispatch_results_remain_readable():
    result = _js(
        f"const child=blockStreamFromSavedSegments([], {json.dumps(SUMMARY)},state.vis);"
        "const delegated=new WebBlockStream();"
        "delegated.toolCall('d','delegate_codex',{});"
        "delegated.toolResult('d','Task completed.',true);"
        "({child:foldedRawFromBlocks(child),delegated:foldedRawFromBlocks(delegated)})",
        extra_files=("ask_user.js", "streaming.js"),
    )
    assert result["child"] == SUMMARY
    assert "Task completed." in result["delegated"]


@needs_node
@pytest.mark.parametrize("failure", ["tool", "error", "abort"])
def test_compaction_failure_is_visible(failure):
    operation = {
        "tool": "bs.toolResult('c','Summarizer unavailable.',false);",
        "error": "bs.turnEnded({}, {error:'Summarizer unavailable.'});",
        "abort": "bs.turnEnded({}, {aborted:true});",
    }[failure]
    result = _js(
        "const bs=new WebBlockStream({toolShowHeader:false});"
        "bs.toolCall('c','compact_context',{});"
        + operation
        + "({hidden:bs.isCompactionOnly(),text:foldedRawFromBlocks(bs)})",
        extra_files=("streaming.js",),
    )
    assert result["hidden"] is False
    if failure == "tool":
        assert "Summarizer unavailable." in result["text"]


@needs_node
def test_replayed_failure_is_reported_once():
    segments = [
        {"kind": "tool", "call_id": "c", "name": "compact_context", "ok": False, "content": "Summarizer unavailable."}
    ]
    result = _js(
        f"const bs=blockStreamFromSavedSegments({json.dumps(segments)},'Summarizer unavailable.',state.vis);"
        "({text:foldedRawFromBlocks(bs),hidden:bs.isCompactionOnly()})",
        extra_files=("streaming.js",),
    )
    assert result["hidden"] is False
    assert result["text"].count("Summarizer unavailable.") == 1


@needs_node
def test_lifecycle_updates_context_without_creating_a_parent_message():
    result = _js(
        "let created=0; const metrics=[];"
        "ensureLiveAssistant=()=>{created++;throw Error('unexpected parent bubble');};"
        "applyLiveContextMetrics=(m)=>metrics.push(m);"
        "renderCanonicalEvent({type:'compaction',phase:'done',child_session_id:'compact-child',"
        "after_tokens:51000,context_window:200000}); ({created,metrics})",
        extra_files=("streaming.js",),
    )
    assert result == {
        "created": 0,
        "metrics": [{"prompt_tokens": 51000, "context_window": 200000, "tokens_estimated": False}],
    }


@needs_node
def test_automatic_compaction_history_does_not_split_surrounding_prose():
    segments = [
        {"kind": "assistant", "content": "Before. "},
        NOTICE,
        {"kind": "assistant", "content": "After."},
    ]
    result = _js(
        f"const bs=blockStreamFromSavedSegments({json.dumps(segments)},'',state.vis);"
        "({text:foldedRawFromBlocks(bs),kinds:bs.visibleBlocks().map(b=>b.kind)})",
        extra_files=("streaming.js",),
    )
    assert result == {"text": "Before. After.", "kinds": ["answer"]}


@pytest.fixture
def compaction_page():
    playwright = pytest.importorskip("playwright.sync_api")
    from test_webui_agents_footer_card_browser import _page_html

    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 960, "height": 720})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("http://compaction.example/**", lambda route: route.fulfill(body=_page_html()))
        page.goto("http://compaction.example/")
        page.add_style_tag(path=STATIC / "style.css")
        page.add_script_tag(path=STATIC / "state.js")
        page.add_script_tag(
            content="""
            const messagesContainer=document.getElementById('messages');
            const welcomeScreen=document.createElement('div');
            function scrollToBottom() {}
            function setWelcomeVisible() {}
            function _turnIndicatorMark() {}
            function updateSendState() {}
            function refreshTreeLive() {}
            function renderHarnessWidget() {}
            function selectedPickerAgent() { return state.currentAgent; }
            const opened=[];
            Object.assign(state,{currentSessionId:'parent',currentAgent:'general'});
            messagesContainer.textContent='';
        """
        )
        for name in (
            "markdown.js",
            "virtual_scroll.js",
            "output_artifacts.js",
            "sessions.js",
            "streaming.js",
            "agent_mon_mini.js",
        ):
            page.add_script_tag(path=STATIC / name)
        page.evaluate("() => { window.saOpenConversation=(id,options)=>opened.push({id,options}); }")
        yield page
        browser.close()
        assert errors == [], errors


@pytest.mark.browser
@pytest.mark.parametrize("origin", ["added", "older", "memory_flush"])
def test_unmeasured_messages_reserve_space_but_hidden_messages_do_not(compaction_page, origin):
    result = compaction_page.evaluate(
        """origin => {
          if (origin === 'added') {
            VirtualScroller.add('assistant','A normal answer.',null,null,null,'general');
          } else {
            state.scroll.renderedStart=1;
            const older=origin === 'memory_flush'
              ? [{role:'user',content:'[MEMORY FLUSH] Save durable state.'},
                 {role:'assistant',content:'Saved.'}]
              : [{role:'assistant',content:'An older answer.'}];
            VirtualScroller.prependOlder(older);
          }
          const unmeasured=VirtualScroller._estimateHeight(0,state.msgCache.length);
          const hidden=document.createElement('div');
          hidden.hidden=true;
          VirtualScroller.measure(0,hidden);
          return {unmeasured,hidden:VirtualScroller._estimateHeight(0,1)};
        }""",
        origin,
    )
    assert result["unmeasured"] > 0
    assert result["hidden"] == 0


@pytest.mark.browser
@pytest.mark.parametrize("width", [390, 960])
def test_chat_stays_clean_after_virtual_rebuild_and_footer_opens_child(compaction_page, width):
    page = compaction_page
    page.set_viewport_size({"width": width, "height": 720})
    page.evaluate(
        """({segments,summary}) => {
          const user=_createMessageElement('user','__SHAREDAGENTS_DELEGATE__:compact_context:{}');
          messagesContainer.appendChild(user);
          const el=_createMessageElement('assistant',summary,null,'general');
          VirtualScroller.add('assistant',summary,el,null,segments,'general');
          messagesContainer.appendChild(el);
          renderBlockTurn({el,bs:blockStreamFromSavedSegments(segments,summary,state.vis)},false);
          window.compactionTurn=el;
          SaAgentMon.update({active:true,hash:'parent',agent:'general',live:false});
          SaAgentMon.applySnapshot({footer:{rows:[
            {hash:'parent',agent:'general',role:'current',state:'idle'},
            {hash:'compact-child',parent_hash:'parent',agent:'compaction',
             title:'Compact context',role:'sub',state:'idle',depth:1}
          ],subs:1,counts:{active:0,total:1,agents:1}}});
        }""",
        {"segments": _segments(), "summary": SUMMARY},
    )
    assert page.locator(".message-delegate-compact").is_hidden()
    assert page.locator(".message-assistant").is_hidden()
    page.evaluate("compactionTurn.remove(); VirtualScroller.rebuild(0)")
    assert page.locator(".message-assistant").is_hidden()
    assert page.locator(".block-compaction-notice, .message-compaction-stats").count() == 0
    page.locator(".ama-summary").click()
    page.locator('.ama-row[data-hash="compact-child"]').click()
    assert page.evaluate("opened") == [{"id": "compact-child", "options": {"fromMonitor": True, "scopeHash": "parent"}}]
    # The child uses ordinary assistant content, without a compact_context tool.
    page.evaluate(
        """summary => {
          state.currentAgent='compaction';
          const el=_createMessageElement('assistant',summary,null,'compaction');
          messagesContainer.appendChild(el);
          renderBlockTurn({el,bs:blockStreamFromSavedSegments([],summary,state.vis)},false);
        }""",
        SUMMARY,
    )
    assert page.locator(".message-assistant:visible").count() == 1
    assert "Keep the current task." in page.locator(".message-assistant:visible").inner_text()


@pytest.mark.browser
@pytest.mark.parametrize("method", ["rebuild", "_materialise"])
def test_live_manual_compaction_keeps_zero_height_after_rebuild(compaction_page, method):
    page = compaction_page
    result = page.evaluate(
        """({method,summary}) => {
          const el=_createMessageElement('assistant','',null,'general');
          VirtualScroller.add('assistant','',el,null,null,'general');
          messagesContainer.appendChild(el);
          const bs=new WebBlockStream();
          bs.toolCall('manual','compact_context',{});
          bs.toolResult('manual',summary,true);
          bs.turnEnded({engine:'local_llm',direct_dispatch:'compact_context'});
          renderBlockTurn({el,bs},false);
          el.remove();
          const rebuilt=VirtualScroller[method](0);
          return {hidden:rebuilt.hidden,height:VirtualScroller._estimateHeight(0,1),
                  content:rebuilt.querySelector('.message-content').textContent};
        }""",
        {"method": method, "summary": SUMMARY},
    )
    assert result["hidden"] is True
    assert result["height"] == 0
    assert "compacted:" not in result["content"]


@pytest.mark.browser
def test_manual_compaction_error_reveals_the_hidden_turn(compaction_page):
    page = compaction_page
    page.evaluate("""() => {
      const el=_createMessageElement('assistant','',null,'general');
      VirtualScroller.add('assistant','',el,null,null,'general');
      messagesContainer.appendChild(el);
      const bs=new WebBlockStream();
      bs.toolCall('manual','compact_context',{});
      renderBlockTurn({el,bs},false);
      if (!el.hidden) throw Error('control turn should start hidden');
      bs.turnEnded({}, {error:'Summarizer unavailable.'});
      renderBlockTurn({el,bs},false);
    }""")
    assert page.locator(".message-assistant").is_visible()
    assert "Summarizer unavailable." in page.locator(".block-error").inner_text()
    assert page.evaluate("VirtualScroller._estimateHeight(0,1)") > 0
