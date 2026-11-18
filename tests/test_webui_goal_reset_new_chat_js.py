"""BUG 2 — the goal row must not survive "New Chat" (JS vectors).

Regression: ``resetGoalTracker()`` was only called from
``openSession`` behind a ``prevSid !== sid`` guard. Neither New Chat path
(``startNewChatDraft()`` nor ``_commitSession()``, which sets
``state.currentSessionId`` directly) cleared the goal slice, and the
self-heal was blocked because ``applyGoalDetail``/``refreshGoalCost`` bail
when ``state.goalSid !== sid``. The previous conversation's goal row stayed
pinned over the fresh chat.

Drives the REAL sessions.js/widgets.js/api.js in node's vm; the ✕
(``/goal clear``) and ⏸/▶ (pause/resume POST) semantics are untouched.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "sharedagents_web" / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _js(vectors, extra_files=()):
    """Evaluate ``vectors`` (JS statements → final expression, may return a
    Promise) with state.js + api.js + widgets.js + sessions.js loaded."""
    assert NODE is not None, "node is required for JS vector tests"
    elements = [
        "sidebar",
        "sidebarBackdrop",
        "sessionList",
        "agentSelect",
        "messagesContainer",
        "messageInput",
        "sendBtn",
        "treeStopBtn",
        "micBtn",
        "realtimeBtn",
        "realtimeVoiceStatus",
        "realtimeListeningCaption",
        "newChatBtn",
        "toggleSidebarBtn",
        "settingsBtn",
        "settingsModal",
        "closeSettingsBtn",
        "currentAgentEl",
        "modelSelect",
        "teacherToggle",
        "teacherToggleState",
        "teacherModelSelect",
        "teacherBadge",
        "statusText",
        "sessionStatus",
        "statusSegments",
        "teacherWidget",
        "goalWidget",
        "harnessWidget",
        "queueWidget",
        "welcomeScreen",
        "chatView",
        "pastePreview",
        "attachBtn",
        "attachInput",
        "optMode",
        "optTransport",
        "optCollapse",
        "topicList",
        "approvalsList",
        "optionsCard",
        "header",
        "saveGw",
    ]
    globals_js = ",".join(f"{n}:mkEl('{n}')" for n in elements)
    script = (
        "const fs=require('fs'),vm=require('vm');"
        "const noop=()=>null;"
        "const mkEl=(id)=>({id,innerHTML:'',textContent:'',hidden:false,style:{},dataset:{},"
        "classList:{add:noop,remove:noop,toggle:noop,contains:()=>false},"
        "addEventListener:noop,removeEventListener:noop,setAttribute:noop,removeAttribute:noop,hasAttribute:()=>false,focus:noop,blur:noop,remove:noop,appendChild:noop,insertBefore:noop,replaceChildren:noop,firstChild:null,childNodes:[],children:[],insertAdjacentHTML:noop,getBoundingClientRect:()=>({top:0,left:0,width:0,height:0,bottom:0,right:0}),scrollTo:noop,"
        "getAttribute:()=>null,querySelector:()=>mkEl(id+'-sub'),"
        "querySelectorAll:()=>[],closest:()=>null});"
        "const els={};"
        "const posted=[];"
        "const ctx=vm.createContext({"
        "window:{addEventListener:noop,location:{href:'http://x/',search:''},"
        "__saLiveSources:new Set(),console},"
        "document:{addEventListener:noop,"
        "getElementById:(id)=>{if(!els[id])els[id]=mkEl(id);return els[id];},"
        "querySelector:()=>null,querySelectorAll:()=>[],createElement:()=>mkEl('x')},"
        "localStorage:{getItem:()=>null,setItem:noop,removeItem:noop},"
        "history:{replaceState:noop},"
        "EventSource:function(){this.readyState=0;this.addEventListener=noop;"
        "this.close=noop;},"
        "IntersectionObserver:function(){this.observe=noop;this.unobserve=noop;this.disconnect=noop;},"
        "ResizeObserver:function(){this.observe=noop;this.unobserve=noop;this.disconnect=noop;},"
        "navigator:{},performance:{now:()=>0},console,setTimeout:(f)=>{try{f&&f()}catch(_e){}return 0},"
        "clearTimeout:noop,setInterval:()=>0,clearInterval:noop,"
        "requestAnimationFrame:noop,"
        "fetch:(u,o)=>{posted.push([String(u),o&&o.method]);"
        "return Promise.resolve({ok:true,status:200,"
        "json:()=>Promise.resolve({session:{id:'NEW1',model:'qwen-x',"
        "agent:'jarvis',revision:0,access:'chat'}})});},"
        # Globals owned by files NOT loaded here (app.js element consts +
        # send.js / render.js chrome).
        "_turn:null,isStreaming:false," + globals_js + ","
        "renderStatusSegments:noop,updateHeader:noop,renderSessionList:noop,"
        "renderTopicList:noop,clearMessages:noop,updateSendState:noop,"
        "renderModelSelect:noop,renderHarnessWidget:noop,"
        "renderSessionStatusBar:noop,applyMonitorReadOnlyUi:noop,"
        "renderAgentSelect:noop,updateAgentMonAccordion:noop,updateRealtimeVisibility:noop,updateMicVisibility:noop,renderWelcomeScreen:noop,updateTreeStopVisibility:noop,refreshSessionOptions:noop,loadSessionOptions:noop,loadCommands:noop,renderOptionsCard:noop,"
        "appendSystemMessage:noop,escapeHtml:(s)=>String(s),clearPastedImages:noop,getPastedImages:()=>[],"
        "formatUsd:(n)=>String(n),"
        "sessionGwHash:(s)=>String(s)});"
        "ctx.posted=posted;"
        "for (const f of "
        + json.dumps(
            [
                "state.js",
                "markdown.js",
                "virtual_scroll.js",
                "api.js",
                "widgets.js",
                "sessions.js",
                "render.js",
                "launch.js",
                *extra_files,
            ]
        )
        + ") "
        f"vm.runInContext(fs.readFileSync({json.dumps(str(STATIC))}+'/'+f,'utf8'),"
        "ctx,{filename:f});"
        f"const __out=vm.runInContext('eval('+{json.dumps(json.dumps(vectors))}+')',ctx);"
        "Promise.resolve(__out).then(function(v){"
        "process.stdout.write(JSON.stringify(v===undefined?null:v));})"
        ".catch(function(e){process.stdout.write(JSON.stringify("
        "{__err:String((e&&e.message)||e)}));});"
    )
    done = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


_GOAL_PRE = (
    "Object.assign(state,{"
    "activeGoal:'ship the four fixes',goalId:'g-42',goalSid:'OLD1',"
    "goalTerminal:'',goalCostUsd:1.5,goalMaxCostUsd:5,goalTurnCount:3,"
    "maxGoalTurns:10,currentSessionId:'OLD1',"
    "agents:['jarvis','build'],currentAgent:'jarvis',pickerAgent:'jarvis',"
    "pickerModels:{jarvis:'qwen-x'},"
    "modelSheet:{phase:'CLOSED',effective:{agent:'jarvis',alias:'qwen-x'},"
    "draft:{agent:'jarvis',alias:'qwen-x'}}});"
)

_READBACK = (
    "JSON.stringify({goalId:state.goalId,activeGoal:state.activeGoal,"
    "goalSid:state.goalSid,goalTerminal:state.goalTerminal,"
    "cost:state.goalCostUsd,max:state.goalMaxCostUsd,"
    "turns:state.goalTurnCount,sid:state.currentSessionId})"
)


def _assert_cleared(out):
    assert "__err" not in out, out.get("__err")
    got = json.loads(out) if isinstance(out, str) else out
    assert got["goalId"] == ""
    assert got["activeGoal"] == ""
    assert got["goalSid"] == ""
    assert got["cost"] is None
    assert got["max"] is None
    assert got["turns"] == 0


def test_start_new_chat_draft_clears_the_goal_tracker():
    """THE FIX: startNewChatDraft() must drop the goal slice next to the
    todos/subagents resets."""
    out = _js(_GOAL_PRE + "startNewChatDraft('',{passive:true});" + _READBACK)
    got = json.loads(out)
    _assert_cleared(got)
    assert got["sid"] == ""  # still a draft


def test_commit_session_clears_the_goal_tracker():
    """THE FIX: _commitSession() sets currentSessionId directly (bypassing
    openSession's reset) — the committed session must not inherit the goal."""
    out = _js(_GOAL_PRE + "_commitSession('topic-x',{agent:'jarvis'}).then(function(){return " + _READBACK + ";})")
    got = json.loads(out)
    _assert_cleared(got)
    assert got["sid"] == "NEW1"


def test_open_session_switch_still_clears_the_goal_tracker():
    """Guard: the existing openSession prevSid!==sid reset keeps working."""
    out = _js(_GOAL_PRE + "openSession({id:'OTHER1',agent:'jarvis',model:'qwen-x'},{epoch:99});" + _READBACK)
    got = json.loads(out)
    _assert_cleared(got)
    assert got["sid"] == "OTHER1"
