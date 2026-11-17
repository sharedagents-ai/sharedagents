"""GET /api/boot carries what an untouched chat page would otherwise fetch.

Owner rule: without user interaction the chat page makes two API
calls — this history read and the conversation stream. Two things the page
used to fetch on its own now ride this read instead:

* the SAVED default model/effort of the agent the page renders — asked for
  with ``?agent=`` (plus the resumed session's own agent), read from the SAME
  gateway endpoint ``GET /api/agents/defs/<agent>`` proxies, so the composer
  seeds exactly what the Agent Selector reads;
* the left menu's Conversations/Topics lists, only when the client says the
  menu is already visible (``?lists=conversations,topics``), built exactly like
  ``GET /api/sessions?limit=20`` and ``GET /api/topics``.

The gateway is faked at ``gateway_request`` (the web's only way out); the
web store, merge, topic grouping and routes are real.
"""

import os
import tempfile

_TMP = os.path.realpath(tempfile.gettempdir())
# Share an already-isolated home when another web test module bound the server
# first (the module imports it once per process); otherwise make one. Never
# the operator's real ~/.sharedagents.
if not os.path.realpath(os.environ.get("SHAREDAGENTS_HOME") or "/").startswith(_TMP + os.sep):
    os.environ["SHAREDAGENTS_HOME"] = tempfile.mkdtemp(prefix="web-boot-")
for _k in (
    "SHAREDAGENTS_AGENTS_HOME",
    "SHAREDAGENTS_WEB_DB",
    "SHAREDAGENTS_SESSIONS_DIR",
    "SHAREDAGENTS_GW_SESSIONS_DIR",
    "SHAREDAGENTS_WEB_DEBUG",
    "SHAREDAGENTS_WEB_TRUST_PROXY",
    "SHAREDAGENTS_EXTERNAL_AUTH",
    "SHAREDAGENTS_WEB_BASE_PATH",
    "SHAREDAGENTS_WEB_BASE_URL",
):
    os.environ.pop(_k, None)
os.environ.setdefault("SHAREDAGENTS_WEB_SECRET", "test-secret-boot")
os.environ["SHAREDAGENTS_SDLC_NO_RECOVERY"] = "1"

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

OWNER = "boot-owner@sharedagents.test"

# What the gateway's GET /control/v1/agents/defs/<agent> answers — the full
# read_definition() shape, of which the page needs only effort + model.
GATEWAY_DEFS = {
    "jarvis": {
        "name": "jarvis",
        "engine": "local_llm",
        "effort": "xhigh",
        "model": "qwen3.8-27b-fp8",
        "model_managed_by": "profile",
        "system": "You are jarvis.",
        "effective_tools": ["bash", "read"],
        "config_sha": "a" * 64,
        "models": [{"id": "qwen3.8-27b-fp8"}],
    },
    "custom_helper": {
        "name": "custom_helper",
        "engine": "local_llm",
        "effort": "high",
        "model": "gpt-6-sol",
        "model_managed_by": None,
        "system": "You trade.",
        "effective_tools": ["bash"],
        "config_sha": "b" * 64,
        "models": [],
    },
}
# A TUI conversation the web store has never seen (gateway root only).
TUI_ROOT = {
    "hash": "20260924T090000000000-0123456789ab",
    "title": "TUI chat",
    "agent": "codex",
    "model": "gpt-5.6-terra",
    "topic": "sharedagents",
    "last_turn_ts": 1790240400,
    "live": False,
    "state": "idle",
}
# More gateway roots than one sidebar page (20), newest first, over 3 topics —
# so a wrong slice or a second, differently-limited read shows.
ROOTS = [TUI_ROOT] + [
    {
        "hash": f"20260923T{i:06d}000000-{i:012x}",
        "title": f"TUI {i}",
        "agent": "jarvis",
        "model": "qwen3.8-27b-fp8",
        "topic": ("ops", "research", "sharedagents")[i % 3],
        "last_turn_ts": 1790240400 - 60 * i,
        "live": False,
        "state": "idle",
    }
    for i in range(1, 25)
]


@pytest.fixture(scope="module")
def web():
    """The web app, imported at test time: the first web test module to import
    it binds its DB home for the whole process, and collecting this module must
    not pre-empt another module's stricter guard (test_web_actor_stamp.py)."""
    from sharedagents_web import server
    from sharedagents_web.routes import sessions as sessions_routes

    if not os.path.realpath(str(server.SESSION_DB_PATH)).startswith(_TMP + os.sep):
        pytest.exit(f"unsafe: web DB bound to {server.SESSION_DB_PATH}", returncode=2)
    return SimpleNamespace(server=server, routes=sessions_routes)


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


@pytest.fixture()
def gateway(monkeypatch, web):
    """The gateway as seen through ``gateway_request``: every call recorded."""
    calls = []

    def fake(method, path, **kwargs):
        calls.append({"method": method, "path": path, "params": kwargs.get("params")})
        if method == "GET" and path.startswith("/control/v1/agents/defs/"):
            name = path.rsplit("/", 1)[1]
            if name in GATEWAY_DEFS:
                return _Resp(dict(GATEWAY_DEFS[name]))
            return _Resp({"error": "unknown_agent", "code": "unknown_agent"}, 404)
        if method == "GET" and path == "/control/v1/orchestrator/sessions":
            # The gateway's roots index: newest first, cut to ``limit``.
            limit = int((kwargs.get("params") or {}).get("limit") or 50)
            return _Resp({"sessions": [dict(r) for r in ROOTS[:limit]]})
        if method == "GET" and path == "/control/v1/orchestrator/approvals":
            return _Resp({"approvals": []})
        if method == "GET" and path == "/":
            return _Resp({"ok": True})
        return None  # anything else: unreachable

    monkeypatch.setattr(web.routes, "gateway_request", fake)
    monkeypatch.setattr(web.server, "gateway_request", fake)
    return calls


@pytest.fixture()
def client(web):
    # The operator's boot: an admin sees every conversation.
    web.server.auth_store.create_user(OWNER, "pw", replace=True, role="admin")
    c = web.server.app.test_client()
    r = c.post("/api/login", json={"email": OWNER, "password": "pw"})
    assert r.status_code == 200, r.get_data(as_text=True)
    return c


def _boot(client, query):
    r = client.get("/api/boot" + query)
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def test_boot_carries_the_rendered_agents_saved_defaults(gateway, client):
    """A New Chat draft asks about its picker agent: boot answers with the
    gateway's resolved default — only effort + model, nothing else of the def.
    Breaks if the defaults come from anywhere but the gateway def (this home
    has no agents on disk) or if the def's other fields leak into boot."""
    body = _boot(client, "?session=&agent=jarvis")
    assert body["agent_defaults"] == {
        "jarvis": {"effort": "xhigh", "model": "qwen3.8-27b-fp8"},
    }


def test_boot_adds_the_resumed_sessions_own_agent(web, gateway, client):
    """A reload of an custom_helper conversation from a browser whose picker says
    jarvis: both agents' defaults ride the read, so the page needs no
    /api/agents/defs call whichever agent it ends up rendering."""
    sid = web.server.store.create(OWNER, "custom_helper", "", "")["id"]
    body = _boot(client, f"?session={sid}&agent=jarvis")
    assert body["session"]["id"] == sid
    assert body["agent_defaults"] == {
        "jarvis": {"effort": "xhigh", "model": "qwen3.8-27b-fp8"},
        "custom_helper": {"effort": "high", "model": "gpt-6-sol"},
    }


@pytest.mark.parametrize("bad", ["../../etc", "jarvis/../x", "a b", "%2e%2e", ""])
def test_boot_refuses_unsafe_agent_names(gateway, client, bad):
    """``?agent=`` is client input spliced into a gateway path: anything that
    is not a plain agent name is dropped before any gateway call."""
    from urllib.parse import quote

    body = _boot(client, "?session=&agent=" + quote(bad, safe=""))
    assert body.get("agent_defaults") in ({}, None)
    assert not [c for c in gateway if c["path"].startswith("/control/v1/agents/defs")]


def test_boot_lists_match_the_rest_endpoints_they_replace(web, gateway, client):
    """With the menu visible at load the client asks for the lists; they must
    be exactly what the lazy loaders would have fetched — gateway (TUI) rows
    included, newest first. Breaks on a web-store-only list, a different
    limit, or topics grouped differently from /api/topics."""
    web.server.store.create(OWNER, "jarvis", "", "")
    body = _boot(client, "?session=&agent=jarvis&lists=conversations,topics")
    boot_reads = [c for c in gateway if c["path"] == "/control/v1/orchestrator/sessions"]
    rest_sessions = client.get("/api/sessions?limit=20").get_json()["sessions"]
    rest_topics = client.get("/api/topics").get_json()["topics"]
    assert body["sidebar"]["sessions"] == rest_sessions
    assert len(rest_sessions) == 20
    assert TUI_ROOT["hash"] in [s["id"] for s in body["sidebar"]["sessions"]]
    assert body["sidebar"]["topics"] == rest_topics
    assert {"ops", "research", "sharedagents"} <= {t["topic"] for t in body["sidebar"]["topics"]}
    # Both lists from ONE gateway roots read (the REST pair makes two).
    assert len(boot_reads) == 1


def test_boot_without_lists_does_not_build_them(gateway, client):
    """Menu hidden (the default, and always on phones): no lists in the read
    and no gateway roots scan paid for them."""
    body = _boot(client, "?session=&agent=jarvis")
    assert "sidebar" not in body
    assert not [c for c in gateway if c["path"] == "/control/v1/orchestrator/sessions"]
