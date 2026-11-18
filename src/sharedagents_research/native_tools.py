"""Native research tools: trusted turn context, automatic source capture and publication."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .ontology import canonical_url, encoded, identity
from .storage_client import configuration, open_runtime_ontology


def runtime_context():
    """Resolve scope from pinned authorization, never model-supplied identities."""
    from sharedagents_session.access_binding import session_access_stamp

    session = os.environ.get("SHAREDAGENTS_SESSION_ID", "")
    turn = os.environ.get("SHAREDAGENTS_TURN_ID", "")
    if not session or not turn:
        raise PermissionError("ontology tools require a native session and turn context")
    raw_pin = os.environ.get("SHAREDAGENTS_ACCESS_PIN")
    if raw_pin:
        try:
            pin = json.loads(raw_pin)
        except ValueError as exc:
            raise PermissionError("ontology authorization pin is malformed") from exc
        if not isinstance(pin, dict) or pin.get("sid") != session or "stamp" not in pin:
            raise PermissionError("ontology authorization pin does not match this session")
    stamp = session_access_stamp(os.environ.get("SHAREDAGENTS_SESSIONS_DIR", ""), session)
    if stamp is not None:
        if not stamp.email:
            raise PermissionError("ontology user identity unavailable")
        scope = "user:" + stamp.email
    else:
        scope = "operator"
    root = Path(
        os.environ.get("SHAREDAGENTS_ONTOLOGY_ROOT")
        or configuration().get("root")
        or Path(os.environ.get("SHAREDAGENTS_HOME", str(Path.home() / ".sharedagents"))) / "ontology"
    )
    return {
        "root": root,
        "scope": scope,
        "actor": scope,
        "run_id": identity(scope, session, turn),
        "model": os.environ.get("SHAREDAGENTS_UPSTREAM_MODEL", ""),
        "agent": os.environ.get("SHAREDAGENTS_AGENT_NAME", ""),
        "session_id": session,
        "turn_id": turn,
    }


def context_store(*, create=False):
    context = runtime_context()
    return context, open_runtime_ontology(context["root"], context["scope"], create=create)


def search(arguments):
    """Read ontology state. A missing store is explicit and never auto-created."""
    action = arguments.get("action", "search")
    try:
        _, store = context_store()
    except FileNotFoundError:
        if action == "status":
            return {"ready": False, "reason": "Begin the first research run to create an empty ontology"}
        if action in {"search", "timeline", "history"}:
            return []
        raise ValueError("ontology is empty; begin a research run first") from None
    if action == "status":
        return store.status()
    if action == "get":
        return store.get(arguments["id"], offset=arguments.get("offset", 0), text_limit=16000)
    if action == "timeline":
        return store.timeline(limit=arguments.get("limit", 100))
    if action == "describe":
        return store.describe(arguments["id"], limit=arguments.get("limit", 50))
    if action == "history":
        return store.export(after=arguments.get("after", 0), limit=arguments.get("limit", 20))
    if action != "search":
        raise ValueError("unknown ontology read action")
    return store.search(arguments.get("query", ""), limit=arguments.get("limit", 20), kind=arguments.get("kind"))


async def capture_source(arguments):
    """Fetch using the established web tool and persist exactly the returned evidence."""
    from sharedagents_agent.tools.impl.search import run_web_fetch_async

    context, store = context_store(create=True)
    # Fail before network egress when the caller did not begin its run.
    store.check_open(context["run_id"], context["actor"])
    url = canonical_url(arguments["url"])
    result = await run_web_fetch_async({"url": url, "format": "markdown"})
    if not result.ok:
        raise ValueError(result.content)
    source = store.capture(
        context["run_id"],
        context["actor"],
        url=arguments["url"],
        text=result.content,
        published_at=arguments.get("published_at"),
        truncated=result.truncated,
    )
    return {
        "source_id": source["id"],
        "url": source["url"],
        "text": source["text"][:16000],
        "next_offset": 16000 if len(source["text"]) > 16000 else None,
        "truncated": source["truncated"],
        "extraction_required": True,
    }


def update(arguments):
    context, store = context_store(create=True)
    action = arguments["action"]
    if action == "begin":
        if not context["model"]:
            raise ValueError("native runtime must stamp the research model before beginning")
        return store.begin(
            context["run_id"],
            actor=context["actor"],
            question=arguments["question"],
            model=context["model"],
            prompt_version="research-ontology-v1",
            agent=context["agent"],
            session_id=context["session_id"],
            turn_id=context["turn_id"],
        )
    if action == "extract":
        return store.extract(
            context["run_id"],
            context["actor"],
            arguments["source_id"],
            entities=arguments.get("entities", []),
            concepts=arguments.get("concepts", []),
            claims=arguments.get("claims", []),
        )
    if action == "finish":
        receipt = store.finish(
            context["run_id"],
            context["actor"],
            answer=arguments["answer"],
            citations=arguments.get("citations", []),
            limitations=arguments.get("limitations", []),
        )
        return {
            "published": True,
            "revision": receipt["revision"],
            "sources": receipt["source_ids"],
            "facts_updated": len(receipt["states"]),
        }
    raise ValueError("unknown ontology mutation action")


def publication_required(profile, calls, results):
    """Only a real successful finish in this turn satisfies the completion gate."""
    if profile != "research":
        return False
    # Provider call IDs can repeat between rounds. Decisions use the executed
    # result's action, never an ID join that could pair a failed finish with an
    # earlier successful begin. A new successful mutation invalidates the receipt.
    required = True
    for result in results:
        if result.get("ok") is not True:
            continue
        if result.get("name") == "ontology_source":
            required = True
        elif result.get("name") == "ontology_update":
            required = result.get("action") != "finish"
    return required


def schemas():
    """Small native schemas shared by the tool registry and MCP facade."""
    name = {"type": "string"}
    item = {
        "type": "object",
        "properties": {
            "key": name,
            "name": name,
            "type": name,
            "external_id": name,
            "description": name,
            "aliases": {"type": "array", "items": name},
            "quote": name,
        },
        "required": ["key", "name", "quote"],
        "additionalProperties": False,
    }
    claim = {
        "type": "object",
        "properties": {
            "subject": name,
            "predicate": name,
            "value": {},
            "quote": name,
            "unit": name,
            "period": name,
            "basis": name,
            "event_at": name,
            "corrects": name,
        },
        "required": ["subject", "predicate", "value", "quote"],
        "additionalProperties": False,
    }
    return {
        "ontology_search": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["status", "search", "get", "describe", "timeline", "history"]},
                "query": name,
                "id": name,
                "kind": name,
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "offset": {"type": "integer", "minimum": 0},
                "after": {"type": "integer", "minimum": 0},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "ontology_source": {
            "type": "object",
            "properties": {"url": name, "published_at": name},
            "required": ["url"],
            "additionalProperties": False,
        },
        "ontology_update": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["begin", "extract", "finish"]},
                "question": name,
                "source_id": name,
                "answer": name,
                "entities": {"type": "array", "items": item, "maxItems": 100},
                "concepts": {"type": "array", "items": item, "maxItems": 100},
                "claims": {"type": "array", "items": claim, "maxItems": 100},
                "citations": {"type": "array", "items": name},
                "limitations": {"type": "array", "items": name},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }


async def run_tool(name, arguments):
    """Native runner adapter; storage failures remain tool failures."""
    from sharedagents_agent.tools.impl._utils import ToolResult

    try:
        if name == "ontology_source":
            value = await capture_source(arguments)
        elif name == "ontology_update":
            value = update(arguments)
        else:
            value = search(arguments)
        return ToolResult(ok=True, content=encoded(value))
    except (ValueError, KeyError, TypeError, PermissionError, RuntimeError, OSError, sqlite3.Error) as exc:
        return ToolResult(
            ok=False,
            content=json.dumps({"error": str(exc), "operation_applied": False, "publication_confirmed": False}),
        )
