"""Native research tools: trusted turn context, automatic source capture and publication."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .ontology import canonical_url, encoded, identity
from .policy import authorization
from .storage_client import configuration, open_runtime_ontology


def runtime_context(*, operation="read"):
    """Resolve scope from pinned authorization, never model-supplied identities."""
    from sharedagents_session.access_binding import session_access_stamp, sidecar_path

    session = os.environ.get("SHAREDAGENTS_SESSION_ID", "")
    turn = os.environ.get("SHAREDAGENTS_TURN_ID", "")
    if not session or not turn:
        raise PermissionError("ontology tools require a native session and turn context")
    directory = os.environ.get("SHAREDAGENTS_SESSIONS_DIR", "")
    if not directory:
        raise PermissionError("ontology tools require the native sessions directory")
    try:
        sidecar_path(directory, session)
        sidecar_path(directory, turn)
    except ValueError as exc:
        raise PermissionError("ontology session or turn identity is invalid") from exc
    raw_pin = os.environ.get("SHAREDAGENTS_ACCESS_PIN")
    if raw_pin:
        try:
            pin = json.loads(raw_pin)
        except ValueError as exc:
            raise PermissionError("ontology authorization pin is malformed") from exc
        if not isinstance(pin, dict) or pin.get("sid") != session or "stamp" not in pin:
            raise PermissionError("ontology authorization pin does not match this session")
    stamp = session_access_stamp(directory, session)
    if stamp is not None:
        if not stamp.email:
            raise PermissionError("ontology user identity unavailable")
        scope = "user:" + stamp.email
    else:
        scope = "operator"
    permission = authorization(
        stamp.email if stamp is not None else "operator", operator=stamp is None, operation=operation
    )
    scope = permission["scope"]
    root = Path(
        os.environ.get("SHAREDAGENTS_ONTOLOGY_ROOT")
        or configuration().get("root")
        or Path(os.environ.get("SHAREDAGENTS_HOME", str(Path.home() / ".sharedagents"))) / "ontology"
    )
    return {
        "root": root,
        "scope": scope,
        "actor": permission["actor"],
        "role": permission["role"],
        "run_id": identity(scope, session, turn),
        "model": os.environ.get("SHAREDAGENTS_UPSTREAM_MODEL", ""),
        "agent": os.environ.get("SHAREDAGENTS_AGENT_NAME", ""),
        "session_id": session,
        "turn_id": turn,
    }


def context_store(*, create=False):
    context = runtime_context(operation="write" if create else "read")
    return context, open_runtime_ontology(context["root"], context["scope"], create=create)


def search(arguments):
    """Read ontology state. A missing store is explicit and never auto-created."""
    action = arguments.get("action", "search")
    try:
        context, store = context_store()
    except FileNotFoundError:
        if action == "status":
            return {"ready": False, "reason": "Begin the first research run to create an empty ontology"}
        if action in {"search", "timeline", "history"}:
            return []
        raise ValueError("ontology is empty; begin a research run first") from None
    if action == "status":
        return store.status()
    if action == "model":
        return (
            store.model_get(arguments["id"])
            if arguments.get("id")
            else store.model_list(
                kind=arguments.get("kind"), limit=arguments.get("limit", 100), after=arguments.get("after_id", "")
            )
        )
    if action == "retrieve":
        return store.model_retrieve(
            context["run_id"], context["actor"], arguments["query"], limit=arguments.get("limit", 20)
        )
    if action == "coverage":
        return store.model_coverage(arguments["required"])
    if action == "path":
        return store.model_path(arguments["id"], arguments["target"])
    if action == "model_export":
        from .interchange import export_model

        return export_model(store, arguments.get("format", "native"))
    if action == "connections":
        from .connectors import configured_sources

        return {"connections": sorted(configured_sources(context["scope"]))}
    if action == "analytics":
        return store.model_analytics()
    if action == "table":
        return store.table_context(arguments["id"], offset=arguments.get("offset", 0), limit=arguments.get("limit", 16))
    if action == "compilation_status":
        return store.compilation_status(arguments["connection"])
    if action == "model_history":
        return store.model_history(
            key=arguments.get("id"), after=arguments.get("after", 0), limit=arguments.get("limit", 100)
        )
    if action == "fact_evidence":
        return store.fact_evidence(
            arguments["id"],
            after=arguments.get("after_id", ""),
            limit=arguments.get("limit", 100),
            revision=arguments.get("revision"),
        )
    if action == "get":
        return store.get(arguments["id"], offset=arguments.get("offset", 0), text_limit=16000)
    if action == "timeline":
        return store.timeline(limit=arguments.get("limit", 100), basis=arguments.get("timeline_basis", "event"))
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
        published_at_verified=False,
        truncated=result.truncated,
    )
    embedder = _embedder()
    await _index_evidence(store, source["id"], embedder)
    return {
        "source_id": source["id"],
        "url": source["url"],
        "text": source["text"][:16000],
        "next_offset": 16000 if len(source["text"]) > 16000 else None,
        "truncated": source["truncated"],
        "extraction_required": True,
        "publisher": source["publisher"],
        "publication_time_status": "proposed" if source["published_at"] else "unknown",
        "retrieval_mode": "hybrid" if embedder else "lexical",
    }


def update(arguments):
    context, store = context_store(create=True)
    action = arguments["action"]
    if action in {"compile", "query", "define", "define_many", "model_import"}:
        store.check_open(context["run_id"], context["actor"])
        if action == "define":
            return store.model_put(
                arguments["kind"],
                arguments["data"],
                actor=context["actor"],
                key=arguments.get("id"),
                expected=arguments.get("expected_revision"),
            )
        if action == "define_many":
            return store.model_put_many(arguments["items"], actor=context["actor"])
        if action == "model_import":
            from .interchange import import_model

            return import_model(store, arguments["document"], actor=context["actor"])
        from .connectors import Source, validate_sql

        source = Source(arguments["connection"], scope=context["scope"])
        try:
            if action == "compile":
                catalog = source.catalog()
                return store.compile_catalog(
                    arguments["connection"],
                    catalog,
                    actor=context["actor"],
                    analysis=source.analyze(catalog) if arguments.get("analyze", False) else None,
                    query_history=[dict(row, dialect=source.dialect) for row in source.query_history()],
                )
            catalog = store.query_catalog(context["run_id"], context["actor"], arguments["connection"])
            validated = validate_sql(
                arguments["sql"],
                catalog,
                dialect=source.dialect,
                retrieved=set(store.retrieved_tables(context["run_id"], context["actor"])),
            )
            result = source.run(validated["sql"], limit=arguments.get("limit", 1000))
            return store.record_query(
                context["run_id"],
                context["actor"],
                connection=arguments["connection"],
                sql=validated["sql"],
                result=result,
                tables=validated["tables"],
            )
        finally:
            source.close()
    if action == "begin":
        if not context["model"]:
            raise ValueError("native runtime must stamp the research model before beginning")
        from sharedagents_agent.agent import agents_root

        prompt = agents_root() / context["agent"] / "AGENTS.md"
        prompt_version = (
            "research-ontology-v2:"
            + identity(prompt.read_text() if prompt.is_file() else "bundled-research-profile")[:16]
        )
        return store.begin(
            context["run_id"],
            actor=context["actor"],
            question=arguments["question"],
            model=context["model"],
            prompt_version=prompt_version,
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


def _embedder():
    from sharedagents_mcp.memory.engine.memory_embed import resolve_embedder

    return resolve_embedder()


def _embedding_space(embedder):
    return identity("embedding-space", embedder.provider, embedder.model, embedder.dims)


async def _index_evidence(store, source_id, embedder):
    rows = store.evidence_chunks(source_id)
    for offset in range(0, len(rows), 20):
        batch = rows[offset : offset + 20]
        if embedder:
            vectors = await embedder.embed_batch([row["text"] for row in batch])
            if len(vectors) != len(batch):
                raise ValueError("embedding service returned an incorrect batch size")
            batch = [dict(row, vector=vector) for row, vector in zip(batch, vectors)]
        store.index_evidence(batch, model=_embedding_space(embedder) if embedder else None)


async def _index_semantics(store, value):
    if "objects" in value:
        objects = value["objects"]
    elif "semantic_ids" in value:
        objects = [store.model_get(key) for key in value["semantic_ids"]]
    elif "ids" in value:
        objects = [store.model_get(key) for key in value["ids"]]
    else:
        objects = [value] if "revision" in value and "data" in value else []
    embedder = _embedder() if objects else None
    if not embedder:
        return "lexical"
    for offset in range(0, len(objects), 20):
        batch = objects[offset : offset + 20]
        texts = [
            " ".join(str(obj["data"].get(key, "")) for key in ("name", "description", "aliases", "synonyms", "sql"))[
                :6000
            ]
            for obj in batch
        ]
        vectors = await embedder.embed_batch(texts)
        if len(vectors) != len(batch):
            raise ValueError("embedding service returned an incorrect batch size")
        store.model_embed(
            [{"id": obj["id"], "revision": obj["revision"], "vector": vector} for obj, vector in zip(batch, vectors)],
            model=_embedding_space(embedder),
        )
    return "hybrid"


async def retrieve(arguments):
    context, store = context_store()
    embedder = _embedder()
    vector = await embedder.embed_query(arguments["query"]) if embedder else None
    model = _embedding_space(embedder) if embedder else None
    value = store.model_retrieve(
        context["run_id"],
        context["actor"],
        arguments["query"],
        limit=arguments.get("limit", 20),
        vector=vector,
        model=model,
    )
    value["evidence"] = store.evidence_search(
        arguments["query"], vector=vector, model=model, limit=arguments.get("limit", 20)
    )
    value["retrieval_mode"] = "hybrid" if embedder else "lexical"
    return value


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
            "cardinality": {"type": "string", "enum": ["one", "many"]},
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
            "object": name,
            "polarity": {"type": "boolean"},
            "modality": {"type": "string", "enum": ["actual", "planned", "expected", "possible", "rumored"]},
        },
        "required": ["subject", "predicate", "quote"],
        "oneOf": [{"required": ["value"]}, {"required": ["object"]}],
        "additionalProperties": False,
    }
    return {
        "ontology_search": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status",
                        "search",
                        "get",
                        "describe",
                        "timeline",
                        "history",
                        "model",
                        "retrieve",
                        "coverage",
                        "path",
                        "model_export",
                        "model_history",
                        "connections",
                        "fact_evidence",
                        "analytics",
                        "table",
                        "compilation_status",
                    ],
                },
                "query": name,
                "id": name,
                "kind": name,
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "offset": {"type": "integer", "minimum": 0},
                "after": {"type": "integer", "minimum": 0},
                "after_id": name,
                "target": name,
                "required": {"type": "array", "items": name, "maxItems": 64},
                "format": {"type": "string", "enum": ["native", "nvidia", "ossie"]},
                "revision": name,
                "connection": name,
                "timeline_basis": {"type": "string", "enum": ["event", "publication"]},
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
                "action": {
                    "type": "string",
                    "enum": ["begin", "extract", "finish", "compile", "query", "define", "define_many", "model_import"],
                },
                "question": name,
                "source_id": name,
                "answer": name,
                "entities": {"type": "array", "items": item, "maxItems": 100},
                "concepts": {"type": "array", "items": item, "maxItems": 100},
                "claims": {"type": "array", "items": claim, "maxItems": 100},
                "citations": {"type": "array", "items": name},
                "limitations": {"type": "array", "items": name},
                "connection": name,
                "sql": name,
                "analyze": {"type": "boolean"},
                "kind": name,
                "id": name,
                "expected_revision": name,
                "data": {"type": "object"},
                "items": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {"kind": name, "data": {"type": "object"}, "id": name, "expected_revision": name},
                        "required": ["kind", "data"],
                        "additionalProperties": False,
                    },
                },
                "document": {"type": "object"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
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
            if arguments["action"] in {"compile", "define", "define_many", "extract", "model_import"}:
                _, store = context_store()
                value["retrieval_mode"] = await _index_semantics(store, value)
                if arguments["action"] in {"compile", "define_many"}:
                    value["objects"] = [
                        {"id": obj["id"], "kind": obj["kind"], "name": obj["data"]["name"], "revision": obj["revision"]}
                        for obj in value["objects"]
                    ]
        elif arguments.get("action") == "retrieve":
            value = await retrieve(arguments)
        else:
            value = search(arguments)
        return ToolResult(ok=True, content=encoded(value))
    except (ValueError, KeyError, TypeError, PermissionError, RuntimeError, OSError, sqlite3.Error) as exc:
        return ToolResult(
            ok=False,
            content=json.dumps({"error": str(exc), "operation_applied": False, "publication_confirmed": False}),
        )
