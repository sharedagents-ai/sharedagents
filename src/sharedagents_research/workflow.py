"""Thin typed workflow over MCP; never imports the store or a provider SDK.

The native research profile uses the same tools directly. Applications may use
this foreground client to run an explicit synthesis callback against frozen
evidence. The callback owns its model authorization and provider limits.
"""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sharedagents_mcp.mcp_client import MCPServerConfig, _mcp_attr, load_mcp_servers, mcp_session


class ResearchError(RuntimeError):
    pass


class KnowledgeClient:
    def __init__(self, session: Any, *, max_bytes: int = 5_000_000):
        self.session = session
        self.max_bytes = max_bytes

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        server: MCPServerConfig | None = None,
        *,
        timeout_s: float = 45,
        max_bytes: int = 5_000_000,
    ) -> AsyncIterator[KnowledgeClient]:
        server = server or load_mcp_servers().get("knowledge")
        if server is None:
            raise ResearchError("Configure the knowledge MCP server before running research")
        # One foreground session owns the complete workflow, including cleanup.
        async with mcp_session(server, timeout_s=timeout_s) as session:
            yield cls(session, max_bytes=max_bytes)

    async def call(self, name: str, **arguments: Any) -> dict:
        result = await self.session.call_tool(name, arguments=arguments)
        blocks = [
            block.text
            for block in (getattr(result, "content", None) or ())
            if getattr(block, "type", "") == "text" and isinstance(getattr(block, "text", None), str)
        ]
        if _mcp_attr(result, "isError", False):
            raise ResearchError("\n".join(blocks) or "Knowledge tool failed: " + name)
        structured = _mcp_attr(result, "structuredContent")
        if isinstance(structured, dict):
            value = structured
            if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > self.max_bytes:
                raise ResearchError("Knowledge result exceeds the configured input limit")
            return value
        raw = "\n".join(blocks)
        if len(raw.encode("utf-8")) > self.max_bytes:
            raise ResearchError("Knowledge result exceeds the configured input limit")
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ResearchError("Knowledge tool returned invalid JSON: " + name) from exc
        if not isinstance(value, dict):
            raise ResearchError("Knowledge tool must return an object: " + name)
        return value

    async def evidence(self, run_id: str) -> dict:
        """Read the complete frozen context, with an explicit size failure.

        Agent-driven use can page individual sections/source text instead. This
        callback interface refuses partial evidence rather than silently slicing.
        """
        return await self.call("get_evidence", run_id=run_id, section="all")


class ResearchWorkflow:
    def __init__(self, client: KnowledgeClient):
        self.client = client

    async def run(
        self,
        query: str,
        *,
        model: str,
        prompt_version: str,
        respond: Callable | None = None,
        request_id: str | None = None,
        related_to: str | None = None,
        start: str | None = None,
        end: str | None = None,
        last_hours: float | None = None,
        parameters: dict | None = None,
    ) -> dict:
        """Pin evidence, synthesize, validate and finish before returning.

        Missing time coverage abstains without invoking the callback. Without a
        callback the run stays open (status awaiting_synthesis) for the chosen agent.
        No web lookup, ingestion, fallback provider or hidden model call occurs.
        """
        client = self.client
        question = await client.call(
            "create_question", query=query, related_to=related_to, start=start, end=end, last_hours=last_hours
        )
        run = await client.call(
            "start_run",
            question_id=question["id"],
            model=model,
            prompt_version=prompt_version,
            request_id=request_id or uuid.uuid4().hex,
            parameters=parameters or {},
        )
        if run.get("completed_result"):
            result = await client.call("get_record", record_id=run["completed_result"])
            return {
                "question": question["id"],
                "run": run["id"],
                "result": result["id"],
                "evidence_digest": run["snapshot_digest"],
                "answer": result["payload"]["answer"],
            }
        snapshot = await client.evidence(run["id"])
        evidence = snapshot["evidence"]
        if respond is None:
            # The data layer never writes the answer: without a synthesis callback the run stays open for the
            # chosen agent, which reads the pinned evidence and calls finish_run itself.
            return {
                "question": question["id"],
                "run": run["id"],
                "result": None,
                "status": "awaiting_synthesis",
                "evidence_digest": snapshot["digest"],
                "evidence_status": evidence.get("status"),
                "matches": evidence.get("matches"),
            }
        if evidence["status"] == "no_coverage":
            answer = {
                "status": "abstained",
                "facts": [],
                "inferences": [],
                "limitations": [
                    "No observed evidence coverage for the requested time window; effects cannot be assessed."
                ],
            }
        else:
            answer = respond(snapshot)
            if inspect.isawaitable(answer):
                answer = await answer
        result = await client.call("finish_run", run_id=run["id"], answer=answer)
        return {
            "question": question["id"],
            "run": run["id"],
            "result": result["id"],
            "evidence_digest": snapshot["digest"],
            "answer": result["payload"]["answer"],
        }
