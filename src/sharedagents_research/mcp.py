"""Authenticated MCP access to the same ontology and native research HTTP APIs.

Run as an owner-configured stdio client. The token is resolved from the approved
runtime environment; all authorization and user scoping stay in the HTTP server.
"""

import asyncio
import os
from time import monotonic
from urllib.parse import urlsplit

from sharedagents_mcp._stdio_server import ToolError, run_stdio_server

from .ontology import encoded


class Client:
    def __init__(self):
        import httpx

        base = os.environ.get("SHAREDAGENTS_RESEARCH_API_URL", "").rstrip("/")
        token = os.environ.get("SHAREDAGENTS_RESEARCH_API_TOKEN", "")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or not token:
            raise ToolError("Configure the research API URL and an authenticated user token")
        self.http = httpx.AsyncClient(
            base_url=base, headers={"Authorization": "Bearer " + token}, timeout=30, follow_redirects=False
        )

    async def call(self, method, path, **kwargs):
        response = await self.http.request(method, path, **kwargs)
        if response.status_code >= 300:
            raise ToolError("Authenticated research API request failed: HTTP " + str(response.status_code))
        return response.json()


async def dispatch(name, arguments):
    client = Client()
    try:
        if name == "ask_question":
            question = arguments.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ToolError("research question required")
            accepted = await client.call("POST", "api/ontology/research", json={"question": question})
            sid = accepted["session_id"]
            deadline = monotonic() + 300
            while monotonic() < deadline:
                published = await client.call(
                    "GET", "api/ontology", params={"action": "publications", "session_id": sid}
                )
                if published.get("value"):
                    return encoded(
                        {"session_id": sid, "published": True, "research": published["value"][0]["research"]}
                    )
                state = await client.call("GET", "api/sessions/" + sid + "/state")
                current = state.get("state") or state.get("session_state")
                if current in {"ask", "permission", "error", "cancelled", "aborted"}:
                    return encoded({"session_id": sid, "completed": False, "state": current})
                await asyncio.sleep(1)
            return encoded(
                {
                    "session_id": sid,
                    "completed": False,
                    "state": "running",
                    "reason": "research is still running; inspect the native session",
                }
            )
        mapping = {
            "ontology_status": "status",
            "search_ontology": "search",
            "get_terms": "list",
            "get_term": "get",
            "check_answerable": "coverage",
            "get_join_path": "path",
            "export_model": "export",
        }
        params = {"action": mapping[name]}
        if name == "get_terms":
            params["kind"] = "term"
        params.update({key: arguments[key] for key in ("id", "target", "limit") if key in arguments})
        if "query" in arguments:
            params["q"] = arguments["query"]
        if "required" in arguments:
            params["term"] = arguments["required"]
        if name == "export_model":
            params["format"] = arguments.get("format", "native")
        result = await client.call("GET", "api/ontology", params=params)
        return encoded(result["value"])
    finally:
        await client.http.aclose()


def main():
    names = (
        "ontology_status",
        "search_ontology",
        "get_terms",
        "get_term",
        "check_answerable",
        "get_join_path",
        "export_model",
        "ask_question",
    )
    properties = {
        "query": {"type": "string"},
        "question": {"type": "string"},
        "id": {"type": "string"},
        "target": {"type": "string"},
        "required": {"type": "array", "items": {"type": "string"}},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "format": {"type": "string", "enum": ["native", "nvidia", "ossie"]},
    }
    definitions = [
        {
            "name": name,
            "description": "Native research ontology: " + name.replace("_", " "),
            "inputSchema": {"type": "object", "properties": properties, "additionalProperties": False},
            "annotations": {"readOnlyHint": name != "ask_question"},
        }
        for name in names
    ]
    handlers = {name: (lambda arguments, name=name: dispatch(name, arguments)) for name in names}
    run_stdio_server("research-ontology", "1.0.0", definitions, handlers)


if __name__ == "__main__":
    main()
