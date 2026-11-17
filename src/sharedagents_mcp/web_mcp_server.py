"""sharedagents web-research MCP server.

Re-exposes the two web tools that used to be builtins — ``WebSearch`` (Brave
Search API) and ``WebFetch`` (URL → markdown) — as ``mcp__web__search`` /
``mcp__web__fetch``. It is a thin wrapper: each handler calls the SAME runner
the builtin used (``run_web_search_async`` / ``run_web_fetch_async`` in
``sharedagents_agent.tools.runners``), so behavior, the Brave-key resolution
(``BRAVE_API_KEY`` / config, inherited via the tool_runner env), and the
markdown extraction are unchanged.

Both tools declare ``readOnlyHint`` — they are read-only GETs. That maps to
READ criticality in discovery, so plan mode allows web research under the
general read-only rule (retiring the old ``_PLAN_NETWORK_ALLOW`` name-list).

Run: ``python -m sharedagents_mcp.web_mcp_server`` (stdio; configured under
``mcpServers.web``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from sharedagents_mcp._stdio_server import result_text, run_stdio_server


def _search(arguments: dict[str, Any]) -> str:
    from sharedagents_agent.tools.runners import run_web_search_async

    return result_text(asyncio.run(run_web_search_async(arguments)))


def _fetch(arguments: dict[str, Any]) -> str:
    from sharedagents_agent.tools.runners import run_web_fetch_async

    return result_text(asyncio.run(run_web_fetch_async(arguments)))


TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search the web via the Brave Search API. Returns a ranked list of "
            "{title, url, description}. Requires a Brave API key (BRAVE_API_KEY "
            "env / config `brave_api_key`)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "count": {
                    "type": "integer",
                    "description": "Number of results (1-10, default 5).",
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict results to these domains.",
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Drop results from these domains.",
                },
                "freshness": {"type": "string", "description": "e.g. pd/pw/pm/py."},
                "country": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": True},
    },
    {
        "name": "fetch",
        "description": (
            "Fetch a URL and return its content. Optional `prompt` is an "
            "extraction hint. HTML → markdown (main article body only); "
            "JSON/text/CSV pass through unchanged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "prompt": {
                    "type": "string",
                    "description": "Optional extraction hint (what to pull out).",
                },
            },
            "required": ["url"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": True},
    },
]

TOOL_DISPATCH = {"search": _search, "fetch": _fetch}


if __name__ == "__main__":
    run_stdio_server("web", "1.0.0", TOOL_DEFS, TOOL_DISPATCH)
