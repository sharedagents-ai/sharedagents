"""Extracted tool-runner family: search (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from sharedagents_model.sampling import apply_sampling, configured_sampling, sampling_row

from ._utils import (
    ToolResult,
    _err,
    _truncate,
)

# ---- network ----

WEBFETCH_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB — opencode webfetch parity
WEBFETCH_DEFAULT_TIMEOUT = 30.0  # seconds
WEBFETCH_MAX_TIMEOUT = 120.0  # seconds — opencode caps at 120
WEBFETCH_MAX_REDIRECTS = 5  # a Web user's fetch follows redirects hop by hop


def _webfetch_timeout(args: dict[str, Any]) -> float:
    """Resolve opencode's ``timeout`` (seconds, max 120), default 30."""
    raw = args.get("timeout")
    try:
        secs = float(raw) if raw is not None else WEBFETCH_DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        secs = WEBFETCH_DEFAULT_TIMEOUT
    if secs <= 0:
        secs = WEBFETCH_DEFAULT_TIMEOUT
    return min(secs, WEBFETCH_MAX_TIMEOUT)


def _turn_is_restricted() -> bool:
    """Whether this runner serves a Web user's turn."""
    sessions_dir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR", "")
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID", "")
    if not (sessions_dir and sid):
        return False
    from sharedagents_session.access_binding import session_access_grants

    return session_access_grants(sessions_dir, sid) is not None


def _host_addresses(host: str) -> list[str]:
    """Every address ``host`` resolves to (an IP literal is itself)."""
    import ipaddress
    import socket

    name = str(host or "").strip("[]")
    try:
        ipaddress.ip_address(name.split("%", 1)[0])
        return [name]
    except ValueError:
        pass
    return sorted({info[4][0] for info in socket.getaddrinfo(name, None, proto=socket.IPPROTO_TCP)})


def _public_address(addr: str) -> bool:
    import ipaddress

    try:
        ip = ipaddress.ip_address(str(addr).split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def _user_fetch_refusal(url: str) -> str | None:
    """Why a Web user's turn may not fetch ``url``, else ``None``.

    Only globally routable destinations: loopback, private, link-local,
    shared (CGNAT) and reserved addresses are this machine's and its LAN's
    services, which a user reaches through their grants, not a fetch.
    Residual: a DNS answer that changes between this check and the connect.
    """
    import httpx

    # The host exactly as HTTPX will connect to it (IDNA 2008 A-label):
    # resolving the Unicode name would let Python's IDNA 2003 map it to a
    # different host ("faß" -> "fass").
    try:
        host = httpx.URL(url).raw_host.decode("ascii")
    except Exception:  # noqa: BLE001 — not a URL HTTPX would fetch
        return "the URL's host is not valid"
    if not host:
        return "the URL has no host"
    try:
        addrs = await asyncio.to_thread(_host_addresses, host)
    except OSError as exc:
        return f"cannot resolve {host}: {exc}"
    if not addrs or not all(_public_address(a) for a in addrs):
        return f"{host} is a private or local address; not allowed for this user"
    return None


async def run_web_fetch_async(args: dict[str, Any]) -> ToolResult:
    """Fetch a URL and return the content in the requested ``format``.

    Mirrors opencode's ``webfetch`` (tool/webfetch.ts): the CONTENT is
    returned directly — there is NO LLM summarization (the Claude-Code
    ``WebFetch`` shape took a ``prompt`` and summarised; that param is
    dropped here). ``format`` is one of ``text`` / ``markdown`` (default)
    / ``html``:

      * ``html``     — the raw response body, unmodified.
      * ``markdown`` — HTML/XML run through trafilatura → markdown
        (headings/lists/tables/links preserved); non-HTML passes through.
      * ``text``     — HTML/XML run through trafilatura → plain text;
        non-HTML passes through.

    HTTP URLs are upgraded to HTTPS. 5 MB response cap. ``timeout`` is in
    seconds (max 120, default 30).
    """
    import httpx

    url = args.get("url")
    if not isinstance(url, str) or not url:
        return _err("url required")
    # opencode requires an http(s) scheme and upgrades http → https.
    if not (url.startswith("http://") or url.startswith("https://")):
        return _err("URL must start with http:// or https://")
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    fmt = str(args.get("format") or "markdown").strip().lower()
    if fmt not in ("text", "markdown", "html"):
        fmt = "markdown"
    timeout = _webfetch_timeout(args)
    # A Web user's turn fetches only public destinations, every redirect hop
    # checked before it is contacted.
    restricted = _turn_is_restricted()
    if restricted:
        refusal = await _user_fetch_refusal(url)
        if refusal:
            return _err(f"[access] {refusal}")
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=not restricted) as c:
            for _hop in range(WEBFETCH_MAX_REDIRECTS + 1):
                # Stream the response so a misdirected fetch at a
                # 500 MB target is aborted at the cap rather than
                # materialised into memory before the truncation kicks in.
                async with c.stream("GET", url) as r:
                    if restricted and r.is_redirect:
                        url = str(r.url.join(r.headers.get("location") or ""))
                        refusal = await _user_fetch_refusal(url)
                        if refusal:
                            return _err(f"[access] {refusal}")
                        continue
                    r.raise_for_status()
                    ctype = r.headers.get("content-type", "").lower()
                    buf = bytearray()
                    async for chunk in r.aiter_bytes():
                        if not chunk:
                            continue
                        if len(buf) + len(chunk) > WEBFETCH_MAX_BYTES:
                            room = WEBFETCH_MAX_BYTES - len(buf)
                            if room > 0:
                                buf.extend(chunk[:room])
                            # Stop reading; the body we've already pulled is
                            # what the caller gets. Surface the cap so the
                            # model knows the page was clipped before
                            # extraction ran.
                            return _err("Response too large (exceeds 5MB limit)")
                        buf.extend(chunk)
                    break
            else:
                return _err("too many redirects")
            try:
                raw = bytes(buf).decode("utf-8", errors="replace")
            except Exception as e:  # noqa: BLE001
                return _err(f"decode failed: {e}")
    except httpx.HTTPError as e:
        return _err(f"{e}")
    # ``html`` format: return the raw body unmodified (opencode returns the
    # untransformed HTML). ``markdown``/``text``: extract via trafilatura for
    # HTML/XML; non-HTML bodies (JSON, plain text, CSV) pass through unchanged
    # in every format.
    is_markup = "html" in ctype or "xml" in ctype
    if fmt == "html" or not is_markup:
        text = raw
    else:
        try:
            import trafilatura
        except ImportError:
            return _err(
                "trafilatura not installed — required for HTML extraction. "
                "It ships in the optional tooling: pip install 'sharedagents[full]'"
            )
        out_format = "txt" if fmt == "text" else "markdown"
        try:
            extracted = trafilatura.extract(
                raw,
                url=url,
                output_format=out_format,
                include_links=(fmt == "markdown"),
                include_tables=True,
                favor_precision=True,
            )
        except Exception as e:  # noqa: BLE001
            return _err(f"trafilatura extraction failed: {e}")
        if not extracted:
            return _err(f"no main content extracted from {url} (content-type: {ctype or 'unknown'}, {len(raw)} bytes)")
        text = extracted
    text, truncated = _truncate(text)
    return ToolResult(content=text, ok=True, truncated=truncated)


# ---- web search (Brave) ----

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def _resolve_brave_api_key() -> str:
    """Pick the Brave API key. Priority: `BRAVE_API_KEY` env > global config.

    The runner is a subprocess; reading the global config from disk
    keeps the secret out of the agent → runner request payload (which
    is otherwise the model-visible args dict).
    """
    env = os.environ.get("BRAVE_API_KEY")
    if env:
        return env
    try:
        from sharedagents_agent.agent import load_global_config

        return str(load_global_config().get("brave_api_key") or "")
    except Exception:  # noqa: BLE001
        return ""


def _websearch_num_results(args: dict[str, Any]) -> int:
    """Resolve opencode's ``numResults`` (default 8), clamped to Brave's range.

    Also accepts the legacy ``count`` alias so pre-parity callers keep working.
    """
    raw = args.get("numResults")
    if raw is None:
        raw = args.get("count")
    try:
        n = int(raw) if raw is not None else 8
    except (TypeError, ValueError):
        n = 8
    return max(1, min(20, n))


async def run_web_search_async(args: dict[str, Any]) -> ToolResult:
    """Web search via Brave, conformed to opencode's ``websearch`` schema.

    Params: ``query`` (required), ``numResults`` (default 8), plus the
    provider-shape fields ``livecrawl`` / ``type`` / ``contextMaxCharacters``
    (default 10000). sharedagents's backend is Brave, not opencode's Exa/Parallel,
    so ``livecrawl`` and ``type`` are accepted for schema parity but not sent
    to Brave (no equivalent knob); ``contextMaxCharacters`` bounds the returned
    result text. Output is the opencode-shaped title/url/snippet list. The
    Claude-Code ``allowed_domains``/``blocked_domains`` params are dropped.
    """
    import httpx

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return _err("query required")
    api_key = _resolve_brave_api_key()
    if not api_key:
        return _err(
            "Brave API key not configured. "
            "Run `sharedagents agent init` (or edit ~/.sharedagents/config.json's "
            "`brave_api_key`) or set the BRAVE_API_KEY env var."
        )
    count_int = _websearch_num_results(args)
    try:
        ctx_max = int(args.get("contextMaxCharacters") or 10000)
    except (TypeError, ValueError):
        ctx_max = 10000
    ctx_max = max(0, ctx_max)
    params: dict[str, Any] = {"q": query.strip(), "count": count_int}
    for k in ("freshness", "country", "language", "search_lang"):
        v = args.get(k)
        if isinstance(v, str) and v:
            params[k] = v
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(BRAVE_SEARCH_URL, headers=headers, params=params)
            if r.status_code == 401:
                return _err("Brave API: 401 unauthorized — bad or missing API key")
            if r.status_code == 429:
                return _err("Brave API: 429 rate-limited (free tier is ~1 qps / 1k qpm)")
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return _err(f"Brave API: {e}")
    results = ((data.get("web") or {}).get("results")) or []
    if not results:
        return ToolResult(
            content=f"# Web search: {query}\n\n(no results)",
            ok=True,
        )
    lines = [f"# Web search: {query}", ""]
    for i, item in enumerate(results, 1):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        desc = (item.get("description") or "").strip()
        lines.append(f"## {i}. {title}")
        if url:
            lines.append(url)
        if desc:
            lines.append("")
            lines.append(desc)
        lines.append("")
    body = "\n".join(lines)
    # opencode's ``contextMaxCharacters`` bounds the returned context string.
    if ctx_max and len(body) > ctx_max:
        body = body[:ctx_max]
        truncated = True
    else:
        body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)


# ---- X / Twitter via xAI Grok x_search ----

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
XAI_TWEETS_DEFAULT_MODEL = "grok-4.3"
XAI_TWEETS_TIMEOUT = 45.0
XAI_TWEETS_MAX_HOURS = 168


def _resolve_xai_api_key() -> str:
    env = os.environ.get("XAI_API_KEY")
    if env:
        return env.strip()
    try:
        from sharedagents_agent.agent import load_global_config

        keys = load_global_config().get("api_keys") or {}
        if isinstance(keys, dict) and keys.get("XAI_API_KEY"):
            return str(keys["XAI_API_KEY"]).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        path = "/etc/environment"
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("XAI_API_KEY=") and "=" in line:
                        return line.split("=", 1)[1].strip().strip("\"'")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _x_tweets_hours(args: dict[str, Any]) -> int:
    raw = args.get("hours")
    try:
        n = int(raw) if raw is not None else 24
    except (TypeError, ValueError):
        n = 24
    return max(1, min(XAI_TWEETS_MAX_HOURS, n))


def _x_tweets_backend_model(args: dict[str, Any] | None = None) -> str:
    """Pin the x_search worker to a Grok id, ignoring the session model.

    Native agents on Qwen/DeepSeek still call this builtin; if they echoed
    their own upstream id into ``args.model`` the xAI Responses call 400s.
    Only explicit ``grok-*`` ids (or ``XAI_TWEETS_MODEL``) may override.
    """
    env = str(os.environ.get("XAI_TWEETS_MODEL") or "").strip()
    if env.lower().startswith("grok-"):
        return env
    raw = str((args or {}).get("model") or "").strip()
    if raw.lower().startswith("grok-"):
        return raw
    return XAI_TWEETS_DEFAULT_MODEL


def _parse_x_tweets_response(data: dict[str, Any]) -> dict[str, Any]:
    """Pull narrative text + citation URLs out of an xAI Responses payload."""
    output = data.get("output") or []
    tool_calls = 0
    narrative = ""
    urls: list[str] = []

    def _add_url(u: str) -> None:
        u = (u or "").strip()
        if u and u not in urls:
            urls.append(u)

    for item in output:
        if not isinstance(item, dict):
            continue
        itype = str(item.get("type") or "")
        if itype in ("custom_tool_call", "function_call", "tool_call"):
            tool_calls += 1
        if "search" in itype or itype.startswith("x_"):
            tool_calls += 1
        if itype == "message":
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    text = str(content.get("text") or "")
                    if text:
                        narrative = text
                    for ann in content.get("annotations") or []:
                        if isinstance(ann, dict) and ann.get("url"):
                            _add_url(str(ann.get("url")))
        for key in ("citations", "sources"):
            block = item.get(key)
            if isinstance(block, list):
                for c in block:
                    if isinstance(c, dict):
                        _add_url(str(c.get("url") or ""))
                    elif isinstance(c, str):
                        _add_url(c)
    top = data.get("citations")
    if isinstance(top, list):
        for c in top:
            if isinstance(c, dict):
                _add_url(str(c.get("url") or ""))
            elif isinstance(c, str):
                _add_url(c)
    if not narrative:
        narrative = str(data.get("output_text") or "").strip()
    return {
        "narrative": narrative.strip(),
        "urls": urls,
        "tool_calls": tool_calls,
    }


async def run_x_tweets_async(args: dict[str, Any]) -> ToolResult:
    """Download live X posts via a host-side Grok x_search call.

    Independent of the session's chat model: Qwen/DeepSeek/etc. emit the
    builtin tool call; this runner always talks to api.x.ai itself.
    """
    from datetime import datetime, timedelta, timezone

    import httpx

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return _err("query required")
    api_key = _resolve_xai_api_key()
    if not api_key:
        return _err("XAI_API_KEY not configured. Set the env var; the session chat model does not need to be Grok.")
    hours = _x_tweets_hours(args)
    model = _x_tweets_backend_model(args)
    from_date = (datetime.now(timezone.utc) - timedelta(hours=hours)).date().isoformat()
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": (
                    f"Search X for the latest posts about: {query.strip()}\n"
                    f"Look back about {hours} hours. Prefer high-follower "
                    "verified sources. Return a 1-2 sentence narrative, then "
                    "list the cited posts as markdown bullets with url, "
                    "author handle, and a short quote."
                ),
            }
        ],
        "tools": [{"type": "x_search", "from_date": from_date}],
    }
    from sharedagents_core.kernel.config import load_global_config

    cfg = load_global_config()
    apply_sampling(
        payload,
        configured_sampling(
            cfg,
            sampling_row(cfg, "https://api.x.ai/v1", model),
            model,
            purpose="x_search",
        ),
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        async with httpx.AsyncClient(timeout=XAI_TWEETS_TIMEOUT) as c:
            r = await c.post(XAI_RESPONSES_URL, headers=headers, json=payload)
            if r.status_code == 401:
                return _err("xAI API: 401 unauthorized — check XAI_API_KEY")
            if r.status_code == 429:
                return _err("xAI API: 429 rate-limited")
            if r.status_code >= 400:
                return _err(f"xAI API: HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
    except httpx.HTTPError as e:
        return _err(f"xAI API: {e}")
    parsed = _parse_x_tweets_response(data if isinstance(data, dict) else {})
    lines = [f"# X tweets: {query.strip()}", ""]
    if parsed["narrative"]:
        lines.append(parsed["narrative"])
        lines.append("")
    if parsed["urls"]:
        lines.append("## Cited posts")
        for u in parsed["urls"]:
            lines.append(f"- {u}")
        lines.append("")
    elif not parsed["narrative"]:
        return _err("xAI x_search returned no narrative and no post URLs")
    lines.append(f"_tool_calls={parsed['tool_calls']} lookback_h={hours} model={model}_")
    body, truncated = _truncate("\n".join(lines))
    return ToolResult(content=body, ok=True, truncated=truncated)


__all__ = [
    "WEBFETCH_MAX_BYTES",
    "WEBFETCH_DEFAULT_TIMEOUT",
    "WEBFETCH_MAX_TIMEOUT",
    "_webfetch_timeout",
    "run_web_fetch_async",
    "BRAVE_SEARCH_URL",
    "_resolve_brave_api_key",
    "_websearch_num_results",
    "run_web_search_async",
    "XAI_RESPONSES_URL",
    "XAI_TWEETS_DEFAULT_MODEL",
    "_resolve_xai_api_key",
    "_x_tweets_hours",
    "_x_tweets_backend_model",
    "_parse_x_tweets_response",
    "run_x_tweets_async",
]
