"""Capture the shipped WebUI with isolated demo data and no live integrations.

Requires the web extra, Playwright, and Chromium. Every browser request is
handled locally from packaged templates/assets or the explicit fixtures below.
No application server, provider account, or production authentication is used.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "sharedagents_web"
STAMP = "2026-01-05T12:00:00Z"
ORIGIN = "https://docs.example"


def demo_cards() -> list[dict]:
    """Explicit illustrative values, unrelated to any real account."""
    cards = []
    for provider, name, pace in [("codex", "Codex", 30), ("claude", "Claude", 60), ("cursor", "Cursor", 85)]:
        cards.append(
            {
                "id": provider,
                "name": name,
                "kind": "subscription",
                "ok": True,
                "plan": "Demo plan",
                "updated_at": STAMP,
                "pace_status": "ok",
                "pace_used_percent": pace,
                "windows": [{"id": "weekly", "label": "Weekly", "used_percent": pace, "remaining_percent": 100 - pace}],
            }
        )
    for provider, name, balance in [("openrouter", "OpenRouter", 75), ("deepseek", "DeepSeek", 40)]:
        cards.append(
            {
                "id": provider,
                "name": name,
                "kind": "balance",
                "ok": True,
                "updated_at": STAMP,
                "windows": [{"id": "balance", "label": "Balance", "unit": "usd", "remaining": balance, "limit": 100}],
            }
        )
    return cards


def capture(output: Path) -> dict:
    env = Environment(loader=FileSystemLoader(WEB / "templates"), autoescape=select_autoescape())
    env.globals["asset_rev"] = lambda _name: "docs"
    cards = demo_cards()
    usage_rows = []
    day = datetime(2026, 1, 5, tzinfo=timezone.utc)
    for offset in range(7):
        for provider, scale in [("Local demo", 1), ("Hosted demo", 2)]:
            usage_rows.append(
                {
                    "bucket": (day - timedelta(days=6 - offset)).date().isoformat(),
                    "provider": provider,
                    "model": "example-model",
                    "tokens": (offset + 2) * 4000 * scale,
                    "requests": (offset + 2) * 3 * scale,
                    "cost_pico": (offset + 2) * 10**9 * scale,
                    "cost_usd": (offset + 2) * 0.001 * scale,
                }
            )
    agents = [{"name": "jarvis"}, {"name": "build"}, {"name": "review"}]
    models = [{"id": "example-model", "alias": "example-model", "label": "Local model (demo)", "provider": "vllm"}]
    topics = [{"topic": "Documentation", "count": 3}, {"topic": "Getting started", "count": 2}]
    tasks = [
        {
            "id": "demo-1",
            "title": "Review the contribution guide",
            "card": "needs_input",
            "state": "paused",
            "agent": "review",
        },
        {
            "id": "demo-2",
            "title": "Add a focused regression test",
            "card": "working",
            "state": "tool_running",
            "agent": "build",
        },
        {
            "id": "demo-3",
            "title": "Set up the development environment",
            "card": "completed",
            "state": "idle",
            "agent": "jarvis",
        },
    ]
    for item in tasks:
        item.update(session_id=item["id"], topic="Documentation", updated_at=STAMP)
    phases = [
        {"n": number, "name": name, "gates": gates, "note": ""}
        for number, name, gates in [
            (1, "Identify and gather requirements", ["describe", "clarify", "contract"]),
            (2, "Plan", ["plan"]),
            (3, "Design", ["design"]),
            (4, "Implement", ["code"]),
            (5, "Test", ["test", "verify"]),
            (6, "Deploy", ["deliver"]),
            (7, "Maintain", []),
        ]
    ]
    pages = {"/": "index.html", "/tasks": "tasks.html", "/usage": "usage.html", "/dev": "dev.html"}
    requests_seen, errors, external_requests = set(), [], []

    def route(request):
        parsed = urlsplit(request.request.url)
        path = parsed.path
        if parsed.netloc != "docs.example":
            external_requests.append(parsed.netloc)
            request.abort()
            return
        requests_seen.add(path)
        if path.startswith("/static/"):
            file = (WEB / path.lstrip("/")).resolve()
            if not file.is_relative_to((WEB / "static").resolve()) or not file.is_file():
                request.fulfill(status=404, body="")
                return
            request.fulfill(path=file, content_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            return
        if path in pages:
            html = env.get_template(pages[path]).render(
                base="",
                public_base=ORIGIN,
                site_title="SharedAgents",
                is_admin=True,
                can_manage_tasks=True,
                current_user_name="Demo",
                current_user_email="user@example.com",
                disable_sse=True,
                realtime_url="",
                phases=phases,
                forge_configured=False,
            )
            request.fulfill(content_type="text/html", body=html)
            return
        data = {}
        if path == "/api/boot":
            data = {
                "session": None,
                "agents": agents,
                "agent_defaults": {a["name"]: {"model": "example-model", "effort": "medium"} for a in agents},
                "health": {},
                "approvals": [],
                "sidebar": {"sessions": [], "topics": topics},
            }
        elif path == "/api/models":
            data = {"models": agents}
        elif path == "/api/models/discover":
            data = {"models": models}
        elif path == "/api/topics":
            data = {"topics": topics}
        elif path == "/api/tasks":
            data = {"items": tasks, "snapshot": {"topics": topics, "sessions": [], "approvals": []}}
        elif path == "/api/tasks/summary-model":
            data = {"models": [], "selected": ""}
        elif path == "/api/usage/subscriptions/catalog":
            data = {
                "source": "fixture",
                "providers": [{"id": p["id"], "name": p["name"], "kind": p["kind"]} for p in cards],
            }
        elif path.startswith("/api/usage/subscriptions/"):
            card = next((p for p in cards if p["id"] == path.rsplit("/", 1)[-1]), {})
            data = {"source": "fixture", "updated_at": STAMP, "provider": card}
        elif path == "/api/usage/agents/health":
            data = {"agents": [], "alerts": [], "updated_at": STAMP}
        elif path == "/api/usage":
            data = {"generated_at": STAMP, "breakdown": {"daily": usage_rows, "hourly": []}, "baseline": {}}
        elif path == "/api/dev/repos":
            data = {"repos": []}
        elif path == "/api/dev/flows":
            data = {"flows": []}
        elif path == "/api/sessions":
            data = {"sessions": []}
        elif path == "/manifest.webmanifest":
            data = {"name": "SharedAgents", "short_name": "SharedAgents", "start_url": "/"}
        request.fulfill(json=data)

    shots = [
        ("welcome-desktop.png", "/", 1440, 900),
        ("tasks-desktop.png", "/tasks", 1440, 900),
        ("tasks-mobile.png", "/tasks", 390, 844),
        ("subscriptions-mobile.png", "/usage#subscriptions", 390, 844),
        ("api-mobile.png", "/usage#api", 390, 844),
        ("activity-desktop.png", "/usage", 1440, 900),
        ("development-desktop.png", "/dev", 1440, 1000),
    ]
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            for name, path, width, height in shots:
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=2,
                    service_workers="block",
                    locale="en-US",
                    timezone_id="UTC",
                )
                try:
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.route("**/*", route)
                    page.clock.install(time=1767614400000)
                    page.goto(ORIGIN + path)
                    page.wait_for_load_state("networkidle")
                    if path.startswith("/tasks"):
                        page.locator(".task-row").first.wait_for()
                    if "#subscriptions" in path or "#api" in path:
                        tab = path.rsplit("#", 1)[-1]
                        page.locator(f"[data-usage-tab='{tab}']").click()
                        page.locator(".usage-sub-card:visible").first.wait_for()
                    if path.startswith("/usage"):
                        page.locator("#usage-chart svg").wait_for(state="attached")
                        alert = page.locator("#usage-error")
                        if alert.count() and alert.is_visible():
                            raise RuntimeError(alert.inner_text())
                    page.evaluate("""() => {
                      const label = document.createElement('div'); label.textContent = 'DEMO DATA';
                      label.style.cssText = 'position:fixed;bottom:8px;right:8px;z-index:99999;padding:5px 9px;border:1px solid #335844;border-radius:7px;background:#101b16;color:#b8dfc8;font:11px system-ui;letter-spacing:.08em;pointer-events:none';
                      document.body.append(label);
                    }""")
                    page.screenshot(path=output / name, animations="disabled")
                finally:
                    context.close()
        finally:
            browser.close()
    report = {
        "captures": [name for name, *_ in shots],
        "fixture_data": True,
        "page_errors": errors,
        "external_requests": external_requests,
    }
    if errors or external_requests:
        raise RuntimeError(json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets/demo")
    args = parser.parse_args()
    print(json.dumps(capture(args.out), indent=2))


if __name__ == "__main__":
    main()
