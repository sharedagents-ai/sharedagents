#!/usr/bin/env python3
"""
Nodriver Browser — Drive a real headed Chromium from Bash.

Usage:
  browser.py [--headless] [--display :N] [--timeout SECS] <action> [--flags...]

Actions: goto, extract, click, fill, screenshot, inspect, eval, scroll,
         open, close, status, login, takeover, persist

Design notes (read before editing):
  * Attaching to a live session uses ``nd.start(host, port)`` — nodriver
    has NO ``connect()`` function. When both host and port are set it
    connects to an already-running Chrome instead of launching one.
  * The CDP port lives at ``browser.config.port`` (NOT ``browser.port``).
  * ``open`` double-forks into the background so the launching Bash call
    returns immediately with the port. The detached holder process keeps
    Chrome alive (if it dies, nodriver's atexit kills Chrome).
  * Chrome's X display comes from the ``DISPLAY`` env var, not a nodriver
    kwarg (nodriver ignores ``display=``). ``--display`` sets the env.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).parent.parent
PROFILES = BASE / "profiles"
SESSIONS = BASE / "sessions"
PROFILES.mkdir(parents=True, exist_ok=True)
SESSIONS.mkdir(parents=True, exist_ok=True)

import nodriver as nd  # noqa: E402

# Heuristic markers that a site served an anti-bot wall rather than content.
_BLOCK_MARKERS = (
    "just a moment",
    "verifying you are human",
    "checking your browser",
    "enable javascript and cookies to continue",
    "access denied",
    "attention required",
    "unusual traffic",
    "captcha",
)


def _load_session(session_name: str) -> dict | None:
    path = SESSIONS / f"{session_name}.live.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("pid"):
                try:
                    os.kill(data["pid"], 0)
                    return data
                except (OSError, ProcessLookupError):
                    pass  # stale
        except Exception:
            pass
    return None


def _save_session(session_name: str, pid: int, port: int, data: dict | None = None):
    path = SESSIONS / f"{session_name}.live.json"
    payload = {"session": session_name, "pid": pid, "port": port, "timestamp": time.time()}
    if data:
        payload.update(data)
    path.write_text(json.dumps(payload, indent=2))


def _remove_session(session_name: str):
    path = SESSIONS / f"{session_name}.live.json"
    if path.exists():
        path.unlink()


# --- one agent per profile ---------------------------------------------------
#
# A second agent attaching to the same Chrome resets the page the first agent
# (or the owner typing a 2FA code) is using. Each command made from an sharedagents
# session takes a lease on the profile, keyed by that session id
# (SHAREDAGENTS_SESSION_ID). A different session is refused while the holder is
# live. Manual use outside sharedagents (no session id) is never blocked.
_LEASE_IDLE_S = 15 * 60
_TERMINAL_JOB = {"done", "completed", "failed", "cancelled", "aborted"}


def _lease_path(session_name: str) -> Path:
    return SESSIONS / f"{session_name}.lease.json"


def _read_lease(session_name: str) -> dict | None:
    try:
        data = json.loads(_lease_path(session_name).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("owner") else None


def _holder_job_state(owner: str) -> str:
    """ "open" / "terminal" for the delegation job driving ``owner``, else ""."""
    home = Path(os.environ.get("SHAREDAGENTS_HOME") or Path.home() / ".sharedagents")
    states = []
    for f in (home / "jobs").glob("job_*.json"):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if rec.get("child_sid") == owner:
            states.append(str(rec.get("status") or ""))
    if not states:
        return ""
    return "open" if any(s not in _TERMINAL_JOB for s in states) else "terminal"


def _lease_live(lease: dict, now: float | None = None) -> bool:
    state = _holder_job_state(str(lease.get("owner") or ""))
    if state:
        return state == "open"
    # Not a delegated child (e.g. a top-level session): idle timeout only.
    current = time.time() if now is None else now
    return current - float(lease.get("at") or 0) < _LEASE_IDLE_S


class _LeaseLock:
    """Exclusive lock around the lease's read-check-write (cross-process)."""

    def __init__(self, session_name: str):
        self._path = SESSIONS / f"{session_name}.lease.lock"
        self._fd = None

    def __enter__(self):
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except ImportError:  # non-POSIX: best effort
            pass
        return self

    def __exit__(self, *exc):
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except ImportError:
            pass
        os.close(self._fd)
        return False


def _claim_lease(session_name: str) -> dict | None:
    """Take or refresh the profile lease; return a refusal dict if held."""
    me = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip()
    if not me:
        return None
    with _LeaseLock(session_name):
        return _claim_lease_locked(session_name, me)


def _claim_lease_locked(session_name: str, me: str) -> dict | None:
    lease = _read_lease(session_name)
    if lease and lease.get("owner") != me and _lease_live(lease):
        return {
            "session": session_name,
            "error": "profile_in_use",
            "holder": lease.get("owner"),
            "message": (
                f"The '{session_name}' browser profile is in use by another "
                f"agent (task {lease.get('owner')}). Only one agent may drive "
                "it at a time. Stop and report this to your coordinator; do "
                "not attach through the DevTools port or launch a second "
                "Chrome on this profile."
            ),
        }
    path = _lease_path(session_name)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps({"owner": me, "at": time.time()}, indent=2))
    os.replace(tmp, path)
    return None


def _release_lease(session_name: str) -> None:
    me = str(os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip()
    with _LeaseLock(session_name):
        lease = _read_lease(session_name)
        if lease and (not me or lease.get("owner") == me):
            try:
                _lease_path(session_name).unlink()
            except OSError:
                pass


def _find_chrome() -> str:
    """Find a usable Chrome/Chromium executable."""
    candidates = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "google-chrome"  # let nodriver's own search handle it


def _resolve_display(headless: bool, display: str | None) -> str | None:
    """Decide which X display Chrome should render on and EXPORT it.

    Chrome reads ``DISPLAY`` from the environment — nodriver ignores any
    ``display=`` kwarg. Precedence: explicit ``--display`` > inherited
    ``DISPLAY`` env > autodetected ``/tmp/.X11-unix`` socket. Returns the
    chosen display (or None for headless) and logs the choice to stderr.
    """
    if headless:
        return None
    chosen = display or os.environ.get("DISPLAY") or None
    if not chosen:
        # Autodetect: pick the lowest-numbered live X socket. On a
        # headless host the operator's xrdp session is often :10.
        socket_dir = Path("/tmp/.X11-unix")
        nums = []
        if socket_dir.is_dir():
            for s in socket_dir.iterdir():
                name = s.name
                if name.startswith("X") and name[1:].isdigit():
                    nums.append(int(name[1:]))
        if nums:
            chosen = f":{min(nums)}"
    if chosen:
        os.environ["DISPLAY"] = chosen
        print(f"[browser] using DISPLAY={chosen}", file=sys.stderr)
    else:
        print(
            "[browser] WARNING: headed mode but no X display found. "
            "Pass --display :N (e.g. --display :10) or --headless.",
            file=sys.stderr,
        )
    return chosen


async def _launch(headless: bool, display: str | None, session_name: str | None) -> nd.Browser:
    """Launch a FRESH Chromium."""
    _resolve_display(headless, display)
    kwargs = {
        "headless": headless,
        # Keep Chrome's sandbox ENABLED (non-root user) so the "unsupported
        # command-line flag --no-sandbox" warning banner never appears.
        "sandbox": True,
        "browser_executable_path": _find_chrome(),
    }
    if session_name:
        profile_dir = PROFILES / session_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs["user_data_dir"] = str(profile_dir)
    return await nd.start(**kwargs)


async def _attach(data: dict) -> nd.Browser:
    """Attach to an already-running Chrome via its CDP host:port."""
    host = data.get("host", "127.0.0.1")
    port = int(data.get("port") or 0)
    if not port:
        raise RuntimeError("live session has no CDP port — reopen with `open`")
    return await nd.start(host=host, port=port)


async def _browser_for(args: argparse.Namespace) -> tuple[nd.Browser, bool]:
    """Return (browser, attached). attached=True means DON'T stop it on exit."""
    session = getattr(args, "session", None)
    if session:
        data = _load_session(session)
        if data and data.get("port"):
            return await _attach(data), True
    return await _launch(args.headless, args.display, session), False


async def _navigate(browser: nd.Browser, args: argparse.Namespace):
    """Navigate to --url if given (and wait), else act on the current page.

    Returns the active page/tab. When attached with no --url, reuses the
    page already on screen — matching the skill's eval/inspect examples.
    """
    url = getattr(args, "url", None)
    if url:
        page = await browser.get(url)
        # Settle: wait for the document to finish parsing (bounded). SPAs
        # may still hydrate after this — use --wait / --wait-for-css for
        # client-rendered content.
        for _ in range(max(int((args.timeout or 30) / 0.25), 1)):
            try:
                if await page.evaluate("document.readyState") == "complete":
                    break
            except Exception:
                pass
            await asyncio.sleep(0.25)
    else:
        page = browser.main_tab
    wait_for_css = getattr(args, "wait_for_css", None)
    if wait_for_css:
        await page.wait_for(selector=wait_for_css, timeout=args.timeout * 1000)
    wait = getattr(args, "wait", 0)
    if wait:
        await asyncio.sleep(wait)
    return page


def _maybe_blocked(text: str | None):
    """Print a BLOCKED line + exit 2 if the page looks like an anti-bot wall."""
    if not text:
        return
    low = text.lower()
    for m in _BLOCK_MARKERS:
        if m in low:
            print(f"BLOCKED: site served an anti-bot wall ({m!r}). Stop; tell the operator.", file=sys.stderr)
            sys.exit(2)


async def cmd_status(args: argparse.Namespace) -> dict:
    session = args.session or "live"
    lease = _read_lease(session)
    holder = {"lease_holder": lease.get("owner")} if lease and _lease_live(lease) else {}
    data = _load_session(session)
    if data:
        return {**data, "running": True, **holder}
    return {"session": session, "running": False, **holder}


async def _serve_open(args: argparse.Namespace):
    """Holder coroutine: launch headed Chrome, record its port, stay alive."""
    session = args.session or "live"
    browser = await _launch(args.headless, args.display, session)
    if getattr(args, "url", None):
        await browser.get(args.url)
    port = getattr(browser.config, "port", 0) or 0
    host = getattr(browser.config, "host", "127.0.0.1") or "127.0.0.1"
    _save_session(session, os.getpid(), port, {"host": host, "running": True})
    stop = asyncio.Future()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, lambda: stop.set_result(None))
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await stop
    finally:
        try:
            browser.stop()
        except Exception:
            pass


def run_open(args: argparse.Namespace) -> int:
    """Double-fork `open` into the background; parent returns immediately.

    The detached grandchild runs the asyncio holder loop and writes the
    session file with the real CDP port. The parent polls for that file
    and prints the JSON handle, so the launching Bash call never blocks.
    """
    session = args.session or "live"
    # Clear any stale handle so the parent's poll only sees the new one.
    _remove_session(session)
    pid = os.fork()
    if pid > 0:
        deadline = time.time() + max(args.timeout or 30, 10)
        while time.time() < deadline:
            data = _load_session(session)
            if data and data.get("port"):
                print(
                    json.dumps(
                        {
                            "session": session,
                            "running": True,
                            "pid": data["pid"],
                            "port": data["port"],
                        }
                    )
                )
                return 0
            time.sleep(0.2)
        print(
            json.dumps(
                {
                    "session": session,
                    "running": False,
                    "error": f"open timed out; see {SESSIONS / (session + '.open.log')}",
                }
            )
        )
        return 1
    # Child → detach fully (setsid + second fork), redirect stdio to a log.
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    logpath = SESSIONS / f"{session}.open.log"
    devnull = os.open(os.devnull, os.O_RDONLY)
    logfd = os.open(str(logpath), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(devnull, 0)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    try:
        asyncio.run(_serve_open(args))
    except Exception:
        traceback.print_exc()
    os._exit(0)


async def cmd_close(args: argparse.Namespace):
    session = args.session or "live"
    data = _load_session(session)
    if data and data.get("pid"):
        try:
            os.kill(data["pid"], signal.SIGTERM)
            time.sleep(1)
        except (OSError, ProcessLookupError):
            pass
    _remove_session(session)
    _release_lease(session)
    print(json.dumps({"session": session, "running": False}))


async def cmd_goto(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        # nodriver Tab has no .text(); get visible text via JS innerText.
        text = await page.evaluate("document.body ? document.body.innerText : document.documentElement.innerText")
        if not isinstance(text, str):
            text = await page.get_content()  # fallback: full HTML
        _maybe_blocked(text)
        print(text[:5000] if text else "(empty page)")
    finally:
        if not attached:
            browser.stop()


async def cmd_extract(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        elems = await page.find_all(args.css)
        results = []
        for el in elems:
            entry = {}
            attrs = args.attr.split(",") if args.attr else ["text"]
            for attr in attrs:
                attr = attr.strip()
                if attr == "text":
                    entry["text"] = el.text  # property, not a coroutine
                elif attr == "html":
                    entry["html"] = await el.get_html()
                else:
                    entry[attr] = (el.attrs or {}).get(attr)
            results.append(entry)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, default=str))
        else:
            for r in results:
                print(r.get("text", "") or json.dumps(r, ensure_ascii=False))
    finally:
        if not attached:
            browser.stop()


async def cmd_click(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        for i in range(args.repeat or 1):
            if i > 0:
                await asyncio.sleep(1)
            if args.text:
                btn = await page.find(args.text)
            elif args.css:
                btn = await page.find(args.css)
            elif args.xpath:
                btn = await page.find(args.xpath)
            else:
                print("Provide --text, --css, or --xpath")
                return
            if btn:
                await btn.click()
                print(f"Clicked ({i + 1}/{args.repeat or 1})")
            else:
                print(f"Element not found (attempt {i + 1})")
                break
    finally:
        if not attached:
            browser.stop()


async def cmd_fill(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        for field_str in args.field:
            selector, _, value = field_str.partition(":")
            el = await page.find(selector)
            if el:
                await el.send_keys(value)
                print(f"Filled {selector}")
            else:
                print(f"Field {selector} not found")
        if args.submit:
            btn = await page.find("button[type=submit]") or await page.find("input[type=submit]")
            if btn:
                await btn.click()
                print("Submitted")
    finally:
        if not attached:
            browser.stop()


async def cmd_screenshot(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        await page.save_screenshot(args.output)
        print(f"Screenshot saved to {args.output}")
    finally:
        if not attached:
            browser.stop()


async def cmd_inspect(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        root = await page.select(args.css)
        if root:

            def _build_tree(el, depth=0):
                if depth > (args.depth or 3):
                    return None
                tag = el.tag_name or "?"
                info = {"tag": tag}
                attrs = el.attrs or {}
                if attrs:
                    info["attrs"] = {k: v for k, v in attrs.items() if k != "_"}
                children = el.children or []
                if children:
                    kids = []
                    for c in children:
                        t = _build_tree(c, depth + 1)
                        if t:
                            kids.append(t)
                    if kids:
                        info["children"] = kids
                return info

            tree = [_build_tree(el) for el in root] if isinstance(root, list) else [_build_tree(root)]
            print(json.dumps(tree, ensure_ascii=False, default=str, indent=2))
    finally:
        if not attached:
            browser.stop()


async def cmd_eval(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        # nodriver returns a raw CDP RemoteObject for object/array results
        # even with return_by_value. Stringify in-page and parse back so
        # `eval` always yields clean JSON. Falls back to raw eval if the
        # expression can't be wrapped (e.g. a bare statement).
        wrapped = f"JSON.stringify((function(){{ return ({args.js}); }})())"
        try:
            raw = await page.evaluate(wrapped)
            result = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            result = await page.evaluate(args.js)
        print(json.dumps(result, ensure_ascii=False, default=str))
    finally:
        if not attached:
            browser.stop()


async def cmd_scroll(args: argparse.Namespace):
    browser, attached = await _browser_for(args)
    try:
        page = await _navigate(browser, args)
        steps = args.steps or 4
        for i in range(steps):
            frac = (i + 1) / steps
            await page.evaluate(f"window.scrollTo({{ top: document.body.scrollHeight * {frac}, behavior: 'smooth' }})")
            await asyncio.sleep(0.5)
        print(f"Scrolled to bottom in {steps} steps")
    finally:
        if not attached:
            browser.stop()


async def cmd_login(args: argparse.Namespace):
    """Open headed, wait for operator to log in, save cookie snapshot."""
    session = args.session or "live"
    browser = await _launch(args.headless, args.display, session)
    if args.url:
        await browser.get(args.url)
    print("Browser open. Log in manually, then press Enter here...", flush=True)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sys.stdin.readline)
    cookies = await browser.cookies.all()
    snap = {"cookies": [dict(c) for c in cookies], "url": args.url}
    (SESSIONS / f"{session}.json").write_text(json.dumps(snap, indent=2))
    print(f"Cookie snapshot saved for session '{session}'")
    browser.stop()


async def cmd_takeover(args: argparse.Namespace):
    """Resume from a cookie snapshot (anonymous-looking visit)."""
    session = args.session or "live"
    snap_path = SESSIONS / f"{session}.json"
    if not snap_path.exists():
        print(f"No cookie snapshot for session '{session}'. Run 'login' first.")
        return
    snap = json.loads(snap_path.read_text())
    browser = await _launch(args.headless, args.display, None)
    page = await browser.get(snap.get("url", "about:blank"))
    for c in snap.get("cookies", []):
        try:
            await browser.cookies.set(**c)
        except Exception:
            pass
    if args.url:
        page = await browser.get(args.url)
    text = await page.evaluate("document.body ? document.body.innerText : document.documentElement.innerText")
    if not isinstance(text, str):
        text = await page.get_content()
    _maybe_blocked(text)
    print(text[:5000] if text else "(empty page)")
    browser.stop()


async def cmd_persist(args: argparse.Namespace):
    session = args.session or "live"
    snap_path = SESSIONS / f"{session}.json"
    if snap_path.exists():
        print(str(snap_path))
    else:
        print(f"No snapshot for session '{session}'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Nodriver Browser CLI")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--display", type=str, help="X display (e.g. :10)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")

    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("status")
    p.add_argument("--session", type=str, default="live")

    p = sub.add_parser("open")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)

    p = sub.add_parser("close")
    p.add_argument("--session", type=str, default="live")

    # Navigation actions — --url is OPTIONAL: when attached to a live
    # session it acts on the current page if omitted.
    p = sub.add_parser("goto")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("extract")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--css", type=str, required=True)
    p.add_argument("--attr", type=str, default="text")
    p.add_argument("--json", action="store_true")
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("click")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--text", type=str)
    p.add_argument("--css", type=str)
    p.add_argument("--xpath", type=str)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("fill")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--field", type=str, action="append", required=True)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("screenshot")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("inspect")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--css", type=str, default="body")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("eval")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--js", type=str, required=True)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("scroll")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--to", type=str, default="bottom")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("login")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("takeover")
    p.add_argument("--session", type=str, default="live")
    p.add_argument("--url", type=str)
    p.add_argument("--wait-for-css", type=str)
    p.add_argument("--wait", type=float, default=0)

    p = sub.add_parser("persist")
    p.add_argument("--session", type=str, default="live")

    args = parser.parse_args()

    if args.action != "status":
        refused = _claim_lease(getattr(args, "session", None) or "live")
        if refused:
            print(json.dumps(refused, ensure_ascii=False))
            sys.exit(3)

    # `open` daemonizes itself (no asyncio at this level) so the launching
    # Bash call returns immediately instead of blocking on the holder loop.
    if args.action == "open":
        sys.exit(run_open(args))

    handlers = {
        "status": cmd_status,
        "close": cmd_close,
        "goto": cmd_goto,
        "extract": cmd_extract,
        "click": cmd_click,
        "fill": cmd_fill,
        "screenshot": cmd_screenshot,
        "inspect": cmd_inspect,
        "eval": cmd_eval,
        "scroll": cmd_scroll,
        "login": cmd_login,
        "takeover": cmd_takeover,
        "persist": cmd_persist,
    }

    handler = handlers.get(args.action)
    if not handler:
        print(f"Unknown action: {args.action}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(handler(args))
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))
    # The browser is already stopped (handlers call browser.stop() for
    # non-attached launches). nodriver's atexit then tries a second
    # cleanup on the closed loop and prints a harmless "Event loop is
    # closed" traceback. We've produced our output — hard-exit to skip
    # that noise so callers see a clean result.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
