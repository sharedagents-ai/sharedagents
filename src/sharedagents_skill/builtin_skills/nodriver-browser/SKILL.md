---
name: nodriver-browser
description: "Drive a real headed Chromium browser via the Bash CLI at ~/.sharedagents/skills/nodriver-browser/browser. Use for: (1) pages mcp__web__fetch can't see (Cloudflare, DataDome, JS-heavy SPAs), (2) interactive login flows where a human types credentials in a visible window, (3) sessions that need to persist cookies across calls, (4) clicking/filling/screenshotting/extracting on real DOMs. NOT for: plain HTTP JSON / static HTML (use mcp__web__fetch), simple web search (use mcp__web__search)."
---

# Nodriver Browser — How to drive Chromium from Bash

This skill is the **only** way to drive a real browser. There are no
`browser_*` tools. You invoke a single CLI launcher from `Bash` and
read its stdout.

## ⚠️ Read this first

1. **One launcher, absolute path:**
   `~/.sharedagents/skills/nodriver-browser/browser <action> [flags]`
2. **Never** chain it with `cd` (`cd … && python browser.py …`) —
   that pattern silently drops env vars and breaks `DISPLAY`. The
   launcher already uses the right venv and resolves the display.
3. **Headed is the default.** Chromium appears on the operator's
   xrdp / Xorg screen. Pass `--headless` only when the operator
   explicitly asks for an unattended run.
4. **Display resolves automatically: flag > `DISPLAY` env >
   autodetected X socket.** The launcher does NOT force `:0` (an old
   bug). If `DISPLAY` is empty (the normal case on headless hosts)
   it autodetects the lowest live `/tmp/.X11-unix/X*` socket, which is
   the operator's xrdp `:10`. **On headless hosts, pass `--display :10`
   explicitly** to be safe — the operator is always on `:10`.
5. **`open` returns immediately.** It double-forks a detached holder
   that keeps Chrome alive and prints `{running, pid, port}` in <1 s.
   You do NOT need `run_in_background`. Holder stdout/stderr go to
   `sessions/<name>.open.log` for debugging.
6. **Global flags go BEFORE the action name:** ``--headless``,
   ``--display``, ``--timeout`` are top-level flags — they must
   come before ``goto``/``extract``/etc.:
   ``browser --headless goto --url ...`` ✅
   ``browser goto --headless --url ...`` ❌
7. **`--url` is optional once a session is live.** Actions attach to
   the running Chrome and, with no `--url`, act on the **current**
   page. Pass `--url` only to navigate first.
8. **Check status first:** ``browser status [--session live]``
   prints JSON ``{running: true/false, pid, port, ...}``. Use
   it before opening or attaching to avoid hitting stale PIDs.
9. **One agent per profile.** Every command from an sharedagents session
   takes a lease on the profile. If another agent holds it, the
   command prints `{"error": "profile_in_use", "holder": ...}` and
   exits 3. Stop and report that to your coordinator. Do not work
   around it by attaching through the DevTools port, launching a
   second Chrome, or copying the profile. `status` shows the holder
   as `lease_holder`; `close` releases your own lease.
10. **Never reset a page someone is using.** Before navigating, read
    the current tab. If it shows a verification code box, a
    confirmation step, or anything the owner is completing, leave it
    alone: no reload, no new login, no "resend code". Report what the
    page is waiting for and stop.

## Quick reference — every action

Every action takes flags after the action name. Long-form `--flags`
only (no short flags). All commands return text or JSON on stdout
and exit `0` on success, non-zero on failure.

| Action | What it does |
|---|---|
| `goto`       | Navigate to a URL and print the page text (first ~5 KB). Accepts `--session` for live attach. |
| `extract`    | Pull elements matching a CSS selector. With `--json`, prints a JSON array. |
| `click`      | Click an element by `--text` / `--css` / `--xpath`. `--repeat N` for pagination. |
| `fill`       | Fill form fields (`--field selector:value` repeated). `--submit` to submit. |
| `screenshot` | Save page to `--output <file>`. |
| `inspect`    | DOM outline rooted at `--css`, depth-limited JSON tree. Use BEFORE writing a flaky extract. |
| `eval`       | Run a JS expression in the live tab; result is JSON-serialised. |
| `scroll`     | Trigger lazy-load (`--to bottom --steps 4` is the default). |
| `open`       | Launch a long-running headed Chrome with a persistent profile. Other calls attach via `--session`. |
| `close`      | Terminate a session opened with `open`. The profile persists on disk. |
| `status`     | Check if a live session is running. Prints JSON: `{running, pid, port, ...}`. |
| `login`      | Open headed, wait for the operator to log in, save a cookie snapshot. Mostly legacy — prefer `open` + manual login + reuse. |
| `takeover`   | Resume a cookie-snapshot session for an anonymous-looking visit. |
| `persist`    | Verify a saved cookie snapshot exists; prints the path. |

## The 90% pattern: persistent live browser

For almost every browse task: open once, attach many times.

```bash
# 1. Open the headed window. The operator sees it. Returns immediately.
~/.sharedagents/skills/nodriver-browser/browser open --session live --url https://example.com

# 2. Every follow-up call attaches via --session live. No new Chrome spawn.
~/.sharedagents/skills/nodriver-browser/browser goto --session live --url https://example.com/page2
~/.sharedagents/skills/nodriver-browser/browser extract --session live --url https://example.com/page2 --css "table tr" --json
~/.sharedagents/skills/nodriver-browser/browser eval --session live --js "document.title"

# 3. Do NOT close it when you're done. Leave the window open for the
#    operator. The on-disk profile at
#    ~/.sharedagents/skills/nodriver-browser/profiles/live/  persists across
#    restarts — Google login from yesterday still works tomorrow.
```

`live` is the default session name. Use a different name only when
the operator wants a separate profile (e.g. `google`, `investing`,
`shopping`).

## Per-action examples

### Navigate + extract page text (anonymous, one-shot)

```bash
~/.sharedagents/skills/nodriver-browser/browser goto --url https://example.com
```

### Wait for an SPA to hydrate

```bash
~/.sharedagents/skills/nodriver-browser/browser goto --url https://example.com/spa --wait-for-css "#main-content" --wait 2
```

### Extract a table as JSON

```bash
~/.sharedagents/skills/nodriver-browser/browser extract --session live --url https://investing.com/economic-calendar/ --css "table.datatable-v2 tr" --attr "text,data-event-id" --json
```

### Click "Load more" 20 times

```bash
~/.sharedagents/skills/nodriver-browser/browser click --session live --url https://example.com/feed --text "Load more" --repeat 20
```

### Fill a search form and submit

```bash
~/.sharedagents/skills/nodriver-browser/browser fill --session live --url https://example.com --field "input[name=q]:search terms" --submit
```

### Run arbitrary JS, get JSON back

```bash
~/.sharedagents/skills/nodriver-browser/browser eval --session live --js 'Array.from(document.querySelectorAll("a")).slice(0,5).map(a => a.href)'
```

### Outline the DOM before writing a fragile selector

```bash
~/.sharedagents/skills/nodriver-browser/browser inspect --session live --css 'body' --depth 2
```

### Trigger lazy-loaded items

```bash
~/.sharedagents/skills/nodriver-browser/browser scroll --session live --to bottom --steps 4
```

### Screenshot for debugging

```bash
~/.sharedagents/skills/nodriver-browser/browser screenshot --session live --url https://example.com --output /tmp/page.png
```

### Check if a session is alive

```bash
~/.sharedagents/skills/nodriver-browser/browser status --session live
# → {"session": "live", "running": true, "pid": 12345, "port": 12345, ...}
```

### Open on a specific display (headless hosts)

```bash
# When the agent doesn't auto-detect DISPLAY on a headless host:
~/.sharedagents/skills/nodriver-browser/browser --display :10 open --session live --url https://example.com
```

## Sessions, profiles, cookies — where state lives

```
~/.sharedagents/skills/nodriver-browser/
├── browser              # The launcher you invoke.
├── scripts/
│   ├── browser.py       # CLI source. Don't invoke directly.
│   └── venv/            # Python venv carrying `nodriver`.
├── profiles/<name>/     # Chrome on-disk profile for `open --session <name>`.
└── sessions/
    ├── <name>.json      # Cookie + localStorage snapshot from `login`.
    └── <name>.live.json # Live-Chrome handle (pid + CDP port) from `open`.
```

- `open --session <name>` writes/updates `<name>.live.json` and the
  matching `profiles/<name>/` dir.
- `login --session <name>` writes `<name>.json` (cookie snapshot)
  after the operator finishes typing credentials.
- Subsequent `goto/extract/click/fill/screenshot/eval/inspect/scroll`
  with `--session <name>` will:
  - **Attach** to the live Chrome if `<name>.live.json` exists and the
    pid is alive.
  - **Otherwise fall back** to a fresh Chromium that hydrates cookies
    from `<name>.json` if that snapshot exists.

## Recovery and pacing

- If a command exits non-zero with **"BLOCKED"** in stderr, the site
  detected automation (Cloudflare, captcha, 403, 429). **Stop.** Don't
  retry. Tell the operator what happened.
- Real Chromium against real sites can earn IP bans. Don't override
  the built-in random delays (0.5 – 2 s between repeated clicks).
- For lazy-loaded grids: `scroll --to bottom --steps 4` first, then
  `extract` or `eval`. If `scrollHeight` stopped growing, there's no
  more content.
- If three consecutive `(action, flags)` calls return zero rows or
  unchanged text, **screenshot the page** before changing your
  selector again. The DOM is probably different from what you think.

## Common mistakes to avoid

| Wrong | Right |
|---|---|
| `cd ~/.sharedagents/skills/nodriver-browser/scripts && venv/bin/python browser.py goto --url …` | `~/.sharedagents/skills/nodriver-browser/browser goto --url …` |
| `DISPLAY=:10 ~/.sharedagents/skills/nodriver-browser/browser …` | `browser --display :10 …` (global flag, not env prefix) |
| Passing `--headless` because Chrome won't show | First check `echo $DISPLAY`. If unset on a headless host, pass `--display :10`. |
| `goto … && extract …` in one shell call | Run two separate `Bash` tool calls so each result is visible in the transcript. |
| Closing a session you opened | Leave `open` sessions running unless the operator asks. |
| `browser goto --headless --url ...` | `browser --headless goto --url ...` (global flags before action) |
| `browser eval --code "..."` | `browser eval --js "..."` (the flag is `--js`, not `--code`) |
| `browser goto --session live --url ...` before checking if session is alive | `browser status --session live` first; if stale, `open` a new one |
| Running `open` with `run_in_background` / assuming it hangs | `open` daemonizes and returns in <1 s. Just call it normally and read the JSON. |
| Re-running `open --session live` while that profile is already open in another Chrome | A profile can't be used by two Chromes at once (Chrome `SingletonLock`). `close` it first, or use a different `--session` name. |
| Passing `--headless` because the window didn't appear | First check the display. On headless hosts, pass `--display :10`. Headless changes behavior — only use it for truly unattended runs. |

## Why headed and why a persistent profile

- The operator **sees** what the agent is doing and can intervene
  (log in, solve a captcha, scroll, click). Anything they do in the
  window persists for the next tool call.
- Profile dir keeps cookies + localStorage + history across restarts
  → no nightly re-login dance.
- Headless Chrome leaks fingerprints (`navigator.webdriver`,
  AudioContext anomalies) that anti-bot systems use. Headed bypasses
  most of that.
- Spawn cost amortizes: one Chrome cold-start per task (~2 s)
  instead of one per tool call.

## When to fall back to `mcp__web__fetch` / `mcp__web__search`

This skill spawns Chrome. That's heavy. Use it only when you need
JS rendering, session state, or anti-bot bypass. For everything
else:

| Target | Tool |
|---|---|
| Public JSON API | `mcp__web__fetch` |
| Plain HTML / RSS | `mcp__web__fetch` |
| "Find me articles about X" | `mcp__web__search` |
| Behind login | this skill |
| Cloudflare-protected | this skill |
| SPA where `view-source` is empty | this skill |
| Need to click / fill / paginate | this skill |

## Bootstrap (only needed on a fresh machine)

The launcher and venv are normally pre-installed. If
`~/.sharedagents/skills/nodriver-browser/browser` is missing, run the
bundled bootstrap — it copies `browser` + `scripts/browser.py` out of
the package and builds the venv (idempotent, safe to re-run to pull
updated scripts):

```bash
bash "$(python3 -c 'import sharedagents_core, pathlib; print(pathlib.Path(sharedagents_core.__file__).parent / "builtin_skills/nodriver-browser/bootstrap.sh")')"
```

Or, from the source checkout:

```bash
bash <sharedagents checkout>/src/sharedagents_skill/builtin_skills/nodriver-browser/bootstrap.sh
```

## Appendix — Web UI interaction patterns

Patterns discovered during real-world use with complex web applications
(iframes, chunked content, form dispatch). Adapt the selectors and
URLs to your own target application.

### Reading content from an iframe

When a page loads its main content inside an `<iframe>`, `goto` and
`extract` only see the outer page. Use `eval` to reach into the
iframe's document:

```bash
# Read the response from the iframe's <main> element
~/.sharedagents/skills/nodriver-browser/browser eval --session live --js '(function(){
  var iframe = document.getElementById("your-iframe-id");
  if (!iframe) return "no iframe";
  var b = iframe.contentWindow.document;
  if (!b) return "cross-origin blocked";
  var main = b.querySelector("main");
  if (!main) return "no main content";
  return JSON.stringify(main.innerText);
})()'
```

**Caveat:** cross-origin iframes block `contentWindow.document` access.
If you see "cross-origin blocked", the iframe loads a different domain
and you cannot read it from the parent page.

### Reading long content in chunks

An iframe's `innerText` can be very long. Read in chunks to avoid
truncation:

```bash
# First 3000 characters
~/.sharedagents/skills/nodriver-browser/browser eval --session live --js '(function(){
  var b = document.getElementById("your-iframe-id").contentWindow.document;
  return JSON.stringify(b.querySelector("main").innerText.substring(0, 3000));
})()'

# Next chunk (3000–8000)
~/.sharedagents/skills/nodriver-browser/browser eval --session live --js '(function(){
  var b = document.getElementById("your-iframe-id").contentWindow.document;
  return JSON.stringify(b.querySelector("main").innerText.substring(3000, 8000));
})()'
```

### Submitting a form via JavaScript dispatch

Some forms use AJAX handlers triggered by a `submit` event rather than
a traditional POST. The `--submit` flag does a traditional form post
which navigates away. Instead, dispatch the event via `eval`:

```bash
~/.sharedagents/skills/nodriver-browser/browser eval --session live --js '(function(){
  var input = document.getElementById("your-input-id");
  var form  = document.getElementById("your-form-id");
  input.value = "YOUR INPUT HERE";
  var e = new Event("submit", { bubbles: true, cancelable: true });
  form.dispatchEvent(e);
  return "submitted";
})()'
```

Always wrap multi-statement JS in an IIFE — the `eval` runner wraps
the expression, so bare semicolons cause `SyntaxError`.

### Authenticating once and reusing cookies

Many sites require login. Open the browser headed, log in manually,
and the persistent profile saves cookies automatically:

```bash
# 1. Open Chrome on the operator's display pointing to the login page
~/.sharedagents/skills/nodriver-browser/browser open --session live --display :10 --url https://your_web_ui_url

# 2. Operator logs in through the visible browser window.
#    The persistent profile saves the session cookie automatically.
```

After login, the `live` profile at
`~/.sharedagents/skills/nodriver-browser/profiles/live/` carries the auth
cookie — no re-login needed until the profile is cleared or the cookie
expires.

### Debugging a local development server

For local debugging, start your development server and point the
browser at it:

```bash
# Start your local server (example: port 8124)
# ... your server command ...

~/.sharedagents/skills/nodriver-browser/browser open --session live --display :10 --url http://127.0.0.1:8124/
```

### Common web UI pitfalls

| Wrong | Right |
|---|---|
| `browser eval --js 'form.submit(); "done"'` | Wrap in IIFE: `--js '(function(){form.submit();return "done"})()'` — bare semicolons cause SyntaxError |
| `browser fill --field "#input:val" --submit` | If the form uses AJAX, use `eval` to dispatch a `submit` event instead |
| `browser goto` after iframe loads | Text extraction only gets the outer page, not iframe content. Use `eval` to read `iframe.contentWindow.document` |
| Trying to click sidebar items by text | Sidebar items may be `<div>` elements, not `<a>` tags. Use CSS selectors or `inspect` first |
