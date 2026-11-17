"""Message input widget + slash-command helpers (extracted from app.py)."""

from __future__ import annotations

import json
from pathlib import Path

from textual.binding import Binding
from textual.widgets import Input, Static, TextArea

from sharedagents_tui.helpers import (
    _PLAIN_KEYS,
)


class CommandHistory:
    """Persisted command history for the chat input.

    Stores a fixed-size list of previously submitted commands in
    ``~/.sharedagents/command_history.json`` (one JSON array of strings).
    Thread-safe for single-process use (the TUI is single-threaded
    under Textual's event loop anyway).
    """

    MAX_ENTRIES = 256

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".sharedagents" / "command_history.json"
        self.path = path
        self.entries: list[str] = []
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = self.path.read_text(encoding="utf-8").strip()
                if raw:
                    self.entries = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            self.entries = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.entries, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def add(self, text: str) -> None:
        """Append a command. Deduplicates against the most recent entry."""
        text = text.strip()
        if not text:
            return
        if self.entries and self.entries[-1] == text:
            return  # skip exact duplicate at top
        self.entries.append(text)
        if len(self.entries) > self.MAX_ENTRIES:
            self.entries = self.entries[-self.MAX_ENTRIES :]
        self._save()

    def navigate(self, index: int) -> str | None:
        """Return the command at *index* (0 = most recent), or None."""
        if not self.entries:
            return None
        # index 0 → most recent (end of list), index N → oldest
        pos = len(self.entries) - 1 - index
        if 0 <= pos < len(self.entries):
            return self.entries[pos]
        return None


# Slash commands the autocomplete popup offers. Each entry is
# ``(token, signature, description)``; the token is the prefix the
# user can match against (no args), the signature is what the popup
# renders (``/model <alias|url>``), and the description is the
# one-line hint. Order is operator-ergonomic: most-common first,
# delegate sugars at the bottom.
SLASH_COMMAND_HELP: list[tuple[str, str, str]] = [
    ("/help", "/help", "show the quick reference card"),
    ("/new", "/new", "abort + clear transcript + start a new session"),
    ("/stop", "/stop", "graceful stop — finish current step, then end (same as 1st ESC)"),
    ("/abort", "/abort", "hard kill now — worker + tools + subagents (same as 2nd ESC)"),
    ("/steer", "/steer <text>", "redirect the running turn (inject, or graceful-pivot a CLI bridge)"),
    ("/restart", "/restart [--hard]", "sweep gateway children (--hard also bounces gateway)"),
    ("/agents", "/agents", "list configured sharedagents agents (with current marked)"),
    ("/agent", "/agent <name> [mode]", "switch this conversation's agent (same transcript)"),
    ("/sessions", "/sessions [i|hash]", "list saved sessions / switch by index or prefix"),
    ("/models", "/models", "list configured models (interactive picker)"),
    ("/model", "/model <alias|url>", "switch the live session's model/upstream"),
    ("/mode", "/mode [id]", "show or set this agent's advertised mode"),
    ("/effort", "/effort <level|default>", "set model-specific reasoning depth"),
    ("/status", "/status", "show agent + endpoint + backend snapshot"),
    ("/task", "/task [add|in|done|del|list]", "operator harness-task store (not the board)"),
    ("/budget", "/budget [usd|add|topup]", "show or set this session's budget cap"),
    ("/loop", "/loop [list|stop|…]", "durable scheduled loops"),
    ("/title", "/title <text>", "set this session's title"),
    ("/memory", "/memory [status|search|…]", "inspect or search agent memory"),
    ("/queue", "/queue", "list pending mid-turn / post-turn prompts"),
    ("/history", "/history", "dump the current session's transcript"),
    ("/refresh", "/refresh", "re-fold durable log (catch up Web/Telegram turns)"),
    ("/compact", "/compact", "squeeze oversized tool results in the saved session"),
    ("/tools", "/tools [on|off]", "toggle inline 🔧 tool-call header (default off)"),
    ("/thinking", "/thinking [on|off]", "show/hide thinking tokens"),
    ("/reasoning", "/reasoning [on|off|stream]", "stream / hide / show thinking tokens"),
    ("/on", "/on", "shortcut for /reasoning on"),
    ("/off", "/off", "shortcut for /reasoning off"),
    ("/stream", "/stream", "shortcut for /reasoning stream"),
    ("/teacher", "/teacher [on|off]", "toggle teacher-model supervision for this session (goal or not)"),
    ("/teacher", "/teacher model <alias>", "pick which model supervises (model default = configured teacher)"),
    ("/mouse", "/mouse", "toggle mouse tracking (text selection vs clickable UI)"),
    ("/todo", "/todo", "show this session's todo checklist"),
    (
        "/subagents",
        "/subagents [N]",
        "list subagents this session has spawned (delegate_agent / claude / codex). Pass N (1-based) to enter that subagent's session.",
    ),
    (
        "/monitor",
        "/monitor",
        "agent monitor — navigate ALL agents' sessions + subagents (webui /agents port); Enter opens any session",
    ),
    (
        "/monitors",
        "/monitors",
        "live shells panel — navigate / kill background shells (alias: /shells; binding: Ctrl+S) — NOT the agent monitor (/monitor)",
    ),
    ("/shells", "/shells", "alias for /monitors — same modal, more findable name"),
    ("/claude", "/claude <prompt>", "bridge the prompt to Claude Code"),
    ("/codex", "/codex <prompt>", "bridge the prompt to OpenAI's Codex CLI"),
    ("/antigravity", "/antigravity <prompt>", "bridge the prompt to Google Antigravity CLI"),
    ("/zcode", "/zcode <prompt>", "bridge the prompt to Z.AI's ZCode CLI (GLM Coding Plan)"),
    ("/plugin", "/plugin <subcmd>", "manage plugins: list / search / install / enable / disable / remove"),
    ("/mcp", "/mcp [filter]", "list MCP servers (Claude Code-compatible) — name, transport, source"),
    ("/skills", "/skills [name]", "list skills (or open one by name) — same store the agent loads from"),
    ("/commands", "/commands", "list file-based slash commands (`commands/*.md`)"),
    ("/context", "/context [N]", "show what's filling the context window (token map per round)"),
]


def file_command_rows() -> list[tuple[str, str, str]]:
    """File-based slash commands (``commands/*.md``) as popup rows.

    Bridges ``sharedagents_command.file_commands.discover_commands()`` into
    the same ``(token, signature, description)`` shape the popup
    renders for built-ins. Includes the operator's
    ``~/.sharedagents/commands/``, ``~/.claude/commands/``, installed
    plugin commands, AND the package's bundled
    ``builtin_commands/`` (where ``/goal`` ships) — so every
    file-defined command is discoverable from the popup, not just
    the ones the TUI has hardcoded knowledge of.

    Fail-soft: any import / discovery error returns an empty list
    so a broken commands dir can't take down the chat input.
    """
    try:
        from sharedagents_command.file_commands import discover_commands

        cmds = discover_commands()
    except Exception:  # noqa: BLE001
        return []
    rows: list[tuple[str, str, str]] = []
    for c in cmds:
        token = f"/{c.name}"
        hint = c.argument_hint.strip()
        sig = f"{token} {hint}" if hint else token
        rows.append((token, sig, c.short_description))
    return rows


def merge_slash_command_rows(
    builtin: list[tuple[str, str, str]],
    file_based: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Combine the hardcoded popup table with file-command rows,
    dropping any file row whose token collides with a hardcoded
    built-in.

    Built-ins win on collision because dispatch in
    ``on_input_submitted`` runs built-in handlers FIRST and only
    unrecognised slashes fall through to file commands — so an
    operator's ``commands/help.md`` would never actually fire,
    listing it in the popup would just be misleading.
    """
    builtin_tokens = {row[0].lower() for row in builtin}
    extras = [r for r in file_based if r[0].lower() not in builtin_tokens]
    return list(builtin) + extras


def _longest_common_prefix(strings: list[str]) -> str:
    """Return the longest common prefix of all strings, or '' if the
    list is empty. Used by Tab-completion when multiple commands
    match the current partial — we fill in as many shared chars as
    we can, then leave the rest to the user."""
    if not strings:
        return ""
    base = strings[0]
    for s in strings[1:]:
        n = min(len(base), len(s))
        i = 0
        while i < n and base[i] == s[i]:
            i += 1
        base = base[:i]
        if not base:
            break
    return base


def resolve_slash_on_submit(
    text: str,
    rows: list[tuple[str, str, str]],
) -> tuple[str | None, str | None]:
    """Resolve an incomplete slash token when the operator hits Enter.

    Returns ``(submit_text, rewrite_input)``:

    * ``submit_text`` — text to submit, or ``None`` to keep editing
      (never send a half-typed ``/ag`` to the model as a chat turn).
    * ``rewrite_input`` — when not submitting, the value to put back
      into the input box (LCP / unique completion + trailing space);
      ``None`` means leave the input alone.

    Already-argumented commands (``/agent plan``) and non-slash text
    pass through unchanged. Exact token matches (``/agent``,
    ``/agents``) submit as-is even when another command shares the
    prefix (``/agent`` vs ``/agents``).
    """
    stripped = (text or "").strip()
    if not stripped.startswith("/") or " " in stripped or "\n" in stripped:
        return stripped or None, None
    if not rows:
        return stripped, None
    # Callers usually pass already-filtered matches; also filter here so
    # a full help table cannot make `/hel` look ambiguous against `/mode`.
    prefix = stripped.lower()
    rows = [row for row in rows if row[0].lower().startswith(prefix)]
    if not rows:
        # Unknown token — submit as-is so dispatch can reject or expand
        # a file command. Never swallow `/notacommand` as "keep typing".
        return stripped, None
    exact = next(
        (row for row in rows if row[0].lower() == stripped.lower()),
        None,
    )
    if exact is not None:
        return exact[0], None
    if len(rows) == 1:
        token, sig, _ = rows[0]
        if "<" in sig:
            # Required-arg command: complete the token + space, wait
            # for the operator to finish (e.g. ``/cla`` → ``/claude ``).
            return None, token + " "
        return token, None
    lcp = _longest_common_prefix([row[0] for row in rows])
    exact_lcp = next(
        (row for row in rows if row[0].lower() == lcp.lower()),
        None,
    )
    if exact_lcp is not None:
        # ``/ag`` → LCP ``/agent``, which is itself a valid command.
        # Prefer submitting the exact command over dropping ``/ag``
        # into the chat as a free-form prompt.
        return exact_lcp[0], None
    if len(lcp) > len(stripped):
        return None, lcp
    # Still ambiguous at the current prefix — keep editing.
    return None, None


class MessageInput(TextArea):
    """Multi-line composition box for the chat input.

    Wraps long messages instead of scrolling horizontally, so a paste
    or a long thought stays readable. Two key bindings make it feel
    like an `Input` for the common case:

    - ``Enter``         — submit the message (single-line behaviour
                          the rest of the app expects).
    - ``Shift+Enter``   — insert a newline; lets the user compose
                          multi-line prompts naturally.
    - ``Up`` / ``Down`` — cycle through command history (like bash).

    Submission is signalled by posting an ``Input.Submitted`` message
    (same event class ``Input`` would post), so ``on_input_submitted``
    handlers stay unchanged. ``value`` proxies to ``self.text``, also
    matching ``Input``'s API — including the ``event.input.value = ""``
    clear pattern in the existing handler.
    """

    BINDINGS = [
        # Three submit/newline bindings to handle every terminal:
        #   - `enter` submits (with kitty disambiguation, this is ONLY
        #     plain Enter — shift+enter / ctrl+enter / ctrl+j produce
        #     distinct key events that don't match this binding).
        #   - `shift+enter` inserts a newline. Requires the terminal to
        #     speak the kitty keyboard protocol (kitty, foot, ghostty,
        #     recent Konsole/WezTerm). `_enable_kitty_keyboard_enhancements`
        #     pushes flags 1+2+4 so the modifier survives.
        #   - `ctrl+j` is the cross-terminal fallback. Ctrl+J is literal
        #     LF (ASCII 10) — every terminal sends it as a distinct
        #     keystroke from Enter (which is CR / ASCII 13), so this
        #     works in plain xterm / gnome-terminal / Alacritty without
        #     needing any protocol support.
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "New line", show=False),
        # tmux/SSH deliver Enter as LF (ctrl+j). Binding that to newline
        # made every Send insert a blank line and dump the next slashes
        # into the model as one chat turn. On plain-key transports, LF
        # submits; Ctrl+N is the newline fallback.
        Binding(
            "ctrl+j",
            "submit" if _PLAIN_KEYS else "insert_newline",
            "Send" if _PLAIN_KEYS else "New line",
            show=False,
            priority=True,
        ),
        Binding("ctrl+n", "insert_newline", "New line", show=False),
        Binding("up", "history_prev", "History ↑", show=False),
        Binding("down", "history_next", "History ↓", show=False),
        # Tab completes the slash command. ``priority=True`` so it
        # wins over TextArea's default ``tab`` (which inserts a tab
        # character) — typing a real tab in chat is rare, completing
        # a half-typed ``/sess`` is common.
        Binding("tab", "complete_slash", "Complete", show=False, priority=True),
    ]

    def __init__(self, placeholder: str = "", **kwargs) -> None:
        super().__init__(
            language=None,
            show_line_numbers=False,
            soft_wrap=True,
            placeholder=placeholder,
            **kwargs,
        )
        self._history = CommandHistory()
        # Index into history during navigation: -1 = typing freely (not
        # in history), 0 = most recent, N = older entries.
        self._history_index: int = -1
        # Text the user was typing *before* they pressed Up. Restored
        # when Down cycles past the oldest entry back to free typing.
        self._draft_text: str = ""

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        # Compatible with `event.input.value = ""` in `on_input_submitted`.
        self.text = text

    def action_submit(self) -> None:
        # Incomplete slash tokens (``/ag``) are resolved first so Enter
        # never ships a truncated command to the model as a chat turn —
        # the failure mode that makes ``/agent`` look broken.
        text = self.value.strip()
        if text.startswith("/") and " " not in text and "\n" not in text:
            rows = self._matching_slash_commands(text)
            submit, rewrite = resolve_slash_on_submit(text, rows)
            if submit is None:
                if rewrite is not None:
                    self.text = rewrite
                    self.cursor_location = (0, len(rewrite))
                    self._refresh_slash_popup()
                try:
                    set_status = getattr(self.app, "_set_status", None)
                    if callable(set_status):
                        set_status("keep typing — Tab completes, Enter needs a full slash command")
                except Exception:  # noqa: BLE001
                    pass
                return
            if submit != text:
                self.text = submit
                text = submit
        if text:
            self._history.add(text)
        # Reset history navigation state on submit.
        self._history_index = -1
        self._draft_text = ""
        self.post_message(Input.Submitted(self, self.value))

    def action_insert_newline(self) -> None:
        self.insert("\n")

    # ---- slash-command autocomplete ----------------------------------

    def _slash_partial(self) -> str | None:
        """Return the leading ``/<word>`` token of the input if the
        composition is exactly a single slash command (no space after,
        no second line). Otherwise ``None`` — the popup stays empty
        and Tab falls through to its TextArea default.

        Why "no space after"? Once the user's typed past the command
        name (``/model `` or ``/sessions <hash>``) the args are
        free-form and the popup would be more in the way than helpful.
        """
        text = self.text
        if not text.startswith("/"):
            return None
        # Suppress popup the moment the user starts typing args or
        # composes multi-line — neither needs command completion.
        if " " in text or "\n" in text:
            return None
        return text

    def _matching_slash_commands(self, partial: str) -> list[tuple[str, str, str]]:
        """All commands whose token starts with ``partial`` (case-fold).

        Returns the full ``(token, signature, description)`` rows so
        callers can render BOTH the signature and the description,
        and so Tab completion has the canonical token to fill in.

        Combines the hardcoded ``SLASH_COMMAND_HELP`` (built-in TUI
        handlers like ``/help``, ``/plugin``, ``/skills``) with the
        live ``discover_commands()`` listing for file-based commands
        like ``/goal``. Hardcoded built-ins win on token collision so
        an operator's ``commands/help.md`` can't accidentally shadow
        the TUI's own ``/help`` handler — matches the dispatch order
        in ``on_input_submitted`` where built-in handlers run first
        and only unrecognised slashes fall through to file commands.
        """
        all_rows = merge_slash_command_rows(SLASH_COMMAND_HELP, file_command_rows())
        if not partial:
            return all_rows
        p = partial.lower()
        return [row for row in all_rows if row[0].lower().startswith(p)]

    def _refresh_slash_popup(self) -> None:
        """Repaint ``#slash-popup`` based on the current input value.

        Called from the TextArea ``Changed`` event hook on the parent
        app. Hidden (empty Static content) when the input doesn't
        start with a slash, or has already moved past the command
        name. Capped at 8 rows so a cursor on ``/`` (which matches
        every command) doesn't push the transcript off-screen.
        """
        try:
            popup = self.app.query_one("#slash-popup", Static)
        except Exception:  # noqa: BLE001
            return
        partial = self._slash_partial()
        if partial is None:
            popup.update("")
            return
        rows = self._matching_slash_commands(partial)
        if not rows:
            popup.update(f"[dim]no slash command matches {partial!r}[/dim]")
            return
        # Render up to 8 rows so a long match list doesn't squash the
        # transcript. Each row: ``  <signature>      <description>``.
        # Highlight the exact-prefix-match column with bold so the
        # operator can scan the column quickly.
        sig_w = max(len(sig) for _, sig, _ in rows[:8])
        sig_w = min(sig_w, 30)  # cap absurdly long signatures
        lines: list[str] = []
        for _token, sig, desc in rows[:8]:
            sig_padded = sig.ljust(sig_w)
            lines.append(f"  [bold cyan]{sig_padded}[/bold cyan]  [dim]{desc}[/dim]")
        if len(rows) > 8:
            lines.append(f"  [dim italic](+{len(rows) - 8} more — keep typing to narrow)[/dim italic]")
        popup.update("\n".join(lines))

    def action_complete_slash(self) -> None:
        """Tab handler. Completes the current slash partial.

        - 0 matches: no-op (Tab does nothing).
        - 1 match: fill the input to the full command, leave a trailing
          space if the command takes args (``/model``, ``/sessions``,
          ``/claude``), else leave bare.
        - 2+ matches: extend to the longest common prefix of the
          matching tokens. The popup keeps showing the remaining
          options.

        When NO slash partial is active, fall through to inserting a
        literal tab — preserves the historical behaviour for users who
        actually want a tab character in their prompt.
        """
        partial = self._slash_partial()
        if partial is None:
            self.insert("\t")
            return
        rows = self._matching_slash_commands(partial)
        if not rows:
            return
        if len(rows) == 1:
            token, sig, _ = rows[0]
            takes_args = "<" in sig or "[" in sig
            self.text = token + (" " if takes_args else "")
            self.cursor_location = (0, len(self.text))
            self._refresh_slash_popup()
            return
        # Multi-match: extend to the LCP of the candidate tokens.
        lcp = _longest_common_prefix([r[0] for r in rows])
        if len(lcp) > len(partial):
            self.text = lcp
            self.cursor_location = (0, len(self.text))
            self._refresh_slash_popup()

    def action_history_prev(self) -> None:
        """Move back one entry in command history (Up arrow)."""
        if not self._history.entries:
            return
        # First press: save current text as draft so Down can restore it.
        if self._history_index == -1:
            self._draft_text = self.text
        # Clamp to available entries.
        self._history_index = min(self._history_index + 1, len(self._history.entries) - 1)
        entry = self._history.navigate(self._history_index)
        if entry is not None:
            self.text = entry

    def action_history_next(self) -> None:
        """Move forward one entry in command history (Down arrow)."""
        if self._history_index <= 0:
            # Past the top → restore draft and go back to free typing.
            if self._history_index == 0:
                self.text = self._draft_text
            self._history_index = -1
            return
        self._history_index -= 1
        entry = self._history.navigate(self._history_index)
        if entry is not None:
            self.text = entry

    # Min/max visual rows the input widget grows between. `MIN_ROWS` is
    # what the user sees before typing anything (so the box isn't a
    # one-line sliver). `MAX_ROWS` is the cap before internal scrolling
    # kicks in — high enough that long pastes / multi-paragraph prompts
    # stay fully visible on a normal-height terminal.
    # Empty prompt is one row; grows as the operator types. Was 3, which
    # left a tall blank band above ``>`` that made the TUI feel broken.
    MIN_ROWS = 1
    MAX_ROWS = 20

    def _autosize(self) -> None:
        """Resize the input box to fit its wrapped content.

        Textual's `height: auto` on a `TextArea` (a `ScrollView` subclass)
        doesn't expand with content — the widget stays at `min-height` and
        the cursor scrolls inside the fixed viewport, making typed text
        appear to "disappear" past row 2/3. We compute the wrapped-line
        count directly from `wrapped_document.height` and pin the widget's
        visual height so the cursor always stays in view.
        """
        try:
            lines = max(1, int(self.wrapped_document.height))
        except Exception:  # noqa: BLE001
            return
        target = max(self.MIN_ROWS, min(self.MAX_ROWS, lines))
        # Setting an int sets cells. Cheap when unchanged — Textual no-ops
        # if the new value matches the current scalar.
        self.styles.height = target

    def on_mount(self) -> None:
        # Ensure the initial size honors `MIN_ROWS` even before any edit.
        self._autosize()

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        # Fired on every edit (including programmatic `self.text = …` via
        # history navigation, so resize tracks history-restored prompts).
        self._autosize()
        # Repaint the slash-command autocomplete panel. Cheap (one
        # ``query_one`` + a Static.update with a precomputed string),
        # so running it on every keystroke is fine.
        self._refresh_slash_popup()
