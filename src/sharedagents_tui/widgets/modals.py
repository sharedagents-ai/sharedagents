"""Modal screens for the sharedagents TUI (extracted from app.py)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    ListItem,
    ListView,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)

from sharedagents_tui.helpers import _humanize_age, _rows_signature


class PermissionModal(ModalScreen[str]):
    """Three-button confirmation modal for `permission_request` events.

    The screen returns one of `"yes"` / `"yes_dont_ask"` / `"no"` via
    `dismiss(...)`. The caller (TUI's `_handle_permission_request`) then
    POSTs the answer to `/v1/turns/{tid}/permission` and the agent
    resumes the dispatch loop.
    """

    CSS = """
    PermissionModal {
        align: center middle;
    }
    #perm-box {
        width: 80%;
        max-width: 100;
        height: auto;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #perm-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    #perm-args {
        background: $boost;
        color: $text;
        padding: 1;
        margin-bottom: 1;
    }
    #perm-buttons {
        height: auto;
        align-horizontal: right;
    }
    #perm-buttons Button { margin-left: 1; }
    """

    BINDINGS = [
        Binding("y", "answer('yes')", "Yes", show=True),
        Binding("a", "answer('yes_dont_ask')", "Always", show=True),
        Binding("n", "answer('no')", "No", show=True),
        Binding("escape", "answer('no')", "No", show=False),
    ]

    def __init__(self, *, tool: str, kind: str, arguments: dict) -> None:
        super().__init__()
        self._tool = tool
        self._kind = kind
        self._arguments = arguments

    def compose(self) -> ComposeResult:
        with Vertical(id="perm-box"):
            yield Label("⚠️  Tool call requires confirmation", id="perm-title")
            yield Label(f"tool: {self._tool}  ·  kind: {self._kind}")
            yield Static(
                json.dumps(self._arguments, indent=2, ensure_ascii=False),
                markup=False,
                id="perm-args",
            )
            with Horizontal(id="perm-buttons"):
                yield Button("Yes (y)", id="b-yes", variant="success")
                yield Button("Yes, don't ask again (a)", id="b-always", variant="primary")
                yield Button("No (n / esc)", id="b-no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {"b-yes": "yes", "b-always": "yes_dont_ask", "b-no": "no"}
        self.dismiss(mapping.get(event.button.id, "no"))

    def action_answer(self, response: str) -> None:
        self.dismiss(response)


class AcpPermissionModal(ModalScreen[str]):
    """Three-button modal for an ACP ``session/request_permission``.

    Distinct from :class:`PermissionModal` (the legacy CLI-engine
    ``permission_request`` event path): this one is wired straight to
    the ACP inbound handler — :meth:`AgentChatApp._on_request_permission`
    pops it from inside the client read loop and the agent's outbound
    request stays parked on its future until ``dismiss(...)`` fires.

    ``dismiss`` returns one of the ACP option ids — ``"allow_once"`` /
    ``"allow_always"`` / ``"reject_once"``. Esc and every error path
    resolve to ``"reject_once"``; otherwise the modal remains open until
    the operator makes an explicit decision.
    """

    # esc/error resolve to reject_once and elapsed wall-clock time never
    # supplies an answer.
    CSS = """
    AcpPermissionModal {
        align: center middle;
    }
    #acp-perm-box {
        width: 80%;
        max-width: 100;
        height: auto;
        border: thick $error;
        background: $panel;
        padding: 1 2;
    }
    #acp-perm-title {
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }
    #acp-perm-args {
        background: $boost;
        color: $text;
        padding: 1;
        margin-bottom: 1;
    }
    #acp-perm-buttons {
        height: auto;
        align-horizontal: right;
    }
    #acp-perm-buttons Button { margin-left: 1; }
    """

    # AgentChatApp registers `Binding("escape", "abort", priority=True)` at
    # App level. Textual checks priority bindings App→screen→focused, so
    # without priority=True here the modal's own Esc binding never fires —
    # pressing Esc terminates the agent instead of rejecting the permission,
    # leaving the modal orphaned on screen. Marking all four priority keeps
    # the contract uniform (Esc / a / s / r all resolve through the modal).
    BINDINGS = [
        Binding("a", "answer('allow_once')", "Allow once", show=True, priority=True),
        Binding("s", "answer('allow_always')", "Allow always", show=True, priority=True),
        Binding("r", "answer('reject_once')", "Reject", show=True, priority=True),
        Binding("escape", "answer('reject_once')", "Reject", show=False, priority=True),
    ]

    def __init__(self, params: dict) -> None:
        super().__init__()
        tool_call = params.get("toolCall") if isinstance(params, dict) else {}
        if not isinstance(tool_call, dict):
            tool_call = {}
        self._title = str(tool_call.get("title") or "tool")
        self._kind = str(tool_call.get("kind") or "other")
        raw_input = tool_call.get("rawInput")
        self._arguments = raw_input if isinstance(raw_input, dict) else {}
        self._resolved = False

    def compose(self) -> ComposeResult:
        with Vertical(id="acp-perm-box"):
            yield Label("⛔  Destructive tool call needs approval", id="acp-perm-title")
            yield Label(f"tool: {self._title}  ·  kind: {self._kind}")
            yield Static(
                json.dumps(self._arguments, indent=2, ensure_ascii=False),
                markup=False,
                id="acp-perm-args",
            )
            with Horizontal(id="acp-perm-buttons"):
                yield Button("Allow once (a)", id="acp-b-allow", variant="success")
                yield Button("Allow always (s)", id="acp-b-always", variant="primary")
                yield Button("Reject (r / esc)", id="acp-b-reject", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "acp-b-allow": "allow_once",
            "acp-b-always": "allow_always",
            "acp-b-reject": "reject_once",
        }
        self.action_answer(mapping.get(event.button.id, "reject_once"))

    def action_answer(self, option_id: str) -> None:
        # Guard against duplicate button/key events — ModalScreen.dismiss is
        # not re-entrant.
        if self._resolved:
            return
        self._resolved = True
        self.dismiss(option_id)


class QuestionsModal(ModalScreen[dict]):
    """Multi-question selection modal for ``session/ask_user``.

    Renders 1–4 questions, each with 2–4 options + an "Other" free-form
    Input. Single-select questions use a ``RadioSet``; multi-select
    questions use a vertical ``Checkbox`` group. Submit returns a dict
    shaped like the wire response::

        {
          "answers": [
            {"question": <str>, "selected": [<label>, ...], "notes": <str>},
            ...
          ]
        }

    Esc / Cancel returns an empty answers list so the agent's bridge
    can synthesize a "user cancelled" tool_result rather than hang.
    """

    CSS = """
    QuestionsModal {
        align: center middle;
    }
    #q-box {
        width: 90%;
        max-width: 120;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #q-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #q-scroll {
        height: auto;
        max-height: 30;
    }
    .q-panel {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
        border-left: thick $accent 50%;
    }
    .q-header {
        text-style: bold;
        color: $warning;
    }
    .q-text {
        margin-bottom: 1;
    }
    .q-notes-label {
        color: $text-muted;
        margin-top: 1;
    }
    #q-buttons {
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }
    #q-buttons Button { margin-left: 1; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    def __init__(self, questions: list[dict]) -> None:
        super().__init__()
        # Defensive copy so we own the structure for the lifetime of
        # the modal — the agent_client sometimes hands us aliased
        # dicts from the JSON-RPC frame.
        self._questions: list[dict] = [dict(q) for q in (questions or []) if isinstance(q, dict)]
        # Map question index → (radio_set_id | None, [(option_label, checkbox_id), ...], notes_input_id)
        # so the submit handler can walk the widget tree by id rather than
        # holding refs (which Textual reuses across screen pushes).
        self._index: list[dict] = []

    def on_mount(self) -> None:
        """Focus the first question's selector so keyboard nav works
        without an extra Tab. Without this, Textual's default tab
        order lands on the Submit button — so number-key bindings
        and arrow-key option nav both miss their target until the
        operator manually Tabs back into the question panel."""
        if not self._index:
            return
        first = self._index[0]
        try:
            if first.get("multi") and first.get("checkboxes"):
                _label, cb_id = first["checkboxes"][0]
                self.query_one(f"#{cb_id}", Checkbox).focus()
            elif first.get("radio_id"):
                self.query_one(f"#{first['radio_id']}", RadioSet).focus()
        except Exception:  # noqa: BLE001
            # Widget tree not ready yet (race against Textual's mount
            # ordering) — fall back to default focus rather than crash
            # the modal on a layout edge case.
            pass

    def _focused_question_index(self) -> int | None:
        """Return the index of the question whose RadioSet / Checkbox /
        RadioButton / notes Input currently has focus, or ``None`` if
        focus is outside the question panels (e.g. on Submit / Cancel).
        Widget ids follow the ``q-{qi}-…`` pattern set in
        :meth:`_compose_question`, so a single regex covers every
        selector kind."""
        f = self.focused
        if f is None:
            return None
        import re

        m = re.match(r"q-(\d+)-", f.id or "")
        if not m:
            return None
        qi = int(m.group(1))
        return qi if 0 <= qi < len(self._index) else None

    def on_key(self, event) -> None:  # noqa: ANN001
        """Number-key (1–4) selects option N of the currently-focused
        question. Yields to focused ``Input`` widgets so typing a
        digit into the "Other (free-form)" field stays a digit instead
        of toggling an option behind the user's back — matches the
        principle of least surprise. Operators on a Cancel/Submit
        button get the same shortcut to set the focused question
        without round-tripping back through Tab."""
        key = event.key
        if key not in ("1", "2", "3", "4"):
            return
        if isinstance(self.focused, Input):
            # Operator is typing in the notes field; the digit belongs
            # to the input, not us.
            return
        qi = self._focused_question_index()
        if qi is None:
            return
        oi = int(key) - 1
        entry = self._index[qi]
        options = entry.get("checkboxes") or []
        if oi >= len(options):
            return
        event.stop()
        _label, widget_id = options[oi]
        try:
            if entry.get("multi"):
                cb = self.query_one(f"#{widget_id}", Checkbox)
                cb.value = not cb.value
                cb.focus()
            else:
                # Set the matching RadioButton; RadioSet's own logic
                # turns the previously-pressed button off.
                rb = self.query_one(f"#{widget_id}", RadioButton)
                rb.value = True
                rb.focus()
        except Exception:  # noqa: BLE001
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="q-box"):
            n = len(self._questions)
            yield Label(
                f"❓  {n} question{'s' if n != 1 else ''}  ·  Ctrl+S submit  ·  Esc cancel",
                id="q-title",
            )
            with VerticalScroll(id="q-scroll"):
                for qi, q in enumerate(self._questions):
                    yield from self._compose_question(qi, q)
            with Horizontal(id="q-buttons"):
                yield Button("Submit (Ctrl+S)", id="q-submit", variant="success")
                yield Button("Cancel (Esc)", id="q-cancel", variant="error")

    def _compose_question(self, qi: int, q: dict) -> ComposeResult:
        header = str(q.get("header") or "").strip() or f"Question {qi + 1}"
        question = str(q.get("question") or "").strip() or "(no question text)"
        multi = bool(q.get("multiSelect"))
        options = q.get("options") or []
        with Vertical(classes="q-panel"):
            yield Static(f"[{header}]", classes="q-header")
            yield Label(question, classes="q-text")
            entry: dict = {"multi": multi, "checkboxes": [], "radio_id": None}
            if multi:
                # One Checkbox per option, stacked vertically.
                for oi, opt in enumerate(options):
                    if not isinstance(opt, dict):
                        continue
                    label = str(opt.get("label") or "").strip()
                    desc = str(opt.get("description") or "").strip()
                    if not label:
                        continue
                    cb_label = f"{label} — {desc}" if desc else label
                    cb_id = f"q-{qi}-cb-{oi}"
                    entry["checkboxes"].append((label, cb_id))
                    yield Checkbox(cb_label, id=cb_id)
            else:
                rs_id = f"q-{qi}-radio"
                entry["radio_id"] = rs_id
                # RadioSet wraps a tuple of RadioButtons; the first
                # button is pre-selected so Submit always has a value.
                buttons: list[RadioButton] = []
                first_set = False
                for oi, opt in enumerate(options):
                    if not isinstance(opt, dict):
                        continue
                    label = str(opt.get("label") or "").strip()
                    desc = str(opt.get("description") or "").strip()
                    if not label:
                        continue
                    rb_label = f"{label} — {desc}" if desc else label
                    rb_id = f"q-{qi}-rb-{oi}"
                    entry["checkboxes"].append((label, rb_id))
                    buttons.append(RadioButton(rb_label, id=rb_id, value=not first_set))
                    first_set = True
                yield RadioSet(*buttons, id=rs_id)
            notes_id = f"q-{qi}-notes"
            entry["notes_id"] = notes_id
            yield Label("Other (free-form, optional):", classes="q-notes-label")
            yield Input(placeholder="…", id=notes_id)
            self._index.append(entry)

    def _collect_answers(self) -> dict:
        answers: list[dict] = []
        for qi, q in enumerate(self._questions):
            try:
                entry = self._index[qi]
            except IndexError:
                continue
            selected: list[str] = []
            if entry["multi"]:
                for label, cb_id in entry["checkboxes"]:
                    try:
                        cb = self.query_one(f"#{cb_id}", Checkbox)
                    except Exception:  # noqa: BLE001
                        continue
                    if cb.value:
                        selected.append(label)
            else:
                rs_id = entry.get("radio_id")
                if rs_id:
                    try:
                        rs = self.query_one(f"#{rs_id}", RadioSet)
                    except Exception:  # noqa: BLE001
                        rs = None
                    if rs is not None and rs.pressed_button is not None:
                        pressed_id = rs.pressed_button.id or ""
                        for label, rb_id in entry["checkboxes"]:
                            if rb_id == pressed_id:
                                selected.append(label)
                                break
            notes = ""
            notes_id = entry.get("notes_id")
            if notes_id:
                try:
                    notes = self.query_one(f"#{notes_id}", Input).value.strip()
                except Exception:  # noqa: BLE001
                    notes = ""
            answers.append(
                {
                    "question": str(q.get("question") or ""),
                    "selected": selected,
                    "notes": notes,
                }
            )
        return {"answers": answers}

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "q-submit":
            self.dismiss(self._collect_answers())
        elif event.button.id == "q-cancel":
            self.dismiss({"answers": [], "cancelled": True})

    def action_submit(self) -> None:
        self.dismiss(self._collect_answers())

    def action_cancel(self) -> None:
        self.dismiss({"answers": [], "cancelled": True})


class MonitorPanelModal(ModalScreen[None]):
    """Live shells panel — list every subprocess monitor for the active
    session, navigate with up/down, press ``x`` to SIGTERM the
    selected one, ``q`` / ``Escape`` to dismiss.

    Two-pane layout:

    - **Top:** every shell the model spawned via ``monitor_start(...)``.
      Source: per-session sidecar at
      ``<sessions_dir>/<sid>.monitors.json`` written by every
      ``monitor_start`` / ``monitor_stop`` runner call. A 1.5 s timer
      refreshes the list so a process that exits behind the operator's
      back disappears without a manual reload.
    - **Bottom:** tail of the currently-selected shell's logfile (path
      carried in the sidecar's ``log_path`` field; the monitor
      subprocess writes to it directly so reading from the TUI is
      process-safe — no IPC needed). Auto-scrolls in follow-mode (``f``
      toggles); ``Ctrl+L`` clears.

    Discoverability: opened via the ``/shells`` slash command, the
    older ``/monitors`` alias, or the ``Ctrl+S`` key binding.

    ``x`` sends SIGTERM directly to the recorded PID — same path
    ``MonitorRegistry._stop_by_sidecar`` uses, just bypassing the
    registry round-trip so the panel is responsive even if no tool
    subprocess is currently alive.
    """

    CSS = """
    MonitorPanelModal {
        align: center middle;
    }
    #mon-box {
        width: 90%;
        max-width: 160;
        height: 90%;
        max-height: 50;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #mon-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    /* Top pane: the list of subprocesses. Fixed minimum + flex up to
       40% of the modal so a long list doesn't crowd the output pane. */
    #mon-list {
        height: 30%;
        min-height: 6;
        border: round $accent 30%;
        background: $boost;
    }
    #mon-list > ListItem {
        padding: 0 1;
        height: 1;
    }
    #mon-list > ListItem.--highlight {
        background: $accent 30%;
    }
    #mon-list > ListItem.dead {
        color: $text-muted;
    }
    /* Bottom pane: tail of the currently-selected shell's logfile. */
    #mon-out-header {
        color: $text-muted;
        margin: 1 0 0 0;
    }
    #mon-out {
        height: 1fr;
        border: round $accent 30%;
        background: $boost;
        padding: 0 1;
        overflow-y: auto;
    }
    #mon-help {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close", show=True),
        Binding("q", "cancel", "Close", show=True),
        Binding("x", "kill", "Kill", show=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("r", "reload", "Reload", show=True),
        Binding("ctrl+l", "clear_output", "Clear", show=True),
        Binding("f", "follow", "Follow", show=True),
    ]

    REFRESH_INTERVAL_S = 1.5
    # Tail of the selected shell's logfile shown in the bottom pane.
    # Big enough to read a useful slice of stdout, small enough that a
    # runaway monitor doesn't lock the modal up.
    OUTPUT_TAIL_BYTES = 16 * 1024
    OUTPUT_TAIL_LINES = 200

    def __init__(self, sessions_dir: Path, session_id: str) -> None:
        super().__init__()
        self._sessions_dir = Path(sessions_dir)
        self._session_id = session_id
        self._rows: list[dict] = []
        self._listview: ListView | None = None
        self._output: RichLog | None = None
        self._output_header: Label | None = None
        # Repaint dedup. Each tick we recompute the visible state
        # (rows list + output pane body) and only push it into the
        # widgets when it actually differs from what was rendered
        # last tick. Without this the periodic ``_refresh`` rebuilt
        # the ListView and re-wrote the RichLog every 1.5 s
        # unconditionally, which made the highlighted row and the
        # log tail flicker visibly even when nothing had changed.
        self._last_rows_sig: tuple | None = None
        self._last_output_sig: tuple | None = None
        # When ``_follow`` is True the bottom pane auto-scrolls to the
        # end on every refresh. Disabled by the operator (Ctrl+L /
        # manual scroll) so they can pin to a particular slice of
        # output without it jumping away on each tick.
        self._follow = True
        self._timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="mon-box"):
            yield Label(
                "🖥️   Shells  ·  ↑/↓ navigate  ·  x kill  ·  r reload  ·  f follow  ·  Ctrl+L clear  ·  q/Esc close",
                id="mon-title",
            )
            # Seed empty; on_mount's first _refresh fetches rows from the
            # gateway (async) and populates the list.
            self._rows = []
            self._listview = ListView(
                *self._build_items(self._rows),
                id="mon-list",
            )
            yield self._listview
            self._output_header = Label(
                "(select a shell to tail its output)",
                id="mon-out-header",
            )
            yield self._output_header
            # ``RichLog`` is purpose-built for tailing line-oriented
            # output: it scrolls itself, handles overflow, and accepts
            # plain strings via ``.write(line)``. We ``.clear()`` + bulk-
            # write on each refresh so the tail always reflects the
            # latest slice of the logfile rather than an ever-growing
            # buffer.
            self._output = RichLog(
                id="mon-out",
                wrap=False,
                highlight=False,
                markup=False,
                auto_scroll=True,
            )
            yield self._output
            yield Label(
                "Tip: shells are spawned by the model via "
                "``monitor_start(...)``. The model also has tools to "
                "send input (``monitor_input``) and read output "
                "(``BashOutput``) — this panel is the operator's view.",
                id="mon-help",
            )

    def on_mount(self) -> None:
        # Periodic refresh so a monitor that exits / is killed elsewhere
        # disappears without manual reload. The same tick re-tails the
        # selected shell's logfile, so the bottom pane updates without
        # the operator pressing anything.
        self._timer = self.set_interval(self.REFRESH_INTERVAL_S, self._refresh)
        # Fetch the initial rows from the gateway right away (async), then
        # the interval keeps them fresh.
        self._refresh()

    def on_list_view_highlighted(self, event: Any) -> None:
        """ListView highlight changed (cursor moved up/down/clicked).
        Re-tail the newly selected shell's logfile immediately so the
        bottom pane responds without waiting for the next refresh tick."""
        # We don't filter on the event source — there's only one ListView
        # in this modal so any highlighted event refers to the shells list.
        self._refresh_output()

    async def _server_cmd(self, text: str) -> dict | None:
        """Run a slash command through the gateway runtime (single source of
        truth). The monitor panel renders shells but does NOT read the
        sidecar / kill processes itself — it goes through ``/monitors`` so
        the same implementation serves TUI and web."""
        client = getattr(self.app, "_client", None)
        sid = getattr(self.app, "_sid", None)
        if client is None or not sid:
            return None
        try:
            return await client.session_command(sid, text, frontend="tui")
        except Exception:  # noqa: BLE001
            return None

    async def _load_rows(self) -> list[dict]:
        env = await self._server_cmd("/monitors")
        if not env or not env.get("ok"):
            return []
        data = env.get("data") if isinstance(env, dict) else None
        if not data and isinstance(env, dict) and isinstance(env.get("result"), dict):
            data = env["result"].get("data")
        return list((data or {}).get("rows") or [])

    def _build_items(self, rows: list[dict]) -> list[ListItem]:
        items: list[ListItem] = []
        if not rows:
            items.append(
                ListItem(
                    Static(
                        "(no live monitors)",
                        markup=False,
                    )
                )
            )
            return items
        for n, r in enumerate(rows, start=1):
            alive = bool(r.get("alive"))
            state = "running" if alive else "exited"
            pid = r.get("pid")
            mid = (r.get("monitor_id") or "")[:12]
            cmd = (r.get("command") or "")[:80]
            row = f" {n:>2}. [{state:<7}] pid={pid!s:<7} {mid}  ·  {cmd}"
            item = ListItem(Static(row, markup=False))
            if not alive:
                item.add_class("dead")
            items.append(item)
        return items

    def _refresh(self) -> None:
        # Data comes from the gateway (async); schedule the fetch+render.
        self.run_worker(self._refresh_async(), exclusive=False, name="mon-refresh")

    async def _refresh_async(self) -> None:
        if self._listview is None:
            return
        prev_idx = self._listview.index or 0
        # The server's /monitors view (sidecar-backed) is authoritative —
        # exited shells drop out there, so no client-side sidecar mutation.
        self._rows = await self._load_rows()
        # Skip the ListView rebuild when nothing visible changed —
        # otherwise the periodic refresh clears + re-appends every
        # row every 1.5 s, briefly dropping the highlighted-row
        # background, which the operator sees as a steady blink.
        sig = _rows_signature(self._rows)
        if sig != self._last_rows_sig:
            self._listview.clear()
            for item in self._build_items(self._rows):
                self._listview.append(item)
            if self._rows:
                self._listview.index = min(prev_idx, len(self._rows) - 1)
            self._last_rows_sig = sig
        await self._refresh_output_async()

    def _selected_row(self) -> dict | None:
        """Return the row dict the operator currently has highlighted,
        or None if the list is empty / no valid index."""
        if not self._rows or self._listview is None:
            return None
        idx = self._listview.index
        if idx is None or not (0 <= idx < len(self._rows)):
            return None
        return self._rows[idx]

    async def _load_tail(self, row: dict) -> str:
        """Tail the selected shell's logfile via the gateway runtime
        (``/monitors tail <id>``) — no direct disk read, so the panel
        works the same whether the TUI is local or remote."""
        mid = row.get("monitor_id") or ""
        if not mid:
            return "(no monitor id)"
        env = await self._server_cmd(f"/monitors tail {mid}")
        if not env or not env.get("ok"):
            return "(tail unavailable)"
        return env.get("text") or ((env.get("result") or {}).get("text") if isinstance(env, dict) else "") or "(empty)"

    def _refresh_output(self) -> None:
        # Tail comes from the gateway (async); schedule the fetch+render.
        self.run_worker(self._refresh_output_async(), exclusive=False, name="mon-tail")

    async def _refresh_output_async(self) -> None:
        """Repaint the bottom pane with the current selection's logfile
        tail. Updates the header to identify which shell is showing.
        Safe to call before the modal has fully mounted (the widget
        handles are None then — bail quietly).

        Skips the RichLog clear+rewrite when the body+header are
        unchanged from the last tick — without that guard the modal
        flickered visibly every 1.5 s on a quiet shell because we
        cleared and re-wrote identical lines."""
        if self._output is None or self._output_header is None:
            return
        row = self._selected_row()
        if row is None:
            empty_sig = ("__empty__",)
            if self._last_output_sig != empty_sig:
                self._output_header.update("(no shells)")
                self._output.clear()
                self._last_output_sig = empty_sig
            return
        mid = (row.get("monitor_id") or "")[:12]
        pid = row.get("pid")
        alive = bool(row.get("alive"))
        state = "● running" if alive else "○ exited"
        cmd_preview = (row.get("command") or "")[:80]
        header = f"{state}  ·  pid={pid}  ·  {mid}  ·  {cmd_preview}"
        if not self._follow:
            header += "  ·  [paused — press f to resume follow]"
        body = await self._load_tail(row)
        sig = (mid, header, body)
        if sig == self._last_output_sig:
            return
        self._output_header.update(header)
        # Bulk replace: clear, then write each line. RichLog appends
        # internally and auto-scrolls; we lean on its own scroll
        # behaviour rather than re-implementing.
        self._output.clear()
        for line in body.splitlines() or [""]:
            self._output.write(line)
        self._last_output_sig = sig
        # ``auto_scroll=True`` (set at construction) keeps us pinned to
        # the bottom when follow-mode is on. If the operator paused
        # follow we leave the scroll position alone.

    def action_clear_output(self) -> None:
        """Ctrl+L — clear the bottom pane visually (next tick refills).
        Useful when a noisy shell has flooded the buffer and the
        operator wants a clean slate to read from."""
        if self._output is not None:
            self._output.clear()

    def action_follow(self) -> None:
        """f — toggle follow-mode for the bottom pane. When OFF, the
        pane keeps the operator's manual scroll position; when ON, it
        auto-scrolls to the end on every refresh.

        Toggling also flips ``RichLog.auto_scroll`` so freshly-written
        lines during the next refresh respect the operator's choice."""
        self._follow = not self._follow
        if self._output is not None:
            self._output.auto_scroll = self._follow
        self._refresh_output()

    def action_reload(self) -> None:
        self._refresh()

    def action_cursor_up(self) -> None:
        if self._listview is not None:
            self._listview.action_cursor_up()
            # ListView fires Highlighted as a Message, which our
            # `on_list_view_highlighted` handler picks up — that's where
            # the output pane re-tail happens. Explicit call here as a
            # belt-and-suspenders in case the message dispatch races a
            # rapid up/up burst.
            self._refresh_output()

    def action_cursor_down(self) -> None:
        if self._listview is not None:
            self._listview.action_cursor_down()
            self._refresh_output()

    def action_kill(self) -> None:
        """SIGTERM the selected monitor's PID, then refresh.

        Two states:

        - **alive row** → SIGTERM the process group (own session, so
          pid==pgid). Sidecar entry stays until the next refresh
          tick observes ``alive=False`` and auto-drops it.
        - **dead row** → drop the stale sidecar entry directly. Lets
          the operator clean up old rows without restarting the panel.
        """
        if not self._rows or self._listview is None:
            return
        idx = self._listview.index or 0
        if not (0 <= idx < len(self._rows)):
            return
        row = self._rows[idx]
        mid = row.get("monitor_id")
        if not mid:
            return

        # Kill goes through the gateway runtime (``/monitors kill <id>``) —
        # the server owns the registry / SIGTERM, so the TUI doesn't touch
        # processes directly. Refresh after.
        async def _do_kill() -> None:
            await self._server_cmd(f"/monitors kill {mid}")
            await self._refresh_async()

        self.run_worker(_do_kill(), exclusive=False, name="mon-kill")
        self._log_panel_event("MONITOR_PANEL_KILL", monitor_id=mid, pid=row.get("pid"), via="server")

    def _log_panel_event(self, event: str, **fields: Any) -> None:
        """Append a structured event line to the shared gateway.log.

        The TUI process logs through the unified JSONL writer so its
        events sit alongside the runtime's in ``logs/gateway.log``,
        tagged with ``agent`` (from the panel's context) and
        ``service="gateway"``. Best-effort: any I/O / validation
        error is swallowed so a logging failure can never break the
        panel.
        """
        try:
            from sharedagents_core.jsonlog import get_writer
            from sharedagents_core.layout import gateway_logfile

            agent = (
                getattr(self, "_agent_name", None) or getattr(getattr(self, "agent_cfg", None), "name", None) or "tui"
            )
            session_id = getattr(self, "_session_id", "") or ""
            w = get_writer(gateway_logfile(), service="gateway")
            w.write(
                agent=str(agent),
                level="INFO",
                msg=event.lower(),
                kind=event,
                session_id=str(session_id) if session_id else None,
                k=dict(fields) if fields else None,
            )
        except (OSError, ValueError):
            pass

    def action_cancel(self) -> None:
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:  # noqa: BLE001
                pass
        self.dismiss(None)


class SessionPickerModal(ModalScreen[str | None]):
    """Modal session picker invoked by `/sessions` (no args).

    Shows each session as a `ListItem` with:

        ● 1.  69ef481f1234  · 5 min ago   · 12 turns  · tui:/home/me/proj
              how to continue on the previous session?

    Up / Down navigate, Enter selects, Esc cancels. The currently
    active session is pre-highlighted with a `●` marker and pre-
    selected on open so a single Enter is a no-op.

    Returns the selected session's hash on success, `None` on cancel.
    """

    CSS = """
    SessionPickerModal {
        align: center middle;
    }
    #picker-box {
        width: 90%;
        max-width: 140;
        height: 80%;
        max-height: 40;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #picker-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #picker-list {
        height: 1fr;
        border: round $accent 30%;
        background: $boost;
    }
    #picker-list > ListItem {
        padding: 0 1;
    }
    #picker-list > ListItem > Vertical {
        height: auto;
    }
    #picker-list > ListItem .session-meta {
        text-wrap: nowrap;
        overflow: hidden hidden;
    }
    #picker-list > ListItem .session-title {
        color: $text-muted;
        text-wrap: wrap;
        width: 1fr;
        /* Titles cap at SESSION_TITLE_MAX_CHARS (160). The picker box is
           up to 140 cols but can be ~28 cols on a narrow split, where 160
           chars wraps to ~6 lines — keep them all visible rather than
           clipping at 3. Short titles stay compact (Vertical is auto). */
        max-height: 6;
        overflow: hidden hidden;
    }
    #picker-list > ListItem.-highlight {
        background: $accent 30%;
    }
    #picker-list > ListItem.heartbeat {
        /* Dim heartbeat sessions a notch — still visible, but they
           don't fight for attention against interactive ones. */
        color: $text-muted;
    }
    #picker-list > ListItem.current {
        color: $success;
        text-style: bold;
    }
    #picker-help {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "pick", "Select", show=True),
    ]

    def __init__(
        self,
        sessions: list[dict],
        current_sid: str = "",
    ) -> None:
        super().__init__()
        self._sessions = sessions
        self._current_sid = current_sid

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-box"):
            yield Label(
                f"📁  Sessions ({len(self._sessions)})  ·  ↑/↓ navigate  ·  Enter select  ·  Esc cancel",
                id="picker-title",
            )
            items: list[ListItem] = []
            initial = 0
            for i, s in enumerate(self._sessions):
                h = s.get("hash") or ""
                short = h[:12]
                marker = "●" if h == self._current_sid else " "
                title = (s.get("title") or "(untitled)").strip()
                origin = s.get("workspace_origin") or "?"
                # Heartbeat sessions get a 💓 badge so they're visually
                # distinct from interactive sessions in the picker.
                # They still appear in the list — the user can switch
                # to one to inspect heartbeat history — they just don't
                # hide in plain sight.
                badge = "💓" if origin == "heartbeat" else "  "
                cwd = s.get("workspace_cwd") or "?"
                turns = s.get("turns") or 0
                age = _humanize_age(s.get("last_activity") or "")
                # Two-row list item: metadata on row 1, wrapped title
                # on row 2 so long titles don't get clipped.
                meta_line = f"{marker} {i + 1:>2}. {badge} {short}  ·  {age:<12}·  {turns:>3} turns  ·  {origin}:{cwd}"
                title_line = title if title != "(untitled)" else ""
                item = ListItem(
                    Vertical(
                        Static(meta_line, markup=False, classes="session-meta"),
                        *(Static(title_line, markup=False, classes="session-title"),) if title_line else (),
                    )
                )
                if origin == "heartbeat":
                    item.add_class("heartbeat")
                if h == self._current_sid:
                    item.add_class("current")
                    initial = i
                items.append(item)
            self._listview = ListView(*items, id="picker-list", initial_index=initial)
            yield self._listview
            yield Label(
                "Tip: `/sessions <index|hash-prefix>` switches without the picker.",
                id="picker-help",
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Enter on a row OR a mouse click — both fire `Selected`.
        idx = self._listview.index or 0
        if 0 <= idx < len(self._sessions):
            self.dismiss(self._sessions[idx].get("hash"))
        else:
            self.dismiss(None)

    def action_pick(self) -> None:
        idx = self._listview.index or 0
        if 0 <= idx < len(self._sessions):
            self.dismiss(self._sessions[idx].get("hash"))
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentMonitorModal(ModalScreen[dict | None]):
    """Cross-agent live monitor — the TUI port of the web ``/agents`` page.

    Renders the SAME ``orchestrator.session_tree()`` (agent → session →
    subagent) the web monitor does, so EVERY agent's sessions are
    visible, not just ones spawned on the active session (the legacy
    ``/subagents`` panel only sees this session's children). The tree is
    re-fetched every ~1.5 s; rows reconcile by a content signature so a
    quiet tree doesn't flicker.

    Navigation:

    - ``↑`` / ``↓`` move the cursor.
    - ``Enter`` on a SESSION row dismisses the modal with that session's
      ``{hash, agent, title}`` — the ``/monitor`` handler then switches
      the TUI onto it (hopping agents first when they differ), so the
      operator can read its replayed history and continue it. ``Enter``
      on an AGENT row toggles its subtree.
    - ``Space`` toggles collapse on the current row (agent or session).
    - ``a`` cycles the "seen ≤" age window (any → 5m → 15m → 1h → 6h →
      24h → any); ``i`` toggles idle/closed sessions (server ``?all=1``
      browse scope); ``/`` focuses the search box (free-text over agent /
      title / hash); ``r`` reloads now; ``q`` / ``Esc`` close.

    Collapse + filter state persist to a sidecar
    (``sessions_root()/.tui_monitor.json``) so they survive reopen — the
    same durability the web monitor gets from localStorage. The filter /
    flatten / age logic is the pure ``sharedagents_tui.monitor_view`` port of
    ``web/static/agents.js`` (unit-tested separately).
    """

    CSS = """
    AgentMonitorModal {
        align: center middle;
    }
    #amon-box {
        width: 92%;
        max-width: 170;
        height: 90%;
        max-height: 52;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #amon-title {
        text-style: bold;
        color: $accent;
    }
    #amon-status {
        color: $text-muted;
        margin-bottom: 1;
    }
    #amon-search {
        margin-bottom: 1;
        border: round $accent 40%;
    }
    #amon-list {
        height: 1fr;
        border: round $accent 30%;
        background: $boost;
    }
    #amon-list > ListItem {
        padding: 0 1;
        height: 1;
    }
    #amon-list > ListItem.--highlight {
        background: $accent 30%;
    }
    #amon-empty {
        color: $text-muted;
        height: 1fr;
        content-align: center middle;
    }
    #amon-help {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close", show=True),
        Binding("q", "cancel", "Close", show=True),
        Binding("enter", "activate", "Open/Toggle", show=True),
        Binding("space", "toggle", "Collapse", show=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("a", "cycle_age", "Age window", show=True),
        Binding("i", "toggle_all", "Idle/closed", show=True),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("r", "reload", "Reload", show=True),
    ]

    REFRESH_INTERVAL_S = 1.5
    # "seen ≤" age windows, in seconds. 0 == any (no cutoff). Mirrors the
    # web monitor's <select> options.
    AGE_WINDOWS = (0, 300, 900, 3600, 21600, 86400)
    AGE_LABELS = {0: "any", 300: "5m", 900: "15m", 3600: "1h", 21600: "6h", 86400: "24h"}

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict] = []
        self._last_tree: dict | None = None
        self._listview: ListView | None = None
        self._status: Label | None = None
        self._search: Input | None = None
        self._empty: Static | None = None
        self._timer = None
        # Structure signature of the last render — keys/kinds/depths/
        # collapse/has_kids ONLY, deliberately NOT the row text. The text
        # carries a relative age ("5s"→"6s") and live round counter that tick
        # every refresh, so folding them into the signature forced a full
        # ListView rebuild every 1.5 s on any live tree → flicker. Now the
        # text is patched in place (see ``_render_tree``) and only a real
        # add / remove / reorder / collapse rebuilds the list.
        self._struct_sig: tuple | None = None
        # The per-row Static widgets and their last-rendered labels, parallel
        # to ``self._rows``, so an in-place patch can target the exact line
        # whose text changed without clearing the list.
        self._row_statics: list[Static] = []
        self._row_labels: list[str] = []
        # Filter + collapse state (hydrated from the sidecar in on_mount).
        self._collapsed: set[str] = set()
        self._query: str = ""
        self._max_age_sec: float = 0
        self._include_all: bool = False

    # -- persistence ------------------------------------------------------
    def _state_path(self) -> Path:
        from sharedagents_core.layout import sessions_root

        return sessions_root() / ".tui_monitor.json"

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self._collapsed = set(data.get("collapsed") or [])
        q = data.get("query")
        if isinstance(q, str):
            # Stored lower-cased (matches_query lower-cases the haystack);
            # normalise on load too so a hand-edited sidecar still matches.
            self._query = q.lower()
        age = data.get("max_age_sec")
        if isinstance(age, (int, float)) and age in self.AGE_WINDOWS:
            self._max_age_sec = age
        self._include_all = bool(data.get("include_all"))

    def _save_state(self) -> None:
        try:
            self._state_path().write_text(
                json.dumps(
                    {
                        "collapsed": sorted(self._collapsed),
                        "query": self._query,
                        "max_age_sec": self._max_age_sec,
                        "include_all": self._include_all,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    # -- compose / mount --------------------------------------------------
    def compose(self) -> ComposeResult:
        self._load_state()
        with Vertical(id="amon-box"):
            yield Label("🛰️   Agent monitor", id="amon-title")
            self._status = Label("connecting…", id="amon-status")
            yield self._status
            self._search = Input(
                value=self._query,
                placeholder="search agent / title / hash  (/ to focus)",
                id="amon-search",
            )
            yield self._search
            self._listview = ListView(id="amon-list")
            yield self._listview
            self._empty = Static("", id="amon-empty")
            self._empty.display = False
            yield self._empty
            yield Label(
                "↑/↓ navigate · Enter open/toggle · Space collapse · "
                "a age · i idle/closed · / search · r reload · q/Esc close",
                id="amon-help",
            )

    def on_mount(self) -> None:
        self._timer = self.set_interval(self.REFRESH_INTERVAL_S, self._refresh)
        self._refresh()
        if self._listview is not None:
            self._listview.focus()

    # -- data fetch + render ---------------------------------------------
    def _refresh(self) -> None:
        self.run_worker(self._refresh_async(), exclusive=False, name="amon-refresh")

    async def _refresh_async(self) -> None:
        # Bail if the modal was dismissed between the timer tick and this
        # worker running — otherwise we'd render into a torn-down widget.
        if not self.is_mounted:
            return
        client = getattr(self.app, "_client", None)
        if client is None:
            return
        try:
            tree = await client.orchestrator_tree(include_all=self._include_all)
        except Exception:  # noqa: BLE001
            tree = {"agents": [], "summary": [], "totals": {}}
        self._last_tree = tree
        self._render_tree()

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _render_tree(self) -> None:
        if not self.is_mounted or self._listview is None or self._last_tree is None:
            return
        from sharedagents_tui import monitor_view as _mv

        now_ms = self._now_ms()
        self._rows = _mv.build_rows(
            self._last_tree,
            collapsed=self._collapsed,
            query=self._query,
            max_age_sec=self._max_age_sec,
            now_ms=now_ms,
        )
        self._update_status(now_ms)
        labels = [self._row_label(r, now_ms) for r in self._rows]
        # STRUCTURE signature — what rows exist, in what order, and their
        # collapse/child shape. It excludes the row TEXT so a quiet-but-live
        # tree (age counter advancing, round badge ticking) patches its
        # existing rows in place instead of clearing + re-appending the whole
        # ListView every 1.5 s. That blanket rebuild was the /monitor flicker.
        struct_sig = tuple((r["key"], r["kind"], r["depth"], r["collapsed"], r["has_kids"]) for r in self._rows)

        # -- empty state --------------------------------------------------
        if not self._rows:
            if self._empty is not None:
                self._empty.display = True
                filtered = self._max_age_sec or self._query
                self._empty.update(
                    "No sessions match the current filters."
                    if filtered
                    else ("No sessions recorded yet." if self._include_all else "No agents are active right now.")
                )
            if struct_sig != self._struct_sig:
                self._listview.clear()
                self._row_statics = []
                self._listview.display = False
            self._struct_sig = struct_sig
            self._row_labels = labels
            return

        if self._empty is not None:
            self._empty.display = False
        self._listview.display = True

        # -- in-place patch: structure unchanged, only text may have ticked.
        # Update just the lines whose text actually changed — no clear(), so
        # the cursor/highlight never blinks and the pane doesn't flicker.
        if struct_sig == self._struct_sig and len(self._row_statics) == len(labels):
            for static, old, new in zip(self._row_statics, self._row_labels, labels):
                if old != new:
                    static.update(new)
            self._row_labels = labels
            return

        # -- structure changed → rebuild the list -------------------------
        prev_idx = self._listview.index or 0
        self._listview.clear()
        self._row_statics = []
        for label in labels:
            static = Static(label, markup=True)
            self._row_statics.append(static)
            self._listview.append(ListItem(static))
        self._struct_sig = struct_sig
        self._row_labels = labels
        self._listview.index = min(prev_idx, len(self._rows) - 1)

    def _update_status(self, now_ms: int) -> None:
        if self._status is None:
            return
        t = (self._last_tree or {}).get("totals") or {}
        agew = self.AGE_LABELS.get(self._max_age_sec, "any")
        scope = "all" if self._include_all else "live"
        q = f" · q={self._query!r}" if self._query else ""
        self._status.update(
            f"{t.get('agents', 0)} agents · {t.get('active_sessions', 0)} "
            f"active · {t.get('subagents', 0)} sub  ·  "
            f"scope={scope} · seen≤{agew}{q}"
        )

    # -- row formatting ---------------------------------------------------
    def _row_label(self, row: dict, now_ms: int) -> str:
        from rich.markup import escape as _esc

        from sharedagents_tui import monitor_view as _mv

        if row["kind"] == "agent":
            a = row["agent"]
            twisty = "▸" if row["collapsed"] else "▾"
            active = a.get("active_sessions") or 0
            sub = a.get("subagents_calling") or 0
            sub_part = f"  [yellow]· {sub} sub[/yellow]" if sub else f"  [dim]· {sub} sub[/dim]"
            return f"{twisty} [bold]{_esc(str(a.get('agent') or '?'))}[/bold]  [cyan]{active} active[/cyan]{sub_part}"
        n = row["node"]
        indent = "  " * row["depth"]
        twisty = ("▸" if row["collapsed"] else "▾") if row["has_kids"] else " "
        state = str(n.get("state") or "idle")
        stale = bool(n.get("stale"))
        live = bool(n.get("live"))
        # Status dot colour mirrors the web monitor's st-* classes.
        if stale:
            dot, state_m = "[yellow]◌[/yellow]", f"[yellow]stale · {state}[/yellow]"
        elif state == "error":
            dot, state_m = "[red]●[/red]", f"[red]{state}[/red]"
        elif live:
            dot, state_m = "[green]●[/green]", f"[green]{state}[/green]"
        else:
            dot, state_m = "[dim]○[/dim]", f"[dim]{state}[/dim]"
        hsh = (n.get("hash") or "")[:8]
        agent_tag = _esc(str(n.get("agent") or "?"))
        title = _esc(str(n.get("title") or "(untitled)"))
        if len(title) > 48:
            title = title[:47] + "…"
        parts = [f"{indent}{twisty} {dot} [dim]{hsh}[/dim] {state_m}"]
        if live and n.get("round") is not None:
            parts.append(f"[dim]r{n.get('round')}[/dim]")
        age = _mv.age_str(n.get("state_updated_at") or n.get("last_turn_ts"), now_ms)
        if age:
            parts.append(f"[dim]{age}[/dim]")
        sc = n.get("subagent_count") or 0
        if sc:
            parts.append(f"[yellow]{sc} sub[/yellow]")
        parts.append(f"[blue]{agent_tag}[/blue] {title}")
        return "  ".join(parts)

    # -- navigation / actions --------------------------------------------
    def _selected_row(self) -> dict | None:
        if not self._rows or self._listview is None:
            return None
        idx = self._listview.index
        if idx is None or not (0 <= idx < len(self._rows)):
            return None
        return self._rows[idx]

    def on_list_view_selected(self, event: Any) -> None:
        # Enter on a row OR a mouse click — both fire Selected.
        self._activate_current()

    def action_activate(self) -> None:
        # NB: named ``activate``, NOT ``select`` — a screen binding action
        # that shares ``ListView``'s built-in ``action_select`` name
        # re-enters during the same Enter dispatch and deadlocks.
        self._activate_current()

    def _activate_current(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row["kind"] == "agent":
            self._toggle_key(row["key"])
            return
        n = row["node"]
        # A session row → open it. Dismiss with enough context for the
        # /monitor handler to switch agent-then-session.
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:  # noqa: BLE001
                pass
        self.dismiss(
            {
                "hash": n.get("hash") or "",
                "agent": n.get("agent") or "",
                "title": n.get("title") or "",
            }
        )

    def action_toggle(self) -> None:
        row = self._selected_row()
        if row is not None:
            self._toggle_key(row["key"])

    def _toggle_key(self, key: str) -> None:
        if not key:
            return
        if key in self._collapsed:
            self._collapsed.discard(key)
        else:
            self._collapsed.add(key)
        self._save_state()
        self._struct_sig = None  # force a full rebuild on next render
        self._render_tree()

    def action_cursor_up(self) -> None:
        if self._listview is not None:
            self._listview.action_cursor_up()

    def action_cursor_down(self) -> None:
        if self._listview is not None:
            self._listview.action_cursor_down()

    def action_cycle_age(self) -> None:
        try:
            i = self.AGE_WINDOWS.index(self._max_age_sec)
        except ValueError:
            i = 0
        self._max_age_sec = self.AGE_WINDOWS[(i + 1) % len(self.AGE_WINDOWS)]
        self._save_state()
        self._struct_sig = None
        self._render_tree()

    def action_toggle_all(self) -> None:
        self._include_all = not self._include_all
        self._save_state()
        self._struct_sig = None
        # include_all changes the server scope → refetch.
        self._refresh()

    def action_focus_search(self) -> None:
        if self._search is not None:
            self._search.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is self._search:
            self._query = (event.value or "").strip().lower()
            self._save_state()
            self._struct_sig = None
            self._render_tree()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in the search box returns focus to the list for navigation.
        if event.input is self._search and self._listview is not None:
            self._listview.focus()

    def action_reload(self) -> None:
        self._refresh()

    def action_cancel(self) -> None:
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:  # noqa: BLE001
                pass
        self.dismiss(None)


class ModelPickerModal(ModalScreen[dict | None]):
    """Modal model picker invoked by `/model` or `/models` (no args).

    Shows each configured model as a `ListItem` with:

        ● 1.  gpu-a         qwen3.6-27b-fp8    ctx=256k  ·  http://gpu-a:8001/v1

    Up / Down navigate, Enter selects, Esc cancels. The currently
    active model is pre-highlighted with a `●` marker.

    Returns the selected model row dict on success, `None` on cancel.
    """

    CSS = """
    ModelPickerModal {
        align: center middle;
    }
    #picker-box {
        width: 90%;
        max-width: 150;
        height: 80%;
        max-height: 30;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #picker-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #picker-list {
        height: 1fr;
        border: round $accent 30%;
        background: $boost;
    }
    #picker-list > ListItem {
        padding: 0 1;
        height: 1;
    }
    #picker-list > ListItem.--highlight {
        background: $accent 30%;
    }
    #picker-list > ListItem.current {
        color: $success;
        text-style: bold;
    }
    #picker-help {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        # Esc is dismissed by ``MainApp.action_abort``'s modal-aware
        # branch — the app-level ``Binding("escape", "abort",
        # priority=True)`` preempts modal-screen bindings in
        # Textual 8.x, so the modal can't claim Esc via its own
        # binding. This entry stays here only so the footer hint
        # renders ``Esc Cancel``.
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "pick", "Select", show=True),
    ]

    def __init__(
        self,
        models: list[dict],
        current_url: str = "",
        *,
        current_alias: str = "",
        current_model_id: str = "",
    ) -> None:
        super().__init__()
        self._models = models
        self._current_url = (current_url or "").rstrip("/")
        # Alias-based "current" matching for engines whose rows have no
        # baseUrl (e.g. ``antigravity_cli`` — the engine owns its own HTTP
        # transport, so there's no URL to compare). Falls back to
        # URL-based matching when this is empty.
        self._current_alias = (current_alias or "").strip()
        # Wire model id of the live session. Needed to disambiguate
        # two rows that share an upstream URL but serve different
        # model ids (e.g. a vLLM endpoint hosting both ``qwen3.6-27b-fp8``
        # and ``Sehyo/Qwen3.5-122B-A10B-NVFP4`` — URL-only matching
        # marks BOTH rows current).
        self._current_model_id = (current_model_id or "").strip()

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-box"):
            yield Label(
                f"🧠  Models ({len(self._models)})  ·  ↑/↓ navigate  ·  Enter select  ·  Esc cancel",
                id="picker-title",
            )
            items: list[ListItem] = []
            initial = 0
            for i, m in enumerate(self._models):
                alias = m.get("alias") or "?"
                # Auto-discover via /v1/models was removed — empty
                # `id` is now a config error. Render "(missing id)"
                # in the picker so the operator notices and edits
                # ~/.sharedagents/config.json before trying to switch.
                raw_model_id = m.get("id") or ""
                model_id = raw_model_id or "(missing id)"
                model_col = "" if raw_model_id and raw_model_id == alias else model_id
                url = (m.get("baseUrl") or "").rstrip("/")
                ctx = m.get("contextWindow") or 0
                # Display label: the operator-facing ``provider`` name when
                # present, falling back to the wire ``backend`` (pre-OpenRouter
                # configs carry only ``backend``).
                provider = m.get("provider") or m.get("backend") or ""
                # Match either by URL (local-LLM, openrouter, …), by the
                # engine wire id (external rows have no upstream URL), or by
                # alias (when the engine exposes a separate display alias).
                # When two rows share a URL (one vLLM upstream serving
                # multiple model ids), require the wire ``id`` to also
                # match — otherwise the picker dots every row that
                # points at the same host.
                url_match = bool(url) and url == self._current_url
                if url_match and self._current_model_id:
                    url_match = raw_model_id == self._current_model_id
                # A model id alone identifies an external row because those
                # rows have no upstream URL. For local rows, only use an id
                # without a URL; when a URL is present ``url_match`` already
                # disambiguates same-id rows on different endpoints.
                id_match = (
                    not self._current_url and bool(self._current_model_id) and (raw_model_id == self._current_model_id)
                )
                is_current = id_match or url_match or (bool(self._current_alias) and alias == self._current_alias)
                marker = "●" if is_current else " "
                ctx_label = f"{ctx // 1024}k" if ctx else "?"
                provider_label = f" ({provider})" if provider else ""
                # antigravity-cli rows have no upstream URL; render the
                # human-readable description in its place so the
                # picker isn't all blank tails.
                tail = url or str(m.get("description") or "")
                row = f"{marker} {i + 1:>2}.  {alias:<20}{model_col:<34}ctx={ctx_label:<6}{provider_label}  ·  {tail}"
                item = ListItem(Static(row, markup=False))
                if is_current:
                    item.add_class("current")
                    initial = i
                items.append(item)
            self._listview = ListView(*items, id="picker-list", initial_index=initial)
            yield self._listview
            yield Label(
                "Tip: `/model <alias>` switches without the picker.",
                id="picker-help",
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = self._listview.index or 0
        if 0 <= idx < len(self._models):
            self.dismiss(dict(self._models[idx]))
        else:
            self.dismiss(None)

    def action_pick(self) -> None:
        idx = self._listview.index or 0
        if 0 <= idx < len(self._models):
            self.dismiss(dict(self._models[idx]))
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """Two-button confirm for irreversible slash commands.

    Web UI surfaces this as an in-chat confirm card. TUI had no
    analogue — ``/compact``, ``/restart``, ``/abort``, ``/task del``,
    and ``/monitors kill`` ran immediately. This modal is TUI-only
    and does not change the server's web-scoped re-challenge.
    """

    CSS = """
    ConfirmModal {
        align: center middle;
    }
    #confirm-box {
        width: 80%;
        max-width: 88;
        height: auto;
        border: thick $error;
        background: $panel;
        padding: 1 2;
    }
    #confirm-title {
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }
    #confirm-body {
        color: $text;
        margin-bottom: 1;
    }
    #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }
    #confirm-buttons Button { margin-left: 1; }
    """

    BINDINGS = [
        Binding("y", "answer('yes')", "Run", show=True),
        Binding("enter", "answer('yes')", "Run", show=False),
        Binding("n", "answer('no')", "Cancel", show=True),
        Binding("escape", "answer('no')", "Cancel", show=False),
    ]

    def __init__(self, *, title: str, prompt: str) -> None:
        super().__init__()
        self._title = title or "Confirm"
        self._prompt = prompt or "This cannot be undone."

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(self._title, id="confirm-title")
            yield Static(self._prompt, markup=False, id="confirm-body")
            with Horizontal(id="confirm-buttons"):
                yield Button("Run (y)", id="b-yes", variant="error")
                yield Button("Cancel (n / esc)", id="b-no", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "b-yes")

    def action_answer(self, response: str) -> None:
        self.dismiss(response == "yes")
