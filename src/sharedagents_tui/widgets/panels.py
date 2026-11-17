"""Panel/footer widgets (plan checklist, subagent box/list) — from app.py."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from sharedagents_tui.helpers import (
    _truncate_smart,
)


class PlanChecklistFooter(Vertical):
    """Persistent ``#plan-checklist`` footer wrapper.

    Renders the TodoWrite plan as a collapsible header + scrollable
    body, matching :class:`HarnessTaskListMessage` for symmetry —
    collapsed by default (``▶ 📐 plan · N items · click to expand``),
    expands on click / space / enter, body wrapped in a
    :class:`VerticalScroll` capped at ``int(app.size.height * 0.4)``.

    The inner Static keeps ``id="plan-checklist"`` so every existing
    ``query_one("#plan-checklist", Static)`` selector at
    :meth:`AgentChatApp._render_plan_checklist` continues to find it.
    The wrapper now ALSO redraws its own header (item count, glyph)
    on each plan-checklist update — the app calls
    :meth:`notify_plan_changed` after writing the inner Static's
    markup.

    Hidden when empty: a fresh session with no plan keeps the inner
    Static empty and shows the header as a zero-row line (``height:
    auto`` collapses to 0). The wrapper is always present in the
    DOM — only the content shrinks.

    Operator scroll bindings (active while the footer has focus):
    ``↑/↓`` + ``j/k`` line scroll, ``PgUp/PgDn`` page scroll,
    ``Home/End`` jump, ``space``/``enter`` toggle collapse. ``Tab``
    moves focus out so the chat input stays reachable.
    """

    BODY_HEIGHT_FRACTION = 0.4

    BINDINGS = [
        Binding("space", "toggle_collapsed", "Toggle", show=False),
        Binding("enter", "toggle_collapsed", "Toggle", show=False),
        Binding("up", "body_scroll('line_up')", "Up", show=False),
        Binding("down", "body_scroll('line_down')", "Down", show=False),
        Binding("k", "body_scroll('line_up')", "Up", show=False),
        Binding("j", "body_scroll('line_down')", "Down", show=False),
        Binding("pageup", "body_scroll('page_up')", "PgUp", show=False),
        Binding("pagedown", "body_scroll('page_down')", "PgDn", show=False),
        Binding("home", "body_scroll('home')", "Home", show=False),
        Binding("end", "body_scroll('end')", "End", show=False),
    ]

    can_focus = True

    def __init__(self) -> None:
        super().__init__(id="plan-checklist-wrap")
        # Collapsed-by-default per a long
        # plan (12 items, the historical hard cap) pushed the input
        # box near the fold. Operator expands explicitly when they
        # want to read the plan.
        self._collapsed: bool = True
        # Track item count so the header can show ``N items`` without
        # the wrapper needing to read into the app's ``_plans``
        # state — the app calls :meth:`notify_plan_changed` after
        # every render.
        self._item_count: int = 0
        self._all_done: bool = False
        self._header_static: Static | None = None
        self._body_static: Static | None = None
        self._body_scroll: VerticalScroll | None = None
        # Mirror the pre-PR-#349 invisible-when-empty UX of the bare
        # ``Static#plan-checklist`` it replaced: a fresh session with
        # no TodoWrite plan shouldn't show a stray header row above
        # the input box. ``notify_plan_changed`` flips this back on
        # when the model emits a plan.
        self.display = False

    # ------ header / body composition --------------------------------

    def _build_header(self) -> str:
        glyph = "▶" if self._collapsed else "▼"
        hint = " · [dim]click to expand[/dim]" if self._collapsed else ""
        done_suffix = " (done)" if (self._all_done and self._item_count > 0) else ""
        if self._item_count == 0:
            return f"[bold magenta]{glyph} 📐 plan[/bold magenta]{hint}"
        return (
            f"[bold magenta]{glyph} 📐 plan · "
            f"{self._item_count} item{'s' if self._item_count != 1 else ''}"
            f"{done_suffix}[/bold magenta]{hint}"
        )

    def compose(self) -> ComposeResult:
        header = Static(
            self._build_header(),
            markup=True,
            classes="plan-checklist-header",
        )
        self._header_static = header
        yield header
        body = Static("", id="plan-checklist", markup=True)
        self._body_static = body
        scroll = VerticalScroll(body, classes="plan-checklist-body-scroll")
        scroll.display = not self._collapsed
        self._body_scroll = scroll
        yield scroll

    # ------ lifecycle ------------------------------------------------

    def on_mount(self) -> None:
        self._apply_body_height_cap()

    def on_resize(self, event=None) -> None:
        self._apply_body_height_cap()

    def _apply_body_height_cap(self) -> None:
        if self._body_scroll is None:
            return
        try:
            screen_h = int(self.app.size.height)
        except Exception:  # noqa: BLE001
            return
        if screen_h <= 0:
            return
        cap = int(screen_h * self.BODY_HEIGHT_FRACTION)
        try:
            self._body_scroll.styles.max_height = cap
        except Exception:  # noqa: BLE001
            pass

    # ------ collapse / scroll ----------------------------------------

    def action_toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def action_body_scroll(self, direction: str) -> None:
        if self._body_scroll is None or self._collapsed:
            return
        try:
            fn = {
                "line_up": self._body_scroll.scroll_up,
                "line_down": self._body_scroll.scroll_down,
                "page_up": self._body_scroll.scroll_page_up,
                "page_down": self._body_scroll.scroll_page_down,
                "home": self._body_scroll.scroll_home,
                "end": self._body_scroll.scroll_end,
            }[direction]
        except KeyError:
            return
        try:
            fn(animate=False)
        except TypeError:
            fn()
        except Exception:  # noqa: BLE001
            pass

    def on_click(self, event) -> None:
        target = getattr(event, "widget", None)
        if target is self._header_static and self._header_static is not None:
            self.set_collapsed(not self._collapsed)

    def set_collapsed(self, value: bool) -> None:
        self._collapsed = bool(value)
        if self._body_scroll is not None:
            self._body_scroll.display = not self._collapsed
        if self._header_static is not None:
            self._header_static.update(self._build_header())

    # ------ app integration ------------------------------------------

    def notify_plan_changed(self, item_count: int, all_done: bool) -> None:
        """Called by :meth:`AgentChatApp._render_plan_checklist` after
        the inner Static is repainted. Updates the header counts +
        glyph WITHOUT changing the collapse state — operator's
        choice survives a TodoWrite re-emit. Toggles wrapper
        ``display`` to hide entirely when empty (no stray header
        row above the input on a fresh session) and reveal when the
        plan has items.
        """
        self._item_count = int(item_count)
        self._all_done = bool(all_done)
        if self._header_static is not None:
            self._header_static.update(self._build_header())
        # Hide-when-empty parity with HarnessTaskListMessage's
        # ``hide_when_empty`` flag — chat footer area collapses to
        # zero rows when there's no plan to show.
        self.display = self._item_count > 0


class SubagentBoxMessage(Vertical):
    """Bordered container that visually frames a single subagent run
    (``delegate_claude`` / ``delegate_codex`` / ``delegate_antigravity`` /
    ``delegate_agent``).

    Why: without this wrapper, the subagent's text widget + its inner
    tool widgets mount directly into the parent transcript, interleaving
    with the parent agent's own output. Operators reported "multiple
    agents' messages appear on the same TUI and it's confusing." The
    box gives each subagent run a colored left border + a header pill
    ``🤖 Subagent: <name> · <task summary>`` so the boundary is
    impossible to miss, and a body that hosts the subagent's
    AssistantMessage + every inner ToolCallMessage / ToolResultMessage
    fired during the run.

    Public surface:
      ``body`` — Vertical child where inner widgets should be mounted.
      ``finalize(ok=True)`` — flip the header from running → done/error.
    """

    def __init__(self, label: str, task_summary: str = "") -> None:
        super().__init__(classes="subagent-box")
        self._label = label or "subagent"
        self._task_summary = (task_summary or "").strip()
        self._header_static: Static | None = None
        self._body: Vertical | None = None
        self._done = False
        self._ok = True
        self._outcome: str | None = None
        # Per-run progress counters. Bumped from the parent TUI's
        # tool_call / tool_result dispatcher whenever an INNER event
        # arrives (a tool the subagent fired itself), with the actual
        # widget mount suppressed so the parent transcript stays
        # readable. The counters render in the header as
        # ``calls=N done=M`` so the operator sees the subagent is
        # alive even though its inner stream is hidden.
        self._inner_call_count: int = 0
        self._inner_done_count: int = 0
        self._inner_last_tool: str = ""
        self._t_start: float = time.monotonic()

    def _summary_line(self) -> str:
        if not self._task_summary:
            return ""
        return _truncate_smart(self._task_summary, 120)

    def _build_header(self) -> str:
        elapsed = max(0.0, time.monotonic() - self._t_start)
        progress = ""
        if self._inner_call_count > 0:
            progress = f" · {self._inner_done_count}/{self._inner_call_count} tools"
            if self._inner_last_tool:
                progress += f" · last: {self._inner_last_tool}"
        if self._done:
            tag = {"cancelled": "STOPPED", "refused": "REFUSED"}.get(self._outcome) or ("✅" if self._ok else "❌")
            colour = "white" if self._outcome in ("cancelled", "refused") else "green" if self._ok else "red"
            head = f"[bold {colour}]🤖 Subagent · {self._label} · {tag}{progress} · {elapsed:.0f}s[/bold {colour}]"
        else:
            head = f"[bold yellow]🤖 Subagent · {self._label} · running…{progress} · {elapsed:.0f}s[/bold yellow]"
        summary = self._summary_line()
        if summary:
            head += f"\n[dim]{summary}[/dim]"
        return head

    def note_inner_call(self, tool_name: str) -> None:
        """Bump the in-flight inner-tool counter. Called by the parent
        TUI's tool_call dispatcher whenever an INNER event arrives
        (a tool the subagent fired itself) while the parent has the
        widget mount suppressed — gives the operator a live ``N tools
        in flight · last: Bash`` indicator without flooding the
        transcript."""
        self._inner_call_count += 1
        self._inner_last_tool = (tool_name or "")[:24]
        if self._header_static is not None:
            self._header_static.update(self._build_header())

    def note_inner_done(self) -> None:
        """Bump the completed-inner-tool counter."""
        self._inner_done_count += 1
        if self._header_static is not None:
            self._header_static.update(self._build_header())

    def compose(self) -> ComposeResult:
        self._header_static = Static(self._build_header(), markup=True, classes="subagent-box-header")
        yield self._header_static
        self._body = Vertical(classes="subagent-box-body")
        yield self._body

    @property
    def body(self) -> Vertical:
        """Mount target for inner widgets (the subagent's
        AssistantMessage + every tool widget the subagent fires)."""
        # Falls back to ``self`` if Textual hasn't composed yet; the
        # caller's mount will still attach correctly under the box.
        return self._body if self._body is not None else self

    def finalize(self, ok: bool = True, outcome: str | None = None) -> None:
        self._done = True
        self._ok = bool(ok)
        self._outcome = outcome
        if self._header_static is not None:
            self._header_static.update(self._build_header())


class SubagentListPanel(Static):
    """Self-refreshing panel rendered by ``/subagents``.

    Why: the panel used to be a one-shot ``Static`` built from sidecar
    JSONs at command time. When ``/subagents`` was invoked mid-turn the
    row text froze at whatever status the sidecars held when the panel
    mounted (typically ``agent/running``); even after the subagent
    finalized and rewrote its sidecar with ``status=done`` + ``elapsed_s``,
    the rendered text never updated. The bottom-status counter (which
    subscribes to in-memory ``_subagent_done`` bumps) DID update, so
    operators saw a split-brain: ``1/2 done`` in the footer while the
    panel still showed both rows as running.

    Fix: keep a list of ``(kind, target, sidecar_path | None,
    static_ident)`` row specs. ``static_ident`` is the pre-built ident
    string for rows whose source isn't a ``.subagent.<call_id>.json``
    sidecar (delegate_{agent,claude,codex} sidecars store a stable
    child session id, not a live status). ``sidecar_path`` is set for
    the per-call subagent sidecars that DO carry live status. On a
    1-second timer we re-read every sidecar_path row, rebuild the
    body, and ``update()`` ourselves. The timer self-cancels once
    every sidecar-backed row reports a terminal status.
    """

    REFRESH_INTERVAL_S = 1.0

    def __init__(
        self,
        rows: list[tuple[str, str, Path | None, str]],
    ) -> None:
        super().__init__("", classes="claude-msg", markup=True)
        # Each entry: (kind, target, sidecar_path_or_None, static_ident).
        # If ``sidecar_path`` is set we re-read it each tick; otherwise
        # ``static_ident`` is the final rendered ident text.
        self._rows = list(rows)
        self._timer: Any = None
        self.update(self._build_body())

    def on_mount(self) -> None:
        # Only schedule a refresh tick if at least one row is sourced
        # from a per-call sidecar (the only kind whose status can
        # change over the panel's lifetime).
        if any(p is not None for _k, _t, p, _i in self._rows):
            self._timer = self.set_interval(self.REFRESH_INTERVAL_S, self._refresh)

    def on_unmount(self) -> None:
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:  # noqa: BLE001
                pass
            self._timer = None

    def _read_sidecar_ident(self, path: Path) -> tuple[str, bool]:
        """Return ``(ident_text, is_terminal)`` for a per-call subagent
        sidecar — same shape ``_cmd_subagents`` built originally so the
        rendered row is byte-identical to a fresh invocation."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tool = data.get("tool") or "agent"
            status = data.get("status") or "?"
            elapsed = data.get("elapsed_s")
            display_status = {"cancelled": "STOPPED", "refused": "REFUSED"}.get(status, status)
            tag = f"{tool}/{display_status}"
            if elapsed is not None:
                tag = f"{tag} {elapsed}s"
        except Exception:  # noqa: BLE001
            tag = "?"
            status = "?"
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
        except Exception:  # noqa: BLE001
            mtime = "??:??:??"
        ident = f"{tag}  ({mtime})"
        is_terminal = status in ("done", "error", "cancelled", "failed", "refused")
        return ident, is_terminal

    def _build_body(self) -> str:
        lines = [
            f"[bold]{len(self._rows)} subagent(s) for this session[/bold]  [dim]/subagents <N> to enter[/dim]\n",
        ]
        for n, (kind, target, sidecar_path, static_ident) in enumerate(self._rows, start=1):
            kind_disp = {
                "agent": "🦞 agent ",
                "claude": "🤖 claude",
                "codex": "🧪 codex ",
            }.get(kind, kind)
            if sidecar_path is not None:
                ident, _terminal = self._read_sidecar_ident(sidecar_path)
            else:
                ident = static_ident
            lines.append(f"  [bold]{n:>2}.[/bold]  {kind_disp}  [cyan]{target:<14}[/cyan]  [dim]{ident}[/dim]")
        return "\n".join(lines)

    def _refresh(self) -> None:
        self.update(self._build_body())
        # Auto-cancel once every sidecar-backed row is terminal — no
        # point polling files that won't change again.
        all_terminal = True
        for _k, _t, sidecar_path, _i in self._rows:
            if sidecar_path is None:
                continue
            _ident, is_terminal = self._read_sidecar_ident(sidecar_path)
            if not is_terminal:
                all_terminal = False
                break
        if all_terminal and self._timer is not None:
            try:
                self._timer.stop()
            except Exception:  # noqa: BLE001
                pass
            self._timer = None
