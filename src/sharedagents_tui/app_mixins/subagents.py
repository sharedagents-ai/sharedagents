"""AgentChatApp subagent sidecar persistence + delegate display.

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Static

from sharedagents_tui.helpers import (
    _reconcile_subagent_sidecars,
)
from sharedagents_tui.widgets.messages import (
    UserMessage,
)
from sharedagents_tui.widgets.panels import (
    SubagentBoxMessage,
    SubagentListPanel,
)


class SubagentMixin:
    """AgentChatApp subagent sidecar persistence + delegate display."""

    def _subagent_sidecar_path(self, call_id: str) -> Any | None:
        """Path for a per-subagent run capture file.

        Pattern: ``<sessions_dir>/<sid>.subagent.<call_id>.json``.
        Holds the full prompt + result so the operator can re-read a
        subagent's run after it finishes (the cloud ``Agent`` tool
        returns only one final result text; without this sidecar the
        intermediate "what did I ask the subagent" is lost the moment
        it scrolls off the transcript). Independent from the
        ``<sid>.delegate_*.json`` sidecars sharedagents's bridges write,
        which are keyed by subagent kind rather than call id.
        """
        sid = getattr(self, "_sid", None)
        if not sid or not call_id:
            return None
        try:
            from pathlib import Path as _Path

            sdir = _Path(self.agent_cfg.sessions_dir)
        except Exception:  # noqa: BLE001
            return None
        return sdir / f"{sid}.subagent.{call_id}.json"

    def _persist_subagent_call(
        self,
        call_id: str,
        tool_name: str,
        args: Any,
    ) -> None:
        """Write the start-of-run record for a subagent call. Called
        from the tool_call dispatch — captures args + a timestamp so
        the result-write later can compute elapsed time. Best-effort;
        any I/O failure is swallowed so it never breaks the live
        stream."""
        path = self._subagent_sidecar_path(call_id)
        if path is None:
            return
        try:
            import json as _json
            import time as _time

            payload = {
                "call_id": call_id,
                "tool": tool_name,
                "args": args if isinstance(args, dict) else {},
                "status": "running",
                "ts_start": _time.time(),
                "ts_start_iso": _time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    _time.localtime(),
                ),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(
                _json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:  # noqa: BLE001
            pass

    def _persist_subagent_result(
        self,
        call_id: str,
        ok: bool,
        content: str,
        outcome: str | None = None,
    ) -> None:
        """Append the result record to the subagent sidecar. Reads the
        existing start-of-run payload, merges in the result + elapsed
        time, and rewrites the file atomically."""
        path = self._subagent_sidecar_path(call_id)
        if path is None or not path.exists():
            return
        try:
            import json as _json
            import time as _time

            payload = _json.loads(path.read_text(encoding="utf-8"))
            ts_end = _time.time()
            payload["ts_end"] = ts_end
            payload["ts_end_iso"] = _time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                _time.localtime(ts_end),
            )
            ts_start = payload.get("ts_start") or ts_end
            payload["elapsed_s"] = round(float(ts_end) - float(ts_start), 3)
            payload["ok"] = bool(ok)
            payload["status"] = outcome or ("done" if ok else "error")
            payload["result"] = content or ""
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(
                _json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:  # noqa: BLE001
            pass

    def _reconcile_orphaned_subagents(self, reason: str = "aborted") -> None:
        """Flip this session's still-``"running"`` subagent sidecars to a
        terminal state. Guards against the status leak where a subagent's
        sidecar never receives its result-write because the matching
        ``tool_result`` wasn't processed by the live box-tracking path
        (turn aborted, or session reattached from another process). Run at
        end-of-turn and on session attach. Best-effort."""
        sid = getattr(self, "_sid", None)
        if not sid:
            return
        try:
            sessions_dir = self.agent_cfg.sessions_dir
        except Exception:  # noqa: BLE001
            return
        reconciled = _reconcile_subagent_sidecars(sessions_dir, sid, reason)
        if reconciled:
            self._tui_log(
                "subagent_reconciled",
                reason=reason,
                count=len(reconciled),
            )

    def _is_delegate_tool(self, name: str) -> bool:
        # ``Agent`` is Claude Code's native subagent primitive (returns
        # a single final result text rather than streaming inner
        # events) — it still benefits from the bordered-box treatment
        # so a parent can visually distinguish "this came from a
        # subagent run" from its own parallel Bash/Read/Edit calls,
        # which was the original complaint.
        return name in (
            "Agent",
            "AgentSync",
            "AgentAsync",
            "delegate_claude",
            "delegate_codex",
            "delegate_antigravity",
            "delegate_agent",
        )

    def _mount_target(self, transcript: Any) -> Any:
        """If a subagent box is on the stack, route inner mounts there
        so the subagent's events nest visually under its container.
        Otherwise mount into the parent transcript."""
        if self._subagent_box_stack:
            return self._subagent_box_stack[-1].body
        return transcript

    def _delegate_summary(self, name: str, args: dict) -> tuple[str, str]:
        """Pretty label + short task summary for a delegate_* call.
        ``label`` reads in the subagent-box header (e.g.
        ``claude · qwen-coder``), ``summary`` is the truncated prompt."""
        label = ""
        if name == "delegate_claude":
            label = "claude"
        elif name == "delegate_codex":
            label = "codex"
        elif name == "delegate_antigravity":
            label = "antigravity"
        elif name == "delegate_agent":
            target = ""
            if isinstance(args, dict):
                target = str(args.get("name") or args.get("agent") or "")
            label = f"agent · {target}" if target else "agent"
        elif name in ("Agent", "AgentSync", "AgentAsync"):
            # Claude Code's native Agent tool: subagent_type is the
            # canonical "what kind of subagent" label.
            stype = ""
            if isinstance(args, dict):
                stype = str(args.get("subagent_type") or "")
            label = f"agent · {stype}" if stype else "agent"
        else:
            label = name
        summary = ""
        if isinstance(args, dict):
            for k in ("task", "prompt", "description"):
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    summary = v.strip().splitlines()[0]
                    break
        return label, summary

    def action_subagent_focus_prev(self) -> None:
        """``alt+up`` — step the subagent-box focus cursor up by one,
        scroll the focused box into view, and stamp a CSS class on it
        so the operator can see which one is selected. Wraps to the
        last box when cursor is on the first.
        """
        self._step_subagent_focus(-1)

    def action_subagent_focus_next(self) -> None:
        """``alt+down`` — symmetric to :meth:`action_subagent_focus_prev`,
        stepping forward through the subagent boxes."""
        self._step_subagent_focus(+1)

    def _step_subagent_focus(self, delta: int) -> None:
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:  # noqa: BLE001
            return
        boxes = list(transcript.query(SubagentBoxMessage))
        if not boxes:
            self._set_status("no subagents to navigate")
            return
        cur = getattr(self, "_subagent_focus_idx", -1)
        new_idx = (0 if delta > 0 else len(boxes) - 1) if cur < 0 or cur >= len(boxes) else (cur + delta) % len(boxes)
        # Clear the old highlight before stamping the new one so two
        # boxes don't appear focused after a step.
        for b in boxes:
            try:
                b.remove_class("subagent-box-focused")
            except Exception:  # noqa: BLE001
                pass
        target = boxes[new_idx]
        try:
            target.add_class("subagent-box-focused")
        except Exception:  # noqa: BLE001
            pass
        # Scroll the focused box into view; ``animate=False`` so the
        # cursor feels snappy when the operator alt+down-mashes through
        # a long subagent list.
        try:
            transcript.scroll_to_widget(target, animate=False)
        except Exception:  # noqa: BLE001
            pass
        self._subagent_focus_idx = new_idx
        self._set_status(f"subagent {new_idx + 1}/{len(boxes)} focused (ctrl+o to open)")

    def action_subagent_open(self) -> None:
        """``ctrl+o`` (or ``alt+enter`` where the terminal delivers it)
        — pop a card showing the focused subagent's sidecar JSON
        (prompt + tool args + result + elapsed). When no subagent is
        focused, prints a hint to the status row instead of opening a
        blank card."""
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:  # noqa: BLE001
            return
        boxes = list(transcript.query(SubagentBoxMessage))
        if not boxes:
            self._set_status("no subagents to open")
            return
        idx = getattr(self, "_subagent_focus_idx", -1)
        if idx < 0 or idx >= len(boxes):
            self._set_status("no subagent focused — alt+up / alt+down first")
            return
        target_box = boxes[idx]
        call_id = next(
            (cid for cid, b in self._subagent_boxes_by_call.items() if b is target_box),
            None,
        )
        # Even after the run finalizes the box stays mounted but the
        # call_id mapping is popped. Fall back to a sequential scan of
        # the sessions dir for the Nth subagent sidecar.
        sidecar_path = None
        if call_id:
            sidecar_path = self._subagent_sidecar_path(call_id)
        if sidecar_path is None or not sidecar_path.exists():
            try:
                from pathlib import Path as _Path

                sdir = _Path(self.agent_cfg.sessions_dir)
                sid = getattr(self, "_sid", None) or ""
                candidates = sorted(
                    sdir.glob(f"{sid}.subagent.*.json"),
                    key=lambda p: p.stat().st_mtime,
                )
                if 0 <= idx < len(candidates):
                    sidecar_path = candidates[idx]
            except Exception:  # noqa: BLE001
                sidecar_path = None
        if sidecar_path is None or not sidecar_path.exists():
            self._set_status(f"subagent {idx + 1}: no sidecar file yet (it may still be running)")
            return
        # Run the inline /subagents-style viewer as a worker so the
        # mount + status updates happen on the right loop.
        self.run_worker(
            self._render_subagent_sidecar(sidecar_path, idx + 1),
            exclusive=False,
            name="subagent-open",
        )

    async def _render_subagent_sidecar(
        self,
        path: Any,
        n: int,
    ) -> None:
        """Mount a Static card showing the focused subagent's sidecar
        contents (description + prompt + result + elapsed). Lives in
        the same transcript as the prompts so it scrolls naturally;
        no modal, no extra dependencies.
        """
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:  # noqa: BLE001
            return
        try:
            import json as _json

            data = _json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            await transcript.mount(
                Static(
                    f"error reading subagent sidecar: {e}",
                    classes="claude-msg",
                )
            )
            return
        from rich.markup import escape as _esc

        args = data.get("args") or {}
        desc = str(args.get("description") or args.get("task") or args.get("name") or "(no description)")
        prompt = str(args.get("prompt") or "")
        tool = data.get("tool") or "agent"
        status = data.get("status") or "?"
        elapsed = data.get("elapsed_s")
        result = data.get("result") or ""
        head = (
            f"[bold cyan]🤖 Subagent {n} · {tool} · {status}"
            f"{f' · {elapsed}s' if elapsed is not None else ''}[/bold cyan]\n"
            f"[bold]{_esc(desc)}[/bold]\n"
        )
        body_parts: list[str] = [head]
        if prompt:
            body_parts.append(f"\n[dim]── prompt ─────────────[/dim]\n{_esc(prompt[:4000])}")
            if len(prompt) > 4000:
                body_parts.append(f"\n[dim](… +{len(prompt) - 4000} chars truncated)[/dim]")
        if result:
            body_parts.append(f"\n\n[dim]── result ─────────────[/dim]\n{_esc(result[:4000])}")
            if len(result) > 4000:
                body_parts.append(f"\n[dim](… +{len(result) - 4000} chars truncated)[/dim]")
        body_parts.append(f"\n\n[dim]sidecar: {_esc(str(path))}[/dim]")
        await transcript.mount(
            Static(
                "".join(body_parts),
                classes="claude-msg",
                markup=True,
            )
        )
        try:
            transcript.scroll_end(animate=False)
        except Exception:  # noqa: BLE001
            pass

    async def _cmd_subagents(self, arg: str = "") -> None:
        """``/subagents [N]`` — list subagent sessions this sharedagents
        session has spawned, or enter the Nth one.

        sharedagents delegates work via three sibling tools — ``delegate_agent``
        (sharedagents-on-sharedagents), ``delegate_claude`` (Claude Code bridge),
        and ``delegate_codex`` (OpenAI Codex bridge). Each persists the
        child session id to a sidecar at
        ``<sessions_dir>/<parent_sid>.delegate_<kind>[.<target>].json``
        so the next call from the same parent resumes the same child.
        ``/subagents`` reads those sidecars and shows what's there.

        Without an argument: print a numbered list. With a number:
        switch the TUI's view to that subagent's session — for
        ``delegate_agent`` we hop into the target sharedagents agent (same
        path ``/agent <name>`` takes), for the external bridges we
        print the bridge's session id and last-modified time so the
        operator can re-attach via the bridge's own CLI.
        """
        from pathlib import Path as _Path

        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /subagents" + (f" {arg}" if arg else "")))
        sid = getattr(self, "_sid", None)
        if not sid:
            await transcript.mount(
                Static(
                    "no active session — start one before listing subagents",
                    classes="claude-msg",
                )
            )
            return
        sessions_dir = _Path(self.agent_cfg.sessions_dir)
        rows: list[tuple[str, str, str, _Path]] = []
        # (kind, target_or_id, mtime_str, sidecar_path) — `target_or_id`
        # is the named target for delegate_agent, "claude" / "codex"
        # otherwise.
        for entry in sorted(sessions_dir.glob(f"{sid}.delegate_agent.*.json")):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
                cid = data.get("session_id") or "—"
            except Exception:  # noqa: BLE001
                cid = "—"
            target = entry.stem.split(".delegate_agent.", 1)[-1]
            mtime = datetime.fromtimestamp(entry.stat().st_mtime).strftime("%H:%M:%S")
            rows.append(
                (
                    "agent",
                    target,
                    f"{cid}  ({mtime})",
                    entry,
                )
            )
        for entry in sorted(sessions_dir.glob(f"{sid}.delegate_claude.json")):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
                cid = data.get("claude_session_id") or "—"
            except Exception:  # noqa: BLE001
                cid = "—"
            mtime = datetime.fromtimestamp(entry.stat().st_mtime).strftime("%H:%M:%S")
            rows.append(("claude", "claude-code", f"{cid}  ({mtime})", entry))
        for entry in sorted(sessions_dir.glob(f"{sid}.delegate_codex.json")):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
                cid = data.get("thread_id") or data.get("session_id") or "—"
            except Exception:  # noqa: BLE001
                cid = "—"
            mtime = datetime.fromtimestamp(entry.stat().st_mtime).strftime("%H:%M:%S")
            rows.append(("codex", "openai-codex", f"{cid}  ({mtime})", entry))
        # Per-call subagent sidecars written by the TUI for every
        # ``Agent`` / ``delegate_*`` tool call (captures prompt + result
        # + elapsed time so the run is verifiable later).
        for entry in sorted(sessions_dir.glob(f"{sid}.subagent.*.json")):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
                tool = data.get("tool") or "agent"
                status = data.get("status") or "?"
                args = data.get("args") or {}
                desc = args.get("description") or args.get("task") or args.get("name") or ""
                if isinstance(desc, str) and len(desc) > 60:
                    desc = desc[:57] + "…"
                elapsed = data.get("elapsed_s")
                tag = f"{tool}/{status}"
                if elapsed is not None:
                    tag = f"{tag} {elapsed}s"
            except Exception:  # noqa: BLE001
                tool = "agent"
                tag = "?"
                desc = ""
            mtime = datetime.fromtimestamp(entry.stat().st_mtime).strftime("%H:%M:%S")
            label = desc or entry.stem
            rows.append(
                (
                    "subagent",
                    label,
                    f"{tag}  ({mtime})",
                    entry,
                )
            )
        if not rows:
            await transcript.mount(
                Static(
                    "no subagents have been spawned from this session yet.\n"
                    "[dim]sharedagents spawns subagents via the canonical "
                    "[bold]Agent[/bold] tool — call "
                    '[bold]Agent(subagent_type="general-purpose"|"claude"|"codex"|"antigravity", '
                    "description=…, prompt=…)[/bold] in a prompt to populate "
                    "this list. Operator-side, the [bold]/claude[/bold] / "
                    "[bold]/codex[/bold] / [bold]/antigravity[/bold] slash commands "
                    "also spawn entries here (session-continuation "
                    "primitives — see CLAUDE.md).[/dim]",
                    classes="claude-msg",
                    markup=True,
                )
            )
            return
        # If `arg` is a 1-based index, hop to that subagent.
        if arg:
            try:
                idx = int(arg)
            except ValueError:
                await transcript.mount(
                    Static(
                        f"usage: /subagents [N]  — N is 1..{len(rows)}",
                        classes="claude-msg",
                    )
                )
                return
            if not (1 <= idx <= len(rows)):
                await transcript.mount(
                    Static(
                        f"index {idx} out of range (have {len(rows)} subagent(s))",
                        classes="claude-msg",
                    )
                )
                return
            kind, target, _ident, _path = rows[idx - 1]
            if kind == "agent":
                # Same path /agent uses — switch to the target sharedagents agent.
                await transcript.mount(
                    Static(
                        f"entering subagent → switching to agent [bold]{target}[/bold]",
                        classes="claude-msg",
                        markup=True,
                    )
                )
                run = getattr(self, "_cmd_agent", None)
                if run is not None:
                    await run(target)
                return
            await transcript.mount(
                Static(
                    f"[bold]{kind}[/bold] subagents are external bridges — reattach via their own CLI ({_ident}).",
                    classes="claude-msg",
                    markup=True,
                )
            )
            return
        # Render the list as a self-refreshing panel. Per-call subagent
        # sidecars (``kind == "subagent"``) carry live status the panel
        # re-reads on a 1s tick; delegate_{agent,claude,codex} sidecars
        # store a stable child session id so their ident is frozen at
        # mount time (passed through ``static_ident``, sidecar_path=None).
        panel_rows: list[tuple[str, str, _Path | None, str]] = []
        for kind, target, ident, path in rows:
            if kind == "subagent":
                panel_rows.append((kind, target, path, ident))
            else:
                panel_rows.append((kind, target, None, ident))
        await transcript.mount(SubagentListPanel(panel_rows))
