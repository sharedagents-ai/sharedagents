"""AgentChatApp plan-checklist + harness-task rendering.

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import re
from typing import Any

from textual.widgets import Static

from sharedagents_tui.widgets.messages import (
    HarnessTaskListMessage,
)
from sharedagents_tui.widgets.panels import (
    PlanChecklistFooter,
)


class TasksMixin:
    """AgentChatApp plan-checklist + harness-task rendering."""

    # ---- plan-checklist (footer-area widget) ---------------------------

    # Single plan per session. Claude Code's TodoWrite semantic is one
    # list whose statuses advance over time — re-emits replace the
    # current plan rather than stacking new ones. ``_plans`` stays a
    # list (length 0 or 1) for renderer continuity with prior code.

    @staticmethod
    def _plan_items(plan: dict) -> list[dict]:
        """Return the items list from a plan-group, materializing the
        legacy ``{steps:[str], cursor:int}`` shape into Claude TodoWrite
        ``{content, activeForm, status}`` dicts on the fly.

        Older plan-groups stored a flat ``steps`` list of strings plus a
        ``cursor`` int; new ones store ``items`` directly. Reading
        through this helper means the renderer + handlers don't need to
        branch on shape.
        """
        items = plan.get("items")
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
        steps = plan.get("steps") or []
        cursor = max(0, min(int(plan.get("cursor") or 0), len(steps)))
        out: list[dict] = []
        for i, step in enumerate(steps):
            out.append(
                {
                    "content": str(step).strip(),
                    "activeForm": "",
                    "status": "completed" if i < cursor else "pending",
                }
            )
        return out

    def _render_plan_checklist(self) -> None:
        """Repaint the persistent ``#plan-checklist`` widget.

        Body rows: ``☑`` completed (struck-through), ``▶`` in-progress
        (highlighted, uses ``activeForm`` so the row reads as a live
        verb-phrase), ``☐`` pending. Header (chevron + ``N items`` +
        ``(done)`` suffix) lives on the :class:`PlanChecklistFooter`
        wrapper — we notify it after writing the body so its header
        counts stay in sync without exposing ``_plans`` directly.
        """
        try:
            widget = self.query_one("#plan-checklist", Static)
        except Exception:  # noqa: BLE001
            return
        wrapper: PlanChecklistFooter | None = None
        try:
            wrapper = self.query_one("#plan-checklist-wrap", PlanChecklistFooter)
        except Exception:  # noqa: BLE001
            wrapper = None
        if not self._plans:
            widget.update("")
            if wrapper is not None:
                wrapper.notify_plan_changed(0, False)
            return
        rows: list[str] = []
        item_count = 0
        all_done = False
        # Single-plan model: render the latest plan only. ``_plans``
        # holds at most one entry but kept as a list for legacy
        # callers and replay scaffolding.
        for plan in self._plans[-1:]:
            items = self._plan_items(plan)
            item_count = len(items)
            all_done = bool(items) and all((it.get("status") in ("completed", "done")) for it in items)
            for it in items:
                status = it.get("status") or "pending"
                content = (it.get("content") or "").strip()
                active = (it.get("activeForm") or "").strip()
                label = active if (status == "in_progress" and active) else content
                if len(label) > 90:
                    label = label[:87] + "…"
                if status in ("completed", "done"):
                    rows.append(f"  [dim strike]☑ {label}[/dim strike]")
                elif status == "in_progress":
                    rows.append(f"  [bold cyan]▶ {label}[/bold cyan]")
                else:
                    rows.append(f"  [dim]☐[/dim] {label}")
        widget.update("\n".join(rows))
        if wrapper is not None:
            wrapper.notify_plan_changed(item_count, all_done)

    def set_plan_checklist(self, items: list) -> None:
        """Replace the session's plan with the given items and render.

        Accepts either:
        - ``list[str]`` — legacy entry point; each string becomes
          ``{content, activeForm: "", status: "pending"}``.
        - ``list[dict]`` — Claude TodoWrite shape, ``{content,
          activeForm, status}``.

        Mirrors Claude Code's TodoWrite: a session has exactly ONE
        plan and re-emits replace its contents (status updates flow
        through ``_handle_todo_tool_call``'s in-place merge so a
        progress nudge doesn't blow away the existing list). Single-
        item "plans" are dropped — the bridge already filters at emit
        time, the TUI guards too.

        No item count cap — the persistent footer is bounded by the
        :class:`PlanChecklistFooter` wrapper's 40%-of-screen-height
        cap + VerticalScroll, so a long plan scrolls inside rather
        than getting silently truncated. Operator can collapse the
        footer to reclaim screen real estate (collapsed by default).
        """
        norm: list[dict] = []
        for it in items:
            if isinstance(it, str):
                content = it.strip()
                if not content:
                    continue
                norm.append(
                    {
                        "content": content,
                        "activeForm": "",
                        "status": "pending",
                    }
                )
            elif isinstance(it, dict):
                content = (it.get("content") or "").strip()
                if not content:
                    continue
                status = (it.get("status") or "pending").strip().lower()
                if status not in ("pending", "in_progress", "completed", "done"):
                    status = "pending"
                norm.append(
                    {
                        "content": content,
                        "activeForm": (it.get("activeForm") or "").strip(),
                        "status": status,
                    }
                )
        if len(norm) < 2:
            return
        self._plans = [{"items": norm}]
        self._render_plan_checklist()

    def advance_plan_step(self) -> None:
        """Mark the next non-completed step on the LATEST plan-group as
        completed. No-op when the latest plan is already fully done.

        Used by the legacy ``todo complete`` action path; the new
        TodoWrite tool emits the full list with explicit per-item
        statuses, so the renderer reads those directly.
        """
        if not self._plans:
            return
        latest = self._plans[-1]
        items = self._plan_items(latest)
        if not items:
            return
        for it in items:
            if it.get("status") not in ("completed", "done"):
                it["status"] = "completed"
                latest["items"] = items
                self._render_plan_checklist()
                return

    def clear_plan_checklist(self) -> None:
        """Drop ALL plan-groups. Called on /new so a new session
        starts with a clean footer."""
        self._plans = []
        self._render_plan_checklist()

    @staticmethod
    def _normalize_todo_arg_item(it: Any) -> dict | None:
        """Coerce one engine todo item into ``{content, activeForm,
        status}``. Returns ``None`` on items with no content."""
        from sharedagents_agent.tools.widget_adapt import adapt_todo_item

        item = adapt_todo_item(it)
        if item is None:
            return None
        return {
            "content": item["content"],
            "activeForm": item.get("activeForm") or "",
            "status": item.get("status") or "pending",
        }

    def _upsert_harness_task(
        self,
        tid: str,
        *,
        subject: str | None = None,
        status: str | None = None,
    ) -> None:
        """Insert / merge a row in the harness-task footer state, then
        repaint. Status only advances forward (pending → in_progress →
        completed) so a stale TaskUpdate can't visually regress a row;
        ``deleted`` overrides everything.
        """
        tid = str(tid or "").strip()
        if not tid:
            return
        row = self._harness_tasks.get(tid)
        if row is None:
            row = {"subject": "", "status": "pending"}
            self._harness_tasks[tid] = row
            self._harness_tasks_order.append(tid)
        if subject is not None:
            sb = str(subject).strip()
            if sb:
                row["subject"] = sb
        if status is not None:
            ns = str(status).strip().lower()
            if ns == "deleted":
                row["status"] = "deleted"
            else:
                rank = {
                    "pending": 0,
                    "in_progress": 1,
                    "completed": 2,
                    "done": 2,
                }
                old_rank = rank.get(row.get("status") or "pending", 0)
                new_rank = rank.get(ns, 0)
                if new_rank >= old_rank:
                    row["status"] = ns
        self._render_harness_tasks()

    def _apply_task_update_args(self, args: dict) -> None:
        """Apply a ``TaskUpdate`` argument bundle to the harness-task
        footer (optimistic update at tool_call time)."""
        if not isinstance(args, dict):
            return
        tid = str(args.get("taskId") or "").strip()
        if not tid:
            return
        self._upsert_harness_task(
            tid,
            subject=args.get("subject"),
            status=args.get("status"),
        )

    def _apply_task_create_from_result(self, args: dict, result_text: str) -> None:
        """Parse the TaskCreate result body to recover the assigned id
        (``"Task #N created successfully: …"``), then upsert into the
        harness-task footer AND the inline ``HarnessTaskListMessage``
        card (mounted lazily on the first parent Task* call)."""
        if not isinstance(args, dict):
            return
        m = re.search(r"Task\s+#(\d+)\s+created", result_text or "")
        if not m:
            return
        tid = m.group(1)
        self._upsert_harness_task(
            tid,
            subject=str(args.get("subject") or ""),
            status="pending",
        )
        # Mirror into the inline transcript card so the operator can
        # scroll back and see the full history. Skip when no card was
        # mounted (e.g. subagent dispatch took the path that bypasses
        # the parent's transcript).
        if self._latest_harness_task_widget is not None:
            try:
                self._latest_harness_task_widget.upsert(
                    tid,
                    subject=str(args.get("subject") or ""),
                    status="pending",
                )
            except Exception:  # noqa: BLE001
                pass

    def _render_subagent_counter(self) -> None:
        """Repaint the persistent ``#subagent-counter`` footer Static.
        Hidden when no subagents have run; otherwise shows ``🤖
        Subagents · N/M done [· K failed] [· in flight]`` so the
        operator sees overall fan-out progress even after individual
        boxes scroll out of the viewport.
        """
        try:
            widget = self.query_one("#subagent-counter", Static)
        except Exception:  # noqa: BLE001
            return
        if self._subagent_total == 0:
            widget.update("")
            return
        in_flight = self._subagent_total - self._subagent_done - self._subagent_failed
        line = f"[bold magenta]🤖 Subagents · {self._subagent_done}/{self._subagent_total} done"
        if self._subagent_failed:
            line += f" · {self._subagent_failed} failed"
        if in_flight > 0:
            line += f" · {in_flight} in flight"
        line += "[/bold magenta]"
        # Navigation hint pinned beneath the counter so the operator
        # knows the SubagentBoxMessages in the transcript are
        # interactive — without it, the boxes look like passive
        # text. The hint disappears once all subagents have
        # completed (no point pointing at scrolled-off boxes).
        if in_flight > 0 or self._subagent_total <= 4:
            line += "  [dim]alt+↑/alt+↓ to step · ctrl+o to open · /subagents N[/dim]"
        widget.update(line)

    def _clear_harness_tasks(self) -> None:
        """Drop every row from the harness-task footer and repaint —
        called on /new, on session reset after a failed reattach, and
        whenever a fresh session-id is created (the prior session's
        tasks no longer apply). Also resets the subagent counter
        (same lifecycle)."""
        self._harness_tasks.clear()
        self._harness_tasks_order.clear()
        self._pending_task_create_calls.clear()
        self._harness_task_call_ids.clear()
        self._render_harness_tasks()
        self._subagent_total = 0
        self._subagent_done = 0
        self._subagent_failed = 0
        self._render_subagent_counter()

    def _render_harness_tasks(self) -> None:
        """Repaint the persistent ``#harness-tasks`` footer widget.

        Shares the inline ``HarnessTaskListMessage`` class so the
        persistent footer gets the same collapse-by-default + 40%-
        height-cap + scrollable-body UX (the previous flat
        ``Static`` grew unbounded). The widget hides
        itself when the task list is empty.
        """
        try:
            widget = self.query_one(
                "#harness-tasks",
                HarnessTaskListMessage,
            )
        except Exception:  # noqa: BLE001
            return
        widget.replace_rows(
            self._harness_tasks,
            self._harness_tasks_order,
        )

    def _is_harness_task_tool(self, name: str) -> bool:
        return name in (
            "TaskCreate",
            "TaskUpdate",
            "TaskList",
            "TaskGet",
            "TaskOutput",
            "TaskStop",
        )

    def _handle_todo_tool_call(self, args: dict) -> None:
        """Update the plan-checklist in response to a model-emitted
        ``TodoWrite`` tool call.

        Two arg shapes are accepted:

        1. **Claude TodoWrite** (current): ``todos: [{content,
           activeForm, status}]`` — the whole list with per-item
           status. ``status ∈ {pending, in_progress, completed}`` and
           ``activeForm`` is the present-continuous verb-phrase shown
           in the renderer while a step is running.
        2. **Legacy sharedagents todo**: ``action`` ∈ {replace, add,
           complete, delete, list} with ``items`` / ``indexes``. Kept
           for older callers.

        Best-effort: malformed args never tank the prompt loop; the
        runner returns its own error to the model and the TUI keeps the
        previous plan-group state.
        """
        try:
            # raw_input may arrive as a JSON string, not a dict.
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return
                if not isinstance(args, dict):
                    return
            # Cursor Update TODOs / OpenCode items → canonical todos.
            # Keep a legacy ``action`` payload on the original keys.
            if "action" not in args:
                from sharedagents_agent.tools.widget_adapt import adapt_todo_args

                adapted = adapt_todo_args(args)
                if adapted:
                    args = adapted
            todos_arg = args.get("todos")
            if isinstance(todos_arg, list) and "action" not in args:
                action = "replace"
                items_in: list[Any] = list(todos_arg)
                indexes: list[Any] = []
            else:
                action = str(args.get("action") or "").strip().lower()
                items_in = list(args.get("items") or [])
                indexes = list(args.get("indexes") or [])

            if action == "replace":
                norm = [n for n in (self._normalize_todo_arg_item(it) for it in items_in) if n is not None]
                if not norm:
                    return
                # When the model re-emits a `replace` whose content list
                # matches the latest in-flight plan-group, merge in
                # place — update statuses + activeForm rather than
                # appending a duplicate group. Without this the TUI
                # would show a second copy of the plan every round the
                # model nudges the list (e.g. flipping one item from
                # pending → in_progress).
                # Single-plan semantics: if the existing plan's content
                # set overlaps the new emit (ignoring order — Gemma 4
                # likes to reshuffle completed rows to the top each
                # round), merge in place and only forward-advance
                # statuses (pending → in_progress → completed) so a
                # stale re-emit can't visually regress a row. Otherwise
                # the new list replaces the plan outright.
                merged = False
                if self._plans and norm:
                    latest_items = self._plan_items(self._plans[-1])
                    old_set = {it.get("content") for it in latest_items}
                    new_set = {n["content"] for n in norm}
                    overlap = old_set & new_set
                    # "Same plan" if the new list is the same set of
                    # items OR a subset/superset that significantly
                    # overlaps — covers the "model dropped one item"
                    # and "model added one item" mid-plan tweaks.
                    if (
                        latest_items
                        and overlap
                        and (old_set == new_set or len(overlap) >= max(len(old_set), len(new_set)) - 1)
                    ):
                        rank = {
                            "pending": 0,
                            "in_progress": 1,
                            "completed": 2,
                            "done": 2,
                        }
                        old_by_content = {it.get("content"): it for it in latest_items}
                        merged_items: list[dict] = []
                        for n in norm:
                            old = old_by_content.get(n["content"], {})
                            old_rank = rank.get(old.get("status") or "pending", 0)
                            new_rank = rank.get(n.get("status") or "pending", 0)
                            chosen = n if new_rank >= old_rank else old
                            merged_items.append(
                                {
                                    "content": n["content"],
                                    "activeForm": n.get("activeForm") or old.get("activeForm") or "",
                                    "status": chosen.get("status") or "pending",
                                }
                            )
                        self._plans[-1] = {"items": merged_items}
                        self._render_plan_checklist()
                        merged = True
                if not merged:
                    self.set_plan_checklist(norm)
            elif action == "add":
                extra = [n for n in (self._normalize_todo_arg_item(it) for it in items_in) if n is not None]
                if not extra:
                    return
                if not self._plans:
                    # `add` with no active plan implicitly creates one.
                    # Mirrors the runner's behaviour.
                    self.set_plan_checklist(extra)
                    return
                latest = self._plans[-1]
                cur = self._plan_items(latest)
                room = max(0, 12 - len(cur))
                cur.extend(extra[:room])
                latest["items"] = cur
                latest.pop("steps", None)
                latest.pop("cursor", None)
                self._render_plan_checklist()
            elif action == "complete":
                if not self._plans:
                    return
                latest = self._plans[-1]
                cur = self._plan_items(latest)
                if not cur:
                    return
                idx_set = {int(i) for i in indexes if isinstance(i, (int, float)) and not isinstance(i, bool)}
                for i, it in enumerate(cur):
                    if i in idx_set and it.get("status") not in ("completed", "done"):
                        it["status"] = "completed"
                latest["items"] = cur
                latest.pop("steps", None)
                latest.pop("cursor", None)
                self._render_plan_checklist()
            elif action == "delete":
                if not self._plans:
                    return
                latest = self._plans[-1]
                cur = self._plan_items(latest)
                if not cur:
                    return
                drop = {int(i) for i in indexes if isinstance(i, (int, float)) and not isinstance(i, bool)}
                kept = [it for i, it in enumerate(cur) if i not in drop]
                latest["items"] = kept
                latest.pop("steps", None)
                latest.pop("cursor", None)
                self._render_plan_checklist()
            # action=list is read-only — don't touch state.
        except Exception:  # noqa: BLE001
            # Best-effort UI sync; never let a malformed args dict
            # take down the prompt loop.
            pass
