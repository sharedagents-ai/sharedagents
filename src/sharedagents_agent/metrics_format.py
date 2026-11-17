"""Single source of truth for the per-turn / per-round telemetry "stats
row" rendered identically across the TUI, Telegram and Web frontends.

The TUI historically built this line inline in ``sharedagents_tui/app.py``
(``_build_metrics_markup`` / ``AssistantMessage.set_metrics``); Telegram
and Web showed nothing. To keep the three interfaces from drifting (the
sharedagents "identical rendering" property — every frontend is a passive
renderer of the SAME session log), the field selection + ordering +
number formatting lives here as plain text. Each frontend wraps the SAME
segments in its own emphasis markup:

* TUI       → ``[dim]─ {line}[/dim]`` (Rich dim markup)
* Telegram  → a plain ``─ {line}`` line appended to the message buffer
              (the buffer is sent without a parse mode, so no escaping)
* Web       → a ``.assist-metrics`` ``<div>`` (a JS port of the same
              field layout lives in ``web/static/app.js``; the
              ``metrics_format_segments`` golden test pins the two in
              lock-step).

Order (left → right):
    <model> · classify <ms> · plan <ms>
      · prefill <ms> @ <prefill_tps> tok/s | accept <ms>
      · ttft <ms> · <N> tools · ttfc <ms>
      · gen <gen_tps> tok/s
      · in <pt> (<cached> cached, <fresh> fresh)/<window>
      · out <ct> tok (think <rt> / result <ot>)
      · total <total> tok · wt <s>

Fields are skipped when their value is missing.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "model_label",
    "metrics_segments",
    "metrics_line",
    "complete_metrics",
]


def model_label(alias: object = "", model_id: object = "") -> str:
    """Render a compact user-facing model label without duplicated parts.

    ``<alias>/<model>`` when both are present and differ (so the operator
    sees BOTH the bucket they picked AND the wire-level id), else whichever
    single value is non-empty.
    """
    alias_s = str(alias or "")
    model_s = str(model_id or "")
    if alias_s and model_s and alias_s != model_s:
        return f"{alias_s}/{model_s}"
    return alias_s or model_s


def metrics_segments(metrics: dict[str, Any] | None) -> list[str]:
    """Return the ordered list of telemetry segments for ``metrics``.

    The plain-text building block shared by every frontend. Returns ``[]``
    when no renderable field is present (callers then render nothing —
    never an empty row).
    """
    if not metrics:
        return []
    m = metrics
    parts: list[str] = []

    model = m.get("model")
    model_alias = m.get("model_alias")
    classify_ms = m.get("classify_ms")
    plan_ms = m.get("plan_ms")
    ttft = m.get("ttft_ms")
    ttfc = m.get("ttfc_ms")
    prefill = m.get("prefill_ms")
    accept = m.get("accept_ms") or m.get("pft_ms")
    prefill_tps = m.get("prefill_tps")
    tps = m.get("tps")
    wt = m.get("wt_ms")
    ct = m.get("completion_tokens")
    rt = m.get("reasoning_tokens")
    ot = m.get("content_tokens")
    pt = m.get("prompt_tokens")
    ptc = m.get("prompt_tokens_cached")
    ptf = m.get("prompt_tokens_fresh")
    total = m.get("total_tokens")
    est = m.get("tokens_estimated")
    ctx_window = m.get("context_window")
    tool_events = m.get("tool_events")

    label = model_label(model_alias, model)
    if label:
        parts.append(label)
    if isinstance(classify_ms, (int, float)):
        parts.append(f"classify {classify_ms:.0f}ms")
    if isinstance(plan_ms, (int, float)):
        parts.append(f"plan {plan_ms:.0f}ms")
    if isinstance(prefill, (int, float)):
        seg = f"prefill {prefill:.0f}ms"
        if isinstance(prefill_tps, (int, float)):
            seg += f" @ {prefill_tps:.0f} tok/s"
        parts.append(seg)
    elif isinstance(accept, (int, float)):
        parts.append(f"accept {accept:.0f}ms")
    if isinstance(ttft, (int, float)):
        seg = f"ttft {ttft:.0f}ms"
        if isinstance(tool_events, (int, float)) and tool_events:
            seg += f" · {int(tool_events)} tool{'s' if tool_events != 1 else ''}"
        if isinstance(ttfc, (int, float)) and ttfc > ttft + 1:
            seg += f" · ttfc {ttfc:.0f}ms"
        parts.append(seg)
    if isinstance(tps, (int, float)):
        # Consolidated CLI engines expose authoritative output-token usage but
        # not token deltas, so their rate is whole-turn effective throughput,
        # not model decode throughput.
        throughput_label = "eff" if m.get("tps_kind") == "effective" else "gen"
        parts.append(f"{throughput_label} {tps:.1f} tok/s" + ("≈" if est else ""))
    if isinstance(pt, (int, float)):
        ctx_seg = f"in {int(pt)}"
        extras: list[str] = []
        if isinstance(ptc, (int, float)) and ptc:
            extras.append(f"{int(ptc)} cached")
        if isinstance(ptf, (int, float)) and (ptc or ptf != pt):
            extras.append(f"{int(ptf)} fresh")
        if extras:
            ctx_seg += f" ({', '.join(extras)})"
        if isinstance(ctx_window, (int, float)) and ctx_window:
            ctx_seg += f"/{int(ctx_window)}"
        parts.append(ctx_seg)
    if isinstance(ct, (int, float)):
        seg = f"out {int(ct)} tok" + ("≈" if est else "")
        if rt is not None or ot is not None:
            rt_disp = int(rt) if isinstance(rt, (int, float)) else 0
            ot_disp = int(ot) if isinstance(ot, (int, float)) else 0
            seg += f" (think {rt_disp} / result {ot_disp})"
        parts.append(seg)
    if isinstance(total, (int, float)) and total != ct:
        parts.append(f"total {int(total)} tok")
    if isinstance(wt, (int, float)):
        parts.append(f"wt {wt / 1000:.1f}s")
    # Cost segment — shown EVERY turn. ``cost_usd`` is this turn's spend (stamped
    # by the turn finalizer); ``goal_cost_usd`` / ``goal_max_cost_usd`` (or
    # ``session_cost_usd``) are the cumulative roll-up from cost_snapshot_metrics.
    cost_usd = m.get("cost_usd")
    if isinstance(cost_usd, (int, float)):
        seg = f"${cost_usd:.4f}" + ("≈" if est else "")
        goal_cost = m.get("goal_cost_usd")
        if isinstance(goal_cost, (int, float)):
            # Budget is never unbounded; a 0/missing value floors
            # to the default (the producer, cost_snapshot_metrics, already floors —
            # this is defensive so the footer never renders ∞).
            gmax = m.get("goal_max_cost_usd")
            if not isinstance(gmax, (int, float)) or gmax <= 0:
                from sharedagents_goal.goal_service import resolve_budget_usd

                gmax = resolve_budget_usd(None)
            budget = f"${gmax:.2f}"
            seg += f" · goal ${goal_cost:.2f}/{budget}"
        else:
            sess = m.get("session_cost_usd")
            if isinstance(sess, (int, float)):
                seg += f" · session ${sess:.2f}"
        parts.append(seg)
    return parts


def metrics_line(metrics: dict[str, Any] | None, *, prefix: str = "─ ", sep: str = " · ") -> str:
    """Plain-text one-line stats row, or ``""`` when nothing renders.

    ``prefix`` defaults to ``"─ "`` and ``sep`` to ``" · "`` — the exact
    glyphs the TUI footer uses, so Telegram/Web read identically.
    """
    parts = metrics_segments(metrics)
    if not parts:
        return ""
    return prefix + sep.join(parts)


def complete_metrics(
    raw: dict[str, Any] | None,
    *,
    prompt_chars: int | None = None,
    thinking_chars: int | None = None,
    content_chars: int | None = None,
    model: str = "",
    context_window: int | None = None,
    turn_s: float | None = None,
    ttft_s: float | None = None,
    gen_window_s: float | None = None,
) -> dict[str, Any]:
    """Return a metrics dict with EVERY stats-row field populated.

    The upstream half of producing the shared row: ``metrics_segments`` lays
    the fields out in the ``main`` (local-vLLM) order but simply skips any the
    engine never measured, so each engine's adapter historically rendered a
    different subset. This estimates the missing fields from char/timing
    signals so every agent renders the SAME line.

    Real values already in ``raw`` are NEVER overwritten — each engine passes
    whatever it actually measured and this fills the gaps. ``tokens_estimated``
    is set True whenever any field had to be synthesized, so the renderer keeps
    the ``≈`` honesty marker on estimated agents while ``main``'s real numbers
    pass through untouched.

    Estimation signals (all optional; a field is derived only when absent from
    ``raw``): char counts feed the token estimators; ``*_s`` are seconds.
    """
    from sharedagents_agent.context_report import (  # lazy: avoid import cycle
        estimate_tokens_for_model,
        estimate_tokens_from_chars,
    )

    out: dict[str, Any] = dict(raw or {})
    raw_had_est = "tokens_estimated" in (raw or {})
    synthesized = False

    def _num(key: str) -> bool:
        return isinstance(out.get(key), (int, float))

    mdl = str(out.get("model") or model or "")
    if mdl and not out.get("model"):
        out["model"] = mdl

    # ---- token counts ----
    if not _num("prompt_tokens") and prompt_chars is not None:
        out["prompt_tokens"] = estimate_tokens_from_chars(max(0, int(prompt_chars)))
        synthesized = True

    if not _num("completion_tokens"):
        cc = (content_chars or 0) + (thinking_chars or 0)
        if cc > 0:
            out["completion_tokens"] = estimate_tokens_for_model(int(cc), mdl)
            synthesized = True

    ct = out.get("completion_tokens")
    if isinstance(ct, (int, float)):
        # think / result split — always present so the (think/result) tail
        # renders for every agent (think 0 when the engine has no thoughts).
        if not _num("reasoning_tokens"):
            rt = estimate_tokens_for_model(int(thinking_chars), mdl) if thinking_chars else 0
            out["reasoning_tokens"] = int(rt)
            if thinking_chars:
                synthesized = True
        if not _num("content_tokens"):
            out["content_tokens"] = max(0, int(ct) - int(out.get("reasoning_tokens") or 0))
            synthesized = True

    if not _num("total_tokens"):
        pt, ctv = out.get("prompt_tokens"), out.get("completion_tokens")
        if isinstance(pt, (int, float)) and isinstance(ctv, (int, float)):
            out["total_tokens"] = int(pt) + int(ctv)
            synthesized = True

    # ---- timings ----
    if not _num("wt_ms") and turn_s is not None and turn_s >= 0:
        out["wt_ms"] = round(turn_s * 1000, 1)
        synthesized = True

    if not _num("ttft_ms") and ttft_s is not None and ttft_s >= 0:
        out["ttft_ms"] = round(ttft_s * 1000, 1)
        synthesized = True

    # prefill ≈ ttft (prefill dominates first-token latency; first decode is
    # negligible) — only synthesized, never overrides a real prefill.
    if not _num("prefill_ms") and _num("ttft_ms"):
        out["prefill_ms"] = out["ttft_ms"]
        synthesized = True

    if not _num("prefill_tps"):
        prefill_ms, pt = out.get("prefill_ms"), out.get("prompt_tokens")
        if isinstance(prefill_ms, (int, float)) and prefill_ms > 0 and isinstance(pt, (int, float)) and pt > 0:
            out["prefill_tps"] = round(int(pt) / (prefill_ms / 1000.0))
            synthesized = True

    # gen tok/s — prefer a real multi-delta window, then the post-ttft slice of
    # wall time, and finally (for a non-streaming engine that returns one blob)
    # whole-turn throughput. Guard a sane denominator (≥250 ms) so a tiny window
    # can't synthesize an absurd rate.
    if not _num("tps"):
        ctv = out.get("completion_tokens")
        if isinstance(ctv, (int, float)) and ctv > 0:
            denom = None
            if gen_window_s is not None and gen_window_s >= 0.25:
                denom = gen_window_s
            elif turn_s is not None and ttft_s is not None and (turn_s - ttft_s) >= 0.25:
                denom = turn_s - ttft_s
            elif turn_s is not None and turn_s >= 0.25:
                denom = turn_s
            if denom:
                out["tps"] = round(int(ctv) / denom, 1)
                synthesized = True

    if not _num("context_window") and context_window:
        out["context_window"] = int(context_window)

    if synthesized and not raw_had_est:
        out["tokens_estimated"] = True
    return out
