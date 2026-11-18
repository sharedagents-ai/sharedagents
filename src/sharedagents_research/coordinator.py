"""Awaited coordination of native research sessions; workers never delegate."""

from __future__ import annotations

import asyncio
from time import monotonic


async def research_many(agent, briefs, *, parallel=8, on_event=None, access=None, actors=None, parent_session_id=""):
    """Run independent native research sessions under the supplied agent's policy.

    No provider client or alternate model loop exists here. Cancellation awaits
    every worker; callers own authorization and the configured research budget.
    """
    from sharedagents_agent.agents.engine import TextDelta, TurnDone, TurnError

    if getattr(agent.cfg, "engine", "local_llm") != "local_llm":
        raise ValueError("ontology research requires the native local_llm engine")
    if (
        getattr(agent.cfg, "_agent_profile_name", "") or getattr(agent.cfg, "agent_profile", "") or agent.cfg.name
    ) != "research":
        raise ValueError("coordinator requires the native research profile")
    if not isinstance(parallel, int) or isinstance(parallel, bool) or not 1 <= parallel <= 32:
        raise ValueError("parallel must be an integer from 1 to 32")
    if isinstance(briefs, (str, bytes)):
        raise ValueError("briefs must be an iterable of research questions")
    pending = iter(enumerate(briefs))
    results = {}

    async def worker(brief):
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError("research questions must be nonempty strings")
        session = agent.create_session(parent_session_id=parent_session_id)
        started = monotonic()
        answer, errors, state = [], [], "incomplete"
        async for event in agent.run_turn(session.hash, brief, access=access, actors=actors):
            if isinstance(event, TextDelta) and event.channel == "content":
                answer.append(event.text)
            elif isinstance(event, TurnError):
                errors.append(event.message)
            elif isinstance(event, TurnDone):
                state = event.state
            if on_event:
                await on_event(session.hash, event)
        return {
            "session": session.hash,
            "state": state,
            "answer": "".join(answer),
            "errors": errors,
            "started": started,
            "finished": monotonic(),
        }

    async def consume():
        # Taking an item has no await: the event loop serializes iterator access.
        # Only the bounded pool exists, even for a large source iterator.
        for index, brief in pending:
            results[index] = await worker(brief)

    tasks = [asyncio.create_task(consume()) for _ in range(parallel)]
    try:
        await asyncio.gather(*tasks)
        return [results[index] for index in sorted(results)]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
