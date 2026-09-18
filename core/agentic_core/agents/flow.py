"""A plain-text picture of the LangGraph path a turn took.

Shown under every answer (and in the trace) so it is obvious which agent the
router picked, what ran after it, and who produced the reply. Deliberately
plain: it renders the same in a chat bubble, a terminal and a log file.

    START -> route(llm) -> eligibility_agent -> policy_agent* -> synthesize -> END
      router   llm plan: needs the balance and the carry-forward rule
      agents   Eligibility Agent (2 tools, 1.4s)
               Leave Policy Agent (1 tool, 0.9s)
      answer   synthesize merged 2 agent answers

`*` marks an agent the orchestrator added by itself (the notification hand-off).
"""

from __future__ import annotations

from typing import Any

ARROW = " -> "
MAX_REASON = 90


def _seconds(ms: int) -> str:
    return f"{ms / 1000:.1f}s"


def _tools(count: int) -> str:
    return "no tools" if count == 0 else ("1 tool" if count == 1 else f"{count} tools")


def _label(key: str, value: str) -> str:
    return f"  {key:<8} {value}"


def flow_lines(
    plan: dict[str, Any],
    results: list[Any],
    *,
    merged: bool,
    error: str | None = None,
) -> list[str]:
    """Build the diagram for one turn. `results` are AgentResult objects, in order."""
    source = plan.get("source") or "rules"
    mode = plan.get("mode") or "agents"
    steps = plan.get("steps") or []
    auto_agents = {s.get("agent") for s in steps if s.get("auto")}
    by_id = {r.agent_id: r for r in results}

    nodes = ["START", f"route({source})"]
    lines: list[str] = []

    if error:
        nodes.append("stop")
        nodes.append("END")
        return [ARROW.join(nodes), _label("router", f"failed: {error[:MAX_REASON]}")]

    if mode == "direct":
        nodes += ["direct", "END"]
        lines.append(ARROW.join(nodes))
        lines.append(_label("router", f"{source} plan: answered without agents"))
        lines.append(_label("answer", "direct (no agent, no tools)"))
        return lines

    ran = [r.agent_id for r in results]
    ordered = ran or [s.get("agent", "?") for s in steps]
    nodes += [f"{a}*" if a in auto_agents else a for a in ordered]
    nodes += ["synthesize", "END"]
    lines.append(ARROW.join(nodes))

    reason = (plan.get("reason") or "").strip().replace("\n", " ")
    if reason:
        lines.append(_label("router", f"{source} plan: {reason[:MAX_REASON]}"))
    else:
        lines.append(_label("router", f"{source} plan: {', '.join(ordered) or 'no agents'}"))

    detail = []
    for agent_id in ordered:
        result = by_id.get(agent_id)
        if result is None:
            detail.append(f"{agent_id} (not run)")
            continue
        bits = [_tools(len(result.tools)), _seconds(result.ms)]
        if agent_id in auto_agents:
            bits.append("auto")
        if not result.ok:
            bits.append("failed")
        detail.append(f"{result.agent_name} ({', '.join(bits)})")
    for i, item in enumerate(detail):
        # one agent per line: a chain wraps badly in a narrow chat bubble
        lines.append(_label("agents", item) if i == 0 else _label("", item))

    main = [r for r in results if r.agent_id not in auto_agents]
    if merged and len(main) > 1:
        lines.append(_label("answer", f"synthesize merged {len(results)} agent answers"))
    elif main:
        lines.append(_label("answer", f"{main[-1].agent_name}, passed through synthesize unchanged"))
    else:
        lines.append(_label("answer", "no agent produced an answer"))
    return lines


def flow_block(
    plan: dict[str, Any],
    results: list[Any],
    *,
    merged: bool,
    error: str | None = None,
) -> str:
    """The same diagram wrapped in a code fence, for the chat bubble."""
    return "```text\n" + "\n".join(flow_lines(plan, results, merged=merged, error=error)) + "\n```"
