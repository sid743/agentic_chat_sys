"""Collects the agent trace for a run and streams readable lines to listeners
(LibreChat shows them in the collapsible "Thoughts" block)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

LEVELS = {"minimal": 0, "normal": 1, "debug": 2}


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={json.dumps(v, default=str, ensure_ascii=False)}" for k, v in (args or {}).items())


def summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return _short(result)
    if not result.get("ok", True):
        label = "denied" if result.get("error") == "forbidden" else result.get("error", "error")
        return f"{label}: {_short(result.get('message') or '', 140)}"
    try:
        if name == "get_my_profile":
            p = result["profile"]
            return f"{p['name']} ({p['employee_id']}), joined {p['date_of_joining']}, probation ends {p['probation_end_date']}"
        if name == "find_employee":
            return f"{result['count']} match(es): " + ", ".join(f"{m['name']} ({m['employee_id']})" for m in result["matches"])
        if name == "get_leave_balance":
            rows = [f"{r['leave_type']} {r['available_days']:g}" for r in result["balances"] if r.get("available_days") is not None]
            return f"{result['employee_name']}: available " + ", ".join(rows)
        if name == "check_leave_eligibility":
            verdict = "eligible" if result["eligible"] else "not eligible - " + "; ".join(result["blocking_reasons"])
            return f"{result['leave_type']}: {verdict}"
        if name == "calculate_leave_days":
            return f"{result['working_days']:g} working day(s) ({result['start_date']} to {result['end_date']})"
        if name == "list_holidays":
            return f"{len(result['holidays'])} holiday(s) for {result['location']}"
        if name in ("create_leave_request", "approve_leave_request", "reject_leave_request", "cancel_leave_request"):
            r = result["request"]
            return f"{r['request_id']} -> {r['status']}" + (f" (approver {r['current_approver_name']})" if r.get("current_approver_name") else "")
        if name == "list_leave_requests":
            return f"{result['count']} request(s)"
        if name == "send_notification":
            n = result["notification"]
            return f"sent {n['channel']} to {n['recipient_name']}: {_short(n['subject'], 80)}"
        if name == "list_notifications":
            return f"{len(result['notifications'])} notification(s)"
        if name in ("search_policies", "search_uploaded_documents"):
            return f"{len(result['results'])} passage(s)"
        if name == "list_uploaded_documents":
            return f"{len(result['documents'])} document(s)"
    except (KeyError, TypeError, ValueError):
        pass
    return _short({k: v for k, v in result.items() if k != "ok"})


class TraceRecorder:
    def __init__(self, level: str = "normal") -> None:
        self.level = LEVELS.get(level, 1)
        self.events: list[dict[str, Any]] = []
        self._listeners: list[Callable[[str], None]] = []
        self._t0 = time.monotonic()

    def subscribe(self, listener: Callable[[str], None]) -> None:
        self._listeners.append(listener)

    def emit(
        self,
        kind: str,
        text: str,
        *,
        agent: str | None = None,
        data: Any = None,
        level: str = "normal",
        record_only: bool = False,
    ) -> None:
        self.events.append(
            {
                "t_ms": int((time.monotonic() - self._t0) * 1000),
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": kind,
                "agent": agent,
                "text": text,
                "data": data,
            }
        )
        if not record_only and LEVELS.get(level, 1) <= self.level:
            for listener in self._listeners:
                listener(text)

    # convenience wrappers --------------------------------------------------------------
    def header(self, model_id: str, actor: dict, actor_source: str) -> None:
        self.emit(
            "start",
            f"Orchestrator | model {model_id} | acting as {actor['name']} "
            f"({actor['id']}, {actor['role']}; identity from {actor_source})\n",
            data={"model": model_id, "actor": actor, "actor_source": actor_source},
            level="minimal",
        )

    def uploads(self, docs: list[dict]) -> None:
        for d in docs:
            status = "indexed" if d.get("status") == "indexed" else "already indexed"
            chunks = f"{d['chunks']} chunk{'s' if d['chunks'] != 1 else ''}"
            self.emit("upload", f"- attachment {d['filename']} {status} ({chunks}) for on-demand retrieval\n", data=d)

    def plan(self, plan: dict, source: str) -> None:
        if plan.get("mode") == "direct":
            text = f"Plan ({source}): answer directly - {plan.get('reason', '')}\n"
        else:
            steps = " -> ".join(s["agent"] for s in plan.get("steps", []))
            text = f"Plan ({source}): {steps}" + (f" - {plan['reason']}" if plan.get("reason") else "") + "\n"
        self.emit("plan", text, data={"plan": plan, "source": source}, level="minimal")

    def agent_start(self, agent_id: str, name: str, task: str, model_id: str) -> None:
        self.emit("agent_start", f"\n[{name}] model {model_id}\n  task: {_short(task, 220)}\n", agent=agent_id,
                  data={"task": task, "model": model_id}, level="minimal")

    def tool_call(self, agent_id: str, name: str, args: dict, result: dict, ms: int) -> None:
        retrieval = name.startswith("search_") and result.get("ok", True)
        self.emit(
            "tool",
            f"  - tool {name}({_short(format_args(args), 120)}) -> {summarize_tool_result(name, result)} ({ms} ms)\n",
            agent=agent_id,
            data={"tool": name, "args": args, "ok": result.get("ok", True), "ms": ms, "result": result},
            level="debug" if retrieval else "normal",
        )

    def rag(self, agent_id: str, scope: str, query: str, citations: list[str]) -> None:
        cites = "; ".join(dict.fromkeys(citations)) or "no matching passages"
        self.emit("rag", f"  - retrieval ({scope}) \"{_short(query, 80)}\" -> {cites}\n", agent=agent_id,
                  data={"scope": scope, "query": query, "citations": citations})

    def flow(self, lines: list[str]) -> None:
        """The plain-text path through the graph, at the end of the trace."""
        if not lines:
            return
        # Recorded for the run log and the console, not streamed: the answer already
        # carries the diagram, and LibreChat wants every reasoning chunk before the
        # first content chunk.
        self.emit("flow", "\nGraph\n" + "\n".join(lines) + "\n", data={"lines": lines}, record_only=True)

    def note(self, text: str, agent: str | None = None, level: str = "normal") -> None:
        self.emit("note", f"{'  ' if agent else ''}- {text}\n", agent=agent, level=level)

    def agent_end(self, agent_id: str, name: str, ms: int, ok: bool) -> None:
        self.emit("agent_end", f"  - {'finished' if ok else 'failed'} in {ms / 1000:.1f} s\n", agent=agent_id,
                  data={"ms": ms, "ok": ok})

    def synth(self, count: int, model_id: str) -> None:
        self.emit("synthesize", f"\n[Synthesizer] model {model_id}: combining {count} agent results\n",
                  data={"count": count, "model": model_id}, level="minimal")

    def error(self, text: str, agent: str | None = None) -> None:
        self.emit("error", f"{'  ' if agent else ''}- error: {text}\n", agent=agent, level="minimal")

    def serializable(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self.events, default=str))
