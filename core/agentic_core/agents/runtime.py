"""Specialist agent loop: model <-> tools until a final answer.

Supports native OpenAI tool calling and, for models without it, a JSON-prompted
tool protocol (selected automatically when a provider rejects `tools`).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from ..llm.base import CallOptions, ChatModel, LLMError, ToolsNotSupported, Usage, parse_json_object
from ..tools import REGISTRY, execute_tool, schemas_for
from .config import AgentSpec
from .context import RequestContext, ToolContext

log = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 6000


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render(template: str, values: dict[str, Any]) -> str:
    try:
        return template.format_map(SafeDict(values))
    except (ValueError, IndexError):
        return template


def prompt_values(request: RequestContext) -> dict[str, Any]:
    actor = request.actor
    uploads = ", ".join(d["filename"] for d in request.uploads) or "none"
    return {
        "today": request.today.isoformat(),
        "actor_name": actor.name,
        "actor_id": actor.id,
        "actor_role": actor.role,
        "actor_title": actor.title,
        "actor_department": actor.department,
        "actor_location": actor.location_code,
        "manager_name": actor.manager_name or "none",
        "uploads": uploads,
    }


@dataclass
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ms: int

    def brief(self) -> dict[str, Any]:
        return {"tool": self.name, "arguments": self.arguments, "ok": self.result.get("ok", True), "ms": self.ms}


@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    task: str
    answer: str
    model: str
    tools: list[ToolRecord] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    ms: int = 0
    usage: Usage = field(default_factory=Usage)

    def summary(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task": self.task,
            "answer": self.answer,
            "ok": self.ok,
            "error": self.error,
            "sources": self.sources,
            "tools": [t.brief() for t in self.tools],
            "ms": self.ms,
        }


def _tool_json(result: dict[str, Any]) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + '..."} (truncated)'
    return text


def _collect(record: ToolRecord, result: AgentResult) -> None:
    data = record.result
    for hit in data.get("results") or []:
        if isinstance(hit, dict) and hit.get("citation"):
            result.sources.append(hit["citation"])
    for ref in data.get("policy_refs") or []:
        result.sources.append(ref)
    if data.get("policy_ref"):
        result.sources.append(data["policy_ref"])
    for event in data.get("events") or []:
        result.events.append(event)


def _history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no earlier messages)"
    lines = []
    for msg in history:
        who = "User" if msg["role"] == "user" else "Assistant"
        content = re.sub(r"\s+", " ", msg["content"]).strip()
        lines.append(f"{who}: {content[:600]}")
    return "\n".join(lines)


def build_user_prompt(task: str, user_message: str, history: list[dict[str, str]], prior: list[AgentResult]) -> str:
    parts = ["Conversation so far:", _history_block(history), ""]
    if prior:
        parts.append("Results from agents that already worked on this message:")
        for r in prior:
            parts.append(f"[{r.agent_name}] {r.answer[:1500]}")
            for event in r.events:
                parts.append(f"  event: {event.get('summary')}")
        parts.append("")
    parts.append(f"Latest user message: {user_message}")
    parts.append(f"Your task from the orchestrator: {task}")
    return "\n".join(parts)


class AgentRunner:
    def __init__(self, spec: AgentSpec, model: ChatModel, request: RequestContext, common_rules: str, max_steps: int) -> None:
        self.spec = spec
        self.model = model
        self.request = request
        self.common_rules = common_rules
        self.max_steps = spec.max_steps or max_steps

    def _system_prompt(self, json_mode: bool) -> str:
        values = prompt_values(self.request)
        prompt = render(self.spec.system_prompt, values) + "\n\n" + render(self.common_rules, values)
        if json_mode:
            tools = "\n".join(f"- {REGISTRY[t].prompt_signature()}" for t in self.spec.tools)
            prompt += (
                "\n\nYou can use these tools:\n" + tools + "\n\n"
                'To call a tool, reply with ONLY a JSON object: {"tool": "<name>", "arguments": {...}}\n'
                'When you have what you need, reply with ONLY: {"final": "<your answer in markdown>"}'
            )
        return prompt

    def _options(self, prior: list[AgentResult], task: str, user_message: str, usage: Usage) -> CallOptions:
        return CallOptions(
            purpose="agent",
            agent_id=self.spec.id,
            temperature=self.spec.temperature,
            usage_sink=usage,
            context={
                "task": task,
                "user_message": user_message,
                "today": self.request.today.isoformat(),
                "actor": self.request.actor.as_dict(),
                "has_uploads": bool(self.request.uploads),
                "events": [e for r in prior for e in r.events],
            },
        )

    async def run(self, task: str, user_message: str, history: list[dict[str, str]], prior: list[AgentResult]) -> AgentResult:
        started = time.monotonic()
        trace = self.request.trace
        result = AgentResult(self.spec.id, self.spec.name, task, "", self.model.id)
        trace.agent_start(self.spec.id, self.spec.name, task, self.model.id)
        user_prompt = build_user_prompt(task, user_message, history, prior)
        options = self._options(prior, task, user_message, result.usage)
        try:
            if self.model.native_tools:
                try:
                    result.answer = await self._run_native(user_prompt, options, result)
                except ToolsNotSupported as exc:
                    trace.note(f"{self.model.id}: {exc}; switching to JSON tool protocol", agent=self.spec.id)
                    self.model.native_tools = False
                    result.answer = await self._run_json(user_prompt, options, result)
            else:
                result.answer = await self._run_json(user_prompt, options, result)
        except LLMError as exc:
            result.ok = False
            result.error = str(exc)
            result.answer = f"The {self.spec.name} could not complete its task: {exc}"
            trace.error(str(exc), agent=self.spec.id)
        result.sources = list(dict.fromkeys(s for s in result.sources if s))
        result.ms = int((time.monotonic() - started) * 1000)
        trace.agent_end(self.spec.id, self.spec.name, result.ms, result.ok)
        return result

    async def _execute(self, name: str, arguments: dict[str, Any], result: AgentResult, seen: dict[str, dict]) -> dict:
        key = name + json.dumps(arguments, sort_keys=True, default=str)
        if key in seen:
            return {**seen[key], "note": "Duplicate call - this is the same result as before; do not call it again."}
        if name not in self.spec.tools:
            output = {"ok": False, "error": "tool_not_allowed", "message": f"{name} is not available to this agent."}
        else:
            t0 = time.monotonic()
            output = await execute_tool(name, ToolContext(self.request, self.spec.id), arguments)
            ms = int((time.monotonic() - t0) * 1000)
            record = ToolRecord(name, arguments, output, ms)
            result.tools.append(record)
            _collect(record, result)
            self.request.trace.tool_call(self.spec.id, name, arguments, output, ms)
        seen[key] = output
        return output

    async def _run_native(self, user_prompt: str, options: CallOptions, result: AgentResult) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(json_mode=False)},
            {"role": "user", "content": user_prompt},
        ]
        tools = schemas_for(self.spec.tools)
        seen: dict[str, dict] = {}
        duplicates = 0
        for _ in range(self.max_steps):
            reply = await self.model.complete(messages, tools, options)
            if not reply.tool_calls:
                if reply.content.strip():
                    return reply.content.strip()
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                before = len(seen)
                output = await self._execute(call.name, call.arguments, result, seen)
                duplicates += int(len(seen) == before)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _tool_json(output)})
            if duplicates >= 2:
                break
        # Step budget exhausted (or empty reply): ask for a final answer without tools.
        messages.append({"role": "user", "content": "Stop calling tools and give your final answer to the user now."})
        final = await self.model.complete(messages, None, options)
        return final.content.strip() or "I couldn't finish this task."

    async def _run_json(self, user_prompt: str, options: CallOptions, result: AgentResult) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(json_mode=True)},
            {"role": "user", "content": user_prompt},
        ]
        seen: dict[str, dict] = {}
        options = replace(options, json_mode=True)  # ask the provider for a JSON object when it can
        for _ in range(self.max_steps):
            reply = await self.model.complete(messages, None, options)
            data = parse_json_object(reply.content)
            if not data:
                return reply.content.strip() or "I couldn't finish this task."
            if "final" in data:
                return str(data["final"]).strip()
            name = data.get("tool") or data.get("name")
            if not name:
                return reply.content.strip()
            arguments = data.get("arguments") or data.get("args") or {}
            output = await self._execute(str(name), arguments if isinstance(arguments, dict) else {}, result, seen)
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": f"Tool result for {name}:\n{_tool_json(output)}"})
        messages.append({"role": "user", "content": 'Reply now with {"final": "<answer>"} only.'})
        reply = await self.model.complete(messages, None, options)
        data = parse_json_object(reply.content)
        return str(data.get("final")) if data and data.get("final") else reply.content.strip()
