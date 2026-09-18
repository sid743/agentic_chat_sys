"""LangGraph orchestrator: route -> specialist agents (sequential hand-offs) -> synthesize."""

from __future__ import annotations

import asyncio
import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..llm.base import CallOptions, ChatModel, LLMError, Usage
from ..llm.registry import ModelRegistry
from ..settings import Settings
from .config import AgentsConfig
from .context import RequestContext
from .flow import flow_block, flow_lines
from .router import heuristic_route, llm_route
from .runtime import AgentResult, AgentRunner, prompt_values, render

log = logging.getLogger(__name__)

TokenSink = Callable[[str], None]


class TurnState(TypedDict, total=False):
    user_message: str
    history: list[dict[str, str]]
    plan: dict[str, Any]
    step: int
    results: Annotated[list[AgentResult], operator.add]
    answer: str
    error: str


@dataclass
class TurnRuntime:
    request: RequestContext
    model: ChatModel
    on_token: TokenSink
    usage: Usage = field(default_factory=Usage)
    emitted: list[str] = field(default_factory=list)

    def emit(self, text: str) -> None:
        if text:
            self.emitted.append(text)
            self.on_token(text)


@dataclass
class TurnOutcome:
    answer: str
    plan: dict[str, Any]
    results: list[AgentResult]
    usage: Usage
    model_id: str
    status: str = "ok"
    error: str | None = None


def _rt(config: RunnableConfig) -> TurnRuntime:
    return config["configurable"]["rt"]


def notification_task(events: list[dict[str, Any]]) -> str:
    lines = ["Send these workflow notifications (one send_notification call per line):"]
    for event in events:
        for n in event.get("notify", []):
            lines.append(
                f'- recipient="{n["employee_id"]}" ({n["name"]}, {n["role"]}) channel="{n.get("channel", "email")}" '
                f'related_request_id="{event["request_id"]}" subject="{n["subject"]}" message="{n["message"]}"'
            )
    return "\n".join(lines)


class Orchestrator:
    def __init__(self, settings: Settings, registry: ModelRegistry, config: AgentsConfig) -> None:
        self.settings = settings
        self.registry = registry
        self.config = config
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        graph = StateGraph(TurnState)
        graph.add_node("route", self._route)
        graph.add_node("direct", self._direct)
        graph.add_node("run_agent", self._run_agent)
        graph.add_node("synthesize", self._synthesize)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route", self._after_route, {"direct": "direct", "agents": "run_agent", "stop": END}
        )
        graph.add_conditional_edges("run_agent", self._after_agent, {"next": "run_agent", "done": "synthesize"})
        graph.add_edge("direct", END)
        graph.add_edge("synthesize", END)
        return graph.compile()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    async def _model_for(self, override: str | None, rt: TurnRuntime) -> ChatModel:
        if override:
            try:
                return await self.registry.resolve(override)
            except LLMError as exc:
                rt.request.trace.note(f"model override '{override}' unavailable ({exc}); using {rt.model.id}")
        return rt.model

    # ------------------------------------------------------------------ nodes
    async def _route(self, state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        rt = _rt(config)
        req = rt.request
        orch = self.config.orchestrator
        agent_ids = list(self.config.agents)
        max_agents = min(orch.max_agents_per_turn, self.settings.max_route_agents)
        values = {**prompt_values(req), "agent_catalog": self.config.catalog(), "max_agents": max_agents}
        router_model = await self._model_for(self.settings.agent_router_model or orch.model, rt)
        try:
            plan = await llm_route(
                router_model,
                render(orch.router_prompt, values),
                state.get("history", [])[-6:],
                state["user_message"],
                agent_ids=agent_ids,
                max_agents=max_agents,
                has_uploads=bool(req.uploads),
                context={"actor": req.actor.as_dict(), "today": req.today.isoformat()},
                usage_sink=rt.usage,
            )
        except LLMError as exc:
            # A rate-limited or unreachable router is not worth failing the turn for:
            # the built-in rules route well enough, and the agents may use another model.
            req.trace.note(f"router model unavailable ({exc}); using the built-in routing rules")
            plan = heuristic_route(state["user_message"], has_uploads=bool(req.uploads), agent_ids=agent_ids)
        # Documents were attached but the planner ignored them: make sure they are used.
        if plan.mode == "agents" and req.new_uploads and all(s.agent != "document_agent" for s in plan.steps):
            if "document_agent" in self.config.agents:
                fallback = heuristic_route(state["user_message"], has_uploads=True, agent_ids=agent_ids)
                if any(s.agent == "document_agent" for s in fallback.steps):
                    plan.steps.insert(0, next(s for s in fallback.steps if s.agent == "document_agent"))
        data = plan.as_dict()
        req.trace.plan(data, plan.source)
        return {"plan": data, "step": 0}

    def _after_route(self, state: TurnState) -> str:
        mode = state["plan"].get("mode")
        if mode == "stop":
            return "stop"
        return "direct" if mode == "direct" else "agents"

    async def _direct(self, state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        rt = _rt(config)
        answer = state["plan"].get("answer") or ""
        if answer:
            await self._stream_text(rt, answer)
            return {"answer": answer}
        orch = self.config.orchestrator
        messages = [
            {"role": "system", "content": render(orch.direct_prompt, prompt_values(rt.request))},
            *state.get("history", [])[-4:],
            {"role": "user", "content": state["user_message"]},
        ]
        options = CallOptions(purpose="direct", usage_sink=rt.usage, context={"actor": rt.request.actor.as_dict()})
        parts = []
        async for delta in rt.model.stream(messages, options):
            parts.append(delta)
            rt.emit(delta)
        return {"answer": "".join(parts)}

    async def _run_agent(self, state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        rt = _rt(config)
        plan = dict(state["plan"])
        steps = [dict(s) for s in plan.get("steps", [])]
        index = state.get("step", 0)
        step = steps[index]
        prior: list[AgentResult] = list(state.get("results", []))
        spec = self.config.agents[step["agent"]]

        pending_events = [e for r in prior for e in r.events]
        task = step["task"]
        if spec.id == self.config.orchestrator.auto_notify_agent and pending_events and "send_notification" not in task:
            task = f"{task}\n\n{notification_task(pending_events)}"

        model = await self._model_for(spec.model, rt)
        runner = AgentRunner(spec, model, rt.request, self.config.common_rules, self.settings.max_agent_steps)
        result = await runner.run(task, state["user_message"], state.get("history", []), prior)
        rt.usage.add(result.usage)

        # Automatic hand-off: workflow changes are always followed by notifications.
        notify_id = self.config.orchestrator.auto_notify_agent
        if result.events and notify_id and notify_id != spec.id:
            remaining = [s["agent"] for s in steps[index + 1 :]]
            if notify_id not in remaining:
                steps.append({"agent": notify_id, "task": notification_task(result.events), "auto": True})
                rt.request.trace.note(f"hand-off: {spec.id} -> {notify_id} ({len(result.events)} workflow event(s))")
        plan["steps"] = steps
        return {"results": [result], "step": index + 1, "plan": plan}

    def _after_agent(self, state: TurnState) -> str:
        return "next" if state.get("step", 0) < len(state["plan"].get("steps", [])) else "done"

    async def _synthesize(self, state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        rt = _rt(config)
        results: list[AgentResult] = state.get("results", [])
        steps = state["plan"].get("steps", [])
        auto_agents = {s["agent"] for s in steps if s.get("auto")}
        main = [r for r in results if r.agent_id not in auto_agents]
        extra = [r for r in results if r.agent_id in auto_agents]
        mode = self.config.orchestrator.synthesize

        if mode == "never" or (mode == "auto" and len(main) <= 1):
            text = "\n\n".join(r.answer for r in main if r.answer) or "I couldn't complete that request."
            if extra:
                text += "\n\n" + "\n".join(r.answer for r in extra if r.answer)
            await self._stream_text(rt, text)
            return {"answer": text}

        orch = self.config.orchestrator
        synth_model = await self._model_for(orch.model, rt)
        rt.request.trace.synth(len(results), synth_model.id)
        findings = "\n\n".join(
            f"[{r.agent_name}]{' (failed)' if not r.ok else ''}\n{r.answer}" for r in results if r.answer
        )
        messages = [
            {"role": "system", "content": render(orch.synthesis_prompt, prompt_values(rt.request))},
            {"role": "user", "content": f"User message: {state['user_message']}\n\nFindings:\n\n{findings}"},
        ]
        options = CallOptions(
            purpose="synthesize",
            usage_sink=rt.usage,
            context={"results": [r.summary() for r in results]},
        )
        parts: list[str] = []
        try:
            async for delta in synth_model.stream(messages, options):
                parts.append(delta)
                rt.emit(delta)
        except LLMError as exc:
            rt.request.trace.error(f"synthesis failed: {exc}")
            fallback = "\n\n".join(f"**{r.agent_name}**\n\n{r.answer}" for r in results if r.answer)
            prefix = "\n\n" if parts else ""
            await self._stream_text(rt, prefix + fallback)
            parts.append(prefix + fallback)
        return {"answer": "".join(parts)}

    @staticmethod
    async def _stream_text(rt: TurnRuntime, text: str, size: int = 40) -> None:
        for i in range(0, len(text), size):
            rt.emit(text[i : i + size])
            await asyncio.sleep(0)

    # ------------------------------------------------------------------ entry point
    async def run(
        self,
        request: RequestContext,
        user_message: str,
        history: list[dict[str, str]],
        on_token: TokenSink,
    ) -> TurnOutcome:
        model = await self.registry.resolve(request.model_id)
        request.model_id = model.id
        rt = TurnRuntime(request=request, model=model, on_token=on_token)
        request.trace.header(model.id, request.actor.as_dict(), request.actor_source)
        if request.new_uploads:
            request.trace.uploads(request.new_uploads)
        state: TurnState = {"user_message": user_message, "history": history, "results": [], "step": 0}
        final = await self.graph.ainvoke(state, config={"configurable": {"rt": rt}, "recursion_limit": 40})
        results = final.get("results", [])
        answer = final.get("answer", "")
        error = final.get("error")
        status = "ok"
        if error:
            status = "error"
            answer = (
                f"I couldn't reach the selected model (`{model.id}`): {error}\n\n"
                "Check the provider settings in `.env`, start your local model server, or pick another model."
            )
            await self._stream_text(rt, answer)
        plan_data = final.get("plan", {})
        merged = self.config.orchestrator.synthesize != "never"
        if self.settings.show_flow:
            request.trace.flow(flow_lines(plan_data, results, merged=merged, error=error))
        if self.settings.show_flow and status == "ok":
            block = "\n\n---\n" + flow_block(plan_data, results, merged=merged)
            rt.emit(block)
            answer += block
        if self.settings.answer_footer and status == "ok":
            names = [r.agent_name for r in results]
            lead = "\n" if self.settings.show_flow else "\n\n---\n"  # the diagram already drew the rule
            footer = (
                f"{lead}_Agents: {', '.join(names) if names else 'orchestrator only'} · model `{model.id}` · "
                f"run `{request.run_id}`_"
            )
            rt.emit(footer)
            answer += footer
        return TurnOutcome(
            answer=answer,
            plan=final.get("plan", {}),
            results=results,
            usage=rt.usage,
            model_id=model.id,
            status=status,
            error=error,
        )
