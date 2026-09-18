"""Routing: an LLM planner with a deterministic keyword fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import CallOptions, ChatModel, LLMError, parse_json_object
from . import nlp

log = logging.getLogger(__name__)


@dataclass
class Step:
    agent: str
    task: str


@dataclass
class RoutePlan:
    mode: str  # agents | direct
    steps: list[Step] = field(default_factory=list)
    reason: str = ""
    answer: str = ""
    source: str = "llm"

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"mode": self.mode, "reason": self.reason, "source": self.source}
        if self.mode == "direct":
            data["answer"] = self.answer
        else:
            data["steps"] = [{"agent": s.agent, "task": s.task} for s in self.steps]
        return data


GREETINGS = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening", "ok", "okay", "cool", "bye")

WORKFLOW_ACTIONS = (
    "apply", "book ", "submit", "raise a", "request leave", "take leave", "take off", "approve", "reject",
    "decline", "cancel", "withdraw",
)
WORKFLOW_QUERIES = (
    "pending request", "my requests", "my leave requests", "request status", "status of my", "awaiting",
    "team's request", "team requests", "my team", "to approve", "leave request",
)
NOTIFY_WORDS = ("notify", "notification", "remind", "inform", "let my manager know", "email my", "message my",
                "tell my manager", "tell hr", "alert")
ELIGIBILITY_WORDS = (
    "balance", "how many days do i", "how many leave", "how many leaves", "days left", "left?", " left ",
    "remaining", "eligible", "eligibility", "entitled", "can i take", "working days", "probation", "joined",
    "holiday", "my profile", "who is my manager", "who's my manager", "accrued", "how many sick", "how many casual",
    "how many annual",
)
POLICY_WORDS = (
    "policy", "policies", "what is", "what's", "what are", "rule", "carry forward", "carry-forward", "encash",
    "parental", "maternity", "paternity", "adoption", "bereavement", "code of conduct", "harass", "gift",
    "information handling", "confidential", "faq", "allowed", "how does", "how do i", "notice period",
    "sick leave", "casual leave", "annual leave", "leave without pay", "optional holiday", "medical certificate",
)
# When the user is asking about an uploaded document, only these pull in the policy agent too.
COMPANY_POLICY_WORDS = ("company policy", "our policy", "hr policy", "leave policy", "compare", "comply", "complian",
                        "in line with", "against policy", "annual leave", "sick leave", "casual leave", "parental")
# With a workflow action, other agents are only added for explicit questions.
EXPLICIT_ELIGIBILITY = ("eligible", "eligibility", "balance", "how many", "enough")
EXPLICIT_POLICY = ("policy say", "what does the policy", "is it allowed", "am i allowed", "rules for")
DOC_WORDS = ("document", "file", "attached", "attachment", "upload", "pdf", "docx", "summar", "this doc",
             "the doc", "contract", "offer letter", "handbook", "report", "according to")


def heuristic_route(
    message: str,
    *,
    has_uploads: bool = False,
    agent_ids: list[str] | None = None,
    max_agents: int = 3,
) -> RoutePlan:
    text = (message or "").strip()
    lowered = f" {text.lower()} "
    available = set(agent_ids or ["policy_agent", "eligibility_agent", "workflow_agent", "notification_agent", "document_agent"])
    steps: list[Step] = []

    def add(agent: str, task: str) -> None:
        if agent in available and all(s.agent != agent for s in steps):
            steps.append(Step(agent, task))

    if nlp.looks_like_override(text):
        add("workflow_agent", f"The user asked: \"{text}\". This tries to bypass leave policy. Do not approve or "
                              "create anything that breaks policy; explain what is allowed instead.")
        add("policy_agent", "Find the policy clauses on self-approval, approvals and policy overrides.")
        return RoutePlan("agents", steps[:max_agents], "Possible policy-override attempt", source="heuristic")

    stripped = lowered.strip(" !.?")
    if stripped in GREETINGS or (len(stripped.split()) <= 3 and any(stripped.startswith(g) for g in GREETINGS)):
        return RoutePlan("direct", reason="Greeting / small talk", source="heuristic")

    notify_intent = nlp.has_any(lowered, NOTIFY_WORDS)
    workflow_action = nlp.has_any(lowered, WORKFLOW_ACTIONS) or (bool(nlp.request_ids(text)) and not notify_intent)
    if workflow_action:
        if nlp.has_any(lowered, EXPLICIT_ELIGIBILITY):
            add("eligibility_agent", f"Look up the personal HR data needed for: {text}")
        if nlp.has_any(lowered, EXPLICIT_POLICY):
            add("policy_agent", f"Find what the HR policies say about: {text}")
        add("workflow_agent", f"Handle this leave workflow request: {text}")
        return RoutePlan("agents", steps[:max_agents], "Workflow action (notifications follow automatically)", source="heuristic")

    doc_intent = has_uploads and nlp.has_any(lowered, DOC_WORDS)
    if doc_intent:
        add("document_agent", f"Answer from the uploaded document(s): {text}")
    if nlp.has_any(lowered, ELIGIBILITY_WORDS) or (nlp.person_names(text) and nlp.has_any(lowered, ("leave", "days"))):
        add("eligibility_agent", f"Look up the personal HR data needed for: {text}")
    policy_words = POLICY_WORDS if not doc_intent else COMPANY_POLICY_WORDS
    if nlp.has_any(lowered, policy_words) and not nlp.has_any(lowered, NOTIFY_WORDS + WORKFLOW_QUERIES):
        add("policy_agent", f"Find what the HR policies say about: {text}")
    if nlp.has_any(lowered, WORKFLOW_QUERIES):
        add("workflow_agent", f"Handle this leave workflow request: {text}")
    if nlp.has_any(lowered, NOTIFY_WORDS):
        add("notification_agent", f"Handle this notification request: {text}")

    if not steps:
        if has_uploads:
            add("document_agent", f"Answer from the uploaded document(s): {text}")
        else:
            add("policy_agent", f"Find what the HR policies say about: {text}")
    return RoutePlan("agents", steps[:max_agents], "Keyword routing", source="heuristic")


def validate_plan(data: dict[str, Any] | None, agent_ids: list[str], max_agents: int) -> RoutePlan | None:
    if not isinstance(data, dict):
        return None
    mode = str(data.get("mode", "agents")).lower()
    if mode == "direct":
        answer = str(data.get("answer") or "").strip()
        return RoutePlan("direct", reason=str(data.get("reason", "")), answer=answer)
    steps = []
    for raw in data.get("steps") or []:
        if not isinstance(raw, dict):
            continue
        agent = str(raw.get("agent", "")).strip()
        task = str(raw.get("task", "")).strip()
        if agent in agent_ids and task and all(s.agent != agent for s in steps):
            steps.append(Step(agent, task))
    if not steps:
        return None
    return RoutePlan("agents", steps[:max_agents], reason=str(data.get("reason", "")))


async def llm_route(
    model: ChatModel,
    system_prompt: str,
    history: list[dict[str, str]],
    message: str,
    *,
    agent_ids: list[str],
    max_agents: int,
    has_uploads: bool,
    context: dict[str, Any],
    usage_sink=None,
) -> RoutePlan:
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": message}]
    options = CallOptions(
        purpose="route",
        json_mode=True,
        temperature=0.0,
        usage_sink=usage_sink,
        context={**context, "user_message": message, "has_uploads": has_uploads, "agent_ids": agent_ids, "max_agents": max_agents},
    )
    try:
        result = await model.complete(messages, None, options)
        plan = validate_plan(parse_json_object(result.content), agent_ids, max_agents)
        if plan is not None:
            plan.source = "llm" if model.provider != "mock" else "mock"
            return plan
        log.warning("Router returned an unusable plan: %.300s", result.content)
    except LLMError as exc:
        log.warning("Router model failed (%s); using keyword routing", exc)
        raise
    fallback = heuristic_route(message, has_uploads=has_uploads, agent_ids=agent_ids, max_agents=max_agents)
    fallback.reason = "LLM plan was not valid JSON; keyword routing used"
    return fallback
