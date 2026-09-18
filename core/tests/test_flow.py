"""The little graph drawn under every answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_core.agents.flow import flow_block, flow_lines


@dataclass
class FakeResult:
    agent_id: str
    agent_name: str
    tools: list = field(default_factory=list)
    ms: int = 1200
    ok: bool = True


def plan(*agents, source="llm", reason="", auto=()):
    return {
        "mode": "agents",
        "source": source,
        "reason": reason,
        "steps": [{"agent": a, "task": "t", **({"auto": True} if a in auto else {})} for a in agents],
    }


def test_chain_shows_route_agents_and_synthesize():
    results = [
        FakeResult("eligibility_agent", "Eligibility Agent", tools=[1, 2], ms=1400),
        FakeResult("policy_agent", "Leave Policy Agent", tools=[1], ms=900),
    ]
    lines = flow_lines(plan("eligibility_agent", "policy_agent", reason="balance and rule"), results, merged=True)
    assert lines[0] == "START -> route(llm) -> eligibility_agent -> policy_agent -> synthesize -> END"
    assert "llm plan: balance and rule" in lines[1]
    assert "Eligibility Agent (2 tools, 1.4s)" in lines[2]
    assert "Leave Policy Agent (1 tool, 0.9s)" in lines[3]
    assert lines[3].startswith("  " + " " * 8)  # continuation lines line up under the label
    assert "synthesize merged 2 agent answers" in lines[4]


def test_single_agent_answer_is_marked_as_passed_through():
    results = [FakeResult("policy_agent", "Leave Policy Agent", tools=[1], ms=800)]
    lines = flow_lines(plan("policy_agent"), results, merged=True)
    assert lines[0].endswith("policy_agent -> synthesize -> END")
    assert "Leave Policy Agent, passed through synthesize unchanged" in lines[-1]


def test_automatic_hand_off_is_starred():
    results = [
        FakeResult("workflow_agent", "Approval & Workflow Agent", tools=[1]),
        FakeResult("notification_agent", "Notification Agent", tools=[1, 2]),
    ]
    steps = plan("workflow_agent", "notification_agent", auto=("notification_agent",))
    lines = flow_lines(steps, results, merged=True)
    assert "notification_agent*" in lines[0]
    assert "auto" in lines[3]
    # the auto agent does not count as the answering agent
    assert "Approval & Workflow Agent, passed through" in lines[-1]


def test_direct_and_error_paths():
    direct = flow_lines({"mode": "direct", "source": "rules", "reason": "greeting"}, [], merged=True)
    assert direct[0] == "START -> route(rules) -> direct -> END"
    assert "direct (no agent, no tools)" in direct[-1]

    stopped = flow_lines(plan("policy_agent"), [], merged=True, error="model unreachable")
    assert stopped[0] == "START -> route(llm) -> stop -> END"
    assert "failed: model unreachable" in stopped[1]


def test_block_is_a_plain_code_fence():
    block = flow_block(plan("policy_agent"), [FakeResult("policy_agent", "Leave Policy Agent")], merged=True)
    assert block.startswith("```text\n") and block.endswith("\n```")
    assert "START -> route(llm)" in block


def test_failed_agent_is_labelled():
    results = [FakeResult("policy_agent", "Leave Policy Agent", tools=[], ms=300, ok=False)]
    lines = flow_lines(plan("policy_agent"), results, merged=True)
    assert "no tools" in lines[2] and "failed" in lines[2]
