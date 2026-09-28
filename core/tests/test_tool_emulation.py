"""Tool calling for text-only models (llm/tool_emulation.py)."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentic_core.llm.base import (
    CallOptions,
    ChatModel,
    ChatResult,
    LLMError,
    ToolCall,
    Usage,
    loads_lenient,
)
from agentic_core.llm.registry import ModelRegistry
from agentic_core.llm.tool_emulation import (
    ANSWER_FROM_RESULTS,
    EmulatedToolsModel,
    TextFunctionModel,
    parse_reply,
    run_tools,
    to_text_messages,
    tool_defs,
    tool_prompt,
    validate,
)
from agentic_core.service import ChatRequest
from agentic_core.tools import schemas_for

from .conftest import make_settings
from .test_cortexa import CREDS, FakeCortexa

TOOLS = schemas_for(["search_policies", "get_leave_balance", "list_leave_requests", "calculate_leave_days"])
DEFS = tool_defs(TOOLS)


# --------------------------------------------------------------------------- lenient JSON
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": "line one\nline two"}', {"a": "line one\nline two"}),  # raw newline in a string
        ('{"a": 1, "b": [1, 2,],}', {"a": 1, "b": [1, 2]}),  # trailing commas
        ("{'a': True, 'b': None}", {"a": True, "b": None}),  # a Python dict
        ('{"a": True, "b": "None of the above",}', {"a": True, "b": "None of the above"}),  # strings untouched
        ("{“a”: “b”}", {"a": "b"}),  # smart quotes
    ],
)
def test_loads_lenient(text, expected):
    assert loads_lenient(text) == expected


def test_loads_lenient_gives_up_cleanly():
    with pytest.raises(ValueError):
        loads_lenient("not json at all {")


# --------------------------------------------------------------------------- parsing
CALL_FORMS = {
    "canonical": '{"tool_calls": [{"name": "search_policies", "arguments": {"query": "carry forward"}}]}',
    "older protocol": '{"tool": "search_policies", "arguments": {"query": "carry forward"}}',
    "OpenAI shape, string arguments": '[{"type": "function", "function": {"name": "search_policies", "arguments": "{\\"query\\": \\"carry forward\\"}"}}]',
    "fenced after prose": 'Let me check.\n```json\n{"tool_calls": [{"name": "search_policies", "arguments": {"query": "carry forward"}}]}\n```',
    "tagged": '<tool_call>{"name": "search_policies", "arguments": {"query": "carry forward"}}</tool_call>',
    "ReAct": 'Thought: I need the policy.\nAction: search_policies\nAction Input: {"query": "carry forward"}',
    "python call": 'search_policies(query="carry forward")',
    "python positional": 'search_policies("carry forward")',
    "python dict": "{'tool_calls': [{'name': 'search_policies', 'arguments': {'query': 'carry forward',}}]}",
    "name as key": '{"search_policies": {"query": "carry forward"}}',
    "think block first": '<think>Which tool?</think>{"tool_calls": [{"name": "search_policies", "arguments": {"query": "carry forward"}}]}',
}


@pytest.mark.parametrize("text", CALL_FORMS.values(), ids=CALL_FORMS.keys())
def test_every_common_call_format_is_understood(text):
    parsed = parse_reply(text, DEFS)
    assert not parsed.errors
    assert [(c.name, c.arguments) for c in parsed.calls] == [("search_policies", {"query": "carry forward"})]
    assert parsed.answer is None


def test_several_calls_in_one_reply():
    parsed = parse_reply(
        '{"tool_calls": [{"name": "get_leave_balance", "arguments": {}},'
        ' {"name": "search_policies", "arguments": {"query": "carry forward"}}]}',
        DEFS,
    )
    assert [c.name for c in parsed.calls] == ["get_leave_balance", "search_policies"]
    assert len({c.id for c in parsed.calls}) == 2


@pytest.mark.parametrize(
    ("text", "answer", "form"),
    [
        ('{"answer": "You have **12** days."}', "You have **12** days.", "answer"),
        ('{"final": "Done."}', "Done.", "answer"),
        ('{"action": "Final Answer", "action_input": "Done."}', "Done.", "answer"),
        ("Thought: done\nFinal Answer: Done.", "Done.", "react"),
        ("You have 12 days of annual leave [1].", "You have 12 days of annual leave [1].", "prose"),
    ],
)
def test_answers(text, answer, form):
    parsed = parse_reply(text, DEFS)
    assert (parsed.answer, parsed.form, parsed.calls, parsed.errors) == (answer, form, [], [])


def test_json_inside_a_prose_answer_is_not_mistaken_for_a_call():
    text = 'Your balances:\n```json\n{"AL": 12, "SL": 5}\n```\nAnything else?'
    parsed = parse_reply(text, DEFS)
    assert parsed.form == "prose" and parsed.answer == text and not parsed.errors


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('{"tool_calls": [{"name": "search_policies", "arguments": {"query": "x"}]}', "not valid JSON"),
        ('{"tool_calls": [{"name": "delete_everything", "arguments": {}}]}', "no tool named 'delete_everything'"),
        ('{"tool_calls": [{"name": "search_policies", "arguments": {"top_k": 3}}]}', "missing required argument(s) query"),
        ('{"tool_calls": [{"name": "list_leave_requests", "arguments": {"scope": "everyone"}}]}', '"mine" | "awaiting_my_approval"'),
        ('{"tool_calls": [{"name": "search_policies", "arguments": "carry forward"}]}', "must be a JSON object"),
        ('{"AL": 12}', "not a tool call or an answer"),
        ("   ", "empty"),
    ],
)
def test_unusable_replies_say_what_is_wrong(text, error):
    parsed = parse_reply(text, DEFS)
    assert not parsed.calls and parsed.answer is None
    assert error in " ".join(parsed.errors)


# --------------------------------------------------------------------------- validation
def test_arguments_are_fitted_to_the_schema():
    call, errors, notes = validate(
        "List-Leave-Request",  # close enough to list_leave_requests
        {"scope": "MINE", "limit": "5", "status": None, "bogus": 1, "employeeId": "E1002"},
        DEFS,
    )
    assert not errors
    assert call.name == "list_leave_requests"
    assert call.arguments == {"scope": "mine", "limit": 5, "employee_id": "E1002"}
    assert any("ignored unknown argument 'bogus'" in n for n in notes)
    assert any("'employeeId' as 'employee_id'" in n for n in notes)


def test_booleans_and_integers_are_coerced_but_nonsense_is_not():
    call, errors, _ = validate("get_leave_balance", {"include_event_based": "yes"}, DEFS)
    assert call.arguments == {"include_event_based": True} and not errors
    _, errors, _ = validate("search_policies", {"query": "x", "top_k": "several"}, DEFS)
    assert "'top_k' should be integer" in errors[0]


# --------------------------------------------------------------------------- transcript
def test_native_history_is_rewritten_as_text():
    messages = [
        {"role": "system", "content": "You are the Leave Policy Agent."},
        {"role": "user", "content": "Carry forward rule?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_policies", "arguments": '{"query": "carry forward"}'}},
            {"id": "c2", "type": "function", "function": {"name": "get_leave_balance", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true, "results": []}'},
        {"role": "tool", "tool_call_id": "c2", "content": '{"ok": true, "AL": 12}'},
    ]
    out = to_text_messages(messages, DEFS, instructions=tool_prompt(DEFS))
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[0]["content"].startswith("You are the Leave Policy Agent.") and "### Tools" in out[0]["content"]
    assert json.loads(out[2]["content"])["tool_calls"][0] == {"name": "search_policies", "arguments": {"query": "carry forward"}}
    results = out[3]["content"]
    assert results.startswith("Tool results:") and "[search_policies] (call c1)" in results and "[get_leave_balance] (call c2)" in results
    assert "How to reply" in results  # the reminder of the format comes last, where it is read
    assert all("tool_calls" not in m or isinstance(m["content"], str) for m in out)


def test_prompt_lists_tools_with_types_enums_and_optional_marks():
    prompt = tool_prompt(DEFS)
    assert "search_policies(query: string, top_k?: integer, doc_code?: string)" in prompt
    assert 'scope?: "mine" | "awaiting_my_approval" | "team" | "all"' in prompt
    assert "start_date: date (YYYY-MM-DD)" in prompt
    assert '{"tool_calls": [{"name": "search_policies", "arguments": {"query": "..."}}]}' in prompt


# --------------------------------------------------------------------------- the wrapper
class ScriptedText(ChatModel):
    """A text-only model that replies from a script and records what it was sent."""

    provider, model, native_tools = "cortexa", "event", False

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, messages, tools=None, options=None):
        assert tools is None, "a text-only model must never be sent tools"
        options = options or CallOptions()
        self.calls.append({"messages": messages, "json_mode": options.json_mode})
        usage = Usage(10, 5)
        if options.usage_sink is not None:
            options.usage_sink.add(usage)
        return ChatResult(content=self.replies.pop(0), usage=usage, model=self.id, meta={"prism_modules": ["planner"]})

    async def stream(self, messages, options=None):
        yield (await self.complete(messages, None, options)).content


CONVO = [{"role": "system", "content": "You are the Leave Policy Agent."}, {"role": "user", "content": "Carry forward rule?"}]
GOOD_CALL = '{"tool_calls": [{"name": "search_policies", "arguments": {"query": "carry forward"}}]}'


def run(model, messages=CONVO, tools=TOOLS, options=None):
    return asyncio.run(model.complete(messages, tools, options))


def test_a_good_call_goes_straight_through():
    text = ScriptedText(GOOD_CALL)
    sink = Usage()
    reply = run(EmulatedToolsModel(text), options=CallOptions(usage_sink=sink))
    assert [c.name for c in reply.tool_calls] == ["search_policies"] and reply.finish_reason == "tool_calls"
    assert reply.meta["repairs"] == 0 and reply.meta["tool_strategy"] == "emulated"
    assert reply.meta["prism_modules"] == ["planner"]  # the inner model's extras survive
    sent = text.calls[0]
    assert sent["json_mode"] and "### Tools" in sent["messages"][0]["content"]
    assert sink.input_tokens == 10 and reply.usage.output_tokens == 5


def test_a_broken_call_is_sent_back_with_the_reason_and_then_used():
    text = ScriptedText('{"tool_calls": [{"name": "search_policies", "arguments": {"top_k": 2}}]}', GOOD_CALL)
    reply = run(EmulatedToolsModel(text))
    assert [c.arguments for c in reply.tool_calls] == [{"query": "carry forward"}]
    assert reply.meta["repairs"] == 1 and reply.meta["reply_forms"] == ["json", "json"]
    repair = text.calls[1]["messages"][-1]["content"]
    assert "missing required argument(s) query" in repair and '{"answer"' in repair
    assert text.calls[1]["messages"][-2] == {"role": "assistant", "content": '{"tool_calls": [{"name": "search_policies", "arguments": {"top_k": 2}}]}'}


def test_prose_before_any_tool_gets_one_nudge_then_is_accepted():
    text = ScriptedText("Carry forward is 10 days.", "Carry forward is 10 days, I am sure.")
    reply = run(EmulatedToolsModel(text))
    assert reply.content == "Carry forward is 10 days, I am sure." and not reply.tool_calls
    assert reply.meta["repairs"] == 1
    assert "call a tool now" in text.calls[1]["messages"][-1]["content"]


def test_prose_nudge_can_lead_to_a_tool_call():
    reply = run(EmulatedToolsModel(ScriptedText("Carry forward is 10 days.", GOOD_CALL)))
    assert [c.name for c in reply.tool_calls] == ["search_policies"]


def test_prose_after_tools_is_the_answer_without_a_nudge():
    history = [*CONVO, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "search_policies", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    text = ScriptedText("Per HR-POL-001 you may carry forward 10 days.")
    reply = run(EmulatedToolsModel(text), messages=history)
    assert reply.content.startswith("Per HR-POL-001") and len(text.calls) == 1


def test_it_gives_up_quietly_so_the_agent_can_ask_for_a_plain_answer():
    text = ScriptedText('{"x": 1}', '{"y": 2}', '{"z": 3}')
    reply = run(EmulatedToolsModel(text, max_repairs=2))
    assert reply.content == "" and not reply.tool_calls and reply.meta["gave_up"]
    assert len(text.calls) == 3


def test_valid_calls_are_kept_when_a_sibling_cannot_be_fixed():
    bad_and_good = '{"tool_calls": [' + GOOD_CALL[16:-2] + ', {"name": "nope", "arguments": {}}]}'
    reply = run(EmulatedToolsModel(ScriptedText(bad_and_good, bad_and_good), max_repairs=1))
    assert [c.name for c in reply.tool_calls] == ["search_policies"] and reply.meta["gave_up"]


def test_without_tools_it_is_a_plain_pass_through():
    text = ScriptedText("Hello!")
    reply = run(EmulatedToolsModel(text), tools=None)
    assert reply.content == "Hello!"
    assert text.calls[0]["messages"] == CONVO and not text.calls[0]["json_mode"]


# --------------------------------------------------------------------------- delegate
class Planner(ChatModel):
    provider, model = "gemini", "flash"

    def __init__(self, *replies, error=None):
        self.replies, self.error, self.seen = list(replies), error, []

    async def complete(self, messages, tools=None, options=None):
        self.seen.append(tools)
        if self.error:
            raise self.error
        return self.replies.pop(0)

    async def stream(self, messages, options=None):
        yield ""


def test_delegate_lets_the_planner_pick_tools_and_the_text_model_answer():
    planner = Planner(
        ChatResult(content="", tool_calls=[ToolCall("p1", "search_policies", {"query": "carry forward"})]),
        ChatResult(content="(planner's own answer, not used)"),
    )
    text = ScriptedText("Cortexa's answer from the results.")
    model = EmulatedToolsModel(text, planner=planner)
    first = run(model)
    assert [c.id for c in first.tool_calls] == ["p1"] and first.meta["tool_strategy"] == "delegate"
    assert planner.seen[0] == TOOLS and not text.calls  # Cortexa is not asked while tools are being picked

    history = [*CONVO, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "p1", "type": "function", "function": {"name": "search_policies", "arguments": '{"query": "carry forward"}'}}]},
        {"role": "tool", "tool_call_id": "p1", "content": '{"ok": true}'}]
    second = run(model, messages=history)
    assert second.content == "Cortexa's answer from the results." and not second.tool_calls
    assert text.calls[0]["messages"][-1]["content"] == ANSWER_FROM_RESULTS
    assert "### Tools" not in json.dumps(text.calls[0]["messages"])


def test_delegate_falls_back_to_emulated_when_the_planner_fails():
    model = EmulatedToolsModel(ScriptedText(GOOD_CALL), planner=Planner(error=LLMError("429 rate limited")))
    reply = run(model)
    assert [c.name for c in reply.tool_calls] == ["search_policies"]
    assert "429" in reply.meta["planner_error"] and reply.meta["tool_strategy"] == "emulated"


# --------------------------------------------------------------------------- registry
def test_registry_builds_each_tool_mode(tmp_path):
    def resolve(**env):
        return asyncio.run(ModelRegistry(make_settings(tmp_path), env={**CREDS, **env}).resolve("cortexa/event"))

    assert resolve().strategy == "emulated"
    delegate = resolve(CORTEXA_TOOL_MODE="delegate", CORTEXA_TOOL_PLANNER="gemini/gemini-3.1-flash-lite", GEMINI_API_KEY="k")
    assert delegate.strategy == "delegate" and delegate.planner.id == "gemini/gemini-3.1-flash-lite"
    with pytest.raises(LLMError, match="CORTEXA_TOOL_PLANNER"):
        resolve(CORTEXA_TOOL_MODE="delegate")
    with pytest.raises(LLMError, match="another provider"):
        resolve(CORTEXA_TOOL_MODE="delegate", CORTEXA_TOOL_PLANNER="cortexa/prism-event")


# --------------------------------------------------------------------------- standalone use
def test_run_tools_with_any_text_function():
    """How the module is used outside the platform: a raw `async (messages) -> str`."""
    replies = iter([
        "Action: get_weather\nAction Input: {\"city\": \"Mumbai\"}",
        '{"answer": "It is 31 C in Mumbai."}',
    ])
    seen = []

    async def ask(messages):
        seen.append(messages)
        return next(replies)

    tools = [{"name": "get_weather", "description": "Current weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    answer, transcript = asyncio.run(run_tools(
        EmulatedToolsModel(TextFunctionModel(ask, "sdk/any")),
        [{"role": "user", "content": "Weather in Mumbai?"}],
        tools,
        {"get_weather": lambda city: {"city": city, "celsius": 31}},
    ))
    assert answer == "It is 31 C in Mumbai."
    assert transcript[-1]["role"] == "tool" and json.loads(transcript[-1]["content"]) == {"city": "Mumbai", "celsius": 31}
    assert '"celsius": 31' in seen[1][-1]["content"]


# --------------------------------------------------------------------------- whole turns on Cortexa
def messy_cortexa(call):
    """A model that keeps the format loosely: a ReAct call with a camelCase argument,
    a call missing its required argument (needs one repair), and a fenced answer."""
    query = call["query"]
    last = query.split("## Message to answer")[-1]
    if "orchestrator of a multi-agent" in query:
        return json.dumps({"mode": "agents", "reason": "balance + policy", "steps": [
            {"agent": "eligibility_agent", "task": "Get the annual leave balance"},
            {"agent": "policy_agent", "task": "Find the carry-forward rule"}]})
    if "Findings:" in query:  # synthesis
        return "You have 12 days of annual leave, and can carry forward up to 10 days (HR-POL-001)."
    if "### How to reply" not in query:
        return "Leave summary"
    if "You are the Employee Eligibility Agent" in query:
        if "Tool results:" in query:
            return '{"answer": "Annual leave balance: 12 days."}'
        return 'Thought: balance first.\nAction: get_leave_balance\nAction Input: {"leaveType": "AL"}'
    if "Your last reply could not be used" in last:
        return '{"tool_calls": [{"name": "search_policies", "arguments": {"query": "carry forward annual leave"}}]}'
    if "Tool results:" in query:
        return '```json\n{"answer": "You can carry forward up to 10 days (HR-POL-001)."}\n```'
    return '{"tool_calls": [{"name": "search_policies", "arguments": {"top_k": 3}}]}'  # forgot the query


def test_a_whole_turn_with_a_model_that_keeps_the_format_loosely(service):
    fake = FakeCortexa(reply=messy_cortexa)
    registry = ModelRegistry(service.settings, env=CREDS)
    registry.cortexa_client_factory = fake.factory
    service.registry = service.orchestrator.registry = registry

    out = asyncio.run(service.complete(ChatRequest(
        messages=[{"role": "user", "content": "How much annual leave can I carry forward?"}],
        model="cortexa/event", conversation_id="conv-messy", employee_id="E1001",
    )))

    assert out["status"] == "ok" and "carry forward up to 10 days" in out["content"]
    assert out["agents"] == ["eligibility_agent", "policy_agent"]
    trace = out["reasoning"]
    assert "get_leave_balance" in trace and "retrieval (policies)" in trace  # both tools really ran
    assert "tool call fixed: get_leave_balance: read argument 'leaveType' as 'leave_type'" in trace
    assert "reply sent back to be fixed 1x (json -> json)" in trace
    assert {c["session_id"] for c in fake.calls} == {"conv-messy"}


# --------------------------------------------------------------------------- hardening
@pytest.mark.parametrize(
    "text",
    ["{" * 20000, '[{"a": "' * 3000, "[" * 3000 + "]" * 3000, "Employees don't lose leave. " * 8000],
    ids=["20k open braces", "unclosed strings", "deep nesting", "200KB prose"],
)
def test_pathological_replies_parse_fast_and_never_raise(text):
    import time

    started = time.monotonic()
    parse_reply(text, DEFS)
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize(
    "prefix",
    ["{" * 5000 + " ok ", "See [Bob's form] then ", 'Pick [the "best" option]: ', '{"tool_calls": ['],
    ids=["after stray braces", "apostrophe in brackets", "quoted word in brackets", "inside a broken outer object"],
)
def test_a_call_is_found_after_noise(prefix):
    parsed = parse_reply(prefix + CALL_FORMS["canonical"], DEFS)
    assert [c.name for c in parsed.calls] == ["search_policies"]


def test_react_edge_cases():
    empty = parse_reply("Action: search_policies\nAction Input: ", DEFS)
    assert "missing required argument(s) query" in empty.errors[0]  # a call attempt, not an answer
    assert [c.name for c in parse_reply("Action: get_leave_balance", DEFS).calls] == ["get_leave_balance"]
    prose = parse_reply("Please talk to your manager. Next Action: escalate", DEFS)
    assert prose.form == "prose" and not prose.errors


def test_exact_argument_name_wins_over_a_look_alike():
    call, errors, notes = validate("search_policies", {"Query": "wrong", "query": "right"}, DEFS)
    assert call.arguments == {"query": "right"} and not errors
    assert any("ignored 'Query'" in n for n in notes)


# --------------------------------------------------------------------------- review findings
WORKFLOW = tool_defs(schemas_for([
    "get_my_profile", "get_leave_balance", "check_leave_eligibility", "calculate_leave_days", "create_leave_request",
    "list_leave_requests", "approve_leave_request", "reject_leave_request", "cancel_leave_request", "find_employee",
    "search_policies",
]))


@pytest.mark.parametrize("wrong", [
    "disapprove_leave_request", "unapprove_leave_request", "remove_leave_request", "delete_leave_request",
    "revoke_leave_request", "retract_leave_request", "get_leave_request", "update_leave_request", "restore_leave_request",
])
def test_a_near_miss_never_becomes_a_different_action(wrong):
    from agentic_core.llm.tool_emulation import resolve_name

    assert resolve_name(wrong, WORKFLOW) == (None, None)
    parsed = parse_reply(json.dumps({"tool_calls": [{"name": wrong, "arguments": {"request_id": "LR-2026-0008"}}]}), WORKFLOW)
    assert not parsed.calls and f"no tool named '{wrong}'" in parsed.errors[0]


@pytest.mark.parametrize(("typed", "meant"), [
    ("list_leave_request", "list_leave_requests"), ("search_policy", "search_policies"),
    ("search_polices", "search_policies"), ("searchPolicies", "search_policies"), ("functions.get_my_profile", "get_my_profile"),
])
def test_harmless_near_misses_still_resolve(typed, meant):
    from agentic_core.llm.tool_emulation import resolve_name

    assert resolve_name(typed, WORKFLOW)[0] == meant


@pytest.mark.parametrize("text", [
    "find_employee(query=...)",
    "{'tool_calls':[{'name':'find_employee','arguments':{'query': ...}}]}",
    '{"tool_calls":[{"name":"search_policies","arguments":{"carry forward"}}]}',
    "{'answer': {1, 2}}",
    'Action: search_policies\nAction Input: {"query": ...}',
    "get_my_profile(x=...)",
    "[" * 10000,
])
def test_non_json_values_are_errors_not_crashes(text):
    parsed = parse_reply(text, WORKFLOW)  # must not raise
    for call in parsed.calls:
        json.dumps(call.arguments)  # and whatever comes out can be sent on as JSON
    assert parsed.errors or parsed.form == "prose"  # the model is told, or it was never a call


def test_parse_json_object_survives_deep_nesting():
    from agentic_core.llm.base import parse_json_object

    assert parse_json_object("[" * 10000) is None


@pytest.mark.parametrize("text", [
    'Here is your balance:\n```json\n{"leave_balance": {"employee_id": "E1001", "AL": 12}}\n```\nAnything else?',
    'The request was updated: {"request_id": "LR-2026-0008", "action": "approved"}.',
    'In the log, entries look like {"tool": "hammer", "count": 2}. Nothing else changed.',
    'Your manager left a note: {"response": "approved for next week"}. Enjoy the break!',
])
def test_data_quoted_inside_a_prose_answer_stays_prose(text):
    parsed = parse_reply(text, WORKFLOW)
    assert (parsed.form, parsed.answer, parsed.calls, parsed.errors) == ("prose", text, [], [])


@pytest.mark.parametrize(("text", "name"), [
    ('create_leave_request(\n    leave_type="AL",\n    start_date="2026-10-05",\n    end_date="2026-10-06",\n)', "create_leave_request"),
    ('search_policies(\n query="x"\n)', "search_policies"),
    ('find_employee(query="E1003")\nThen I will check their balance.', "find_employee"),
    ("Action: get_my_profile\n\nI need your profile first.", "get_my_profile"),
    ('<function=search_policies>{"query": "x"}</function>', "search_policies"),
])
def test_call_attempts_never_reach_the_user_as_an_answer(text, name):
    parsed = parse_reply(text, WORKFLOW)
    assert parsed.answer is None
    assert [c.name for c in parsed.calls] == [name] or any(name in e for e in parsed.errors)


@pytest.mark.parametrize(("text", "answer"), [
    ('{"answer": "Tap “Apply” now", }', "Tap “Apply” now"),
    ("{'answer': 'It’s 12 days'}", "It’s 12 days"),
    ('{"answer": "Pick one of [AL, SL, ]", }', "Pick one of [AL, SL, ]"),
    ('{"answer": "You can carry forward up to 10 days, and the rest', "You can carry forward up to 10 days, and the rest"),
])
def test_answer_text_is_never_altered_or_leaked_as_json(text, answer):
    assert parse_reply(text, WORKFLOW).answer == answer


def test_smart_quotes_inside_arguments_are_kept():
    defs = tool_defs(schemas_for(["send_notification"]))
    parsed = parse_reply(
        '{"tool_calls": [{"name": "send_notification", "arguments": {"recipient": "E1002", "subject": "Leave",'
        ' "message": "Use the “Apply” button", "related_request_id": None}}]}',
        defs,
    )
    assert not parsed.errors and parsed.calls[0].arguments["message"] == "Use the “Apply” button"


def test_none_strings_mean_not_given_for_optional_fields():
    call, errors, _ = validate("get_leave_balance", {"employee_id": "None", "leave_type": "null"}, WORKFLOW)
    assert call.arguments == {} and not errors


def test_delegate_does_not_blame_the_planner_for_the_text_models_errors():
    class Failing(ScriptedText):
        async def complete(self, messages, tools=None, options=None):
            raise LLMError("cortexa 503")

    model = EmulatedToolsModel(Failing(), planner=Planner(ChatResult(content="done")))
    with pytest.raises(LLMError, match="cortexa 503"):
        run(model)


def test_planners_must_be_native_so_there_are_no_loops(tmp_path):
    providers = tmp_path / "providers.yaml"
    providers.write_text(
        "providers:\n"
        "  a: {type: openai, base_url: http://a, api_key: k, models: [m1], tool_mode: delegate, tool_planner: b/m2}\n"
        "  b: {type: openai, base_url: http://b, api_key: k, models: [m2], tool_mode: delegate, tool_planner: a/m1}\n"
        "  c: {type: openai, base_url: http://c, api_key: k, models: [m3], tool_mode: delegate, tool_planner: d/m4}\n"
        "  d: {type: openai, base_url: http://d, api_key: k, models: [m4], tool_mode: emulated}\n",
        encoding="utf-8",
    )
    reg = ModelRegistry(make_settings(tmp_path), providers_file=providers, env={})
    with pytest.raises(LLMError, match="needs native tool calling"):
        asyncio.run(reg.resolve("a/m1"))
    with pytest.raises(LLMError, match="needs native tool calling"):
        asyncio.run(reg.resolve("c/m3"))


def test_a_prose_answer_between_broken_replies_is_not_lost():
    reply = run(EmulatedToolsModel(ScriptedText('{"x": 1}', "Carry forward is 10 days.", '{"y": 2}'), max_repairs=2))
    assert reply.content == "Carry forward is 10 days." and reply.meta["gave_up"]


def test_run_tools_asks_for_a_plain_answer_when_the_model_gives_up():
    replies = iter(['{"x": 1}', '{"y": 2}', '{"z": 3}', "Here is a plain answer."])

    async def ask(messages):
        return next(replies)

    answer, _ = asyncio.run(run_tools(EmulatedToolsModel(TextFunctionModel(ask)), CONVO, TOOLS, {}))
    assert answer == "Here is a plain answer."
