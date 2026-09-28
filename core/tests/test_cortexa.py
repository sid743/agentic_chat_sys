"""Cortexa Core as a model provider."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agentic_core.llm.base import (
    CALL_SCOPE,
    CallOptions,
    LLMError,
    ToolsNotSupported,
    Usage,
)
from agentic_core.llm.cortexa import CortexaChatModel, flatten
from agentic_core.llm.registry import ModelRegistry
from agentic_core.llm.tool_emulation import EmulatedToolsModel
from agentic_core.service import ChatRequest

from .conftest import make_settings

CREDS = {"CORTEXA_APP_ID": "app-123", "CORTEXA_CLIENT_SECRET": "secret-456", "CORTEXA_CORE_BASE_URL": "http://cortexa.test"}


# --------------------------------------------------------------------------- fakes
class FakeInteractions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, query, *, entrypoint="event", session_id=None, user_id=None, trace_id=None):
        call = {"query": query, "entrypoint": entrypoint, "session_id": session_id, "user_id": user_id, "trace_id": trace_id}
        self.owner.calls.append(call)
        if self.owner.error:
            raise self.owner.error
        text = self.owner.reply(call) if callable(self.owner.reply) else self.owner.reply
        extra = {"planner": {"steps": 2}, "composer": {"ok": True}} if entrypoint == "prism-event" else {}
        return SimpleNamespace(query=query, response=text, **extra)

    async def heartbeat(self):
        if self.owner.error:
            raise self.owner.error
        return SimpleNamespace(status="connected")


class FakeCortexa:
    """Stands in for AsyncCortexaEnterpriseClient: same shape, no network."""

    def __init__(self, reply="Hello from Cortexa", error=None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []
        self.opened = self.closed = 0

    def factory(self, app_id, client_secret, base_url, timeout, max_retries):
        self.seen = {"app_id": app_id, "client_secret": client_secret, "base_url": base_url}
        return self

    async def __aenter__(self):
        self.opened += 1
        self.interactions = FakeInteractions(self)
        return self

    async def __aexit__(self, *exc):
        self.closed += 1


def model(fake, entrypoint="event"):
    return CortexaChatModel(
        provider="cortexa",
        model=entrypoint,
        app_id="app-123",
        client_secret="secret-456",
        base_url="http://cortexa.test/",
        client_factory=fake.factory,
    )


# --------------------------------------------------------------------------- flattening
def test_chat_is_flattened_into_one_query():
    query = flatten(
        [
            {"role": "system", "content": "You are the Leave Policy Agent."},
            {"role": "user", "content": "What is the carry-forward rule?"},
            {"role": "assistant", "content": '{"tool": "search_policies", "arguments": {"query": "carry forward"}}'},
            {"role": "user", "content": "Tool result for search_policies:\n{...}"},
        ],
        json_mode=True,
    )
    assert query.index("## Instructions") < query.index("## Conversation so far") < query.index("## Message to answer")
    assert "You are the Leave Policy Agent." in query
    assert "Assistant: {\"tool\": \"search_policies\"" in query
    assert query.split("## Message to answer\n")[1].startswith("Tool result for search_policies")
    assert "Reply with ONLY a single JSON object" in query


def test_plain_question_has_no_json_instruction():
    query = flatten([{"role": "user", "content": "Hi"}])
    assert query == "## Message to answer\nHi"


# --------------------------------------------------------------------------- adapter
def test_complete_sends_one_query_with_session_metadata():
    fake = FakeCortexa(reply="  The answer.  ")
    usage = Usage()
    token = CALL_SCOPE.set({"session_id": "conv-1", "user_id": "E1001", "trace_id": "run-9"})
    try:
        result = asyncio.run(model(fake).complete([{"role": "user", "content": "Q?"}], None, CallOptions(usage_sink=usage)))
    finally:
        CALL_SCOPE.reset(token)

    assert result.content == "The answer."
    assert result.model == "cortexa/event"
    call = fake.calls[0]
    assert call["entrypoint"] == "event"
    assert (call["session_id"], call["user_id"], call["trace_id"]) == ("conv-1", "E1001", "run-9")
    assert fake.seen == {"app_id": "app-123", "client_secret": "secret-456", "base_url": "http://cortexa.test"}
    assert usage.input_tokens > 0 and usage.output_tokens > 0  # estimated: the API reports none
    assert fake.opened == fake.closed == 1  # a client per call, always closed


def test_prism_event_reports_its_modules():
    result = asyncio.run(model(FakeCortexa(), "prism-event").complete([{"role": "user", "content": "Q?"}]))
    assert result.meta["entrypoint"] == "prism-event"
    assert result.meta["prism_modules"] == ["composer", "planner"]


def test_native_tools_are_declined_so_the_json_protocol_is_used():
    m = model(FakeCortexa())
    assert m.native_tools is False
    with pytest.raises(ToolsNotSupported):
        asyncio.run(m.complete([{"role": "user", "content": "Q"}], tools=[{"type": "function"}]))


def test_unknown_entrypoint_is_refused():
    with pytest.raises(LLMError, match="event, prism-event"):
        model(FakeCortexa(), "gpt-5")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("AuthenticationError", "CORTEXA_APP_ID and CORTEXA_CLIENT_SECRET"),
        ("RateLimitError", "Rate limit reached"),
        ("APIConnectionError", "Could not connect"),
        ("NotFoundError", "CORTEXA_CORE_BASE_URL"),
    ],
)
def test_sdk_errors_become_readable_messages(name, expected):
    error = type(name, (Exception,), {})("boom")
    error.response_body = '{"detail": "quota"}'
    with pytest.raises(LLMError, match=expected):
        asyncio.run(model(FakeCortexa(error=error)).complete([{"role": "user", "content": "Q"}]))


# --------------------------------------------------------------------------- registry
def test_provider_stays_off_without_credentials(tmp_path):
    reg = ModelRegistry(make_settings(tmp_path), env={})
    assert not reg.providers["cortexa"].enabled
    with pytest.raises(LLMError, match="CORTEXA_APP_ID"):
        asyncio.run(reg.resolve("cortexa/event"))


def test_provider_lists_both_entrypoints_when_configured(tmp_path):
    reg = ModelRegistry(make_settings(tmp_path), env=CREDS)
    provider = reg.providers["cortexa"]
    assert provider.enabled and provider.app_id == "app-123" and provider.api_key == "secret-456"
    assert provider.base_url == "http://cortexa.test"
    ids = [m.id for m in asyncio.run(reg.available_models()) if m.provider == "cortexa"]
    assert ids == ["cortexa/event", "cortexa/prism-event"]
    resolved = asyncio.run(reg.resolve("cortexa/prism-event"))
    # Cortexa has no tool-calling API, so it is wrapped in one (llm/tool_emulation.py).
    assert isinstance(resolved, EmulatedToolsModel) and isinstance(resolved.inner, CortexaChatModel)
    assert resolved.native_tools and resolved.id == "cortexa/prism-event" and resolved.strategy == "emulated"


def test_json_tool_mode_leaves_cortexa_unwrapped(tmp_path):
    reg = ModelRegistry(make_settings(tmp_path), env={**CREDS, "CORTEXA_TOOL_MODE": "json"})
    resolved = asyncio.run(reg.resolve("cortexa/event"))
    assert isinstance(resolved, CortexaChatModel) and not resolved.native_tools


# --------------------------------------------------------------------------- the real SDK, network faked
def test_real_sdk_request_shape(monkeypatch):
    sdk = pytest.importorskip("cortexa_enterprise_sdk")
    from cortexa_enterprise_sdk import _base_client

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        if request.url.path == "/api/core/prism-event":
            return httpx.Response(200, json={"query": "q", "response": "prism answer", "planner": {"steps": 3}})
        return httpx.Response(200, json={"query": "q", "response": "standard answer"})

    real = httpx.AsyncClient
    monkeypatch.setattr(_base_client.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    m = CortexaChatModel(provider="cortexa", model="event", app_id="app-1", client_secret="sec-2", base_url="http://core.test")
    token = CALL_SCOPE.set({"session_id": "conv-7", "user_id": "E1002", "trace_id": "run-7"})
    try:
        out = asyncio.run(m.complete([{"role": "user", "content": "Am I eligible?"}]))
    finally:
        CALL_SCOPE.reset(token)

    assert out.content == "standard answer"
    assert seen["path"] == "/api/core/event"
    assert seen["headers"]["app-id"] == "app-1" and seen["headers"]["client-secret"] == "sec-2"
    assert seen["body"]["query"].endswith("Am I eligible?")
    assert seen["body"]["session_metadata"] == {"session_id": "conv-7", "user_id": "E1002", "trace_id": "run-7"}

    prism = CortexaChatModel(provider="cortexa", model="prism-event", app_id="a", client_secret="s", base_url="http://core.test")
    out = asyncio.run(prism.complete([{"role": "user", "content": "Q"}]))
    assert seen["path"] == "/api/core/prism-event"
    assert out.content == "prism answer" and out.meta["prism_modules"] == ["planner"]
    assert sdk.__version__


def test_real_sdk_401_is_explained(monkeypatch):
    pytest.importorskip("cortexa_enterprise_sdk")
    from cortexa_enterprise_sdk import _base_client

    real = httpx.AsyncClient
    monkeypatch.setattr(
        _base_client.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad creds")), **kw),
    )
    m = CortexaChatModel(provider="cortexa", model="event", app_id="a", client_secret="s", base_url="http://core.test")
    with pytest.raises(LLMError, match="CORTEXA_APP_ID and CORTEXA_CLIENT_SECRET"):
        asyncio.run(m.complete([{"role": "user", "content": "Q"}]))


# --------------------------------------------------------------------------- a whole turn on Cortexa
def cortexa_brain(call):
    """Plays the parts the platform asks for, the way an instruction-following model would."""
    query = call["query"]
    if "orchestrator of a multi-agent" in query:
        return json.dumps({
            "mode": "agents",
            "reason": "policy question",
            "steps": [{"agent": "policy_agent", "task": "Explain the parental leave policy"}],
        })
    answer = "Primary caregivers get 26 weeks (HR-POL-002 §4.1)."
    if "### How to reply" in query:  # emulated tool calling
        if "Tool results:" in query:
            return json.dumps({"answer": answer})
        return json.dumps({"tool_calls": [{"name": "search_policies", "arguments": {"query": "parental leave"}}]})
    if "Tool result for search_policies" in query:  # the older JSON protocol
        return json.dumps({"final": answer})
    if "reply with ONLY a JSON object" in query:
        return json.dumps({"tool": "search_policies", "arguments": {"query": "parental leave"}})
    return "Parental leave"


@pytest.mark.parametrize("tool_mode", ["emulated", "json"])
def test_a_full_turn_runs_on_cortexa(service, tmp_path, tool_mode):
    fake = FakeCortexa(reply=cortexa_brain)
    registry = ModelRegistry(service.settings, env={**CREDS, "CORTEXA_TOOL_MODE": tool_mode})
    registry.cortexa_client_factory = fake.factory
    service.registry = registry
    service.orchestrator.registry = registry

    request = ChatRequest(
        messages=[{"role": "user", "content": "What is our parental leave policy?"}],
        model="cortexa/event",
        conversation_id="conv-cortexa",
        employee_id="E1001",
    )
    out = asyncio.run(service.complete(request))

    assert out["status"] == "ok"
    assert out["agents"] == ["policy_agent"]
    assert "26 weeks" in out["content"]
    assert "retrieval (policies)" in out["reasoning"]  # the JSON protocol drove a real tool call
    # every call carried the turn's identity, so Cortexa's logs line up with ours
    assert {c["session_id"] for c in fake.calls} == {"conv-cortexa"}
    assert {c["trace_id"] for c in fake.calls} == {out["run_id"]}
    assert {c["user_id"] for c in fake.calls} == {"E1001"}


# --------------------------------------------------------------------------- cortexa-check
def run_check_with(fake, provider_env=CREDS, tmp_path=None):
    from agentic_core.llm.cortexa_check import run_check

    provider = ModelRegistry(make_settings(tmp_path), env=provider_env).providers["cortexa"]
    lines: list[str] = []
    code = asyncio.run(run_check(provider, client_factory=fake.factory, out=lines.append))
    return code, "\n".join(lines)


def well_behaved(call):
    query = call["query"]
    if "### How to reply" in query:
        if "Tool results:" in query:
            return '{"answer": "The notice period is 60 days (HR-POL-003 section 2)."}'
        return '{"tool_calls": [{"name": "lookup_policy", "arguments": {"topic": "resignation notice period"}}]}'
    return '{"ok": true, "word": "hi"}' if "JSON" in query else "pong"


def test_check_passes_and_never_prints_the_secret(tmp_path):
    code, text = run_check_with(FakeCortexa(reply=well_behaved), tmp_path=tmp_path)
    assert code == 0
    assert "heartbeat: connected" in text and "'pong'" in text and "JSON replies parse" in text
    assert 'Cortexa called a tool: lookup_policy({"topic": "resignation notice period"})' in text
    assert "answered from the tool result" in text and "Recommended" not in text
    assert "secret-456" not in text and "app-123" not in text  # masked


def test_check_counts_repairs(tmp_path):
    replies = iter(["pong", '{"ok": true}', 'lookup_policy(subject="notice")', "lookup_policy(topic='notice')",
                    '{"answer": "60 days."}'])
    code, text = run_check_with(FakeCortexa(reply=lambda call: next(replies)), tmp_path=tmp_path)
    assert code == 0 and "Cortexa called a tool after 1 repair(s)" in text and "1 repair(s) in total" in text


def test_check_recommends_delegate_when_cortexa_ignores_tools(tmp_path):
    code, text = run_check_with(FakeCortexa(reply="Sure! Here is some prose."), tmp_path=tmp_path)
    assert code == 0 and "reply was not JSON" in text and "AGENT_ROUTER_MODEL" in text
    assert "answered without calling the tool" in text
    assert "Recommended: CORTEXA_TOOL_MODE=delegate and CORTEXA_TOOL_PLANNER=gemini/gemini-3.1-flash-lite" in text


def test_check_fails_on_bad_credentials(tmp_path):
    error = type("AuthenticationError", (Exception,), {})("401")
    code, text = run_check_with(FakeCortexa(error=error), tmp_path=tmp_path)
    assert code == 1 and "CORTEXA_APP_ID and CORTEXA_CLIENT_SECRET" in text


def test_check_names_what_is_missing(tmp_path):
    code, text = run_check_with(FakeCortexa(), provider_env={"CORTEXA_APP_ID": "app-123"}, tmp_path=tmp_path)
    assert code == 1 and "missing in .env: CORTEXA_CLIENT_SECRET" in text
