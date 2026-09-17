"""A scripted OpenAI-compatible server (httpx transport) that behaves like a real
tool-calling model, so the full HTTP path can be tested without network access."""

from __future__ import annotations

import json
import time

import httpx


def _completion(message: dict, finish: str = "stop") -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fake-model",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _sse(text: str) -> bytes:
    chunks = []
    for piece in [text[i : i + 7] for i in range(0, len(text), 7)]:
        chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "fake-model",
                       "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
    chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "fake-model",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "fake-model", "choices": [],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


class FakeLLM:
    """mode: native | json | no_tools (rejects the tools parameter like some Ollama models)."""

    def __init__(self, mode: str = "native") -> None:
        self.mode = mode
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "fake-model"}, {"id": "embed-x"}]})
        body = json.loads(request.content)
        self.requests.append(body)
        messages = body["messages"]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        last = messages[-1]

        if body.get("stream"):
            text = "<think>internal notes</think>Combined answer: balance and policy both checked. Sources: Leave Policy (HR-POL-001) v3.2 §3.5"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(text))

        if "orchestrator of a multi-agent" in system:
            plan = {
                "mode": "agents",
                "reason": "needs balance and policy",
                "steps": [
                    {"agent": "eligibility_agent", "task": "Get the user's annual leave balance"},
                    {"agent": "policy_agent", "task": "Find the carry-forward rule for annual leave"},
                ],
            }
            return httpx.Response(200, json=_completion({"content": "```json\n" + json.dumps(plan) + "\n```"}))

        if "tools" in body:
            if self.mode == "no_tools":
                return httpx.Response(400, json={"error": {"message": "registry/fake-model does not support tools", "type": "invalid_request_error"}})
            if last["role"] == "tool":
                return httpx.Response(200, json=_completion({"content": f"Final answer using tool data: {last['content'][:80]}"}))
            name, args = ("get_leave_balance", {"leave_type": "AL"}) if "Eligibility Agent" in system else (
                "search_policies", {"query": "carry forward annual leave"})
            call = {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            return httpx.Response(200, json=_completion({"content": None, "tool_calls": [call]}, "tool_calls"))

        if "reply with ONLY a JSON object" in system:
            assert body.get("response_format") == {"type": "json_object"}
            if last["role"] == "user" and last["content"].startswith("Tool result for"):
                return httpx.Response(200, json=_completion({"content": json.dumps({"final": "JSON-mode final answer."})}))
            name = "get_leave_balance" if "Eligibility Agent" in system else "search_policies"
            args = {} if name == "get_leave_balance" else {"query": "carry forward"}
            return httpx.Response(200, json=_completion({"content": json.dumps({"tool": name, "arguments": args})}))

        return httpx.Response(200, json=_completion({"content": "Leave Balance Question"}))


PROVIDERS_YAML = """
providers:
  fake:
    type: openai
    base_url: http://fake.local/v1
    api_key: test
    fetch_models: true
    model_exclude: "embed"
    models: [fake-model]
    tool_mode: native
  fakejson:
    type: openai
    base_url: http://fakejson.local/v1
    api_key: test
    models: [fake-model]
    tool_mode: json
  notools:
    type: openai
    base_url: http://notools.local/v1
    api_key: test
    models: [fake-model]
  offline:
    type: openai
    local: true
    base_url: http://offline.local/v1
    api_key: none
    fetch_models: true
    models: []
  limited:
    type: openai
    base_url: http://limited.local/v1
    api_key: test
    models: [fake-model]
  mock:
    type: mock
    models: [hr-demo]
default_priority: [fake, mock]
"""
