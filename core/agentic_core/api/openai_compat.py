"""OpenAI-compatible endpoints so LibreChat can use the agent core as a custom endpoint.

- GET  /v1/models
- POST /v1/chat/completions   (stream and non-stream)

The agent trace is streamed as `reasoning_content` deltas, which LibreChat renders
in its collapsible "Thoughts" block; the answer is streamed as normal `content`.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from ..llm.base import Usage
from ..service import ChatRequest, ChatService
from .deps import get_service, require_api_key

router = APIRouter(tags=["openai-compatible"])


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None


def _header(request: Request, *names: str) -> str:
    for name in names:
        value = request.headers.get(name, "").strip()
        if value and not value.startswith("{{"):
            return value
    return ""


def _chat_request(body: ChatCompletionRequest, request: Request) -> ChatRequest:
    return ChatRequest(
        messages=body.messages,
        model=body.model,
        user_email=_header(request, "x-user-email", "x-librechat-user-email"),
        user_name=_header(request, "x-user-name"),
        conversation_id=_header(request, "x-conversation-id", "x-librechat-conversation-id"),
        employee_id=_header(request, "x-employee-id"),
    )


def _usage_dict(usage: Usage | None) -> dict[str, int]:
    usage = usage or Usage()
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
@router.get("/models", dependencies=[Depends(require_api_key)], include_in_schema=False)
async def list_models(service: ChatService = Depends(get_service)) -> dict[str, Any]:
    models = await service.registry.available_models()
    data = [{"id": "auto", "object": "model", "created": 0, "owned_by": "agentic-core"}]
    data += [{"id": m.id, "object": "model", "created": 0, "owned_by": m.provider} for m in models]
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
@router.post("/chat/completions", dependencies=[Depends(require_api_key)], include_in_schema=False)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    service: ChatService = Depends(get_service),
):
    chat = _chat_request(body, request)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = body.model or "auto"
    field = request.app.state.settings.reasoning_field

    if not body.stream:
        result = await service.complete(chat)
        content = result["content"]
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if field in ("reasoning_content", "reasoning") and result["reasoning"]:
            message[field] = result["reasoning"]
        elif field == "think" and result["reasoning"]:
            message["content"] = f"<think>\n{result['reasoning']}\n</think>\n\n{content}"
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": _usage_dict(result.get("usage")),
                "agentic_core": {
                    "run_id": result.get("run_id"),
                    "status": result.get("status"),
                    "agents": result.get("agents", []),
                    "resolved_model": result.get("model"),
                    "latency_ms": result.get("latency_ms"),
                },
            }
        )

    include_usage = bool((body.stream_options or {}).get("include_usage"))

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def events():
        yield chunk({"role": "assistant", "content": ""})
        think_open = False
        done: dict[str, Any] = {}
        async for event in service.stream(chat):
            if event.kind == "reasoning":
                if field in ("reasoning_content", "reasoning"):
                    yield chunk({field: event.text})
                elif field == "think":
                    if not think_open:
                        think_open = True
                        yield chunk({"content": "<think>\n"})
                    yield chunk({"content": event.text})
            elif event.kind == "content":
                if think_open:
                    think_open = False
                    yield chunk({"content": "\n</think>\n\n"})
                yield chunk({"content": event.text})
            elif event.kind == "done":
                done = event.data
        if think_open:
            yield chunk({"content": "\n</think>\n\n"})
        yield chunk({}, "stop")
        if include_usage:
            usage_payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [],
                "usage": _usage_dict(done.get("usage")),
            }
            yield f"data: {json.dumps(usage_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
