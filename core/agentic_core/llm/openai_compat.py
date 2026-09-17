"""Chat model for any OpenAI-compatible API: OpenAI, Groq, Ollama, LM Studio,
vLLM, OpenRouter, Together, Mistral, DeepSeek, Azure-compatible gateways, ...
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from .base import (
    CallOptions,
    ChatModel,
    ChatResult,
    LLMError,
    ThinkFilter,
    ToolCall,
    ToolsNotSupported,
    Usage,
    estimate_tokens,
    strip_think,
)

log = logging.getLogger(__name__)

# Reasoning-style models that reject a custom temperature.
_NO_TEMPERATURE = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)


def _loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"_raw": raw}


class OpenAICompatModel(ChatModel):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 90.0,
        headers: dict[str, str] | None = None,
        native_tools: bool = True,
        json_mode: bool = True,
        default_temperature: float | None = 0.2,
        extra_body: dict[str, Any] | None = None,
        http_client: Any = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.native_tools = native_tools
        self.json_mode = json_mode
        self.accepts_temperature = not _NO_TEMPERATURE.match(model.split("/")[-1])
        self.default_temperature = default_temperature
        self.extra_body = extra_body or {}
        self.stream_usage = True
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=2,
            default_headers=headers or None,
            http_client=http_client,
        )

    # ------------------------------------------------------------------ helpers
    def _kwargs(self, messages, options: CallOptions) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        temperature = options.temperature if options.temperature is not None else self.default_temperature
        if temperature is not None and self.accepts_temperature:
            kwargs["temperature"] = temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    def _friendly(self, exc: Exception) -> LLMError:
        where = f"{self.provider} ({self.base_url})"
        if isinstance(exc, openai.AuthenticationError):
            return LLMError(f"Authentication failed for {where}. Check the API key in .env.")
        if isinstance(exc, openai.PermissionDeniedError):
            return LLMError(f"Access denied by {where} for model '{self.model}'.")
        if isinstance(exc, openai.NotFoundError):
            return LLMError(f"Model '{self.model}' was not found at {where}. Is it pulled/deployed?")
        if isinstance(exc, openai.RateLimitError):
            return LLMError(f"Rate limit reached at {where}. Try again shortly or pick another model.")
        if isinstance(exc, openai.APITimeoutError):
            return LLMError(f"Timed out waiting for {where}.")
        if isinstance(exc, openai.APIConnectionError):
            return LLMError(f"Could not connect to {where}. Is the server running and reachable?")
        if isinstance(exc, openai.APIStatusError):
            return LLMError(f"{where} returned HTTP {exc.status_code}: {str(exc)[:300]}")
        return LLMError(f"Call to {where} failed: {exc}")

    async def _create(self, kwargs: dict[str, Any], has_tools: bool):
        """Create a completion, adapting to common provider quirks once each."""
        for _attempt in range(4):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if has_tools and ("tool" in msg or "function" in msg):
                    if "tool_use_failed" in msg or "failed to call a function" in msg:
                        raise ToolsNotSupported(f"{self.id} produced an invalid tool call") from exc
                    raise ToolsNotSupported(f"{self.id} does not accept native tool calls") from exc
                if "response_format" in kwargs and ("response_format" in msg or "json" in msg):
                    kwargs.pop("response_format", None)
                    self.json_mode = False
                    continue
                if "temperature" in kwargs and "temperature" in msg:
                    kwargs.pop("temperature", None)
                    self.accepts_temperature = False
                    continue
                if "stream_options" in kwargs and "stream_options" in msg:
                    kwargs.pop("stream_options", None)
                    self.stream_usage = False
                    continue
                raise self._friendly(exc) from exc
            except openai.OpenAIError as exc:
                raise self._friendly(exc) from exc
        raise LLMError(f"{self.id}: request could not be adapted to the provider")

    # ------------------------------------------------------------------ API
    async def complete(self, messages, tools=None, options=None) -> ChatResult:
        options = options or CallOptions()
        kwargs = self._kwargs(messages, options)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if options.json_mode and self.json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._create(kwargs, has_tools=bool(tools))
        if not resp.choices:
            raise LLMError(f"{self.id} returned no choices")
        choice = resp.choices[0]
        message = choice.message
        calls = []
        for i, tc in enumerate(message.tool_calls or []):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            calls.append(ToolCall(id=tc.id or f"call_{i}", name=fn.name, arguments=_loads(fn.arguments)))
        content = strip_think(message.content or "")
        usage = Usage()
        if resp.usage:
            usage = Usage(resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
        else:
            usage = Usage(estimate_tokens(messages), estimate_tokens(content))
        if options.usage_sink is not None:
            options.usage_sink.add(usage)
        return ChatResult(content=content, tool_calls=calls, usage=usage, model=self.id, finish_reason=choice.finish_reason)

    async def stream(self, messages, options=None) -> AsyncIterator[str]:
        options = options or CallOptions()
        kwargs = self._kwargs(messages, options)
        kwargs["stream"] = True
        if self.stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        stream = await self._create(kwargs, has_tools=False)
        think = ThinkFilter()
        produced = []
        got_usage = False
        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None) and options.usage_sink is not None:
                    options.usage_sink.add(Usage(chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0))
                    got_usage = True
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    visible = think.feed(text)
                    if visible:
                        produced.append(visible)
                        yield visible
        except openai.OpenAIError as exc:
            raise self._friendly(exc) from exc
        tail = think.flush()
        if tail:
            produced.append(tail)
            yield tail
        if not got_usage and options.usage_sink is not None:
            options.usage_sink.add(Usage(estimate_tokens(messages), estimate_tokens("".join(produced))))
