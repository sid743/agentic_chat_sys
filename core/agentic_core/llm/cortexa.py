"""Cortexa Core as a model provider, through the Cortexa Enterprise SDK.

Cortexa Core is not an OpenAI-style chat API. Its SDK takes one natural-language
`query` string and returns one `response` string, over two entrypoints:

    event        the standard path
    prism-event  the same, plus the output of Cortexa's own reasoning modules
                 (planner, collector, challenger, composer, ...)

So this adapter:

- flattens the chat (system prompt, conversation, tool results) into one query;
- declares no native tool calling, so the agents use the JSON tool protocol -
  the query asks for a JSON object and the reply is parsed like any other model's;
- tags every call with session_id = conversation, user_id = employee, trace_id =
  run id, so a turn can be followed from this platform's logs into Cortexa's;
- has no token counts from the API, so usage is estimated.

The SDK is optional: nothing imports it until a `cortexa/...` model is used.
Install the wheel you were given (see README, "Cortexa Core").
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from .base import (
    CALL_SCOPE,
    CallOptions,
    ChatModel,
    ChatResult,
    LLMError,
    ToolsNotSupported,
    Usage,
    estimate_tokens,
    strip_think,
)

log = logging.getLogger(__name__)

ENTRYPOINTS = ("event", "prism-event")
# Reasoning modules that prism-event reports on; recorded in the trace when present.
PRISM_MODULES = (
    "decision_gate",
    "intent_classifier",
    "workload_classifier",
    "collector",
    "challenger",
    "planner",
    "communicator",
    "composer",
)

INSTALL_HINT = (
    "The Cortexa SDK is not installed. Install the wheel you were given: "
    "`pip install cortexa_enterprise_sdk-0.1.0-py3-none-any.whl` (on Python 3.11 add "
    "`--ignore-requires-python`), or put it in core/vendor/ and rebuild the Docker image."
)


def flatten(messages: list[dict[str, Any]], *, json_mode: bool = False) -> str:
    """Turn chat messages into the single query string Cortexa Core expects."""
    system = [m.get("content") or "" for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") != "system"]
    latest = turns[-1] if turns and turns[-1].get("role") == "user" else None
    earlier = turns[:-1] if latest else turns

    parts: list[str] = []
    if system:
        parts.append("## Instructions\n" + "\n\n".join(s.strip() for s in system if s.strip()))
    if earlier:
        lines = []
        for m in earlier:
            who = {"user": "User", "assistant": "Assistant", "tool": "Tool result"}.get(m.get("role", ""), "Note")
            content = m.get("content") or ""
            if isinstance(content, list):  # multi-part content: keep the text parts
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            lines.append(f"{who}: {str(content).strip()}")
        parts.append("## Conversation so far\n" + "\n\n".join(lines))
    if latest:
        parts.append("## Message to answer\n" + str(latest.get("content") or "").strip())
    if json_mode:
        parts.append(
            "## Output format\nReply with ONLY a single JSON object. No prose before or after it, "
            "no markdown code fences."
        )
    return "\n\n".join(parts)


def _default_client(app_id: str, client_secret: str, base_url: str, timeout: float, max_retries: int) -> Any:
    try:
        from cortexa_enterprise_sdk import AsyncCortexaEnterpriseClient
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LLMError(INSTALL_HINT) from exc
    return AsyncCortexaEnterpriseClient(
        app_id=app_id,
        client_secret=client_secret,
        timeout=timeout,
        max_retries=max_retries,
        _base_url_override=base_url,
    )


class CortexaChatModel(ChatModel):
    native_tools = False  # one query string in, one string out: tools go through the JSON protocol

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        app_id: str,
        client_secret: str,
        base_url: str,
        timeout: float = 90.0,
        max_retries: int = 2,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if model not in ENTRYPOINTS:
            raise LLMError(f"Unknown Cortexa entrypoint '{model}'. Use one of: {', '.join(ENTRYPOINTS)}.")
        self.provider = provider
        self.model = model
        self.app_id = app_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.client_factory = client_factory or _default_client
        self.native_tools = False

    # ------------------------------------------------------------------ errors
    def _friendly(self, exc: Exception) -> LLMError:
        """Map SDK exceptions (matched by name, so the SDK stays optional) to plain sentences."""
        where = f"Cortexa Core ({self.base_url})"
        kind = type(exc).__name__
        body = str(getattr(exc, "response_body", "") or "").strip()
        detail = f" Server says: {body[:240]}" if body else ""
        if kind == "AuthenticationError":
            return LLMError(f"Authentication failed for {where}. Check CORTEXA_APP_ID and CORTEXA_CLIENT_SECRET.")
        if kind == "PermissionDeniedError":
            return LLMError(f"{where} refused these credentials for '{self.model}'.{detail}")
        if kind == "NotFoundError":
            return LLMError(f"{where} has no '{self.model}' endpoint. Check CORTEXA_CORE_BASE_URL.{detail}")
        if kind == "RateLimitError":
            return LLMError(f"Rate limit reached at {where}.{detail} Wait a moment or pick another model.")
        if kind == "BadRequestError":
            return LLMError(f"{where} rejected the request.{detail}")
        if kind == "InternalServerError":
            return LLMError(f"{where} had an internal error.{detail}")
        if kind == "APIConnectionError":
            return LLMError(f"Could not connect to {where}. Is CORTEXA_CORE_BASE_URL right, and reachable from here?")
        return LLMError(f"Call to {where} failed: {exc}")

    # ------------------------------------------------------------------ API
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: CallOptions | None = None,
    ) -> ChatResult:
        options = options or CallOptions()
        if tools:
            # The agent runtime reads native_tools and never sends tools here; if something
            # does, this sends it down the JSON-protocol path instead of failing.
            raise ToolsNotSupported(f"{self.id} takes a single query; tools go through the JSON protocol")
        query = flatten(messages, json_mode=options.json_mode)
        scope = CALL_SCOPE.get() or {}

        client = self.client_factory(self.app_id, self.client_secret, self.base_url, self.timeout, self.max_retries)
        try:
            # A client per call: its HTTP pool is bound to the running event loop, and the
            # platform runs one loop per request in some paths (CLI, tests).
            async with client:
                reply = await client.interactions.create(
                    query=query,
                    entrypoint=self.model,
                    session_id=scope.get("session_id") or None,
                    user_id=scope.get("user_id") or None,
                    trace_id=scope.get("trace_id") or None,
                )
        except LLMError:
            raise
        except Exception as exc:  # every SDK error becomes a readable LLMError
            raise self._friendly(exc) from exc

        text = strip_think(str(getattr(reply, "response", "") or "")).strip()
        prism = {name: getattr(reply, name) for name in PRISM_MODULES if getattr(reply, name, None) is not None}
        usage = Usage(estimate_tokens(query), estimate_tokens(text))  # the API reports no token counts
        if options.usage_sink is not None:
            options.usage_sink.add(usage)
        meta: dict[str, Any] = {"entrypoint": self.model}
        if prism:
            meta["prism_modules"] = sorted(prism)
            meta["prism"] = prism
        return ChatResult(content=text, usage=usage, model=self.id, finish_reason="stop", meta=meta)

    async def stream(self, messages: list[dict[str, Any]], options: CallOptions | None = None) -> AsyncIterator[str]:
        """The HTTP API is not streamed, so the whole reply arrives at once and is passed on in pieces."""
        result = await self.complete(messages, None, options)
        text = result.content
        for i in range(0, len(text), 60):
            yield text[i : i + 60]
