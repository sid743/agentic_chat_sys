"""Provider-neutral chat model interface."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Who and what the current turn is about, for providers that can tag their calls with it
# (Cortexa takes session_id / user_id / trace_id). Set once per turn by the orchestrator.
CALL_SCOPE: ContextVar[dict[str, str] | None] = ContextVar("call_scope", default=None)


class LLMError(RuntimeError):
    """A provider call failed. The message is safe to show to end users."""


class ToolsNotSupported(LLMError):
    """The model/provider rejected native tool calling."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage | None) -> None:
        if other is None:
            return
        self.input_tokens += other.input_tokens or 0
        self.output_tokens += other.output_tokens or 0


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str | None = None
    # Provider-specific extras worth recording (e.g. Cortexa prism modules).
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallOptions:
    """Per-call options. `context` is only read by the offline mock model and is
    never sent to real providers."""

    purpose: str = "chat"  # route | agent | synthesize | title | direct
    agent_id: str | None = None
    json_mode: bool = False
    temperature: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    usage_sink: Usage | None = None


class ChatModel(ABC):
    provider: str = ""
    model: str = ""
    native_tools: bool = True

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: CallOptions | None = None,
    ) -> ChatResult: ...

    @abstractmethod
    def stream(self, messages: list[dict[str, Any]], options: CallOptions | None = None) -> AsyncIterator[str]: ...


# ----------------------------------------------------------------------------- helpers
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks some local reasoning models emit."""
    if not text or "<think" not in text.lower():
        return text or ""
    return _THINK_RE.sub("", text).strip()


class ThinkFilter:
    """Streaming filter that drops <think>...</think> sections from content deltas."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buffer = ""
        self.inside = False

    def feed(self, chunk: str) -> str:
        self.buffer += chunk
        out = []
        while self.buffer:
            if self.inside:
                idx = self.buffer.find(self.CLOSE)
                if idx == -1:
                    # keep a tail that may hold a partial closing tag
                    self.buffer = self.buffer[-(len(self.CLOSE) - 1):]
                    break
                self.buffer = self.buffer[idx + len(self.CLOSE):]
                self.inside = False
                continue
            idx = self.buffer.find(self.OPEN)
            if idx == -1:
                safe = len(self.buffer)
                for k in range(len(self.OPEN) - 1, 0, -1):
                    if self.buffer.endswith(self.OPEN[:k]):
                        safe = len(self.buffer) - k
                        break
                out.append(self.buffer[:safe])
                self.buffer = self.buffer[safe:]
                break
            out.append(self.buffer[:idx])
            self.buffer = self.buffer[idx + len(self.OPEN):]
            self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buffer
        self.buffer = ""
        return rest


_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_PY_LITERAL = re.compile(r"\b(True|False|None)\b")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _fix_bare_json(segment: str) -> str:
    segment = _TRAILING_COMMA.sub(r"\1", segment)
    return _PY_LITERAL.sub(lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)], segment)


def _outside_strings(text: str, fix: Any) -> str:
    """Apply `fix` to the parts of `text` that are not inside double-quoted strings."""
    out, buf, i, n = [], [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            out.append(fix("".join(buf)))
            buf = []
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        buf.append(ch)
        i += 1
    out.append(fix("".join(buf)))
    return "".join(out)


def _json_compatible(value: Any, depth: int = 0) -> bool:
    if depth > 100:
        return False
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_compatible(v, depth + 1) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_compatible(v, depth + 1) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def loads_lenient(text: str) -> Any:
    """json.loads that also takes what models commonly get wrong: raw newlines inside
    strings, trailing commas, Python literals (True/None, single-quoted dicts) and smart
    quotes used as delimiters. Repairs are tried from least to most invasive, and
    only outside string values, so a string's content is never changed when the text
    parses without that repair. Raises ValueError when nothing works."""
    if not isinstance(text, str):
        raise ValueError("not a string")  # noqa: TRY004 - callers handle ValueError only
    text = text.strip()
    smart = text.translate(_SMART_QUOTES)
    for candidate in dict.fromkeys((text, _outside_strings(text, _fix_bare_json), _outside_strings(smart, _fix_bare_json))):
        try:
            return json.loads(candidate, strict=False)
        except (json.JSONDecodeError, RecursionError):
            pass
    import ast

    for candidate in dict.fromkeys((text, smart)):  # a Python literal: {'a': 'b', 'c': None,}
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if _json_compatible(value):  # not Ellipsis, sets, bytes, tuples
            return value
    raise ValueError("not valid JSON")


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object in a model reply."""
    if not text:
        return None
    text = strip_think(text).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    for candidate in candidates:
        try:
            value = loads_lenient(candidate)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        start = candidate.find("{")
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = loads_lenient(candidate[start : i + 1])
                            if isinstance(value, dict):
                                return value
                        except ValueError:
                            pass
                        break
            start = candidate.find("{", start + 1)
    return None


def estimate_tokens(value: Any) -> int:
    if not value:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    return max(1, len(value) // 4)
