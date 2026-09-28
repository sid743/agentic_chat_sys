"""Tool calling for models that have no tool-calling API.

Some models take text and return text: Cortexa Core through its SDK, plain completion
endpoints, many small local models. The agents are written against the native
interface instead:

    reply = await model.complete(messages, tools)      # tools = OpenAI function schemas
    reply.tool_calls                                   # [ToolCall(id, name, arguments), ...]

`EmulatedToolsModel` wraps a text-only model and gives it that interface, so the
agents, the orchestrator and the tools do not change:

    messages + tool schemas
      -> one text conversation: the tool list and a strict reply format are added to
         the system prompt; earlier tool calls and tool results are written out as text
      -> the text model replies
      -> parse_reply(): JSON, <tool_call> tags, ReAct "Action:" lines, name(arg=...) calls
      -> validate(): known tool, required arguments present, types and enums coerced
      -> not usable? say exactly what was wrong and ask again (up to `max_repairs` times)
      -> ChatResult(tool_calls=[...])  or  ChatResult(content=<answer>)

Two strategies:

    emulated   the text model chooses the tools (above).
    delegate   a second model with native tool calling (`planner`) chooses the tools;
               the text model writes the answer from the tool results. For a text
               model that will not keep to a reply format.

This file needs only the standard library and base.py, so the two files can be
copied into another project. `TextFunctionModel` and `run_tools` below show that use.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .base import (
    CallOptions,
    ChatModel,
    ChatResult,
    LLMError,
    ToolCall,
    ToolsNotSupported,
    Usage,
    estimate_tokens,
    loads_lenient,
    strip_think,
)

ANSWER_KEYS = ("answer", "final", "final_answer", "response", "reply")
ARGUMENT_KEYS = ("arguments", "args", "parameters", "params", "input", "action_input", "tool_input")
NEXT_STEP = 'Next: call more tools, or give your answer. Use one of the two JSON forms under "How to reply".'
ANSWER_FROM_RESULTS = "Write your reply to the user now, using the tool results above."


# ============================================================================ tool definitions
@dataclass
class ToolDef:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def properties(self) -> dict[str, dict[str, Any]]:
        return self.parameters.get("properties") or {}

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required") or [])


def tool_defs(tools: list[dict[str, Any]]) -> dict[str, ToolDef]:
    """Accepts OpenAI tool schemas ({"type": "function", "function": {...}}) or bare
    {"name", "description", "parameters"} dicts."""
    defs: dict[str, ToolDef] = {}
    for tool in tools or []:
        spec = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = spec.get("name")
        if name:
            defs[name] = ToolDef(name, spec.get("description") or "", spec.get("parameters") or {})
    return defs


# ============================================================================ prompt
def _branches(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The non-null alternatives of a property schema (anyOf/oneOf/type lists)."""
    alts = schema.get("anyOf") or schema.get("oneOf")
    if alts:
        return [a for a in alts if a.get("type") != "null"]
    types = schema.get("type")
    if isinstance(types, list):
        return [{**schema, "type": t} for t in types if t != "null"]
    return [schema]


def _type_label(schema: dict[str, Any]) -> str:
    labels = []
    for branch in _branches(schema):
        if "enum" in branch:
            labels.append(" | ".join(json.dumps(v) for v in branch["enum"]))
        elif branch.get("type") == "array":
            labels.append(f"list of {_type_label(branch.get('items') or {})}")
        elif branch.get("format") == "date":
            labels.append("date (YYYY-MM-DD)")
        else:
            labels.append(str(branch.get("type") or "any"))
    return " | ".join(labels) or "any"


def signature(tool: ToolDef) -> str:
    required = set(tool.required)
    args = [f"{k}{'' if k in required else '?'}: {_type_label(p)}" for k, p in tool.properties.items()]
    lines = [f"- {tool.name}({', '.join(args)})"]
    if tool.description:
        lines.append(f"    {tool.description}")
    for key, prop in tool.properties.items():
        if prop.get("description"):
            lines.append(f"    {key}: {prop['description']}")
    return "\n".join(lines)


def tool_prompt(defs: dict[str, ToolDef]) -> str:
    first = next(iter(defs.values()))
    example_args = {k: "..." for k in first.required[:1]} or {}
    example = json.dumps({"tool_calls": [{"name": first.name, "arguments": example_args}]})
    catalog = "\n".join(signature(t) for t in defs.values())
    return (
        "### Tools\n"
        "You can call these tools. Arguments marked ? are optional.\n\n"
        f"{catalog}\n\n"
        "### How to reply\n"
        "Reply in exactly one of these two forms, with nothing before or after the JSON.\n\n"
        "To call tools (one or more; they run in order):\n"
        f"{example}\n\n"
        "To give your answer, once you have what you need:\n"
        '{"answer": "<your reply to the user, in markdown>"}\n\n'
        "- If the answer depends on company records or policy text, call a tool first. Do not answer those from memory.\n"
        "- Use only the tool names and argument names listed above.\n"
        '- Tool results come back in a message that starts with "Tool results". Then call more tools, or answer.'
    )


# ============================================================================ transcript
def _text(content: Any) -> str:
    if isinstance(content, list):  # multi-part content: keep the text parts
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def to_text_messages(
    messages: list[dict[str, Any]],
    defs: dict[str, ToolDef] | None = None,
    *,
    instructions: str | None = None,
) -> list[dict[str, Any]]:
    """Rewrite a native tool-calling conversation as plain system/user/assistant text.

    Assistant tool calls become the JSON the model is asked to write; tool results
    become one user message per batch, labelled with the tool name and call id."""
    out: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    results: list[str] = []

    def flush() -> None:
        if results:
            body = "Tool results:\n\n" + "\n\n".join(results)
            if defs:
                body += "\n\n" + NEXT_STEP
            out.append({"role": "user", "content": body})
            results.clear()

    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id") or ""
            name = message.get("name") or call_names.get(call_id, "tool")
            results.append(f"[{name}] (call {call_id})\n{_text(message.get('content'))}")
            continue
        flush()
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                fn = call.get("function") or {}
                arguments = fn.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = loads_lenient(arguments or "{}")
                    except ValueError:
                        pass
                call_names[call.get("id") or ""] = fn.get("name") or ""
                calls.append({"name": fn.get("name"), "arguments": arguments or {}})
            out.append({"role": "assistant", "content": json.dumps({"tool_calls": calls}, ensure_ascii=False)})
            continue
        out.append({"role": role, "content": _text(message.get("content"))})
    flush()

    if instructions:
        for message in out:
            if message["role"] == "system":
                message["content"] = f"{message['content'].rstrip()}\n\n{instructions}"
                break
        else:
            out.insert(0, {"role": "system", "content": instructions})
    return out


# ============================================================================ parsing
@dataclass
class ParsedReply:
    calls: list[ToolCall] = field(default_factory=list)  # valid calls only
    answer: str | None = None
    errors: list[str] = field(default_factory=list)  # fed back to the model on a repair
    notes: list[str] = field(default_factory=list)  # fixes applied silently (renames, coercions)
    form: str = "empty"  # json | tagged | react | python | answer | prose | empty


def json_values(text: str) -> list[Any]:
    """Every JSON object/array in the text: fenced blocks first, then the whole text,
    then each balanced {...} or [...] found by scanning."""
    fenced = re.findall(r"```(?:json|javascript|js)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    found: list[Any] = []
    for block in fenced:
        try:
            found.append(loads_lenient(block))
        except ValueError:
            found.extend(_scan(block))
    if found:
        return found
    try:
        value = loads_lenient(text)
        if isinstance(value, (dict, list)):
            return [value]
    except ValueError:
        pass
    return _scan(text)


MAX_PARSE_ATTEMPTS = 60


def _bracket_pairs(text: str) -> list[tuple[int, int]]:
    """Matched {...} / [...] spans, in one pass (linear, whatever the input).

    A quote only opens a string where a JSON key or value can start (after { [ , :),
    so an apostrophe or a quoted word in prose does not swallow the rest of the text."""
    pairs: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    closers = {"{": "}", "[": "]"}
    last = ""  # last non-space character outside strings
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if stack and ch in "\"'" and last in "{[,:":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i, last = j + 1, ch
            continue
        if ch in closers:
            stack.append((closers[ch], i))
        elif ch in "}]":
            if stack and stack[-1][0] == ch:
                pairs.append((stack.pop()[1], i))
            else:
                stack.clear()  # mismatched: start over from the next opener
        if not ch.isspace():
            last = ch
        i += 1
    return sorted(pairs)


def _scan(text: str) -> list[Any]:
    """Parse the outermost bracketed spans that are valid JSON; if an outer span is
    not, try the spans inside it."""
    values: list[Any] = []
    taken_until = -1
    attempts = 0
    for start, end in _bracket_pairs(text):
        if start <= taken_until:
            continue  # inside a span that already parsed
        if attempts >= MAX_PARSE_ATTEMPTS:
            break
        attempts += 1
        try:
            values.append(loads_lenient(text[start : end + 1]))
            taken_until = end
        except ValueError:
            pass
    return values


NOT_LITERAL = "(arguments that are not literal values)"


def _as_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return loads_lenient(value or "{}")
        except ValueError:
            return value  # left as-is; validation reports it
    return {} if value is None else value


# Keys that name a tool only in a reply that is a tool call ("tool") versus keys that
# also turn up in ordinary data ("name", "action", ...). The second kind must name a
# real tool exactly before a value is read as a call.
CLEAR_NAME_KEYS = ("tool", "tool_name")
AMBIGUOUS_NAME_KEYS = ("name", "action", "function", "call")


def _call_from(item: Any, defs: dict[str, ToolDef], *, strict: bool = False) -> tuple[str, Any] | None:
    """Recognise one tool call in the shapes models commonly produce. `strict` (JSON
    found inside a prose reply) accepts only names of real tools, spelled right."""
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("function"), dict):  # OpenAI shape
        fn = item["function"]
        if isinstance(fn.get("name"), str) and (not strict or resolve_name(fn["name"], defs, fuzzy=False)[0]):
            return fn["name"], _as_arguments(fn.get("arguments"))
    args_key = next((k for k in ARGUMENT_KEYS if k in item), None)
    for key in (*CLEAR_NAME_KEYS, *AMBIGUOUS_NAME_KEYS):
        name = item.get(key)
        if not isinstance(name, str) or not name.strip():
            continue
        exact = resolve_name(name, defs, fuzzy=False)[0] is not None
        final = _norm(name) in ("final_answer", "final", "answer")
        if strict and not (exact or final):
            continue
        if key in AMBIGUOUS_NAME_KEYS and not (exact or final or (args_key and key == "name")):
            continue
        return name, _as_arguments(item.get(args_key)) if args_key else {}
    if len(item) == 1:  # {"search_policies": {...}}
        (key, value), = item.items()
        if resolve_name(key, defs, fuzzy=False)[0] and isinstance(value, (dict, str, type(None))):
            return key, _as_arguments(value)
    return None


def _dumps(value: Any, limit: int = 200) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)[:limit]


def _answer_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _interpret(
    value: Any, defs: dict[str, ToolDef], *, strict: bool = False
) -> tuple[list[tuple[str, Any]], str | None, list[str], bool]:
    """-> (calls, answer, errors, recognised)"""
    if isinstance(value, list):
        calls = [_call_from(item, defs, strict=strict) for item in value]
        if value and all(calls):
            return [c for c in calls if c], None, [], True
        return [], None, [], False
    if not isinstance(value, dict):
        return [], None, [], False
    for key in ("tool_calls", "calls", "tools", "actions"):
        items = value.get(key)
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, list) and items:
            calls, errors = [], []
            for item in items:
                call = _call_from(item, defs, strict=strict)
                if call:
                    calls.append(call)
                elif not strict:
                    errors.append(f"Could not read this tool call: {_dumps(item)}")
            if calls or errors:
                return calls, None, errors, True
    call = _call_from(value, defs, strict=strict)
    if call and _norm(call[0]) in ("final_answer", "final", "answer"):  # {"action": "Final Answer", "action_input": ...}
        return [], _answer_text(call[1]), [], True
    if call:
        return [call], None, [], True
    # Inside prose only this module's own answer keys count; "response" or "reply" in a
    # data sample is just data.
    for key in ANSWER_KEYS[:3] if strict else ANSWER_KEYS:
        if key in value and value[key] not in (None, ""):
            return [], _answer_text(value[key]), [], True
    return [], None, [], False


def _only_json(text: str) -> bool:
    """True when the reply is a JSON value and nothing else (a fenced block counts)."""
    return _json_body(text) is not None


def _json_body(text: str) -> Any:
    fence = re.fullmatch(r"```(?:json|javascript|js)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    try:
        value = loads_lenient(fence.group(1) if fence else text)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


# A broken attempt at a tool call: never show that to a user.
_BROKEN_CALL = re.compile(r"[\"']tool_calls[\"']\s*:")
_BROKEN_CALL_START = re.compile(r"^\s*(?:```(?:json)?\s*)?[\[{].*[\"'](?:tool|arguments|name)[\"']\s*:", re.DOTALL)
# A JSON answer cut off before its closing quote (token limit): salvage the text.
_CUT_ANSWER = re.compile(r"^\s*(?:```(?:json)?\s*)?\{\s*[\"']?(?:answer|final|final_answer)[\"']?\s*:\s*\"(.*)$", re.DOTALL)

_TAGGED = re.compile(r"<(tool_call|function_call|tool_use)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE)
_FUNCTION_TAG = re.compile(r"<function=([\w.\-]+)>\s*(.*?)\s*</function>", re.DOTALL | re.IGNORECASE)  # Llama style
_REACT_ACTION = re.compile(r"^[ \t]*Action[ \t]*:[ \t]*`?([\w.\-]+)`?[ \t]*$", re.MULTILINE | re.IGNORECASE)
_REACT_INPUT = re.compile(
    r"\s*Action[ \t]*Input[ \t]*:[ \t]*(.*?)(?=^[ \t]*(?:Observation|Thought|Action)[ \t]*:|\Z)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)
_REACT_FINAL = re.compile(r"Final\s*Answer\s*:\s*(.+)", re.DOTALL | re.IGNORECASE)
_PY_CALL = re.compile(r"^\s*(?:`{1,3}(?:python|py)?\s*)?([A-Za-z_][\w.\-]*)\((.*)\)\s*;?\s*`{0,3}\s*$", re.DOTALL)


def _python_call(text: str, defs: dict[str, ToolDef], *, fuzzy: bool) -> tuple[str, Any] | None:
    """`search_policies(query="carry forward", top_k=3)` -> ("search_policies", {...})."""
    match = _PY_CALL.match(text)
    if not match:
        return None
    name = resolve_name(match.group(1), defs, fuzzy=fuzzy)[0]
    if name is None:
        return None
    try:
        node = ast.parse(f"_({match.group(2)})", mode="eval").body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if not isinstance(node, ast.Call):
        return None
    try:
        args: dict[str, Any] = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        for key, arg in zip(defs[name].properties, node.args):  # positional -> property order
            args.setdefault(key, ast.literal_eval(arg))
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return match.group(1), NOT_LITERAL
    if not all(_json_value(v) for v in args.values()):
        return match.group(1), NOT_LITERAL  # e.g. query=... or a set
    return match.group(1), args


def _json_value(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
        return not isinstance(value, tuple)
    except (TypeError, ValueError):
        return False


def _react_calls(text: str, defs: dict[str, ToolDef]) -> list[tuple[str, Any]]:
    calls = []
    for action in _REACT_ACTION.finditer(text):
        name = action.group(1)
        given = _REACT_INPUT.match(text, action.end())
        if given is None:
            if resolve_name(name, defs, fuzzy=False)[0] is None:
                continue  # "Action: escalate" in prose, with no Action Input: not a call
            calls.append((name, {}))
            continue
        body = given.group(1).strip()
        values = json_values(body) if body else []
        calls.append((name, values[0] if values else (body or {})))
    return calls


def _salvage_cut_answer(text: str) -> str | None:
    match = _CUT_ANSWER.match(text)
    if not match:
        return None
    body = re.sub(r"\"?\s*\}?\s*(?:```)?\s*$", "", match.group(1))
    try:
        return json.loads(f'"{body}"', strict=False)
    except json.JSONDecodeError:
        return body.replace("\\n", "\n").replace('\\"', '"')


def parse_reply(text: str, defs: dict[str, ToolDef]) -> ParsedReply:
    """Find tool calls or an answer in a model's text reply, and validate the calls."""
    text = strip_think(text or "").strip()
    if not text:
        return ParsedReply(errors=["The reply was empty."], form="empty")

    raw: list[tuple[str, Any]] = []
    answer: str | None = None
    form = ""
    errors: list[str] = []

    tagged = _TAGGED.findall(text)
    function_tags = _FUNCTION_TAG.findall(text)
    if tagged or function_tags:
        form = "tagged"
        for _, body in tagged:
            values = json_values(body)
            calls, _, value_errors, ok = _interpret(values[0], defs) if values else ([], None, [], False)
            if ok and (calls or value_errors):
                raw.extend(calls)
                errors.extend(value_errors)
            else:
                errors.append(f"Could not read this tool call: {body[:200]}")
        for name, body in function_tags:
            values = json_values(body) if body.strip() else [{}]
            raw.append((name, values[0] if values else body))
    else:
        whole = _json_body(text)
        strict = whole is None  # JSON found inside prose is read strictly
        unrecognised = []
        recognised = False
        for value in [whole] if whole is not None else json_values(text):
            calls, value_answer, value_errors, ok = _interpret(value, defs, strict=strict)
            if not ok:
                unrecognised.append(value)
                continue
            recognised = True
            raw.extend(calls)
            errors.extend(value_errors)
            if value_answer is not None and answer is None:
                answer = value_answer
        if raw or (recognised and errors):
            form = "json"
        elif answer is not None:
            form = "answer"
            if strict and not text.lstrip().startswith(("{", "`")):
                answer = None  # an answer object quoted inside prose: the prose is the reply
        elif whole is not None:
            form = "json"
            keys = sorted(whole) if isinstance(whole, dict) else []
            errors.append(
                "The JSON was not a tool call or an answer"
                + (f" (it had the keys {', '.join(keys)})." if keys else ".")
            )
        elif (cut := _salvage_cut_answer(text)) is not None:
            form, answer = "answer", cut
        elif _BROKEN_CALL.search(text) or _BROKEN_CALL_START.match(text):
            form = "json"
            errors.append("The tool call was not valid JSON. Check quotes, commas and brackets.")

    if not raw and answer is None and not errors:
        react = _react_calls(text, defs)
        final = _REACT_FINAL.search(text)
        body = re.sub(r"^```(?:python|py)?\s*|\s*```$", "", text).strip()
        whole_call = _python_call(body, defs, fuzzy=True)
        line_calls = [c for c in (_python_call(ln, defs, fuzzy=False) for ln in text.splitlines()) if c]
        if react:
            form = "react"
            raw.extend(react)
        elif final:
            form, answer = "react", final.group(1).strip()
        elif whole_call:
            form = "python"
            raw.append(whole_call)
        elif line_calls:
            form = "python"
            raw.extend(line_calls)
        else:
            form, answer = "prose", text

    parsed = ParsedReply(answer=answer if not raw else None, errors=errors, form=form)
    if raw and answer is not None:
        parsed.notes.append("The reply had both tool calls and an answer; the tool calls were used.")
    for name, arguments in raw:
        call, call_errors, notes = validate(name, arguments, defs)
        parsed.notes.extend(notes)
        parsed.errors.extend(call_errors)
        if call:
            parsed.calls.append(call)
    return parsed


# ============================================================================ validation
def _norm(name: str) -> str:
    name = re.sub(r"^(functions|tools|tool)[.:]", "", name.strip().strip("`\"'"), flags=re.IGNORECASE)
    name = re.sub(r"\(\)$", "", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)  # camelCase -> camel_Case
    return re.sub(r"[\s\-]+", "_", name).lower()


def _singular(name: str) -> str:
    def one(word: str) -> str:
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        return word

    return "_".join(one(w) for w in name.split("_"))


def _distance(a: str, b: str) -> int:
    """Levenshtein distance (tool names are short)."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def resolve_name(name: str, defs: dict[str, ToolDef], *, fuzzy: bool = True) -> tuple[str | None, str | None]:
    """-> (tool name, note).

    Exact, then case/separator-insensitive. With `fuzzy`, also a near miss - but only
    with the same leading verb, and only when exactly one tool fits, so that
    'disapprove_leave_request' can never become 'approve_leave_request' and
    'get_leave_request' never becomes 'reject_leave_request'."""
    if name in defs:
        return name, None
    wanted = _norm(str(name))
    normalised = {_norm(t): t for t in defs}
    if wanted in normalised:
        return normalised[wanted], None
    if not fuzzy or not wanted:
        return None, None
    verb = wanted.split("_", 1)[0]
    same_verb = [n for n in normalised if n.split("_", 1)[0] == verb]
    fits = [n for n in same_verb if _singular(n) == _singular(wanted)] or [
        n for n in same_verb if _distance(n, wanted) <= 2
    ]
    if len(fits) == 1:
        return normalised[fits[0]], f"read tool '{name}' as '{normalised[fits[0]]}'"
    return None, None


def _coerce(value: Any, schema: dict[str, Any]) -> tuple[bool, Any]:
    """Fit a value to a property schema. Only safe conversions ("5" -> 5, "true" -> True)."""
    branches = _branches(schema) or [{}]
    enums = [v for b in branches for v in (b.get("enum") or [])]
    if enums:
        if value in enums:
            return True, value
        if isinstance(value, str):
            for option in enums:
                if isinstance(option, str) and option.lower() == value.strip().lower():
                    return True, option
        return False, value
    types = [b.get("type") for b in branches if b.get("type")]
    if not types:
        return True, value
    for kind in types:
        if kind == "string":
            if isinstance(value, str):
                return True, value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True, str(value)
        elif kind == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return True, value
            if isinstance(value, float) and value.is_integer():
                return True, int(value)
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
                return True, int(value)
        elif kind == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True, value
            if isinstance(value, str):
                try:
                    return True, float(value)
                except ValueError:
                    pass
        elif kind == "boolean":
            if isinstance(value, bool):
                return True, value
            if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1", "false", "no", "0"):
                return True, value.strip().lower() in ("true", "yes", "1")
            if value in (0, 1):
                return True, bool(value)
        elif kind == "array":
            if isinstance(value, list):
                return True, value
            if isinstance(value, str):
                try:
                    parsed = loads_lenient(value)
                    if isinstance(parsed, list):
                        return True, parsed
                except ValueError:
                    pass
            return True, [value]
        elif kind == "object":
            if isinstance(value, dict):
                return True, value
            if isinstance(value, str):
                try:
                    parsed = loads_lenient(value)
                    if isinstance(parsed, dict):
                        return True, parsed
                except ValueError:
                    pass
        else:
            return True, value
    return False, value


def validate(name: str, arguments: Any, defs: dict[str, ToolDef]) -> tuple[ToolCall | None, list[str], list[str]]:
    """-> (call or None, errors for the model, notes for the trace)."""
    tool_name, note = resolve_name(str(name), defs)
    notes = [note] if note else []
    if tool_name is None:
        return None, [f"There is no tool named '{name}'. Available tools: {', '.join(defs)}."], notes
    tool = defs[tool_name]
    if not isinstance(arguments, dict):
        shown = "values that are not literals" if arguments == NOT_LITERAL else _dumps(arguments, 80)
        return None, [f"The arguments for {tool_name} must be a JSON object with literal values, not {shown}."], notes

    props = tool.properties
    by_norm = {_norm(k): k for k in props}
    clean: dict[str, Any] = {}
    errors: list[str] = []
    ordered = sorted(arguments.items(), key=lambda kv: kv[0] not in props)  # exact names first
    for key, value in ordered:
        target = key if key in props else by_norm.get(_norm(str(key)))
        if target is not None and target != key and target in clean:
            notes.append(f"{tool_name}: ignored '{key}' because '{target}' was also given")
            continue
        if target is None:
            if "properties" in tool.parameters or tool.parameters.get("additionalProperties") is False:
                notes.append(f"{tool_name}: ignored unknown argument '{key}'")
                continue
            target = key
        if target != key:
            notes.append(f"{tool_name}: read argument '{key}' as '{target}'")
        if value is None or (
            isinstance(value, str) and value.strip().lower() in ("none", "null") and target not in tool.required
        ):
            continue  # null (or "None") means "not given": the tool's default applies
        if not _json_value(value):
            errors.append(f"{tool_name}: '{target}' must be a literal JSON value.")
            continue
        ok, coerced = _coerce(value, props.get(target, {}))
        if not ok:
            errors.append(f"{tool_name}: '{target}' should be {_type_label(props[target])}, got {_dumps(value, 80)}.")
            continue
        clean[target] = coerced
    missing = [r for r in tool.required if r not in clean]
    if missing:
        errors.append(f"{tool_name}: missing required argument(s) {', '.join(missing)}. Signature: {signature(tool).splitlines()[0][2:]}")
    if errors:
        return None, errors, notes
    return ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=tool_name, arguments=clean), [], notes


def repair_prompt(parsed: ParsedReply, *, prose: bool = False) -> str:
    if prose:
        lines = [
            "Your reply was not in one of the two required JSON forms.",
            "If the question needs company records or policy text, call a tool now.",
            'If it does not, send the same reply as {"answer": "..."}.',
        ]
    else:
        lines = ["Your last reply could not be used:", *[f"- {e}" for e in parsed.errors]]
        if parsed.calls:
            lines.append(f"(These calls were fine: {', '.join(c.name for c in parsed.calls)}. Send them again with the fixed ones.)")
    lines += [
        "",
        "Reply with ONLY one JSON object, either",
        '{"tool_calls": [{"name": "<tool>", "arguments": {...}}]}',
        "or",
        '{"answer": "<your reply to the user>"}',
    ]
    return "\n".join(lines)


# ============================================================================ the model wrapper
class EmulatedToolsModel(ChatModel):
    """Gives a text-only ChatModel the native tool-calling interface.

    `reply.meta` records how it went: tool_strategy, repairs, reply_forms, tool_notes
    (plus whatever the inner model reported, e.g. Cortexa's prism modules)."""

    def __init__(
        self,
        inner: ChatModel,
        *,
        max_repairs: int = 2,
        retry_prose_before_tools: bool = True,
        planner: ChatModel | None = None,
    ) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.max_repairs = max(0, int(max_repairs))
        self.retry_prose_before_tools = retry_prose_before_tools
        self.planner = planner
        self.native_tools = True

    @property
    def strategy(self) -> str:
        return "delegate" if self.planner is not None else "emulated"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: CallOptions | None = None,
    ) -> ChatResult:
        options = options or CallOptions()
        if not tools:
            return await self.inner.complete(to_text_messages(messages), None, options)
        defs = tool_defs(tools)
        if not defs:
            return await self.inner.complete(to_text_messages(messages), None, options)
        if self.planner is not None:
            try:
                plan = await self.planner.complete(messages, tools, options)
            except (ToolsNotSupported, LLMError) as exc:
                # The planner is down or rate limited: the text model picks the tools itself.
                result = await self._emulate(messages, defs, options)
                result.meta["planner_error"] = f"{self.planner.id}: {exc}"
                return result
            return await self._delegate(messages, plan, options)
        return await self._emulate(messages, defs, options)

    def stream(self, messages: list[dict[str, Any]], options: CallOptions | None = None) -> AsyncIterator[str]:
        return self.inner.stream(to_text_messages(messages), options)

    # ------------------------------------------------------------------ strategies
    async def _emulate(self, messages: list[dict[str, Any]], defs: dict[str, ToolDef], options: CallOptions) -> ChatResult:
        convo = to_text_messages(messages, defs, instructions=tool_prompt(defs))
        tools_used = any(m.get("role") == "tool" for m in messages)
        json_options = replace(options, json_mode=True)
        usage = Usage()
        meta: dict[str, Any] = {"tool_strategy": "emulated", "repairs": 0, "reply_forms": [], "tool_notes": []}
        prose_retried = False
        parsed = ParsedReply()
        last_answer: str | None = None  # a prose answer that was sent back once, kept in case nothing better comes

        for attempt in range(self.max_repairs + 1):
            reply = await self.inner.complete(convo, None, json_options)
            usage.add(reply.usage)
            for key, value in reply.meta.items():
                meta.setdefault(key, value)
            try:
                parsed = parse_reply(reply.content, defs)
            except Exception as exc:  # noqa: BLE001 - a reply that breaks the parser is just unusable
                parsed = ParsedReply(errors=[f"The reply could not be read ({type(exc).__name__})."], form="unreadable")
            meta["reply_forms"].append(parsed.form)
            meta["tool_notes"].extend(parsed.notes)

            if parsed.calls and not parsed.errors:
                return self._result("", parsed.calls, usage, meta)
            if parsed.answer is not None and not parsed.errors:
                retry_prose = (
                    parsed.form == "prose"
                    and self.retry_prose_before_tools
                    and not tools_used
                    and not prose_retried
                )
                if not retry_prose or attempt == self.max_repairs:
                    return self._result(parsed.answer, [], usage, meta)
                prose_retried = True
                last_answer = parsed.answer
            if attempt == self.max_repairs:
                break
            meta["repairs"] += 1
            convo = [
                *convo,
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": repair_prompt(parsed, prose=parsed.form == "prose")},
            ]

        # Out of repairs: use what is usable, or return nothing and let the caller ask
        # for a final answer without tools (the agent runtime does exactly that).
        meta["gave_up"] = True
        if parsed.calls:
            return self._result("", parsed.calls, usage, meta)
        if parsed.answer is not None:
            return self._result(parsed.answer, [], usage, meta)
        if last_answer is not None:
            return self._result(last_answer, [], usage, meta)
        return self._result("", [], usage, meta)

    async def _delegate(self, messages: list[dict[str, Any]], plan: ChatResult, options: CallOptions) -> ChatResult:
        assert self.planner is not None
        meta: dict[str, Any] = {"tool_strategy": "delegate", "planner": self.planner.id}
        if plan.tool_calls:
            return ChatResult(content="", tool_calls=plan.tool_calls, usage=plan.usage, model=self.id,
                              finish_reason="tool_calls", meta=meta)
        # The planner wants no more tools: the text model writes the answer.
        convo = to_text_messages(messages)
        if any(m.get("role") == "tool" for m in messages):
            convo.append({"role": "user", "content": ANSWER_FROM_RESULTS})
        reply = await self.inner.complete(convo, None, options)
        return ChatResult(content=reply.content, usage=reply.usage, model=self.id, finish_reason="stop",
                          meta={**reply.meta, **meta})

    def _result(self, content: str, calls: list[ToolCall], usage: Usage, meta: dict[str, Any]) -> ChatResult:
        return ChatResult(
            content=content,
            tool_calls=calls,
            usage=usage,
            model=self.id,
            finish_reason="tool_calls" if calls else "stop",
            meta=meta,
        )


# ============================================================================ standalone use
class TextFunctionModel(ChatModel):
    """Any `async (messages) -> str` function as a ChatModel, e.g. a raw SDK call:

        async def ask(messages):
            reply = await client.interactions.create(query=flatten(messages))
            return reply.response

        model = EmulatedToolsModel(TextFunctionModel(ask, "cortexa/event"))
    """

    native_tools = False

    def __init__(self, fn: Callable[[list[dict[str, Any]]], Awaitable[str]], model_id: str = "custom/text") -> None:
        self.fn = fn
        self.provider, _, self.model = model_id.partition("/")

    async def complete(self, messages, tools=None, options=None) -> ChatResult:  # type: ignore[override]
        if tools:
            raise ToolsNotSupported(f"{self.id} is text-only; wrap it in EmulatedToolsModel")
        text = await self.fn(messages)
        usage = Usage(estimate_tokens(json.dumps(messages, default=str)), estimate_tokens(text))
        if options and options.usage_sink is not None:
            options.usage_sink.add(usage)
        return ChatResult(content=text or "", usage=usage, model=self.id, finish_reason="stop")

    async def stream(self, messages, options=None) -> AsyncIterator[str]:  # type: ignore[override]
        yield (await self.complete(messages, None, options)).content


async def run_tools(
    model: ChatModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    functions: dict[str, Callable[..., Any]],
    *,
    max_steps: int = 6,
) -> tuple[str, list[dict[str, Any]]]:
    """A minimal agent loop: model -> tools -> model until it answers.
    `functions` maps tool names to plain or async Python functions taking keyword
    arguments. Returns (answer, full message list)."""
    import inspect

    messages = list(messages)
    for _ in range(max_steps):
        reply = await model.complete(messages, tools)
        if not reply.tool_calls and reply.content.strip():
            return reply.content.strip(), messages
        if not reply.tool_calls:
            break  # nothing usable: ask for a plain answer below
        messages.append({
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, default=str)}}
                for c in reply.tool_calls
            ],
        })
        for call in reply.tool_calls:
            try:
                output = functions[call.name](**call.arguments)
                if inspect.isawaitable(output):
                    output = await output
            except Exception as exc:  # noqa: BLE001 - the model sees the error and can recover
                output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output, default=str)})
    messages.append({"role": "user", "content": "Stop calling tools and give your final answer now."})
    final = await model.complete(messages, None)
    return final.content.strip(), messages
