"""`python -m agentic_core cortexa-check`: is Cortexa Core usable from here?

Checks, in order, and stops at the first hard failure:

1. the SDK wheel is installed;
2. CORTEXA_APP_ID, CORTEXA_CLIENT_SECRET and CORTEXA_CORE_BASE_URL are set;
3. the heartbeat answers (informational: a server without it is not a failure);
4. a plain question gets an answer             <- proves URL + credentials work;
5. a JSON-only question gets parseable JSON     <- whether the router can plan on Cortexa;
6. a tool-calling round trip through the emulation layer (llm/tool_emulation.py):
   does Cortexa ask for a sample tool, and use its result in the answer?
   <- decides between CORTEXA_TOOL_MODE=emulated and delegate.

Prints only masked credentials. Exit code 0 = usable, 1 = not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .base import CallOptions, LLMError, parse_json_object
from .cortexa import ENTRYPOINTS, INSTALL_HINT, CortexaChatModel, _default_client
from .tool_emulation import EmulatedToolsModel

Out = Callable[[str], None]

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_policy",
        "description": "Look up a company HR policy by topic. Returns the policy text.",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "What to look up"}},
            "required": ["topic"],
        },
    },
}
PROBE_MESSAGES = [
    {"role": "system", "content": "You are an HR assistant for Acme. Company policies are only available "
                                  "through tools: never answer a policy question from memory."},
    {"role": "user", "content": "What is the notice period for resignation at our company?"},
]
PROBE_RESULT = {"ok": True, "policy": "HR-POL-003 section 2: the notice period is 60 days for permanent employees."}


def mask(value: str, *, secret: bool = False) -> str:
    if not value:
        return "(not set)"
    if secret or len(value) <= 8:
        return f"set ({len(value)} characters)"
    return f"{value[:4]}...{value[-2:]} ({len(value)} characters)"


async def run_check(provider: Any, *, entrypoint: str = "event", timeout: float = 90.0,
                    client_factory: Callable[..., Any] | None = None, out: Out = print) -> int:
    ok, warn, bad = "  [ok]  ", "  [warn]", "  [FAIL]"
    factory = client_factory or _default_client

    out("Cortexa Core check")
    # 1. SDK
    if client_factory is None:
        try:
            import cortexa_enterprise_sdk as sdk
        except ImportError:
            out(f"{bad} {INSTALL_HINT}")
            return 1
        out(f"{ok} SDK installed (cortexa_enterprise_sdk {getattr(sdk, '__version__', '?')})")

    # 2. configuration
    app_id, secret, base_url = provider.app_id or "", provider.api_key or "", (provider.base_url or "").rstrip("/")
    out(f"        CORTEXA_APP_ID         {mask(app_id)}")
    out(f"        CORTEXA_CLIENT_SECRET  {mask(secret, secret=True)}")
    out(f"        CORTEXA_CORE_BASE_URL  {base_url or '(not set)'}")
    missing = [n for n, v in (("CORTEXA_APP_ID", app_id), ("CORTEXA_CLIENT_SECRET", secret),
                              ("CORTEXA_CORE_BASE_URL", base_url)) if not v or v == "user_provided"]
    if missing:
        out(f"{bad} missing in .env: {', '.join(missing)}")
        return 1
    if entrypoint not in ENTRYPOINTS:
        out(f"{bad} unknown entrypoint '{entrypoint}' (use {', '.join(ENTRYPOINTS)})")
        return 1
    model = CortexaChatModel(provider=provider.name, model=entrypoint, app_id=app_id, client_secret=secret,
                             base_url=base_url, timeout=timeout, max_retries=1, client_factory=factory)

    # 3. heartbeat (informational)
    try:
        async with factory(app_id, secret, base_url, 15.0, 0) as client:
            beat = await client.interactions.heartbeat()
        out(f"{ok} heartbeat: {getattr(beat, 'status', beat)}")
    except Exception as exc:  # noqa: BLE001 - any failure here is reported, not raised
        if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
            out(f"{bad} heartbeat: {model._friendly(exc)}")
            return 1
        out(f"{warn} heartbeat did not answer ({type(exc).__name__}); trying a real question anyway")

    # 4. a plain question
    started = time.monotonic()
    try:
        reply = await model.complete([{"role": "user", "content": "Reply with the single word: pong"}])
    except LLMError as exc:
        out(f"{bad} {entrypoint}: {exc}")
        return 1
    seconds = time.monotonic() - started
    out(f"{ok} {entrypoint} answered in {seconds:.1f}s: {reply.content[:120]!r}")
    if reply.meta.get("prism_modules"):
        out(f"        modules reported: {', '.join(reply.meta['prism_modules'])}")

    # 5. JSON discipline, which the router and the agents' tool calls rely on
    try:
        reply = await model.complete(
            [{"role": "system", "content": 'Answer with a JSON object of the form {"ok": true, "word": "<any word>"}.'},
             {"role": "user", "content": "Go."}],
            None,
            CallOptions(json_mode=True),
        )
        data = parse_json_object(reply.content)
    except LLMError as exc:
        data, reply = None, None
        out(f"{warn} JSON test call failed: {exc}")
    if data is not None:
        out(f"{ok} JSON replies parse, so the router can plan on Cortexa")
    elif reply is not None:
        out(f"{warn} reply was not JSON: {reply.content[:120]!r}")
        out("        The router will fall back to its built-in rules. For LLM routing set")
        out("        AGENT_ROUTER_MODEL=gemini/gemini-3.1-flash-lite (or another JSON-reliable model).")

    # 6. tool calling, through the same emulation layer the agents use
    advice = await _tool_probe(model, provider, out)

    out(f"\nUsable. Pick cortexa/{entrypoint} in the model menu, or run:")
    out(f'  python -m agentic_core chat "What is our parental leave policy?" --model cortexa/{entrypoint}')
    if advice:
        out(f"\nRecommended: {advice}")
    return 0


async def _tool_probe(model: CortexaChatModel, provider: Any, out: Out) -> str:
    """One tool round trip. Returns advice for .env ("" when nothing needs changing)."""
    ok, warn = "  [ok]  ", "  [warn]"
    mode = getattr(provider, "tool_mode", "emulated") or "emulated"
    planner = getattr(provider, "tool_planner", "") or ""
    out(f"        tool mode: {mode}" + (f" (planner {planner})" if mode == "delegate" else ""))
    emulated = EmulatedToolsModel(model, max_repairs=getattr(provider, "tool_repairs", 2))
    delegate_advice = (
        "CORTEXA_TOOL_MODE=delegate and CORTEXA_TOOL_PLANNER=gemini/gemini-3.1-flash-lite "
        "(Gemini picks the tools, Cortexa writes the answers)"
    )
    try:
        first = await emulated.complete(PROBE_MESSAGES, [PROBE_TOOL])
    except LLMError as exc:
        out(f"{warn} tool test call failed: {exc}")
        return ""
    repairs = first.meta.get("repairs", 0)
    if not first.tool_calls:
        out(f"{warn} Cortexa answered without calling the tool: {first.content[:120]!r}")
        out("        With CORTEXA_TOOL_MODE=emulated its agents would answer from memory, not your data.")
        return "" if mode == "delegate" else delegate_advice
    call = first.tool_calls[0]
    fixed = f" after {repairs} repair(s)" if repairs else ""
    out(f"{ok} Cortexa called a tool{fixed}: {call.name}({json.dumps(call.arguments)})")

    history = [
        *PROBE_MESSAGES,
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}]},
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(PROBE_RESULT)},
    ]
    try:
        second = await emulated.complete(history, [PROBE_TOOL])
    except LLMError as exc:
        out(f"{warn} follow-up call failed: {exc}")
        return ""
    repairs += second.meta.get("repairs", 0)
    if second.tool_calls:
        out(f"{warn} it asked for another tool instead of answering: {second.tool_calls[0].name}")
    elif "60" in second.content:
        out(f"{ok} it answered from the tool result: {second.content[:100]!r}")
    else:
        out(f"{warn} its answer did not use the tool result: {second.content[:100]!r}")
        return "" if mode == "delegate" else delegate_advice
    if repairs:
        out(f"        {repairs} repair(s) in total: each costs one extra Cortexa call, so turns are slower.")
    if mode != "emulated" and not second.tool_calls:
        return "CORTEXA_TOOL_MODE=emulated (Cortexa handles tool calls itself; no planner needed)"
    return ""
