"""Tool registry: typed arguments (pydantic) -> OpenAI tool schemas + safe execution."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from ..agents.context import ToolContext
from ..db.models import AuditLog
from ..db.session import session_scope

log = logging.getLogger(__name__)


class NoArgs(BaseModel):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Any]
    kind: str = "data"  # data | action | retrieval

    def openai_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        schema["type"] = "object"
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }

    def prompt_signature(self) -> str:
        props = self.args_model.model_json_schema().get("properties", {})
        required = set(self.args_model.model_json_schema().get("required", []))
        args = []
        for key, prop in props.items():
            typ = prop.get("type") or next((a.get("type") for a in prop.get("anyOf", []) if a.get("type") != "null"), "any")
            args.append(f"{key}{'' if key in required else '?'}: {typ}")
        return f"{self.name}({', '.join(args)}) - {self.description}"


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, args_model: type[BaseModel] = NoArgs, kind: str = "data"):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = ToolSpec(name=name, description=description, args_model=args_model, handler=func, kind=kind)
        return func

    return decorator


def audit(
    ctx: ToolContext,
    session,
    action: str,
    *,
    target_type: str = "",
    target_id: str = "",
    outcome: str = "success",
    **details,
) -> None:
    """Write an audit record. Pass the tool's open session so SQLite never sees two writers."""
    entry = AuditLog(
        actor_id=ctx.actor.id,
        agent=ctx.agent_id,
        action=action,
        target_type=target_type,
        target_id=target_id or "",
        outcome=outcome,
        details=json.loads(json.dumps(details, default=str)),
        conversation_id=ctx.request.conversation_id,
        run_id=ctx.request.run_id,
    )
    try:
        if session is not None:
            session.add(entry)
            return
        with session_scope() as own:
            own.add(entry)
    except Exception:  # noqa: BLE001 - auditing must never break a tool
        log.exception("Failed to write audit log")


def forbidden(message: str, policy_ref: str = "HR-POL-004 §4", **extra) -> dict[str, Any]:
    return {"ok": False, "error": "forbidden", "message": message, "policy_ref": policy_ref, **extra}


async def execute_tool(name: str, ctx: ToolContext, arguments: dict[str, Any] | None) -> dict[str, Any]:
    spec = REGISTRY.get(name)
    if spec is None:
        return {"ok": False, "error": "unknown_tool", "message": f"No tool named '{name}'."}
    arguments = {k: v for k, v in (arguments or {}).items() if not k.startswith("_")}
    try:
        args = spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        issues = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return {"ok": False, "error": "invalid_arguments", "message": issues}
    try:
        if inspect.iscoroutinefunction(spec.handler):
            result = await spec.handler(ctx, args)
        else:
            result = await asyncio.to_thread(spec.handler, ctx, args)
    except (ValueError, LookupError) as exc:
        return {"ok": False, "error": "invalid_request", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        log.exception("Tool %s failed", name)
        return {"ok": False, "error": "tool_error", "message": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, dict):
        result.setdefault("ok", True)
        return result
    return {"ok": True, "result": result}


def schemas_for(names: list[str]) -> list[dict[str, Any]]:
    return [REGISTRY[n].openai_schema() for n in names if n in REGISTRY]
