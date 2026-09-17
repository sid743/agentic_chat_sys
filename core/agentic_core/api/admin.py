"""Admin / observability API and the built-in console page."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, inspect, select

from .. import __version__
from ..db.models import BROWSABLE_TABLES, AgentRun
from ..db.seed import seed_database
from ..db.session import session_scope
from ..service import ChatService
from .deps import get_service, require_admin

router = APIRouter(tags=["admin"])
STATIC = Path(__file__).resolve().parent.parent / "static"


def _row(obj) -> dict:
    data = {}
    for column in inspect(obj).mapper.column_attrs:
        value = getattr(obj, column.key)
        if isinstance(value, (date, datetime)):
            value = value.isoformat()
        data[column.key] = value
    return data


@router.get("/", include_in_schema=False)
@router.get("/admin", include_in_schema=False)
async def console_page():
    return FileResponse(STATIC / "admin.html", media_type="text/html")


@router.get("/health")
async def health():
    return {"status": "ok", "version": __version__}


@router.get("/api/overview", dependencies=[Depends(require_admin)])
async def overview(service: ChatService = Depends(get_service)):
    settings = service.settings
    models = await service.registry.available_models()
    with session_scope() as session:
        counts = {name: session.scalar(select(func.count()).select_from(model)) for name, model in BROWSABLE_TABLES.items()}
        runs = session.scalar(select(func.count()).select_from(AgentRun))
        tokens = session.execute(
            select(func.coalesce(func.sum(AgentRun.input_tokens), 0), func.coalesce(func.sum(AgentRun.output_tokens), 0))
        ).one()
        avg_latency = session.scalar(select(func.avg(AgentRun.latency_ms)))
    return {
        "version": __version__,
        "today": settings.today().isoformat(),
        "default_model": await service.registry.default_model_id(),
        "router_model": settings.agent_router_model or None,
        "models": [m.__dict__ for m in models],
        "providers": service.registry.describe(),
        "agents": [
            {"id": a.id, "name": a.name, "tools": a.tools, "model": a.model or "selected model"}
            for a in service.agents.agents.values()
        ],
        "embeddings": service.knowledge.embedder.signature,
        "vector_store": settings.qdrant_url or f"embedded ({settings.resolved_qdrant_path})",
        "database": settings.resolved_database_url.split("@")[-1],
        "reasoning_field": settings.reasoning_field,
        "auth_enabled": bool(settings.agent_core_api_key),
        "tables": counts,
        "runs": {"count": runs, "input_tokens": tokens[0], "output_tokens": tokens[1], "avg_latency_ms": int(avg_latency or 0)},
    }


@router.get("/api/graph", dependencies=[Depends(require_admin)])
async def graph(service: ChatService = Depends(get_service)):
    return {"mermaid": service.orchestrator.mermaid()}


@router.get("/api/runs", dependencies=[Depends(require_admin)])
async def list_runs(limit: int = Query(50, ge=1, le=500), conversation_id: str | None = None):
    with session_scope() as session:
        stmt = select(AgentRun).order_by(AgentRun.started_at.desc()).limit(limit)
        if conversation_id:
            stmt = stmt.where(AgentRun.conversation_id == conversation_id)
        rows = session.scalars(stmt).all()
        return {
            "runs": [
                {k: v for k, v in _row(r).items() if k not in ("trace", "final_answer")} | {"answer_preview": r.final_answer[:160]}
                for r in rows
            ]
        }


@router.get("/api/runs/{run_id}", dependencies=[Depends(require_admin)])
async def get_run(run_id: str):
    with session_scope() as session:
        run = session.get(AgentRun, run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return _row(run)


@router.get("/api/tables", dependencies=[Depends(require_admin)])
async def tables():
    return {"tables": list(BROWSABLE_TABLES)}


@router.get("/api/tables/{name}", dependencies=[Depends(require_admin)])
async def table_rows(name: str, limit: int = Query(200, ge=1, le=2000)):
    model = BROWSABLE_TABLES.get(name)
    if model is None:
        raise HTTPException(404, "Unknown table")
    with session_scope() as session:
        pk = inspect(model).primary_key[0]
        order = pk.desc() if name in ("audit_log", "notifications") else pk
        rows = session.scalars(select(model).order_by(order).limit(limit)).all()
        return {"table": name, "rows": [_row(r) for r in rows]}


@router.post("/api/reset", dependencies=[Depends(require_admin)])
async def reset_demo(service: ChatService = Depends(get_service)):
    seeded = await asyncio.to_thread(seed_database, True)
    indexed = await asyncio.to_thread(service.knowledge.reset_all)
    return {"seeded": seeded, "indexed": indexed}
