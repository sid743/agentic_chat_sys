"""FastAPI application factory.  Run with:  uvicorn agentic_core.main:app --port 8088"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .agents.config import load_agents_config
from .api import admin, documents, openai_compat
from .db.seed import seed_database
from .db.session import init_engine
from .llm.registry import ModelRegistry
from .rag.service import KnowledgeService
from .service import ChatService
from .settings import Settings, get_settings

log = logging.getLogger("agentic_core")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())
    # openai>=3 ships its own httpx fork ("httpx2"); keep both quiet.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_service(settings: Settings) -> ChatService:
    """Initialise storage, seed data, the policy index, models and agents."""
    init_engine(settings.resolved_database_url)
    if settings.auto_seed:
        seed_database()
    knowledge = KnowledgeService(settings)
    knowledge.index_policies()
    registry = ModelRegistry(settings)
    agents = load_agents_config(settings.agents_file)
    return ChatService(settings, registry, knowledge, agents)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings)
        service = await asyncio.to_thread(build_service, settings)
        app.state.service = service
        default_model = await service.registry.default_model_id()
        log.info(
            "Agent core %s ready - default model %s, %d agents, embeddings %s, auth %s",
            __version__,
            default_model,
            len(service.agents.agents),
            service.knowledge.embedder.signature,
            "on" if settings.agent_core_api_key else "OFF",
        )
        yield
        service.knowledge.close()

    app = FastAPI(
        title="Agentic HR Platform - agent core",
        version=__version__,
        description="Multi-agent HR assistant behind an OpenAI-compatible API (LibreChat custom endpoint).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(openai_compat.router)
    app.include_router(documents.router)
    app.include_router(admin.router)
    return app


app = create_app()
