from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CORE))

# Deterministic environment for every test (set before the package is imported).
os.environ["DEMO_TODAY"] = "2026-09-17"
os.environ["AGENT_DEFAULT_MODEL"] = "mock/hr-demo"
for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "AGENT_CORE_API_KEY", "AGENT_ROUTER_MODEL"):
    os.environ.pop(key, None)

from agentic_core.agents.context import Actor, RequestContext, ToolContext  # noqa: E402
from agentic_core.db.models import Employee  # noqa: E402
from agentic_core.db.session import session_scope  # noqa: E402
from agentic_core.main import build_service  # noqa: E402
from agentic_core.settings import Settings, get_settings  # noqa: E402
from agentic_core.tracing import TraceRecorder  # noqa: E402

TODAY = date(2026, 9, 17)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        data_dir=tmp_path,
        qdrant_path=":memory:",
        demo_today=TODAY,
        agent_default_model="mock/hr-demo",
        agent_router_model="",
        agent_core_api_key="",
        answer_footer=False,
        embeddings_provider="hash",
        log_level="WARNING",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def service(settings):
    svc = build_service(settings)
    yield svc
    svc.knowledge.close()


def tool_context(service, employee_id: str, conversation_id: str = "test-conv", agent: str = "test_agent") -> ToolContext:
    with session_scope() as s:
        actor = Actor.from_employee(s.get(Employee, employee_id))
    request = RequestContext(
        run_id="run-test",
        conversation_id=conversation_id,
        user_email="",
        user_name="",
        actor=actor,
        actor_source="test",
        today=TODAY,
        model_id="mock/hr-demo",
        settings=service.settings,
        knowledge=service.knowledge,
        trace=TraceRecorder("debug"),
    )
    return ToolContext(request=request, agent_id=agent)
