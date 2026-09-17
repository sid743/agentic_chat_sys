"""Per-request context shared by the orchestrator, agents and tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..rag.service import KnowledgeService
    from ..settings import Settings
    from ..tracing import TraceRecorder


@dataclass
class Actor:
    id: str
    name: str
    first_name: str
    email: str
    role: str
    title: str
    department: str
    location_code: str
    employment_type: str
    manager_id: str | None = None
    manager_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_employee(cls, emp) -> Actor:
        return cls(
            id=emp.id,
            name=emp.full_name,
            first_name=emp.first_name,
            email=emp.email,
            role=emp.role,
            title=emp.title,
            department=emp.department.name if emp.department else emp.department_code,
            location_code=emp.location_code,
            employment_type=emp.employment_type,
            manager_id=emp.manager_id,
            manager_name=emp.manager.full_name if emp.manager else None,
        )


@dataclass
class RequestContext:
    run_id: str
    conversation_id: str
    user_email: str
    user_name: str
    actor: Actor
    actor_source: str
    today: date
    model_id: str
    settings: Settings
    knowledge: KnowledgeService
    trace: TraceRecorder
    uploads: list[dict] = field(default_factory=list)
    new_uploads: list[dict] = field(default_factory=list)


@dataclass
class ToolContext:
    request: RequestContext
    agent_id: str

    @property
    def actor(self) -> Actor:
        return self.request.actor

    @property
    def today(self) -> date:
        return self.request.today
