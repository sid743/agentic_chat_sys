"""Dummy HR database schema (synthetic data only - not a real HRIS).

SQLite by default; any SQLAlchemy URL (e.g. PostgreSQL) works via DATABASE_URL.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


class Department(Base):
    __tablename__ = "departments"

    code: Mapped[str] = mapped_column(String(8), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    head_employee_id: Mapped[str | None] = mapped_column(String(10), nullable=True)


class Location(Base):
    __tablename__ = "locations"

    code: Mapped[str] = mapped_column(String(8), primary_key=True)
    city: Mapped[str] = mapped_column(String(60))
    state: Mapped[str] = mapped_column(String(60))
    country: Mapped[str] = mapped_column(String(60), default="India")


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(10), primary_key=True)  # E1001
    first_name: Mapped[str] = mapped_column(String(60))
    last_name: Mapped[str] = mapped_column(String(60))
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(80))
    department_code: Mapped[str] = mapped_column(ForeignKey("departments.code"))
    location_code: Mapped[str] = mapped_column(ForeignKey("locations.code"))
    manager_id: Mapped[str | None] = mapped_column(ForeignKey("employees.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="employee")  # employee | manager | hr_admin
    employment_type: Mapped[str] = mapped_column(String(20), default="full_time")  # full_time | contract | intern
    date_of_joining: Mapped[date] = mapped_column(Date)
    probation_months: Mapped[int] = mapped_column(Integer, default=6)
    status: Mapped[str] = mapped_column(String(20), default="active")

    manager: Mapped[Employee | None] = relationship(remote_side="Employee.id", foreign_keys=[manager_id])
    department: Mapped[Department] = relationship()
    location: Mapped[Location] = relationship()

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class LeaveType(Base):
    __tablename__ = "leave_types"

    code: Mapped[str] = mapped_column(String(8), primary_key=True)  # AL, SL, CL ...
    name: Mapped[str] = mapped_column(String(80))
    paid: Mapped[bool] = mapped_column(Boolean, default=True)
    annual_quota_days: Mapped[float] = mapped_column(Float, default=0)
    # monthly: accrues each completed month | upfront: credited at start (pro-rated) | event: per life event
    accrual: Mapped[str] = mapped_column(String(10), default="upfront")
    requires_probation_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    min_service_days: Mapped[int] = mapped_column(Integer, default=0)
    max_consecutive_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    advance_notice_days: Mapped[int] = mapped_column(Integer, default=0)
    carry_forward_max: Mapped[float] = mapped_column(Float, default=0)
    # HR approval needed when working days exceed this value (0 = always, NULL = never)
    hr_approval_over_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    eligible_employment_types: Mapped[str] = mapped_column(String(60), default="full_time,contract")
    document_required_after_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy_ref: Mapped[str] = mapped_column(String(80), default="")
    description: Mapped[str] = mapped_column(Text, default="")


class LeaveBalance(Base):
    """Entitlement ledger. Used/pending days are derived from leave_requests."""

    __tablename__ = "leave_balances"
    __table_args__ = (UniqueConstraint("employee_id", "leave_type", "year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[str] = mapped_column(ForeignKey("employees.id"), index=True)
    leave_type: Mapped[str] = mapped_column(ForeignKey("leave_types.code"))
    year: Mapped[int] = mapped_column(Integer)
    entitled_days: Mapped[float] = mapped_column(Float, default=0)
    carried_forward: Mapped[float] = mapped_column(Float, default=0)
    adjustment: Mapped[float] = mapped_column(Float, default=0)


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # LR-2026-0001
    employee_id: Mapped[str] = mapped_column(ForeignKey("employees.id"), index=True)
    leave_type: Mapped[str] = mapped_column(ForeignKey("leave_types.code"))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    working_days: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text, default="")
    # pending_manager | pending_hr | approved | rejected | cancelled
    status: Mapped[str] = mapped_column(String(20), default="pending_manager", index=True)
    current_approver_id: Mapped[str | None] = mapped_column(ForeignKey("employees.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
    source: Mapped[str] = mapped_column(String(10), default="seed")  # seed | chat | api
    conversation_id: Mapped[str | None] = mapped_column(String(80), nullable=True)

    employee: Mapped[Employee] = relationship(foreign_keys=[employee_id])
    approver: Mapped[Employee | None] = relationship(foreign_keys=[current_approver_id])


class ApprovalStep(Base):
    __tablename__ = "approval_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("leave_requests.id"), index=True)
    level: Mapped[str] = mapped_column(String(10))  # manager | hr
    approver_id: Mapped[str] = mapped_column(ForeignKey("employees.id"))
    decision: Mapped[str] = mapped_column(String(10))  # approved | rejected
    comment: Mapped[str] = mapped_column(Text, default="")
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Holiday(Base):
    __tablename__ = "holidays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    holiday_date: Mapped[date] = mapped_column(Date, index=True)
    name: Mapped[str] = mapped_column(String(80))
    location_code: Mapped[str] = mapped_column(String(8), default="ALL")  # ALL or a location code
    optional: Mapped[bool] = mapped_column(Boolean, default=False)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recipient_id: Mapped[str] = mapped_column(ForeignKey("employees.id"), index=True)
    sender_id: Mapped[str | None] = mapped_column(ForeignKey("employees.id"), nullable=True)
    channel: Mapped[str] = mapped_column(String(10), default="email")  # email | teams | in_app
    subject: Mapped[str] = mapped_column(String(160))
    body: Mapped[str] = mapped_column(Text)
    related_request_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(10), default="sent")  # simulated delivery
    created_by_agent: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(10), nullable=True)
    agent: Mapped[str] = mapped_column(String(40), default="")
    action: Mapped[str] = mapped_column(String(60))
    target_type: Mapped[str] = mapped_column(String(30), default="")
    target_id: Mapped[str] = mapped_column(String(40), default="")
    outcome: Mapped[str] = mapped_column(String(10), default="success")  # success | denied | error
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    conversation_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)


class ConversationState(Base):
    __tablename__ = "conversation_state"

    conversation_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    acting_employee_id: Mapped[str | None] = mapped_column(String(10), nullable=True)
    user_email: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    scope: Mapped[str] = mapped_column(String(12))  # policy | conversation
    conversation_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(200))
    doc_code: Mapped[str] = mapped_column(String(30), default="")
    version: Mapped[str] = mapped_column(String(20), default="")
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    collection: Mapped[str] = mapped_column(String(80))
    uploaded_by: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(80), index=True)
    user_email: Mapped[str] = mapped_column(String(120), default="")
    actor_id: Mapped[str | None] = mapped_column(String(10), nullable=True)
    model: Mapped[str] = mapped_column(String(120), default="")
    status: Mapped[str] = mapped_column(String(12), default="ok")  # ok | error
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    agents: Mapped[str] = mapped_column(String(200), default="")
    user_message: Mapped[str] = mapped_column(Text, default="")
    final_answer: Mapped[str] = mapped_column(Text, default="")
    trace: Mapped[list] = mapped_column(JSON, default=list)


# Tables that the admin page may display (name -> model)
BROWSABLE_TABLES = {
    "employees": Employee,
    "departments": Department,
    "locations": Location,
    "leave_types": LeaveType,
    "leave_balances": LeaveBalance,
    "leave_requests": LeaveRequest,
    "approval_steps": ApprovalStep,
    "holidays": Holiday,
    "notifications": Notification,
    "audit_log": AuditLog,
    "documents": Document,
    "conversation_state": ConversationState,
}
