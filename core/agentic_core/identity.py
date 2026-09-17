"""Maps the calling LibreChat user (or API client) to an employee in the dummy HR database.

Order: per-conversation /act-as override -> X-Employee-Id header -> USER_EMPLOYEE_MAP
-> employee with the same email -> DEFAULT_EMPLOYEE_ID.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db.models import ConversationState, Employee
from .settings import Settings


def find_employee(session: Session, key: str) -> Employee | None:
    key = (key or "").strip()
    if not key:
        return None
    if "@" in key:
        return session.scalar(select(Employee).where(func.lower(Employee.email) == key.lower()))
    emp = session.get(Employee, key.upper())
    if emp:
        return emp
    full_name = func.lower(Employee.first_name + " " + Employee.last_name)
    matches = session.scalars(select(Employee).where(full_name == key.lower())).all()
    return matches[0] if len(matches) == 1 else None


def resolve_actor(
    session: Session,
    settings: Settings,
    *,
    conversation_id: str,
    user_email: str = "",
    employee_id_header: str = "",
) -> tuple[Employee, str]:
    if settings.allow_act_as:
        state = session.get(ConversationState, conversation_id)
        if state and state.acting_employee_id:
            emp = session.get(Employee, state.acting_employee_id)
            if emp:
                return emp, "act-as"
    if employee_id_header:
        emp = find_employee(session, employee_id_header)
        if emp:
            return emp, "X-Employee-Id header"
    email = (user_email or "").strip().lower()
    if email:
        mapped = settings.user_map().get(email)
        if mapped and (emp := session.get(Employee, mapped)):
            return emp, "USER_EMPLOYEE_MAP"
        emp = find_employee(session, email)
        if emp:
            return emp, "matching email"
    emp = session.get(Employee, settings.default_employee_id)
    if emp is None:
        emp = session.scalars(select(Employee).order_by(Employee.id)).first()
    if emp is None:
        raise RuntimeError("The HR database has no employees; run the seed step")
    return emp, "default persona"


def set_act_as(session: Session, conversation_id: str, employee_id: str | None, user_email: str = "") -> None:
    state = session.get(ConversationState, conversation_id)
    if state is None:
        state = ConversationState(conversation_id=conversation_id, user_email=user_email)
        session.add(state)
    state.acting_employee_id = employee_id
