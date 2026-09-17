"""HR tools over the dummy database.

Access rules (HR-POL-004 §4) and workflow rules (HR-POL-001 §6) are enforced
here, in code, so a model cannot talk its way around them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field
from sqlalchemy import func, or_, select

from ..agents import nlp
from ..agents.context import ToolContext
from ..db.models import ApprovalStep, Employee, Holiday, LeaveRequest, LeaveType, Notification
from ..db.session import session_scope
from ..domain import leave as rules
from ..settings import get_settings
from .base import audit, forbidden, tool

LEAVE_TYPE_ALIASES = {
    "annual": "AL", "annual leave": "AL", "privilege": "AL", "privilege leave": "AL", "pl": "AL",
    "earned leave": "AL", "paid leave": "AL", "vacation": "AL",
    "sick": "SL", "sick leave": "SL",
    "casual": "CL", "casual leave": "CL",
    "maternity": "PPL", "primary caregiver": "PPL", "parental": "PPL", "parental leave": "PPL",
    "paternity": "SPL", "secondary caregiver": "SPL",
    "bereavement": "BL", "bereavement leave": "BL",
    "unpaid": "LWP", "leave without pay": "LWP", "lwp": "LWP", "loss of pay": "LWP",
}


def _leave_type(value):
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.upper() in {"AL", "SL", "CL", "PPL", "SPL", "BL", "LWP"}:
        return text.upper()
    lowered = text.lower()
    if lowered in LEAVE_TYPE_ALIASES:
        return LEAVE_TYPE_ALIASES[lowered]
    guess = nlp.guess_leave_type(text, None)
    if guess:
        return guess
    raise ValueError("unknown leave type; use one of AL, SL, CL, PPL, SPL, BL, LWP")


def _date(value):
    if value is None or value == "" or isinstance(value, date):
        return value or None
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        parsed = nlp.parse_dates(text, get_settings().today())
        if parsed:
            return parsed[0]
    raise ValueError("use YYYY-MM-DD")


def _upper(value):
    return str(value).strip().upper() if value else value


LeaveTypeCode = Annotated[str, BeforeValidator(_leave_type)]
OptLeaveType = Annotated[str | None, BeforeValidator(_leave_type)]
Day = Annotated[date, BeforeValidator(_date)]
OptDay = Annotated[date | None, BeforeValidator(_date)]
EmpId = Annotated[str | None, BeforeValidator(_upper)]
ReqId = Annotated[str, BeforeValidator(_upper)]


# --------------------------------------------------------------------------- helpers
def _employee(session, employee_id: str | None, ctx: ToolContext) -> Employee:
    emp = session.get(Employee, employee_id or ctx.actor.id)
    if emp is None:
        raise LookupError(f"No employee with id {employee_id}")
    return emp


def _can_view_restricted(viewer: Employee, target: Employee) -> bool:
    return viewer.id == target.id or viewer.role == "hr_admin" or target.manager_id == viewer.id


def _directory_entry(emp: Employee) -> dict:
    return {
        "employee_id": emp.id,
        "name": emp.full_name,
        "title": emp.title,
        "department": emp.department.name if emp.department else emp.department_code,
        "location": emp.location.city if emp.location else emp.location_code,
        "manager_name": emp.manager.full_name if emp.manager else None,
        "email": emp.email,
    }


def _event(kind: str, req: LeaveRequest, summary: str, notify: list[dict]) -> dict:
    return {"type": kind, "request_id": req.id, "summary": summary, "notify": notify}


def _notify_entry(emp: Employee | None, role: str, subject: str, message: str, channel: str = "email") -> list[dict]:
    if emp is None:
        return []
    return [{"employee_id": emp.id, "name": emp.full_name, "role": role, "subject": subject, "message": message, "channel": channel}]


def _describe(req: LeaveRequest, lt: LeaveType | None = None) -> str:
    name = lt.name if lt else req.leave_type
    return f"{name} from {req.start_date.isoformat()} to {req.end_date.isoformat()} ({req.working_days:g} working day(s))"


# --------------------------------------------------------------------------- profile & directory
@tool(
    "get_my_profile",
    "Get the requesting employee's own HR profile: id, title, department, location, manager, "
    "date of joining, service days, probation status, role and direct reports.",
)
def get_my_profile(ctx: ToolContext, args: BaseModel) -> dict:
    with session_scope() as s:
        emp = _employee(s, None, ctx)
        reports = s.scalars(select(Employee).where(Employee.manager_id == emp.id).order_by(Employee.id)).all()
        p_end = rules.probation_end(emp)
        return {
            "profile": {
                "employee_id": emp.id,
                "name": emp.full_name,
                "email": emp.email,
                "title": emp.title,
                "department": emp.department.name,
                "location": f"{emp.location.city} ({emp.location_code})",
                "role": emp.role,
                "employment_type": emp.employment_type,
                "date_of_joining": emp.date_of_joining.isoformat(),
                "service_days": rules.service_days(emp, ctx.today),
                "probation_end_date": p_end.isoformat(),
                "probation_complete": ctx.today >= p_end,
                "manager_id": emp.manager_id,
                "manager_name": emp.manager.full_name if emp.manager else None,
            },
            "direct_reports": [{"employee_id": r.id, "name": r.full_name, "title": r.title} for r in reports],
            "as_of": ctx.today.isoformat(),
        }


class FindEmployeeArgs(BaseModel):
    query: str = Field(description="Name, employee id (E1003) or email to look up")


@tool(
    "find_employee",
    "Look up colleagues in the company directory (name, title, department, location, manager). "
    "Directory data only - never balances or leave details.",
    FindEmployeeArgs,
)
def find_employee(ctx: ToolContext, args: FindEmployeeArgs) -> dict:
    q = args.query.strip()
    with session_scope() as s:
        full_name = func.lower(Employee.first_name + " " + Employee.last_name)
        stmt = select(Employee).where(
            or_(
                func.upper(Employee.id) == q.upper(),
                func.lower(Employee.email) == q.lower(),
                full_name.like(f"%{q.lower()}%"),
                func.lower(Employee.first_name) == q.lower(),
                func.lower(Employee.last_name) == q.lower(),
            )
        )
        matches = s.scalars(stmt.order_by(Employee.id).limit(5)).all()
        return {"matches": [_directory_entry(m) for m in matches], "count": len(matches)}


# --------------------------------------------------------------------------- balances & eligibility
class BalanceArgs(BaseModel):
    employee_id: EmpId = Field(None, description="Employee id; omit for the requesting employee")
    leave_type: OptLeaveType = Field(None, description="Optional leave type code: AL, SL, CL, PPL, SPL, BL, LWP")
    include_event_based: bool = Field(False, description="Also list event-based leave (parental, bereavement)")


@tool(
    "get_leave_balance",
    "Get leave balances (carried forward, accrued, used, pending, available) for the requesting employee, "
    "or for another employee if the requester is their manager or HR. Access is enforced.",
    BalanceArgs,
)
def get_leave_balance(ctx: ToolContext, args: BalanceArgs) -> dict:
    with session_scope() as s:
        viewer = _employee(s, None, ctx)
        target = _employee(s, args.employee_id, ctx)
        if not _can_view_restricted(viewer, target):
            audit(ctx, s, "view_leave_balance", target_type="employee", target_id=target.id, outcome="denied")
            return forbidden(
                f"{viewer.full_name} is not allowed to view {target.full_name}'s leave balance. Only the employee, "
                "their manager or HR administrators can see it.",
                target_name=target.full_name,
            )
        rows = rules.balances_for(s, target, ctx.today)
        if args.leave_type:
            rows = [r for r in rows if r["leave_type"] == args.leave_type]
        elif not args.include_event_based:
            rows = [r for r in rows if r["accrual"] != "event" and r["leave_type"] != "LWP"]
        if target.id != viewer.id:
            audit(ctx, s, "view_leave_balance", target_type="employee", target_id=target.id)
        return {
            "employee_id": target.id,
            "employee_name": target.full_name,
            "is_self": target.id == viewer.id,
            "as_of": ctx.today.isoformat(),
            "balances": rows,
            "notes": [
                "available = carried forward + accrued/credited - used (approved) - pending",
                "Annual Leave accrues 1.5 days per completed month (HR-POL-001 §3.1)",
            ],
        }


class EligibilityArgs(BaseModel):
    leave_type: LeaveTypeCode = Field(description="Leave type code: AL, SL, CL, PPL, SPL, BL or LWP")
    start_date: OptDay = Field(None, description="Planned first day (YYYY-MM-DD), optional")
    end_date: OptDay = Field(None, description="Planned last day (YYYY-MM-DD), optional")
    employee_id: EmpId = Field(None, description="Employee id; omit for the requesting employee")


@tool(
    "check_leave_eligibility",
    "Check whether an employee can take a leave type (probation, service, employment type, balance, notice, "
    "overlaps, limits). Returns each rule checked with policy references.",
    EligibilityArgs,
)
def check_leave_eligibility(ctx: ToolContext, args: EligibilityArgs) -> dict:
    with session_scope() as s:
        viewer = _employee(s, None, ctx)
        target = _employee(s, args.employee_id, ctx)
        if not _can_view_restricted(viewer, target):
            audit(ctx, s, "check_eligibility", target_type="employee", target_id=target.id, outcome="denied")
            return forbidden(
                f"You are not allowed to check {target.full_name}'s leave eligibility.", target_name=target.full_name
            )
        lt = s.get(LeaveType, args.leave_type)
        start, end = args.start_date, args.end_date or args.start_date
        return rules.evaluate_eligibility(s, target, lt, ctx.today, start, end if start else None)


class DaysArgs(BaseModel):
    start_date: Day
    end_date: Day
    location_code: str | None = Field(None, description="MUM, PUN, BLR, HYD or DEL; defaults to the employee's location")


@tool(
    "calculate_leave_days",
    "Count working days between two dates (inclusive), excluding weekends and public holidays for a location.",
    DaysArgs,
)
def calculate_leave_days(ctx: ToolContext, args: DaysArgs) -> dict:
    location = (args.location_code or ctx.actor.location_code).upper()
    if (args.end_date - args.start_date).days > 400:
        raise ValueError("date range is too long (max ~13 months)")
    with session_scope() as s:
        count = rules.count_working_days(s, location, args.start_date, args.end_date)
    return {"start_date": args.start_date.isoformat(), "end_date": args.end_date.isoformat(), "location": location, **count.as_dict()}


class HolidayArgs(BaseModel):
    year: int | None = None
    location_code: str | None = Field(None, description="MUM, PUN, BLR, HYD or DEL; defaults to the employee's location")
    upcoming_only: bool = True


@tool("list_holidays", "List public and optional holidays for a location (synthetic demo calendar).", HolidayArgs)
def list_holidays(ctx: ToolContext, args: HolidayArgs) -> dict:
    location = (args.location_code or ctx.actor.location_code).upper()
    with session_scope() as s:
        stmt = select(Holiday).where(or_(Holiday.location_code == "ALL", Holiday.location_code == location))
        if args.year:
            stmt = stmt.where(Holiday.holiday_date >= date(args.year, 1, 1), Holiday.holiday_date <= date(args.year, 12, 31))
        if args.upcoming_only:
            stmt = stmt.where(Holiday.holiday_date >= ctx.today)
        rows = s.scalars(stmt.order_by(Holiday.holiday_date)).all()
        return {
            "location": location,
            "holidays": [
                {
                    "date": h.holiday_date.isoformat(),
                    "weekday": h.holiday_date.strftime("%A"),
                    "name": h.name,
                    "optional": h.optional,
                }
                for h in rows
            ],
            "note": "Synthetic demo calendar. Employees may take 2 optional holidays per year (HR-POL-001 §9.2).",
        }


@tool("list_leave_types", "List leave types with their rules (quota, accrual, notice, limits, approvals, policy reference).")
def list_leave_types(ctx: ToolContext, args: BaseModel) -> dict:
    with session_scope() as s:
        rows = s.scalars(select(LeaveType).order_by(LeaveType.code)).all()
        return {
            "leave_types": [
                {
                    "code": lt.code,
                    "name": lt.name,
                    "paid": lt.paid,
                    "annual_quota_days": lt.annual_quota_days,
                    "accrual": lt.accrual,
                    "requires_probation_complete": lt.requires_probation_complete,
                    "min_service_days": lt.min_service_days,
                    "max_consecutive_days": lt.max_consecutive_days,
                    "advance_notice_days": lt.advance_notice_days,
                    "carry_forward_max": lt.carry_forward_max,
                    "hr_approval_over_days": lt.hr_approval_over_days,
                    "eligible_employment_types": lt.eligible_employment_types,
                    "policy_ref": lt.policy_ref,
                    "description": lt.description,
                }
                for lt in rows
            ]
        }


# --------------------------------------------------------------------------- workflow
class CreateRequestArgs(BaseModel):
    leave_type: LeaveTypeCode = Field(description="AL, SL, CL, PPL, SPL, BL or LWP")
    start_date: Day = Field(description="First day of leave, YYYY-MM-DD")
    end_date: Day = Field(description="Last day of leave, YYYY-MM-DD")
    reason: str = Field("", description="Short reason shared with the approver")


@tool(
    "create_leave_request",
    "Submit a leave request for the requesting employee (never for someone else). Runs all eligibility "
    "checks first and routes it to the right approver(s).",
    CreateRequestArgs,
    kind="action",
)
def create_leave_request(ctx: ToolContext, args: CreateRequestArgs) -> dict:
    if args.end_date < args.start_date:
        raise ValueError("end_date is before start_date")
    with session_scope() as s:
        emp = _employee(s, None, ctx)
        lt = s.get(LeaveType, args.leave_type)
        check = rules.evaluate_eligibility(s, emp, lt, ctx.today, args.start_date, args.end_date)
        if not check["eligible"]:
            audit(ctx, s, "create_leave_request", target_type="employee", target_id=emp.id, outcome="denied",
                  leave_type=lt.code, start=args.start_date, end=args.end_date, reasons=check["blocking_reasons"])
            return {
                "ok": False,
                "error": "not_eligible",
                "message": f"The {lt.name} request cannot be submitted.",
                "reasons": check["blocking_reasons"],
                "policy_refs": check["policy_refs"],
            }
        days = check["date_range"]["working_days"]
        levels = rules.approval_levels(emp, lt, days)
        if levels[0] == "manager":
            approver = emp.manager
            status = "pending_manager"
        else:
            approver = rules.hr_approver_for(s, emp)
            status = "pending_hr"
        req = LeaveRequest(
            id=rules.next_request_id(s, args.start_date.year),
            employee_id=emp.id,
            leave_type=lt.code,
            start_date=args.start_date,
            end_date=args.end_date,
            working_days=days,
            reason=args.reason[:500],
            status=status,
            current_approver_id=approver.id if approver else None,
            source="chat",
            conversation_id=ctx.request.conversation_id,
        )
        s.add(req)
        s.flush()
        s.refresh(req)
        desc = _describe(req, lt)
        audit(ctx, s, "create_leave_request", target_type="leave_request", target_id=req.id, leave_type=lt.code,
              days=days, approver=approver.id if approver else None)
        event = _event(
            "leave_request_submitted",
            req,
            f"{emp.full_name} submitted {req.id}: {desc}.",
            _notify_entry(approver, "approver", f"Leave request {req.id} awaiting your approval",
                          f"{emp.full_name} requested {desc}. Reason: {req.reason or 'not given'}.")
            + _notify_entry(emp, "requester", f"Leave request {req.id} submitted",
                            f"Your request for {desc} is pending approval by {approver.full_name if approver else 'HR'}.",
                            "in_app"),
        )
        data = rules.request_to_dict(req)
        data["leave_type_name"] = lt.name
        return {
            "request": data,
            "approval_levels": levels,
            "warnings": check["warnings"],
            "policy_refs": check["policy_refs"],
            "events": [event],
        }


class ListRequestsArgs(BaseModel):
    scope: Literal["mine", "awaiting_my_approval", "team", "all"] = Field(
        "mine", description="mine | awaiting_my_approval | team (direct reports) | all (HR only)"
    )
    status: Literal["pending", "approved", "rejected", "cancelled", "any"] | None = "any"
    employee_id: EmpId = None
    limit: int = Field(20, ge=1, le=100)


@tool(
    "list_leave_requests",
    "List leave requests: your own, those awaiting your approval, your team's (managers) or everyone's (HR).",
    ListRequestsArgs,
)
def list_leave_requests(ctx: ToolContext, args: ListRequestsArgs) -> dict:
    with session_scope() as s:
        viewer = _employee(s, None, ctx)
        stmt = select(LeaveRequest)
        if args.employee_id:
            target = _employee(s, args.employee_id, ctx)
            if not _can_view_restricted(viewer, target):
                audit(ctx, s, "list_leave_requests", target_type="employee", target_id=target.id, outcome="denied")
                return forbidden(f"You are not allowed to view {target.full_name}'s leave requests.", target_name=target.full_name)
            stmt = stmt.where(LeaveRequest.employee_id == target.id)
        elif args.scope == "mine":
            stmt = stmt.where(LeaveRequest.employee_id == viewer.id)
        elif args.scope == "awaiting_my_approval":
            stmt = stmt.where(LeaveRequest.current_approver_id == viewer.id, LeaveRequest.status.in_(rules.PENDING_STATUSES))
        elif args.scope == "team":
            if viewer.role not in ("manager", "hr_admin"):
                return forbidden("Only managers can list their team's requests.")
            team_ids = select(Employee.id).where(Employee.manager_id == viewer.id)
            stmt = stmt.where(LeaveRequest.employee_id.in_(team_ids))
        elif args.scope == "all":
            if viewer.role != "hr_admin":
                return forbidden("Only HR administrators can list all requests.")
        status = args.status or "any"
        if status == "pending":
            stmt = stmt.where(LeaveRequest.status.in_(rules.PENDING_STATUSES))
        elif status != "any":
            stmt = stmt.where(LeaveRequest.status == status)
        rows = s.scalars(stmt.order_by(LeaveRequest.start_date.desc()).limit(args.limit)).all()
        return {"scope": args.scope, "status": status, "count": len(rows), "requests": [rules.request_to_dict(r) for r in rows]}


class DecisionArgs(BaseModel):
    request_id: ReqId = Field(description="Leave request id, e.g. LR-2026-0008")
    comment: str = Field("", description="Comment shared with the employee")


class RejectArgs(BaseModel):
    request_id: ReqId
    reason: str = Field(description="Reason for rejection (required by HR-POL-001 §6.7)")


def _load_request(s, request_id: str) -> LeaveRequest:
    req = s.get(LeaveRequest, request_id)
    if req is None:
        raise LookupError(f"No leave request with id {request_id}")
    return req


def _decide(ctx: ToolContext, request_id: str, decision: str, comment: str) -> dict:
    with session_scope() as s:
        actor = _employee(s, None, ctx)
        req = _load_request(s, request_id)
        requester = req.employee
        lt = s.get(LeaveType, req.leave_type)
        action = "approve_leave_request" if decision == "approved" else "reject_leave_request"
        if req.status not in rules.PENDING_STATUSES:
            return {"ok": False, "error": "invalid_state", "message": f"{req.id} is already {req.status}."}
        if requester.id == actor.id:
            audit(ctx, s, action, target_type="leave_request", target_id=req.id, outcome="denied", rule="self_approval")
            return forbidden("Nobody can approve or reject their own leave request.", "HR-POL-001 §6.3")
        level = "manager" if req.status == "pending_manager" else "hr"
        allowed = (
            (level == "manager" and (requester.manager_id == actor.id or actor.role == "hr_admin"))
            or (level == "hr" and actor.role == "hr_admin")
        )
        if not allowed:
            audit(ctx, s, action, target_type="leave_request", target_id=req.id, outcome="denied", level=level)
            who = "the employee's manager (or HR)" if level == "manager" else "an HR administrator"
            return forbidden(f"{req.id} is waiting for {who}; you are not an approver for it.", "HR-POL-001 §6.1-6.2")
        on_behalf = level == "manager" and requester.manager_id != actor.id
        s.add(ApprovalStep(request_id=req.id, level=level, approver_id=actor.id, decision=decision,
                           comment=(comment + (" (HR on behalf of manager)" if on_behalf else "")).strip()))
        desc = _describe(req, lt)
        notify: list[dict] = []
        if decision == "rejected":
            req.status = "rejected"
            req.current_approver_id = None
            summary = f"{actor.full_name} rejected {req.id} ({requester.full_name}, {desc})."
            notify += _notify_entry(requester, "requester", f"Leave request {req.id} rejected",
                                    f"{actor.full_name} rejected your request for {desc}. Reason: {comment}")
        else:
            levels = rules.approval_levels(requester, lt, req.working_days)
            if level == "manager" and "hr" in levels:
                hr = rules.hr_approver_for(s, requester)
                req.status = "pending_hr"
                req.current_approver_id = hr.id if hr else None
                summary = f"{actor.full_name} approved {req.id} at manager level; it now needs HR approval."
                notify += _notify_entry(hr, "approver", f"Leave request {req.id} needs HR approval",
                                        f"{requester.full_name}'s request for {desc} was approved by {actor.full_name} and needs HR approval.")
                notify += _notify_entry(requester, "requester", f"Leave request {req.id} approved by manager",
                                        f"Your request for {desc} was approved by {actor.full_name} and is now with HR.", "in_app")
            else:
                req.status = "approved"
                req.current_approver_id = None
                summary = f"{actor.full_name} approved {req.id} ({requester.full_name}, {desc}); it is fully approved."
                notify += _notify_entry(requester, "requester", f"Leave request {req.id} approved",
                                        f"Your request for {desc} was approved by {actor.full_name}. {comment}".strip())
        req.updated_at = datetime.now().replace(microsecond=0)
        s.flush()
        s.refresh(req)
        audit(ctx, s, action, target_type="leave_request", target_id=req.id, level=level, new_status=req.status)
        return {
            "request": rules.request_to_dict(req),
            "decision": decision,
            "level": level,
            "events": [_event(f"leave_request_{decision}", req, summary, notify)],
        }


@tool(
    "approve_leave_request",
    "Approve a pending leave request as its current approver (manager, or HR at the HR step). Self-approval is blocked.",
    DecisionArgs,
    kind="action",
)
def approve_leave_request(ctx: ToolContext, args: DecisionArgs) -> dict:
    return _decide(ctx, args.request_id, "approved", args.comment)


@tool("reject_leave_request", "Reject a pending leave request as its current approver; a reason is required.", RejectArgs, kind="action")
def reject_leave_request(ctx: ToolContext, args: RejectArgs) -> dict:
    if not args.reason.strip():
        raise ValueError("A rejection reason is required (HR-POL-001 §6.7)")
    return _decide(ctx, args.request_id, "rejected", args.reason)


class CancelArgs(BaseModel):
    request_id: ReqId
    reason: str = ""


@tool(
    "cancel_leave_request",
    "Cancel the requesting employee's own leave request (pending, or approved but not yet started).",
    CancelArgs,
    kind="action",
)
def cancel_leave_request(ctx: ToolContext, args: CancelArgs) -> dict:
    with session_scope() as s:
        actor = _employee(s, None, ctx)
        req = _load_request(s, args.request_id)
        if req.employee_id != actor.id and actor.role != "hr_admin":
            audit(ctx, s, "cancel_leave_request", target_type="leave_request", target_id=req.id, outcome="denied")
            return forbidden("You can only cancel your own leave requests.", "HR-POL-001 §6.5")
        if req.status in ("rejected", "cancelled"):
            return {"ok": False, "error": "invalid_state", "message": f"{req.id} is already {req.status}."}
        if req.status == "approved" and req.start_date <= ctx.today and actor.role != "hr_admin":
            return {"ok": False, "error": "invalid_state",
                    "message": "This leave has already started; contact HR to change it (HR-POL-001 §6.5)."}
        previous_approver = req.approver
        was = req.status
        req.status = "cancelled"
        req.current_approver_id = None
        req.updated_at = datetime.now().replace(microsecond=0)
        lt = s.get(LeaveType, req.leave_type)
        desc = _describe(req, lt)
        notify_target = previous_approver if was in rules.PENDING_STATUSES else req.employee.manager
        s.flush()
        s.refresh(req)
        audit(ctx, s, "cancel_leave_request", target_type="leave_request", target_id=req.id, previous_status=was)
        return {
            "request": rules.request_to_dict(req),
            "previous_status": was,
            "events": [
                _event(
                    "leave_request_cancelled",
                    req,
                    f"{actor.full_name} cancelled {req.id} ({desc}).",
                    _notify_entry(notify_target, "approver", f"Leave request {req.id} cancelled",
                                  f"{req.employee.full_name} cancelled the request for {desc}. {args.reason}".strip()),
                )
            ],
        }


# --------------------------------------------------------------------------- notifications
class NotifyArgs(BaseModel):
    recipient: str = Field(
        description="Employee id (E1005), email, or one of: me, my_manager, hr, requester, approver "
        "(requester/approver need related_request_id)"
    )
    subject: str = Field(max_length=160)
    message: str = Field(max_length=2000)
    channel: Literal["email", "teams", "in_app"] = "email"
    related_request_id: Annotated[str | None, BeforeValidator(_upper)] = None


@tool(
    "send_notification",
    "Send a (simulated) email / Teams / in-app notification to yourself, your manager, your direct reports, "
    "HR, or people involved in a leave request you are part of.",
    NotifyArgs,
    kind="action",
)
def send_notification(ctx: ToolContext, args: NotifyArgs) -> dict:
    with session_scope() as s:
        actor = _employee(s, None, ctx)
        req = s.get(LeaveRequest, args.related_request_id) if args.related_request_id else None
        key = args.recipient.strip()
        low = key.lower()
        recipient: Employee | None = None
        if low in ("me", "self", "myself"):
            recipient = actor
        elif low in ("my_manager", "manager"):
            recipient = actor.manager
        elif low == "hr":
            recipient = rules.hr_approver_for(s, actor)
        elif low == "requester" and req:
            recipient = req.employee
        elif low == "approver" and req:
            recipient = req.approver or (req.employee.manager if req.employee else None)
        elif "@" in key:
            recipient = s.scalar(select(Employee).where(func.lower(Employee.email) == low))
        else:
            recipient = s.get(Employee, key.upper())
        if recipient is None:
            return {"ok": False, "error": "unknown_recipient", "message": f"Could not resolve recipient '{args.recipient}'."}

        involved: set[str] = set()
        if req is not None:
            past = s.scalars(select(ApprovalStep.approver_id).where(ApprovalStep.request_id == req.id)).all()
            involved = {req.employee_id, req.current_approver_id or "", req.employee.manager_id or "", *past}
        allowed = (
            actor.role == "hr_admin"
            or recipient.id == actor.id
            or recipient.id == actor.manager_id
            or recipient.manager_id == actor.id
            or recipient.role == "hr_admin"
            or (req is not None and actor.id in involved and recipient.id in involved)
        )
        if not allowed:
            audit(ctx, s, "send_notification", target_type="employee", target_id=recipient.id, outcome="denied")
            return forbidden(
                f"You can't message {recipient.full_name} from the HR assistant; notifications are limited to "
                "yourself, your manager, your reports, HR and people on a shared leave request.",
                "HR-POL-004 §5",
            )
        note = Notification(
            recipient_id=recipient.id,
            sender_id=actor.id,
            channel=args.channel,
            subject=args.subject,
            body=args.message,
            related_request_id=req.id if req else None,
            status="sent",
            created_by_agent=ctx.agent_id,
        )
        s.add(note)
        s.flush()
        audit(ctx, s, "send_notification", target_type="employee", target_id=recipient.id, channel=args.channel,
              related_request_id=note.related_request_id)
        return {
            "notification": {
                "id": note.id,
                "recipient_id": recipient.id,
                "recipient_name": recipient.full_name,
                "channel": note.channel,
                "subject": note.subject,
                "related_request_id": note.related_request_id,
                "delivery": "simulated (stored in the notifications table)",
            }
        }


class ListNotificationsArgs(BaseModel):
    limit: int = Field(10, ge=1, le=50)


@tool("list_notifications", "List the requesting employee's most recent notifications.", ListNotificationsArgs)
def list_notifications(ctx: ToolContext, args: ListNotificationsArgs) -> dict:
    with session_scope() as s:
        rows = s.scalars(
            select(Notification)
            .where(Notification.recipient_id == ctx.actor.id)
            .order_by(Notification.created_at.desc(), Notification.id.desc())
            .limit(args.limit)
        ).all()
        return {
            "notifications": [
                {
                    "id": n.id,
                    "channel": n.channel,
                    "subject": n.subject,
                    "body": n.body,
                    "related_request_id": n.related_request_id,
                    "created_at": n.created_at.isoformat(),
                }
                for n in rows
            ]
        }
