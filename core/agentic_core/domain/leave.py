"""HR leave domain rules (deterministic, no LLM involved).

Agents call these through tools, so policy enforcement never depends on what a
model decides to do.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..db.models import Employee, Holiday, LeaveBalance, LeaveRequest, LeaveType

PENDING_STATUSES = ("pending_manager", "pending_hr")
ACTIVE_STATUSES = (*PENDING_STATUSES, "approved")


# --------------------------------------------------------------------------- dates
def add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def completed_months(start: date, end: date) -> int:
    """Number of whole months between start and end (month anniversaries passed)."""
    if end <= start:
        return 0
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if add_months(start, months) > end:
        months -= 1
    return max(months, 0)


def probation_end(emp: Employee) -> date:
    return add_months(emp.date_of_joining, emp.probation_months or 0)


def service_days(emp: Employee, today: date) -> int:
    return max((today - emp.date_of_joining).days, 0)


def round_half(value: float) -> float:
    return round(value * 2) / 2


def pro_rata(quota: float, doj: date, year: int) -> float:
    """Pro-rated entitlement for someone joining during `year`.

    The joining month counts when the employee joined on or before the 15th.
    """
    if doj.year < year:
        return quota
    if doj.year > year:
        return 0.0
    months = 12 - doj.month + (1 if doj.day <= 15 else 0)
    return round_half(quota * months / 12)


# --------------------------------------------------------------------------- calendar
def holidays_between(session: Session, location_code: str, start: date, end: date) -> dict[date, str]:
    rows = session.scalars(
        select(Holiday).where(
            Holiday.holiday_date >= start,
            Holiday.holiday_date <= end,
            Holiday.optional.is_(False),
            or_(Holiday.location_code == "ALL", Holiday.location_code == location_code),
        )
    ).all()
    return {row.holiday_date: row.name for row in rows}


@dataclass
class DayCount:
    working_days: float
    calendar_days: int
    weekends: int
    holidays: dict[date, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "working_days": self.working_days,
            "calendar_days": self.calendar_days,
            "weekend_days": self.weekends,
            "public_holidays": [
                {"date": d.isoformat(), "name": n} for d, n in sorted(self.holidays.items())
            ],
        }


def count_working_days(session: Session, location_code: str, start: date, end: date) -> DayCount:
    if end < start:
        raise ValueError("end_date is before start_date")
    hols = holidays_between(session, location_code, start, end)
    weekends = 0
    working = 0
    hit_holidays: dict[date, str] = {}
    day = start
    while day <= end:
        if day.weekday() >= 5:
            weekends += 1
        elif day in hols:
            hit_holidays[day] = hols[day]
        else:
            working += 1
        day += timedelta(days=1)
    return DayCount(float(working), (end - start).days + 1, weekends, hit_holidays)


# --------------------------------------------------------------------------- balances
def _used_and_pending(session: Session, employee_id: str, leave_type: str, year: int) -> tuple[float, float]:
    rows = session.execute(
        select(LeaveRequest.status, func.coalesce(func.sum(LeaveRequest.working_days), 0.0))
        .where(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.leave_type == leave_type,
            LeaveRequest.start_date >= date(year, 1, 1),
            LeaveRequest.start_date <= date(year, 12, 31),
            LeaveRequest.status.in_(ACTIVE_STATUSES),
        )
        .group_by(LeaveRequest.status)
    ).all()
    used = sum(total for status, total in rows if status == "approved")
    pending = sum(total for status, total in rows if status in PENDING_STATUSES)
    return float(used), float(pending)


def accrued_to_date(lt: LeaveType, emp: Employee, bal: LeaveBalance | None, today: date) -> float:
    if bal is None:
        return 0.0
    if lt.accrual != "monthly":
        return bal.entitled_days
    year_start = date(bal.year, 1, 1)
    start = max(year_start, emp.date_of_joining)
    end = min(today, date(bal.year, 12, 31))
    rate = lt.annual_quota_days / 12
    return min(bal.entitled_days, round_half(completed_months(start, end) * rate))


def leave_balance(session: Session, emp: Employee, lt: LeaveType, today: date, year: int | None = None) -> dict:
    year = year or today.year
    bal = session.scalar(
        select(LeaveBalance).where(
            LeaveBalance.employee_id == emp.id,
            LeaveBalance.leave_type == lt.code,
            LeaveBalance.year == year,
        )
    )
    used, pending = _used_and_pending(session, emp.id, lt.code, year)
    row: dict = {
        "leave_type": lt.code,
        "leave_type_name": lt.name,
        "year": year,
        "accrual": lt.accrual,
        "used_days": used,
        "pending_days": pending,
        "policy_ref": lt.policy_ref,
    }
    if lt.accrual == "event":
        row.update(
            {
                "entitlement_per_event_days": lt.annual_quota_days,
                "available_days": None,
                "note": "Event-based leave: entitlement applies per qualifying event.",
            }
        )
        return row
    accrued = accrued_to_date(lt, emp, bal, today)
    carried = bal.carried_forward if bal else 0.0
    adjustment = bal.adjustment if bal else 0.0
    row.update(
        {
            "entitled_days": bal.entitled_days if bal else 0.0,
            "carried_forward_days": carried,
            "accrued_to_date_days": accrued,
            "adjustment_days": adjustment,
            "available_days": round(carried + accrued + adjustment - used - pending, 2),
        }
    )
    return row


def balances_for(session: Session, emp: Employee, today: date, year: int | None = None) -> list[dict]:
    types = session.scalars(select(LeaveType).order_by(LeaveType.code)).all()
    rows = []
    for lt in types:
        if emp.employment_type not in lt.eligible_employment_types.split(","):
            continue
        rows.append(leave_balance(session, emp, lt, today, year))
    return rows


# --------------------------------------------------------------------------- eligibility
@dataclass
class Check:
    rule: str
    passed: bool
    detail: str
    blocking: bool = True
    policy_ref: str = ""

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "passed": self.passed,
            "blocking": self.blocking,
            "detail": self.detail,
            "policy_ref": self.policy_ref,
        }


def find_overlaps(session: Session, employee_id: str, start: date, end: date, exclude_id: str | None = None):
    stmt = select(LeaveRequest).where(
        LeaveRequest.employee_id == employee_id,
        LeaveRequest.status.in_(ACTIVE_STATUSES),
        LeaveRequest.start_date <= end,
        LeaveRequest.end_date >= start,
    )
    if exclude_id:
        stmt = stmt.where(LeaveRequest.id != exclude_id)
    return session.scalars(stmt).all()


def evaluate_eligibility(
    session: Session,
    emp: Employee,
    lt: LeaveType,
    today: date,
    start: date | None = None,
    end: date | None = None,
) -> dict:
    checks: list[Check] = []
    earliest: date | None = None

    allowed_types = lt.eligible_employment_types.split(",")
    checks.append(
        Check(
            "employment_type",
            emp.employment_type in allowed_types,
            f"{lt.name} is available to: {', '.join(allowed_types)}; employee is {emp.employment_type}.",
            policy_ref=lt.policy_ref,
        )
    )

    if lt.requires_probation_complete:
        p_end = probation_end(emp)
        passed = today >= p_end
        if not passed:
            earliest = p_end
        checks.append(
            Check(
                "probation",
                passed,
                (
                    f"Probation completed on {p_end.isoformat()}."
                    if passed
                    else f"Probation ends on {p_end.isoformat()}; {lt.name} can be availed only after probation."
                ),
                policy_ref="HR-POL-001 §2.3",
            )
        )

    if lt.min_service_days:
        served = service_days(emp, today)
        passed = served >= lt.min_service_days
        if not passed:
            candidate = emp.date_of_joining + timedelta(days=lt.min_service_days)
            earliest = max(earliest, candidate) if earliest else candidate
        checks.append(
            Check(
                "minimum_service",
                passed,
                f"Requires {lt.min_service_days} days of service; employee has {served} days.",
                policy_ref=lt.policy_ref,
            )
        )

    balance = leave_balance(session, emp, lt, today, (start or today).year)
    day_count: DayCount | None = None
    if start and end:
        day_count = count_working_days(session, emp.location_code, start, end)
        days = day_count.working_days
        checks.append(Check("working_days", days > 0, f"Requested range has {days:g} working day(s).", blocking=days <= 0))

        if balance.get("available_days") is not None and lt.paid:
            available = balance["available_days"]
            checks.append(
                Check(
                    "balance",
                    available >= days,
                    f"Available {lt.name} balance is {available:g} day(s); request needs {days:g}.",
                    policy_ref=lt.policy_ref,
                )
            )
        elif lt.accrual == "event" and days > lt.annual_quota_days:
            checks.append(
                Check(
                    "event_entitlement",
                    False,
                    f"{lt.name} entitlement is {lt.annual_quota_days:g} day(s) per event; request needs {days:g}.",
                    policy_ref=lt.policy_ref,
                )
            )

        if lt.max_consecutive_days is not None and days > lt.max_consecutive_days:
            checks.append(
                Check(
                    "max_consecutive",
                    False,
                    f"{lt.name} allows at most {lt.max_consecutive_days:g} consecutive working day(s).",
                    policy_ref=lt.policy_ref,
                )
            )

        if lt.code == "SL":
            passed = start >= today - timedelta(days=7)
            checks.append(
                Check(
                    "retroactive_window",
                    passed,
                    "Sick leave can be recorded retroactively for up to 7 days.",
                    policy_ref=lt.policy_ref,
                )
            )
        elif start < today:
            checks.append(Check("start_in_future", False, "Leave (other than sick leave) cannot start in the past."))

        notice_days = (start - today).days
        if lt.advance_notice_days and notice_days < lt.advance_notice_days:
            checks.append(
                Check(
                    "advance_notice",
                    False,
                    f"Policy asks for {lt.advance_notice_days} days' notice; this request gives {max(notice_days, 0)}. "
                    "It can still be submitted but will be flagged as short notice.",
                    blocking=False,
                    policy_ref=lt.policy_ref,
                )
            )

        if lt.document_required_after_days is not None and days > lt.document_required_after_days:
            checks.append(
                Check(
                    "supporting_document",
                    True,
                    f"A supporting document is required for more than {lt.document_required_after_days:g} day(s).",
                    blocking=False,
                    policy_ref=lt.policy_ref,
                )
            )

        overlaps = find_overlaps(session, emp.id, start, end)
        checks.append(
            Check(
                "no_overlap",
                not overlaps,
                "No overlapping requests."
                if not overlaps
                else "Overlaps existing request(s): " + ", ".join(f"{o.id} ({o.status})" for o in overlaps),
            )
        )

        if lt.hr_approval_over_days is not None and days > lt.hr_approval_over_days:
            checks.append(
                Check(
                    "hr_approval",
                    True,
                    "HR approval will be required in addition to the manager.",
                    blocking=False,
                    policy_ref="HR-POL-001 §6.2",
                )
            )

    eligible = all(c.passed for c in checks if c.blocking)
    result = {
        "employee_id": emp.id,
        "employee_name": emp.full_name,
        "leave_type": lt.code,
        "leave_type_name": lt.name,
        "paid": lt.paid,
        "eligible": eligible,
        "checks": [c.as_dict() for c in checks],
        "blocking_reasons": [c.detail for c in checks if c.blocking and not c.passed],
        "warnings": [c.detail for c in checks if not c.blocking and (not c.passed or c.rule != "working_days")],
        "balance": balance,
        "policy_refs": sorted({c.policy_ref for c in checks if c.policy_ref} | {lt.policy_ref}),
        "as_of": today.isoformat(),
    }
    if earliest:
        result["earliest_eligible_date"] = earliest.isoformat()
    if day_count:
        result["date_range"] = {"start_date": start.isoformat(), "end_date": end.isoformat(), **day_count.as_dict()}
    return result


# --------------------------------------------------------------------------- workflow
def approval_levels(emp: Employee, lt: LeaveType, days: float) -> list[str]:
    levels = ["manager"] if emp.manager_id else []
    if lt.hr_approval_over_days is not None and days > lt.hr_approval_over_days:
        levels.append("hr")
    return levels or ["hr"]


def hr_approver_for(session: Session, emp: Employee) -> Employee | None:
    """HR approver: the most recently created active hr_admin who is not the requester
    (in the seed data that is the HR business partner, falling back to the CPO)."""
    return session.scalars(
        select(Employee)
        .where(Employee.role == "hr_admin", Employee.id != emp.id, Employee.status == "active")
        .order_by(Employee.id.desc())
    ).first()


def next_request_id(session: Session, year: int) -> str:
    prefix = f"LR-{year}-"
    ids = session.scalars(select(LeaveRequest.id).where(LeaveRequest.id.like(f"{prefix}%"))).all()
    highest = max((int(i.rsplit("-", 1)[1]) for i in ids), default=0)
    return f"{prefix}{highest + 1:04d}"


def request_to_dict(req: LeaveRequest) -> dict:
    return {
        "request_id": req.id,
        "employee_id": req.employee_id,
        "employee_name": req.employee.full_name if req.employee else None,
        "leave_type": req.leave_type,
        "start_date": req.start_date.isoformat(),
        "end_date": req.end_date.isoformat(),
        "working_days": req.working_days,
        "status": req.status,
        "current_approver_id": req.current_approver_id,
        "current_approver_name": req.approver.full_name if req.approver else None,
        "reason": req.reason,
        "created_at": req.created_at.isoformat() if req.created_at else None,
    }
