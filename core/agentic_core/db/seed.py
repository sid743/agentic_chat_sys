"""Synthetic seed data for the dummy HR database.

Everything here is fictional and labelled as demo data. Emails use example.com.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..domain.leave import count_working_days, pro_rata
from .models import (
    ApprovalStep,
    Department,
    Employee,
    Holiday,
    LeaveBalance,
    LeaveRequest,
    LeaveType,
    Location,
    Notification,
)
from .session import reset_schema, session_scope

log = logging.getLogger(__name__)

SEED_YEAR = 2026

DEPARTMENTS = [
    ("ENG", "Engineering", "E1005"),
    ("FIN", "Finance", "E1009"),
    ("HR", "Human Resources", "E1000"),
    ("SAL", "Sales", "E1008"),
    ("OPS", "Operations", "E1011"),
    ("MKT", "Marketing", "E1017"),
]

LOCATIONS = [
    ("MUM", "Mumbai", "Maharashtra"),
    ("PUN", "Pune", "Maharashtra"),
    ("BLR", "Bengaluru", "Karnataka"),
    ("HYD", "Hyderabad", "Telangana"),
    ("DEL", "Gurugram", "Haryana"),
]

# id, first, last, title, dept, location, manager, role, employment type, date of joining
EMPLOYEES = [
    ("E1000", "Kavita", "Rao", "Chief People Officer", "HR", "MUM", None, "hr_admin", "full_time", "2015-04-01"),
    ("E1011", "Meera", "Joshi", "Chief Operating Officer", "OPS", "MUM", None, "manager", "full_time", "2014-01-06"),
    ("E1005", "Neha", "Kapoor", "Engineering Manager", "ENG", "MUM", "E1011", "manager", "full_time", "2019-09-02"),
    ("E1008", "Vikram", "Singh", "Regional Sales Manager", "SAL", "BLR", "E1011", "manager", "full_time", "2018-06-04"),
    ("E1009", "Ananya", "Iyer", "Finance Controller", "FIN", "MUM", "E1011", "manager", "full_time", "2017-11-13"),
    ("E1017", "Pooja", "Patil", "Marketing Manager", "MKT", "PUN", "E1011", "manager", "full_time", "2020-03-09"),
    ("E1010", "Rohan", "Desai", "HR Business Partner", "HR", "MUM", "E1000", "hr_admin", "full_time", "2020-08-24"),
    ("E1001", "Aarav", "Mehta", "Senior Software Engineer", "ENG", "MUM", "E1005", "employee", "full_time", "2022-07-11"),
    ("E1002", "Priya", "Nair", "Data Analyst", "ENG", "MUM", "E1005", "employee", "full_time", "2026-06-15"),
    ("E1003", "John", "Smith", "Account Executive", "SAL", "BLR", "E1008", "employee", "full_time", "2021-02-01"),
    ("E1004", "Sneha", "Kulkarni", "QA Engineer", "ENG", "PUN", "E1005", "employee", "full_time", "2024-01-08"),
    ("E1006", "Arjun", "Reddy", "Backend Engineer", "ENG", "HYD", "E1005", "employee", "full_time", "2025-03-17"),
    ("E1007", "Fatima", "Shaikh", "Financial Analyst", "FIN", "MUM", "E1009", "employee", "full_time", "2023-05-22"),
    ("E1012", "Karan", "Malhotra", "Sales Development Representative", "SAL", "DEL", "E1008", "employee", "contract", "2025-11-03"),
    ("E1013", "Isha", "Banerjee", "Software Engineering Intern", "ENG", "BLR", "E1005", "employee", "intern", "2026-07-01"),
    ("E1014", "David", "Fernandes", "DevOps Engineer", "ENG", "PUN", "E1005", "employee", "full_time", "2021-10-18"),
    ("E1015", "Lakshmi", "Menon", "Payroll Specialist", "HR", "BLR", "E1010", "employee", "full_time", "2022-02-14"),
    ("E1016", "Sameer", "Khan", "Operations Analyst", "OPS", "HYD", "E1011", "employee", "full_time", "2024-08-05"),
    ("E1018", "Nikhil", "Gupta", "Content Strategist", "MKT", "DEL", "E1017", "employee", "full_time", "2025-06-30"),
    ("E1019", "Emily", "Carter", "Solutions Consultant", "SAL", "BLR", "E1008", "employee", "full_time", "2023-09-11"),
]

# Leave-type rulebook mirrors data/policies/*.md
LEAVE_TYPES = [
    dict(
        code="AL", name="Annual Leave", paid=True, annual_quota_days=18, accrual="monthly",
        requires_probation_complete=True, min_service_days=0, max_consecutive_days=None,
        advance_notice_days=7, carry_forward_max=10, hr_approval_over_days=10,
        eligible_employment_types="full_time,contract", document_required_after_days=None,
        policy_ref="HR-POL-001 §3",
        description="18 days per year, accrues 1.5 days per completed month; usable after probation.",
    ),
    dict(
        code="SL", name="Sick Leave", paid=True, annual_quota_days=10, accrual="upfront",
        requires_probation_complete=False, min_service_days=0, max_consecutive_days=None,
        advance_notice_days=0, carry_forward_max=0, hr_approval_over_days=10,
        eligible_employment_types="full_time,contract,intern", document_required_after_days=2,
        policy_ref="HR-POL-001 §4",
        description="10 days per year credited in January (pro-rated for joiners).",
    ),
    dict(
        code="CL", name="Casual Leave", paid=True, annual_quota_days=6, accrual="upfront",
        requires_probation_complete=False, min_service_days=0, max_consecutive_days=2,
        advance_notice_days=1, carry_forward_max=0, hr_approval_over_days=None,
        eligible_employment_types="full_time,contract", document_required_after_days=None,
        policy_ref="HR-POL-001 §5",
        description="6 days per year for short personal needs; max 2 consecutive days.",
    ),
    dict(
        code="PPL", name="Parental Leave - Primary Caregiver", paid=True, annual_quota_days=182,
        accrual="event", requires_probation_complete=False, min_service_days=80,
        max_consecutive_days=None, advance_notice_days=30, carry_forward_max=0, hr_approval_over_days=0,
        eligible_employment_types="full_time,contract", document_required_after_days=0,
        policy_ref="HR-POL-002 §4.1",
        description="26 weeks paid leave per birth/adoption event for the primary caregiver.",
    ),
    dict(
        code="SPL", name="Parental Leave - Secondary Caregiver", paid=True, annual_quota_days=15,
        accrual="event", requires_probation_complete=False, min_service_days=90,
        max_consecutive_days=15, advance_notice_days=14, carry_forward_max=0, hr_approval_over_days=0,
        eligible_employment_types="full_time,contract", document_required_after_days=None,
        policy_ref="HR-POL-002 §4.2",
        description="15 working days within six months of the birth/adoption.",
    ),
    dict(
        code="BL", name="Bereavement Leave", paid=True, annual_quota_days=5, accrual="event",
        requires_probation_complete=False, min_service_days=0, max_consecutive_days=5,
        advance_notice_days=0, carry_forward_max=0, hr_approval_over_days=None,
        eligible_employment_types="full_time,contract", document_required_after_days=None,
        policy_ref="HR-POL-001 §7",
        description="Up to 5 working days on the death of an immediate family member.",
    ),
    dict(
        code="LWP", name="Leave Without Pay", paid=False, annual_quota_days=30, accrual="upfront",
        requires_probation_complete=False, min_service_days=0, max_consecutive_days=30,
        advance_notice_days=14, carry_forward_max=0, hr_approval_over_days=0,
        eligible_employment_types="full_time,contract", document_required_after_days=None,
        policy_ref="HR-POL-001 §8",
        description="Unpaid leave up to 30 days per year, after paid leave is exhausted; HR approval required.",
    ),
]

AL_CARRY_FORWARD = {
    "E1000": 10, "E1011": 10, "E1005": 8, "E1008": 6, "E1009": 10, "E1017": 4, "E1010": 7,
    "E1001": 6, "E1003": 10, "E1004": 3, "E1006": 0, "E1007": 5, "E1014": 9, "E1015": 2,
    "E1016": 1, "E1018": 0, "E1019": 4,
}

# Synthetic holiday calendar for the demo (not an official list).
HOLIDAYS = [
    ("2026-01-01", "New Year's Day", "ALL", True),
    ("2026-01-14", "Makar Sankranti / Pongal", "BLR", False),
    ("2026-01-14", "Makar Sankranti / Pongal", "HYD", False),
    ("2026-01-26", "Republic Day", "ALL", False),
    ("2026-03-04", "Holi", "ALL", False),
    ("2026-03-20", "Eid al-Fitr", "ALL", True),
    ("2026-04-03", "Good Friday", "ALL", False),
    ("2026-05-01", "Maharashtra Day", "MUM", False),
    ("2026-05-01", "Maharashtra Day", "PUN", False),
    ("2026-05-01", "May Day", "BLR", False),
    ("2026-05-27", "Bakri Eid", "ALL", True),
    ("2026-08-15", "Independence Day", "ALL", False),
    ("2026-09-14", "Ganesh Chaturthi", "MUM", False),
    ("2026-09-14", "Ganesh Chaturthi", "PUN", False),
    ("2026-09-14", "Ganesh Chaturthi", "HYD", False),
    ("2026-10-02", "Gandhi Jayanti", "ALL", False),
    ("2026-10-20", "Dussehra", "ALL", False),
    ("2026-11-09", "Diwali (observed)", "ALL", False),
    ("2026-11-10", "Diwali - Balipratipada", "MUM", False),
    ("2026-11-10", "Diwali - Balipratipada", "PUN", False),
    ("2026-11-10", "Diwali - Balipratipada", "BLR", False),
    ("2026-11-24", "Guru Nanak Jayanti", "DEL", False),
    ("2026-12-25", "Christmas Day", "ALL", False),
    ("2027-01-01", "New Year's Day", "ALL", True),
    ("2027-01-26", "Republic Day", "ALL", False),
]

# id, employee, type, start, end, status, approver chain [(level, approver, decision, comment, decided_at)], reason, created_at
LEAVE_REQUESTS = [
    ("LR-2026-0001", "E1001", "AL", "2026-03-02", "2026-03-06", "approved",
     [("manager", "E1005", "approved", "Enjoy the break!", "2026-02-20 10:12")], "Family trip to Goa", "2026-02-18 09:30"),
    ("LR-2026-0002", "E1001", "SL", "2026-05-18", "2026-05-19", "approved",
     [("manager", "E1005", "approved", "Get well soon.", "2026-05-20 11:00")], "Viral fever", "2026-05-18 08:05"),
    ("LR-2026-0003", "E1003", "AL", "2026-04-13", "2026-04-27", "approved",
     [("manager", "E1008", "approved", "Approved - hand over the Q2 pipeline to Emily.", "2026-03-30 16:40"),
      ("hr", "E1010", "approved", "OK per policy.", "2026-03-31 10:05")], "Visiting family in the UK", "2026-03-27 12:00"),
    ("LR-2026-0004", "E1003", "CL", "2026-08-07", "2026-08-07", "approved",
     [("manager", "E1008", "approved", "", "2026-08-05 09:15")], "Personal errand", "2026-08-04 18:20"),
    ("LR-2026-0005", "E1004", "AL", "2026-10-19", "2026-10-23", "pending_manager", [], "Dussehra holidays with family", "2026-09-10 14:02"),
    ("LR-2026-0006", "E1006", "SL", "2026-09-08", "2026-09-09", "approved",
     [("manager", "E1005", "approved", "", "2026-09-10 09:00")], "Dental procedure", "2026-09-08 07:45"),
    ("LR-2026-0007", "E1014", "AL", "2026-11-11", "2026-11-13", "pending_manager", [], "Diwali travel", "2026-09-12 19:30"),
    ("LR-2026-0008", "E1001", "AL", "2026-10-26", "2026-10-30", "pending_manager", [], "Trek in Himachal", "2026-09-15 11:11"),
    ("LR-2026-0009", "E1007", "AL", "2026-12-21", "2026-12-31", "pending_manager", [], "Year-end vacation", "2026-09-16 10:00"),
    ("LR-2026-0010", "E1019", "AL", "2026-06-29", "2026-07-03", "rejected",
     [("manager", "E1008", "rejected", "Quarter-end closing week - please pick another week.", "2026-06-10 12:30")],
     "Short holiday", "2026-06-08 15:00"),
    ("LR-2026-0011", "E1012", "CL", "2026-09-25", "2026-09-25", "pending_manager", [], "Moving house", "2026-09-16 17:45"),
    ("LR-2026-0012", "E1016", "PPL", "2026-11-02", "2027-04-30", "pending_hr",
     [("manager", "E1011", "approved", "Congratulations! Approved from my side.", "2026-09-02 10:20")],
     "Primary caregiver leave for adoption", "2026-08-28 09:00"),
]

NOTIFICATIONS = [
    ("E1005", "E1001", "email", "Leave request LR-2026-0008 awaiting your approval",
     "Aarav Mehta requested Annual Leave from 2026-10-26 to 2026-10-30 (5 working days).", "LR-2026-0008", "2026-09-15 11:12"),
    ("E1001", None, "in_app", "Leave request LR-2026-0008 submitted",
     "Your Annual Leave request is pending approval by Neha Kapoor.", "LR-2026-0008", "2026-09-15 11:12"),
    ("E1010", "E1011", "email", "LR-2026-0012 needs HR approval",
     "Sameer Khan's primary caregiver leave was approved by the manager and now needs HR approval.",
     "LR-2026-0012", "2026-09-02 10:21"),
    ("E1019", "E1008", "email", "Leave request LR-2026-0010 rejected",
     "Your Annual Leave request was rejected: Quarter-end closing week - please pick another week.",
     "LR-2026-0010", "2026-06-10 12:31"),
]


def _d(value: str) -> date:
    return date.fromisoformat(value)


def _dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def seed(session: Session) -> dict:
    for code, name, head in DEPARTMENTS:
        session.add(Department(code=code, name=name, head_employee_id=head))
    for code, city, state in LOCATIONS:
        session.add(Location(code=code, city=city, state=state))
    session.flush()

    for emp_id, first, last, title, dept, loc, mgr, role, etype, doj in EMPLOYEES:
        session.add(
            Employee(
                id=emp_id,
                first_name=first,
                last_name=last,
                email=f"{first.lower()}.{last.lower()}@example.com",
                title=title,
                department_code=dept,
                location_code=loc,
                manager_id=mgr,
                role=role,
                employment_type=etype,
                date_of_joining=_d(doj),
                probation_months=6,
            )
        )
    session.flush()

    for spec in LEAVE_TYPES:
        session.add(LeaveType(**spec))
    for day, name, loc, optional in HOLIDAYS:
        session.add(Holiday(holiday_date=_d(day), name=name, location_code=loc, optional=optional))
    session.flush()

    employees = {e.id: e for e in session.scalars(select(Employee)).all()}
    leave_types = {lt.code: lt for lt in session.scalars(select(LeaveType)).all()}

    for emp in employees.values():
        for lt in leave_types.values():
            if lt.accrual == "event":
                continue
            if emp.employment_type not in lt.eligible_employment_types.split(","):
                continue
            entitled = pro_rata(lt.annual_quota_days, emp.date_of_joining, SEED_YEAR)
            carried = float(AL_CARRY_FORWARD.get(emp.id, 0)) if lt.code == "AL" else 0.0
            session.add(
                LeaveBalance(
                    employee_id=emp.id,
                    leave_type=lt.code,
                    year=SEED_YEAR,
                    entitled_days=entitled,
                    carried_forward=carried,
                    adjustment=0.0,
                )
            )
    session.flush()

    for req_id, emp_id, lt_code, start, end, status, chain, reason, created in LEAVE_REQUESTS:
        emp = employees[emp_id]
        days = count_working_days(session, emp.location_code, _d(start), _d(end)).working_days
        approver = None
        if status == "pending_manager":
            approver = emp.manager_id
        elif status == "pending_hr":
            approver = "E1010"
        session.add(
            LeaveRequest(
                id=req_id,
                employee_id=emp_id,
                leave_type=lt_code,
                start_date=_d(start),
                end_date=_d(end),
                working_days=days,
                reason=reason,
                status=status,
                current_approver_id=approver,
                created_at=_dt(created),
                updated_at=_dt(chain[-1][4]) if chain else _dt(created),
                source="seed",
            )
        )
        session.flush()
        for level, approver_id, decision, comment, decided in chain:
            session.add(
                ApprovalStep(
                    request_id=req_id,
                    level=level,
                    approver_id=approver_id,
                    decision=decision,
                    comment=comment,
                    decided_at=_dt(decided),
                )
            )

    for recipient, sender, channel, subject, body, related, created in NOTIFICATIONS:
        session.add(
            Notification(
                recipient_id=recipient,
                sender_id=sender,
                channel=channel,
                subject=subject,
                body=body,
                related_request_id=related,
                status="sent",
                created_by_agent="seed",
                created_at=_dt(created),
            )
        )

    return {
        "employees": len(EMPLOYEES),
        "leave_types": len(LEAVE_TYPES),
        "holidays": len(HOLIDAYS),
        "leave_requests": len(LEAVE_REQUESTS),
        "notifications": len(NOTIFICATIONS),
    }


def is_seeded() -> bool:
    with session_scope() as session:
        return session.scalar(select(Employee.id).limit(1)) is not None


def seed_database(reset: bool = False) -> dict:
    """Create tables and load the synthetic data. Idempotent unless reset=True."""
    if reset:
        reset_schema()
    if not reset and is_seeded():
        return {"skipped": True}
    with session_scope() as session:
        counts = seed(session)
    log.info("Seeded dummy HR database: %s", counts)
    return counts
