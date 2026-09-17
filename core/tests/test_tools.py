import asyncio

import pytest

from agentic_core.db.models import AuditLog, LeaveRequest, Notification
from agentic_core.db.session import session_scope
from agentic_core.tools import REGISTRY, execute_tool

from .conftest import tool_context


def run(service, employee, tool, **args):
    return asyncio.run(execute_tool(tool, tool_context(service, employee), args))


def test_every_tool_has_a_valid_schema():
    for spec in REGISTRY.values():
        schema = spec.openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"


def test_employee_cannot_see_colleague_balance(service):
    result = run(service, "E1001", "get_leave_balance", employee_id="E1003")
    assert result["ok"] is False and result["error"] == "forbidden"
    assert result["target_name"] == "John Smith"
    with session_scope() as s:
        denied = s.query(AuditLog).filter_by(action="view_leave_balance", outcome="denied").one()
        assert denied.actor_id == "E1001" and denied.target_id == "E1003"


def test_manager_and_hr_can_see_report_balance(service):
    assert run(service, "E1008", "get_leave_balance", employee_id="E1003")["ok"] is True  # John's manager
    assert run(service, "E1010", "get_leave_balance", employee_id="E1003")["ok"] is True  # HR admin
    assert run(service, "E1005", "get_leave_balance", employee_id="E1003")["ok"] is False  # other manager


def test_directory_lookup_has_no_restricted_fields(service):
    result = run(service, "E1001", "find_employee", query="john smith")
    assert result["count"] == 1
    assert set(result["matches"][0]) == {"employee_id", "name", "title", "department", "location", "manager_name", "email"}


def test_create_request_happy_path_and_events(service):
    result = run(service, "E1001", "create_leave_request", leave_type="annual leave",
                 start_date="2026-11-11", end_date="2026-11-13", reason="Family function")
    assert result["ok"] is True
    req = result["request"]
    assert req["request_id"] == "LR-2026-0013"
    assert req["status"] == "pending_manager" and req["current_approver_id"] == "E1005"
    notify = result["events"][0]["notify"]
    assert {n["employee_id"] for n in notify} == {"E1005", "E1001"}


def test_probation_blocks_annual_leave_request(service):
    result = run(service, "E1002", "create_leave_request", leave_type="AL", start_date="2026-10-12", end_date="2026-10-13")
    assert result["ok"] is False and result["error"] == "not_eligible"
    assert any("probation" in r.lower() for r in result["reasons"])
    with session_scope() as s:
        assert s.query(LeaveRequest).filter_by(employee_id="E1002").count() == 0


def test_long_leave_needs_manager_then_hr(service):
    created = run(service, "E1014", "create_leave_request", leave_type="AL", start_date="2026-12-01", end_date="2026-12-15")
    assert created["ok"] is True and created["approval_levels"] == ["manager", "hr"]
    rid = created["request"]["request_id"]
    step1 = run(service, "E1005", "approve_leave_request", request_id=rid)
    assert step1["request"]["status"] == "pending_hr"
    assert step1["request"]["current_approver_id"] == "E1010"
    # A manager cannot do the HR step
    assert run(service, "E1005", "approve_leave_request", request_id=rid)["error"] == "forbidden"
    step2 = run(service, "E1010", "approve_leave_request", request_id=rid, comment="ok")
    assert step2["request"]["status"] == "approved"


def test_self_approval_and_non_approver_are_blocked(service):
    created = run(service, "E1005", "create_leave_request", leave_type="CL", start_date="2026-09-25", end_date="2026-09-25")
    rid = created["request"]["request_id"]
    assert created["request"]["current_approver_id"] == "E1011"
    assert run(service, "E1005", "approve_leave_request", request_id=rid)["policy_ref"] == "HR-POL-001 §6.3"
    assert run(service, "E1001", "approve_leave_request", request_id="LR-2026-0005")["error"] == "forbidden"
    assert run(service, "E1011", "approve_leave_request", request_id=rid)["request"]["status"] == "approved"


def test_reject_requires_reason_and_cancel_rules(service):
    assert run(service, "E1005", "reject_leave_request", request_id="LR-2026-0005", reason=" ")["error"] == "invalid_request"
    rejected = run(service, "E1005", "reject_leave_request", request_id="LR-2026-0005", reason="Release freeze")
    assert rejected["request"]["status"] == "rejected"
    # Someone else's request cannot be cancelled; own pending one can
    assert run(service, "E1001", "cancel_leave_request", request_id="LR-2026-0009")["error"] == "forbidden"
    cancelled = run(service, "E1001", "cancel_leave_request", request_id="LR-2026-0008")
    assert cancelled["request"]["status"] == "cancelled"
    # Leave that already happened cannot be cancelled by the employee
    assert run(service, "E1001", "cancel_leave_request", request_id="LR-2026-0001")["error"] == "invalid_state"


def test_notification_permissions(service):
    ok = run(service, "E1001", "send_notification", recipient="my_manager", subject="Hi", message="FYI")
    assert ok["ok"] and ok["notification"]["recipient_id"] == "E1005"
    blocked = run(service, "E1001", "send_notification", recipient="E1019", subject="Hi", message="Spam")
    assert blocked["error"] == "forbidden"
    related = run(service, "E1005", "send_notification", recipient="requester", subject="About your leave",
                  message="Approved soon", related_request_id="LR-2026-0005")
    assert related["notification"]["recipient_id"] == "E1004"
    with session_scope() as s:
        assert s.query(Notification).filter_by(created_by_agent="test_agent").count() == 2


def test_listing_scopes(service):
    assert run(service, "E1001", "list_leave_requests", scope="team")["error"] == "forbidden"
    awaiting = run(service, "E1005", "list_leave_requests", scope="awaiting_my_approval")
    assert {r["request_id"] for r in awaiting["requests"]} == {"LR-2026-0005", "LR-2026-0007", "LR-2026-0008"}
    assert run(service, "E1010", "list_leave_requests", scope="all", status="pending")["count"] == 6


@pytest.mark.parametrize(
    "tool, args, error",
    [
        ("create_leave_request", {"leave_type": "ZZ", "start_date": "2026-10-01", "end_date": "2026-10-01"}, "invalid_arguments"),
        ("create_leave_request", {"leave_type": "AL", "start_date": "not a date", "end_date": "2026-10-01"}, "invalid_arguments"),
        ("approve_leave_request", {"request_id": "LR-2026-9999"}, "invalid_request"),
        ("no_such_tool", {}, "unknown_tool"),
    ],
)
def test_invalid_calls_return_errors(service, tool, args, error):
    assert run(service, "E1001", tool, **args)["error"] == error


def test_lenient_date_and_leave_type_parsing(service):
    result = run(service, "E1001", "calculate_leave_days", start_date="16 Nov 2026", end_date="2026-11-20")
    assert result["working_days"] == 5
    result = run(service, "E1001", "check_leave_eligibility", leave_type="Sick Leave")
    assert result["leave_type"] == "SL"
