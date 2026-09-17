from datetime import date

from agentic_core.db.models import Employee, LeaveType
from agentic_core.db.session import session_scope
from agentic_core.domain import leave as rules

from .conftest import TODAY


def test_seed_counts(service):
    with session_scope() as s:
        assert s.query(Employee).count() == 20
        assert s.query(LeaveType).count() == 7


def test_working_days_skip_weekends_and_location_holidays(service):
    with session_scope() as s:
        # 9-13 Nov 2026 in Mumbai: Diwali (9th, all) and Balipratipada (10th, MUM) are holidays
        count = rules.count_working_days(s, "MUM", date(2026, 11, 9), date(2026, 11, 15))
        assert count.working_days == 3
        assert count.weekends == 2
        assert set(count.holidays) == {date(2026, 11, 9), date(2026, 11, 10)}
        # Gurugram does not observe Balipratipada
        assert rules.count_working_days(s, "DEL", date(2026, 11, 9), date(2026, 11, 13)).working_days == 4


def test_completed_months_and_pro_rata():
    assert rules.completed_months(date(2026, 1, 1), date(2026, 9, 17)) == 8
    assert rules.completed_months(date(2026, 6, 15), date(2026, 9, 17)) == 3
    assert rules.completed_months(date(2026, 6, 15), date(2026, 9, 14)) == 2
    assert rules.pro_rata(18, date(2026, 6, 15), 2026) == 10.5  # joined on/before the 15th: June counts
    assert rules.pro_rata(18, date(2026, 6, 16), 2026) == 9.0
    assert rules.pro_rata(18, date(2020, 1, 1), 2026) == 18


def test_annual_leave_balance_for_long_tenured_employee(service):
    with session_scope() as s:
        aarav = s.get(Employee, "E1001")
        al = s.get(LeaveType, "AL")
        bal = rules.leave_balance(s, aarav, al, TODAY)
        # 6 carried + 12 accrued (8 months x 1.5) - 4 used (Holi fell inside LR-0001) - 5 pending
        assert bal["carried_forward_days"] == 6
        assert bal["accrued_to_date_days"] == 12
        assert bal["used_days"] == 4
        assert bal["pending_days"] == 5
        assert bal["available_days"] == 9


def test_new_joiner_cannot_use_annual_leave_during_probation(service):
    with session_scope() as s:
        priya = s.get(Employee, "E1002")
        result = rules.evaluate_eligibility(s, priya, s.get(LeaveType, "AL"), TODAY)
        assert result["eligible"] is False
        assert result["earliest_eligible_date"] == "2026-12-15"
        assert "HR-POL-001 §2.3" in result["policy_refs"]
        # Sick leave is fine during probation
        assert rules.evaluate_eligibility(s, priya, s.get(LeaveType, "SL"), TODAY)["eligible"] is True


def test_casual_leave_limit_and_overlap_checks(service):
    with session_scope() as s:
        aarav = s.get(Employee, "E1001")
        cl = s.get(LeaveType, "CL")
        too_long = rules.evaluate_eligibility(s, aarav, cl, TODAY, date(2026, 10, 5), date(2026, 10, 7))
        assert too_long["eligible"] is False
        assert any("at most 2" in r for r in too_long["blocking_reasons"])
        overlap = rules.evaluate_eligibility(s, aarav, s.get(LeaveType, "AL"), TODAY, date(2026, 10, 28), date(2026, 10, 29))
        assert overlap["eligible"] is False
        assert any("LR-2026-0008" in r for r in overlap["blocking_reasons"])


def test_short_notice_is_a_warning_not_a_blocker(service):
    with session_scope() as s:
        aarav = s.get(Employee, "E1001")
        result = rules.evaluate_eligibility(s, aarav, s.get(LeaveType, "AL"), TODAY, date(2026, 9, 21), date(2026, 9, 22))
        assert result["eligible"] is True
        assert any("notice" in w for w in result["warnings"])


def test_approval_levels(service):
    with session_scope() as s:
        aarav = s.get(Employee, "E1001")
        coo = s.get(Employee, "E1011")
        assert rules.approval_levels(aarav, s.get(LeaveType, "AL"), 5) == ["manager"]
        assert rules.approval_levels(aarav, s.get(LeaveType, "AL"), 11) == ["manager", "hr"]
        assert rules.approval_levels(aarav, s.get(LeaveType, "PPL"), 60) == ["manager", "hr"]
        assert rules.approval_levels(coo, s.get(LeaveType, "AL"), 2) == ["hr"]  # no manager -> HR
        assert rules.next_request_id(s, 2026) == "LR-2026-0013"
