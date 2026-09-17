"""Small, dependency-free text helpers used by the heuristic router and the
offline mock model (dates, leave types, request ids, names)."""

from __future__ import annotations

import re
from datetime import date, timedelta

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}

_MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DAY_RE = r"(\d{1,2})(?:st|nd|rd|th)?"

REQUEST_ID_RE = re.compile(r"\bLR-\d{4}-\d{4}\b", re.IGNORECASE)
EMPLOYEE_ID_RE = re.compile(r"\bE\d{4}\b", re.IGNORECASE)


def request_ids(text: str) -> list[str]:
    return [m.upper() for m in REQUEST_ID_RE.findall(text or "")]


def employee_ids(text: str) -> list[str]:
    return [m.upper() for m in EMPLOYEE_ID_RE.findall(text or "")]


def _year_for(month: int, day: int, today: date, explicit: str | None) -> int:
    if explicit:
        year = int(explicit)
        return year + 2000 if year < 100 else year
    candidate = date(today.year, month, min(day, 28 if month == 2 else day))
    # Dates more than a month in the past most likely mean next year.
    return today.year + 1 if candidate < today - timedelta(days=31) else today.year


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_dates(text: str, today: date) -> list[date]:
    """Find dates in free text, in order of appearance."""
    if not text:
        return []
    lowered = text.lower()
    found: list[tuple[int, date]] = []

    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered):
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append((m.start(), d))

    # 12/10/2026 or 12-10-2026 (day first, en-IN)
    for m in re.finditer(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b", lowered):
        year = int(m.group(3))
        year = year + 2000 if year < 100 else year
        d = _safe_date(year, int(m.group(2)), int(m.group(1)))
        if d:
            found.append((m.start(), d))

    # "12-14 Nov", "12 to 14 November 2026", "12 and 14 nov"
    for m in re.finditer(
        rf"\b{_DAY_RE}\s*(?:-|–|to|till|until|and|through)\s*{_DAY_RE}\s+(?:of\s+)?{_MONTH_RE}\.?(?:,?\s+(\d{{4}}))?",
        lowered,
    ):
        month = MONTHS[m.group(3)[:3] if m.group(3)[:4] != "sept" else "sep"]
        year = _year_for(month, int(m.group(1)), today, m.group(4))
        for idx, day_str in ((m.start(1), m.group(1)), (m.start(2), m.group(2))):
            d = _safe_date(year, month, int(day_str))
            if d:
                found.append((idx, d))

    # "Nov 12-14"
    for m in re.finditer(rf"\b{_MONTH_RE}\.?\s+{_DAY_RE}\s*(?:-|–|to)\s*{_DAY_RE}\b(?:,?\s+(\d{{4}}))?", lowered):
        month = MONTHS[m.group(1)[:3]]
        year = _year_for(month, int(m.group(2)), today, m.group(4))
        for idx, day_str in ((m.start(2), m.group(2)), (m.start(3), m.group(3))):
            d = _safe_date(year, month, int(day_str))
            if d:
                found.append((idx, d))

    # "12 Oct 2026", "12th October"
    for m in re.finditer(rf"\b{_DAY_RE}\s+(?:of\s+)?{_MONTH_RE}\b\.?(?:,?\s+(\d{{4}}))?", lowered):
        month = MONTHS[m.group(2)[:3]]
        year = _year_for(month, int(m.group(1)), today, m.group(3))
        d = _safe_date(year, month, int(m.group(1)))
        if d:
            found.append((m.start(), d))

    # "Oct 12", "October 12th, 2026"
    for m in re.finditer(rf"\b{_MONTH_RE}\.?\s+{_DAY_RE}\b(?:,?\s+(\d{{4}}))?", lowered):
        month = MONTHS[m.group(1)[:3]]
        year = _year_for(month, int(m.group(2)), today, m.group(3))
        d = _safe_date(year, month, int(m.group(2)))
        if d:
            found.append((m.start(), d))

    relative = {
        "day after tomorrow": today + timedelta(days=2),
        "tomorrow": today + timedelta(days=1),
        "today": today,
    }
    for phrase, d in relative.items():
        idx = lowered.find(phrase)
        if idx != -1 and not any(abs(idx - pos) < 3 for pos, _ in found):
            found.append((idx, d))
            if phrase == "day after tomorrow":
                lowered = lowered.replace("tomorrow", "        ")

    for m in re.finditer(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
        target = WEEKDAYS[m.group(1)]
        delta = (target - today.weekday()) % 7 or 7
        found.append((m.start(), today + timedelta(days=delta)))

    m = re.search(r"\bnext week\b", lowered)
    if m:
        monday = today + timedelta(days=(7 - today.weekday()))
        found.append((m.start(), monday))
        found.append((m.start() + 1, monday + timedelta(days=4)))

    # de-duplicate while keeping order of first appearance
    seen: set[date] = set()
    ordered = []
    for _, d in sorted(found, key=lambda item: item[0]):
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def date_range(text: str, today: date) -> tuple[date, date] | None:
    dates = parse_dates(text, today)
    if not dates:
        return None
    if len(dates) == 1:
        return dates[0], dates[0]
    start, end = dates[0], dates[1]
    if end < start:
        start, end = end, start
    return start, end


LEAVE_TYPE_KEYWORDS = [
    ("PPL", ("maternity", "primary caregiver", "primary-caregiver", "adoption leave", "adopting")),
    ("SPL", ("paternity", "secondary caregiver", "secondary-caregiver")),
    ("BL", ("bereavement", "funeral", "passed away", "death in")),
    ("LWP", ("unpaid", "without pay", "lwp", "loss of pay")),
    ("SL", ("sick", "unwell", "fever", "doctor", "medical", "ill ")),
    ("CL", ("casual",)),
    ("AL", ("annual", "privilege", "vacation", "holiday trip", "earned leave", "paid leave", " pl ", " al ")),
]


def guess_leave_type(text: str, default: str | None = "AL") -> str | None:
    lowered = f" {(text or '').lower()} "
    if "parental" in lowered and not any(k in lowered for k in ("primary", "secondary", "maternity", "paternity")):
        return "PPL"
    for code, words in LEAVE_TYPE_KEYWORDS:
        if any(w in lowered for w in words):
            return code
    return default


_NAME_STOP = {
    "annual", "sick", "casual", "parental", "leave", "policy", "code", "conduct", "information", "handling",
    "hr", "faq", "i", "can", "what", "how", "is", "my", "the", "please", "hi", "hello", "thanks", "thank",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "primary", "secondary", "caregiver", "bereavement", "without", "pay", "request", "approve",
    "reject", "cancel", "notify", "tell", "show", "list", "give", "does", "do", "has", "have", "am",
    "diwali", "holi", "christmas", "dussehra", "good", "independence", "republic", "day",
}


def person_names(text: str) -> list[str]:
    """Capitalised 'First Last' pairs that are probably people's names."""
    names = []
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text or ""):
        first, last = m.group(1), m.group(2)
        if first.lower() in _NAME_STOP or last.lower() in _NAME_STOP:
            continue
        names.append(f"{first} {last}")
    return names


def keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def has_any(text: str, phrases: tuple[str, ...] | list[str]) -> bool:
    lowered = (text or "").lower()
    return any(p in lowered for p in phrases)


OVERRIDE_PATTERNS = (
    r"\bignore\b.*\b(policy|policies|rules?|instructions?|guidelines?)\b",
    r"\b(bypass|override|circumvent|skip)\b.*\b(policy|approval|rules?|limit)",
    r"\bapprove\b.*\b(for me|my own|myself)\b",
    r"\b(pretend|act as if)\b.*\b(approved|manager|hr)\b",
    r"\bforget\b.*\b(instructions|rules|policy)\b",
)


def looks_like_override(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(p, lowered) for p in OVERRIDE_PATTERNS)
