"""Deterministic offline "model".

It lets the platform, the demo and the test-suite run with no API keys or local
model. It follows the same protocol as real models (tool calls, then a final
answer); its decisions are rule-based and its answers are templated from tool
results, so answers are plainer than a real LLM's.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import date
from typing import Any

from .base import CallOptions, ChatModel, ChatResult, ToolCall, Usage, estimate_tokens

TITLE_PREFIXES = ("provide a concise", "analyze this conversation")


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{uuid.uuid4().hex[:10]}", name=name, arguments={k: v for k, v in arguments.items() if v is not None})


def _history(messages: list[dict[str, Any]]) -> list[tuple[str, dict, dict]]:
    """(tool name, arguments, result) for every tool call already executed."""
    calls: dict[str, tuple[str, dict]] = {}
    done: list[tuple[str, dict, dict]] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls[tc.get("id")] = (fn.get("name"), args)
        elif msg.get("role") == "tool":
            name, args = calls.get(msg.get("tool_call_id"), ("?", {}))
            try:
                result = json.loads(msg.get("content") or "{}")
            except json.JSONDecodeError:
                result = {"ok": False, "raw": msg.get("content")}
            done.append((name, args, result))
    return done


def _last(done, name):
    for tool, args, result in reversed(done):
        if tool == name:
            return result
    return None


def _called(done, name, **match) -> bool:
    for tool, args, _ in done:
        if tool == name and all(args.get(k) == v for k, v in match.items()):
            return True
    return False


def _fmt_days(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _fmt_range(start: str, end: str) -> str:
    try:
        s, e = date.fromisoformat(start), date.fromisoformat(end)
    except (TypeError, ValueError):
        return f"{start} to {end}"
    if s == e:
        return s.strftime("%d %b %Y")
    if s.year == e.year:
        return f"{s.strftime('%d %b')} to {e.strftime('%d %b %Y')}"
    return f"{s.strftime('%d %b %Y')} to {e.strftime('%d %b %Y')}"


def _best_sentences(text: str, query: str, limit: int = 2) -> str:
    words = {w for w in re.findall(r"[a-z]{3,}", query.lower())} - {"what", "the", "our", "for", "and", "how", "can", "does", "policy"}
    body = re.sub(r"^.*?\n", "", text, count=1) if "\n" in text else text  # drop breadcrumb line
    body = re.sub(r"\*\*(.+?)[.:]?\*\*[:.]?\s*", r"\1: ", body)  # "**Label.** text" -> "Label: text"
    body = body.replace("**", "")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 20]
    if not sentences:
        return body.strip()[:300]
    scored = sorted(
        enumerate(sentences),
        key=lambda item: (-len(words & set(re.findall(r"[a-z]{3,}", item[1].lower()))), item[0]),
    )
    chosen = sorted(scored[:limit], key=lambda item: item[0])
    return " ".join(s for _, s in chosen)


class MockChatModel(ChatModel):
    provider = "mock"
    native_tools = True

    def __init__(self, model: str = "hr-demo") -> None:
        self.model = model

    # ------------------------------------------------------------------ API
    async def complete(self, messages, tools=None, options=None) -> ChatResult:
        options = options or CallOptions()
        ctx = options.context or {}
        purpose = options.purpose
        content = ""
        calls: list[ToolCall] = []

        first_user = next((m.get("content") for m in messages if m.get("role") == "user"), "") or ""
        if purpose == "title" or (isinstance(first_user, str) and first_user.lower().startswith(TITLE_PREFIXES)):
            content = self._title(ctx.get("user_message") or first_user)
        elif purpose == "route":
            from ..agents.router import heuristic_route  # local import avoids a cycle

            plan = heuristic_route(
                ctx.get("user_message", ""),
                has_uploads=bool(ctx.get("has_uploads")),
                agent_ids=ctx.get("agent_ids"),
                max_agents=ctx.get("max_agents", 3),
            )
            content = json.dumps(plan.as_dict())
        elif purpose == "synthesize":
            content = self._synthesize(ctx)
        elif purpose == "direct":
            content = self._direct(ctx)
        elif purpose == "agent":
            done = _history(messages)
            planner = getattr(self, f"_plan_{options.agent_id}", None)
            call = planner(ctx, done) if planner and tools else None
            if call is not None and len(done) < 8:
                calls = [call]
            else:
                finisher = getattr(self, f"_final_{options.agent_id}", None)
                content = finisher(ctx, done) if finisher else "Done."
        else:
            content = "I'm the offline demo model. Configure a real provider for open-ended chat."

        usage = Usage(estimate_tokens(messages), estimate_tokens(content) + 10 * len(calls))
        if options.usage_sink is not None:
            options.usage_sink.add(usage)
        return ChatResult(content=content, tool_calls=calls, usage=usage, model=self.id, finish_reason="tool_calls" if calls else "stop")

    async def stream(self, messages, options=None) -> AsyncIterator[str]:
        result = await self.complete(messages, None, options)
        text = result.content
        for i in range(0, len(text), 24):
            yield text[i : i + 24]

    # ------------------------------------------------------------------ generic
    @staticmethod
    def _title(text: str) -> str:
        text = re.sub(r"(?s)^.*?Conversation:\s*", "", text or "")
        text = re.sub(r"(?im)^(user|assistant|ai)\s*:\s*", "", text)
        words = re.findall(r"[A-Za-z0-9']+", text)[:5]
        return " ".join(w.capitalize() for w in words) or "HR Assistant Chat"

    @staticmethod
    def _direct(ctx: dict) -> str:
        name = (ctx.get("actor") or {}).get("first_name") or "there"
        return (
            f"Hi {name}! I'm the HR multi-agent assistant (running on the offline demo model). "
            "I can explain HR policies, check your leave balance and eligibility, submit or approve "
            "leave requests, send notifications, and answer questions about documents you upload."
        )

    @staticmethod
    def _synthesize(ctx: dict) -> str:
        results = [r for r in ctx.get("results", []) if r.get("answer")]
        if not results:
            return "I couldn't complete that request."
        if len(results) == 1:
            return results[0]["answer"]
        parts = []
        sources: list[str] = []
        for r in results:
            answer = r["answer"]
            match = re.search(r"\n*\s*Sources?:\s*(.+)$", answer, re.IGNORECASE | re.DOTALL)
            if match:
                sources.extend(s.strip() for s in re.split(r";\s*", match.group(1).strip()) if s.strip())
                answer = answer[: match.start()].rstrip()
            parts.append(f"**{r['agent_name']}**\n\n{answer}")
        text = "\n\n".join(parts)
        if sources:
            text += "\n\nSources: " + "; ".join(dict.fromkeys(sources))
        return text

    # ------------------------------------------------------------------ policy agent
    @staticmethod
    def _policy_query(ctx) -> str:
        task = (ctx.get("task") or "").split("\n")[0].strip()
        generic = "Find what the HR policies say about:"
        if task and not task.startswith(generic):
            return task
        return ctx.get("user_message") or task.replace(generic, "").strip()

    def _plan_policy_agent(self, ctx, done):
        if not _called(done, "search_policies"):
            return _call("search_policies", query=self._policy_query(ctx))
        return None

    def _final_policy_agent(self, ctx, done):
        result = _last(done, "search_policies") or {}
        hits = result.get("results") or []
        if not hits:
            return "I couldn't find anything about that in the HR policy documents. Please contact your HR Business Partner."
        query = self._policy_query(ctx)
        lines, sources = [], []
        hits = [h for i, h in enumerate(hits) if i == 0 or h.get("section") != "overview"]
        for hit in hits[:3]:
            if hit.get("section") == "overview":
                items = [ln[2:] for ln in hit.get("text", "").splitlines() if ln.startswith("- ")][:8]
                lines.append(f"- **{hit.get('title')} - overview**:\n" + "\n".join(f"  - {i}" for i in items))
            else:
                excerpt = _best_sentences(hit.get("text", ""), query)
                lines.append(f"- **{hit.get('title')} §{hit.get('section') or '-'}**: {excerpt}")
            sources.append(hit.get("citation"))
        return "Here is what the HR policies say:\n\n" + "\n".join(lines) + "\n\nSources: " + "; ".join(dict.fromkeys(sources))

    # ------------------------------------------------------------------ eligibility agent
    def _plan_eligibility_agent(self, ctx, done):
        from ..agents import nlp

        text = f"{ctx.get('user_message', '')} {ctx.get('task', '')}"
        actor = ctx.get("actor") or {}
        today = date.fromisoformat(ctx.get("today"))
        names = [n for n in nlp.person_names(ctx.get("user_message", "")) if n.lower() != (actor.get("name") or "").lower()]
        other_ids = [i for i in nlp.employee_ids(text) if i != actor.get("id")]
        lowered = text.lower()

        if names or other_ids:
            if not other_ids:
                if not _called(done, "find_employee"):
                    return _call("find_employee", query=names[0])
                found = (_last(done, "find_employee") or {}).get("matches") or []
                other_ids = [found[0]["employee_id"]] if found else []
            if other_ids and not _called(done, "get_leave_balance", employee_id=other_ids[0]):
                return _call("get_leave_balance", employee_id=other_ids[0])
            return None

        if nlp.has_any(lowered, ("holiday",)) and not _called(done, "list_holidays"):
            return _call("list_holidays", upcoming_only=True)

        rng = nlp.date_range(ctx.get("user_message", ""), today)
        if rng and nlp.has_any(lowered, ("working day", "how many days", "count")) and not _called(done, "calculate_leave_days"):
            return _call("calculate_leave_days", start_date=rng[0].isoformat(), end_date=rng[1].isoformat())

        if nlp.has_any(lowered, ("eligib", "entitled", "can i take", "can i apply", "allowed to take", "probation", "joined")):
            leave_type = nlp.guess_leave_type(text, "AL")
            if not _called(done, "get_my_profile"):
                return _call("get_my_profile")
            if not _called(done, "check_leave_eligibility"):
                args = {"leave_type": leave_type}
                if rng and rng[0] != today:
                    args.update(start_date=rng[0].isoformat(), end_date=rng[1].isoformat())
                return _call("check_leave_eligibility", **args)
            if leave_type == "AL" and not _called(done, "get_leave_balance"):
                return _call("get_leave_balance")
            return None

        if nlp.has_any(lowered, ("manager", "profile", "who am i", "my details")) and not _called(done, "get_my_profile"):
            return _call("get_my_profile")

        if not _called(done, "get_leave_balance") and not done:
            leave_type = nlp.guess_leave_type(text, None)
            return _call("get_leave_balance", leave_type=leave_type)
        return None

    def _final_eligibility_agent(self, ctx, done):
        parts: list[str] = []
        refs: list[str] = []
        for tool, args, result in done:
            if not result.get("ok", True):
                if result.get("error") == "forbidden":
                    target = result.get("target_name") or args.get("employee_id") or "that employee"
                    parts.append(
                        f"I can't share {target}'s leave information. Leave balances are Restricted data: only the "
                        "employee, their manager or an HR administrator can view them. You could ask them directly or contact HR."
                    )
                    refs.append(result.get("policy_ref") or "HR-POL-004 §4")
                elif tool == "find_employee":
                    continue
                else:
                    parts.append(f"I couldn't complete `{tool}`: {result.get('message') or result.get('error')}.")
                continue
            if tool == "get_my_profile":
                p = result.get("profile", {})
                if not any(t == "check_leave_eligibility" for t, _, _ in done):
                    parts.append(
                        f"You are {p.get('name')} ({p.get('employee_id')}), {p.get('title')} in {p.get('department')}, "
                        f"{p.get('location')}. You joined on {p.get('date_of_joining')} ({p.get('service_days')} days ago); "
                        f"probation {'ended' if p.get('probation_complete') else 'ends'} on {p.get('probation_end_date')}. "
                        f"Your manager is {p.get('manager_name') or 'not set'}."
                    )
            elif tool == "check_leave_eligibility":
                name = result.get("leave_type_name")
                if result.get("eligible"):
                    text = f"**You are eligible for {name}.**"
                    bal = (result.get("balance") or {}).get("available_days")
                    if bal is not None:
                        text += f" Available balance: {_fmt_days(bal)} day(s)."
                    rng = result.get("date_range")
                    if rng:
                        text += f" {_fmt_range(rng['start_date'], rng['end_date'])} is {_fmt_days(rng['working_days'])} working day(s)."
                else:
                    text = f"**You are not eligible for {name} yet.** " + " ".join(result.get("blocking_reasons") or [])
                    if result.get("earliest_eligible_date"):
                        text += f" You can use it from {result['earliest_eligible_date']}."
                for warning in result.get("warnings") or []:
                    text += f"\n- Note: {warning}"
                parts.append(text)
                refs.extend(result.get("policy_refs") or [])
            elif tool == "get_leave_balance":
                rows = result.get("balances") or []
                who = result.get("employee_name") or "Your"
                header = "Your leave balance" if result.get("is_self", True) else f"Leave balance for {who}"
                table = [
                    f"{header} (as of {result.get('as_of')}):",
                    "",
                    "| Leave type | Available | Accrued/credited | Used | Pending |",
                    "|---|---|---|---|---|",
                ]
                for r in rows:
                    if r.get("available_days") is None:
                        continue
                    table.append(
                        f"| {r['leave_type_name']} | {_fmt_days(r.get('available_days'))} | "
                        f"{_fmt_days(r.get('accrued_to_date_days'))} (+{_fmt_days(r.get('carried_forward_days'))} carried) | "
                        f"{_fmt_days(r.get('used_days'))} | {_fmt_days(r.get('pending_days'))} |"
                    )
                parts.append("\n".join(table))
                refs.extend(sorted({r.get("policy_ref") for r in rows if r.get("policy_ref")}))
            elif tool == "calculate_leave_days":
                hol = ", ".join(f"{h['name']} ({h['date']})" for h in result.get("public_holidays", [])) or "none"
                parts.append(
                    f"{_fmt_range(result['start_date'], result['end_date'])} has {_fmt_days(result['working_days'])} working day(s) "
                    f"({result['weekend_days']} weekend day(s); public holidays: {hol})."
                )
            elif tool == "list_holidays":
                items = result.get("holidays") or []
                lines = [f"- {h['date']} ({h['weekday']}): {h['name']}{' (optional)' if h.get('optional') else ''}" for h in items[:8]]
                parts.append(f"Upcoming holidays for {result.get('location')}:\n" + ("\n".join(lines) or "- none"))
        if not parts:
            return "I couldn't find the information needed for that."
        text = "\n\n".join(parts)
        refs = [r for r in dict.fromkeys(refs) if r]
        if refs:
            text += "\n\nSources: " + "; ".join(refs)
        return text

    # ------------------------------------------------------------------ workflow agent
    def _plan_workflow_agent(self, ctx, done):
        from ..agents import nlp

        message = ctx.get("user_message", "")
        text = f"{message} {ctx.get('task', '')}"
        lowered = message.lower()
        today = date.fromisoformat(ctx.get("today"))
        if nlp.looks_like_override(message):
            return None
        ids = nlp.request_ids(message) or nlp.request_ids(ctx.get("task", ""))
        if ids and nlp.has_any(lowered, ("approve",)):
            if not _called(done, "approve_leave_request"):
                return _call("approve_leave_request", request_id=ids[0], comment="Approved via HR assistant")
            return None
        if ids and nlp.has_any(lowered, ("reject", "decline")):
            if not _called(done, "reject_leave_request"):
                match = re.search(r"(?i)\b(?:because of|because|due to|reason:?)\s+(.*)", message)
                reason = (match.group(1).strip() if match else "") or "Rejected via HR assistant"
                return _call("reject_leave_request", request_id=ids[0], reason=reason[:200])
            return None
        if ids and nlp.has_any(lowered, ("cancel", "withdraw")):
            if not _called(done, "cancel_leave_request"):
                return _call("cancel_leave_request", request_id=ids[0])
            return None
        rng = nlp.date_range(message, today)
        if rng and nlp.has_any(lowered, ("apply", "book", "submit", "request", "raise", "take", "file")):
            if not _called(done, "create_leave_request"):
                return _call(
                    "create_leave_request",
                    leave_type=nlp.guess_leave_type(text, "AL"),
                    start_date=rng[0].isoformat(),
                    end_date=rng[1].isoformat(),
                    reason="Submitted via HR assistant",
                )
            return None
        if not _called(done, "list_leave_requests"):
            role = (ctx.get("actor") or {}).get("role")
            if nlp.has_any(lowered, ("team", "approval", "approve", "awaiting", "direct report")) and role in ("manager", "hr_admin"):
                return _call("list_leave_requests", scope="awaiting_my_approval")
            status = "pending" if "pending" in lowered else None
            return _call("list_leave_requests", scope="mine", status=status)
        return None

    def _final_workflow_agent(self, ctx, done):
        from ..agents import nlp

        if nlp.looks_like_override(ctx.get("user_message", "")):
            return (
                "I can't do that. Leave can only be approved by your approvers, nobody can approve their own leave, "
                "and requests to ignore or override the leave policy must be refused. If you need leave, I can check "
                "your balance or submit a normal request for your manager to review.\n\n"
                "Sources: HR-POL-001 v3.2 §6.3; HR-POL-001 v3.2 §10"
            )
        parts = []
        for tool, args, result in done:
            if not result.get("ok", True):
                reasons = result.get("reasons") or []
                msg = result.get("message") or result.get("error")
                text = f"I couldn't complete that ({tool.replace('_', ' ')}): {msg}"
                if reasons:
                    text += "\n" + "\n".join(f"- {r}" for r in reasons)
                parts.append(text)
                continue
            req = result.get("request") or {}
            if tool == "create_leave_request":
                text = (
                    f"Submitted **{req.get('request_id')}**: {req.get('leave_type_name', req.get('leave_type'))}, "
                    f"{_fmt_range(req.get('start_date'), req.get('end_date'))} ({_fmt_days(req.get('working_days'))} working day(s)). "
                    f"Status: pending approval by {req.get('current_approver_name')}."
                )
                if result.get("approval_levels") and len(result["approval_levels"]) > 1:
                    text += " HR approval is also required after your manager."
                for warning in result.get("warnings") or []:
                    text += f"\n- Note: {warning}"
                parts.append(text)
            elif tool in ("approve_leave_request", "reject_leave_request", "cancel_leave_request"):
                verb = {"approve_leave_request": "Approved", "reject_leave_request": "Rejected", "cancel_leave_request": "Cancelled"}[tool]
                status = req.get("status")
                text = f"{verb} **{req.get('request_id')}** ({req.get('employee_name')}, {_fmt_range(req.get('start_date'), req.get('end_date'))})."
                if status == "pending_hr":
                    text += f" It now needs HR approval from {req.get('current_approver_name')}."
                elif status == "approved":
                    text += " The request is now fully approved."
                parts.append(text)
            elif tool == "list_leave_requests":
                rows = result.get("requests") or []
                if not rows:
                    parts.append("There are no matching leave requests.")
                    continue
                lines = ["| Request | Employee | Type | Dates | Days | Status | Approver |", "|---|---|---|---|---|---|---|"]
                for r in rows[:10]:
                    lines.append(
                        f"| {r['request_id']} | {r['employee_name']} | {r['leave_type']} | "
                        f"{_fmt_range(r['start_date'], r['end_date'])} | {_fmt_days(r['working_days'])} | "
                        f"{r['status']} | {r.get('current_approver_name') or '-'} |"
                    )
                parts.append("\n".join(lines))
        return "\n\n".join(parts) or "No workflow action was needed."

    # ------------------------------------------------------------------ notification agent
    def _plan_notification_agent(self, ctx, done):
        from ..agents import nlp

        events = ctx.get("events") or []
        sent = [args for tool, args, _ in done if tool == "send_notification"]
        for event in events:
            for target in event.get("notify", []):
                key = (target.get("employee_id"), event.get("request_id"))
                if any((a.get("recipient"), a.get("related_request_id")) == key for a in sent):
                    continue
                return _call(
                    "send_notification",
                    recipient=target.get("employee_id"),
                    subject=target.get("subject") or event.get("summary", "Leave update")[:120],
                    message=target.get("message") or event.get("summary", ""),
                    channel=target.get("channel", "email"),
                    related_request_id=event.get("request_id"),
                )
        if events:
            return None
        lowered = ctx.get("user_message", "").lower()
        if nlp.has_any(lowered, ("notify", "remind", "inform", "let my manager", "email my", "message my", "tell my")):
            if not sent:
                recipient = "my_manager" if "manager" in lowered else "hr" if " hr" in f" {lowered}" else "my_manager"
                ids = nlp.request_ids(ctx.get("user_message", ""))
                return _call(
                    "send_notification",
                    recipient=recipient,
                    subject="Message from " + ((ctx.get("actor") or {}).get("name") or "an employee"),
                    message=ctx.get("user_message", "")[:500],
                    channel="email",
                    related_request_id=ids[0] if ids else None,
                )
            return None
        if not _called(done, "list_notifications"):
            return _call("list_notifications", limit=10)
        return None

    def _final_notification_agent(self, ctx, done):
        parts = []
        for tool, args, result in done:
            if tool == "send_notification":
                if result.get("ok"):
                    n = result.get("notification", {})
                    parts.append(f"Notified {n.get('recipient_name')} by {n.get('channel')}: \"{n.get('subject')}\".")
                else:
                    parts.append(f"Could not notify {args.get('recipient')}: {result.get('message') or result.get('error')}.")
            elif tool == "list_notifications":
                items = result.get("notifications") or []
                if not items:
                    parts.append("You have no notifications.")
                else:
                    lines = [f"- {n['created_at'][:16]} [{n['channel']}] {n['subject']}" for n in items]
                    parts.append("Your latest notifications:\n" + "\n".join(lines))
        return "\n".join(parts) or "No notifications were needed."

    # ------------------------------------------------------------------ document agent
    def _plan_document_agent(self, ctx, done):
        if not _called(done, "list_uploaded_documents"):
            return _call("list_uploaded_documents")
        if not _called(done, "search_uploaded_documents"):
            return _call("search_uploaded_documents", query=ctx.get("user_message") or ctx.get("task", ""), top_k=4)
        return None

    def _final_document_agent(self, ctx, done):
        docs = (_last(done, "list_uploaded_documents") or {}).get("documents") or []
        hits = (_last(done, "search_uploaded_documents") or {}).get("results") or []
        if not docs:
            return "There are no documents uploaded in this conversation yet. Attach a file (PDF, DOCX, TXT or MD) and ask again."
        query = ctx.get("user_message", "")
        names = ", ".join(d["filename"] for d in docs)
        if not hits:
            return f"I searched {names} but found nothing relevant to your question."
        summarise = "summar" in query.lower() or "overview" in query.lower()
        if not summarise:
            hits = [h for h in hits if h.get("section") != "overview"] or hits
        lines = []
        for hit in hits[:3]:
            excerpt = _best_sentences(hit.get("text", ""), query, limit=3 if summarise else 2)
            where = hit.get("heading") or (f"page {hit['page']}" if hit.get("page") else hit.get("section")) or "excerpt"
            lines.append(f"- **{hit.get('filename')}** ({where}): {excerpt}")
        lead = "Summary of the most relevant parts" if summarise else "From your uploaded document(s)"
        return f"{lead}:\n\n" + "\n".join(lines) + "\n\nSources: " + "; ".join(dict.fromkeys(h.get("citation") for h in hits[:3]))
