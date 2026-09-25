"""A JSON record of every turn, written to disk for later reference.

One file per conversation per day, one JSON object per line (JSONL), under
`DATA_DIR/chatlogs/<date>/<conversation-id>.jsonl`. Each line holds the whole
story of one turn: who asked, as whom, which model, the router's plan, every
agent and every tool call with its arguments and result, the citations, token
counts, timings, uploads and the full trace.

JSONL because appending a line is atomic enough for this, the file stays
readable while it is being written, and `jq` can chew through it:

    jq -r 'select(.status=="ok") | [.ts, .actor.id, .latency_ms, .message] | @tsv' *.jsonl

The same records are in the `agent_runs` table; these files are the portable
copy, for handing to someone who does not have the database.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
VERSION = 1


def safe_name(value: str) -> str:
    """A conversation id that is safe as a file name, and never empty."""
    cleaned = _UNSAFE.sub("-", (value or "").strip()).strip("-.")
    return (cleaned or "unknown")[:80]


def _clip(text: str | None, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def _tool_result(result: dict[str, Any], limit: int) -> Any:
    """Keep tool results whole when they are small, summarised when they are not."""
    try:
        encoded = json.dumps(result, default=str)
    except (TypeError, ValueError):
        return {"unserialisable": str(result)[:limit]}
    if len(encoded) <= limit:
        return result
    return {
        "truncated": True,
        "chars": len(encoded),
        "ok": result.get("ok", True),
        "preview": encoded[:limit],
    }


def turn_record(
    *,
    request: Any,
    message: str,
    answer: str,
    status: str,
    latency_ms: int,
    usage: Any,
    results: list[Any],
    trace: Any,
    plan: dict[str, Any] | None = None,
    flow: list[str] | None = None,
    tool_result_chars: int = 2000,
    text_chars: int = 20000,
) -> dict[str, Any]:
    actor = request.actor
    agents = []
    tool_calls = 0
    for result in results:
        tools = []
        for record in result.tools:
            tool_calls += 1
            tools.append(
                {
                    "name": record.name,
                    "arguments": record.arguments,
                    "ok": bool(record.result.get("ok", True)),
                    "ms": record.ms,
                    "result": _tool_result(record.result, tool_result_chars),
                }
            )
        agents.append(
            {
                "id": result.agent_id,
                "name": result.agent_name,
                "task": _clip(result.task, 2000),
                "model": result.model,
                "ok": result.ok,
                "error": result.error,
                "ms": result.ms,
                "answer": _clip(result.answer, text_chars),
                "sources": list(result.sources),
                "events": list(result.events),
                "tools": tools,
            }
        )

    return {
        "v": VERSION,
        "run_id": request.run_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conversation_id": request.conversation_id,
        "status": status,
        "latency_ms": latency_ms,
        "user": {"email": request.user_email, "name": request.user_name},
        "actor": {
            "id": actor.id,
            "name": actor.name,
            "role": actor.role,
            "title": actor.title,
            "department": actor.department,
            "location": actor.location_code,
            "manager": actor.manager_name,
            "identity_source": request.actor_source,
        },
        "model": request.model_id,
        "demo_today": request.today.isoformat(),
        "message": _clip(message, text_chars),
        "answer": _clip(answer, text_chars),
        "plan": plan or {},
        "flow": flow or [],
        "agents": agents,
        "counts": {"agents": len(results), "tool_calls": tool_calls},
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        },
        "uploads": list(request.uploads),
        "trace": trace.serializable() if hasattr(trace, "serializable") else [],
    }


class ChatLog:
    """Appends turn records. Never raises: a failed log must not fail a chat."""

    def __init__(self, directory: Path, enabled: bool = True, tool_result_chars: int = 2000) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.tool_result_chars = tool_result_chars

    def path_for(self, conversation_id: str, when: datetime | None = None) -> Path:
        when = when or datetime.now(timezone.utc)
        return self.directory / when.strftime("%Y-%m-%d") / f"{safe_name(conversation_id)}.jsonl"

    def write(self, record: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        path = self.path_for(record.get("conversation_id", ""))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            return path
        except OSError:
            log.exception("Could not append to the chat log %s", path)
            return None

    # ------------------------------------------------------------------ reading
    def files(self, limit: int = 200) -> list[dict[str, Any]]:
        """Newest first: what is on disk, without reading the contents."""
        if not self.directory.is_dir():
            return []
        found = []
        for path in self.directory.glob("*/*.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue
            found.append(
                {
                    "date": path.parent.name,
                    "conversation_id": path.stem,
                    "path": f"{path.parent.name}/{path.name}",
                    "bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                }
            )
        found.sort(key=lambda item: item["modified"], reverse=True)
        return found[:limit]

    def read(self, date: str, conversation_id: str) -> list[dict[str, Any]]:
        path = self.directory / safe_name(date) / f"{safe_name(conversation_id)}.jsonl"
        if not path.is_file():
            return []
        turns = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except ValueError:
                turns.append({"unparsable": line[:500]})
        return turns
