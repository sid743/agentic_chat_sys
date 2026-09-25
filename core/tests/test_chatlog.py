"""The JSON transcript written for every turn."""

from __future__ import annotations

import asyncio
import json

from agentic_core.chatlog import ChatLog, safe_name
from agentic_core.service import ChatRequest


def ask(service, text, employee="", conversation="conv-log"):
    request = ChatRequest(
        messages=[{"role": "user", "content": text}],
        model="mock/hr-demo",
        employee_id=employee,
        conversation_id=conversation,
        user_email="sid@example.com",
        user_name="Sid",
    )
    return asyncio.run(service.complete(request))


def test_file_names_survive_odd_conversation_ids(tmp_path):
    log = ChatLog(tmp_path)
    # separators are replaced, so nothing can climb out of the log folder
    assert "/" not in safe_name("conv/../etc/passwd")
    assert safe_name("conv/../etc/passwd") == "conv-..-etc-passwd"
    assert safe_name("..") == "unknown"
    assert safe_name("") == "unknown"
    assert log.path_for("abc123").name == "abc123.jsonl"


def test_a_turn_is_written_with_the_whole_story(service, tmp_path):
    out = ask(service, "What is our parental leave policy?")
    files = service.chat_log.files()
    assert len(files) == 1 and files[0]["conversation_id"] == "conv-log"

    turns = service.chat_log.read(files[0]["date"], "conv-log")
    assert len(turns) == 1
    turn = turns[0]

    assert turn["status"] == "ok"
    assert turn["run_id"] == out["run_id"]
    assert turn["conversation_id"] == "conv-log"
    assert turn["user"] == {"email": "sid@example.com", "name": "Sid"}
    assert turn["actor"]["id"] == "E1001" and turn["actor"]["identity_source"]
    assert turn["model"] == "mock/hr-demo"
    assert turn["message"].startswith("What is our parental")
    assert "HR-POL-002" in turn["answer"]
    assert turn["plan"]["mode"] == "agents"
    assert turn["flow"][0].startswith("START -> route(")
    assert turn["counts"]["agents"] == 1 and turn["counts"]["tool_calls"] >= 1
    assert turn["latency_ms"] >= 0
    assert isinstance(turn["usage"]["input_tokens"], int)
    assert turn["trace"], "the trace travels with the record"

    agent = turn["agents"][0]
    assert agent["id"] == "policy_agent" and agent["ok"] is True
    assert agent["sources"], "citations are kept"
    tool = agent["tools"][0]
    assert tool["name"] == "search_policies"
    assert tool["arguments"] and tool["ok"] is True
    assert tool["result"], "tool results are kept for later reference"


def test_turns_append_to_one_file_per_conversation(service):
    ask(service, "What is our parental leave policy?", conversation="conv-many")
    ask(service, "And the notice period for annual leave?", conversation="conv-many")
    ask(service, "What is our parental leave policy?", conversation="conv-other")

    files = {f["conversation_id"]: f for f in service.chat_log.files()}
    assert set(files) == {"conv-many", "conv-other"}
    assert len(service.chat_log.read(files["conv-many"]["date"], "conv-many")) == 2
    assert len(service.chat_log.read(files["conv-other"]["date"], "conv-other")) == 1


def test_every_line_is_valid_json(service):
    ask(service, "Apply for casual leave next Tuesday", employee="E1014", conversation="conv-json")
    path = service.chat_log.path_for("conv-json")
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_big_tool_results_are_summarised_not_dropped(tmp_path):
    from agentic_core.chatlog import _tool_result

    small = {"ok": True, "rows": [1, 2, 3]}
    assert _tool_result(small, 2000) == small

    big = {"ok": True, "text": "x" * 5000}
    clipped = _tool_result(big, 200)
    assert clipped["truncated"] is True and clipped["ok"] is True
    assert clipped["chars"] > 200 and len(clipped["preview"]) == 200


def test_logging_can_be_turned_off(service):
    service.chat_log.enabled = False
    ask(service, "What is our parental leave policy?", conversation="conv-off")
    assert service.chat_log.files() == []


def test_a_broken_log_never_breaks_a_chat(service, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ChatLog, "write", explode)
    out = ask(service, "What is our parental leave policy?", conversation="conv-broken")
    assert out["status"] == "ok" and "HR-POL-002" in out["content"]
