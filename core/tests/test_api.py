import json

import pytest
from fastapi.testclient import TestClient

from agentic_core.main import create_app

from .conftest import make_settings
from .helpers import make_pdf


@pytest.fixture
def client(tmp_path):
    app = create_app(make_settings(tmp_path, agent_core_api_key="secret"))
    with TestClient(app) as c:
        yield c


AUTH = {"Authorization": "Bearer secret"}


def sse_events(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def test_auth_required(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/overview").status_code == 401


def test_models_endpoint_lists_auto_and_mock(client):
    data = client.get("/v1/models", headers=AUTH).json()
    ids = [m["id"] for m in data["data"]]
    assert ids[0] == "auto" and "mock/hr-demo" in ids
    assert client.get("/models", headers=AUTH).status_code == 200  # baseURL without /v1


def test_non_stream_completion(client):
    body = {"model": "mock/hr-demo", "messages": [{"role": "user", "content": "What is our parental leave policy?"}]}
    data = client.post("/v1/chat/completions", headers=AUTH, json=body).json()
    message = data["choices"][0]["message"]
    assert data["object"] == "chat.completion"
    assert "HR-POL-002" in message["content"]
    assert "[Leave Policy Agent]" in message["reasoning_content"]
    assert data["usage"]["total_tokens"] > 0
    assert data["agentic_core"]["agents"] == ["policy_agent"]


def test_stream_matches_librechat_expectations(client):
    headers = {**AUTH, "X-User-Email": "priya.nair@example.com", "X-Conversation-Id": "lc-123"}
    body = {
        "model": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "I joined three months ago. Am I eligible for paid leave?"}],
    }
    with client.stream("POST", "/v1/chat/completions", headers=headers, json=body) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_events(response)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if e != "[DONE]"]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    reasoning_idx = [i for i, c in enumerate(chunks) if c["choices"] and c["choices"][0]["delta"].get("reasoning_content")]
    content_idx = [i for i, c in enumerate(chunks) if c["choices"] and c["choices"][0]["delta"].get("content")]
    assert reasoning_idx and content_idx and max(reasoning_idx) < min(content_idx)
    for i in reasoning_idx:  # LibreChat treats a chunk as reasoning only when content is empty
        assert not chunks[i]["choices"][0]["delta"].get("content")
    answer = "".join(chunks[i]["choices"][0]["delta"]["content"] for i in content_idx)
    reasoning = "".join(chunks[i]["choices"][0]["delta"]["reasoning_content"] for i in reasoning_idx)
    assert "Priya Nair" in reasoning and "matching email" in reasoning  # identity from X-User-Email
    assert "2026-12-15" in answer
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] > 0


def test_think_tag_mode(tmp_path):
    app = create_app(make_settings(tmp_path, reasoning_field="think"))
    with TestClient(app) as c:
        body = {"model": "mock/hr-demo", "stream": True, "messages": [{"role": "user", "content": "hello"}]}
        with c.stream("POST", "/v1/chat/completions", json=body) as response:
            events = [e for e in sse_events(response) if e != "[DONE]"]
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"])
    assert text.startswith("<think>\n") and "</think>" in text
    assert not any("reasoning_content" in e["choices"][0]["delta"] for e in events if e["choices"])


def test_librechat_upload_as_text_flow(client):
    doc = "# Travel Policy\n\n## Per diem\n\nDomestic per diem is INR 3,500 per day.\n\n## Flights\n\nEconomy class for flights under 6 hours."
    content = 'Attached document(s):\n```md# "travel.md"\n' + doc + "\n\n```\nWhat is the per diem according to the attached file?"
    headers = {**AUTH, "X-Conversation-Id": "lc-upload"}
    body = {"model": "mock/hr-demo", "messages": [{"role": "user", "content": content}]}
    data = client.post("/v1/chat/completions", headers=headers, json=body).json()
    assert "INR 3,500" in data["choices"][0]["message"]["content"]
    docs = client.get("/v1/documents", headers=AUTH, params={"conversation_id": "lc-upload"}).json()["documents"]
    assert [d["filename"] for d in docs] == ["travel.md"]


def test_document_upload_endpoint_and_search(client):
    pdf = make_pdf(["Night shift allowance is INR 400 per shift."])
    res = client.post(
        "/v1/documents",
        headers=AUTH,
        files={"file": ("allowances.pdf", pdf, "application/pdf")},
        data={"conversation_id": "api-conv"},
    )
    assert res.status_code == 200 and res.json()["status"] == "indexed"
    hits = client.get(
        "/v1/documents/search", headers=AUTH,
        params={"q": "night shift allowance", "scope": "conversation", "conversation_id": "api-conv"},
    ).json()["results"]
    assert hits and hits[0]["filename"] == "allowances.pdf"
    bad = client.post("/v1/documents", headers=AUTH, files={"file": ("x.bin", b"\x00\xff\x00\xfe", "application/octet-stream")},
                      data={"conversation_id": "api-conv"})
    assert bad.status_code == 415


def test_admin_endpoints(client):
    client.post("/v1/chat/completions", headers=AUTH, json={"model": "mock/hr-demo", "messages": [{"role": "user", "content": "Show my notifications"}]})
    overview = client.get("/api/overview", headers=AUTH).json()
    assert overview["tables"]["employees"] == 20
    assert overview["runs"]["count"] == 1
    runs = client.get("/api/runs", headers=AUTH).json()["runs"]
    detail = client.get(f"/api/runs/{runs[0]['id']}", headers=AUTH).json()
    assert detail["trace"]
    rows = client.get("/api/tables/leave_requests", headers=AUTH).json()["rows"]
    assert len(rows) == 12
    assert client.get("/api/tables/secrets", headers=AUTH).status_code == 404
    mermaid = client.get("/api/graph", headers=AUTH).json()["mermaid"]
    assert "route" in mermaid and "run_agent" in mermaid and "synthesize" in mermaid
    assert client.get("/admin").status_code == 200
    reset = client.post("/api/reset", headers=AUTH).json()
    assert reset["seeded"]["employees"] == 20
