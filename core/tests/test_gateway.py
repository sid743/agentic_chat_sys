"""The demo gateway: auto sign-in, our own branding, everything else untouched."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.testclient import TestClient

from agentic_core import gateway

INDEX_HTML = (
    "<!doctype html><html><head>"
    '<meta name="description" content="LibreChat - An open source chat application" />'
    "<title>LibreChat</title></head><body>chat</body></html>"
)


def upstream(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/auth/login":
        body = json.loads(request.content)
        if body == {"email": "demo@example.com", "password": "s3cret"}:
            return httpx.Response(
                200,
                json={"token": "jwt"},
                headers=[
                    ("set-cookie", "refreshToken=abc; Path=/; HttpOnly"),
                    ("set-cookie", "token_provider=librechat; Path=/"),
                ],
            )
        return httpx.Response(401, json={"message": "Invalid credentials"})
    if path == "/manifest.webmanifest":
        return httpx.Response(200, json={"name": "LibreChat", "short_name": "LibreChat", "icons": []})
    if path.startswith("/assets/"):
        return httpx.Response(200, content=b"upstream-bytes", headers={"content-type": "image/png"})
    if path == "/api/auth/refresh":
        # LibreChat rejects a stale or missing session
        if "refreshToken=good" in request.headers.get("cookie", ""):
            return httpx.Response(200, json={"token": "still-valid"})
        return httpx.Response(401, json={"message": "Refresh token not provided"})
    if path == "/api/auth/logout":
        return httpx.Response(200, json={"message": "logged out upstream"})
    if path == "/api/messages":
        return httpx.Response(200, json={"ok": True})
    if path == "/api/stream":
        return httpx.Response(200, content=b"data: one\n\ndata: two\n\n", headers={"content-type": "text/event-stream"})
    return httpx.Response(200, text=INDEX_HTML, headers={"content-type": "text/html; charset=utf-8"})


@pytest.fixture
def client(tmp_path, monkeypatch):
    brand = tmp_path / "brand"
    brand.mkdir()
    (brand / "logo.svg").write_text("<svg>ours</svg>", encoding="utf-8")
    monkeypatch.setattr(gateway, "BRAND_DIR", brand)
    monkeypatch.setattr(gateway, "APP_TITLE", "Agentic HR Platform")
    monkeypatch.setattr(gateway, "APP_DESCRIPTION", "Internal demo, synthetic data")
    monkeypatch.setattr(gateway, "DEMO_EMAIL", "demo@example.com")
    monkeypatch.setattr(gateway, "DEMO_PASSWORD", "s3cret")
    monkeypatch.setattr(gateway, "AUTO_LOGIN", True)
    with TestClient(gateway.create_app()) as test_client:
        test_client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        yield test_client


HTML = {"accept": "text/html"}


def test_page_is_rebranded(client):
    page = client.get("/", headers=HTML)
    assert "<title>Agentic HR Platform</title>" in page.text
    assert "Internal demo, synthetic data" in page.text
    assert "LibreChat" not in page.text


def test_first_visit_gets_a_demo_session(client):
    page = client.get("/", headers=HTML)
    cookies = page.headers.get_list("set-cookie")
    assert any(c.startswith("refreshToken=") for c in cookies)
    assert any(c.startswith("token_provider=") for c in cookies)


def test_existing_session_is_left_alone(client):
    page = client.get("/", headers={**HTML, "cookie": "refreshToken=already-here"})
    assert not any(c.startswith("refreshToken=") for c in page.headers.get_list("set-cookie"))


def test_sign_in_pages_go_to_the_chat(client):
    for path in ("/login", "/register", "/forgot-password"):
        reply = client.get(path, headers=HTML, follow_redirects=False)
        assert reply.status_code == 302 and reply.headers["location"] == "/"


def test_login_page_returns_when_auto_login_is_off(client, monkeypatch):
    monkeypatch.setattr(gateway, "AUTO_LOGIN", False)
    reply = client.get("/login", headers=HTML, follow_redirects=False)
    assert reply.status_code == 200


def test_our_icons_replace_theirs(client):
    assert client.get("/assets/logo.svg").text == "<svg>ours</svg>"
    # anything we do not brand still comes from LibreChat
    assert client.get("/assets/anyscale.png").content == b"upstream-bytes"


def test_manifest_carries_our_name(client):
    data = client.get("/manifest.webmanifest").json()
    assert data["name"] == data["short_name"] == "Agentic HR Platform"


def test_api_and_streams_pass_through(client):
    assert client.get("/api/messages").json() == {"ok": True}
    stream = client.get("/api/stream")
    assert stream.headers["content-type"] == "text/event-stream"
    assert stream.text == "data: one\n\ndata: two\n\n"


def test_health_endpoint(client):
    assert client.get("/gateway/health").json()["status"] == "ok"


def test_expired_session_is_replaced_instead_of_showing_the_login_form(client):
    # The login form is a route inside the app, so the only fix is a refresh that works.
    reply = client.post("/api/auth/refresh", headers={"cookie": "refreshToken=stale"})
    assert reply.status_code == 200
    assert reply.json()["token"] == "jwt"
    assert any(c.startswith("refreshToken=") for c in reply.headers.get_list("set-cookie"))


def test_valid_session_refreshes_normally(client):
    reply = client.post("/api/auth/refresh", headers={"cookie": "refreshToken=good"})
    assert reply.status_code == 200
    assert reply.json()["token"] == "still-valid"
    assert not reply.headers.get_list("set-cookie")


def test_logout_does_not_strand_the_next_visitor(client):
    reply = client.post("/api/auth/logout", headers={"cookie": "refreshToken=good"})
    assert reply.status_code == 200
    assert not reply.headers.get_list("set-cookie")
    # the session is untouched, so the app signs straight back in
    assert client.post("/api/auth/refresh", headers={"cookie": "refreshToken=good"}).status_code == 200


def test_health_reports_a_broken_demo_account(client, monkeypatch):
    assert client.get("/gateway/health").json()["demo_login"] == "ok"
    monkeypatch.setattr(gateway, "DEMO_PASSWORD", "wrong-password")
    body = client.get("/gateway/health").json()
    assert body["status"] == "degraded"
    assert "HTTP 401" in body["demo_login"]
    assert "create-user" in body["fix"]
