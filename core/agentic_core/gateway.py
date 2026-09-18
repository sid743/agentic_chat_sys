"""Demo front door.

A small proxy that sits in front of LibreChat so the demo opens straight into a
chat window:

- signs a demo user in automatically, so there is no login or signup page;
- serves our own icons, app name and page title in place of LibreChat's.

Everything else is passed through untouched, including streamed responses.

    python -m uvicorn agentic_core.gateway:app --host 0.0.0.0 --port 3080

Settings (all from the environment / .env):
    GATEWAY_UPSTREAM      where LibreChat is        (default http://librechat:3080)
    DOMAIN_CLIENT         the public URL of this gateway, used as the Origin
    APP_TITLE             name shown in the tab and page title
    BRAND_DIR             folder with logo.svg and the icon PNGs
    DEMO_USER_EMAIL       account the gateway signs in as
    DEMO_USER_PASSWORD    its password (setup_env.py generates one)
    GATEWAY_AUTO_LOGIN    false disables the auto sign-in and shows the login page
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger(__name__)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("false", "0", "no", "off")


UPSTREAM = (os.getenv("GATEWAY_UPSTREAM") or "http://librechat:3080").rstrip("/")
PUBLIC_URL = (os.getenv("DOMAIN_CLIENT") or "http://localhost:3080").rstrip("/")
APP_TITLE = os.getenv("APP_TITLE") or "Agentic HR Platform"
APP_DESCRIPTION = os.getenv("APP_DESCRIPTION") or f"{APP_TITLE} - internal demo with synthetic data"
BRAND_DIR = Path(os.getenv("BRAND_DIR") or "/app/brand")
DEMO_EMAIL = os.getenv("DEMO_USER_EMAIL") or ""
DEMO_PASSWORD = os.getenv("DEMO_USER_PASSWORD") or ""
AUTO_LOGIN = _flag("GATEWAY_AUTO_LOGIN", True) and bool(DEMO_EMAIL and DEMO_PASSWORD)

# Headers that belong to a single hop and must not be forwarded.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
# Files replaced from BRAND_DIR when they exist there.
BRAND_FILES = {
    "logo.svg", "favicon-16x16.png", "favicon-32x32.png",
    "apple-touch-icon-180x180.png", "icon-192x192.png", "maskable-icon.png",
}
# Pages that only exist to sign in; with auto-login they are dead ends.
AUTH_PAGES = ("/login", "/register", "/forgot-password", "/reset-password", "/verify")


def rebrand_html(text: str) -> str:
    text = re.sub(r"<title>.*?</title>", f"<title>{escape(APP_TITLE)}</title>", text, count=1, flags=re.S)
    text = re.sub(
        r'(<meta\s+name="description"\s+content=")(.*?)(")',
        lambda m: m.group(1) + escape(APP_DESCRIPTION) + m.group(3),
        text,
        count=1,
        flags=re.S,
    )
    return text


def rebrand_manifest(raw: bytes) -> bytes:
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    data["name"] = APP_TITLE
    data["short_name"] = APP_TITLE
    data["description"] = APP_DESCRIPTION
    return json.dumps(data).encode()


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
    if wants_html(request):
        # We rewrite the page, so ask for it uncompressed. Everything else keeps
        # compression (a browser asking for brotli would otherwise come back as
        # bytes we cannot edit).
        headers["accept-encoding"] = "identity"
    client = request.client.host if request.client else ""
    if client:
        forwarded = request.headers.get("x-forwarded-for")
        headers["x-forwarded-for"] = f"{forwarded}, {client}" if forwarded else client
    headers.setdefault("x-forwarded-proto", request.url.scheme)
    return headers


def _response_headers(upstream: httpx.Response, drop_body_headers: bool = False) -> list[tuple[bytes, bytes]]:
    out: list[tuple[bytes, bytes]] = []
    for key, value in upstream.headers.multi_items():
        lower = key.lower()
        if lower in HOP_BY_HOP:
            continue
        if drop_body_headers and lower in ("content-length", "content-encoding"):
            continue
        out.append((key.encode("latin-1"), value.encode("latin-1")))
    return out


async def demo_session(client: httpx.AsyncClient) -> tuple[list[str], bytes, str]:
    """Sign the demo user in upstream.

    Returns (cookies, body, problem). `problem` is empty when it worked, and
    otherwise says why - which is what /gateway/health reports.
    """
    try:
        reply = await client.post(
            f"{UPSTREAM}/api/auth/login",
            json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD},
            headers={"origin": PUBLIC_URL, "referer": f"{PUBLIC_URL}/", "content-type": "application/json"},
        )
    except httpx.HTTPError as exc:
        problem = f"cannot reach LibreChat: {exc}"
        log.warning("Demo sign-in %s", problem)
        return [], b"", problem
    if reply.status_code != 200:
        problem = f"HTTP {reply.status_code} for {DEMO_EMAIL}: {reply.text[:160]}"
        log.warning(
            "Demo sign-in failed (%s). Create the account: "
            "docker compose exec -T librechat npm run create-user -- "
            "<email> \"Demo User\" <username> <password> --email-verified=true",
            problem,
        )
        return [], b"", problem
    return reply.headers.get_list("set-cookie"), reply.content, ""


async def demo_cookies(client: httpx.AsyncClient) -> list[str]:
    cookies, _body, _problem = await demo_session(client)
    return cookies


async def _proxy_json(client: httpx.AsyncClient, request: Request, path: str) -> Response | None:
    """Forward a small JSON request and buffer the reply (used for /api/auth/refresh)."""
    try:
        reply = await client.request(
            request.method,
            f"{UPSTREAM}{path}",
            headers=_forward_headers(request),
            content=await request.body(),
        )
    except httpx.HTTPError as exc:
        log.warning("Upstream %s unreachable: %s", UPSTREAM, exc)
        return None
    body = reply.content  # httpx already decoded any compression, so re-state the length
    response = Response(body, status_code=reply.status_code)
    response.raw_headers = _response_headers(reply, drop_body_headers=True) + [
        (b"content-length", str(len(body)).encode())
    ]
    return response


def wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return request.method == "GET" and "text/html" in accept


async def handle(request: Request) -> Response:
    client: httpx.AsyncClient = request.app.state.client
    path = "/" + request.path_params.get("path", "")

    # 1. Our own icons and logo, when the brand folder provides them.
    if path.startswith("/assets/"):
        name = path.rsplit("/", 1)[-1]
        candidate = BRAND_DIR / name
        if name in BRAND_FILES and candidate.is_file():
            return FileResponse(candidate, headers={"cache-control": "public, max-age=300"})

    # 2. Sign-in pages are dead ends once auto-login is on.
    if AUTO_LOGIN and request.method == "GET" and any(path == p or path.startswith(p + "/") for p in AUTH_PAGES):
        return RedirectResponse("/", status_code=302)

    # 3. Keep the session alive. The login form is a page inside the app, not a URL we
    #    can redirect, so the only way to never see it is to make sure the app's session
    #    check always succeeds: when LibreChat rejects the refresh, sign in again here.
    if AUTO_LOGIN and request.method == "POST" and path == "/api/auth/refresh":
        refreshed = await _proxy_json(client, request, path)
        if refreshed is not None and refreshed.status_code == 200:
            return refreshed
        cookies, body, _problem = await demo_session(client)
        if not cookies:
            return refreshed if refreshed is not None else Response(status_code=401)
        response = Response(body, status_code=200, media_type="application/json")
        response.raw_headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            *[(b"set-cookie", c.encode("latin-1")) for c in cookies],
        ]
        return response

    # 4. Signing out of a shared demo account would only strand the next visitor.
    if AUTO_LOGIN and request.method == "POST" and path == "/api/auth/logout":
        return Response(b'{"message":"ok"}', status_code=200, media_type="application/json")

    url = f"{UPSTREAM}{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    upstream_request = client.build_request(
        request.method,
        url,
        headers=_forward_headers(request),
        content=request.stream(),
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        log.warning("Upstream %s unreachable: %s", UPSTREAM, exc)
        return Response(
            f"<!doctype html><title>{escape(APP_TITLE)}</title>"
            "<body style='font-family:system-ui;padding:3rem'>"
            f"<h1>{escape(APP_TITLE)}</h1><p>The chat service is still starting. Refresh in a moment.</p>",
            status_code=502,
            media_type="text/html; charset=utf-8",
        )
    content_type = upstream.headers.get("content-type", "")

    # 3. The web app manifest carries a name; make it ours.
    if path == "/manifest.webmanifest":
        body = rebrand_manifest(await upstream.aread())
        await upstream.aclose()
        return Response(body, status_code=upstream.status_code, media_type="application/manifest+json")

    # 4. HTML: retitle, and attach the demo session on the way in.
    if "text/html" in content_type:
        raw = await upstream.aread()
        await upstream.aclose()
        text = raw.decode(upstream.encoding or "utf-8", errors="replace")
        if "<" not in text[:500]:  # still compressed: pass it through rather than corrupt it
            response = Response(raw, status_code=upstream.status_code)
            response.raw_headers = _response_headers(upstream)
            return response
        body = rebrand_html(text).encode("utf-8")
        headers = [h for h in _response_headers(upstream, drop_body_headers=True) if h[0].lower() != b"content-type"]
        headers.append((b"content-type", b"text/html; charset=utf-8"))
        headers.append((b"content-length", str(len(body)).encode()))
        if AUTO_LOGIN and "refreshToken=" not in request.headers.get("cookie", ""):
            for cookie in await demo_cookies(client):
                headers.append((b"set-cookie", cookie.encode("latin-1")))
        response = Response(body, status_code=upstream.status_code)
        response.raw_headers = headers
        return response

    # 5. Everything else streams through untouched (SSE, uploads, bundles).
    async def stream():
        try:
            if upstream.is_stream_consumed:  # already buffered (test transports)
                yield upstream.content
            else:
                async for chunk in upstream.aiter_raw():
                    yield chunk
        finally:
            await upstream.aclose()

    response = StreamingResponse(stream(), status_code=upstream.status_code)
    response.raw_headers = _response_headers(upstream)
    return response


def _is_local(request: Request) -> bool:
    """True for loopback and private callers; the details are for the operator only."""
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True  # test clients and unix sockets
    return address.is_loopback or address.is_private


async def healthz(request: Request) -> Response:
    """Says whether the demo sign-in actually works, which is the thing that breaks."""
    if not _is_local(request):
        return Response(json.dumps({"status": "ok"}), media_type="application/json")
    data: dict[str, Any] = {"status": "ok", "upstream": UPSTREAM, "auto_login": AUTO_LOGIN}
    if AUTO_LOGIN:
        data["demo_user"] = DEMO_EMAIL
        _cookies, _body, problem = await demo_session(request.app.state.client)
        data["demo_login"] = "ok" if not problem else problem
        if problem:
            data["status"] = "degraded"
            data["fix"] = (
                'docker compose exec -T librechat npm run create-user -- '
                f'"{DEMO_EMAIL}" "Demo User" "{DEMO_EMAIL.split("@")[0]}" "<DEMO_USER_PASSWORD from .env>" '
                "--email-verified=true </dev/null"
            )
    return Response(json.dumps(data), media_type="application/json")


@asynccontextmanager
async def lifespan(app: Starlette):
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
        follow_redirects=False,
    )
    log.info("Gateway ready: upstream %s, auto sign-in %s", UPSTREAM, "on" if AUTO_LOGIN else "off")
    if not AUTO_LOGIN:
        log.info("Auto sign-in is off; set DEMO_USER_EMAIL and DEMO_USER_PASSWORD to turn it on")
    try:
        yield
    finally:
        await app.state.client.aclose()


def create_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/gateway/health", healthz),
            Route("/{path:path}", handle, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]),
        ],
        lifespan=lifespan,
    )


app = create_app()
