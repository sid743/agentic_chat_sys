"""Shared FastAPI dependencies."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from ..service import ChatService
from ..settings import Settings


def get_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service is starting up")
    return service


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def _presented_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def require_api_key(request: Request) -> None:
    expected = request.app.state.settings.agent_core_api_key
    if not expected:
        return
    if not hmac.compare_digest(_presented_key(request), expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid or missing API key", "type": "invalid_request_error"}},
        )


def require_admin(request: Request) -> None:
    if not request.app.state.settings.admin_auth_required:
        return
    require_api_key(request)
