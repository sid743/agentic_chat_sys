"""Document endpoints (for API clients and the admin page; LibreChat uploads arrive inside chat messages)."""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from ..rag.parsing import UnsupportedDocument
from ..service import ChatService
from .deps import get_service, require_api_key, require_admin

router = APIRouter(prefix="/v1/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post("", dependencies=[Depends(require_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(..., description="Conversation the document belongs to"),
    uploaded_by: str = Form(""),
    service: ChatService = Depends(get_service),
):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 25 MB)")
    try:
        return await asyncio.to_thread(
            service.knowledge.ingest_upload,
            conversation_id=conversation_id,
            filename=file.filename or "upload",
            data=data,
            uploaded_by=uploaded_by,
        )
    except UnsupportedDocument as exc:
        raise HTTPException(415, str(exc)) from exc


@router.get("", dependencies=[Depends(require_api_key)])
async def list_documents(
    scope: Literal["policy", "conversation"] | None = None,
    conversation_id: str | None = None,
    service: ChatService = Depends(get_service),
):
    return {"documents": await asyncio.to_thread(service.knowledge.list_documents, scope, conversation_id)}


@router.get("/search", dependencies=[Depends(require_api_key)])
async def search_documents(
    q: str = Query(..., min_length=2),
    scope: Literal["policy", "conversation"] = "policy",
    conversation_id: str | None = None,
    top_k: int = Query(5, ge=1, le=20),
    service: ChatService = Depends(get_service),
):
    hits = await asyncio.to_thread(
        service.knowledge.search, q, scope=scope, conversation_id=conversation_id, top_k=top_k
    )
    return {"query": q, "results": [h.as_dict() for h in hits]}


@router.delete("/{doc_id}", dependencies=[Depends(require_api_key)])
async def delete_document(doc_id: str, service: ChatService = Depends(get_service)):
    if not await asyncio.to_thread(service.knowledge.delete_document, doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


@router.post("/policies", dependencies=[Depends(require_admin)])
async def add_policy(file: UploadFile = File(...), service: ChatService = Depends(get_service)):
    """Add or replace a policy document (Markdown with front matter recommended) and re-index."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "policy.md")
    if not name.lower().endswith((".md", ".txt", ".pdf", ".docx")):
        raise HTTPException(415, "Policies must be .md, .txt, .pdf or .docx")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 25 MB)")
    target = service.settings.policies_dir / name
    target.write_bytes(data)
    try:
        return await asyncio.to_thread(service.knowledge.index_policies)
    except UnsupportedDocument as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(415, str(exc)) from exc


@router.post("/policies/reindex", dependencies=[Depends(require_admin)])
async def reindex_policies(force: bool = False, service: ChatService = Depends(get_service)):
    return await asyncio.to_thread(service.knowledge.index_policies, force)
