"""Retrieval tools. Agents call these only when they need knowledge (on-demand RAG)."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from ..agents.context import ToolContext
from .base import tool


class PolicySearchArgs(BaseModel):
    query: str = Field(description="What to look for, in natural language")
    top_k: int = Field(5, ge=1, le=10)
    doc_code: str | None = Field(
        None, description="Optional document filter: HR-POL-001 (Leave), HR-POL-002 (Parental), "
        "HR-POL-003 (Conduct), HR-POL-004 (Information Handling), HR-FAQ-001 (FAQ)"
    )


@tool(
    "search_policies",
    "Search the approved HR policy documents. Returns passages with a ready-to-use citation "
    "(title, document id, version, section).",
    PolicySearchArgs,
    kind="retrieval",
)
async def search_policies(ctx: ToolContext, args: PolicySearchArgs) -> dict:
    knowledge = ctx.request.knowledge
    hits = await asyncio.to_thread(
        knowledge.search, args.query, scope="policy", top_k=args.top_k, doc_code=args.doc_code
    )
    ctx.request.trace.rag(ctx.agent_id, "policies", args.query, [h.citation for h in hits])
    return {
        "query": args.query,
        "results": [h.as_dict() for h in hits],
        "instructions": "Answer only from these passages and cite the 'citation' values you used.",
    }


class UploadSearchArgs(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=10)
    filename: str | None = Field(None, description="Limit the search to one uploaded file")


@tool(
    "search_uploaded_documents",
    "Search the documents the user uploaded in this conversation.",
    UploadSearchArgs,
    kind="retrieval",
)
async def search_uploaded_documents(ctx: ToolContext, args: UploadSearchArgs) -> dict:
    knowledge = ctx.request.knowledge
    hits = await asyncio.to_thread(
        knowledge.search,
        args.query,
        scope="conversation",
        conversation_id=ctx.request.conversation_id,
        top_k=args.top_k,
        filename=args.filename,
    )
    ctx.request.trace.rag(ctx.agent_id, "uploads", args.query, [h.citation for h in hits])
    return {"query": args.query, "results": [h.as_dict(1200) for h in hits]}


@tool("list_uploaded_documents", "List the documents uploaded in this conversation (name, size, chunks).")
async def list_uploaded_documents(ctx: ToolContext, args: BaseModel) -> dict:
    docs = await asyncio.to_thread(
        ctx.request.knowledge.list_documents, "conversation", ctx.request.conversation_id
    )
    return {
        "documents": [
            {"filename": d["filename"], "characters": d["characters"], "chunks": d["chunks"], "uploaded_at": d["created_at"]}
            for d in docs
        ]
    }
