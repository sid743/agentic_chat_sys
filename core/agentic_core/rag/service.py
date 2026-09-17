"""Knowledge service: indexes HR policies and chat uploads, answers retrieval
queries. Retrieval only runs when an agent calls a search tool (on-demand RAG)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import delete, select

from ..db.models import Document
from ..db.session import session_scope
from ..settings import Settings
from .chunking import chunk_document
from .embeddings import build_embedder, tokenize
from .parsing import ParsedDocument, parse_bytes, parse_text
from .store import VectorStore

log = logging.getLogger(__name__)

POLICY_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}


@dataclass
class SearchHit:
    doc_id: str
    title: str
    doc_code: str
    version: str
    section: str
    heading: str
    filename: str
    page: int | None
    text: str
    score: float
    citation: str
    scope: str

    def as_dict(self, max_chars: int = 900) -> dict:
        data = asdict(self)
        if len(self.text) > max_chars:
            data["text"] = self.text[:max_chars].rstrip() + " ..."
        return data


def _citation(payload: dict) -> str:
    if payload.get("scope") == "policy":
        parts = [payload.get("title", "")]
        if payload.get("doc_code"):
            parts.append(f"({payload['doc_code']})")
        if payload.get("version"):
            parts.append(f"v{payload['version']}")
        cite = " ".join(p for p in parts if p)
        if payload.get("section"):
            cite += f" §{payload['section']}"
        return cite
    cite = payload.get("filename", "uploaded document")
    if payload.get("page"):
        cite += f" p.{payload['page']}"
    elif payload.get("section"):
        cite += f" §{payload['section']}"
    return cite


class KnowledgeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = build_embedder(settings)
        self.store = VectorStore(settings, self.embedder.dim)
        signature = self.embedder.signature
        self.policy_collection = f"hr_policies_{signature}"
        self.upload_collection = f"chat_uploads_{signature}"

    # ------------------------------------------------------------------ indexing
    def _index(self, parsed: ParsedDocument, *, scope: str, doc_id: str, collection: str, base_payload: dict) -> int:
        chunks = chunk_document(parsed, base_payload["title"])
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        payloads, ids = [], []
        for chunk in chunks:
            payloads.append(
                {
                    **base_payload,
                    "doc_id": doc_id,
                    "scope": scope,
                    "section": chunk.section,
                    "heading": chunk.heading,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                }
            )
            ids.append(VectorStore.point_id(doc_id, str(chunk.index)))
        self.store.upsert(collection, vectors, payloads, ids)
        return len(chunks)

    def index_policies(self, force: bool = False) -> dict:
        folder = Path(self.settings.policies_dir)
        summary = {"indexed": [], "unchanged": [], "removed": []}
        seen_codes: set[str] = set()
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in POLICY_SUFFIXES) if folder.exists() else []
        for path in files:
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            parsed = parse_bytes(path.name, data)
            meta = parsed.metadata
            code = str(meta.get("doc_id") or path.stem)
            seen_codes.add(code)
            with session_scope() as session:
                existing = session.scalar(select(Document).where(Document.scope == "policy", Document.doc_code == code))
                if existing and existing.sha256 == sha and existing.collection == self.policy_collection and not force:
                    summary["unchanged"].append(code)
                    continue
                if existing:
                    session.delete(existing)
            self.store.delete(self.policy_collection, {"doc_code": code, "scope": "policy"})
            title = str(meta.get("title") or path.stem.replace("_", " "))
            version = str(meta.get("version") or "")
            doc_id = f"pol-{code.lower()}"
            count = self._index(
                parsed,
                scope="policy",
                doc_id=doc_id,
                collection=self.policy_collection,
                base_payload={
                    "title": title,
                    "doc_code": code,
                    "version": version,
                    "filename": path.name,
                    "effective_date": str(meta.get("effective_date") or ""),
                    "conversation_id": None,
                },
            )
            with session_scope() as session:
                session.add(
                    Document(
                        id=doc_id,
                        scope="policy",
                        conversation_id=None,
                        filename=path.name,
                        title=title,
                        doc_code=code,
                        version=version,
                        sha256=sha,
                        chunk_count=count,
                        char_count=len(parsed.text),
                        collection=self.policy_collection,
                        uploaded_by="system",
                    )
                )
            summary["indexed"].append(code)
        with session_scope() as session:
            stale = session.scalars(select(Document).where(Document.scope == "policy")).all()
            for doc in stale:
                if doc.doc_code not in seen_codes:
                    self.store.delete(self.policy_collection, {"doc_code": doc.doc_code, "scope": "policy"})
                    session.delete(doc)
                    summary["removed"].append(doc.doc_code)
        if summary["indexed"] or summary["removed"]:
            log.info("Policy index updated: %s", summary)
        return summary

    def ingest_upload(
        self,
        *,
        conversation_id: str,
        filename: str,
        text: str | None = None,
        data: bytes | None = None,
        uploaded_by: str = "",
    ) -> dict:
        """Index a document for one conversation. Re-uploads are de-duplicated."""
        if data is None and text is None:
            raise ValueError("text or data is required")
        raw = data if data is not None else text.encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        doc_id = "up-" + hashlib.sha1(f"{conversation_id}:{sha}".encode()).hexdigest()[:20]
        with session_scope() as session:
            existing = session.get(Document, doc_id)
            if existing and existing.collection == self.upload_collection:
                return {**_doc_dict(existing), "status": "unchanged"}
            if existing:
                session.delete(existing)
        parsed = parse_bytes(filename, data) if data is not None else parse_text(filename, text or "")
        title = str(parsed.metadata.get("title") or filename)
        count = self._index(
            parsed,
            scope="conversation",
            doc_id=doc_id,
            collection=self.upload_collection,
            base_payload={
                "title": title,
                "doc_code": "",
                "version": "",
                "filename": filename,
                "conversation_id": conversation_id,
            },
        )
        with session_scope() as session:
            doc = Document(
                id=doc_id,
                scope="conversation",
                conversation_id=conversation_id,
                filename=filename,
                title=title,
                sha256=sha,
                chunk_count=count,
                char_count=len(parsed.text),
                collection=self.upload_collection,
                uploaded_by=uploaded_by,
            )
            session.add(doc)
            session.flush()
            info = _doc_dict(doc)
        return {**info, "status": "indexed"}

    # ------------------------------------------------------------------ querying
    def search(
        self,
        query: str,
        *,
        scope: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
        doc_code: str | None = None,
        filename: str | None = None,
    ) -> list[SearchHit]:
        top_k = top_k or self.settings.rag_top_k
        if scope == "policy":
            collection = self.policy_collection
            conditions = {"scope": "policy", "doc_code": doc_code}
        else:
            if not conversation_id:
                return []
            collection = self.upload_collection
            conditions = {"conversation_id": conversation_id, "filename": filename}
        conditions = {k: v for k, v in conditions.items() if v}
        vector = self.embedder.embed([query])[0]
        points = self.store.hybrid_search(collection, query, vector, top_k * 2, conditions)
        # Boost chunks whose document title is fully named in the query
        # ("parental leave policy" -> Parental Leave Policy).
        query_tokens = set(tokenize(query))
        for point in points:
            title_tokens = set(tokenize(point.payload.get("title", "")))
            if title_tokens and title_tokens <= query_tokens:
                point.score = round(point.score * (1 + 0.1 * len(title_tokens)), 4)
        points = sorted(points, key=lambda pt: pt.score, reverse=True)
        if scope == "policy" and not doc_code:
            points = self._with_overview(points, query_tokens, collection)
        points = points[:top_k]
        hits = []
        for point in points:
            p = point.payload
            hits.append(
                SearchHit(
                    doc_id=p.get("doc_id", ""),
                    title=p.get("title", ""),
                    doc_code=p.get("doc_code", ""),
                    version=p.get("version", ""),
                    section=p.get("section", ""),
                    heading=p.get("heading", ""),
                    filename=p.get("filename", ""),
                    page=p.get("page"),
                    text=p.get("text", ""),
                    score=point.score,
                    citation=_citation(p),
                    scope=p.get("scope", scope),
                )
            )
        return hits

    def _with_overview(self, points, query_tokens: set[str], collection: str):
        """For broad questions that just name a policy ("what is our parental leave
        policy?"), put that policy's overview chunk first."""
        with session_scope() as session:
            docs = session.execute(select(Document.doc_code, Document.title).where(Document.scope == "policy")).all()
        matches = []
        for code, title in docs:
            title_tokens = set(tokenize(title))
            if title_tokens and title_tokens <= query_tokens and len(query_tokens - title_tokens) <= 2:
                matches.append((len(title_tokens), code))
        if not matches:
            return points
        best = max(size for size, _ in matches)
        for _, code in [m for m in matches if m[0] == best]:
            overview = self.store.get_points(collection, {"doc_code": code, "section": "overview"}, limit=1)
            if overview:
                top = points[0].score if points else 1.0
                overview[0].score = round(top * 1.01, 4)
                points = overview + [p for p in points if p.id != overview[0].id]
        return points

    def list_documents(self, scope: str | None = None, conversation_id: str | None = None) -> list[dict]:
        with session_scope() as session:
            stmt = select(Document).order_by(Document.created_at)
            if scope:
                stmt = stmt.where(Document.scope == scope)
            if conversation_id:
                stmt = stmt.where(Document.conversation_id == conversation_id)
            return [_doc_dict(d) for d in session.scalars(stmt).all()]

    def has_uploads(self, conversation_id: str) -> bool:
        with session_scope() as session:
            return (
                session.scalar(select(Document.id).where(Document.conversation_id == conversation_id).limit(1))
                is not None
            )

    def delete_document(self, doc_id: str) -> bool:
        with session_scope() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return False
            collection = doc.collection
            session.execute(delete(Document).where(Document.id == doc_id))
        self.store.delete(collection, {"doc_id": doc_id})
        return True

    def reset_all(self) -> dict:
        """Drop every indexed chunk (policies and uploads) and re-index the policies."""
        for name in (self.policy_collection, self.upload_collection):
            self.store.drop(name)
        with session_scope() as session:
            session.execute(delete(Document))
        return self.index_policies(force=True)

    def close(self) -> None:
        self.store.close()


def _doc_dict(doc: Document) -> dict:
    return {
        "id": doc.id,
        "scope": doc.scope,
        "conversation_id": doc.conversation_id,
        "filename": doc.filename,
        "title": doc.title,
        "doc_code": doc.doc_code,
        "version": doc.version,
        "chunks": doc.chunk_count,
        "characters": doc.char_count,
        "uploaded_by": doc.uploaded_by,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }
