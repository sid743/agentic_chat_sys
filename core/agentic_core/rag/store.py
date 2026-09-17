"""Qdrant-backed vector store (embedded local mode by default) + BM25 for hybrid search."""

from __future__ import annotations

import math
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..settings import Settings
from .embeddings import tokenize


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    path = settings.resolved_qdrant_path
    if path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=path)


def _filter(conditions: dict[str, Any] | None) -> qm.Filter | None:
    if not conditions:
        return None
    return qm.Filter(
        must=[qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in conditions.items() if v is not None]
    )


@dataclass
class StoredPoint:
    id: str
    score: float
    payload: dict[str, Any]


class BM25:
    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.ids = [d[0] for d in docs]
        self.tfs = [Counter(tokenize(d[1])) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            for t in terms:
                if t not in tf:
                    continue
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg or 1))
                score += self.idf.get(t, 0.0) * freq * (self.k1 + 1) / denom
            if score > 0:
                scores.append((self.ids[i], score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]


class VectorStore:
    def __init__(self, settings: Settings, dim: int) -> None:
        self.client = make_client(settings)
        self.dim = dim
        self._lock = threading.RLock()
        self._bm25: dict[tuple, tuple[BM25, dict[str, dict]]] = {}

    def ensure_collection(self, name: str) -> None:
        with self._lock:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
                )

    @staticmethod
    def point_id(*parts: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(parts)))

    def upsert(self, collection: str, vectors: list[list[float]], payloads: list[dict[str, Any]], ids: list[str]) -> None:
        self.ensure_collection(collection)
        with self._lock:
            self.client.upsert(
                collection_name=collection,
                points=[qm.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)],
            )
            self._invalidate(collection)

    def delete(self, collection: str, conditions: dict[str, Any]) -> None:
        if not self.client.collection_exists(collection):
            return
        with self._lock:
            self.client.delete(
                collection_name=collection,
                points_selector=qm.FilterSelector(filter=_filter(conditions)),
            )
            self._invalidate(collection)

    def drop(self, collection: str) -> None:
        with self._lock:
            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)
            self._invalidate(collection)

    def _invalidate(self, collection: str) -> None:
        for key in [k for k in self._bm25 if k[0] == collection]:
            self._bm25.pop(key, None)

    def _corpus(self, collection: str, conditions: dict[str, Any] | None) -> list[tuple[str, dict]]:
        points: list[tuple[str, dict]] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=collection,
                scroll_filter=_filter(conditions),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend((str(p.id), p.payload or {}) for p in batch)
            if offset is None:
                break
        return points

    def get_points(self, collection: str, conditions: dict[str, Any], limit: int = 4) -> list[StoredPoint]:
        if not self.client.collection_exists(collection):
            return []
        with self._lock:
            batch, _ = self.client.scroll(
                collection_name=collection,
                scroll_filter=_filter(conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        return [StoredPoint(id=str(p.id), score=0.0, payload=p.payload or {}) for p in batch]

    def hybrid_search(
        self,
        collection: str,
        query: str,
        vector: list[float],
        limit: int,
        conditions: dict[str, Any] | None = None,
        min_dense_score: float = 0.3,
    ) -> list[StoredPoint]:
        if not self.client.collection_exists(collection):
            return []
        with self._lock:
            dense = self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=max(limit * 4, 20),
                query_filter=_filter(conditions),
                with_payload=True,
            ).points
            key = (collection, tuple(sorted((conditions or {}).items())))
            if key not in self._bm25:
                corpus = self._corpus(collection, conditions)
                if len(self._bm25) >= 128:  # keep the lexical index cache bounded
                    self._bm25.pop(next(iter(self._bm25)))
                self._bm25[key] = (BM25([(pid, payload.get("text", "")) for pid, payload in corpus]), dict(corpus))
            bm25, corpus_payloads = self._bm25[key]
            lexical = bm25.search(query, max(limit * 4, 20))

        # Reciprocal-rank fusion of dense and lexical rankings
        fused: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        dense_scores: dict[str, float] = {}
        for rank, point in enumerate(dense):
            pid = str(point.id)
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (60 + rank)
            payloads[pid] = point.payload or {}
            dense_scores[pid] = float(point.score)
        for rank, (pid, _score) in enumerate(lexical):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (60 + rank)
            payloads.setdefault(pid, corpus_payloads.get(pid, {}))
        if not lexical:  # nothing matched lexically: rely on (thresholded) dense similarity
            fused = {pid: score for pid, score in dense_scores.items() if score >= min_dense_score}
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [StoredPoint(id=pid, score=round(score, 4), payload=payloads[pid]) for pid, score in ranked]

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass
