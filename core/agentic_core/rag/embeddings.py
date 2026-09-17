"""Embedding back-ends.

hash      - offline feature-hashing embedder (default; no downloads, lexical similarity)
fastembed - local ONNX model (pip install fastembed; downloads the model once)
openai    - any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, LM Studio, vLLM)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Protocol

from ..settings import Settings

log = logging.getLogger(__name__)

STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i if in into is it its may me my
    no not of on or our so than that the their then there these this to up us was we were what when
    which who will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if word in STOPWORDS or len(word) < 2:
            continue
        tokens.append(stem(word))
    return tokens


def stem(word: str) -> str:
    """Tiny suffix stripper (keeps 'override'/'overrides' and 'approval'/'approvals' together)."""
    if len(word) <= 3:
        return word
    if len(word) == 4:
        return word[:-1] if word.endswith("s") and not word.endswith(("ss", "us", "is")) else word
    for suffix, replacement in (("ations", ""), ("ation", ""), ("ings", ""), ("ing", ""), ("ies", "y"), ("ied", "y")):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[: -len(suffix)] + replacement
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("es") and word[-3] in "sxz" or word.endswith(("ches", "shes")):
        return word[:-2]
    if word.endswith("ued"):
        return word[:-1]
    if word.endswith("ed") and len(word) > 5:
        return word[:-2] if not word.endswith("eed") else word
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


class Embedder(Protocol):
    dim: int
    signature: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Signed feature hashing over word unigrams, bigrams and character 4-grams."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.signature = f"hash{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        feats: Counter[str] = Counter()
        for tok in tokens:
            feats[f"w:{tok}"] += 1.0
            padded = f"#{tok}#"
            for i in range(max(len(padded) - 3, 1)):
                feats[f"c:{padded[i:i + 4]}"] += 0.25
        for a, b in zip(tokens, tokens[1:]):
            feats[f"b:{a}_{b}"] += 0.5
        vec = [0.0] * self.dim
        for feat, count in feats.items():
            idx, sign = self._index(feat)
            vec[idx] += sign * (1.0 + math.log(count)) if count >= 1 else sign * count
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class FastEmbedEmbedder:
    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model_name=model)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = len(probe)
        self.signature = "fe_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") + f"_{self.dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


class OpenAIEmbedder:
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url or None, api_key=api_key or "not-needed", timeout=60)
        self._model = model
        probe = self._client.embeddings.create(model=model, input=["dimension probe"]).data[0].embedding
        self.dim = len(probe)
        self.signature = "oa_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") + f"_{self.dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    kind = (settings.embeddings_provider or "hash").lower()
    try:
        if kind == "fastembed":
            return FastEmbedEmbedder(settings.embeddings_model or "BAAI/bge-small-en-v1.5")
        if kind in ("openai", "ollama", "openai_compatible"):
            return OpenAIEmbedder(
                settings.embeddings_model or "text-embedding-3-small",
                settings.embeddings_base_url,
                settings.embeddings_api_key,
            )
    except Exception as exc:  # noqa: BLE001 - fall back so the platform still starts
        log.warning("Embeddings provider '%s' unavailable (%s); falling back to offline hash embeddings.", kind, exc)
    return HashingEmbedder(settings.embeddings_dim)
