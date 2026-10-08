"""Embedding providers. API-based (no torch) so the app fits in 512MB."""
import hashlib
import math
import re
from collections import OrderedDict
from typing import Protocol

import httpx

from app.config import Settings


class Embedder(Protocol):
    """Anything that turns text into a fixed-size vector."""

    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and offline dev.

    Not semantically smart, but texts sharing words get higher cosine similarity.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbedder:
    """OpenAI embeddings via plain HTTPS (avoids heavy SDK dependencies)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.api_key, self.model, self.dim = api_key, model, 1536

    def embed(self, text: str) -> list[float]:
        r = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


class CachedEmbedder:
    """LRU wrapper so the same text is embedded once per request (recall, cache and dedup share it)."""

    def __init__(self, inner: Embedder, size: int = 2048) -> None:
        self.inner, self.dim, self.size = inner, inner.dim, size
        self._c: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = self.misses = 0

    def embed(self, text: str) -> list[float]:
        if text in self._c:
            self._c.move_to_end(text)
            self.hits += 1
            return self._c[text]
        self.misses += 1
        vec = self.inner.embed(text)
        self._c[text] = vec
        if len(self._c) > self.size:
            self._c.popitem(last=False)
        return vec


def build_embedder(s: Settings) -> Embedder:
    """Pick OpenAI when configured, otherwise the offline hash embedder."""
    use_openai = s.embedding_provider == "openai" or (
        s.embedding_provider == "auto" and s.openai_api_key
    )
    return CachedEmbedder(OpenAIEmbedder(s.openai_api_key, s.embedding_model) if use_openai else HashEmbedder())
