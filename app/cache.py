"""Semantic cache on Qdrant: reuse answers for near-duplicate queries.

Scoping (privacy): answers produced WITHOUT personal memory go to the 'global' scope and are shared;
answers that used a user's memories are stored under 'user:<id>' and only visible to that user.
"""
import re
import time
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector, MatchAny, MatchValue, PointStruct, VectorParams,
)

from app.memory.embeddings import Embedder
from app.qdrant_utils import ensure_collection

GLOBAL = "global"
_TIME = re.compile(r"\b(today|now|current(ly)?|latest|tonight|tomorrow|yesterday|weather|price|news|score|stock)\b", re.I)
_CTX = re.compile(r"\b(it|that|this|those|these|they|them|he|she|his|her|above|previous|earlier|again|continue|more)\b", re.I)
_PERSONAL = re.compile(r"\b(my|me|i|i'm|mine|myself|we|our)\b", re.I)


def is_cacheable(message: str) -> bool:
    """Skip time-sensitive, context-dependent ('what about that?') and personal statements."""
    return len(message.split()) >= 2 and not (_TIME.search(message) or _CTX.search(message) or _PERSONAL.search(message))


@dataclass
class CacheHit:
    answer: str
    similarity: float
    model: str
    scope: str
    saved_cost: float
    saved_prompt_tokens: int


class SemanticCache:
    """Cosine-similarity cache with TTL."""

    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = "costmind_cache",
                 threshold: float = 0.92, ttl_seconds: float = 86400) -> None:
        self.client, self.embedder, self.collection = client, embedder, collection
        self.threshold, self.ttl = threshold, ttl_seconds
        ensure_collection(client, collection, embedder.dim)

    def lookup(self, user_id: str, query: str) -> CacheHit | None:
        flt = Filter(must=[FieldCondition(key="scope", match=MatchAny(any=[GLOBAL, f"user:{user_id}"]))])
        points = self.client.query_points(self.collection, query=self.embedder.embed(query),
                                          query_filter=flt, limit=3, with_payload=True).points
        now = time.time()
        for h in points:  # best first
            if h.score < self.threshold:
                break
            p = h.payload
            if now - p["created_at"] > self.ttl:
                continue
            return CacheHit(p["answer"], h.score, p["model"], p["scope"], p["cost"], p["prompt_tokens"])
        return None

    def store(self, user_id: str | None, query: str, answer: str, model: str, cost: float, prompt_tokens: int) -> None:
        """user_id=None stores in the shared global scope."""
        payload = {"scope": GLOBAL if user_id is None else f"user:{user_id}", "query": query, "answer": answer,
                   "model": model, "cost": cost, "prompt_tokens": prompt_tokens, "created_at": time.time()}
        self.client.upsert(self.collection, points=[PointStruct(id=str(uuid.uuid4()), vector=self.embedder.embed(query), payload=payload)])

    def delete_user(self, user_id: str) -> None:
        """Purge a user's private cache entries ('forget me')."""
        flt = Filter(must=[FieldCondition(key="scope", match=MatchValue(value=f"user:{user_id}"))])
        self.client.delete(self.collection, points_selector=FilterSelector(filter=flt))
