"""Long-term vector memory on Qdrant, isolated per user via payload filter."""
import time
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector, MatchValue, PayloadSchemaType,
    PointStruct, VectorParams,
)

from app.memory.embeddings import Embedder
from app.qdrant_utils import ensure_collection
from app.memory.scoring import estimate_importance, relevance_score


@dataclass
class Memory:
    """A recalled memory with its component scores."""

    id: str
    text: str
    kind: str
    importance: float
    created_at: float
    similarity: float = 0.0
    score: float = 0.0


class LongTermMemory:
    """Store and recall per-user memories with relevance-scored re-ranking."""

    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str, weights: dict) -> None:
        self.client, self.embedder, self.collection, self.weights = client, embedder, collection, weights
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        ensure_collection(self.client, self.collection, self.embedder.dim)
        try:  # required for filtering on Qdrant Cloud; harmless locally
            self.client.create_payload_index(self.collection, "user_id", PayloadSchemaType.KEYWORD)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _user_filter(user_id: str) -> Filter:
        return Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

    def add(self, user_id: str, text: str, kind: str = "fact", importance: float | None = None) -> str:
        """Embed and store a memory. Returns its id."""
        mid = str(uuid.uuid4())
        payload = {
            "user_id": user_id, "text": text, "kind": kind,
            "importance": estimate_importance(text) if importance is None else importance,
            "created_at": time.time(),
        }
        self.client.upsert(self.collection, points=[PointStruct(id=mid, vector=self.embedder.embed(text), payload=payload)])
        return mid

    def search(self, user_id: str, query: str, k: int = 5) -> list[Memory]:
        """Over-fetch by similarity, then re-rank by similarity + recency + importance."""
        hits = self.client.query_points(
            self.collection, query=self.embedder.embed(query),
            query_filter=self._user_filter(user_id), limit=k * 4, with_payload=True,
        ).points
        out = []
        for h in hits:
            p = h.payload
            m = Memory(str(h.id), p["text"], p["kind"], p["importance"], p["created_at"], similarity=h.score)
            m.score = relevance_score(h.score, m.created_at, m.importance, **self.weights)
            out.append(m)
        return sorted(out, key=lambda m: m.score, reverse=True)[:k]

    def list_all(self, user_id: str, limit: int = 200) -> list[Memory]:
        pts, _ = self.client.scroll(self.collection, scroll_filter=self._user_filter(user_id), limit=limit, with_payload=True)
        return [Memory(str(p.id), p.payload["text"], p.payload["kind"], p.payload["importance"], p.payload["created_at"]) for p in pts]

    def delete_all(self, user_id: str) -> None:
        """'Forget me': remove every memory for this user."""
        self.client.delete(self.collection, points_selector=FilterSelector(filter=self._user_filter(user_id)))
