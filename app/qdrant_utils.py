"""Qdrant helpers."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Create the collection, or fail loudly if it exists with a different vector size.

    Switching embedding providers (e.g. offline hash 256-d -> OpenAI 1536-d) on a reused Qdrant
    otherwise fails later with a confusing error on the first upsert.
    """
    if client.collection_exists(name):
        size = getattr(client.get_collection(name).config.params.vectors, "size", None)
        if size is not None and size != dim:
            raise RuntimeError(
                f"Qdrant collection '{name}' has vector dimension {size} but the current embedder produces {dim}. "
                f"Delete the collection (or set a different collection name via MEMORY_COLLECTION / CACHE_COLLECTION).")
        return
    client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
