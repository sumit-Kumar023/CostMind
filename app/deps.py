"""Dependency wiring (cached singletons)."""
from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import get_settings
from app.llm import build_llm
from app.memory.embeddings import build_embedder
from app.memory.long_term import LongTermMemory
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer


@lru_cache
def get_qdrant() -> QdrantClient:
    """One shared Qdrant client. QDRANT_URL=':memory:' uses an embedded in-process Qdrant."""
    s = get_settings()
    if s.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, check_compatibility=False)


@lru_cache
def get_embedder():
    """One shared (LRU-cached) embedder."""
    return build_embedder(get_settings())


@lru_cache
def get_memory() -> MemoryManager:
    """Build the MemoryManager."""
    s = get_settings()
    weights = dict(w_sim=s.w_similarity, w_rec=s.w_recency, w_imp=s.w_importance, half_life_days=s.recency_half_life_days)
    long = LongTermMemory(get_qdrant(), get_embedder(), s.memory_collection, weights)
    short = ShortTermBuffer(s.short_term_window, None if s.redis_url == "memory" else s.redis_url)
    return MemoryManager(short, long, s.memory_top_k, build_llm(s), s.cheap_model, s.dedup_threshold)


@lru_cache
def get_agent():
    """Build the LangGraph agent with router + budgets."""
    from app.budget import BudgetManager
    from app.config import parse_list
    from app.graph import Agent
    from app.router import ModelRouter

    s = get_settings()
    models = {"cheap": s.cheap_model, "mid": s.mid_model, "premium": s.premium_model}
    router = ModelRouter(build_llm(s), models, parse_list(s.fallback_models))
    budget = BudgetManager(s.max_usd_per_user_per_day, s.low_budget_fraction,
                           None if s.redis_url == "memory" else s.redis_url, s.max_usd_global_per_day)
    cache = None
    if s.cache_enabled:
        from app.cache import SemanticCache
        cache = SemanticCache(get_qdrant(), get_embedder(), s.cache_collection, s.cache_similarity_threshold, s.cache_ttl_hours * 3600)
    return Agent(get_memory(), router, budget, s, cache, get_analytics())


@lru_cache
def get_analytics():
    """SQLite-backed analytics store."""
    from app.analytics import AnalyticsStore

    return AnalyticsStore(get_settings().analytics_db)
