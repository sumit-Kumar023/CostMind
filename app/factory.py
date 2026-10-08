"""Build a fresh, fully isolated Agent (own in-memory Qdrant, buffers, budgets, analytics)."""
from qdrant_client import QdrantClient

from app.analytics import AnalyticsStore
from app.budget import BudgetManager
from app.cache import SemanticCache
from app.config import Settings, parse_list
from app.graph import Agent
from app.llm import LLMClient, build_llm
from app.memory.embeddings import Embedder, build_embedder
from app.memory.long_term import LongTermMemory
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer
from app.router import ModelRouter


def build_agent(s: Settings, llm: LLMClient | None = None, embedder: Embedder | None = None) -> Agent:
    """Construct an Agent with no external services (used by benchmarks and experiments)."""
    llm, embedder, client = llm or build_llm(s), embedder or build_embedder(s), QdrantClient(location=":memory:")
    weights = dict(w_sim=s.w_similarity, w_rec=s.w_recency, w_imp=s.w_importance, half_life_days=s.recency_half_life_days)
    memory = MemoryManager(ShortTermBuffer(s.short_term_window), LongTermMemory(client, embedder, s.memory_collection, weights),
                           s.memory_top_k, llm, s.cheap_model, s.dedup_threshold)
    router = ModelRouter(llm, {"cheap": s.cheap_model, "mid": s.mid_model, "premium": s.premium_model},
                         parse_list(s.fallback_models))
    cache = SemanticCache(client, embedder, s.cache_collection, s.cache_similarity_threshold,
                          s.cache_ttl_hours * 3600) if s.cache_enabled else None
    return Agent(memory, router, BudgetManager(s.max_usd_per_user_per_day, s.low_budget_fraction, None, s.max_usd_global_per_day), s, cache, AnalyticsStore())
