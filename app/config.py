"""Central configuration loaded from environment variables / .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings. Never hardcode secrets; set them in .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "CostMind"
    app_version: str = "0.1.0"
    app_env: str = "dev"
    log_level: str = "INFO"
    api_key: str = ""  # if set, every endpoint except /health requires header X-API-Key
    warmup: bool = True  # preload agent + LiteLLM in a background thread at startup

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    cheap_model: str = "gpt-4o-mini"
    mid_model: str = "gpt-4o-mini"
    premium_model: str = "gpt-4o"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    redis_url: str = "redis://localhost:6379"

    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"  # auto | openai | hash

    llm_provider: str = "auto"  # auto | litellm | mock
    dedup_threshold: float = 0.92

    # Memory
    short_term_window: int = 6  # number of recent turns kept in the buffer
    memory_top_k: int = 5
    memory_collection: str = "costmind_memories"
    w_similarity: float = 0.6
    w_recency: float = 0.25
    w_importance: float = 0.15
    recency_half_life_days: float = 14.0

    max_tokens_per_request: int = 4000
    max_usd_per_user_per_day: float = 0.50
    max_usd_global_per_day: float = 2.0  # hard stop for ALL users combined (protects your API credit)
    max_usd_per_request: float = 0.05
    low_budget_fraction: float = 0.2
    confidence_threshold: float = 0.75
    # Cache + memory-aware routing
    cache_enabled: bool = True
    cache_collection: str = "costmind_cache"
    cache_similarity_threshold: float = 0.92
    cache_ttl_hours: float = 24.0
    memory_shortcut_enabled: bool = True
    memory_answer_threshold: float = 0.5  # min similarity to answer a recall question straight from memory
    min_recall_similarity: float = 0.2    # drop recalled memories below this (saves tokens)
    analytics_db: str = "costmind.db"  # SQLite file (":memory:" in tests)
    fallback_models: str = ""  # comma-separated, tried when a provider call fails


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def parse_list(value: str) -> list[str]:
    """Split a comma-separated env value into a clean list."""
    return [v.strip() for v in value.split(",") if v.strip()]
