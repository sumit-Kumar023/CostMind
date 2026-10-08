"""CostMind FastAPI app: memory-enabled, cost-aware chat with analytics."""
import logging
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import redis
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request
from pydantic import BaseModel, Field

from app.budget import BudgetExceeded
from app.config import get_settings
from app.deps import get_agent, get_memory
from app.router import AllModelsFailed

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("costmind")

USER_ID_RE = r"^[A-Za-z0-9_.@-]+$"
UserId = Annotated[str, Path(min_length=1, max_length=64, pattern=USER_ID_RE)]


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """If API_KEY is configured, every endpoint except /health* needs a matching X-API-Key header."""
    if not settings.api_key or request.url.path.startswith("/health"):
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _warmup() -> None:
    """Build the agent and import LiteLLM in the background so the first real request isn't slow."""
    try:
        from app.llm import LiteLLMClient

        agent = get_agent()
        if isinstance(agent.router.llm, LiteLLMClient):
            import litellm  # noqa: F401  (heavy import, ~140MB)
        logger.info("warm-up complete")
    except Exception as exc:  # noqa: BLE001
        logger.warning("warm-up failed (will retry lazily on first request): %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.warmup:
        threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan,
              dependencies=[Depends(require_api_key)])


# ------------------------------------------------------------------ health
@app.get("/health")
def health() -> dict:
    """Liveness probe. Cheap and dependency-free (used by Render)."""
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


def _check_redis() -> bool:
    try:
        return bool(redis.from_url(settings.redis_url, socket_connect_timeout=1).ping())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis check failed: %s", exc)
        return False


def _check_qdrant() -> bool:
    if settings.qdrant_url == ":memory:":
        return True
    try:
        headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
        return httpx.get(f"{settings.qdrant_url}/healthz", headers=headers, timeout=2).status_code == 200
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant check failed: %s", exc)
        return False


@app.get("/health/deps")
def health_deps() -> dict:
    """Readiness probe: are Redis and Qdrant reachable?"""
    deps = {"redis": _check_redis() if settings.redis_url != "memory" else True, "qdrant": _check_qdrant()}
    return {"status": "ok" if all(deps.values()) else "degraded", "dependencies": deps}


# ------------------------------------------------------------------ chat
class ChatIn(BaseModel):
    """Chat request."""

    user_id: str = Field(min_length=1, max_length=64, pattern=USER_ID_RE)
    message: str = Field(min_length=1, max_length=4000)


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    """Answer a message using memory + cost-aware routing."""
    try:
        return get_agent().chat(body.user_id, body.message)
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except AllModelsFailed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ------------------------------------------------------------------ memory
class MemoryIn(BaseModel):
    """Manually add a long-term memory (dev/testing)."""

    text: str = Field(min_length=1, max_length=2000)
    importance: float | None = Field(default=None, ge=0, le=1)


class TurnIn(BaseModel):
    """A completed user/assistant exchange (dev endpoint; /chat records turns automatically)."""

    user: str = Field(min_length=1, max_length=4000)
    assistant: str = Field(min_length=1, max_length=8000)


@app.post("/memory/{user_id}")
def add_memory(user_id: UserId, body: MemoryIn) -> dict:
    """Store a long-term memory for a user."""
    return {"id": get_memory().remember(user_id, body.text, importance=body.importance)}


@app.get("/memory/{user_id}")
def list_memory(user_id: UserId, q: str | None = None, k: int = 5) -> dict:
    """List a user's memories, or recall the top-k relevant ones when `q` is given."""
    mm = get_memory()
    items = mm.long.search(user_id, q, min(max(k, 1), 20)) if q else mm.list_memories(user_id)
    return {"user_id": user_id, "count": len(items), "memories": [m.__dict__ for m in items]}


@app.delete("/memory/{user_id}")
def delete_memory(user_id: UserId) -> dict:
    """GDPR-style 'forget me': memories, short-term buffer, private cache entries and analytics rows."""
    get_agent().forget_user(user_id)
    return {"user_id": user_id, "deleted": True}


@app.post("/memory/{user_id}/turn")
def record_turn(user_id: UserId, body: TurnIn) -> dict:
    """Record a turn: buffer it, extract facts, compress old context if needed."""
    return get_memory().record_turn(user_id, body.user, body.assistant)


@app.get("/memory/{user_id}/context")
def get_context(user_id: UserId, q: str, k: int = 5) -> dict:
    """What the agent would see for query `q`: recent turns + recalled memories."""
    ctx = get_memory().build_context(user_id, q, min(max(k, 1), 20))
    return {"recent": ctx["recent"], "recalled": [m.__dict__ for m in ctx["recalled"]]}


# ------------------------------------------------------------------ stats / metrics
@app.get("/stats/memory")
def memory_stats() -> dict:
    """Compression / extraction counters."""
    return get_memory().stats


@app.get("/stats/agent")
def agent_stats() -> dict:
    """In-process counters: cache hits, memory answers, LLM calls avoided %."""
    return get_agent().metrics()


@app.get("/metrics")
def metrics() -> dict:
    """Aggregated analytics: cost, savings vs always-premium, LLM calls avoided, latency, model usage."""
    agent = get_agent()
    return agent.analytics.summary(maintenance_cost=agent.memory.stats["maintenance_cost_usd"])


@app.get("/metrics/recent")
def metrics_recent(limit: int = 50, user_id: str | None = None) -> dict:
    """Most recent per-request analytics rows (newest first)."""
    return {"rows": get_agent().analytics.recent(min(max(limit, 1), 500), user_id)}
