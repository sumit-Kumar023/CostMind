import pytest
from qdrant_client import QdrantClient

from app.analytics import AnalyticsStore
from app.budget import BudgetManager
from app.cache import SemanticCache
from app.config import Settings
from app.graph import Agent
from app.llm import MockLLM, estimate_cost
from app.memory.embeddings import HashEmbedder
from app.memory.long_term import LongTermMemory
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer
from app.router import ModelRouter

W = dict(w_sim=0.6, w_rec=0.25, w_imp=0.15, half_life_days=14)
MODELS = {"cheap": "m-cheap", "mid": "m-mid", "premium": "m-prem"}


def rec(route="llm", cost=0.01, base=0.05, lat=100.0, user="u", model="m-cheap", esc=0):
    return dict(user_id=user, route=route, model=model, tier="cheap", escalations=esc, confidence=0.9, cost=cost,
                prompt_tokens=100, completion_tokens=20, latency_ms=lat, retrieved_memories=0, route_reason="r",
                message_chars=10, baseline_cost_est=base)


def test_summary_math():
    a = AnalyticsStore()
    for _ in range(3):
        a.log(**rec())                                   # 3 LLM calls: cost .03, baseline .15
    a.log(**rec("cache", cost=0, lat=10))                # baseline .05
    a.log(**rec("memory", cost=0, lat=5, model="memory"))
    s = a.summary(maintenance_cost=0.01)
    assert s["requests"] == 5 and s["llm_calls"] == 3
    assert s["llm_calls_avoided_pct"] == 40.0
    assert s["total_cost_usd"] == pytest.approx(0.04)
    assert s["baseline_cost_usd_est"] == pytest.approx(0.25)
    assert s["savings_pct_est"] == pytest.approx(84.0)   # (0.25 - 0.04) / 0.25
    assert s["by_route"] == {"llm": 3, "cache": 1, "memory": 1}
    assert s["p95_latency_ms"] == 100.0
    assert s["avg_latency_ms_by_route"]["cache"] == 10.0


def test_empty_summary_is_safe():
    s = AnalyticsStore().summary()
    assert s["requests"] == 0 and s["savings_pct_est"] == 0.0 and s["p95_latency_ms"] == 0.0


def test_delete_user_and_recent_filter():
    a = AnalyticsStore()
    a.log(**rec(user="a")); a.log(**rec(user="b"))
    assert len(a.recent(10, "a")) == 1
    a.delete_user("a")
    assert [r["user_id"] for r in a.recent(10)] == ["b"]


def make_agent():
    s = Settings(cheap_model="m-cheap", mid_model="m-mid", premium_model="m-prem", memory_answer_threshold=0.4)
    client, emb = QdrantClient(location=":memory:"), HashEmbedder()
    mem = MemoryManager(ShortTermBuffer(3), LongTermMemory(client, emb, "t", W), llm=MockLLM())
    return Agent(mem, ModelRouter(MockLLM(), MODELS, retry_delay=0), BudgetManager(1.0), s,
                 SemanticCache(client, emb, "c", 0.92, 3600), AnalyticsStore())


def test_agent_logs_every_request_with_savings():
    a = make_agent()
    a.chat("u", "My name is Rahul")
    a.chat("u", "what is my name")
    a.chat("u", "Explain DNS with an example")
    a.chat("v", "Explain DNS with an example")
    s = a.analytics.summary(a.memory.stats["maintenance_cost_usd"])
    assert s["requests"] == 4 and s["by_route"] == {"llm": 2, "memory": 1, "cache": 1}
    assert s["llm_calls_avoided_pct"] == 50.0
    assert s["baseline_cost_usd_est"] > s["total_cost_usd"] > 0 and s["savings_pct_est"] > 0


def test_raw_message_never_stored():
    a = make_agent()
    a.chat("u", "my secret password is hunter2")
    assert "message" not in a.analytics.recent(1)[0]
    assert "hunter2" not in str(a.analytics.recent(1)[0])


def test_forget_user_purges_analytics():
    a = make_agent()
    a.chat("u", "hello there")
    a.forget_user("u")
    assert a.analytics.recent(10, "u") == []


def test_estimate_cost_table_fallback():
    assert estimate_cost("gpt-4o", 1000, 1000) == pytest.approx(0.0125)
    assert estimate_cost("unknown-model-x", 1000, 0, prefer_litellm=True) > 0


def test_metrics_endpoints():
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    c.post("/chat", json={"user_id": "metrics-user", "message": "Explain UDP with an example"})
    m = c.get("/metrics").json()
    assert m["requests"] >= 1 and "savings_pct_est" in m and "by_model" in m
    rows = c.get("/metrics/recent", params={"limit": 5, "user_id": "metrics-user"}).json()["rows"]
    assert rows and rows[0]["user_id"] == "metrics-user"
