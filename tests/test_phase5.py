import pytest
from qdrant_client import QdrantClient

from app.budget import BudgetManager
from app.cache import SemanticCache, is_cacheable
from app.config import Settings
from app.graph import Agent
from app.llm import MockLLM
from app.memory.embeddings import CachedEmbedder, HashEmbedder
from app.memory.long_term import LongTermMemory, Memory
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer
from app.memory_router import answer_from_memory, is_recall_question, rewrite_fact
from app.router import ModelRouter

W = dict(w_sim=0.6, w_rec=0.25, w_imp=0.15, half_life_days=14)
MODELS = {"cheap": "m-cheap", "mid": "m-mid", "premium": "m-prem"}


def make(cache=True, **kw):
    kw.setdefault("memory_answer_threshold", 0.4)
    s = Settings(cheap_model="m-cheap", mid_model="m-mid", premium_model="m-prem", **kw)
    client, emb = QdrantClient(location=":memory:"), HashEmbedder()
    mem = MemoryManager(ShortTermBuffer(3), LongTermMemory(client, emb, "t", W), llm=MockLLM())
    c = SemanticCache(client, emb, "c", 0.92, 3600) if cache else None
    return Agent(mem, ModelRouter(MockLLM(), MODELS, retry_delay=0), BudgetManager(1.0), s, c)


def test_embedder_lru_embeds_once():
    class Counting(HashEmbedder):
        n = 0

        def embed(self, t):
            Counting.n += 1
            return super().embed(t)

    e = CachedEmbedder(Counting())
    e.embed("hello"); e.embed("hello"); e.embed("world")
    assert Counting.n == 2 and e.hits == 1


def test_cacheability_rules():
    assert is_cacheable("Explain DNS with an example")
    assert not is_cacheable("what is the weather today")
    assert not is_cacheable("what about that one")
    assert not is_cacheable("My name is Sam")


def test_cache_scopes_and_ttl():
    client, emb = QdrantClient(location=":memory:"), HashEmbedder()
    c = SemanticCache(client, emb, "c", 0.92, 3600)
    c.store(None, "What is Python programming", "A language.", "m", 0.01, 100)
    c.store("alice", "Alice private question here", "secret", "m", 0.01, 100)
    assert c.lookup("bob", "What is Python programming").answer == "A language."
    assert c.lookup("bob", "totally unrelated quantum chromodynamics") is None
    assert c.lookup("alice", "Alice private question here").answer == "secret"
    assert c.lookup("bob", "Alice private question here") is None
    c.ttl = -1
    assert c.lookup("bob", "What is Python programming") is None


def test_rewrite_and_recall_helpers():
    assert rewrite_fact("User's name is Rahul") == "Your name is Rahul"
    assert rewrite_fact("User loves cricket") == "You love cricket"
    assert rewrite_fact("User is allergic to peanuts") == "You are allergic to peanuts"
    assert is_recall_question("what is my name?") and not is_recall_question("what is DNS?")
    low = [Memory("1", "User loves tea", "fact", 0.5, 0, similarity=0.1)]
    assert answer_from_memory(low, 0.5) is None


def test_second_identical_query_is_cache_hit_with_zero_cost():
    a = make()
    first = a.chat("u1", "Explain DNS with an example")
    second = a.chat("u2", "Explain DNS with an example")  # global scope -> shared
    assert not first["cache_hit"] and second["cache_hit"]
    assert second["cost"] == 0 and second["prompt_tokens"] == 0 and second["model"] == "cache"
    m = a.metrics()
    assert m["cache_hits"] == 1 and m["llm_calls_avoided_pct"] == 50.0 and m["saved_usd_est"] > 0


def test_answers_using_personal_memory_stay_private():
    a = make()
    a.chat("u1", "I love cricket")
    assert not a.chat("u1", "Explain cricket with an example")["cache_hit"]
    assert a.chat("u1", "Explain cricket with an example")["cache_hit"]
    assert not a.chat("u2", "Explain cricket with an example")["cache_hit"]  # u2 must not see u1's private answer


def test_recall_question_answered_from_memory_without_llm():
    a = make()
    a.chat("u", "My name is Rahul and I love cricket")
    out = a.chat("u", "what is my name")
    assert out["answered_from_memory"] and out["model"] == "memory" and out["cost"] == 0
    assert "Your name is Rahul" in out["reply"]
    assert a.metrics()["memory_answers"] == 1


def test_weak_memory_match_starts_cheap_instead():
    a = make(memory_answer_threshold=0.99)
    a.chat("u", "My name is Rahul")
    out = a.chat("u", "what is my name, analyze the trade-offs step by step")
    assert not out["answered_from_memory"] and out["tier"] == "cheap" and "recall question" in out["route_reason"]


def test_low_confidence_answers_are_not_cached():
    a = make()
    a.chat("u", "a tricky question for you")  # escalates to premium (conf .95) -> cached
    assert a.chat("u", "a tricky question for you")["cache_hit"]
    b = make(confidence_threshold=0.99)  # nothing reaches the bar
    b.chat("u", "Explain DNS with an example")
    assert not b.chat("u", "Explain DNS with an example")["cache_hit"]


def test_forget_user_purges_private_cache():
    a = make()
    a.chat("u1", "I love cricket")
    a.chat("u1", "Explain cricket with an example")
    a.forget_user("u1")
    assert a.cache.lookup("u1", "Explain cricket with an example") is None


def test_api_cache_and_stats():
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    body = {"user_id": "apiuser", "message": "Explain TCP handshakes with an example"}
    assert c.post("/chat", json=body).json()["cache_hit"] is False
    assert c.post("/chat", json=body).json()["cache_hit"] is True
    assert c.get("/stats/agent").json()["cache_hits"] >= 1
