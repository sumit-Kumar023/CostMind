import pytest
from qdrant_client import QdrantClient

from app.budget import BudgetExceeded, BudgetManager
from app.config import Settings
from app.graph import Agent, build_messages
from app.llm import MockLLM
from app.memory.embeddings import HashEmbedder
from app.memory.long_term import LongTermMemory, Memory
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer
from app.router import AllModelsFailed, ModelRouter, classify_complexity, parse_confidence

W = dict(w_sim=0.6, w_rec=0.25, w_imp=0.15, half_life_days=14)
MODELS = {"cheap": "m-cheap", "mid": "m-mid", "premium": "m-prem"}


def make_agent(cap=1.0, llm=None, fallbacks=None, **kw):
    s = Settings(cheap_model="m-cheap", mid_model="m-mid", premium_model="m-prem", max_usd_per_user_per_day=cap, **kw)
    mem = MemoryManager(ShortTermBuffer(3), LongTermMemory(QdrantClient(location=":memory:"), HashEmbedder(), "t", W), llm=MockLLM())
    router = ModelRouter(llm or MockLLM(), MODELS, fallbacks, retry_delay=0)
    budget = BudgetManager(cap, s.low_budget_fraction)
    return Agent(mem, router, budget, s), budget


def test_classifier_tiers():
    assert classify_complexity("hi there")[0] == "cheap"
    assert classify_complexity("Compare and analyze the trade-offs of microservices vs monolith, step by step")[0] == "premium"
    assert classify_complexity("Explain DNS with an example")[0] == "mid"
    assert classify_complexity("Explain how DNS works")[0] == "cheap"


def test_parse_confidence():
    assert parse_confidence("Answer.\nCONFIDENCE: 0.82") == ("Answer.", 0.82)
    assert parse_confidence("no score") == ("no score", None)


def test_simple_query_stays_cheap_and_exits_early():
    agent, _ = make_agent()
    out = agent.chat("u", "hello")
    assert out["model"] == "m-cheap" and out["escalations"] == 0 and out["confidence"] >= 0.75


def test_low_confidence_escalates_to_premium():
    agent, _ = make_agent()
    out = agent.chat("u", "a tricky question")
    assert out["route_path"] == ["cheap", "mid", "premium"]
    assert out["model"] == "m-prem" and out["escalations"] == 2
    assert "escalate" in out["route_reason"]


def test_per_request_cost_cap_stops_escalation():
    agent, _ = make_agent(max_usd_per_request=0.0)
    out = agent.chat("u", "a tricky question")
    assert out["escalations"] == 0 and out["model"] == "m-cheap"


def test_low_budget_caps_to_cheap_and_reduces_k():
    agent, budget = make_agent(cap=1.0)
    budget.record("u", 0.9)
    out = agent.chat("u", "a tricky question")
    assert out["tier"] == "cheap" and out["escalations"] == 0
    assert out["retrieval_k"] == 1


def test_budget_exceeded_raises():
    agent, budget = make_agent(cap=0.01)
    budget.record("u", 0.02)
    with pytest.raises(BudgetExceeded):
        agent.chat("u", "hello")


def test_spend_is_recorded():
    agent, budget = make_agent()
    out = agent.chat("u", "hello")
    assert budget.spent("u") == pytest.approx(out["cost"], abs=1e-6)


def test_memory_used_in_prompt_and_facts_saved():
    agent, _ = make_agent()
    agent.chat("u", "My name is Rahul")
    assert any("Rahul" in m.text for m in agent.memory.list_memories("u"))
    msgs, _ = build_messages(agent.memory.build_context("u", "name?"), "name?", 4000)
    assert "Rahul" in msgs[0]["content"]


def test_prompt_trimmed_to_token_cap():
    ctx = {"recalled": [Memory(str(i), "fact " * 50, "fact", 0.5, 0.0) for i in range(5)],
           "recent": [{"role": "user", "content": "x" * 400}] * 4}
    msgs, dropped = build_messages(ctx, "q", cap_tokens=150)
    assert dropped > 0


class FlakyLLM(MockLLM):
    def __init__(self, bad):
        self.bad = bad

    def complete(self, messages, model="mock", max_tokens=300):
        if model in self.bad:
            raise RuntimeError("provider down")
        return super().complete(messages, model, max_tokens)


def test_fallback_when_primary_fails():
    agent, _ = make_agent(llm=FlakyLLM({"m-cheap"}), fallbacks=["m-backup"])
    out = agent.chat("u", "hello")
    assert out["model"] == "m-backup"
    assert [a["ok"] for a in out["attempts"]] == [False, False, True]


def test_all_models_fail():
    r = ModelRouter(FlakyLLM({"m-cheap"}), MODELS, retry_delay=0)
    with pytest.raises(AllModelsFailed):
        r.call("cheap", [{"role": "user", "content": "x"}])


def test_next_tier_skips_duplicate_models():
    r = ModelRouter(MockLLM(), {"cheap": "a", "mid": "a", "premium": "b"})
    assert r.next_tier("cheap") == "premium" and r.next_tier("premium") is None
