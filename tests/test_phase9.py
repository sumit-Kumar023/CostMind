import pytest
from qdrant_client import QdrantClient

from app.analytics import AnalyticsStore
from app.budget import BudgetManager
from app.cache import SemanticCache
from app.config import Settings
from app.graph import Agent, build_messages
from app.llm import CONFIDENCE_REMINDER, LLMResponse, MockLLM
from app.memory.embeddings import HashEmbedder
from app.memory.long_term import LongTermMemory
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer
from app.router import ModelRouter
from scripts.tune_memory_threshold import analyze, cosine

W = dict(w_sim=0.6, w_rec=0.25, w_imp=0.15, half_life_days=14)
MODELS = {"cheap": "m-cheap", "mid": "m-mid", "premium": "m-prem"}


class ForgetfulLLM(MockLLM):
    """Chat answers WITHOUT the CONFIDENCE line (what gpt-4o-mini did for casual chat)."""

    def complete(self, messages, model="mock", max_tokens=300):
        res = super().complete(messages, model, max_tokens)
        if "TASK: CHAT" in messages[0]["content"]:
            res = LLMResponse(res.text.split("\nCONFIDENCE")[0], res.prompt_tokens, res.completion_tokens, res.cost, res.model)
        return res


def make(llm):
    s = Settings(cheap_model="m-cheap", mid_model="m-mid", premium_model="m-prem")
    client, emb = QdrantClient(location=":memory:"), HashEmbedder()
    mem = MemoryManager(ShortTermBuffer(3), LongTermMemory(client, emb, "t", W), llm=MockLLM())
    return Agent(mem, ModelRouter(llm, MODELS, retry_delay=0), BudgetManager(1.0), s, SemanticCache(client, emb, "c", 0.92, 3600), AnalyticsStore())


def test_missing_confidence_is_not_escalated():
    a = make(ForgetfulLLM())
    out = a.chat("u", "hey how is it going today")
    assert out["confidence"] is None and out["escalations"] == 0 and out["model"] == "m-cheap"
    assert "no confidence score" in out["route_reason"]
    assert a.stats["unscored"] == 1


def test_unscored_answers_are_not_cached():
    a = make(ForgetfulLLM())
    a.chat("u", "Explain DNS with an example")
    assert a.chat("u", "Explain DNS with an example")["cache_hit"] is False


def test_scored_low_confidence_still_escalates():
    a = make(MockLLM())
    out = a.chat("u", "a tricky question")
    assert out["escalations"] == 2 and out["model"] == "m-prem"


def test_analytics_handles_unscored_rows():
    a = make(ForgetfulLLM())
    a.chat("u", "hello there friend")
    s = a.analytics.summary()
    assert s["unscored_pct"] == 100.0 and s["avg_confidence"] == 0.0


def test_reminder_goes_on_last_user_turn_but_not_into_memory():
    a = make(MockLLM())
    ctx = {"recalled": [], "recent": []}
    msgs, _ = build_messages(ctx, "What is DNS?", 4000)
    assert msgs[-1]["content"].endswith(CONFIDENCE_REMINDER) and "CONFIDENCE" in msgs[0]["content"]
    a.chat("u", "My name is Rahul")
    assert all(CONFIDENCE_REMINDER not in m["content"] for m in a.memory.short.get("u"))


def test_threshold_tuner_math_and_separation():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0) and cosine([1, 0], [0, 1]) == 0.0
    r = analyze(HashEmbedder().embed)
    assert set(r) == {"correct_min", "correct_avg", "wrong_max", "separable", "suggested"}
    assert r["correct_avg"] > 0
