import json

import pytest
from qdrant_client import QdrantClient

import benchmarks.run_benchmark as rb
from app.budget import BudgetManager
from app.cache import SemanticCache
from app.config import Settings
from app.graph import Agent
from app.llm import MockLLM
from app.memory.embeddings import HashEmbedder
from app.memory.long_term import LongTermMemory
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer
from app.router import ModelRouter
from benchmarks.generate_queries import generate
from scripts import diagnose_recall

W = dict(w_sim=0.6, w_rec=0.25, w_imp=0.15, half_life_days=14)


def make_agent():
    s = Settings(cheap_model="m-cheap", mid_model="m-mid", premium_model="m-prem")
    client, emb = QdrantClient(location=":memory:"), HashEmbedder()
    mem = MemoryManager(ShortTermBuffer(3), LongTermMemory(client, emb, "t", W), llm=MockLLM())
    return Agent(mem, ModelRouter(MockLLM(), {"cheap": "m-cheap", "mid": "m-mid", "premium": "m-prem"}, retry_delay=0),
                 BudgetManager(1.0), s, SemanticCache(client, emb, "c", 0.92, 3600))


def test_stage_timings_for_llm_and_cache_routes():
    a = make_agent()
    llm = a.chat("u", "Explain DNS with an example")["timings"]
    assert {"retrieve", "shortcut", "classify", "generate", "finish"} <= set(llm) and all(v >= 0 for v in llm.values())
    cache = a.chat("v", "Explain DNS with an example")
    assert cache["cache_hit"] and {"retrieve", "shortcut", "finish_shortcut"} <= set(cache["timings"]) and "generate" not in cache["timings"]


def test_escalated_generate_time_accumulates():
    out = make_agent().chat("u", "a tricky question")
    assert out["escalations"] == 2 and "escalate" in out["timings"] and out["timings"]["generate"] >= 0


def _result():
    return rb.RunResult(n=4, cost=1.5, prompt_tokens=900, completion_tokens=100, latency_ms_total=8000, recall_total=2,
                        recall_correct=2, answers=["a", "b", "c", "d"], routes=["llm"] * 4)


def test_baseline_cache_roundtrip_and_key_isolation(tmp_path):
    qs = generate(320)[:4]
    path = str(tmp_path / "b.json")
    key = rb.dataset_key(qs, "gpt-4o", False)
    rb.save_baseline(path, key, _result())
    got = rb.load_baseline(path, key)
    assert got.cost == 1.5 and got.answers == ["a", "b", "c", "d"] and got.avg_latency_ms == 2000
    assert rb.load_baseline(path, rb.dataset_key(qs, "gpt-4o-mini", False)) is None   # different premium model
    assert rb.load_baseline(path, rb.dataset_key(qs, "gpt-4o", True)) is None         # a MOCK baseline can never pass as real
    assert rb.load_baseline(path, rb.dataset_key(qs[:3], "gpt-4o", False)) is None    # different dataset
    assert rb.load_baseline(str(tmp_path / "missing.json"), key) is None


def test_baseline_seeded_from_results(tmp_path):
    qs = generate(320)
    res = {"mock": False, "n_queries": 320, "models": {"premium": "gpt-4o"},
           "baseline": {"cost_usd": 1.27, "prompt_tokens": 567088, "completion_tokens": 40000, "avg_latency_ms": 2130, "recall_acc_pct": 100.0}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(res))
    b = rb.baseline_from_results(str(p), qs, "gpt-4o")
    assert b.cost == 1.27 and b.n == 320 and b.recall_total == 24 and b.recall_correct == 24 and b.answers == []
    assert rb.baseline_from_results(str(p), qs[:100], "gpt-4o") is None               # query count differs
    assert rb.baseline_from_results(str(p), qs, "gpt-4o-mini") is None                # premium model differs
    p.write_text(json.dumps({**res, "mock": True}))
    assert rb.baseline_from_results(str(p), qs, "gpt-4o") is None                     # mock results are never reusable


def test_by_complexity_buckets_scores():
    qs = [{"complexity": "simple"}, {"complexity": "complex"}, {"complexity": "complex"}]
    per_item = [{"idx": 0, "ref": 5, "test": 5}, {"idx": 1, "ref": 5, "test": 3}, {"idx": 2, "ref": 4, "test": 2}]
    out = rb.by_complexity(per_item, [0, 1, 2], qs)
    assert out["simple"] == {"n": 1, "reference_mean": 5.0, "tested_mean": 5.0}
    assert out["complex"] == {"n": 2, "reference_mean": 4.5, "tested_mean": 2.5}


def test_end_to_end_mock_reuses_baseline_and_runs_cheap_baseline(tmp_path, monkeypatch):
    calls = []
    real = rb.run_baseline
    monkeypatch.setattr(rb, "run_baseline", lambda llm, model, *a, **k: (calls.append(model), real(llm, model, *a, **k))[1])
    argv = ["--mock", "--limit", "120", "--yes", "--cheap-baseline", "--out", str(tmp_path / "r.json"), "--baseline-cache", str(tmp_path / "b.json")]
    r1 = rb.main(argv)
    assert calls == ["gpt-4o", "gpt-4o-mini"] and "cheap_baseline" in r1
    calls.clear()
    r2 = rb.main(argv)
    assert calls == ["gpt-4o-mini"]                                      # premium baseline came from the cache
    assert r2["baseline"]["cost_usd"] == r1["baseline"]["cost_usd"]
    c = r2["costmind"]
    assert set(c["latency_by_route_ms"]) <= {"llm", "cache", "memory"} and "llm" in c["stage_ms_avg_by_route"]
    recalls = [q for q in r2["per_query"] if q["kind"] == "recall"]
    assert all({"expected", "reply", "correct"} <= set(q) for q in recalls)
    assert "Stage latency" in (tmp_path / "r.md").read_text()


def test_diagnose_recall_structure():
    s = Settings(cheap_model="m-cheap", mid_model="m-mid", premium_model="m-prem", memory_answer_threshold=0.3)
    qs = [q for q in generate(320) if q["user_id"].startswith("p")][:80]
    rows = diagnose_recall.run(s, MockLLM(), qs)
    assert all({"user", "question", "expected", "route", "reply", "ok", "stored_facts", "recalled"} <= set(r) for r in rows)
