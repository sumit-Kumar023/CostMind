from qdrant_client import QdrantClient

from app.llm import MockLLM
from app.memory.embeddings import HashEmbedder
from app.memory.long_term import LongTermMemory
from app.memory.maintenance import parse_facts
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermBuffer

W = dict(w_sim=0.6, w_rec=0.25, w_imp=0.15, half_life_days=14)


def make(window=2, client=None, short=None):
    client = client or QdrantClient(location=":memory:")
    ltm = LongTermMemory(client, HashEmbedder(), "t", W)
    return MemoryManager(short or ShortTermBuffer(window), ltm, llm=MockLLM()), client


def test_parse_facts_handles_fences_and_garbage():
    assert parse_facts('```json\n[{"fact":"User likes tea","importance":0.7}]\n```')[0].text == "User likes tea"
    assert parse_facts("not json at all") == []
    assert parse_facts('[{"nope":1}]') == []


def test_fact_extraction_and_dedup():
    m, _ = make()
    r1 = m.record_turn("u", "My name is Rahul and I love cricket", "Nice to meet you!")
    assert r1["facts_stored"] == 2
    r2 = m.record_turn("u", "My name is Rahul", "Hello again")
    assert r2["facts_stored"] == 0 and r2["duplicates_skipped"] == 1


def test_compression_summarises_and_saves_tokens():
    m, _ = make(window=2)  # max 4 messages
    long_text = "word " * 120
    saved_total = 0
    for i in range(4):
        out = m.record_turn("u", f"{long_text} turn{i}", f"{long_text} reply{i}")
        saved_total += out["tokens_saved"]
    assert m.stats["compressions"] >= 1
    assert saved_total > 0
    assert len(m.short.get("u")) <= m.short.max_messages
    assert any(x.kind == "summary" for x in m.list_memories("u"))


def test_cross_session_recall_with_fresh_short_term():
    m1, client = make()
    m1.record_turn("rahul", "I love cricket", "Great sport!")
    # "new session": brand-new manager + empty short-term buffer, same persistent vector store
    m2, _ = make(client=client, short=ShortTermBuffer(2))
    ctx = m2.build_context("rahul", "which cricket sport do I love")
    assert ctx["recent"] == []
    assert "cricket" in ctx["recalled"][0].text


def test_forget_user_clears_everything():
    m, _ = make()
    m.record_turn("u", "I love tea", "ok")
    m.forget_user("u")
    assert m.list_memories("u") == [] and m.short.get("u") == []


def test_extraction_gated_to_personal_messages():
    m, _ = make()
    out = m.record_turn("u", "What is DNS?", "A naming system.")
    assert out["facts_stored"] == 0 and m.stats["extractions_skipped"] == 1
    assert m.record_turn("u", "I love tea", "ok")["facts_stored"] == 1
