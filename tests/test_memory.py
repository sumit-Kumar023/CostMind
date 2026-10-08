from qdrant_client import QdrantClient

from app.memory.embeddings import HashEmbedder
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermBuffer

W = dict(w_sim=0.6, w_rec=0.25, w_imp=0.15, half_life_days=14)


def make_ltm():
    return LongTermMemory(QdrantClient(location=":memory:"), HashEmbedder(), "t", W)


def test_short_term_hard_cap_and_trim():
    b = ShortTermBuffer(window=2)  # max 4, hard cap 12
    for i in range(30):
        b.add("u", "user", f"m{i}")
    assert len(b.get("u")) == 12
    b.trim_oldest("u", 10)
    assert [m["content"] for m in b.get("u")] == ["m28", "m29"]


def test_recall_returns_relevant_first():
    ltm = make_ltm()
    ltm.add("u1", "User loves cricket and plays every weekend")
    ltm.add("u1", "User is allergic to peanuts")
    top = ltm.search("u1", "what sport does the user like cricket", k=1)[0]
    assert "cricket" in top.text


def test_users_are_isolated():
    ltm = make_ltm()
    ltm.add("alice", "Alice loves chess")
    assert ltm.search("bob", "chess", k=5) == []


def test_delete_all_forgets_user():
    ltm = make_ltm()
    ltm.add("u", "something to forget")
    ltm.delete_all("u")
    assert ltm.list_all("u") == []
