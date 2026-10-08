from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)


def test_memory_roundtrip():
    c.post("/memory/rahul", json={"text": "Rahul loves cricket"})
    c.post("/memory/rahul", json={"text": "Rahul is allergic to peanuts"})
    r = c.get("/memory/rahul", params={"q": "which sport cricket", "k": 1}).json()
    assert "cricket" in r["memories"][0]["text"]
    assert c.get("/memory/rahul").json()["count"] == 2
    assert c.delete("/memory/rahul").json()["deleted"] is True
    assert c.get("/memory/rahul").json()["count"] == 0


def test_turn_context_stats_endpoints():
    r = c.post("/memory/amit/turn", json={"user": "My name is Amit", "assistant": "Hi Amit"}).json()
    assert r["facts_stored"] == 1
    ctx = c.get("/memory/amit/context", params={"q": "what is my name"}).json()
    assert len(ctx["recent"]) == 2 and "Amit" in ctx["recalled"][0]["text"]
    assert c.get("/stats/memory").json()["facts_stored"] >= 1
    c.delete("/memory/amit")


def test_chat_endpoint():
    r = c.post("/chat", json={"user_id": "chatter", "message": "My name is Sam"})
    assert r.status_code == 200
    d = r.json()
    for key in ("reply", "model", "cost", "prompt_tokens", "cache_hit", "answered_from_memory", "route_reason"):
        assert key in d
    c.delete("/memory/chatter")
