from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_deps_shape():
    r = client.get("/health/deps")
    assert r.status_code == 200
    assert set(r.json()["dependencies"]) == {"redis", "qdrant"}
