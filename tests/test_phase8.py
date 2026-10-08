import pytest
import yaml
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.budget import BudgetExceeded, BudgetManager
from app.config import Settings, get_settings
from app.main import app
from app.memory.embeddings import HashEmbedder
from app.memory.long_term import LongTermMemory
from scripts.preflight import evaluate

c = TestClient(app)


def test_global_cap_blocks_every_user():
    b = BudgetManager(daily_cap_usd=1.0, global_cap_usd=0.05)
    b.record("alice", 0.06)
    with pytest.raises(BudgetExceeded, match="Service-wide"):
        b.check("bob")                      # bob never spent anything, but the whole service is capped
    assert b.global_spent() == pytest.approx(0.06)


def test_global_cap_optional():
    b = BudgetManager(daily_cap_usd=1.0)
    b.record("alice", 100)
    b.check("bob")                          # no global cap configured -> fine


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", "s3cret")
    body = {"user_id": "authuser", "message": "Explain DNS with an example"}
    assert c.get("/health").status_code == 200                       # liveness always open
    assert c.post("/chat", json=body).status_code == 401
    assert c.post("/chat", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/metrics").status_code == 401
    assert c.post("/chat", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200
    c.delete("/memory/authuser", headers={"X-API-Key": "s3cret"})


def test_open_when_no_api_key_configured():
    assert get_settings().api_key == ""
    assert c.get("/metrics").status_code == 200


@pytest.mark.parametrize("body", [
    {"user_id": "u", "message": ""},
    {"user_id": "u", "message": "x" * 4001},
    {"user_id": "bad id!", "message": "hi"},
    {"user_id": "u" * 65, "message": "hi"},
])
def test_chat_input_validation(body):
    assert c.post("/chat", json=body).status_code == 422


def test_path_user_id_validated():
    assert c.get("/memory/bad%20id!").status_code == 422


def test_embedding_dimension_mismatch_is_explained():
    client = QdrantClient(location=":memory:")
    w = dict(w_sim=.6, w_rec=.25, w_imp=.15, half_life_days=14)
    LongTermMemory(client, HashEmbedder(8), "col", w)
    with pytest.raises(RuntimeError, match="dimension"):
        LongTermMemory(client, HashEmbedder(16), "col", w)
    LongTermMemory(client, HashEmbedder(8), "col", w)               # same dim is fine


def test_preflight_flags_unsafe_production_config():
    bad = Settings(llm_provider="auto", app_env="production", openai_api_key="", api_key="", qdrant_url="http://localhost:6333", redis_url="redis://localhost:6379")
    status = {n: st for st, n, _ in evaluate(bad)}
    assert status["LLM key"] == "fail" and status["API_KEY"] == "fail" and status["QDRANT_URL"] == "fail"
    good = Settings(llm_provider="auto", app_env="production", openai_api_key="sk-x", api_key="k", embedding_provider="openai",
                    qdrant_url="https://x.cloud.qdrant.io:6333", qdrant_api_key="q", redis_url="redis://red-abc:6379")
    assert all(st != "fail" for st, _, _ in evaluate(good))


def test_render_blueprint_is_well_formed():
    bp = yaml.safe_load(open("render.yaml"))
    by_name = {s["name"]: s for s in bp["services"]}
    assert set(by_name) == {"costmind-api", "costmind-dashboard", "costmind-redis"}
    kv = by_name["costmind-redis"]
    assert kv["type"] == "keyvalue" and kv["ipAllowList"] == [] and kv["plan"] == "free"
    api = by_name["costmind-api"]
    assert api["healthCheckPath"] == "/health" and "$PORT" in api["startCommand"]
    env = {e["key"]: e for e in api["envVars"]}
    assert env["REDIS_URL"]["fromService"]["name"] == "costmind-redis" and env["API_KEY"]["generateValue"] is True
    assert env["OPENAI_API_KEY"]["sync"] is False
    dash = {e["key"]: e for e in by_name["costmind-dashboard"]["envVars"]}
    assert dash["API_KEY"]["fromService"]["name"] == "costmind-api"
    assert "dashboard/requirements.txt" in by_name["costmind-dashboard"]["buildCommand"]


def test_requirements_split():
    prod = open("requirements.txt").read()
    assert "pytest" not in prod and "streamlit" not in prod
    assert "pytest" in open("requirements-dev.txt").read()


def test_env_example_parses_to_safe_defaults():
    """Regression: inline '# comments' after empty values used to become the value (API_KEY='# set in production...')."""
    from dotenv import dotenv_values

    vals = dotenv_values(".env.example")          # reads the file only, independent of your shell / real .env
    assert vals["API_KEY"] == "" and vals["FALLBACK_MODELS"] == "" and vals["OPENAI_API_KEY"] == ""
    assert not [k for k, v in vals.items() if (v or "").startswith("#")]
    assert vals["CHEAP_MODEL"] == "gpt-4o-mini" and vals["MAX_USD_GLOBAL_PER_DAY"] == "2.0"
