import os

os.environ["QDRANT_URL"] = ":memory:"
os.environ["REDIS_URL"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["ANALYTICS_DB"] = ":memory:"
os.environ["WARMUP"] = "false"

# Tests must never see (or use) real credentials from the developer's .env or shell.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "API_KEY", "QDRANT_API_KEY", "FALLBACK_MODELS"):
    os.environ[_k] = ""
