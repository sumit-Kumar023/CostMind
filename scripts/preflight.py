"""Pre-deploy checklist. Run with your production env vars set:  python -m scripts.preflight

Exit code 1 if any check FAILS (warnings don't fail).
"""
import sys

from app.config import Settings, get_settings
from app.memory.embeddings import HashEmbedder, build_embedder

Check = tuple[str, str, str]  # (status ok|warn|fail, name, detail)


def _local(url: str) -> bool:
    return url in ("", ":memory:", "memory") or "localhost" in url or "127.0.0.1" in url


def evaluate(s: Settings) -> list[Check]:
    """Offline configuration checks (no network)."""
    prod, out = s.app_env == "production", []
    has_key = bool(s.openai_api_key or s.anthropic_api_key)
    if s.llm_provider == "mock":
        out.append(("warn" if not prod else "fail", "LLM provider", "LLM_PROVIDER=mock returns fake answers"))
    elif has_key:
        out.append(("ok", "LLM key", "OPENAI_API_KEY / ANTHROPIC_API_KEY present"))
    else:
        out.append(("fail", "LLM key", "no OPENAI_API_KEY or ANTHROPIC_API_KEY - the app would silently use the mock LLM"))
    emb = build_embedder(s)
    if isinstance(getattr(emb, "inner", emb), HashEmbedder):
        out.append(("fail" if prod else "warn", "Embeddings", "offline hash embedder in use: recall/cache will be poor. Set OPENAI_API_KEY"))
    else:
        out.append(("ok", "Embeddings", f"API embeddings ({s.embedding_model})"))
    out.append(("ok", "API_KEY", "set: endpoints require X-API-Key") if s.api_key else
               ("fail" if prod else "warn", "API_KEY", "empty: anyone with the URL can spend your API credit"))
    g = s.max_usd_global_per_day
    out.append(("ok", "Global daily cap", f"${g:.2f}/day") if 0 < g <= 10 else ("warn", "Global daily cap", f"${g:.2f}/day - is that intended?"))
    for name, url in (("QDRANT_URL", s.qdrant_url), ("REDIS_URL", s.redis_url)):
        out.append(("fail" if prod and _local(url) else "ok" if not _local(url) else "warn", name,
                    "points at localhost/in-memory" if _local(url) else "external service configured"))
    if prod and not s.qdrant_api_key and "qdrant.io" in s.qdrant_url:
        out.append(("fail", "QDRANT_API_KEY", "Qdrant Cloud needs an API key"))
    return out


def network_checks(s: Settings) -> list[Check]:
    """Reachability checks for Qdrant and Redis."""
    out: list[Check] = []
    try:
        import redis

        redis.from_url(s.redis_url, socket_connect_timeout=3).ping()
        out.append(("ok", "Redis reachable", s.redis_url.split("@")[-1]))
    except Exception as exc:  # noqa: BLE001
        out.append(("fail", "Redis reachable", str(exc)[:100]))
    try:
        from qdrant_client import QdrantClient

        names = [c.name for c in QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=8).get_collections().collections]
        out.append(("ok", "Qdrant reachable", f"collections: {names}"))
    except Exception as exc:  # noqa: BLE001
        out.append(("fail", "Qdrant reachable", str(exc)[:100]))
    return out


def main() -> int:
    s = get_settings()
    results = evaluate(s) + ([] if s.qdrant_url == ":memory:" else network_checks(s))
    icon = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    for status, name, detail in results:
        print(f"{icon[status]} {name}: {detail}")
    failed = sum(1 for r in results if r[0] == "fail")
    print(f"\n{failed} failed, {sum(1 for r in results if r[0] == 'warn')} warnings")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
