"""Per-request analytics in SQLite (cost, tokens, latency, route, cache/memory hits, baseline estimate).

Note: on Render's free tier the filesystem is ephemeral, so this resets on redeploy.
Point ANALYTICS_DB at a persistent disk (or swap in Postgres) if you need history to survive.
"""
import math
import sqlite3
import threading
import time

_COLS = ["ts", "user_id", "route", "model", "tier", "escalations", "confidence", "cost", "prompt_tokens",
         "completion_tokens", "latency_ms", "retrieved_memories", "route_reason", "message_chars", "baseline_cost_est"]


class AnalyticsStore:
    """Thread-safe SQLite store (FastAPI runs sync endpoints in a thread pool)."""

    def __init__(self, path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, user_id TEXT, route TEXT, model TEXT, tier TEXT,
                escalations INTEGER, confidence REAL, cost REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
                latency_ms REAL, retrieved_memories INTEGER, route_reason TEXT, message_chars INTEGER,
                baseline_cost_est REAL)""")
            self._db.commit()

    def log(self, **rec) -> None:
        rec.setdefault("ts", time.time())
        with self._lock:
            self._db.execute(f"INSERT INTO requests ({','.join(_COLS)}) VALUES ({','.join('?' * len(_COLS))})",
                             [rec[c] for c in _COLS])
            self._db.commit()

    def recent(self, limit: int = 50, user_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM requests", []
        if user_id:
            q += " WHERE user_id = ?"; args.append(user_id)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
        return [dict(r) for r in rows]

    def avg_llm_baseline(self) -> float:
        """Average premium-baseline cost of past LLM requests (used to estimate savings of memory answers)."""
        with self._lock:
            v = self._db.execute("SELECT AVG(baseline_cost_est) FROM requests WHERE route='llm'").fetchone()[0]
        return float(v or 0.0)

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM requests WHERE user_id = ?", (user_id,))
            self._db.commit()

    def summary(self, maintenance_cost: float = 0.0) -> dict:
        """Aggregates for /metrics and the dashboard. `maintenance_cost` = cheap-model memory upkeep (summaries/facts)."""
        with self._lock:
            rows = [dict(r) for r in self._db.execute("SELECT * FROM requests").fetchall()]
        n = len(rows)
        by = lambda key: _count(rows, key)  # noqa: E731
        cost = sum(r["cost"] for r in rows)
        baseline = sum(r["baseline_cost_est"] for r in rows)
        total = cost + maintenance_cost
        avoided = sum(1 for r in rows if r["route"] != "llm")
        lat = sorted(r["latency_ms"] for r in rows)
        llm = [r for r in rows if r["route"] == "llm"]
        scored = [r for r in llm if r["confidence"] is not None]
        route_lat: dict[str, list[float]] = {}
        for r in rows:
            route_lat.setdefault(r["route"], []).append(r["latency_ms"])
        return {
            "requests": n,
            "llm_calls": len(llm),
            "llm_calls_avoided_pct": _pct(avoided, n),
            "cache_hits": sum(1 for r in rows if r["route"] == "cache"),
            "memory_answers": sum(1 for r in rows if r["route"] == "memory"),
            "chat_cost_usd": round(cost, 6),
            "maintenance_cost_usd": round(maintenance_cost, 6),
            "total_cost_usd": round(total, 6),
            "baseline_cost_usd_est": round(baseline, 6),
            "savings_pct_est": _pct(baseline - total, baseline),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "avg_latency_ms": round(sum(lat) / n, 1) if n else 0.0,
            "p95_latency_ms": lat[max(0, math.ceil(0.95 * n) - 1)] if n else 0.0,
            "avg_latency_ms_by_route": {k: round(sum(v) / len(v), 1) for k, v in route_lat.items()},
            "avg_confidence": round(sum(r["confidence"] for r in scored) / len(scored), 3) if scored else 0.0,
            "unscored_pct": _pct(len(llm) - len(scored), len(llm)),
            "escalation_rate_pct": _pct(sum(1 for r in llm if r["escalations"] > 0), len(llm)),
            "by_route": by("route"), "by_model": by("model"), "by_tier": by("tier"),
        }


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def _pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0
