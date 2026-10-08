"""Relevance scoring: weighted similarity + recency + importance."""
import math
import re
import time

SECONDS_PER_DAY = 86400.0


def recency_score(created_at: float, now: float | None = None, half_life_days: float = 14.0) -> float:
    """Exponential decay in [0, 1]: 1.0 when brand new, 0.5 after one half-life."""
    now = time.time() if now is None else now
    age_days = max(0.0, now - created_at) / SECONDS_PER_DAY
    return math.exp(-math.log(2) * age_days / half_life_days)


def relevance_score(
    similarity: float,
    created_at: float,
    importance: float,
    w_sim: float = 0.6,
    w_rec: float = 0.25,
    w_imp: float = 0.15,
    half_life_days: float = 14.0,
    now: float | None = None,
) -> float:
    """Combine signals into one score in [0, 1] (weights are normalised)."""
    sim = min(1.0, max(0.0, similarity))
    imp = min(1.0, max(0.0, importance))
    total = (w_sim + w_rec + w_imp) or 1.0
    rec = recency_score(created_at, now, half_life_days)
    return (w_sim * sim + w_rec * rec + w_imp * imp) / total


_HIGH = re.compile(r"\b(my name|i am|i'm|i live|allergic|birthday|i work|i love|i hate|my goal|prefer)\b", re.I)


def estimate_importance(text: str) -> float:
    """Cheap heuristic importance (Phase 3 upgrades this with a cheap LLM)."""
    base = 0.8 if _HIGH.search(text) else 0.4
    return min(1.0, base + (0.1 if len(text) > 80 else 0.0))
