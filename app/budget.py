"""Per-request, per-user (daily) and GLOBAL (daily) budgets. Redis-backed with in-memory fallback."""
import datetime as dt
import logging

import redis

logger = logging.getLogger("costmind.budget")


class BudgetExceeded(Exception):
    """Raised when a user (or the whole service) has used up today's USD budget."""


class BudgetManager:
    """Tracks daily USD spend per user and globally, and derives budget-aware behaviour."""

    def __init__(self, daily_cap_usd: float, low_fraction: float = 0.2, redis_url: str | None = None,
                 global_cap_usd: float | None = None) -> None:
        self.cap, self.low_fraction, self.global_cap = daily_cap_usd, low_fraction, global_cap_usd
        self._local: dict[str, float] = {}
        self._r = None
        if redis_url:
            try:
                c = redis.from_url(redis_url, socket_connect_timeout=1)
                c.ping()
                self._r = c
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable (%s); budgets kept in memory", exc)

    # ---- storage primitives
    @staticmethod
    def _key(user_id: str) -> str:
        return f"costmind:spend:{user_id}:{dt.date.today().isoformat()}"

    @staticmethod
    def _gkey() -> str:
        return f"costmind:spend-global:{dt.date.today().isoformat()}"

    def _get(self, key: str) -> float:
        return float(self._r.get(key) or 0.0) if self._r else self._local.get(key, 0.0)

    def _add(self, key: str, cost: float) -> None:
        if self._r:
            self._r.incrbyfloat(key, cost)
            self._r.expire(key, 172800)
        else:
            self._local[key] = self._local.get(key, 0.0) + cost

    # ---- public API
    def spent(self, user_id: str) -> float:
        return self._get(self._key(user_id))

    def global_spent(self) -> float:
        return self._get(self._gkey())

    def record(self, user_id: str, cost: float) -> None:
        self._add(self._key(user_id), cost)
        self._add(self._gkey(), cost)

    def remaining(self, user_id: str) -> float:
        return max(0.0, self.cap - self.spent(user_id))

    def check(self, user_id: str) -> None:
        """Hard stops: the service-wide daily cap first, then the user's own cap."""
        if self.global_cap is not None and self.global_spent() >= self.global_cap:
            raise BudgetExceeded(f"Service-wide daily budget of ${self.global_cap:.2f} reached. Try again tomorrow.")
        if self.remaining(user_id) <= 0:
            raise BudgetExceeded(f"Daily budget of ${self.cap:.2f} exhausted for user '{user_id}'")

    def is_low(self, user_id: str) -> bool:
        """Soft limit: remaining budget is small, so stay on the cheap tier."""
        return self.remaining(user_id) <= self.cap * self.low_fraction

    def retrieval_k(self, user_id: str, base_k: int) -> int:
        """Budget-aware retrieval: recall fewer memories as the budget shrinks."""
        frac = self.remaining(user_id) / self.cap if self.cap else 0.0
        if frac <= 0.1:
            return 1
        if frac <= 0.3:
            return max(1, base_k // 2)
        return base_k
