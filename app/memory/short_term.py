"""Short-term buffer (Redis-backed, in-memory fallback).

Holds up to `max_messages` before compression is triggered. A hard cap (3x) guards
against unbounded growth if compression fails.
"""
import json
import logging
from collections import defaultdict, deque

import redis

logger = logging.getLogger("costmind.memory")


class ShortTermBuffer:
    """Recent turns per user (2 messages per turn)."""

    def __init__(self, window: int = 6, redis_url: str | None = None) -> None:
        self.max_messages = window * 2
        self.hard_cap = self.max_messages * 3
        self._r = None
        if redis_url:
            try:
                client = redis.from_url(redis_url, socket_connect_timeout=1, decode_responses=True)
                client.ping()
                self._r = client
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis unavailable (%s); using in-memory short-term buffer", exc)
        self._local: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.hard_cap))

    @staticmethod
    def _key(user_id: str) -> str:
        return f"costmind:st:{user_id}"

    def add(self, user_id: str, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        if self._r:
            k = self._key(user_id)
            self._r.rpush(k, json.dumps(msg))
            self._r.ltrim(k, -self.hard_cap, -1)
        else:
            self._local[user_id].append(msg)

    def get(self, user_id: str) -> list[dict]:
        if self._r:
            return [json.loads(m) for m in self._r.lrange(self._key(user_id), 0, -1)]
        return list(self._local[user_id])

    def trim_oldest(self, user_id: str, n: int) -> None:
        """Drop the n oldest messages (after they were summarised)."""
        if self._r:
            self._r.ltrim(self._key(user_id), n, -1)
        else:
            for _ in range(min(n, len(self._local[user_id]))):
                self._local[user_id].popleft()

    def clear(self, user_id: str) -> None:
        if self._r:
            self._r.delete(self._key(user_id))
        else:
            self._local.pop(user_id, None)
