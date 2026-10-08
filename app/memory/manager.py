"""Facade combining short-term, long-term, compression and fact extraction."""
from app.llm import LLMClient, estimate_tokens
from app.memory.long_term import LongTermMemory, Memory
from app.memory.maintenance import ContextCompressor, FactExtractor, looks_personal
from app.memory.short_term import ShortTermBuffer


class MemoryManager:
    """Single entry point the agent graph uses."""

    def __init__(
        self,
        short: ShortTermBuffer,
        long: LongTermMemory,
        top_k: int = 5,
        llm: LLMClient | None = None,
        cheap_model: str = "gpt-4o-mini",
        dedup_threshold: float = 0.92,
    ) -> None:
        self.short, self.long, self.top_k, self.dedup_threshold = short, long, top_k, dedup_threshold
        self.compressor = ContextCompressor(llm, cheap_model) if llm else None
        self.extractor = FactExtractor(llm, cheap_model) if llm else None
        self.stats = {
            "compressions": 0, "tokens_saved": 0, "facts_stored": 0,
            "facts_skipped_duplicate": 0, "extractions_skipped": 0, "maintenance_cost_usd": 0.0,
        }

    # ---- read path
    def build_context(self, user_id: str, query: str, k: int | None = None) -> dict:
        """Recent turns + top-k relevant long-term memories (summaries and facts)."""
        return {"recent": self.short.get(user_id), "recalled": self.long.search(user_id, query, k or self.top_k)}

    # ---- write path
    def record_turn(self, user_id: str, user_msg: str, assistant_msg: str, extract: bool = True) -> dict:
        """Persist a finished turn: buffer it, extract facts, compress if the buffer is full."""
        self.short.add(user_id, "user", user_msg)
        self.short.add(user_id, "assistant", assistant_msg)
        stored, skipped = self._store_facts(user_id, user_msg, assistant_msg) if extract else (0, 0)
        tokens_saved = self._maybe_compress(user_id)
        return {"facts_stored": stored, "duplicates_skipped": skipped, "compressed": tokens_saved is not None,
                "tokens_saved": tokens_saved or 0}

    def _store_facts(self, user_id: str, user_msg: str, assistant_msg: str) -> tuple[int, int]:
        if not self.extractor:
            return 0, 0
        if not looks_personal(user_msg):  # no LLM call for "What is DNS?"
            self.stats["extractions_skipped"] += 1
            return 0, 0
        facts, cost = self.extractor.extract(user_msg, assistant_msg)
        self.stats["maintenance_cost_usd"] += cost
        stored = skipped = 0
        for f in facts:
            if self._is_duplicate(user_id, f.text):
                skipped += 1
                continue
            self.long.add(user_id, f.text, kind="fact", importance=f.importance)
            stored += 1
        self.stats["facts_stored"] += stored
        self.stats["facts_skipped_duplicate"] += skipped
        return stored, skipped

    def _is_duplicate(self, user_id: str, text: str) -> bool:
        hits = self.long.search(user_id, text, k=1)
        return bool(hits) and hits[0].similarity >= self.dedup_threshold

    def _maybe_compress(self, user_id: str) -> int | None:
        """Summarise the oldest messages once the buffer overflows. Returns tokens saved or None."""
        if not self.compressor:
            return None
        msgs = self.short.get(user_id)
        if len(msgs) <= self.short.max_messages:
            return None
        n = len(msgs) - self.short.max_messages // 2  # keep the newest half of the window
        old = msgs[:n]
        res, removed = self.compressor.summarize(old)
        self.long.add(user_id, res.text, kind="summary", importance=0.6)
        self.short.trim_oldest(user_id, n)
        saved = removed - estimate_tokens(res.text)
        self.stats["compressions"] += 1
        self.stats["tokens_saved"] += saved
        self.stats["maintenance_cost_usd"] += res.cost
        return saved

    # ---- misc
    def remember(self, user_id: str, text: str, **kw) -> str:
        return self.long.add(user_id, text, **kw)

    def forget_user(self, user_id: str) -> None:
        self.long.delete_all(user_id)
        self.short.clear(user_id)

    def list_memories(self, user_id: str) -> list[Memory]:
        return self.long.list_all(user_id)
