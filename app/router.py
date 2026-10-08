"""Complexity classification, confidence parsing, and model calls with retry + fallback."""
import logging
import re
import time

from app.llm import LLMClient, LLMResponse

logger = logging.getLogger("costmind.router")

TIERS = ["cheap", "mid", "premium"]
_HARD = re.compile(
    r"\b(analy[sz]e|compare|design|architecture|prove|derive|optimi[sz]e|trade-?offs?|step[- ]by[- ]step|"
    r"debug|refactor|implement|algorithm|evaluate|strategy|why does)\b", re.I)
_MEDIUM = re.compile(r"\b(explain|summari[sz]e|write|draft|how (do|does|to)|difference|list|translate|example)\b", re.I)
_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|function\s*\w*\(|SELECT .* FROM", re.I)
_CONF = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.I)


class AllModelsFailed(Exception):
    """Every candidate model (including fallbacks) failed."""


def classify_complexity(message: str) -> tuple[str, str]:
    """Free, instant heuristic classifier. Returns (tier, human-readable reason)."""
    score, why = 0, []
    words = len(message.split())
    if words > 80:
        score += 2; why.append(f"long ({words} words)")
    elif words > 30:
        score += 1; why.append(f"medium length ({words} words)")
    if _CODE.search(message):
        score += 2; why.append("contains code")
    if hard := {m.group(0).lower() for m in _HARD.finditer(message)}:
        score += min(4, 2 * len(hard)); why.append(f"reasoning keywords {sorted(hard)}")
    if med := {m.group(0).lower() for m in _MEDIUM.finditer(message)}:
        score += min(2, len(med)); why.append(f"generation/explanation task {sorted(med)}")
    if message.count("?") > 1:
        score += 1; why.append("multi-part question")
    tier = "premium" if score >= 4 else "mid" if score >= 2 else "cheap"
    return tier, (", ".join(why) or "short, simple query") + f" -> score {score}"


def parse_confidence(text: str) -> tuple[str, float | None]:
    """Strip the trailing 'CONFIDENCE: x' line. Returns (answer, confidence), or None when the model omitted it.

    A missing score is a formatting lapse, not low confidence, so callers must not treat None as 'unsure'.
    """
    matches = _CONF.findall(text)
    answer = _CONF.sub("", text).strip()
    return answer, (min(1.0, max(0.0, float(matches[-1]))) if matches else None)


class ModelRouter:
    """Maps tiers to models and calls them with one retry each, then fallback models."""

    def __init__(self, llm: LLMClient, models: dict[str, str], fallbacks: list[str] | None = None,
                 retry_delay: float = 0.3) -> None:
        self.llm, self.models, self.fallbacks, self.retry_delay = llm, models, fallbacks or [], retry_delay

    def next_tier(self, tier: str) -> str | None:
        """Next tier that actually uses a different model (skips duplicate tiers)."""
        for t in TIERS[TIERS.index(tier) + 1:]:
            if self.models[t] != self.models[tier]:
                return t
        return None

    def call(self, tier: str, messages: list[dict], max_tokens: int = 600) -> tuple[LLMResponse, list[dict]]:
        """Try the tier's model (2 attempts), then each fallback. Returns (response, attempt log)."""
        candidates = [self.models[tier]] + [m for m in self.fallbacks if m != self.models[tier]]
        log: list[dict] = []
        for model in candidates:
            for attempt in (1, 2):
                try:
                    res = self.llm.complete(messages, model, max_tokens)
                    log.append({"model": model, "attempt": attempt, "ok": True})
                    return res, log
                except Exception as exc:  # noqa: BLE001
                    logger.warning("model %s attempt %d failed: %s", model, attempt, exc)
                    log.append({"model": model, "attempt": attempt, "ok": False, "error": str(exc)[:120]})
                    if attempt == 1 and self.retry_delay:
                        time.sleep(self.retry_delay)
        raise AllModelsFailed(f"All models failed: {[a['model'] for a in log]}")
