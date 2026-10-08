"""Memory maintenance done by the CHEAP model: summarisation and fact extraction."""
import json
import re
from dataclasses import dataclass

from app.llm import LLMClient, LLMResponse, estimate_tokens

SUMMARIZE_SYS = (
    "TASK: SUMMARIZE\nCompress the conversation into 2-4 short sentences. Keep names, preferences, "
    "decisions, numbers and open questions. Drop pleasantries."
)
EXTRACT_SYS = (
    "TASK: EXTRACT_FACTS\nExtract durable facts about the USER worth remembering long-term "
    "(identity, preferences, goals, constraints, allergies). Reply with ONLY a JSON list like "
    '[{"fact": "User likes cricket", "importance": 0.7}] with importance 0-1. Reply [] if none.'
)


_PERSONAL = re.compile(r"\b(my|me|i|i'm|i've|mine|myself|we|our)\b", re.I)


def looks_personal(text: str) -> bool:
    """Cheap gate: only messages that talk about the user can contain user facts worth extracting."""
    return bool(_PERSONAL.search(text))


@dataclass
class Fact:
    text: str
    importance: float


def parse_facts(raw: str) -> list[Fact]:
    """Robustly parse the model's JSON (handles code fences and garbage)."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        data = json.loads(raw)
        return [
            Fact(str(d["fact"]).strip(), min(1.0, max(0.0, float(d.get("importance", 0.5)))))
            for d in data
            if isinstance(d, dict) and d.get("fact")
        ]
    except (ValueError, TypeError, KeyError):
        return []


class ContextCompressor:
    """Summarise old turns so the prompt stays small."""

    def __init__(self, llm: LLMClient, model: str) -> None:
        self.llm, self.model = llm, model

    def summarize(self, messages: list[dict]) -> tuple[LLMResponse, int]:
        """Return (summary response, tokens removed from the prompt)."""
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        res = self.llm.complete(
            [{"role": "system", "content": SUMMARIZE_SYS}, {"role": "user", "content": convo}], self.model
        )
        return res, estimate_tokens(convo)


class FactExtractor:
    """Pull durable user facts out of each turn."""

    def __init__(self, llm: LLMClient, model: str) -> None:
        self.llm, self.model = llm, model

    def extract(self, user_msg: str, assistant_msg: str) -> tuple[list[Fact], float]:
        content = f"USER: {user_msg}\nASSISTANT: {assistant_msg}"
        res = self.llm.complete(
            [{"role": "system", "content": EXTRACT_SYS}, {"role": "user", "content": content}], self.model
        )
        return parse_facts(res.text), res.cost
