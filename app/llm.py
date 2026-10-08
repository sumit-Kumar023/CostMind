"""LLM access layer: LiteLLM in production, a deterministic mock for offline dev/tests."""
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings


# Appended to the last user turn of CHAT prompts: models obey a format instruction far more reliably there than in the system prompt.
CONFIDENCE_REMINDER = "\n\n[Reply normally, then end with a final line: CONFIDENCE: <number 0-1>]"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used for savings metrics."""
    return max(1, len(text) // 4)


@dataclass
class LLMResponse:
    """Normalised completion result including cost, for analytics."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    model: str


class LLMClient(Protocol):
    def complete(self, messages: list[dict], model: str, max_tokens: int = 300) -> LLMResponse: ...


class LiteLLMClient:
    """Thin wrapper over LiteLLM (imported lazily to keep startup RAM low)."""

    def complete(self, messages: list[dict], model: str, max_tokens: int = 300) -> LLMResponse:
        import litellm

        r = litellm.completion(model=model, messages=messages, max_tokens=max_tokens, temperature=0)
        try:
            cost = float(litellm.completion_cost(completion_response=r))
        except Exception:  # noqa: BLE001
            cost = 0.0
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, cost, model)


_PRICES = {"gpt-4o-mini": (0.00015, 0.0006), "gpt-4o": (0.0025, 0.01)}  # USD per 1k tokens (in, out)


def _is_premium(model: str) -> bool:
    return model == "gpt-4o" or any(x in model for x in ("prem", "opus"))


def _price(model: str) -> tuple[float, float]:
    return _PRICES.get(model) or ((0.0025, 0.01) if _is_premium(model) else (0.00015, 0.0006))


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, prefer_litellm: bool = False) -> float:
    """Estimate USD cost of a call. Uses LiteLLM's price map when asked, else the built-in table."""
    if prefer_litellm:
        try:
            import litellm

            a, b = litellm.cost_per_token(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            return float(a + b)
        except Exception:  # noqa: BLE001
            pass
    p_in, p_out = _price(model)
    return prompt_tokens / 1000 * p_in + completion_tokens / 1000 * p_out


class MockLLM:
    """Deterministic fake LLM with realistic pricing. Detects the task from the system prompt tag."""

    def complete(self, messages: list[dict], model: str = "mock", max_tokens: int = 300) -> LLMResponse:
        system, user = messages[0]["content"], messages[-1]["content"]
        if "TASK: EXTRACT_FACTS" in system:
            text = json.dumps(self._extract(user))
        elif "TASK: CHAT" in system:
            user = user.split(CONFIDENCE_REMINDER)[0]
            hard = len(user) > 150 or "tricky" in user.lower()
            conf = 0.95 if _is_premium(model) else (0.4 if hard else 0.9)
            text = f"[{model}] Echo: {user[:80]}\nCONFIDENCE: {conf}"
        else:  # SUMMARIZE
            text = "Summary: " + " ".join(user.split())[:120]
        pt = estimate_tokens(" ".join(m["content"] for m in messages))
        ct = estimate_tokens(text)
        p_in, p_out = _price(model)
        return LLMResponse(text, pt, ct, pt / 1000 * p_in + ct / 1000 * p_out, model)

    @staticmethod
    def _extract(user: str) -> list[dict]:
        line = next((ln[5:] for ln in user.splitlines() if ln.startswith("USER:")), user)
        out = []
        if m := re.search(r"my name is (\w+)", line, re.I):
            out.append({"fact": f"User's name is {m.group(1)}", "importance": 0.9})
        if m := re.search(r"allergic to ([^.,!?]+)", line, re.I):
            out.append({"fact": f"User is allergic to {m.group(1).strip()}", "importance": 0.95})
        if m := re.search(r"\bi (love|like|hate|enjoy) ([^.,!?]+)", line, re.I):
            out.append({"fact": f"User {m.group(1).lower()}s {m.group(2).strip()}", "importance": 0.7})
        return out


def build_llm(s: Settings) -> LLMClient:
    """Use LiteLLM when a provider key exists, else the offline mock."""
    has_key = bool(s.openai_api_key or s.anthropic_api_key)
    if s.llm_provider == "mock" or (s.llm_provider == "auto" and not has_key):
        return MockLLM()
    if s.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", s.openai_api_key)
    if s.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", s.anthropic_api_key)
    return LiteLLMClient()
