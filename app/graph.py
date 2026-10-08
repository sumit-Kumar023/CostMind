"""LangGraph agent: retrieve -> classify -> generate -> (escalate -> generate)* -> finish."""
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.budget import BudgetManager
from app.cache import SemanticCache, is_cacheable
from app.config import Settings
from app.llm import CONFIDENCE_REMINDER, MockLLM, estimate_cost, estimate_tokens
from app.memory.manager import MemoryManager
from app.memory_router import answer_from_memory, is_recall_question
from app.router import ModelRouter, classify_complexity, parse_confidence

SYSTEM = """TASK: CHAT
You are CostMind, a helpful assistant with long-term memory of the user.
{memory}Answer concisely. ALWAYS end with a final line exactly like `CONFIDENCE: 0.85`
(0-1: how sure you are that your answer is correct and complete), even for casual chat."""


class State(TypedDict, total=False):
    user_id: str
    message: str
    context: dict
    k: int
    tier: str
    reason: str
    allow_escalation: bool
    answer: str
    confidence: float | None
    model: str
    escalations: int
    cost: float
    prompt_tokens: int
    completion_tokens: int
    attempts: list
    path: list
    dropped_context: int
    memory_result: dict
    recall_q: bool
    shortcut: str | None
    similarity: float
    saved_cost: float
    saved_tokens: int
    last_prompt_tokens: int
    last_completion_tokens: int


def build_messages(ctx: dict, message: str, cap_tokens: int) -> tuple[list[dict], int]:
    """Assemble the prompt, dropping lowest-ranked memories then oldest turns to fit the token cap."""
    recalled, recent, dropped = list(ctx["recalled"]), list(ctx["recent"]), 0

    def assemble() -> list[dict]:
        mem = "Known about the user:\n" + "\n".join(f"- {m.text}" for m in recalled) + "\n\n" if recalled else ""
        return [{"role": "system", "content": SYSTEM.format(memory=mem)}, *recent, {"role": "user", "content": message + CONFIDENCE_REMINDER}]

    msgs = assemble()
    while estimate_tokens(" ".join(m["content"] for m in msgs)) > cap_tokens and (recalled or recent):
        (recalled.pop() if recalled else recent.pop(0))
        dropped += 1
        msgs = assemble()
    return msgs, dropped


class Agent:
    """Memory-enabled, cost-aware agent."""

    def __init__(self, memory: MemoryManager, router: ModelRouter, budget: BudgetManager, settings: Settings,
                 cache: SemanticCache | None = None, analytics=None) -> None:
        self.memory, self.router, self.budget, self.s, self.cache = memory, router, budget, settings, cache
        self.analytics = analytics
        self.prefer_litellm = not isinstance(router.llm, MockLLM)  # real price map only with real models
        self.stats = {"requests": 0, "cache_hits": 0, "memory_answers": 0, "llm_calls": 0, "escalations": 0,
                      "total_cost_usd": 0.0, "saved_usd_est": 0.0, "saved_prompt_tokens": 0, "unscored": 0}
        self.graph = self._build()

    # ---- nodes
    def _retrieve(self, st: State) -> State:
        k = self.budget.retrieval_k(st["user_id"], self.s.memory_top_k)
        ctx = self.memory.build_context(st["user_id"], st["message"], k)
        ctx["recalled"] = [m for m in ctx["recalled"] if m.similarity >= self.s.min_recall_similarity]
        return {"k": k, "context": ctx, "recall_q": is_recall_question(st["message"])}

    def _shortcut(self, st: State) -> State:
        """Skip the LLM entirely if memory or the semantic cache can answer."""
        if self.s.memory_shortcut_enabled and st["recall_q"]:
            hit = answer_from_memory(st["context"]["recalled"], self.s.memory_answer_threshold)
            if hit:
                return {"shortcut": "memory", "answer": hit[0], "similarity": hit[1], "model": "memory", "tier": "none",
                        "confidence": 1.0, "path": ["memory"], "reason": "recall question answered directly from long-term memory"}
        if self.cache is not None and is_cacheable(st["message"]):
            c = self.cache.lookup(st["user_id"], st["message"])
            if c:
                return {"shortcut": "cache", "answer": c.answer, "similarity": c.similarity, "model": "cache", "tier": "none",
                        "confidence": 1.0, "path": ["cache"], "saved_cost": c.saved_cost, "saved_tokens": c.saved_prompt_tokens,
                        "reason": f"semantic cache hit (sim {c.similarity:.2f}, scope {c.scope})"}
        return {"shortcut": None}

    def _route_shortcut(self, st: State) -> str:
        return "finish_shortcut" if st.get("shortcut") else "classify"

    def _finish_shortcut(self, st: State) -> State:
        # still buffer the turn for continuity, but skip the paid fact-extraction call
        return {"memory_result": self.memory.record_turn(st["user_id"], st["message"], st["answer"], extract=False)}

    def _classify(self, st: State) -> State:
        tier, reason = classify_complexity(st["message"])
        allow = True
        if st["recall_q"] and tier != "cheap" and any(m.kind == "fact" for m in st["context"]["recalled"]):
            tier, reason = "cheap", reason + " | recall question with relevant memory: start cheap"
        if self.budget.is_low(st["user_id"]):
            tier, allow, reason = "cheap", False, reason + " | low budget: capped to cheap tier"
        return {"tier": tier, "reason": reason, "allow_escalation": allow, "path": [tier]}

    def _generate(self, st: State) -> State:
        msgs, dropped = build_messages(st["context"], st["message"], self.s.max_tokens_per_request)
        res, attempts = self.router.call(st["tier"], msgs)
        answer, conf = parse_confidence(res.text)
        return {
            "answer": answer, "confidence": conf, "model": res.model,
            "last_prompt_tokens": res.prompt_tokens, "last_completion_tokens": res.completion_tokens, "dropped_context": dropped,
            "cost": st["cost"] + res.cost, "prompt_tokens": st["prompt_tokens"] + res.prompt_tokens,
            "completion_tokens": st["completion_tokens"] + res.completion_tokens,
            "attempts": st["attempts"] + attempts,
        }

    def _after_generate(self, st: State) -> str:
        if st["confidence"] is None:  # model omitted the score: a formatting lapse, not doubt -> don't pay to escalate
            return "finish"
        if st["confidence"] >= self.s.confidence_threshold:  # early exit
            return "finish"
        if not st["allow_escalation"] or st["cost"] >= self.s.max_usd_per_request:
            return "finish"
        return "escalate" if self.router.next_tier(st["tier"]) else "finish"

    def _escalate(self, st: State) -> State:
        nxt = self.router.next_tier(st["tier"])
        return {"tier": nxt, "escalations": st["escalations"] + 1, "path": st["path"] + [nxt],
                "reason": st["reason"] + f" | low confidence ({st['confidence']:.2f}) -> escalate to {nxt}"}

    def _finish(self, st: State) -> State:
        self.budget.record(st["user_id"], st["cost"])
        if self.cache is not None and is_cacheable(st["message"]) \
                and st["confidence"] is not None and st["confidence"] >= self.s.confidence_threshold:
            private = bool(st["context"]["recalled"])  # answer used personal memory -> keep it private
            self.cache.store(st["user_id"] if private else None, st["message"], st["answer"], st["model"],
                             st["cost"], st["prompt_tokens"])
        return {"memory_result": self.memory.record_turn(st["user_id"], st["message"], st["answer"])}

    def _build(self):
        g = StateGraph(State)
        for name, fn in [("retrieve", self._retrieve), ("shortcut", self._shortcut), ("finish_shortcut", self._finish_shortcut), ("classify", self._classify), ("generate", self._generate),
                         ("escalate", self._escalate), ("finish", self._finish)]:
            g.add_node(name, fn)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "shortcut")
        g.add_conditional_edges("shortcut", self._route_shortcut, {"classify": "classify", "finish_shortcut": "finish_shortcut"})
        g.add_edge("finish_shortcut", END)
        g.add_edge("classify", "generate")
        g.add_conditional_edges("generate", self._after_generate, {"escalate": "escalate", "finish": "finish"})
        g.add_edge("escalate", "generate")
        g.add_edge("finish", END)
        return g.compile()

    # ---- public
    def chat(self, user_id: str, message: str) -> dict:
        """Run one turn. Raises BudgetExceeded / AllModelsFailed."""
        self.budget.check(user_id)
        t0 = time.perf_counter()
        st = self.graph.invoke({
            "user_id": user_id, "message": message, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "escalations": 0, "attempts": [], "path": [], "shortcut": None, "dropped_context": 0,
            "saved_cost": 0.0, "saved_tokens": 0,
        })
        sc = st.get("shortcut")
        if sc is None and st["confidence"] is None:
            st["reason"] += " | model gave no confidence score: accepted without escalation (not cached)"
            self.stats["unscored"] += 1
        self._update_stats(st, sc)
        out = {
            "reply": st["answer"], "model": st["model"], "tier": st["tier"], "route_path": st["path"],
            "route_reason": st["reason"], "confidence": st["confidence"], "escalations": st["escalations"],
            "cost": round(st["cost"], 6), "prompt_tokens": st["prompt_tokens"],
            "completion_tokens": st["completion_tokens"], "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "cache_hit": sc == "cache", "answered_from_memory": sc == "memory",
            "similarity": round(st.get("similarity", 0.0), 3),
            "retrieval_k": st["k"], "retrieved_memories": len(st["context"]["recalled"]), "dropped_context_items": st["dropped_context"],
            "budget_remaining_usd": round(self.budget.remaining(user_id), 6),
            "memory": st["memory_result"], "attempts": st["attempts"],
        }
        if self.analytics is not None:
            self._log(user_id, message, st, sc, out)
        return out

    def _baseline_cost(self, st: State, sc: str | None) -> float:
        """What this request would have cost on a single premium-model call (same final prompt)."""
        prem = self.router.models["premium"]
        if sc == "cache":
            return estimate_cost(prem, st["saved_tokens"], estimate_tokens(st["answer"]), self.prefer_litellm)
        if sc == "memory":
            return self.analytics.avg_llm_baseline()
        return estimate_cost(prem, st["last_prompt_tokens"], st["last_completion_tokens"], self.prefer_litellm)

    def _log(self, user_id: str, message: str, st: State, sc: str | None, out: dict) -> None:
        """Persist one analytics row. The raw message is NOT stored (privacy), only its length."""
        self.analytics.log(
            user_id=user_id, route=sc or "llm", model=out["model"], tier=out["tier"], escalations=out["escalations"],
            confidence=out["confidence"], cost=out["cost"], prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"], latency_ms=out["latency_ms"], retrieved_memories=out["retrieved_memories"],
            route_reason=out["route_reason"], message_chars=len(message), baseline_cost_est=self._baseline_cost(st, sc),
        )

    def _update_stats(self, st: State, shortcut: str | None) -> None:
        S = self.stats
        S["requests"] += 1
        if shortcut == "cache":
            S["cache_hits"] += 1
            S["saved_usd_est"] += st["saved_cost"]
            S["saved_prompt_tokens"] += st["saved_tokens"]
        elif shortcut == "memory":
            S["memory_answers"] += 1
            # estimate: what an average LLM request has cost so far
            S["saved_usd_est"] += S["total_cost_usd"] / S["llm_calls"] if S["llm_calls"] else 0.0
        else:
            S["llm_calls"] += 1
            S["total_cost_usd"] += st["cost"]
            S["escalations"] += st["escalations"]

    def metrics(self) -> dict:
        """Aggregate counters; llm_calls_avoided_pct is your Z metric."""
        S = self.stats
        avoided = S["cache_hits"] + S["memory_answers"]
        return {**S, "llm_calls_avoided_pct": round(100 * avoided / S["requests"], 1) if S["requests"] else 0.0}

    def forget_user(self, user_id: str) -> None:
        """'Forget me': memories, buffer and private cache entries."""
        self.memory.forget_user(user_id)
        if self.cache is not None:
            self.cache.delete_user(user_id)
        if self.analytics is not None:
            self.analytics.delete_user(user_id)
