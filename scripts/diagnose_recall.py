"""Why did recall accuracy drop?  Replays ONLY the personal users' queries and explains every recall question.

    python -m scripts.diagnose_recall          (real models; roughly $0.05-0.15)
    python -m scripts.diagnose_recall --mock   (free dry run)

Each miss is labelled with a root cause:
  shortcut returned wrong fact | fact never extracted | fact stored but not recalled | fact was in the prompt but answer missed it
"""
import argparse

from app.config import Settings, get_settings
from app.factory import build_agent
from app.llm import LLMClient, MockLLM, build_llm
from benchmarks.generate_queries import load_or_create


def run(s: Settings, llm: LLMClient, queries: list[dict]) -> list[dict]:
    """Replay `queries` (users are isolated, so a per-user subset behaves exactly as in the full benchmark).

    Returns one row per recall question: user, question, expected, route, model, reply, ok, cause, stored_facts, recalled.
    """
    agent = build_agent(s, llm)
    rows: list[dict] = []
    for q in queries:
        if q["kind"] != "recall":
            agent.chat(q["user_id"], q["message"])
            continue
        uid, exp = q["user_id"], q["expected"].lower()
        ctx = agent.memory.build_context(uid, q["message"], s.memory_top_k)       # what retrieval sees BEFORE the turn
        kept = [m for m in ctx["recalled"] if m.similarity >= s.min_recall_similarity]
        out = agent.chat(uid, q["message"])
        ok = exp in out["reply"].lower()
        route = "memory" if out["answered_from_memory"] else "llm"
        facts = [m for m in agent.memory.list_memories(uid) if m.kind == "fact"]
        cause = ""
        if not ok:
            if route == "memory":
                cause = "shortcut returned wrong fact"
            elif not any(exp in m.text.lower() for m in facts):
                cause = "fact never extracted"
            elif not any(exp in m.text.lower() for m in kept):
                cause = "fact stored but not recalled"
            else:
                cause = "fact was in the prompt but answer missed it"
        rows.append({"user": uid, "question": q["message"], "expected": q["expected"], "route": route, "model": out["model"],
                     "reply": out["reply"], "ok": ok, "cause": cause, "stored_facts": [m.text for m in facts],
                     "recalled": [{"kind": m.kind, "similarity": round(m.similarity, 2), "text": m.text} for m in kept]})
    return rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--queries", default="benchmarks/queries.json")
    a = ap.parse_args(argv)

    s = get_settings().model_copy(update={"max_usd_per_user_per_day": 1e9, "max_usd_global_per_day": 1e9})
    llm = MockLLM() if a.mock else build_llm(s)
    queries = [q for q in load_or_create(a.queries) if q["user_id"].startswith("p")]
    print(f"Replaying {len(queries)} queries for the personal users | MEMORY_ANSWER_THRESHOLD={s.memory_answer_threshold}\n")
    rows = run(s, llm, queries)

    causes: dict[str, int] = {}
    by_route = {"memory": [0, 0], "llm": [0, 0]}          # [correct, total]
    for r in rows:
        by_route[r["route"]][1] += 1
        by_route[r["route"]][0] += r["ok"]
        if r["ok"]:
            continue
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        print(f"MISS [{r['user']}] {r['question']!r}  expected {r['expected']!r}")
        print(f"   cause : {r['cause']}   (route={r['route']}, model={r['model']})")
        print(f"   reply : {r['reply'][:140]!r}")
        print(f"   recalled (sim>=floor): {[(m['kind'], m['similarity'], m['text'][:40]) for m in r['recalled']]}")
        print(f"   stored facts        : {r['stored_facts']}\n")

    ok, total = sum(r["ok"] for r in rows), len(rows)
    print("================ RECALL DIAGNOSIS ================")
    print(f"correct {ok}/{total}  ({100 * ok / total:.0f}%)" if total else "no recall questions found")
    print(f"answered from memory: {by_route['memory'][0]}/{by_route['memory'][1]} correct | via LLM: {by_route['llm'][0]}/{by_route['llm'][1]} correct")
    print(f"causes of misses: {causes or 'none'}")
    return {"correct": ok, "total": total, "causes": causes, "by_route": by_route}


if __name__ == "__main__":
    main()
