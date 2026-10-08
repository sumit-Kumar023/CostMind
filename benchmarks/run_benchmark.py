"""Benchmark: ALWAYS-PREMIUM baseline (full per-user history) vs CostMind. Produces X, Y, Z.

  python -m benchmarks.run_benchmark --mock --limit 100        # free dry run (numbers NOT valid)
  python -m benchmarks.run_benchmark --judge                   # real run (needs API key; costs a few dollars)
  python -m benchmarks.run_benchmark --sweep --sweep-conf 0.6,0.75,0.9 --sweep-cache 0.9,0.95
"""
import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.factory import build_agent
from app.llm import LLMClient, LLMResponse, MockLLM, build_llm
from app.memory.embeddings import CachedEmbedder, OpenAIEmbedder, build_embedder
from app.llm import estimate_tokens
from app.router import parse_confidence
from benchmarks.generate_queries import load_or_create

BASE_SYS = "TASK: CHAT\nYou are a helpful assistant. Answer concisely."  # same brevity instruction CostMind gets
EMBED_USD_PER_TOKEN = 0.02 / 1_000_000                                    # text-embedding-3-small


@dataclass
class RunResult:
    """Totals for one system over the whole query list."""

    n: int = 0
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    avoided: int = 0
    latency_ms_total: float = 0.0
    recall_total: int = 0
    recall_correct: int = 0
    answers: list = field(default_factory=list)
    routes: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    route_n: dict = field(default_factory=dict)       # route -> request count
    lat_by_route: dict = field(default_factory=dict)  # route -> total ms
    stage_ms: dict = field(default_factory=dict)      # route -> stage -> total ms

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.n if self.n else 0.0

    @property
    def recall_acc(self) -> float:
        return 100 * self.recall_correct / self.recall_total if self.recall_total else 0.0


def pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def compute_metrics(base: RunResult, cm: RunResult) -> dict:
    """X = cost reduction %, Y = prompt-token reduction %, Z = LLM calls avoided %."""
    return {
        "X_cost_reduction_pct": pct(base.cost - cm.cost, base.cost),
        "Y_prompt_token_reduction_pct": pct(base.prompt_tokens - cm.prompt_tokens, base.prompt_tokens),
        "Z_llm_calls_avoided_pct": pct(cm.avoided, cm.n),
    }


# ---------------------------------------------------------------- instrumentation
class TallyLLM:
    """Wraps any LLMClient and records EVERY call (chat, escalations, memory upkeep) - the source of truth for CostMind totals."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner, self.calls = inner, []

    def complete(self, messages: list[dict], model: str, max_tokens: int = 300) -> LLMResponse:
        res = self.inner.complete(messages, model, max_tokens)
        kind = "chat" if "TASK: CHAT" in messages[0]["content"] else "maintenance"
        self.calls.append((kind, res.prompt_tokens, res.completion_tokens, res.cost))
        return res

    def total(self, idx: int, kind: str | None = None) -> float:
        return sum(c[idx] for c in self.calls if kind in (None, c[0]))


class CountingEmbedder:
    """Counts tokens actually sent to the embedding provider (cache misses only)."""

    def __init__(self, inner) -> None:
        self.inner, self.dim, self.tokens = inner, inner.dim, 0

    def embed(self, text: str) -> list[float]:
        self.tokens += estimate_tokens(text)
        return self.inner.embed(text)


def _retry(fn, mock: bool, tries: int = 3):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(0 if mock else 2 * (i + 1))


def correct(q: dict, answer: str) -> bool:
    return bool(q.get("expected")) and q["expected"].lower() in answer.lower()


# ---------------------------------------------------------------- the two systems
def run_baseline(llm: LLMClient, premium: str, queries: list[dict], mock: bool, log=print) -> RunResult:
    """Always the premium model, with the user's full conversation history in every prompt."""
    r, hist = RunResult(), {}
    for i, q in enumerate(queries, 1):
        h = hist.setdefault(q["user_id"], [])
        msgs = [{"role": "system", "content": BASE_SYS}, *h, {"role": "user", "content": q["message"]}]
        t0 = time.perf_counter()
        res = _retry(lambda: llm.complete(msgs, premium, 600), mock)
        r.latency_ms_total += (time.perf_counter() - t0) * 1000
        ans, _ = parse_confidence(res.text)
        h += [{"role": "user", "content": q["message"]}, {"role": "assistant", "content": ans}]
        r.n += 1; r.cost += res.cost; r.prompt_tokens += res.prompt_tokens; r.completion_tokens += res.completion_tokens
        r.answers.append(ans); r.routes.append("llm")
        if q["kind"] == "recall":
            r.recall_total += 1; r.recall_correct += correct(q, ans)
        if i % 25 == 0:
            log(f"  baseline {i}/{len(queries)}  ${r.cost:.4f}")
    return r


def run_costmind(settings: Settings, base_llm: LLMClient, queries: list[dict], label: str = "costmind", log=print) -> RunResult:
    """CostMind end-to-end on a fresh, isolated agent. Totals come from the call tally (not the agent's own bookkeeping)."""
    tally = TallyLLM(base_llm)
    counting = CountingEmbedder(build_embedder(settings).inner)
    agent = build_agent(settings, tally, CachedEmbedder(counting))
    r = RunResult()
    for i, q in enumerate(queries, 1):
        out = agent.chat(q["user_id"], q["message"])
        r.n += 1; r.latency_ms_total += out["latency_ms"]
        r.answers.append(out["reply"])
        route = "cache" if out["cache_hit"] else "memory" if out["answered_from_memory"] else "llm"
        r.routes.append(route)
        r.route_n[route] = r.route_n.get(route, 0) + 1
        r.lat_by_route[route] = r.lat_by_route.get(route, 0.0) + out["latency_ms"]
        for stage, ms in out["timings"].items():
            r.stage_ms.setdefault(route, {}).setdefault(stage, 0.0)
            r.stage_ms[route][stage] += ms
        if q["kind"] == "recall":
            r.recall_total += 1; r.recall_correct += correct(q, out["reply"])
        if i % 25 == 0:
            log(f"  {label} {i}/{len(queries)}  ${tally.total(3):.4f}")
    embed_cost = counting.tokens * EMBED_USD_PER_TOKEN if isinstance(counting.inner, OpenAIEmbedder) else 0.0
    r.cost = tally.total(3) + embed_cost
    r.prompt_tokens, r.completion_tokens = int(tally.total(1)), int(tally.total(2))
    r.avoided = sum(1 for x in r.routes if x != "llm")
    r.breakdown = {"chat_cost_usd": round(tally.total(3, "chat"), 6), "maintenance_cost_usd": round(tally.total(3, "maintenance"), 6),
                   "embedding_cost_usd": round(embed_cost, 6), "chat_prompt_tokens": int(tally.total(1, "chat")),
                   "maintenance_prompt_tokens": int(tally.total(1, "maintenance")),
                   "extractions_skipped": agent.memory.stats["extractions_skipped"], "compressions": agent.memory.stats["compressions"]}
    r.summary = agent.analytics.summary(agent.memory.stats["maintenance_cost_usd"])
    return r


# ---------------------------------------------------------------- quality: LLM judge
JUDGE_SYS = ("TASK: JUDGE\nYou grade two answers to the same question for correctness and helpfulness. "
             'Reply with ONLY JSON like {"a": 4, "b": 5} using integers 1-5 for Answer A and Answer B.')


def judge_pairs(llm: LLMClient, model: str, items: list[tuple[str, str, str]], seed: int = 0) -> dict:
    """items = (question, baseline_answer, costmind_answer). Order is randomised per pair to avoid position bias."""
    rng, base_s, cm_s, failed, per_item = random.Random(seed), [], [], 0, []
    for idx, (q, b, c) in enumerate(items):
        swap = rng.random() < 0.5
        a_txt, b_txt = (c, b) if swap else (b, c)
        res = llm.complete([{"role": "system", "content": JUDGE_SYS},
                            {"role": "user", "content": f"Question: {q}\n\nAnswer A: {a_txt}\n\nAnswer B: {b_txt}"}], model, 60)
        m = re.search(r"\{.*?\}", res.text, re.S)
        try:
            d = json.loads(m.group(0))
            sa, sb = int(d["a"]), int(d["b"])
            if not (1 <= sa <= 5 and 1 <= sb <= 5):
                raise ValueError
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        sb_base, sb_cm = (sb, sa) if swap else (sa, sb)
        base_s.append(sb_base); cm_s.append(sb_cm); per_item.append({"idx": idx, "ref": sb_base, "test": sb_cm})
    n = len(base_s)
    mb, mc = (sum(base_s) / n, sum(cm_s) / n) if n else (0.0, 0.0)
    return {"judged": n, "failed": failed, "baseline_mean": round(mb, 2), "costmind_mean": round(mc, 2),
            "quality_retained_pct": pct(mc, mb), "costmind_at_least_as_good_pct": pct(sum(c >= b for b, c in zip(base_s, cm_s)), n), "per_item": per_item}


def by_complexity(per_item: list[dict], picks: list[int], queries: list[dict]) -> dict:
    """Mean judge score of reference vs tested system, bucketed by the query's complexity label."""
    buckets: dict[str, list[dict]] = {}
    for it in per_item:
        buckets.setdefault(queries[picks[it["idx"]]]["complexity"], []).append(it)
    return {c: {"n": len(v), "reference_mean": round(sum(x["ref"] for x in v) / len(v), 2),
                "tested_mean": round(sum(x["test"] for x in v) / len(v), 2)} for c, v in sorted(buckets.items())}


# ---------------------------------------------------------------- reusable baselines (the premium baseline is ~80% of the bill)
def dataset_key(queries: list[dict], model: str, mock: bool) -> str:
    h = hashlib.sha256(f"{model}|{mock}|".encode())
    for q in queries:
        h.update(f"{q['user_id']}|{q['message']}\n".encode())
    return h.hexdigest()[:16]


def save_baseline(path: str, key: str, r: RunResult) -> None:
    Path(path).write_text(json.dumps({"key": key, "saved_at": time.time(), "n": r.n, "cost": r.cost, "prompt_tokens": r.prompt_tokens,
                                      "completion_tokens": r.completion_tokens, "latency_ms_total": r.latency_ms_total,
                                      "recall_total": r.recall_total, "recall_correct": r.recall_correct, "answers": r.answers}))


def load_baseline(path: str, key: str) -> "RunResult | None":
    """Return the cached baseline only if it was produced from exactly this dataset + model."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("key") != key:
        return None
    return RunResult(n=d["n"], cost=d["cost"], prompt_tokens=d["prompt_tokens"], completion_tokens=d["completion_tokens"],
                     latency_ms_total=d["latency_ms_total"], recall_total=d["recall_total"], recall_correct=d["recall_correct"],
                     answers=d["answers"], routes=["llm"] * d["n"])


def baseline_from_results(path: str, queries: list[dict], premium: str) -> "RunResult | None":
    """Seed the baseline from a previous (real) results.json so you don't pay for it twice. No answers -> judge unavailable."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("mock") or d.get("n_queries") != len(queries) or d["models"]["premium"] != premium:
        return None
    b, n = d["baseline"], len(queries)
    recall_total = sum(q["kind"] == "recall" for q in queries)
    return RunResult(n=n, cost=b["cost_usd"], prompt_tokens=b["prompt_tokens"], completion_tokens=b["completion_tokens"],
                     latency_ms_total=b["avg_latency_ms"] * n, recall_total=recall_total,
                     recall_correct=round(b["recall_acc_pct"] / 100 * recall_total), answers=[], routes=["llm"] * n)


# ---------------------------------------------------------------- reporting
def write_reports(res: dict, out: Path) -> None:
    out.write_text(json.dumps(res, indent=1))
    m, b, c, q = res["metrics"], res["baseline"], res["costmind"], res.get("quality")
    md = ["# Benchmark results", ""]
    if res["mock"]:
        md += ["> **MOCK RUN - synthetic responses. These numbers are NOT valid; do not use them on a resume.**", ""]
    md += [f"{res['n_queries']} queries - baseline: always `{res['models']['premium']}` with full per-user history - "
           f"CostMind: cheap=`{res['models']['cheap']}`, premium=`{res['models']['premium']}`", "",
           "| Metric | Baseline | CostMind | Result |", "|---|---|---|---|",
           f"| Total cost | ${b['cost_usd']:.4f} | ${c['cost_usd']:.4f} | **X = {m['X_cost_reduction_pct']}% cheaper** |",
           f"| Prompt tokens | {b['prompt_tokens']:,} | {c['prompt_tokens']:,} | **Y = {m['Y_prompt_token_reduction_pct']}%** |",
           f"| LLM calls avoided | 0% | {m['Z_llm_calls_avoided_pct']}% | **Z = {m['Z_llm_calls_avoided_pct']}%** |",
           f"| Avg latency | {b['avg_latency_ms']:.0f} ms | {c['avg_latency_ms']:.0f} ms | |",
           f"| Recall accuracy | {b['recall_acc_pct']:.0f}% | {c['recall_acc_pct']:.0f}% | |"]
    if q:
        md.append(f"| Judge score (1-5) | {q['baseline_mean']} | {q['costmind_mean']} | {q['quality_retained_pct']}% quality retained |")
    ch = res.get("cheap_baseline")
    if ch:
        md.append(f"| *Always-cheap baseline* | cost ${ch['cost_usd']:.4f} | prompt tokens {ch['prompt_tokens']:,} | avg latency {ch['avg_latency_ms']:.0f} ms |")
    md += ["", f"CostMind cost breakdown: {c['breakdown']}", "", f"Latency by route (ms): {c.get('latency_by_route_ms')}",
           f"Stage latency by route (avg ms): {c.get('stage_ms_avg_by_route')}", ""]
    if q and q.get("by_complexity"):
        md += ["## Judge score by query complexity (premium baseline = reference)", "", "| Complexity | n | Premium | CostMind |" + (" Always-cheap |" if ch and ch.get("quality") else ""),
               "|---|---|---|---|" + ("---|" if ch and ch.get("quality") else "")]
        for cx, v in q["by_complexity"].items():
            row = f"| {cx} | {v['n']} | {v['reference_mean']} | {v['tested_mean']} |"
            if ch and ch.get("quality"):
                row += f" {ch['quality']['by_complexity'].get(cx, {}).get('tested_mean', '-')} |"
            md.append(row)
        md.append("")
    if res.get("sweep"):
        md += ["## Threshold sweep", "", "| confidence thr | cache sim thr | X | Y | Z | recall acc | escalation rate |", "|---|---|---|---|---|---|---|"]
        md += [f"| {s['confidence_threshold']} | {s['cache_similarity_threshold']} | {s['X_cost_reduction_pct']}% | "
               f"{s['Y_prompt_token_reduction_pct']}% | {s['Z_llm_calls_avoided_pct']}% | {s['recall_acc_pct']:.0f}% | {s['escalation_rate_pct']}% |"
               for s in res["sweep"]]
    out.with_suffix(".md").write_text("\n".join(md))


def _side(r: RunResult) -> dict:
    return {"cost_usd": round(r.cost, 6), "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
            "avg_latency_ms": round(r.avg_latency_ms, 1), "recall_acc_pct": round(r.recall_acc, 1), "breakdown": r.breakdown}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="benchmarks/queries.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="use the offline mock LLM (free; numbers not valid)")
    ap.add_argument("--judge", action="store_true"); ap.add_argument("--judge-n", type=int, default=60)
    ap.add_argument("--judge-model", default="")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-conf", default="0.6,0.75,0.9"); ap.add_argument("--sweep-cache", default="0.9,0.95")
    ap.add_argument("--out", default="benchmarks/results.json")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--baseline-cache", default="benchmarks/baseline_cache.json", help="reuse the premium baseline across runs")
    ap.add_argument("--fresh-baseline", action="store_true", help="ignore the cache and re-run the premium baseline")
    ap.add_argument("--baseline-from-results", default="", help="seed the baseline from an earlier results.json (judge then unavailable)")
    ap.add_argument("--cheap-baseline", action="store_true", help="also run an always-cheap (full history) baseline")
    a = ap.parse_args(argv)

    s = get_settings().model_copy(update={"max_usd_per_user_per_day": 1e9, "max_usd_global_per_day": 1e9})  # don't let budgets throttle the benchmark
    llm = MockLLM() if a.mock else build_llm(s)
    mock = isinstance(llm, MockLLM)
    queries = load_or_create(a.queries)
    queries = queries[: a.limit] if a.limit else queries
    print(f"\n=== CostMind benchmark: {len(queries)} queries | cheap={s.cheap_model} premium={s.premium_model} | {'MOCK' if mock else 'REAL'} ===")
    if mock:
        print("!! MOCK MODE: synthetic responses. Numbers below are NOT valid for a resume. Add an API key for real results.\n")
    elif not a.yes:
        print(f"Real run. Rough cost: ${0.004 * len(queries):.2f}-${0.010 * len(queries):.2f} (baseline dominates), ~10-15 min.")
        if input("Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("aborted")

    key = dataset_key(queries, s.premium_model, mock)
    base = None
    if a.baseline_from_results:
        base = baseline_from_results(a.baseline_from_results, queries, s.premium_model)
        if base is None:
            raise SystemExit(f"{a.baseline_from_results} doesn't match this run (needs a real run with the same query count and premium model).")
        print(f"Seeded baseline from {a.baseline_from_results} (no premium calls; the judge needs cached answers, so it is skipped).")
    elif not a.fresh_baseline:
        base = load_baseline(a.baseline_cache, key)
        if base:
            print(f"Reusing cached premium baseline from {a.baseline_cache} (use --fresh-baseline to redo it).")
    if base is None:
        print("Running baseline (always premium, full history)...")
        base = run_baseline(llm, s.premium_model, queries, mock)
        save_baseline(a.baseline_cache, key, base)
    print("Running CostMind...")
    cm = run_costmind(s, llm, queries)
    metrics = compute_metrics(base, cm)

    res = {"mock": mock, "n_queries": len(queries), "models": {"cheap": s.cheap_model, "mid": s.mid_model, "premium": s.premium_model},
           "config": {"confidence_threshold": s.confidence_threshold, "cache_similarity_threshold": s.cache_similarity_threshold,
                      "memory_answer_threshold": s.memory_answer_threshold, "short_term_window": s.short_term_window},
           "metrics": metrics, "baseline": _side(base), "costmind": {**_side(cm), "by_route": cm.summary["by_route"],
           "by_model": cm.summary["by_model"], "escalation_rate_pct": cm.summary["escalation_rate_pct"],
           "latency_by_route_ms": {k: round(v / cm.route_n[k], 1) for k, v in cm.lat_by_route.items()},
           "stage_ms_avg_by_route": {rt: {st: round(ms / cm.route_n[rt], 1) for st, ms in stages.items()} for rt, stages in cm.stage_ms.items()}},
           "per_query": [{"id": q["id"], "kind": q["kind"], "complexity": q["complexity"], "route": r,
                          **({"expected": q["expected"], "reply": a_[:240], "correct": correct(q, a_)} if q["kind"] == "recall" else {})}
                         for q, r, a_ in zip(queries, cm.routes, cm.answers)]}

    can_judge = a.judge and not mock and len(base.answers) == len(queries)
    if a.judge and not can_judge:
        print("(judge skipped: mock mode, or the baseline came from results.json and has no stored answers. Use --fresh-baseline once.)")
    cheap = None
    if a.cheap_baseline:
        print("Running always-cheap baseline (full history)...")
        cheap = run_baseline(llm, s.cheap_model, queries, mock)
        res["cheap_baseline"] = _side(cheap)
    if can_judge:
        rng = random.Random(1)
        eligible = [i for i, q in enumerate(queries) if q["kind"] in ("fresh", "dup") and cm.routes[i] != "memory"]
        hard = [i for i in eligible if queries[i]["complexity"] == "complex"]
        rest = [i for i in eligible if queries[i]["complexity"] != "complex"]
        picks = sorted(set(rng.sample(rest, min(a.judge_n, len(rest))) + hard[:30]))  # always include the hard queries
        jm = a.judge_model or s.premium_model
        q_cm = judge_pairs(llm, jm, [(queries[i]["message"], base.answers[i], cm.answers[i]) for i in picks])
        q_cm["by_complexity"] = by_complexity(q_cm.pop("per_item"), picks, queries)
        res["quality"] = q_cm
        if cheap is not None:
            q_ch = judge_pairs(llm, jm, [(queries[i]["message"], base.answers[i], cheap.answers[i]) for i in picks])
            q_ch["by_complexity"] = by_complexity(q_ch.pop("per_item"), picks, queries)
            res["cheap_baseline"]["quality"] = q_ch

    if a.sweep:
        res["sweep"] = []
        for conf in [float(x) for x in a.sweep_conf.split(",")]:
            for thr in [float(x) for x in a.sweep_cache.split(",")]:
                print(f"Sweep: confidence>={conf}, cache sim>={thr}")
                r = run_costmind(s.model_copy(update={"confidence_threshold": conf, "cache_similarity_threshold": thr}), llm, queries, "sweep")
                res["sweep"].append({"confidence_threshold": conf, "cache_similarity_threshold": thr, **compute_metrics(base, r),
                                     "recall_acc_pct": r.recall_acc, "escalation_rate_pct": r.summary["escalation_rate_pct"]})

    write_reports(res, Path(a.out))
    X, Y, Z = metrics["X_cost_reduction_pct"], metrics["Y_prompt_token_reduction_pct"], metrics["Z_llm_calls_avoided_pct"]
    print("\n================ RESULTS ================")
    print(f"X  cost reduction         : {X}%   (${base.cost:.4f} -> ${cm.cost:.4f})")
    print(f"Y  prompt-token reduction : {Y}%   ({base.prompt_tokens:,} -> {cm.prompt_tokens:,}, incl. memory upkeep)")
    print(f"Z  LLM calls avoided      : {Z}%   ({cm.avoided} of {cm.n}; routes {cm.summary['by_route']})")
    print(f"Latency avg               : {base.avg_latency_ms:.0f} ms -> {cm.avg_latency_ms:.0f} ms")
    print(f"Recall accuracy           : {base.recall_acc:.0f}% -> {cm.recall_acc:.0f}%")
    if res.get("quality"):
        print(f"Judge quality retained    : {res['quality']['quality_retained_pct']}%  ({res['quality']})")
    print(f"Latency by route (ms)     : { {k: round(v / cm.route_n[k]) for k, v in cm.lat_by_route.items()} }")
    for rt, stages in cm.stage_ms.items():
        print(f"  stages [{rt:6}] avg ms   : { {st: round(ms / cm.route_n[rt]) for st, ms in stages.items()} }")
    if cheap is not None:
        print(f"Always-cheap baseline     : ${cheap.cost:.4f}, {cheap.avg_latency_ms:.0f} ms avg, recall {cheap.recall_acc:.0f}%")
    if res.get("quality", {}).get("by_complexity"):
        print(f"Judge by complexity       : {res['quality']['by_complexity']}")
        if cheap is not None and res["cheap_baseline"].get("quality"):
            print(f"  always-cheap vs premium : {res['cheap_baseline']['quality']['by_complexity']}")
    print(f"CostMind breakdown        : {cm.breakdown}")
    if not mock:
        print("\nResume bullet:")
        if Y > 0:
            print(f"  Cut LLM inference cost {X:.0f}% and prompt tokens {Y:.0f}% vs an always-premium, full-history baseline over {cm.n} queries, "
                  f"avoiding {Z:.0f}% of LLM calls via semantic caching and memory-aware routing.")
        else:
            print(f"  Cut LLM inference cost {X:.0f}% vs an always-premium baseline over {cm.n} queries, avoiding {Z:.0f}% of LLM calls "
                  "via semantic caching and memory-aware routing.   (Y was not positive on this dataset - don't claim it.)")
    print(f"\nWrote {a.out} and {Path(a.out).with_suffix('.md')}")
    return res


if __name__ == "__main__":
    main()
