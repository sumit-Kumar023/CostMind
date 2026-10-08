import json

import pytest

from benchmarks.generate_queries import generate
from benchmarks.run_benchmark import RunResult, compute_metrics, judge_pairs, main


def test_dataset_shape_and_determinism():
    d = generate(320)
    assert len(d) >= 320 and generate(320) == d
    assert len({q["id"] for q in d}) == len(d)
    generic = [q for q in d if q["kind"] in ("fresh", "dup")]
    dup_share = sum(q["kind"] == "dup" for q in generic) / len(generic)
    assert 0.2 <= dup_share <= 0.3
    fresh = [q for q in generic if q["kind"] == "fresh"]
    share = lambda c: sum(q["complexity"] == c for q in fresh) / len(fresh)  # noqa: E731
    assert 0.55 <= share("simple") <= 0.65 and 0.25 <= share("medium") <= 0.35 and 0.07 <= share("complex") <= 0.13


def test_recalls_come_after_their_facts_and_have_answers():
    d = generate(320)
    for q in (x for x in d if x["kind"] == "recall"):
        facts = [x for x in d if x["kind"] == "fact" and x["user_id"] == q["user_id"]]
        assert len(facts) == 3 and all(f["id"] < q["id"] for f in facts) and q["expected"]


def test_duplicates_follow_an_original():
    d = generate(320)
    seen = set()
    for q in d:
        if q["kind"] == "dup":
            assert any(q["message"].lower().strip(" ?.!") .replace("please ", "").replace("hey, ", "").replace(" thanks", "")
                       .startswith(s[:12]) for s in seen)
        elif q["kind"] == "fresh":
            seen.add(q["message"].lower())


def test_metric_formulas():
    base = RunResult(n=100, cost=4.20, prompt_tokens=600_000)
    cm = RunResult(n=100, cost=1.89, prompt_tokens=270_000, avoided=23)
    m = compute_metrics(base, cm)
    assert m == {"X_cost_reduction_pct": 55.0, "Y_prompt_token_reduction_pct": 55.0, "Z_llm_calls_avoided_pct": 23.0}
    assert compute_metrics(RunResult(), RunResult())["X_cost_reduction_pct"] == 0.0


class FakeJudge:
    """Scores 5 for answers containing GOOD else 3; one call returns garbage."""

    def __init__(self):
        self.n = 0

    def complete(self, messages, model, max_tokens=60):
        from app.llm import LLMResponse
        self.n += 1
        txt = messages[-1]["content"]
        a = txt.split("Answer A:")[1].split("Answer B:")[0]
        b = txt.split("Answer B:")[1]
        out = "not json" if self.n == 1 else json.dumps({"a": 5 if "GOOD" in a else 3, "b": 5 if "GOOD" in b else 3})
        return LLMResponse(out, 1, 1, 0.0, model)


def test_judge_unscrambles_position_bias_and_counts_failures():
    items = [(f"q{i}", "GOOD baseline", "meh costmind") for i in range(21)]
    r = judge_pairs(FakeJudge(), "j", items)
    assert r["failed"] == 1 and r["judged"] == 20
    assert r["baseline_mean"] == 5.0 and r["costmind_mean"] == 3.0 and r["quality_retained_pct"] == 60.0
    assert r["costmind_at_least_as_good_pct"] == 0.0


def test_end_to_end_mock_run_writes_reports(tmp_path):
    out = tmp_path / "r.json"
    res = main(["--mock", "--limit", "150", "--out", str(out), "--yes", "--judge"])
    assert res["mock"] is True and res["n_queries"] == 150
    m = res["metrics"]
    assert m["X_cost_reduction_pct"] > 0 and m["Z_llm_calls_avoided_pct"] > 0
    assert set(res["costmind"]["by_route"]) <= {"llm", "cache", "memory"}
    assert json.loads(out.read_text())["mock"] is True
    assert "NOT valid" in out.with_suffix(".md").read_text()
