"""Pick MEMORY_ANSWER_THRESHOLD from YOUR embedding model.   python -m scripts.tune_memory_threshold

Compares recall questions against (a) the fact that answers them and (b) facts/questions that must NOT match.
Uses a few dozen embedding calls (well under $0.01). Needs OPENAI_API_KEY (offline it only demos the idea).
"""
import math

from app.config import get_settings
from app.memory.embeddings import build_embedder

FACTS = {"name": "User's name is {n}", "love": "User loves {h}", "allergy": "User is allergic to {a}"}
QUESTIONS = {"name": "What is my name?", "love": "What do I love?", "allergy": "What am I allergic to?"}
PEOPLE = [("Rahul", "cricket", "peanuts"), ("Priya", "chess", "shellfish"), ("Amit", "painting", "pollen"),
          ("Neha", "hiking", "dust"), ("Karan", "cooking", "penicillin"), ("Sneha", "photography", "eggs")]
UNRELATED = ["What is the capital of France?", "What is 12 times 13?", "Who wrote Hamlet?", "What is DNS?"]


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def analyze(embed) -> dict:
    """Return similarity samples for matching pairs vs pairs that must not match, plus a suggested threshold."""
    good, bad = [], []
    for n, h, a in PEOPLE:
        facts = {"name": FACTS["name"].format(n=n), "love": FACTS["love"].format(h=h), "allergy": FACTS["allergy"].format(a=a)}
        for k, q in QUESTIONS.items():
            for fk, ft in facts.items():
                (good if fk == k else bad).append(cosine(embed(q), embed(ft)))
        bad += [cosine(embed(q), embed(ft)) for q in UNRELATED for ft in facts.values()]
    lo, hi = min(good), max(bad)
    return {"correct_min": lo, "correct_avg": sum(good) / len(good), "wrong_max": hi,
            "separable": lo > hi, "suggested": round((lo + hi) / 2, 2) if lo > hi else None}


def main() -> None:
    s = get_settings()
    r = analyze(build_embedder(s).embed)
    print(f"embedding provider      : {type(getattr(build_embedder(s), 'inner', None)).__name__}")
    print(f"matching pairs          : min {r['correct_min']:.2f}  avg {r['correct_avg']:.2f}")
    print(f"pairs that must NOT match: max {r['wrong_max']:.2f}")
    print(f"current MEMORY_ANSWER_THRESHOLD = {s.memory_answer_threshold}")
    if r["separable"]:
        print(f"\nClean separation. Suggested: MEMORY_ANSWER_THRESHOLD={r['suggested']}")
        if s.memory_answer_threshold > r["correct_min"]:
            print("  -> your current value is ABOVE the weakest correct match, so some recall questions will skip the shortcut.")
    else:
        print(f"\nNo clean separation (weakest correct {r['correct_min']:.2f} <= strongest wrong {r['wrong_max']:.2f}).")
        print(f"Use a conservative value just above {r['wrong_max']:.2f} (wrong answers are worse than a missed shortcut).")


if __name__ == "__main__":
    main()
