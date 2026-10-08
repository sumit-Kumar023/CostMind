"""Deterministic benchmark dataset: ~60% simple / 30% medium / 10% complex generic queries,
~25% near-duplicate paraphrases, and personal-fact + recall scenarios (with ground-truth answers).

Usage: python -m benchmarks.generate_queries [--total 320]
"""
import argparse
import json
import random
from pathlib import Path

COUNTRIES = "France Japan Brazil Canada Egypt India Kenya Norway Peru Spain Italy Chile Ghana Cuba Nepal Iran Iraq Laos Mali Oman Qatar Togo Chad Fiji Haiti Malta Yemen Zambia Sweden Poland".split()
TERMS = ["photosynthesis", "inflation", "a black hole", "DNA", "an algorithm", "gravity", "democracy", "a vaccine", "entropy", "an API",
         "blockchain", "a neuron", "supply and demand", "machine learning", "a prime number", "the water cycle", "a compiler", "osmosis",
         "a database index", "encryption", "an eclipse", "the Renaissance", "a hash table", "cloud computing", "a mortgage", "GDP",
         "an ecosystem", "a recession", "a firewall", "latency"]
ACRONYMS = "HTTP CPU GDP DNS RAM SQL HTML USB NASA UNESCO API JSON SSD VPN GPS PDF CSS TCP LAN OCR".split()
UNITS = [("minutes", "day"), ("seconds", "hour"), ("centimeters", "meter"), ("ounces", "pound"), ("days", "leap year"), ("months", "decade"),
         ("grams", "kilogram"), ("feet", "mile"), ("hours", "week"), ("inches", "foot"), ("milliliters", "liter"), ("weeks", "year"),
         ("meters", "kilometer"), ("pints", "gallon"), ("days", "week"), ("hours", "day"), ("minutes", "hour"), ("quarters", "dollar"),
         ("cents", "dollar"), ("yards", "mile")]
BOOKS = ["Hamlet", "Pride and Prejudice", "1984", "Moby Dick", "The Odyssey", "Don Quixote", "Frankenstein", "Dracula", "The Hobbit",
         "Ulysses", "Emma", "Macbeth", "Les Miserables", "The Iliad", "Walden"]
PRIMES = [17, 21, 29, 33, 37, 49, 51, 53, 57, 61]
MULTS = [(12, 13), (7, 8), (9, 14), (15, 16), (11, 11), (6, 19), (13, 17), (8, 12), (14, 15), (18, 7)]
CONCEPTS = ["recursion", "the TCP handshake", "compound interest", "the greenhouse effect", "a REST API", "Bayes' theorem", "inflation",
            "load balancing", "the immune system", "supply chains", "garbage collection", "public-key cryptography", "the French Revolution",
            "neural networks", "version control", "opportunity cost", "the Doppler effect", "caching", "photosynthesis", "containerization",
            "machine learning bias", "a hash map", "the CAP theorem", "continuous integration", "plate tectonics", "the scientific method",
            "SQL joins", "microservices", "DNS", "OAuth"]
MEDIUM_T = ["Explain {c} with an example and list two common pitfalls.", "Summarize {c} and explain how it works, with an example.",
            "Write a short explanation of {c} and give an example."]
PAIRS = [("SQL", "NoSQL", "a high-traffic e-commerce site"), ("microservices", "a monolith", "a five-person startup"),
         ("REST", "GraphQL", "a mobile app backend"), ("Kafka", "RabbitMQ", "an event-driven payments system"),
         ("Postgres", "MongoDB", "a social network"), ("Kubernetes", "serverless", "a bursty ML inference workload"),
         ("TCP", "UDP", "a real-time multiplayer game"), ("Redis", "Memcached", "a session store"),
         ("batch", "stream processing", "fraud detection"), ("React", "Vue", "a large enterprise dashboard"),
         ("a monorepo", "polyrepo", "a 200-engineer company"), ("gradient boosting", "deep learning", "tabular churn prediction")]
COMPLEX_T = ["Compare and analyze the trade-offs of {a} versus {b} for {s}, step by step, and recommend one.",
             "Design the architecture for {s} using {a} or {b}, evaluate the trade-offs, and justify your strategy step by step."]
NAMES = ["Rahul", "Priya", "Amit", "Neha", "Karan", "Sneha", "Vikram", "Anjali"]
HOBBIES = ["cricket", "chess", "painting", "hiking", "cooking", "photography", "gardening", "cycling"]
ALLERGENS = ["peanuts", "shellfish", "pollen", "dust", "penicillin", "eggs", "soy", "latex"]

PARAPHRASE = [lambda m: m.lower(), lambda m: "Please " + m[0].lower() + m[1:], lambda m: "Hey, " + m,
              lambda m: m.rstrip("?.") + "?", lambda m: m + " Thanks!"]


def _simple_pool() -> list[str]:
    return ([f"What is the capital of {c}?" for c in COUNTRIES] + [f"Define {t} in one sentence." for t in TERMS]
            + [f"What does {a} stand for?" for a in ACRONYMS] + [f"How many {a} are in a {b}?" for a, b in UNITS]
            + [f"Who wrote {b}?" for b in BOOKS] + [f"Is {n} a prime number?" for n in PRIMES]
            + [f"What is {a} times {b}?" for a, b in MULTS])


def generate(total: int = 320, seed: int = 42) -> list[dict]:
    """Build the ordered query list. Same seed -> identical dataset."""
    rng = random.Random(seed)
    n_personal_users = len(NAMES)
    personal_q = n_personal_users * 6                      # 3 facts + 3 recalls each
    generic_total = total - personal_q
    n_dups = round(generic_total * 0.25)
    n_base = generic_total - n_dups
    n_simple, n_medium = round(n_base * 0.60), round(n_base * 0.30)
    n_complex = n_base - n_simple - n_medium

    simple = rng.sample(_simple_pool(), n_simple)
    medium = rng.sample([t.format(c=c) for c in CONCEPTS for t in MEDIUM_T], n_medium)
    cplx = rng.sample([t.format(a=a, b=b, s=s) for a, b, s in PAIRS for t in COMPLEX_T], n_complex)
    base = ([(m, "simple") for m in simple] + [(m, "medium") for m in medium] + [(m, "complex") for m in cplx])
    rng.shuffle(base)

    # a few heavy users with long conversations (where context compression pays off) + the personal users
    users = [f"u{i:02d}" for i in range(1, 11)] + [f"p{i}" for i in range(1, n_personal_users + 1)]
    weights = [8, 8, 8, 2, 2, 2, 1, 1, 1, 1] + [3] * n_personal_users
    seq = [{"message": m, "complexity": c, "kind": "fresh"} for m, c in base]
    for _ in range(n_dups):                                 # near-duplicates always come after their original
        j = rng.randrange(len(seq))
        while seq[j]["kind"] == "dup":
            j = rng.randrange(len(seq))
        src = seq[j]
        k = rng.randrange(j + 1, len(seq) + 1)
        seq.insert(k, {"message": rng.choice(PARAPHRASE)(src["message"]), "complexity": src["complexity"], "kind": "dup"})
    for item in seq:
        item["user_id"] = rng.choices(users, weights)[0]

    inserts = []                                            # (position, item): facts early, recalls late
    n = len(seq)
    for i in range(n_personal_users):
        uid, name, hobby, allergen = f"p{i + 1}", NAMES[i], HOBBIES[i], ALLERGENS[i]
        facts = [f"My name is {name}.", f"I love {hobby}.", f"I am allergic to {allergen}."]
        recalls = [("What is my name?", name), ("What do I love?", hobby), ("What am I allergic to?", allergen)]
        for f in facts:
            inserts.append((rng.randrange(0, int(n * 0.30)), {"user_id": uid, "message": f, "complexity": "simple", "kind": "fact"}))
        for q, ans in recalls:
            inserts.append((rng.randrange(int(n * 0.60), n), {"user_id": uid, "message": q, "complexity": "simple", "kind": "recall", "expected": ans}))
    for pos, item in sorted(inserts, key=lambda x: x[0], reverse=True):
        seq.insert(pos, item)
    for i, item in enumerate(seq):
        item["id"] = i
    return seq


def load_or_create(path: str | Path, total: int = 320) -> list[dict]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    data = generate(total)
    path.write_text(json.dumps(data, indent=1))
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=320)
    ap.add_argument("--out", default="benchmarks/queries.json")
    a = ap.parse_args()
    data = generate(a.total)
    Path(a.out).write_text(json.dumps(data, indent=1))
    kinds = {}
    for d in data:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"wrote {len(data)} queries to {a.out}: {kinds}")
