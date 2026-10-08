"""Memory-aware routing: answer personal recall questions from memory without calling an LLM."""
import re

_RECALL = re.compile(r"^\s*(what|what's|whats|which|who|where|when|how (old|many|much)|do i|am i|did i|remind me|tell me)\b", re.I)
# Questions about someone/something ELSE ("my brother's name", "my dog's name") look like recall questions and score
# high against the user's own facts, so no similarity threshold can separate them. Send those to the LLM instead.
_THIRD_PARTY = re.compile(
    r"\bmy\s+\w+'s\b|\b(his|her|their|brother|sister|mother|father|mom|dad|wife|husband|friend|dog|cat|boss|colleague|"
    r"son|daughter|partner|girlfriend|boyfriend|neighbou?r|teacher|parents?|kids?|children|family)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(my|i|me|mine|i'm|i am)\b", re.I)


def is_recall_question(message: str) -> bool:
    """Short first-person question like 'what is my name?'."""
    return (len(message.split()) <= 14 and bool(_RECALL.search(message)) and bool(_FIRST_PERSON.search(message))
            and not _THIRD_PARTY.search(message))


def rewrite_fact(text: str) -> str:
    """'User loves cricket' -> 'You love cricket' (second person)."""
    if text.startswith("User's "):
        return "Your " + text[7:]
    if text.startswith("User is "):
        return "You are " + text[8:]
    return re.sub(r"^User (\w+)s ", r"You \1 ", text)


def answer_from_memory(recalled: list, threshold: float) -> tuple[str, float] | None:
    """Build an answer from strongly matching fact memories, or None if not confident."""
    facts = sorted((m for m in recalled if m.kind == "fact" and m.similarity >= threshold), key=lambda m: m.score, reverse=True)[:2]
    if not facts:
        return None
    return "From what I remember: " + "; ".join(rewrite_fact(f.text) for f in facts) + ".", facts[0].similarity
