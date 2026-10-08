import time

from app.memory.scoring import estimate_importance, recency_score, relevance_score

DAY = 86400


def test_recency_half_life():
    now = time.time()
    assert recency_score(now, now) == 1.0
    assert abs(recency_score(now - 14 * DAY, now, 14) - 0.5) < 1e-6


def test_newer_beats_older_at_equal_similarity():
    now = time.time()
    new = relevance_score(0.8, now, 0.5, now=now)
    old = relevance_score(0.8, now - 60 * DAY, 0.5, now=now)
    assert new > old


def test_important_beats_trivial():
    now = time.time()
    assert relevance_score(0.7, now, 0.9, now=now) > relevance_score(0.7, now, 0.1, now=now)


def test_score_bounded():
    now = time.time()
    assert 0 <= relevance_score(5, now, 5, now=now) <= 1


def test_importance_heuristic():
    assert estimate_importance("My name is Rahul") > estimate_importance("ok thanks")
