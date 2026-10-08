"""Post-deploy smoke test.  python -m scripts.smoke_test https://costmind-api.onrender.com --api-key KEY

Free Render services take ~1 minute to wake up, so timeouts are generous.
"""
import argparse
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--api-key", default="")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    h = {"X-API-Key": a.api_key} if a.api_key else {}
    c = httpx.Client(timeout=120, headers=h)
    uid, results = "smoke-test-user", []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    t0 = time.time()
    r = c.get(f"{base}/health")
    check("health (wakes the service)", r.status_code == 200, f"{time.time() - t0:.0f}s")
    d = c.get(f"{base}/health/deps").json()
    check("dependencies", d["status"] == "ok", str(d["dependencies"]))
    if a.api_key:
        check("auth enforced", httpx.post(f"{base}/chat", json={"user_id": uid, "message": "hi"}, timeout=30).status_code == 401)
    q = "Explain DNS with an example"
    r1 = c.post(f"{base}/chat", json={"user_id": uid, "message": q})
    check("chat works", r1.status_code == 200, f"-> {r1.json().get('model') if r1.status_code == 200 else r1.text[:100]}")
    if r1.status_code == 200:
        r2 = c.post(f"{base}/chat", json={"user_id": uid, "message": q}).json()
        check("semantic cache hit on repeat", r2["cache_hit"] is True, f"(cost ${r2['cost']})")
    c.post(f"{base}/chat", json={"user_id": uid, "message": "My name is Smoke"})
    r3 = c.post(f"{base}/chat", json={"user_id": uid, "message": "what is my name"}).json()
    check("memory recall", "smoke" in r3["reply"].lower(), f"-> {r3['reply'][:60]!r}")
    m = c.get(f"{base}/metrics").json()
    check("metrics", m["requests"] >= 3, f"requests={m['requests']} avoided={m['llm_calls_avoided_pct']}%")
    check("forget-me", c.delete(f"{base}/memory/{uid}").json().get("deleted") is True)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
