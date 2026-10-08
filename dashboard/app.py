"""CostMind dashboard: cost saved, route/model usage, latency, and a live chat tester."""
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}
LIVE_CHAT = os.getenv("ENABLE_LIVE_CHAT", "true").lower() != "false"  # set false to stop public visitors spending your credit
st.set_page_config(page_title="CostMind Dashboard", page_icon="💸", layout="wide")


@st.cache_data(ttl=5)
def fetch(path: str, params: tuple = ()) -> dict:
    """GET from the API (60s timeout to survive Render cold starts)."""
    r = requests.get(f"{API}{path}", params=dict(params), headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


st.title("💸 CostMind — memory-enabled, cost-aware agent")

with st.sidebar:
    if LIVE_CHAT:
        st.header("Try it live")
        user_id = st.text_input("User ID", "demo-user")
        message = st.text_area("Message", "Hi, I'm Rahul and I love cricket")
        if st.button("Send", type="primary"):
            try:
                res = requests.post(f"{API}/chat", json={"user_id": user_id, "message": message}, headers=HEADERS, timeout=90)
                res.raise_for_status()
                d = res.json()
                st.success(d["reply"])
                via = "semantic cache" if d["cache_hit"] else "memory" if d["answered_from_memory"] else d["model"]
                st.caption(f"Served by **{via}** · ${d['cost']:.6f} · {d['latency_ms']:.0f} ms")
                st.caption(f"Why: {d['route_reason']}")
                st.cache_data.clear()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Chat failed: {exc}")
    if st.button("Refresh metrics"):
        st.cache_data.clear()

try:
    m = fetch("/metrics")
    rows = fetch("/metrics/recent", (("limit", 300),))["rows"]
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot reach the API at {API}: {exc}")
    st.stop()

if not m["requests"]:
    st.info("No requests yet. Send a message from the sidebar.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests", m["requests"])
c2.metric("Est. cost saved vs always-premium", f"{m['savings_pct_est']}%")
c3.metric("LLM calls avoided", f"{m['llm_calls_avoided_pct']}%", help="cache hits + answered from memory")
c4.metric("Avg / p95 latency", f"{m['avg_latency_ms']:.0f} / {m['p95_latency_ms']:.0f} ms")
c5.metric("Total spend", f"${m['total_cost_usd']:.4f}", help=f"incl. ${m['maintenance_cost_usd']:.4f} memory upkeep")

left, right = st.columns(2)
with left:
    st.subheader("How requests were served")
    st.bar_chart(pd.Series(m["by_route"], name="requests"))
    st.subheader("Avg latency by route (ms)")
    st.bar_chart(pd.Series(m["avg_latency_ms_by_route"], name="ms"))
with right:
    st.subheader("Model usage")
    st.bar_chart(pd.Series(m["by_model"], name="requests"))
    st.subheader("Cumulative cost vs always-premium baseline")
    df = pd.DataFrame(rows).sort_values("id")
    curve = pd.DataFrame({"CostMind": df["cost"].cumsum(), "Always premium (est.)": df["baseline_cost_est"].cumsum()})
    st.line_chart(curve.reset_index(drop=True))

st.caption(f"LLM requests: avg confidence {m['avg_confidence']} · escalation rate {m['escalation_rate_pct']}% · "
           f"prompt tokens {m['prompt_tokens']:,}. Baseline = one premium call with the same final prompt; "
           "the benchmark script measures the real thing.")
st.subheader("Recent requests")
cols = ["id", "user_id", "route", "model", "tier", "escalations", "confidence", "cost", "latency_ms", "route_reason"]
st.dataframe(pd.DataFrame(rows)[cols], use_container_width=True, hide_index=True)
