## Results

**TL;DR:** On a 320-query benchmark, CostMind cut total LLM cost by **83.2%** ($1.27 → $0.21) and prompt tokens by **63.4%** versus an always-premium, full-history baseline, with the same LLM-judged quality score (4.7/5). The price: **+19.2% average latency** and **9 pp lower memory-recall accuracy**. Full numbers and caveats are below.

### Benchmark setup

- **Command:** `python -m benchmarks.run_benchmark --judge --sweep`
- **Workload:** 320 queries: 60% simple / 30% medium / 10% complex, ~25% near-duplicates, personal-fact recall scenarios, and a few long-running "power users".
- **Baseline:** always the premium model, sent each user's full conversation history.
- **CostMind totals include everything:** escalations, memory upkeep (summaries, fact extraction) and embeddings.

### Headline numbers

| Metric | Baseline | CostMind | Change |
|---|---:|---:|---:|
| Total cost | $1.2700 | $0.2135 | **-83.2%** (saves $1.0565) |
| Prompt tokens | 567,088 | 207,398 | **-63.4%** |
| LLM calls avoided | 0 | 57 / 320 | **17.8% of calls** |
| Avg latency | 2,130 ms | 2,540 ms | +19.2% (worse) |
| Memory-recall accuracy | 100% | 91% | -9 pp (worse) |
| Answer quality (LLM judge, 1-5) | 4.7 | 4.7 | Equal (n = 60) |

### Trade-offs (read this before relying on the savings)

- **Latency is higher.** Average latency rose from 2.13 s to 2.54 s.
- **Memory recall is lower.** Recall accuracy fell from 100% to 91%, because the full-history baseline sees everything, while CostMind retrieves and compresses.
- **Quality parity is limited evidence.** The 4.7/5 score comes from an LLM judge on 60 of the 320 queries. It shows no measurable quality drop on that sample; it is not proof of identical correctness.
- **When CostMind is a good fit:** cost-sensitive workloads where a small recall and latency penalty is acceptable. **When it isn't:** use cases that need perfect recall or the lowest possible latency.

### Where the money goes

| Component | Cost | Share |
|---|---:|---:|
| Chat inference | $0.204615 | 95.8% |
| Memory maintenance | $0.008685 | 4.1% |
| Embeddings | $0.000197 | 0.1% |
| **Total** | **$0.2135** | |

Memory upkeep is cheap because extraction is selective: **233** memory extractions were skipped and **59** context compressions were run.

### Notes on measurement

- The benchmark is the authoritative measurement. Dashboard savings are estimates based on the configured premium model.
- Cost figures use the per-token prices configured in the repo; changing the premium model or prices will change the savings.
