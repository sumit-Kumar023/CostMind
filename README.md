## Results

Measured by `python -m benchmarks.run_benchmark --judge --sweep` on **320 queries** (60% simple / 30% medium / 10% complex, ~25% near-duplicates, personal-fact recall scenarios, a few long-running "power users").

Baseline = always the premium model with each user's full conversation history. CostMind totals include escalations, memory upkeep (summaries, fact extraction) and embeddings.

| Metric | Baseline | CostMind | Result |
|---|---:|---:|---:|
| Total cost | $1.2700 | $0.2135 | **83.2% cheaper** |
| Prompt tokens | 567,088 | 207,398 | **63.4% fewer** |
| LLM calls avoided | 0% | 17.8% (57/320) | **57 calls avoided** |
| Avg latency | 2,130 ms | 2,540 ms | **19.2% higher** |
| Memory-recall accuracy | 100% | 91% | **9 pp lower** |
| Answer quality (LLM judge, 1-5) | 4.7 | 4.7 | **100% retained** |

**Benchmark interpretation:** CostMind reduced cost substantially, but the optimization has trade-offs: average latency increased from 2.13s to 2.54s and memory-recall accuracy decreased from 100% to 91%. The equal 4.7/5 LLM-judge score was measured on 60 evaluated queries and should not be interpreted as proof of identical correctness.

**Cost breakdown:**
- Chat inference: `$0.204615`
- Memory maintenance: `$0.008685`
- Embeddings: `$0.000197`
- Total CostMind spend: approximately `$0.2135`
- Memory extractions skipped: `233`
- Context compressions: `59`

The benchmark is the authoritative measurement; dashboard savings are estimates based on the configured premium model.
