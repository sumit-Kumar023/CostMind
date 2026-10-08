# Benchmark results

320 queries - baseline: always `gpt-4o` with full per-user history - CostMind: cheap=`gpt-4o-mini`, premium=`gpt-4o`

| Metric | Baseline | CostMind | Result |
|---|---|---|---|
| Total cost | $1.2700 | $0.2135 | **X = 83.2% cheaper** |
| Prompt tokens | 567,088 | 207,398 | **Y = 63.4%** |
| LLM calls avoided | 0% | 17.8% | **Z = 17.8%** |
| Avg latency | 2130 ms | 3700 ms | |
| Recall accuracy | 100% | 83% | |
| Judge score (1-5) | 4.7 | 4.7 | 100.0% quality retained |

CostMind cost breakdown: {'chat_cost_usd': 0.204615, 'maintenance_cost_usd': 0.008685, 'embedding_cost_usd': 0.000197, 'chat_prompt_tokens': 170160, 'maintenance_prompt_tokens': 37238, 'extractions_skipped': 233, 'compressions': 59}
