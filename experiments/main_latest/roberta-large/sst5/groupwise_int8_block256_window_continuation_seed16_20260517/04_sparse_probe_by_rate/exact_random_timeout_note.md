# exact_random Sparse Probe Note

The requested sparse probe mode was `exact_random` if available, otherwise `bernoulli`.

`exact_random` was available, but the p=0.003 group produced no `probe_stats.jsonl` rows after about 38 minutes of sustained H100 utilization. The process was terminated to avoid spending the remaining local H100 budget on exact mask generation before any sparse-rate table could be produced.

The continuation therefore uses a labeled `bernoulli` fallback for the same p / h_active grid:

- p = 0.003
- p = 0.01
- p = 0.03
- p = 0.1

All Bernoulli sparse probe rows keep `sparse_mode=bernoulli` in the JSONL and summaries.

