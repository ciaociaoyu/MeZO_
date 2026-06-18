# Audit Report

## Existing Interval Metrics

- facebook/opt-1.3b / sst-5 / int8 / dense: 21 rows, h=[np.float64(1e-07), np.float64(3e-07), np.float64(1e-06), np.float64(3e-06), np.float64(1e-05), np.float64(3e-05), np.float64(0.0001), np.float64(0.0003), np.float64(0.001), np.float64(0.002), np.float64(0.003), np.float64(0.005), np.float64(0.01), np.float64(0.03)]
- facebook/opt-1.3b / sst-5 / int8 / sparse_p0p1: 12 rows, h=[np.float64(1e-07), np.float64(3e-07), np.float64(1e-06), np.float64(3e-06), np.float64(1e-05), np.float64(3e-05), np.float64(0.0001), np.float64(0.0003), np.float64(0.001), np.float64(0.003), np.float64(0.01), np.float64(0.03)]
- roberta-large / sst-5 / int4 / dense: 37 rows, h=[np.float64(1e-08), np.float64(3e-08), np.float64(1e-07), np.float64(3e-07), np.float64(1e-06), np.float64(3e-06), np.float64(1e-05), np.float64(3e-05), np.float64(0.0001), np.float64(0.0003), np.float64(0.001), np.float64(0.0015), np.float64(0.002), np.float64(0.003), np.float64(0.004), np.float64(0.005), np.float64(0.01), np.float64(0.03), np.float64(0.1)]
- roberta-large / sst-5 / int4 / sparse_p0p1: 15 rows, h=[np.float64(1e-08), np.float64(3e-08), np.float64(1e-07), np.float64(3e-07), np.float64(1e-06), np.float64(3e-06), np.float64(1e-05), np.float64(3e-05), np.float64(0.0001), np.float64(0.0003), np.float64(0.001), np.float64(0.003), np.float64(0.01), np.float64(0.03), np.float64(0.1)]
- roberta-large / sst-5 / int8 / dense: 35 rows, h=[np.float64(1e-08), np.float64(3e-08), np.float64(1e-07), np.float64(3e-07), np.float64(1e-06), np.float64(3e-06), np.float64(1e-05), np.float64(3e-05), np.float64(0.0001), np.float64(0.0003), np.float64(0.001), np.float64(0.0015), np.float64(0.002), np.float64(0.003), np.float64(0.004), np.float64(0.005), np.float64(0.01), np.float64(0.03), np.float64(0.1)]
- roberta-large / sst-5 / int8 / sparse_p0p1: 15 rows, h=[np.float64(1e-08), np.float64(3e-08), np.float64(1e-07), np.float64(3e-07), np.float64(1e-06), np.float64(3e-06), np.float64(1e-05), np.float64(3e-05), np.float64(0.0001), np.float64(0.0003), np.float64(0.001), np.float64(0.003), np.float64(0.01), np.float64(0.03), np.float64(0.1)]

## Existing Training Results

- facebook/opt-1.3b / mnli / int4 / dense: 15 rows
- facebook/opt-1.3b / rte / int4 / dense: 16 rows
- facebook/opt-1.3b / sst-2 / int4 / dense: 15 rows
- facebook/opt-1.3b / sst-5 / int4 / dense: 17 rows
- facebook/opt-1.3b / sst-5 / int8 / dense: 8 rows
- facebook/opt-1.3b / trec / int4 / dense: 18 rows
- roberta-large / sst-5 / int4 / dense: 11 rows
- roberta-large / sst-5 / int8 / dense: 19 rows

## Missing / Prioritized

- OPT-1.3B / SST-5 / INT8 / dense+sparse interval geometry and loss nMSE
- RoBERTa and OPT / TREC,RTE / INT8 loss-level nMSE
- Short 300-step pilots for default vs selected h where full logs are absent
- INT4 secondary probes after INT8 coverage is complete

## Notes

- Workflow did not launch long training; it uses existing logs and probe outputs.
