# V10 Supplement Results (20260626)

Generated from existing MeZO / ZO perturbation-radius artifacts.

Git commit at build time: `07c96b8c93a576c66f468835f06a3d8861b1b27a`

This folder is an audit-and-aggregation supplement. It does not launch new
training. Missing multi-seed experiments are explicitly marked as missing or
insufficient rather than fabricated.

## Contents

- `priority1_high_precision_existing_runs.csv`
- `priority1_variance_components.csv`
- `sparse_prefix_true_mse_visibility.csv`
- `prefix_int4_multiseed_per_run.csv`
- `prefix_int4_multiseed_aggregate.csv`
- `sparse_int4_multiseed_per_run.csv`
- `sparse_int4_multiseed_aggregate.csv`
- `probe_update_mismatch_diagnostics.csv`
- `audit_prefix_rte_sparse_trec.csv`
- `v10_table_values_audit.csv`
- `figures/*.png` and `figures/*.pdf`
- `raw_run_summaries/` copied source summary CSVs
- `paper_update_notes.md`

## Priority status

1. High-precision plateau multi-seed variance decomposition:
   existing h-sweeps are mostly seed16 only. The required paired seeds
   {16,32,64,128,256} were not found, so variance decomposition is marked
   insufficient.

2. Sparse/prefix INT4 probe diagnostics:
   sparse p=0.1 true-MSE/visibility data were found for several tasks from
   existing probe summaries. Prefix true-MSE/correlation summaries were found,
   but many prefix summaries do not include active fraction/norm-ratio geometry.

3. Prefix INT4 multi-seed confirmation:
   existing paper-facing table data are primarily seed16. Aggregates are
   emitted, but n_seeds indicates whether a row is genuinely multi-seed.

4. Sparse INT4 multi-seed confirmation:
   existing paper-facing table data are primarily seed16. Aggregates are
   emitted, but n_seeds indicates whether a row is genuinely multi-seed.

5. Probe/update mismatch diagnostic:
   canonical SST-5 dense INT8/INT4 diagnostics and available sparse p=0.1
   diagnostics are aggregated. Prefix mismatch geometry is incomplete.

## Important caveats

- Do not interpret missing nMSE/visibility fields as zero.
- Do not mix old highest-abs sparse mask runs with seed-fixed/task-gradient
  sparse runs in the main table.
- Do not use residual-grid or QZO/QES-like runs as mainline optimizer claims.
- Prefix RTE and sparse TREC are audited separately for comparability.
