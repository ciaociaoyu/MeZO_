# Paper Update Notes for V10

## Central interpretation

The aggregated results support the V10 framing that h is a finite-difference
probe guardrail. Inside a safe region, h-policy differences are not reliably an
accuracy knob. Low precision, sparse, and prefix settings are stress tests.

## Priority 1

The existing high-precision RoBERTa/SST-5 h sweeps are useful for showing a
plateau, but not enough for the requested paired multi-seed variance
decomposition. `priority1_variance_components.csv` records this as
`insufficient_existing_multiseed_data`.

Paper use: state as a planned/needed supplement unless new paired seeds are
run.

## Priority 2

`sparse_prefix_true_mse_visibility.csv` aggregates probe-level evidence.
Sparse p=0.1 includes true directional nMSE/correlation and visibility
diagnostics. Prefix summaries include true directional nMSE/correlation where
available but often lack active fraction and norm-ratio.

Paper use: sparse p=0.1 can support default-safe visibility claims; prefix
default-failure needs either training evidence or additional geometry probes if
active/norm diagnostics are required.

## Priorities 3 and 4

Existing seed-fixed INT4 sparse/prefix tables are mostly seed16. The aggregate
CSVs preserve `n_seeds`, so single-seed rows are clearly visible.

Paper use: do not call these multi-seed confirmations unless `n_seeds >= 3`.

## Priority 5

`probe_update_mismatch_diagnostics.csv` combines canonical dense INT8/INT4
SST-5 mismatch diagnostics with available sparse p=0.1 diagnostics.

Paper use: supports the low-precision boundary-case section. Prefix mismatch
needs an additional probe if the paper needs active fraction and per-coordinate
jump distributions for prefix.

## Audit tasks

- Prefix RTE: see `audit_prefix_rte_sparse_trec.csv`. Include only if same
  family/full fixed/default/reference rows exist.
- Sparse TREC: see `audit_prefix_rte_sparse_trec.csv`. In prior final artifacts
  sparse TREC often appears as medium/incomplete; keep appendix unless complete.
- V10 table value consistency: see `v10_table_values_audit.csv`.
- Residual-grid/QES-like runs: keep as update-side diagnostics only.
