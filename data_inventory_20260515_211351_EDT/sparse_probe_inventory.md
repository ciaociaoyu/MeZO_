
# Sparse Probe Inventory

## Scope

Primary source: `experiments/int8_update_sparse_plan/probe_window_h100_20260512/sparse_probe_summary.csv`. The same H100 package archive contains raw sparse probe JSONL stats and plots by both raw h and h_active.

## Available Data

- Sparse rates p: 0.003;0.01;0.03;0.1.
- h_raw grid: 4.108e-05;7.5e-05;8.216e-05;0.0001299;0.00015;0.0001643;0.0002372;0.0002598;0.0003;0.0003286;0.0004743;0.0005196;0.0006;0.0006573;0.0009487;0.00103923;0.0012;0.00131453;0.00189737;0.00207846;0.0024;0.00379473;0.00415692;0.00758947.
- h_active grid: 0.00075;0.0015;0.003;0.006;0.012;0.024.
- Sparse mode: bernoulli.
- Sparse rescale: inv_sqrt_p.
- Rows parsed: 24.
- Curves can be plotted raw h vs h_active: True.
- Training follow-up exists: True.

## Best h_active By corr_fd_true Within Each p

| p | h_raw | h_active | corr_fd_true | nMSE_fd_true | alignment | norm_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0.003 | 0.0003286 | 0.006 | 0.990691 | 0.0174639 | 0.977526 | 1.02313 |
| 0.01 | 0.0012 | 0.012 | 0.982456 | 0.0349333 | 0.99075 | 1.00939 |
| 0.03 | 0.00103923 | 0.006 | 0.981233 | 0.0370592 | 0.976939 | 1.0237 |
| 0.1 | 0.00189737 | 0.006 | 0.982461 | 0.0374947 | 0.978559 | 1.02194 |

## Overall Best Row

| p | h_raw | h_active | corr_fd_true | nMSE_fd_true |
| --- | --- | --- | --- | --- |
| 0.003 | 0.0003286 | 0.006 | 0.990691 | 0.0174639 |

## Training Follow-up

- 300-step sparse validation rows: 2.
- Future sparse training summary rows: 0.
- `sparse_partial/` update stats present in future package: True.

## Notes

- The table uses `h_active = h_raw / sqrt(p)` and `inv_sqrt_p` rescaling.
- Good sparse settings appear in the source around shared `h_active` values across p. Treat this as probe evidence unless paired with completed training rows.
- The future sparse training line is incomplete in local summaries: `summary_sparse.csv` has no data rows.
