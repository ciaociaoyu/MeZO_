
# Dense Probe Inventory

## Scope

Primary source: `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv`. The matching H100 archive is `experiments/int8_update_sparse_plan/probe_window_h100_20260512/probe_window_h100_20260512_results.tar.gz`.

## Available Data

- Precision modes: bf16;fp32;int8.
- h grid: 1e-05;3e-05;0.0001;0.0003;0.001;0.0015;0.002;0.003;0.004;0.005;0.01.
- Rows parsed: 33.
- Plots exist: True.
- Commands/configs exist: True.
- Raw JSONL probe stats exist: True.

## Best h By corr_fd_true

| precision | h_raw | corr_fd_true | nMSE_fd_true | alignment | norm_ratio | window_candidate |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 0.001 | 0.997958 | 0.00462076 | 1 | 1 | True |
| fp32 | 1e-05 | 1 | 6.666e-07 | 1 | 1 | True |
| int8 | 0.003 | 0.937567 | 0.126393 | 0.946725 | 1.05627 | True |

## Representative Evidence Rows

| precision | h_raw | alignment | norm_ratio | corr_fd_true | nMSE_fd_true | sign_agreement | window_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | 1e-05 | 0.0745006 | 13.424 | 0.120965 | 71.044 | 0.48 | False |
| int8 | 0.0003 | 0.40759 | 2.45352 | 0.593827 | 1.34663 | 0.68 | False |
| int8 | 0.003 | 0.946725 | 1.05627 | 0.937567 | 0.126393 | 0.94 | True |
| bf16 | 0.001 | 1 | 1 | 0.997958 | 0.00462076 | 0.98 | True |
| fp32 | 1e-05 | 1 | 1 | 1 | 6.666e-07 | 1 | True |

## Notes

- The dense probe contains alignment, norm ratio, finite-difference/true-gradient correlation, normalized MSE, and sign agreement in one table.
- INT8 small h rows are present and are marked `window_candidate=False` at `1e-5`, `3e-5`, `1e-4`, and `3e-4` in the source table.
- The source table is diagnostic; it does not by itself establish final fine-tuning accuracy.
