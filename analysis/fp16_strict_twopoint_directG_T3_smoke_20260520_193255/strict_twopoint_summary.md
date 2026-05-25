# Strict Symmetric Two-Point H-Star Summary

Output directory: `analysis/fp16_strict_twopoint_directG_T3_smoke_20260520_193255`

This is probe-only. The empirical oracle h is used only as a reference, not for selecting any h-star.

## Table 1: h4 vs strict h6

| model | dataset | seed | h4_fdG | h6_fdG_S3 | h6_trueG_S3 | h_oracle | oracle/h4 | oracle/h6_fdG | nMSE ratio h4 | nMSE ratio h6_fdG | pass h4 | pass h6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

## Table 2: Group comparison

| group | median oracle/h4 | median oracle/h6_fdG | h4 pass rate | h6 pass rate | h4 strict pass | h6 strict pass |
|---|---:|---:|---:|---:|---:|---:|
| A_seed | n/a | n/a | n/a | n/a | n/a | n/a |
| B_task | n/a | n/a | n/a | n/a | n/a | n/a |
| C_model | n/a | n/a | n/a | n/a | n/a | n/a |
| overall | n/a | n/a | n/a | n/a | n/a | n/a |

## Table 3: G estimation

| model | dataset | seed | G_true | G_fd_multi | G_fd/G_true |
|---|---|---:|---:|---:|---:|

## Table 4: T3 stability

| model | dataset | seed | selected h3 values | S3_sq_multi | stability flags |
|---|---|---:|---|---:|---|

## Interpretation Notes

- `h6_fdG_S3` is the deployable-like strict two-point formula using FP16 finite-difference G and clean-FP32 third-moment estimates.
- `h6_trueG_S3` is diagnostic: it tests the formula with exact FP32 gradient norm.
- `h_oracle` is only the grid min-nMSE reference and was not used to choose h.
