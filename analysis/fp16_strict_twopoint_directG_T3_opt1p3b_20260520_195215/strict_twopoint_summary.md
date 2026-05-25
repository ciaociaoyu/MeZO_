# Strict Symmetric Two-Point H-Star Summary

Output directory: `analysis/fp16_strict_twopoint_directG_T3_opt1p3b_20260520_195215`

This is probe-only. The empirical oracle h is used only as a reference, not for selecting any h-star.

## Table 1: h4 vs strict h6

| model | dataset | seed | h4_fdG | h6_fdG_S3 | h6_trueG_S3 | h_oracle | oracle/h4 | oracle/h6_fdG | nMSE ratio h4 | nMSE ratio h6_fdG | pass h4 | pass h6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| OPT-1.3B | sst-2 | 16 | 3.69527e-05 | 0.000785319 | 0.000736471 | 0.0001 | 2.70616 | 0.127337 | 5.3446 | 3.83409 | False | False |
| OPT-1.3B | sst-2 | 17 | 3.70934e-05 | 0.000493058 | 0.00049405 | 0.0001 | 2.6959 | 0.202816 | 6.5077 | 1.61657 | False | True |

## Table 2: Group comparison

| group | median oracle/h4 | median oracle/h6_fdG | h4 pass rate | h6 pass rate | h4 strict pass | h6 strict pass |
|---|---:|---:|---:|---:|---:|---:|
| A_seed | n/a | n/a | n/a | n/a | n/a | n/a |
| B_task | n/a | n/a | n/a | n/a | n/a | n/a |
| C_model | 2.70103 | 0.165076 | 0 | 0.5 | 0 | 0 |
| overall | 2.70103 | 0.165076 | 0 | 0.5 | 0 | 0 |

## Table 3: G estimation

| model | dataset | seed | G_true | G_fd_multi | G_fd/G_true |
|---|---|---:|---:|---:|---:|
| OPT-1.3B | sst-2 | 16 | 43.9095 | 53.2392 | 1.21247 |
| OPT-1.3B | sst-2 | 17 | 55.4148 | 55.0819 | 0.993992 |

## Table 4: T3 stability

| model | dataset | seed | selected h3 values | S3_sq_multi | stability flags |
|---|---|---:|---|---:|---|
| OPT-1.3B | sst-2 | 16 | 0.004 | 1.88669e+13 | fallback_T3 |
| OPT-1.3B | sst-2 | 17 | 0.001;0.004 | 3.2972e+14 |  |

## Interpretation Notes

- `h6_fdG_S3` is the deployable-like strict two-point formula using FP16 finite-difference G and clean-FP32 third-moment estimates.
- `h6_trueG_S3` is diagnostic: it tests the formula with exact FP32 gradient norm.
- `h_oracle` is only the grid min-nMSE reference and was not used to choose h.
