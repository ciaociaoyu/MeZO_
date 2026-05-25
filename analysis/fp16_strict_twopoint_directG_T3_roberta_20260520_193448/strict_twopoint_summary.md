# Strict Symmetric Two-Point H-Star Summary

Output directory: `analysis/fp16_strict_twopoint_directG_T3_roberta_20260520_193448`

This is probe-only. The empirical oracle h is used only as a reference, not for selecting any h-star.

## Table 1: h4 vs strict h6

| model | dataset | seed | h4_fdG | h6_fdG_S3 | h6_trueG_S3 | h_oracle | oracle/h4 | oracle/h6_fdG | nMSE ratio h4 | nMSE ratio h6_fdG | pass h4 | pass h6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| roberta-large | sst-5 | 16 | 9.91903e-05 | 0.00104253 | 0.00106292 | 0.0003 | 3.02449 | 0.287762 | 1.56627 | 1.10015 | True | True |
| roberta-large | sst-5 | 17 | 0.000130445 | 0.00092437 | 0.000931074 | 0.0003 | 2.29982 | 0.324545 | 5.09227 | 1.86974 | True | True |
| roberta-large | sst-5 | 18 | 0.000110329 | 0.000940434 | 0.00096095 | 0.0003 | 2.71913 | 0.319002 | 6.13889 | 2.35902 | True | True |
| roberta-large | sst-5 | 19 | 0.000121119 | 0.000897969 | 0.000922837 | 0.0003 | 2.47691 | 0.334087 | 7.10459 | 4.83193 | True | True |
| roberta-large | rte | 16 | 0.000146553 | 0.00106993 | 0.00103122 | 0.001 | 6.82347 | 0.934638 | 1.24843 | 0.992869 | True | True |
| roberta-large | rte | 17 | 0.000123098 | 0.000956004 | 0.000971268 | 0.0003 | 2.43709 | 0.313806 | 0.709295 | 1.10764 | True | True |
| roberta-large | rte | 18 | 0.000164258 | 0.000990222 | 0.00104411 | 0.001 | 6.088 | 1.00987 | 1.71025 | 0.9926 | True | True |

## Table 2: Group comparison

| group | median oracle/h4 | median oracle/h6_fdG | h4 pass rate | h6 pass rate | h4 strict pass | h6 strict pass |
|---|---:|---:|---:|---:|---:|---:|
| A_seed | 2.59802 | 0.321773 | 1 | 1 | 0 | 0 |
| B_task | 6.088 | 0.934638 | 1 | 1 | 0.333 | 0.667 |
| C_model | n/a | n/a | n/a | n/a | n/a | n/a |
| overall | 2.71913 | 0.324545 | 1 | 1 | 0.143 | 0.286 |

## Table 3: G estimation

| model | dataset | seed | G_true | G_fd_multi | G_fd/G_true |
|---|---|---:|---:|---:|---:|
| roberta-large | sst-5 | 16 | 13.3596 | 12.6055 | 0.943556 |
| roberta-large | sst-5 | 17 | 17.7746 | 17.3934 | 0.978555 |
| roberta-large | sst-5 | 18 | 18.7191 | 17.5456 | 0.937309 |
| roberta-large | sst-5 | 19 | 16.1515 | 14.8807 | 0.921317 |
| roberta-large | rte | 16 | 13.4481 | 15.0203 | 1.11691 |
| roberta-large | rte | 17 | 9.33162 | 8.89855 | 0.953591 |
| roberta-large | rte | 18 | 14.1571 | 12.0762 | 0.853013 |

## Table 4: T3 stability

| model | dataset | seed | selected h3 values | S3_sq_multi | stability flags |
|---|---|---:|---|---:|---|
| roberta-large | sst-5 | 16 | 0.0015;0.003;0.004 | 1.11085e+12 |  |
| roberta-large | sst-5 | 17 | 0.001;0.0015;0.002;0.003 | 4.35272e+12 |  |
| roberta-large | sst-5 | 18 | 0.001;0.0015;0.002;0.003;0.004 | 3.99422e+12 |  |
| roberta-large | sst-5 | 19 | 0.001;0.0015;0.002;0.003;0.004 | 3.79092e+12 |  |
| roberta-large | rte | 16 | 0.001;0.0015;0.002;0.004 | 1.34985e+12 |  |
| roberta-large | rte | 17 | 0.001;0.0015;0.002;0.003;0.004 | 9.31e+11 |  |
| roberta-large | rte | 18 | 0.001;0.0015;0.002;0.003;0.004 | 1.38846e+12 |  |

## Interpretation Notes

- `h6_fdG_S3` is the deployable-like strict two-point formula using FP16 finite-difference G and clean-FP32 third-moment estimates.
- `h6_trueG_S3` is diagnostic: it tests the formula with exact FP32 gradient norm.
- `h_oracle` is only the grid min-nMSE reference and was not used to choose h.
