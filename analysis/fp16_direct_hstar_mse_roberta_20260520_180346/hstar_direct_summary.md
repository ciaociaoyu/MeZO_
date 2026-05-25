# Direct FP16 h-star MSE evaluation

Output directory: `analysis/fp16_direct_hstar_mse_roberta_20260520_180346`

The selector first computes continuous `hstar_cont` from Delta/G/L, then applies a visibility-only clamp to get `h_selected`. The empirical grid optimum is an oracle reference only.

## Table 1: Direct hstar evaluation

| model | dataset | seed | hstar_cont | nMSE(hstar_cont) | corr(hstar_cont) | h_selected | nMSE(h_selected) | corr(h_selected) | empirical min h | selected nMSE ratio | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| roberta-large | sst-5 | 16 | 0.000103027 | 0.00412867 | 0.997821 | 0.000110144 | 0.00419088 | 0.997921 | 0.0003 | 1.3109 | True |
| roberta-large | sst-5 | 17 | 0.000133832 | 0.00214085 | 0.998982 | 0.000133832 | 0.00214085 | 0.998982 | 0.0003 | 3.88514 | True |
| roberta-large | sst-5 | 18 | 0.000109847 | 0.00142258 | 0.999269 | 0.000109847 | 0.00142258 | 0.999269 | 0.0003 | 3.85719 | True |
| roberta-large | sst-5 | 19 | 0.000124647 | 0.00158698 | 0.999269 | 0.000124647 | 0.00158698 | 0.999269 | 0.0003 | 4.91049 | True |
| roberta-large | rte | 16 | 0.000148041 | 0.0253563 | 0.987195 | 0.000148041 | 0.0253563 | 0.987195 | 0.001 | 1.22515 | True |
| roberta-large | rte | 17 | 0.000125468 | 0.00218555 | 0.998815 | 0.000125468 | 0.00218555 | 0.998815 | 0.0003 | 0.543834 | True |
| roberta-large | rte | 18 | 0.000166424 | 0.00409942 | 0.997959 | 0.000166424 | 0.00409942 | 0.997959 | 0.001 | 1.00682 | True |

## Table 2: Clamp behavior

| model | dataset | seed | hstar_cont | h_selected | clamp changed? | reason |
|---|---|---:|---:|---:|---|---|
| roberta-large | sst-5 | 16 | 0.000103027 | 0.000110144 | True | binary_visibility_clamp |
| roberta-large | sst-5 | 17 | 0.000133832 | 0.000133832 | False | raw_hstar_visible |
| roberta-large | sst-5 | 18 | 0.000109847 | 0.000109847 | False | raw_hstar_visible |
| roberta-large | sst-5 | 19 | 0.000124647 | 0.000124647 | False | raw_hstar_visible |
| roberta-large | rte | 16 | 0.000148041 | 0.000148041 | False | raw_hstar_visible |
| roberta-large | rte | 17 | 0.000125468 | 0.000125468 | False | raw_hstar_visible |
| roberta-large | rte | 18 | 0.000166424 | 0.000166424 | False | raw_hstar_visible |

## Table 3: Group pass rates

| group | raw hstar pass rate | selected pass rate | strict selected pass rate | median selected nMSE ratio | max selected nMSE ratio |
|---|---:|---:|---:|---:|---:|
| A_seed | 1 | 1 | 0 | 3.87116 | 4.91049 |
| B_task | 1 | 1 | 0.667 | 1.00682 | 1.22515 |
| C_model | n/a | n/a | n/a | n/a | n/a |

## Interpretation Notes

- `hstar_cont` is formula-derived from Delta/G/L only.
- `h_selected` is chosen only by visibility diagnostics, not by MSE/correlation.
- `empirical_min_nmse_h` is an oracle grid reference and was not used for selection.
