# Direct FP16 h-star MSE evaluation

Output directory: `analysis/fp16_direct_hstar_mse_opt1p3b_20260520_182256`

The selector first computes continuous `hstar_cont` from Delta/G/L, then applies a visibility-only clamp to get `h_selected`. The empirical grid optimum is an oracle reference only.

## Table 1: Direct hstar evaluation

| model | dataset | seed | hstar_cont | nMSE(hstar_cont) | corr(hstar_cont) | h_selected | nMSE(h_selected) | corr(h_selected) | empirical min h | selected nMSE ratio | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OPT-1.3B | sst-2 | 16 | 3.70448e-05 | 0.0594512 | 0.971825 | 4.74838e-05 | 0.0435211 | 0.977408 | 0.0001 | 3.5076 | False |
| OPT-1.3B | sst-2 | 17 | 3.65734e-05 | 0.0979448 | 0.957988 | 4.70299e-05 | 0.04132 | 0.985952 | 0.0001 | 2.80573 | True |

## Table 2: Clamp behavior

| model | dataset | seed | hstar_cont | h_selected | clamp changed? | reason |
|---|---|---:|---:|---:|---|---|
| OPT-1.3B | sst-2 | 16 | 3.70448e-05 | 4.74838e-05 | True | binary_visibility_clamp |
| OPT-1.3B | sst-2 | 17 | 3.65734e-05 | 4.70299e-05 | True | binary_visibility_clamp |

## Table 3: Group pass rates

| group | raw hstar pass rate | selected pass rate | strict selected pass rate | median selected nMSE ratio | max selected nMSE ratio |
|---|---:|---:|---:|---:|---:|
| A_seed | n/a | n/a | n/a | n/a | n/a |
| B_task | n/a | n/a | n/a | n/a | n/a |
| C_model | 0 | 0.5 | 0 | 3.15667 | 3.5076 |

## Interpretation Notes

- `hstar_cont` is formula-derived from Delta/G/L only.
- `h_selected` is chosen only by visibility diagnostics, not by MSE/correlation.
- `empirical_min_nmse_h` is an oracle grid reference and was not used for selection.
