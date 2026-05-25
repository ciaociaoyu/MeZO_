# Direct FP16 h-star MSE evaluation

Output directory: `analysis/fp16_direct_hstar_mse_smoke_20260520_180301`

The selector first computes continuous `hstar_cont` from Delta/G/L, then applies a visibility-only clamp to get `h_selected`. The empirical grid optimum is an oracle reference only.

## Table 1: Direct hstar evaluation

| model | dataset | seed | hstar_cont | nMSE(hstar_cont) | corr(hstar_cont) | h_selected | nMSE(h_selected) | corr(h_selected) | empirical min h | selected nMSE ratio | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| roberta-large | sst-5 | 16 | 0.000224813 | 0.00167111 | 1 | 0.000224813 | 0.00167111 | 1 | 0.0003 | 23.5841 | True |

## Table 2: Clamp behavior

| model | dataset | seed | hstar_cont | h_selected | clamp changed? | reason |
|---|---|---:|---:|---:|---|---|
| roberta-large | sst-5 | 16 | 0.000224813 | 0.000224813 | False | raw_hstar_visible |

## Table 3: Group pass rates

| group | raw hstar pass rate | selected pass rate | strict selected pass rate | median selected nMSE ratio | max selected nMSE ratio |
|---|---:|---:|---:|---:|---:|
| A_seed | 1 | 1 | 0 | 23.5841 | 23.5841 |
| B_task | n/a | n/a | n/a | n/a | n/a |
| C_model | n/a | n/a | n/a | n/a | n/a |

## Interpretation Notes

- `hstar_cont` is formula-derived from Delta/G/L only.
- `h_selected` is chosen only by visibility diagnostics, not by MSE/correlation.
- `empirical_min_nmse_h` is an oracle grid reference and was not used for selection.
