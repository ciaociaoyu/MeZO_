# FP16 Direct H-Star MSE Summary

Output directory: `analysis/fp16_direct_hstar_mse_20260520_183229_combined`

This is probe-only. No training, INT8/INT4/BF16/sparse/residual-grid, or full h-sweep jobs were launched.

## Direct hstar evaluation

| model | dataset | seed | hstar_cont | nmse(cont) | corr(cont) | vis(cont) | h_selected | nmse(selected) | corr(selected) | empirical min h | selected nmse ratio | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roberta-large | sst-5 | 16 | 0.000103 | 0.004129 | 0.9978 | False | 0.00011 | 0.004191 | 0.9979 | 0.0003 | 1.311 | True |
| roberta-large | sst-5 | 17 | 0.000134 | 0.002141 | 0.999 | True | 0.000134 | 0.002141 | 0.999 | 0.0003 | 3.885 | True |
| roberta-large | sst-5 | 18 | 0.00011 | 0.001423 | 0.9993 | True | 0.00011 | 0.001423 | 0.9993 | 0.0003 | 3.857 | True |
| roberta-large | sst-5 | 19 | 0.000125 | 0.001587 | 0.9993 | True | 0.000125 | 0.001587 | 0.9993 | 0.0003 | 4.91 | True |
| roberta-large | rte | 16 | 0.000148 | 0.02536 | 0.9872 | True | 0.000148 | 0.02536 | 0.9872 | 0.001 | 1.225 | True |
| roberta-large | rte | 17 | 0.000125 | 0.002186 | 0.9988 | True | 0.000125 | 0.002186 | 0.9988 | 0.0003 | 0.5438 | True |
| roberta-large | rte | 18 | 0.000166 | 0.004099 | 0.998 | True | 0.000166 | 0.004099 | 0.998 | 0.001 | 1.007 | True |
| OPT-1.3B | sst-2 | 16 | 3.7e-05 | 0.05945 | 0.9718 | False | 4.75e-05 | 0.04352 | 0.9774 | 0.0001 | 3.508 | False |
| OPT-1.3B | sst-2 | 17 | 3.66e-05 | 0.09794 | 0.958 | False | 4.7e-05 | 0.04132 | 0.986 | 0.0001 | 2.806 | True |

## Clamp behavior

| model | dataset | seed | hstar_cont | h_selected | clamp changed | reason |
|---|---:|---:|---:|---:|---:|---|
| roberta-large | sst-5 | 16 | 0.000103 | 0.00011 | True | binary_visibility_clamp |
| roberta-large | sst-5 | 17 | 0.000134 | 0.000134 | False | raw_hstar_visible |
| roberta-large | sst-5 | 18 | 0.00011 | 0.00011 | False | raw_hstar_visible |
| roberta-large | sst-5 | 19 | 0.000125 | 0.000125 | False | raw_hstar_visible |
| roberta-large | rte | 16 | 0.000148 | 0.000148 | False | raw_hstar_visible |
| roberta-large | rte | 17 | 0.000125 | 0.000125 | False | raw_hstar_visible |
| roberta-large | rte | 18 | 0.000166 | 0.000166 | False | raw_hstar_visible |
| OPT-1.3B | sst-2 | 16 | 3.7e-05 | 4.75e-05 | True | binary_visibility_clamp |
| OPT-1.3B | sst-2 | 17 | 3.66e-05 | 4.7e-05 | True | binary_visibility_clamp |

## Group pass rates

| group | n | raw hstar pass rate | selected pass rate | strict selected pass rate | median selected nmse ratio | max selected nmse ratio |
|---|---:|---:|---:|---:|---:|---:|
| A_seed | 4 | 1 | 1 | 0 | 3.871 | 4.91 |
| B_task | 3 | 1 | 1 | 0.667 | 1.007 | 1.225 |
| C_model | 2 | 0 | 0.5 | 0 | 3.157 | 3.508 |
| overall | 9 | 0.778 | 0.889 | 0.222 | 2.806 | 4.91 |

## Interpretation

- `hstar_cont` was computed from Delta, G, and L before looking at MSE/corr.
- `h_selected` was obtained only by the visibility rule, not by minimizing MSE.
- The empirical min-MSE grid h is included only as an oracle reference for ratios.
- RoBERTa generalizes well by the main pass criterion: all SST-5 and RTE settings pass after direct/visibility-clamped evaluation.
- OPT-1.3B/SST-2 is harder: raw h-star failed both seeds, and visibility clamp rescued one of two seeds; this suggests the RoBERTa-calibrated selector may need a model-family correction or a stricter lower visibility bound for OPT.
