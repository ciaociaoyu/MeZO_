# FP16 h-star generalization summary

Generated: 2026-05-20T04:28:22.790502

Primary selector: `calibrated_hstar_absG_Lclean32_q90`.

## Seed robustness: RoBERTa-large/SST-5 FP16

| model | dataset | seed | hstar h | empirical min-MSE h | nmse ratio | corr gap | pass | L_h2 | G_h |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| roberta-large | sst-5 | 16 | 0.0001 | 0.0003 | 1.814 | 0.00152 | True | 1e-05 | 0.0003 |
| roberta-large | sst-5 | 17 | 0.0001 | 0.0003 | 7.059 | 0.001481 | True | 1e-05 | 0.0003 |
| roberta-large | sst-5 | 18 | 0.0001 | 0.0003 | 6.088 | 0.001002 | True | 1e-05 | 0.0003 |
| roberta-large | sst-5 | 19 | 0.0001 | 0.0003 | 13.89 | 0.00186 | True | 3e-05 | 0.0003 |

## Task robustness: RoBERTa-large/RTE FP16

| model | dataset | seed | hstar h | empirical min-MSE h | nmse ratio | corr gap | pass | L_h2 | G_h |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| roberta-large | rte | 16 | 0.0003 | 0.001 | 1.16 | 0.001636 | True | 0.0003 | 0.0003 |
| roberta-large | rte | 17 | 0.0001 | 0.0003 | 1.296 | 0.000573 | True | 1e-05 | 0.0003 |
| roberta-large | rte | 18 | 0.0001 | 0.001 | 1.602 | 0.001021 | True | 1e-05 | 0.0003 |

## Model robustness: OPT/SST-2 FP16

| model | dataset | seed | hstar h | empirical min-MSE h | nmse ratio | corr gap | pass | L_h2 | G_h |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| _none completed_ | | | | | | | | | |

## Skipped settings

- C_model OPT-1.3B sst-2 seed 16: AttributeError("'OPTConfig' object has no attribute 'apply_lora'")
- C_model OPT-1.3B sst-2 seed 17: AttributeError("'OPTConfig' object has no attribute 'apply_lora'")

