# FP16 h-star generalization summary

Combined output: `analysis/fp16_hstar_generalization_20260520_043617_combined`

Primary selector: `calibrated_hstar_absG_Lclean32_q90`.

Directions: RoBERTa uses 32 probe / 8 L directions; OPT-1.3B uses 16 probe / 4 L directions due runtime and torch-env constraints.

| group | model | dataset | seed | hstar nearest | empirical min-MSE h | nmse ratio | corr gap | pass | strict pass | L_h2 | G_h |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| A_seed | roberta-large | sst-5 | 16 | 0.0001 | 0.0003 | 1.81447 | 0.00151973 | True | False | 1e-05 | 0.0003 |
| A_seed | roberta-large | sst-5 | 17 | 0.0001 | 0.0003 | 7.05883 | 0.00148111 | True | False | 1e-05 | 0.0003 |
| A_seed | roberta-large | sst-5 | 18 | 0.0001 | 0.0003 | 6.08797 | 0.00100171 | True | False | 1e-05 | 0.0003 |
| A_seed | roberta-large | sst-5 | 19 | 0.0001 | 0.0003 | 13.8949 | 0.00185957 | True | False | 3e-05 | 0.0003 |
| B_task | roberta-large | rte | 16 | 0.0003 | 0.001 | 1.1596 | 0.001636 | True | False | 0.0003 | 0.0003 |
| B_task | roberta-large | rte | 17 | 0.0001 | 0.0003 | 1.29577 | 0.000572986 | True | False | 1e-05 | 0.0003 |
| B_task | roberta-large | rte | 18 | 0.0001 | 0.001 | 1.60194 | 0.00102095 | True | False | 1e-05 | 0.0003 |
| C_model | OPT-1.3B | sst-2 | 16 | 3e-05 | 0.0003 | 6.85247 | 0.0399666 | False | False | 3e-05 | 0.0001 |
| C_model | OPT-1.3B | sst-2 | 17 | 3e-05 | 0.0001 | 7.51221 | 0.0495727 | False | False | 3e-05 | 0.0001 |

## Pass rates

- `A_seed`: 4/4 primary pass; 0/4 strict nMSE pass.
- `B_task`: 3/3 primary pass; 0/3 strict nMSE pass.
- `C_model`: 0/2 primary pass; 0/2 strict nMSE pass.

## Interpretation

- RoBERTa-large/SST-5: h-star consistently selects `1e-4`, while empirical min-MSE is `3e-4`; correlation remains close enough to pass the configured corr-gap rule.
- RoBERTa-large/RTE: h-star selects `1e-4` to `3e-4`; empirical min-MSE is `3e-4` or `1e-3`; all pass by correlation gap, not by strict nMSE ratio.
- OPT-1.3B/SST-2: h-star selects `3e-5`, below the empirical low-MSE window (`1e-4` to `3e-4`), and fails both pass criteria.
- No training was launched; this is probe/curvature analysis only.
