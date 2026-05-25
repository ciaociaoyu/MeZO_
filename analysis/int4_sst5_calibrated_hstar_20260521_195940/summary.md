# INT4 SST-5 calibrated h-star check

Output directory: `/scratch/jy03364/MeZO_/analysis/int4_sst5_calibrated_hstar_20260521_195940`

Selector tested: `calibrated_hstar_absG_Lclean32_q90` adapted to RTNClip INT4 by using INT4 group-scale RMS as Delta.

| quantity | value |
|---|---:|
| Delta INT4 scale RMS | 0.0146615 |
| G absG | 42.7614 |
| h_G | 0.001 |
| G selection | manual_int4_primary_hG_1e-3 |
| L clean32 q90 | 2.36719e-05 |
| h2_L | 1e-05 |
| L selection | plateau_q90_primary |
| d trainable | 407938142 |
| hstar_cont | 0.00402876 |
| nearest grid h | 0.004 |

## Comparison to current INT4 sweep

| reference | h | metric |
|---|---:|---:|
| sweep best best_eval_acc | 0.001 | 0.478923 |
| sweep best last_eval_acc | 0.001 | 0.478923 |
| sweep best lowbit visibility nMSE | 0.01 | 0.100231 |
| hstar_cont / best-training-h | 4.02876 | |
| nearest-grid / best-training-h | 4 | |

Interpretation: the formula lands far above the current INT4 training-optimal h. The low-bit displacement visibility metric keeps improving out to 1e-2, but training accuracy peaks around 1e-3, so INT4 needs an additional nonlocal/training-stability upper gate beyond visibility.
