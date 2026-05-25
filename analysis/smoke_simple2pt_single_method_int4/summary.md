# INT4 SST-5 simple2pt_corrected h-star check

Output directory: `analysis/smoke_simple2pt_single_method_int4`

Selector tested: `simple2pt_corrected` (修正简单两点法).

| quantity | value |
|---|---:|
| precision branch | int4 |
| Delta RTNClip scale RMS | 0.00706799 |
| selected Delta mode | scale_rms_over_sqrt6 |
| selected Delta | 0.00288549 |
| G clean32 absG at h=3e-4 | 31.0138 |
| G clean32 median absG | 31.0138 |
| selected G mode | clean32_absG_median_1e-4_3e-4_1e-3 |
| selected G | 31.0138 |
| L clean32 q90 | 4.67549e-05 |
| h2_L | 1e-05 |
| L selection | plateau_q90_primary |
| d trainable | 407988407 |
| hstar_cont | 0.00108298 |
| nearest grid h | 0.001 |

## Comparison to current INT4 sweep

| reference | h | metric |
|---|---:|---:|
| sweep best best_eval_acc | 0.001 | 0.478923 |
| sweep best last_eval_acc | 0.001 | 0.478923 |
| sweep best lowbit visibility nMSE | 0.01 | 0.100231 |
| hstar_cont / best-training-h | 1.08298 | |
| nearest-grid / best-training-h | 1 | |

Interpretation: `simple2pt_corrected` is now the only enabled h-star selector: scale RMS / sqrt(6) Delta, clean32 absG median G, and L_clean32 q90.
