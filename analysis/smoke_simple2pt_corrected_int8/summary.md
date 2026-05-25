# INT8 SST-5 simple2pt_corrected h-star check

Output directory: `analysis/smoke_simple2pt_corrected_int8`

Selector tested: `simple2pt_corrected` (修正简单两点法).

| quantity | value |
|---|---:|
| precision branch | int8 |
| Delta INT4 scale RMS | 0.00093823 |
| Delta scale RMS / sqrt(6) | 0.000383031 |
| Delta empirical snap RMS at h=1e-3 | 0.000392684 |
| selected Delta mode | raw_scale_rms |
| selected Delta | 0.00093823 |
| G absG | 15.9112 |
| G clean32 absG at h=3e-4 | 21.3602 |
| selected G mode | clean32_absG |
| selected G | 21.4189 |
| h_G | 0.001 |
| G selection | manual_int4_primary_hG_1e-3 |
| L clean32 q90 | 1.16902e-05 |
| h2_L | 1e-05 |
| L selection | plateau_q90_primary |
| d trainable | 407938142 |
| hstar_cont | 0.0010264 |
| nearest grid h | 0.001 |

## Comparison to current INT8 sweep

| reference | h | metric |
|---|---:|---:|
| sweep best best_eval_acc | 0.001 | 0.476581 |
| sweep best last_eval_acc | 0.001 | 0.476581 |
| sweep best lowbit visibility nMSE | 0.01 | nan |
| hstar_cont / best-training-h | 1.0264 | |
| nearest-grid / best-training-h | 1 | |

Interpretation: the formula lands far above the current INT4 training-optimal h. The low-bit displacement visibility metric keeps improving out to 1e-2, but training accuracy peaks around 1e-3, so INT4 needs an additional nonlocal/training-stability upper gate beyond visibility.
