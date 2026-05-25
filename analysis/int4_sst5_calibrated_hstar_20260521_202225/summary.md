# INT4 SST-5 calibrated h-star check

Output directory: `/scratch/jy03364/MeZO_/analysis/int4_sst5_calibrated_hstar_20260521_202225`

Selector tested: `calibrated_hstar_absG_Lclean32_q90` adapted to RTNClip INT4 by using INT4 group-scale RMS as Delta.

| quantity | value |
|---|---:|
| Delta INT4 scale RMS | 0.0146615 |
| Delta scale RMS / sqrt(6) | 0.00598553 |
| Delta empirical snap RMS at h=1e-3 | 0.00399205 |
| G absG | 42.7614 |
| G clean32 absG at h=3e-4 | 15.7166 |
| h_G | 0.001 |
| G selection | manual_int4_primary_hG_1e-3 |
| L clean32 q90 | 2.36719e-05 |
| h2_L | 1e-05 |
| L selection | plateau_q90_primary |
| d trainable | 407938142 |
| hstar_cont | 0.00402876 |
| nearest grid h | 0.004 |

## Variant h-star rows

| Delta mode | G mode | hstar | nearest h | hstar / best-training-h |
|---|---|---:|---:|---:|
| raw_scale_rms | int4_absG_h1e-3 | 0.00402876 | 0.004 | 4.02876 |
| raw_scale_rms | clean32_absG_h0.0003 | 0.00244244 | 0.002 | 2.44244 |
| raw_scale_rms | clean32_absG_median_1e-4_3e-4_1e-3 | 0.00242735 | 0.002 | 2.42735 |
| scale_rms_over_sqrt6 | int4_absG_h1e-3 | 0.00257415 | 0.003 | 2.57415 |
| scale_rms_over_sqrt6 | clean32_absG_h0.0003 | 0.00156058 | 0.0015 | 1.56058 |
| scale_rms_over_sqrt6 | clean32_absG_median_1e-4_3e-4_1e-3 | 0.00155094 | 0.0015 | 1.55094 |
| empirical_snap_rms_h1e-3 | int4_absG_h1e-3 | 0.00210223 | 0.002 | 2.10223 |
| empirical_snap_rms_h1e-3 | clean32_absG_h0.0003 | 0.00127448 | 0.0015 | 1.27448 |
| empirical_snap_rms_h1e-3 | clean32_absG_median_1e-4_3e-4_1e-3 | 0.00126661 | 0.0015 | 1.26661 |
| empirical_snap_rms_first_lowbit_pass | int4_absG_h1e-3 | 0.00210223 | 0.002 | 2.10223 |
| empirical_snap_rms_first_lowbit_pass | clean32_absG_h0.0003 | 0.00127448 | 0.0015 | 1.27448 |
| empirical_snap_rms_first_lowbit_pass | clean32_absG_median_1e-4_3e-4_1e-3 | 0.00126661 | 0.0015 | 1.26661 |

## Comparison to current INT4 sweep

| reference | h | metric |
|---|---:|---:|
| sweep best best_eval_acc | 0.001 | 0.478923 |
| sweep best last_eval_acc | 0.001 | 0.478923 |
| sweep best lowbit visibility nMSE | 0.01 | 0.100231 |
| hstar_cont / best-training-h | 4.02876 | |
| nearest-grid / best-training-h | 4 | |

Interpretation: the formula lands far above the current INT4 training-optimal h. The low-bit displacement visibility metric keeps improving out to 1e-2, but training accuracy peaks around 1e-3, so INT4 needs an additional nonlocal/training-stability upper gate beyond visibility.
