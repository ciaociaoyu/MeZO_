# INT4 SST-5 calibrated h-star check

Output directory: `analysis/int8_sst5_calibrated_hstar_20260521_newmethod`

Selector tested: `calibrated_hstar_absG_Lclean32_q90` adapted to RTNClip INT4 by using INT4 group-scale RMS as Delta.

| quantity | value |
|---|---:|
| Delta INT4 scale RMS | 0.00093823 |
| Delta scale RMS / sqrt(6) | 0.000383031 |
| Delta empirical snap RMS at h=1e-3 | 0.000392689 |
| G absG | 14.2283 |
| G clean32 absG at h=3e-4 | 15.7166 |
| h_G | 0.001 |
| G selection | manual_int4_primary_hG_1e-3 |
| L clean32 q90 | 2.36719e-05 |
| h2_L | 1e-05 |
| L selection | plateau_q90_primary |
| d trainable | 407938142 |
| hstar_cont | 0.000587878 |
| nearest grid h | 0.001 |

## Variant h-star rows

| Delta mode | G mode | hstar | nearest h | hstar / best-training-h |
|---|---|---:|---:|---:|
| raw_scale_rms | int4_absG_h1e-3 | 0.000587878 | 0.001 | 0.587878 |
| raw_scale_rms | clean32_absG_h0.0003 | 0.000617859 | 0.001 | 0.617859 |
| raw_scale_rms | clean32_absG_median_1e-4_3e-4_1e-3 | 0.000614042 | 0.001 | 0.614042 |
| scale_rms_over_sqrt6 | int4_absG_h1e-3 | 0.000375621 | 0.0003 | 0.375621 |
| scale_rms_over_sqrt6 | clean32_absG_h0.0003 | 0.000394777 | 0.0003 | 0.394777 |
| scale_rms_over_sqrt6 | clean32_absG_median_1e-4_3e-4_1e-3 | 0.000392338 | 0.0003 | 0.392338 |
| empirical_snap_rms_h1e-3 | int4_absG_h1e-3 | 0.000380327 | 0.0003 | 0.380327 |
| empirical_snap_rms_h1e-3 | clean32_absG_h0.0003 | 0.000399723 | 0.0003 | 0.399723 |
| empirical_snap_rms_h1e-3 | clean32_absG_median_1e-4_3e-4_1e-3 | 0.000397254 | 0.0003 | 0.397254 |
| empirical_snap_rms_first_lowbit_pass | int4_absG_h1e-3 | 0.000341054 | 0.0003 | 0.341054 |
| empirical_snap_rms_first_lowbit_pass | clean32_absG_h0.0003 | 0.000358447 | 0.0003 | 0.358447 |
| empirical_snap_rms_first_lowbit_pass | clean32_absG_median_1e-4_3e-4_1e-3 | 0.000356233 | 0.0003 | 0.356233 |

## Comparison to current INT4 sweep

| reference | h | metric |
|---|---:|---:|
| sweep best best_eval_acc | 0.001 | 0.476581 |
| sweep best last_eval_acc | 0.001 | 0.476581 |
| sweep best lowbit visibility nMSE | 0.01 | nan |
| hstar_cont / best-training-h | 0.587878 | |
| nearest-grid / best-training-h | 1 | |

Interpretation: the formula lands far above the current INT4 training-optimal h. The low-bit displacement visibility metric keeps improving out to 1e-2, but training accuracy peaks around 1e-3, so INT4 needs an additional nonlocal/training-stability upper gate beyond visibility.
