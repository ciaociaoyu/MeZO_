# INT8 h-star dequantized-scale estimate

Output directory: `analysis/int8_hstar_dequant_estimate_20260520_202220`

Method: use existing INT8 fake-quant/dequant probe artifacts. Delta is derived from dequantized reconstruction MSE as `sqrt(12 * MSE)` (grid-step RMS under a uniform rounding-error model). G_fd is the median abs-G over stable INT8 probe h in `[1e-3, 1.5e-3, 2e-3, 3e-3]`. L and S3 are clean FP32 geometry proxies from the seed16 initial state; this is formula calculation, not new training.

## Key components

- RTNClip/G128 Delta scale RMS from dequant recon MSE: `0.000934923`
- groupwise256 representative Delta scale RMS from dequant recon MSE: `0.00203592`
- INT8 stable G_fd median: `15.5128`
- clean32 proxy G_true: `13.3596`
- clean32 L q90: `3.50666e-05`
- clean32 S3_sq: `1.11085e+12`
- clean32 rho3_q90: `2.12355e-07`
- all trainable d: `407938142`
- RTNClip quantized-value d: `355563520`

## Main RTNClip/G128 formulas with G_fd

| oracle_source | formula_name | hstar_cont | empirical_min_nmse_h | h_over_empirical_min_nmse_h | empirical_min_nmse_h_over_h |
| --- | --- | --- | --- | --- | --- |
| current_rtnclip_g128_reprobe | old_h4_Lsmooth_all_trainable | 0.000503453 | 0.0015 | 0.335635 | 2.97942 |
| current_groupwise256_main_latest | old_h4_Lsmooth_all_trainable | 0.000503453 | 0.0015 | 0.335635 | 2.97942 |
| older_probe_window_h100_20260512 | old_h4_Lsmooth_all_trainable | 0.000503453 | 0.003 | 0.167818 | 5.95885 |
| current_rtnclip_g128_reprobe | new_h6_rho3_gate_all_trainable | 0.0013002 | 0.0015 | 0.866799 | 1.15367 |
| current_groupwise256_main_latest | new_h6_rho3_gate_all_trainable | 0.0013002 | 0.0015 | 0.866799 | 1.15367 |
| older_probe_window_h100_20260512 | new_h6_rho3_gate_all_trainable | 0.0013002 | 0.003 | 0.433399 | 2.30734 |
| current_rtnclip_g128_reprobe | old_h4_Lsmooth_rtnclip_quantized_values | 0.000539259 | 0.0015 | 0.359506 | 2.78159 |
| current_groupwise256_main_latest | old_h4_Lsmooth_rtnclip_quantized_values | 0.000539259 | 0.0015 | 0.359506 | 2.78159 |
| older_probe_window_h100_20260512 | old_h4_Lsmooth_rtnclip_quantized_values | 0.000539259 | 0.003 | 0.179753 | 5.56319 |
| current_rtnclip_g128_reprobe | new_h6_rho3_gate_rtnclip_quantized_values | 0.00139267 | 0.0015 | 0.928446 | 1.07707 |
| current_groupwise256_main_latest | new_h6_rho3_gate_rtnclip_quantized_values | 0.00139267 | 0.0015 | 0.928446 | 1.07707 |
| older_probe_window_h100_20260512 | new_h6_rho3_gate_rtnclip_quantized_values | 0.00139267 | 0.003 | 0.464223 | 2.15414 |
| current_rtnclip_g128_reprobe | new_h6_S3_empirical_third_moment | 0.00307904 | 0.0015 | 2.05269 | 0.487165 |
| current_groupwise256_main_latest | new_h6_S3_empirical_third_moment | 0.00307904 | 0.0015 | 2.05269 | 0.487165 |
| older_probe_window_h100_20260512 | new_h6_S3_empirical_third_moment | 0.00307904 | 0.003 | 1.02635 | 0.974331 |

## Interpretation

- Against the current RTNClip/G128 and main groupwise256 probe reference, empirical min-nMSE h is `1.5e-3`. The old h4 formula gives about `5.0e-4` with all trainable d, or `5.4e-4` with RTNClip quantized-value d, so it is about `2.8x-3.0x` too small.
- The strict S3 h6 formula gives about `3.08e-3` with RTNClip Delta/G_fd: close to the older INT8 probe optimum `3e-3`, but about `2.05x` larger than the current RTNClip/groupwise256 empirical min.
- The conservative rho3 gate h6 version gives `1.30e-3` (all trainable d) or `1.39e-3` (RTNClip quantized-value d), which is closest to the current `1.5e-3` oracle.
- So for INT8, old h4 is still biased low. New S3 h6 moves into the right order, and the rho3 gate variant is the best match to the current dequantized INT8 probe reference.
