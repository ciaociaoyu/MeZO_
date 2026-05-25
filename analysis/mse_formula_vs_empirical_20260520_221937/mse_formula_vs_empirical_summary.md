# Formula-predicted MSE vs empirical MSE

Output directory: `analysis/mse_formula_vs_empirical_20260520_221937`

Definitions: `old_h4_full_bound = Delta^2 G^2/(4h^2) + 2 Delta L G sqrt(d(d+2)) + 4 h^2 L^2 d(d+2)`. `strict_h6_S3 = Delta^2 G^2/(4h^2) + S3_sq h^4/36`.

## Summary

| setting | formula | actual_min_h | pred_min_h | pred_min_over_actual_min_h | pred_over_actual_at_actual_min | median_pred_over_actual | log10_curve_corr | spearman_rank_corr | scaled_log10_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32_seed16 | old_h4_full_bound | 1e-05 | 1e-05 | 1 | 131.3 | 229.6 | 0.9747 | 1 | 0.4785 |
| fp32_seed16 | strict_h6_S3 | 1e-05 | 3e-05 | 3 | 0.07444 | 0.1231 | 0.8968 | 0.9727 | 2.094 |
| fp16_seed16 | old_h4_full_bound | 0.0003 | 0.0001 | 0.3333 | 173.2 | 534.6 | 0.7196 | 0.7545 | 0.7218 |
| fp16_seed16 | strict_h6_S3 | 0.0003 | 0.001 | 3.333 | 1.656 | 0.5035 | 0.8079 | 0.7545 | 1.092 |
| int8_rtnclip_g128_seed16 | old_h4_full_bound | 0.0015 | 0.0003 | 0.2 | 273.4 | 228.5 | 0.7647 | 0.6455 | 0.6843 |
| int8_rtnclip_g128_seed16 | strict_h6_S3 | 0.0015 | 0.003 | 2 | 3.605 | 3.527 | 0.9487 | 0.7727 | 1.497 |

## Readout

- Old h4/L-bound predicts the minimum h too small for FP16/INT8, and its absolute MSE scale is badly over-conservative once h is above the lower-bound region. This is why L is not trustworthy as an absolute MSE predictor.
- Strict h6/S3 gives a better U-shaped mechanism and often a better order of h, but its raw MSE scale still misses by factors of a few and its minimum shifts high for FP16/INT8 current data.
- After allowing only one scalar rescale, curve-shape error is much smaller than raw error. That means the trend is partially right, but constants/L/S3 are not calibrated enough for a numerical MSE bound.
- A more reliable L should be estimated with clean FP32 Hessian-vector products or gradient-difference curvature, not loss second differences alone. Loss second differences are usable for h selection diagnostics, not for absolute MSE claims.
