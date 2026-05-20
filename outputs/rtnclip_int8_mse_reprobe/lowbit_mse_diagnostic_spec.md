# Low-Bit MSE Diagnostic Field Spec

Low-bit ZO diagnostics must not use a single generic `mse` field. The fields below have separate meanings and must not be substituted for each other.

## Weight Reconstruction

- `weight_recon_mse`: mean squared error of `Q_t(w_t)` versus the unperturbed FP16 master `w_t`.
- `weight_recon_rel_mse`: pooled squared error divided by pooled `||w_t||^2`.
- `weight_recon_sqnr_db`: `10 log10(||w_t||^2 / ||Q_t(w_t)-w_t||^2)`.
- Legacy `recon_mse_global` may be copied into `weight_recon_mse`, but it is not an h-window finite-difference MSE.

## Perturbed Reconstruction

- `plus_recon_mse`, `minus_recon_mse`: reconstruction MSE of `Q_t(w_t +/- h u)` against each perturbed floating state.
- `plus_recon_rel_mse`, `minus_recon_rel_mse`: pooled relative versions of the same diagnostics.

## Effective Displacement / Visibility

- `delta_visibility_mse`: MSE of `Q_t(w_t+h u)-Q_t(w_t-h u)` against `2h u`.
- `delta_visibility_nmse`: pooled normalized version of the same visibility error.
- `delta_visibility_rel_l2`: relative L2 visibility error.
- `alignment`, `norm_ratio`, `code_change_frac`, `active_frac`, `clip_frac`, and `saturation_frac` describe quantization geometry.

`delta_visibility_nmse` is not truncation error. Large `h` can have low visibility error while still being a poor finite-difference estimator.

## True-Gradient Finite-Difference Quality

- `fd_true_mse`: pooled MSE of `d_h^Q(u)` versus the unquantized true directional derivative `grad L(w_t)^T u`.
- `fd_true_nmse`: pooled normalized MSE against the true derivative energy.
- `fd_true_rmse`: square root of `fd_true_mse`.
- `corr_fd_true`: correlation across directions between quantized finite differences and true directional derivatives.
- `fd_true_bias`: mean `d_h^Q(u)-d_true(u)`.
- `fd_true_available`: false when true gradients are unavailable or OOM. Do not fill these fields from reconstruction metrics.

## Richardson / Locality

- `richardson_absdiff`: mean absolute difference between `d_h^Q` and `d_{h/2}^Q`.
- `richardson_rmse_rel`: pooled relative RMSE of `d_h^Q-d_{h/2}^Q` normalized by `d_{h/2}^Q`.
- `richardson_relerr`: median per-direction relative difference.
- `richardson_available`: false when the paired `h/2` probe is unavailable.

Richardson metrics and true-gradient metrics are the locality diagnostics. Weight reconstruction MSE and delta visibility nMSE must not be used as h-window truncation MSE.
