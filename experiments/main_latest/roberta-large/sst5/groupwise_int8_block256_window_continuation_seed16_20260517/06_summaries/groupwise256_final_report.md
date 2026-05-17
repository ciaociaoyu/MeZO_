# groupwise_int8_block256 Final Report

## 1. Actual Quantizer

This run uses `groupwise_int8_block256`: symmetric group-wise INT8 quantization with group/block size 256 and `calibration_samples=0`.

It is not exact GPTQ. No Hessian-based GPTQ calibration is used in this code path.

## 2. Dense Window

Best dense probe h by corr_fd_true: `0.0015` with corr `0.990373` and nMSE `0.0202737`.

h=1e-3: corr `0.98278`, nMSE `0.0335388`.
h=1.5e-3: corr `0.990373`, nMSE `0.0202737`.
h=2e-3: corr `0.989376`, nMSE `0.0221241`.
h=3e-3: corr `0.976402`, nMSE `0.0524592`.
h=1e-2: corr `0.21594`, nMSE `1.06311`.

Interpretation should treat h=1e-2 as unreliable if derivative correlation/locality is poor, even when probe geometry looks active.

## 3. Dense Training

Best dense FP16-master training row: `dense_groupwise256_fp16master_h1e-3_step5000` with best_eval_acc `0.487119` and last_eval_acc `0.25644`.

Late collapse should be judged from the gap between best_eval_acc and last_eval_acc plus last_eval_loss.

## 4. Residual Grid

Best residual-grid row: `residual_grid_groupwise256_h3e-3_lr7e-5_clip3_step2000` with best_eval_acc `0.447307`, last_eval_acc `0.447307`, grid_error_norm `0`, and scale_drift_max `0`.

Residual-grid remains mechanically clean only if grid_error_norm, scale_drift_max, and residual_bound_violation_frac stay zero or numerically negligible.

## 5. Sparse Rate

Best sparse probe row: p=`0.003`, h_active=`0.012`, corr `0.997417`, nMSE `0.00487749`.

Sparse training should only be started after this probe table points to a stable h_active/p pair.

## 6. Recommendation

Keep this as a robustness / quantizer-ablation setting unless repeated training confirms that the wider groupwise block-256 window improves stability, not just probe geometry.

If groupwise block-256 is promoted later, use the best dense-training h from this continuation rather than h=1e-2, because h=1e-2 remains a locality risk.
