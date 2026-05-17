# Groupwise-256 INT8 vs Previous INT8 Summary

Exact GPTQ was not available. This directory is a GPTQ-256-requested rerun, but the actual quantizer used was the explicitly labeled fallback `groupwise_int8_block256`.

## Probe Window

The probe-window conclusion is robust. Previous INT8 runs indicated useful signal around `h=2e-3` to `3e-3`, with very small h distorted and `h=1e-2` failing by locality. The groupwise/block256 rerun shifts the strongest point slightly lower:

- best h by `corr_fd_true`: `1.5e-3`, corr `0.990373`, nMSE `0.020274`
- `h=2e-3`: corr `0.989376`, nMSE `0.022124`
- `h=3e-3`: corr `0.976402`, nMSE `0.052459`
- `h=1e-2`: probe geometry looks active/aligned, but derivative quality fails, corr `0.21594`, nMSE `1.06311`

Interpretation: block/group size 256 does not remove the h-window; it makes the best point look closer to `1.5e-3` to `2e-3`.

## Dense FP16-Master

The representative dense FP16-master run at `h=2e-3` reached the best peak accuracy in this rerun:

- run: `dense_gw256_fp16master_h2e-3_step5000`
- best_eval_acc: `0.474239`
- best_eval_loss: `1.206379`
- last_eval_acc: `0.360656`
- last_eval_loss: `1.460353`

This supports the previous conclusion that FP16-master update is the strongest route by peak training signal, but the 5k run was unstable after its 2500-step peak.

## Direct INT8 Update

Direct INT8 remains distorted:

- run: `direct_gw256_h3e-3_lr1e-5_step100`
- active_frac: `0.986147`
- cos_intended_actual: `0.349846`
- actual_over_intended_norm_ratio: `2.85881`
- best_eval_acc: `0.284543`

Compared with previous direct tensor-INT8 distortion, the norm ratio is less extreme, but the update is still near-dense and the training result is poor.

## Residual-Grid

Residual-grid remains mechanically clean:

- run: `residual_grid_gw256_h3e-3_lr7e-5_clip3_step2000`
- best_eval_acc: `0.460187`
- last_eval_acc: `0.447307`
- active_frac: `0.129804`
- global_acc_actual_cos: `0.417574`
- global_actual_over_acc_norm_ratio: `0.717596`
- grid_error_norm: `0`
- scale_drift_max: `0`
- residual_bound_violation_frac: `0`

This matches the previous interpretation: residual-grid is useful as a diagnostic and secondary backend candidate, but not yet the main training route.

## Sparse

Sparse was only smoke-tested in this rerun. No full comparison should be made yet.

## Recommendation

Treat this as an appendix robustness ablation. It supports the main perturbation-visibility/h-window story under groupwise/block256 INT8, but it is not exact GPTQ and does not justify replacing the main quantizer setting.
