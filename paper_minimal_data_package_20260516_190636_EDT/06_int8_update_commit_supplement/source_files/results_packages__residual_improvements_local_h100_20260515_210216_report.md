# Residual-Grid Improvement Report

Run root: `/scratch/jy03364/MeZO_/runs/residual_improvements_local_h100_20260515_210216`

## Commands

All exact launched commands are recorded verbatim in:

`/scratch/jy03364/MeZO_/runs/residual_improvements_local_h100_20260515_210216/commands.txt`

This file contains 10 commands: real-model debug, 1-step training smoke, 7 sequential 500-step matrix runs, and 1 promoted 2k run. All training commands used `CUDA_VISIBLE_DEVICES=0` and local H100 only.

## Code Changes

- `medium_models/src/int8_residual_grid.py`
  - Added EF-aware diagnostics against accumulated `acc = residual + delta`: `acc_actual_cos`, `actual_over_acc_norm_ratio`, `ef_error_norm`, `ef_error_max`, residual before/after norms.
  - Added `residual_commit_threshold`, `top_abs_acc`, `norm_budget`, and `residual_decay`.
  - Kept default behavior unchanged: threshold `0`, selection `all`, decay `1.0`, tensor scale.
  - Added tensor-only scale-mode scaffolding; channel/block intentionally raise until implemented.
- `medium_models/src/trainer.py`
  - Wired new residual arguments into `ResidualGridUpdater`.
  - Aggregated global and per-layer EF-aware metrics.
  - Extended debug one-step CSV with old delta metrics and new acc metrics.
- `medium_models/run.py`
  - Added CLI/config validation for new residual arguments.
- `medium_models/tests/test_residual_grid_update.py`
  - Added synthetic checks for threshold, top_abs_acc, norm_budget, no-op EF conservation.
- `scripts/summarize_residual_improvements.py`
  - Added summary CSV/MD and plots for old/new geometry metrics.

## Sanity Checks

- `python -m py_compile medium_models/src/int8_residual_grid.py medium_models/src/trainer.py medium_models/run.py scripts/summarize_residual_improvements.py`: passed. Existing trainer docstring escape warning only.
- `python -m unittest medium_models.tests.test_residual_grid_update`: 10 tests passed.
- Real-model debug one-step:
  - Path: `debug_real_model_one_step/debug/`
  - Synthetic residual tests passed for round/floor/stochastic.
  - No-op check was inactive.
  - One-step CSV includes `cos_delta_actual`, `cos_acc_actual`, `actual_over_delta_norm_ratio`, `actual_over_acc_norm_ratio`, `ef_error_norm`, `ef_error_max`.
- 1-step training smoke wrote global update stats with `global_ef_error_norm=0`, `grid_error_norm=0`.

## 500-Step Results

| run_name | best_acc | last_acc | last_loss | acc_cos_last | actual/acc_last | delta_cos_last | actual/delta_last | selected_active_last | clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| residual_best_current_all_step500 | 0.37119 | 0.37119 | 1.51532 | 0.26174 | 0.50238 | 0.20314 | 4.83640 | 0.04078 | yes |
| residual_threshold0p75_step500 | 0.34192 | 0.33607 | 1.52551 | 0.09477 | 0.12178 | 0.09037 | 1.31871 | 0.00741 | yes |
| residual_threshold1p0_step500 | 0.35246 | 0.35246 | 1.52423 | 0.07137 | 0.06925 | 0.06840 | 0.91917 | 0.00417 | yes |
| residual_topacc_active0p02_step500 | 0.36417 | 0.36417 | 1.52269 | 0.23703 | 0.45096 | 0.16357 | 4.35770 | 0.01870 | yes |
| residual_topacc_active0p04_step500 | 0.37119 | 0.37119 | 1.51436 | 0.25816 | 0.49526 | 0.18518 | 4.76708 | 0.02728 | yes |
| residual_normbudget_acc_cap2_step500 | 0.37119 | 0.37119 | 1.51532 | 0.26174 | 0.50238 | 0.20314 | 4.83640 | 0.04078 | yes |
| residual_normbudget_acc_cap3_step500 | 0.37119 | 0.37119 | 1.51532 | 0.26174 | 0.50238 | 0.20314 | 4.83640 | 0.04078 | yes |

Clean means `grid_error_norm=0`, `scale_drift_max=0`, `residual_bound_violation_frac=0`, and `ef_error_norm=0`.

## Interpretation

EF-aware metrics show the current residual-grid commit is better aligned with accumulated `acc` than with current-step `delta`.

For the all-commit baseline at 500 steps:

- Old delta metric: `cos_delta_actual=0.203`, `actual/delta=4.836`.
- New acc metric: `cos_acc_actual=0.262`, `actual/acc=0.502`.

The old metric is still useful for current-step distortion, but it overstates the error-feedback commit distortion because the commit target is accumulated residual state, not just the current perturbation delta.

Thresholding at `0.75` and `1.0` reduced `actual/acc` strongly, but it also reduced `acc_actual_cos`, active fraction, and accuracy. These thresholds are too conservative for training as configured.

`top_abs_acc` with target active `0.04` was the best budgeted result: it matched baseline 500-step accuracy, slightly improved 500-step eval loss, and reduced selected active fraction from about `0.0408` to `0.0273`.

`norm_budget` with cap `2` or `3` was effectively a no-op under the new acc reference because actual/acc was already well below those caps.

## Promotion

Selected for 2k promotion:

`residual_topacc_active0p04_step500_promote_step2000`

Reason:

- It was the only non-no-op 500-step variant matching all-commit accuracy.
- It reduced selected active fraction while keeping acc-aware geometry close to baseline.
- Its diagnostics were clean.

2k result:

| run_name | best_acc | last_acc | best_loss | last_loss | acc_cos_last | actual/acc_last | selected_active_last | clean |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| residual_topacc_active0p04_step500_promote_step2000 | 0.45902 | 0.45902 | 1.38946 | 1.38946 | 0.23926 | 0.46174 | 0.02473 | yes |

This is close to, but does not beat or match, the current all-commit residual 2k reference of about `0.46253`. Therefore seeds 1 and 2 were not launched.

## Backend Decision

Residual-grid should remain a diagnostic and secondary backend for now.

The EF-aware diagnostics make the backend look mechanically healthier than the old delta-only metrics suggested, and `top_abs_acc=0.04` is a promising budgeted commit variant. But the best longer-run accuracy still belongs to all-commit residual-grid, and the budgeted variant did not beat the current 2k reference.

## Recommended Next Action

Run one future 5k candidate only if residual-grid remains worth spending H100 time:

- `top_abs_acc`, `residual_target_active_frac=0.04`, seed 0, same h/lr/clip.

Also consider a narrower norm-budget follow-up, because cap `2` and `3` were no-ops:

- `norm_budget`, `residual_budget_reference=acc`, cap around `0.45` to `0.55`.

Do not prioritize block-wise scale next. The current tensor-scale implementation is mechanically clean; block/channel scale should wait until a budgeted commit setting shows stronger long-run training value.
