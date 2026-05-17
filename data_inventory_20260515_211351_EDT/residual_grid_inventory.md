# Residual Grid Inventory

## Scope

Residual-grid is treated here as a secondary update-commit diagnostic/backend, not as the main perturbation-visibility contribution.

## Pre-fix Versus Post-fix

- Pre-fix evidence: `runs/int8_residual_round2_20260512_180500/summary.csv` shows residual-over-scale values inflated by near-zero frozen scales; use as diagnostic only.
- Post-fix evidence: `experiments/int8_update_sparse_plan/results/residual_consistency_20260512_190000_key_results.tar.gz` includes corrected scale initialization/checks, zero grid error, and zero scale drift in the listed summaries.
- Later H100 residual package: `results_packages/residual_local_h100_20260515_172944_essential.tar.gz` includes sanity checks plus 500-step runs and one 2k promoted residual run.

## Consistency Diagnostics

| check | status | evidence |
| --- | --- | --- |
| no-op | available | `noop_update_check.csv`, no-op run summaries, sanity_summary.json |
| pre-snap/post-snap grid | available | `grid_stats.csv`, grid_error after snap = 0 in summaries |
| synthetic residual | available | `synthetic_residual_test.json` passes round/floor/stochastic |
| one-step equation check | available | `one_step_equation_check.csv`, max_abs_q_diff = 0 for selected tensors |
| scale drift | available | `scale_drift_summary.csv`, `scale_drift_last_step.csv`, and summary fields |
| residual-bound violation | available | summary fields report zero violation for post-fix round-mode runs |
| per-layer update stats | available compactly | per-layer last-step CSV in key archive; full raw per-layer logs intentionally excluded there |

## First-Round / Pre-fix Short Results

| run | backend | steps | best_acc | final_acc | cos_last | norm_ratio_last | residual_p99_last | grid_error | nan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_int8_lr1e-5 | direct_int8 | 51 | 0.282201 | 0.282201 | 0.0291871 | 34.2319 | 0 | 0 | False |
| noop_residual_grid_lr0 | residual_grid | 2 | 0.28103 | 0.28103 |  |  | 0 | 0 | False |
| residual_grid_lr1e-4 | residual_grid | 51 | 0.289227 | 0.289227 | 0.3324 | 3.00792 | 1.811e+04 | 0 | False |
| residual_grid_lr1e-5 | residual_grid | 51 | 0.284543 | 0.284543 | 0.110731 | 2.68405 | 2224.27 | 0 | False |
| residual_grid_lr3e-5 | residual_grid | 51 | 0.296253 | 0.296253 | 0.221903 | 4.08492 | 6359.09 | 0 | False |
| residual_grid_round_step1_lr1e-4_clip5 | residual_grid | 51 | 0.298595 | 0.298595 | 0.251889 | 3.02414 | 4298.78 | 0 | False |
| residual_grid_stoch_step1_lr1e-4_clip5 | residual_grid | 51 | 0.302108 | 0.302108 | 0.0402686 | 24.8136 | 4501.89 | 0 | False |
| residual_grid_stoch_step1_lr3e-4_clip10 | residual_grid | 51 | 0.298595 | 0.298595 | 0.165847 | 5.87151 | 9133.27 | 0 | False |
| residual_grid_stoch_step1_lr3e-4_clip5 | residual_grid | 51 | 0.297424 | 0.297424 | 0.091536 | 10.8527 | 4548.54 | 0 | False |

## Post-fix / Short Residual Results

| run | steps | best_eval_acc | last_eval_acc | lr | cos_last | norm_ratio_last | violation | grid_error | scale_drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_grid_round_lr3e-5_step500 | 501 | 0.309133 | 0.309133 | 3e-05 | 0.0726682 | 13.7647 | 0 | 0 | 0 |
| residual_grid_round_step1_lr1e-4_clip5_step500 | 501 | 0.368852 | 0.332553 | 0.0001 | 0.130911 | 7.63645 | 0 | 0 | 0 |
| residual_grid_stoch_step1_lr3e-4_clip10_step500 | 501 | 0.272834 | 0.271663 | 0.0003 | 0.165258 | 5.89287 | 0 | 0 | 0 |

## Longer Residual Evidence

The residual local H100 essential package contains a 2k promoted residual run:

| run | steps | best_eval_acc | last_eval_acc | lr | clip | cos_last | norm_ratio_last | violation | grid_error | scale_drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_grid_round_step1_lr1e-4_clip2_step500 | 501 | 0.312646 | 0.312646 | 0.0001 | 2 | 0.163656 | 5.46288 | 0 | 0 | 0 |
| residual_grid_round_step1_lr5e-5_clip5_step500 | 501 | 0.367681 | 0.367681 | 5e-05 | 5 | 0.140206 | 7.132 | 0 | 0 | 0 |
| residual_grid_round_step1_lr7e-5_clip3_step500 | 501 | 0.371194 | 0.371194 | 7e-05 | 3 | 0.203144 | 4.8364 | 0 | 0 | 0 |
| residual_grid_round_step1_lr7e-5_clip3_step500_promote_step2000 | 2001 | 0.462529 | 0.462529 | 7e-05 | 3 | 0.203599 | 4.91229 | 0 | 0 | 0 |

## Evidence Use

- Use post-fix consistency archives for update-commit distortion and clean residual mechanics claims.
- Use pre-fix residual-over-scale anomalies only as diagnostic history, not final evidence.
- The 2k promoted residual run is stronger than a pure 50/500-step check but still short of a long-run backend claim.
- Residual-grid belongs in appendix or secondary analysis unless longer multi-seed evidence is added.
