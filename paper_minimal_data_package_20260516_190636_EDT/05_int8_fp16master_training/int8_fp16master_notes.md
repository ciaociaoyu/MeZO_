# INT8-Forward + FP16-Master Training Notes

Use precise language: these are **INT8-forward + FP16-master** runs, not true INT8 update.

Sources include:

- `experiments/int8_update_sparse_plan/probe_window_h100_20260512/window_training_summary.csv`: 300-step early validation.
- `experiments/future_probe_window_training_20260512_235627_package/summary_all.csv`: seed0 2k h sweep / anchors.
- `experiments/int8_update_sparse_plan/next_round_window_sparse_20260516/dense_5k_runs.csv` and `experiments/int8_update_sparse_plan/next_round_window_sparse_20260516/dense_5k_by_h.csv`: 5k selected h=2e-3 and h=3e-3, seeds 0/1/2.

Interpretation:

- Estimated INT8 dense probe window: roughly `1e-3` to `3e-3`, with best probe correlation around `3e-3`.
- Inside-window training candidates: `h=2e-3` and `h=3e-3`.
- Outside / weak candidates: `h=3e-4` below-window and `h=1e-2` large/locality-failure side.
- `h=1e-2` is worse in the available seed0 2k package.
- Best acc and last acc can differ substantially; some runs peak and then collapse or drift. Use both best and last metrics.

5k aggregate by h from `experiments/int8_update_sparse_plan/next_round_window_sparse_20260516/dense_5k_by_h.csv`:

```text
{'h_raw': '0.002', 'num_seeds': '3', 'seeds': '0 1 2', 'mean_best_eval_acc': '0.471506635', 'std_best_eval_acc_population': '0.007175947', 'mean_last_eval_acc': '0.332552693', 'std_last_eval_acc_population': '0.077890126', 'mean_last_eval_loss': '1.493677139', 'best_single_run': 'dense_int8_fp16master_h2e-3_seed0_step5000', 'best_single_best_eval_acc': '0.480093677', 'best_single_last_run': 'dense_int8_fp16master_h2e-3_seed1_step5000', 'best_single_last_eval_acc': '0.440281030'}
{'h_raw': '0.003', 'num_seeds': '3', 'seeds': '0 1 2', 'mean_best_eval_acc': '0.460577674', 'std_best_eval_acc_population': '0.007175947', 'mean_last_eval_acc': '0.403590945', 'std_last_eval_acc_population': '0.063738203', 'mean_last_eval_loss': '1.420228044', 'best_single_run': 'dense_int8_fp16master_h3e-3_seed0_step5000', 'best_single_best_eval_acc': '0.470725995', 'best_single_last_run': 'dense_int8_fp16master_h3e-3_seed1_step5000', 'best_single_last_eval_acc': '0.455503513'}
```

Caveats:

- h=2e-3 and h=3e-3 have 3 seeds / 5k and are the strongest INT8 training evidence.
- h=1e-3 and h=1e-2 anchors are mainly seed0 / 2k in the copied data.
