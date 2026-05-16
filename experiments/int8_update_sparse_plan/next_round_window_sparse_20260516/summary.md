# Next-round INT8 window/sparse/residual result package

This package contains compact summaries, manifests, commands, job IDs, and script snapshots only. Raw logs and checkpoints are intentionally excluded.

## Source run directories
- `runs/next_round_window_sparse_20260515_163221_chunk8_l4` -> `cluster_dense_l4_first_chunk` (L4)
- `runs/next_round_window_sparse_20260515_205233_offset8_limit7_l4` -> `sparse_l4_offset8_limit7` (L4)
- `runs/next_round_window_sparse_20260516_010554_offset15_limit4_h100` -> `sparse_h100_offset15_limit4` (H100)
- `runs/next_round_window_sparse_20260516_010556_offset19_limit6_a100` -> `sparse_a100_offset19_limit6` (A100)
- `runs/next_round_residual_local_20260515_163229` -> `residual_local_h100` (local H100)

## Job IDs and GPU groups
- L4 first chunk: `45295771` for source cases 0-7, eventually throttled to 7 live tasks.
- L4 second chunk: `45306349` for source cases 8-14, throttled to 7 live tasks.
- H100 split: `45317863` for source cases 15-18, throttled to 4 live tasks.
- A100 split: `45317864` for source cases 19-24, throttled to 6 live tasks.
- Local residual: sequential local H100 run, no SLURM job ID.

## Coverage
- Combined rows: 28
- Dense rows: 7 total, 6 are 5k promoted rows.
- Sparse screening rows: 18 / 18.
- Residual rows: 3 / 3.

## Dense 5k result
- Best by mean peak accuracy: h=`0.002`, mean best_eval_acc=`0.471506635`.
- Best by mean last accuracy: h=`0.003`, mean last_eval_acc=`0.403590945`.
- Interpretation: h=2e-3 has the higher peak mean; h=3e-3 has better last-step stability in this 5k batch.

## Sparse 500-step screening result
- Best by last accuracy: `sparse_int8_fp16master_p0p01_hactive0p006_lr1e-5_seed0_step500`, p=`0.01`, h_active=`0.006`, lr=`1e-05`, last_eval_acc=`0.3793911007025761`.
- Best by peak accuracy: `sparse_int8_fp16master_p0p01_hactive0p006_lr1e-5_seed0_step500`, p=`0.01`, h_active=`0.006`, lr=`1e-05`, best_eval_acc=`0.3793911007025761`.
- Interpretation: lr=1e-5 is the cleanest sparse setting. lr=3e-5 often reaches a peak then degrades by the last eval.

## Residual local result
- Best residual run: `residual_grid_round_step1_lr7e-5_clip3_step500`, best_eval_acc=`0.3711943793911007`, last_eval_loss=`1.5153218507766724`.
- Residual geometry stayed clean: residual_bound_violation_frac_last=0 and scale_drift_max=0 for all three runs.

## Included aggregate files
- `all_summary.csv`
- `dense_all.csv`
- `dense_5k_runs.csv`
- `dense_5k_by_h.csv`
- `sparse_screen_all.csv`
- `sparse_screen_by_setting_aggregate.csv`
- `residual_all.csv`
- `residual_by_run.csv`

## Recommendation
- Promote dense h=3e-3 for stability-oriented 10k-20k validation; keep h=2e-3 as the peak-accuracy comparator.
- Promote sparse p=0.01, h_active=0.006, lr=1e-5 first; p=0.01, h_active=0.012, lr=1e-5 is the second sparse candidate.
- Keep residual_grid as a secondary diagnostic/backend unless a longer run shows a clear accuracy gain.
