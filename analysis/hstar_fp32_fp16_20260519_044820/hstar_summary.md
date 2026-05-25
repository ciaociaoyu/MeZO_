# Offline H-Star Analysis: FP32/FP16 RoBERTa-large SST-5

Analysis directory: `/scratch/jy03364/MeZO_/analysis/hstar_fp32_fp16_20260519_044820`

This is an offline analysis over cached checkpoint probe JSONL files. It did not launch training.

## Main Selector Table

| precision | checkpoint | selector | hstar_cont | nearest h | nmse | corr | empirical best h | empirical best nmse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fp16 | initial_cached_step0 | old_hstar_absG_deltaUlp_Lq90 | 3.541e-05 | 3e-05 | 0.01644 | 0.9927 | 0.0003 | 0.002974 |
| fp16 | initial_cached_step0 | old_hstar_codeG_deltaUlp_Lq90 | 3.326e-05 | 3e-05 | 0.01644 | 0.9927 | 0.0003 | 0.002974 |
| fp16 | initial_cached_step0 | old_hstar_richardsonG_deltaUlp_Lq90 | 3.483e-05 | 3e-05 | 0.01644 | 0.9927 | 0.0003 | 0.002974 |
| fp32 | initial_cached_step0 | old_hstar_absG_deltaUlp_Lq90 | 1.562e-06 | 1e-05 | 1.553e-06 | 1 | 1e-05 | 1.553e-06 |
| fp32 | initial_cached_step0 | old_hstar_codeG_deltaUlp_Lq90 | 1.535e-06 | 1e-05 | 1.553e-06 | 1 | 1e-05 | 1.553e-06 |
| fp32 | initial_cached_step0 | old_hstar_richardsonG_deltaUlp_Lq90 | 1.816e-06 | 1e-05 | 1.553e-06 | 1 | 1e-05 | 1.553e-06 |

## Interpretation

- The saved checkpoint probes cover `checkpoint_step=0` for every h. Step-1000/final trajectory probe curves were not cached, so they are listed as skipped rather than recomputed.
- `codeG` is the existing two-point estimator formula found in `medium_models/src/trainer.py`: `sqrt(pi/2) * mean(abs(d_hat))`; it is not a signed mean.
- `absG` adds an h-stability selector around the same absolute-moment G estimator.
- `richardsonG` uses cached `d2(h)` and `d2(2h)` pairs to form `(4*d2(h)-d2(2h))/3`; this is empirical and separate from the old theorem.
- `L` uses a cached symmetric-curvature proxy `(loss_plus - 2*loss_base + loss_minus)/h^2` because the shared-step `theta+delta, theta+2delta` losses were not cached.
- Continuous h-star values outside the grid are clipped only for the `hstar_clipped` column; the quality table reports nearest-grid cached MSE/corr.

## Warnings
- fp32/initial_cached_step0: no_on_disk_step0_checkpoint; using reference h=1e-3 step_1000 checkpoint for Delta ULP only
- fp32/initial_cached_step0: empirical_snap_rms skipped because cached probes do not contain per-coordinate actual_delta-intended_delta
- fp16/initial_cached_step0: no_on_disk_step0_checkpoint; using reference h=1e-3 step_1000 checkpoint for Delta ULP only
- fp16/initial_cached_step0: empirical_snap_rms skipped because cached probes do not contain per-coordinate actual_delta-intended_delta

## Output Files

- `hstar_estimates.csv`
- `hstar_grid_mse.csv`
- `hstar_diagnostics.json`
- `plots/`
