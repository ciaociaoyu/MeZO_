# Discovery Report

Generated: 2026-06-28T18:29:56
Git commit: `bf7d32db55fb58377c8f697d75727caf98a89618`

## Source

- Reused probe-only source directory: `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627`
- No new training was run.
- No model reload was required for this robust dual-scheme pass; it reuses saved raw probe metrics.

## Checkpoints Found/Selected

- fp32: `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627/checkpoints/task_start_seed16_deterministic_fp32_master.pt` (deterministic task-start model load)
- fp16: `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627/checkpoints/task_start_seed16_deterministic_fp16_master.pt` (deterministic task-start model load)
- int8: `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627/checkpoints/task_start_seed16_deterministic_int8_master.pt` (deterministic task-start model load)
- int4: `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627/checkpoints/task_start_seed16_deterministic_int4_master.pt` (deterministic task-start model load)

## Accuracy Sweep Data

- fp16: 15 h points; sources: `/scratch/jy03364/MeZO_/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_left_tail_seed16_bs64_ckpt1k_20260519/summaries/merged_fp32_fp16_h_sweep_1e-9_to_1e-2.csv`
- fp32: 17 h points; sources: `/scratch/jy03364/MeZO_/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_left_tail_seed16_bs64_ckpt1k_20260519/summaries/merged_fp32_fp16_h_sweep_1e-9_to_1e-2.csv`
- int4: 11 h points; sources: `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch_summary.csv`
- int8: 11 h points; sources: `/scratch/jy03364/MeZO_/outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch/int8_hsearch_summary.csv`

## Precision Modes

- FP32 and FP16: high-precision forward modes from existing sweep/probe.
- INT8 and INT4: G128 RTNClip shared-grid fake quantization on Linear.weight from the existing probe.

## Fallback Choices

- Practical windows are computed even when smooth rho fit is unstable.
- Pure plug-in windows use direct Delta/L estimates when available and otherwise emit unavailable status rows.
- FP32/FP16 Delta estimates are numeric visibility approximations, not hardware ulp claims.
