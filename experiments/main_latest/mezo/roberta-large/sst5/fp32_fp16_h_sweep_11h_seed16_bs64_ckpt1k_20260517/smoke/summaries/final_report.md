# Latest Main FP32/FP16 H-Sweep Summary

Experiment root: `/scratch/jy03364/MeZO_/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517/smoke`
Completed runs: 2 / 2
Failed/incomplete runs: 0

## FP32
- Best by best_eval_acc: `smoke_fp32_h1e-3_seed16_bs64_ckpt25` h=`0.001` acc=`0.2775175644028103`
- Best by last_eval_acc: `smoke_fp32_h1e-3_seed16_bs64_ckpt25` h=`0.001` acc=`0.2775175644028103`

## FP16
- Best by best_eval_acc: `smoke_fp16_h1e-3_seed16_bs64_ckpt25` h=`0.001` acc=`0.24824355971896955`
- Best by last_eval_acc: `smoke_fp16_h1e-3_seed16_bs64_ckpt25` h=`0.001` acc=`0.24824355971896955`

## Contract Checks
- Scope is FP32/FP16 only; no INT8/INT4/sparse/residual-grid runs are in the manifest.
- Manifest uses RoBERTa-large and full SST-5 only.
- Launcher sets `DATALOADER_SHUFFLE=True`; run logs should contain RandomSampler lines from the Trainer override.
- FP16 rows use `precision_mode=fp16` and `zo_quantization=fp16`; BF16 is not used.

## Files
- `summaries/summary_all.csv`
- `summaries/checkpoint_inventory.csv`
- `summaries/failed_or_incomplete_runs.csv`
- `plots/plot_training_acc_vs_h.csv`
- `plots/plot_training_loss_vs_h.csv`
- `plots/plot_probe_vs_h.csv`
- `plots/plot_mse_vs_acc.csv`
