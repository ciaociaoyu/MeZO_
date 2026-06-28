# Latest Main FP32/FP16 H-Sweep Summary

Experiment root: `/scratch/jy03364/MeZO_/v10_medium_supplement_results_20260627/raw_runs/high_precision_sst5_fp32_multiseed`
Completed runs: 0 / 0
Failed/incomplete runs: 0

## FP32
- Best by best_eval_acc: `None` h=`None` acc=`None`
- Best by last_eval_acc: `None` h=`None` acc=`None`

## FP16
- Best by best_eval_acc: `None` h=`None` acc=`None`
- Best by last_eval_acc: `None` h=`None` acc=`None`

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
