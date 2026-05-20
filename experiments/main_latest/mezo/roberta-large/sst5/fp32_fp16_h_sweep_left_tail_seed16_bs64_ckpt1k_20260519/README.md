# latest_main_roberta_sst5_fp32_fp16_left_tail_seed16_bs64_ckpt1k

Left-tail extension for the latest clean RoBERTa-large / SST-5 FP32+FP16 MeZO h-sweep.

- Parent context: `/scratch/jy03364/MeZO_/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517`.
- Scope: FP32 and FP16 only, dense MeZO directions only.
- New h values: 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6.
- Dataset: full SST-5 (`dataset_mode=full`), seed=16, data_seed=16.
- Dataloader shuffle: explicitly enabled by `DATALOADER_SHUFFLE=True`.
- Batch size: 64, gradient accumulation: 1, lr: 1e-06.
- Checkpoints: every 1000 steps plus final, best_acc, best_loss.
- Guards disabled: random prediction guard and ZO probe health guard.
- Scheduling: 4 lane jobs maximum; each lane owns one GPU and runs rows serially.
- Excluded: INT8, INT4, BF16, sparse, residual-grid, LoRA, RTE, OPT, other datasets/models.
