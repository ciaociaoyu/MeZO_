# latest_main_roberta_sst5_fp32_fp16_hsweep_seed16_bs64_ckpt1k

Latest checkpointed main MeZO h-sweep for RoBERTa-large / SST-5.

- Scope: FP32 and FP16 only.
- Excluded: INT8, INT4, sparse directions, residual_grid, LoRA, RTE, OPT, MNLI.
- Dataset: full SST-5 (`dataset_mode=full`), seed=16, data_seed=16.
- Dataloader shuffle: enabled explicitly by `DATALOADER_SHUFFLE=True`.
- Batch size: 64.
- h-grid: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1p5e-3, 2e-3, 3e-3, 4e-3, 5e-3, 1e-2.
- Checkpoints: every 1000 steps plus final, best_acc, best_loss under each run's `checkpoints/`.
- Scheduling: 7 lanes; lanes 0-2 request H100, lanes 3-6 request A100.
