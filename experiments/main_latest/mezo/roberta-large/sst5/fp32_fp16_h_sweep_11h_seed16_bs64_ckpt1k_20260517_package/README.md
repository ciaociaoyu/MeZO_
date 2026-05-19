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

## Package contents

This package is the lightweight, git-trackable result bundle for the completed FP32/FP16 run. It intentionally excludes checkpoint/model artifacts such as `checkpoints/`, `pytorch_model.bin`, `model.safetensors`, optimizer states, scheduler states, and RNG states.

Included:
- `summaries/`: final aggregate CSV/MD reports, checkpoint inventory, and failure table.
- `plots/`: plot-ready CSVs and generated PNGs.
- `run_artifacts/`: per-run configs, summaries, eval metrics, and probe metrics.
- `lane_manifests/`, `jobs/`, `commands.txt`, `config_manifest.json`, and `run_manifest.csv`: launch and scheduler provenance.
- `scripts/`: copies of the launcher, lane runner, and summarizer used for this experiment.

Omitted from the git package to keep it lightweight:
- raw model/checkpoint files and optimizer state;
- full per-run `stderr.log` and SLURM `.err` logs;
- dense per-step `metrics.csv` / `metrics_logs/` files, which remain available in the local raw result directory.

Final status:
- Completed runs: 22 / 22.
- Failed runs: 0.
- FP32 best eval accuracy: `h=3e-5`, `best_eval_acc=0.48711943793911006`.
- FP32 best last eval accuracy: `h=2e-3`, `last_eval_acc=0.48009367681498827`.
- FP16 best and last eval accuracy: `h=1e-5`, `0.4637002341920375`.
