# INT8 Update And Sparse Probe Plan

Date: 2026-05-12.

## Repository Detection

- Training entrypoint: `medium_models/run.py`, normally launched through `medium_models/mezo.sh`.
- Local INT8 diagnostic wrapper detected: `experiments/int8_error_origin_probe/run_roberta_sst5_int8_local.sh`.
- Main config style: environment variables passed into `medium_models/mezo.sh`, then extra Python CLI flags are forwarded to `run_fewshot.sh` / `run.py`.
- Existing MeZO/ZO flags include `--zero_order_optim`, `--efficient_zero_order`, `--zero_order_eps`, `--zo_probe_every`, `--zo_probe_num_seeds`, `--zo_two_point_precision`.
- Existing INT8/QuZO flags include `--zo_quantization int8`, `--zo_quantization_bits`, `--zo_update_backend`, `--residual_dtype`, `--residual_commit_mode`, `--residual_max_code_step`, `--int8_freeze_scale`, `--save_update_stats_jsonl`.
- Scheduler detected: SLURM (`sinfo`, `sbatch`, `squeue` are available).
- Existing GPU syntax: `#SBATCH --partition=gpu_p` plus `#SBATCH --gres=gpu:H100:1` for H100 examples. An older A100 example uses `#SBATCH --partition=a100` plus `#SBATCH --gres=gpu:a100:1`. L4 nodes are visible in `gpu_p` with GRES `gpu:L4:4`, so the generated script maps L4 fallback to `--gres=gpu:L4:1`.

## Generated Scripts

- Local residual first round: `scripts/run_int8_residual_round2_local.sh`
- Sparse probe submitter: `scripts/submit_sparse_probe_hsweep.sh`
- Sparse probe summarizer: `scripts/summarize_sparse_probe_hsweep.py`

## Local Residual Commands

Local H100 check:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Launched local residual round2:

```bash
TS=20260512_180500 CUDA_VISIBLE_DEVICES=0 CONDA_ENV=ciao scripts/run_int8_residual_round2_local.sh
```

Output directory:

```text
runs/int8_residual_round2_20260512_180500/
```

Generated summaries:

```text
runs/int8_residual_round2_20260512_180500/summary.csv
runs/int8_residual_round2_20260512_180500/summary.md
```

Status: completed all requested first-round local runs. The no-op residual run produced zero active updates and zero actual update norm. The direct INT8 baseline showed low update alignment and a large norm mismatch. Residual-grid runs kept committed weights on-grid (`grid_error_norm=0`) and improved alignment, but residual-over-scale diagnostics are inflated by near-zero scales (`scale_min` around `1.8e-12`).

The first attempt used `TS=20260512_173500` and failed during the no-op residual run because direct `torch.quantile` over a full RoBERTa tensor was too large. The implementation now samples at most 1,000,000 entries for residual/scale quantiles.

## Sparse Probe Submission

Generated sparse h-sweep manifest:

```text
runs/sparse_probe_hsweep_20260512_173500/manifest.tsv
```

Generated SLURM array script:

```text
runs/sparse_probe_hsweep_20260512_173500/jobs/sparse_probe_hsweep_array.sbatch
```

Initial full submission command:

```bash
TS=20260512_173500 CONDA_ENV=ciao NUM_PROBE_DIRECTIONS=50 MAX_ARRAY_CONCURRENCY=4 scripts/submit_sparse_probe_hsweep.sh
```

That full 25-task array was rejected by SLURM with `QOSMaxSubmitJobPerUserLimit`, so it was split into smaller arrays:

```bash
sbatch --array=0-0%1 runs/sparse_probe_hsweep_20260512_173500/jobs/sparse_probe_hsweep_array.sbatch
sbatch --array=1-4%2 runs/sparse_probe_hsweep_20260512_173500/jobs/sparse_probe_hsweep_array.sbatch
sbatch --array=5-12%2 runs/sparse_probe_hsweep_20260512_173500/jobs/sparse_probe_hsweep_array.sbatch
```

Submitted job IDs:

```text
45161347 array=0-0%1
45161348 array=1-4%2
45161349 array=5-12%2
```

The remaining `array=13-24%2` submission was also rejected by the QOS submit limit and remains unsubmitted in the manifest.

Sparse summary command, once JSONL logs exist:

```bash
scripts/summarize_sparse_probe_hsweep.py runs/sparse_probe_hsweep_20260512_173500
```

Partial sparse summary generated from completed dense cases:

```text
runs/sparse_probe_hsweep_20260512_173500/summary.csv
runs/sparse_probe_hsweep_20260512_173500/summary.md
```

Current sparse status at this note update: dense cases `0-6` completed. The first two sparse exact-random cases (`p=0.03`, factors `0.25` and `0.5`) were still running and had created empty `probe_stats.jsonl` files, so the partial summary currently contains dense rows only. The exact-random sparse probe path is substantially slower on the full RoBERTa-large trainable parameter set because it generates exact masks over very large tensors.
