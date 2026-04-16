# Experiment Record: 2026-04-16

This file is a lightweight status snapshot of the main experiments currently active in this repository.

## 0. Pilot Matrix Status

The table below reflects the current status of the pilot matrix as of `2026-04-16`.

| Model | Baseline | Dataset | Precision | Current status | Notes |
|---|---|---|---|---|---|
| OPT-1.3B | MeZO | MNLI | FP16 | Running | Relaunched cleanly as job `44457037`; current fresh result dir has no finished `summary.jsonl` yet |
| OPT-1.3B | MeZO | SST-5 | FP16 | Completed | `14/14 completed` |
| RoBERTa-large | MeZO | MNLI | FP16 | Completed | `9 completed + 5 skipped_nan_guard` |
| RoBERTa-large | MeZO | SST-5 | FP16 | Completed | `9 completed + 5 skipped_nan_guard` |
| OPT-1.3B | MeZO | MNLI | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465908` |
| OPT-1.3B | MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465909` |
| RoBERTa-large | MeZO | MNLI | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465906` |
| RoBERTa-large | MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465907` |
| OPT-1.3B | Sparse MeZO | MNLI | FP16 | Submitted, pending | Smoke passed; full 14-value search queued as job `44526707` |
| OPT-1.3B | Sparse MeZO | SST-5 | FP16 | Submitted, pending | Smoke passed; full 14-value search queued as job `44526706` |
| RoBERTa-large | Sparse MeZO | MNLI | FP16 | Running | Current partial summary: `1 completed + 4 skipped_nan_guard` |
| RoBERTa-large | Sparse MeZO | SST-5 | FP16 | Running | Current partial summary: `2 completed + 4 skipped_nan_guard` |
| OPT-1.3B | Sparse MeZO | MNLI | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465912` |
| OPT-1.3B | Sparse MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465913` |
| RoBERTa-large | Sparse MeZO | MNLI | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465910` |
| RoBERTa-large | Sparse MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465911` |

## 0.1 Experiment Setting Sheet

The table below records the current hyperparameter settings for the main experiment families. Here:

- `14-value grid` = `1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2`
- `8-value grid` = `1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3`
- `INT8/INT4` on the pilot path both mean QuZO low-bit perturb/update, not `load_int8`

| Model | Baseline | Dataset | Precision | H grid | Max steps / epochs | Batch size | Grad accum | LR | Eval steps | Logging steps | ZO probe every | Sparse ratio | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| RoBERTa-large | MeZO | MNLI | FP16 | 14-value grid | `50000` steps | 32 | 1 | `1e-6` | `5000` | 10 | 200 | – | `--zo_two_point_precision fp16 --zo_quantization_bits 16` |
| RoBERTa-large | MeZO | SST-5 | FP16 | 14-value grid | `50000` steps | 32 | 1 | `1e-6` | `5000` | 10 | 200 | – | `--zo_two_point_precision fp16 --zo_quantization_bits 16` |
| RoBERTa-large | Sparse MeZO | MNLI | FP16 | 14-value grid | `50000` steps | 32 | 1 | `1e-6` | `5000` | 10 | 200 | 0.25 | `percentile_per_layer`, `trainable_only` |
| RoBERTa-large | Sparse MeZO | SST-5 | FP16 | 14-value grid | `50000` steps | 32 | 1 | `1e-6` | `5000` | 10 | 200 | 0.25 | `percentile_per_layer`, `trainable_only` |
| OPT-1.3B | MeZO | MNLI | FP16 | 14-value grid | `1` epoch | 16 | 1 | `1e-6` | `5000` | 10 | 200 | – | `--load_float16 --zo_quantization_bits 16 --num_dev 0` |
| OPT-1.3B | MeZO | SST-5 | FP16 | 14-value grid | `5` epochs | 16 | 1 | `1e-6` | `5000` | 10 | 100 | – | `--load_float16 --zo_quantization_bits 16` |
| OPT-1.3B | Sparse MeZO | MNLI | FP16 | 14-value grid | `1` epoch | 16 | 1 | `1e-6` | `5000` | 10 | 200 | 0.25 | `--load_float16 --zo_quantization_bits 16` |
| OPT-1.3B | Sparse MeZO | SST-5 | FP16 | 14-value grid | `5` epochs | 16 | 1 | `1e-6` | `5000` | 10 | 100 | 0.25 | `--load_float16 --zo_quantization_bits 16` |
| RoBERTa-large | MeZO | MNLI | INT8 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int8` pilot |
| RoBERTa-large | MeZO | SST-5 | INT8 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int8` pilot |
| RoBERTa-large | Sparse MeZO | MNLI | INT8 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int8` pilot |
| RoBERTa-large | Sparse MeZO | SST-5 | INT8 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int8` pilot |
| OPT-1.3B | MeZO | MNLI | INT8 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int8` pilot, `num_dev=0`, `num_epochs=1` |
| OPT-1.3B | MeZO | SST-5 | INT8 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int8` pilot, `num_epochs=5` |
| OPT-1.3B | Sparse MeZO | MNLI | INT8 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int8` pilot, `num_dev=0`, `num_epochs=1` |
| OPT-1.3B | Sparse MeZO | SST-5 | INT8 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int8` pilot, `num_epochs=5` |
| RoBERTa-large | MeZO | MNLI | INT4 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int4` pilot |
| RoBERTa-large | MeZO | SST-5 | INT4 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int4` pilot |
| RoBERTa-large | Sparse MeZO | MNLI | INT4 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int4` pilot |
| RoBERTa-large | Sparse MeZO | SST-5 | INT4 | 8-value grid | `10000` steps | 32 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int4` pilot |
| OPT-1.3B | MeZO | MNLI | INT4 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int4` pilot, `num_dev=0`, `num_epochs=1` |
| OPT-1.3B | MeZO | SST-5 | INT4 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | – | QuZO `int4` pilot, `num_epochs=5` |
| OPT-1.3B | Sparse MeZO | MNLI | INT4 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int4` pilot, `num_dev=0`, `num_epochs=1` |
| OPT-1.3B | Sparse MeZO | SST-5 | INT4 | 8-value grid | `10000` steps | 16 | 1 | `1e-6` | `1000` | 10 | 200 | 0.25 | QuZO `int4` pilot, `num_epochs=5` |

## 1. Running / Pending Full H-Sweeps

As of 2026-04-16, the following experiment jobs are running:

### 1.1 QuZO 16-bit OPT-1.3B MNLI

- Job ID: `44457037`
- Job name: `hsweep14h_quzo16_opt13b_mnli`
- Status: `RUNNING`
- Launcher: [opt13b_mnli_quzo16_14h.sh](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/jobs/opt13b_mnli_quzo16_14h.sh)
- Result directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli)
- Log directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli)

Notes:
- This is the relaunched run after fixing the `Trainer._maybe_log_save_evaluate` compatibility bug in `large_models/trainer.py`.
- The current run starts clean in a fresh result/log directory.
- The first `h` is `1e-8`, which historically is numerically fragile for `opt-1.3b + MNLI + fp16/quzo16`.

### 1.2 Sparse MeZO 16-bit RoBERTa-large MNLI

- Job ID: `44285153`
- Job name: `hsweep14h_sparsemezo16_roberta_mnli`
- Status: `RUNNING`
- Result directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/mnli](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/mnli)
- Log directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/roberta-large/mnli](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/roberta-large/mnli)

Latest observed progress:
- Current `h`: `3e-6`
- Latest observed global step: about `35770 / 50000`
- `sparse_active_fraction` remains around `0.25`

### 1.3 Sparse MeZO 16-bit RoBERTa-large SST-5

- Job ID: `44285154`
- Job name: `hsweep14h_sparsemezo16_roberta_sst5`
- Status: `RUNNING`
- Result directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/sst5](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/sst5)
- Log directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/roberta-large/sst5](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/roberta-large/sst5)

Latest observed progress:
- Current `h`: `3e-6`
- Latest observed global step: about `46110 / 50000`
- `sparse_active_fraction` remains around `0.25`

### 1.4 Sparse MeZO 16-bit OPT-1.3B MNLI

- Job ID: `44526707`
- Job name: `hsweep14h_sparsemezo16_opt13b_mnli`
- Status: `PENDING`
- Launcher: [opt13b_mnli_sparse_mezo16_14h.sh](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/jobs/opt13b_mnli_sparse_mezo16_14h.sh)
- Result directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/opt-1.3b/mnli](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/opt-1.3b/mnli)
- Log directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/opt-1.3b/mnli](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/opt-1.3b/mnli)

Notes:
- A targeted smoke test at `h=1e-4` completed successfully before submission.
- Current queue reason is `Priority`; the job is not blocked by a dependency.

### 1.5 Sparse MeZO 16-bit OPT-1.3B SST-5

- Job ID: `44526706`
- Job name: `hsweep14h_sparsemezo16_opt13b_sst5`
- Status: `PENDING`
- Launcher: [opt13b_sst5_sparse_mezo16_14h.sh](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/jobs/opt13b_sst5_sparse_mezo16_14h.sh)
- Result directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/opt-1.3b/sst5](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/opt-1.3b/sst5)
- Log directory: [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/opt-1.3b/sst5](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/opt-1.3b/sst5)

Notes:
- A targeted smoke test at `h=1e-4` completed successfully before submission.
- Current queue reason is `Priority`; the job is not blocked by a dependency.

## 2. Completed Main H-Sweeps

These runs already have finished `summary.jsonl` files and can be treated as completed sweeps.

### 2.1 QuZO 16-bit RoBERTa-large SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/roberta-large/sst5/summary.jsonl)
- Status counts:
  - `completed`: `9`
  - `skipped_nan_guard`: `5`

### 2.2 QuZO 16-bit RoBERTa-large MNLI

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/roberta-large/mnli/summary.jsonl)
- Status counts:
  - `completed`: `9`
  - `skipped_nan_guard`: `5`

### 2.3 QuZO 16-bit OPT-1.3B SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/sst5/summary.jsonl)
- Status counts:
  - `completed`: `14`

### 2.4 QuZO 8-bit RoBERTa-large SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo8/roberta-large/sst5/summary.jsonl)
- Status counts:
  - `completed`: `10`
  - `skipped_nan_guard`: `4`

## 3. In-Progress Sparse MeZO Summaries

These runs already have partial sweep summaries but are still running.

### 3.1 Sparse MeZO 16-bit RoBERTa-large MNLI

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/mnli/summary.jsonl)
- Current recorded status counts:
  - `completed`: `1`
  - `skipped_nan_guard`: `4`
- Last recorded finished `h`: `1e-6`

### 3.2 Sparse MeZO 16-bit RoBERTa-large SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/sst5/summary.jsonl)
- Current recorded status counts:
  - `completed`: `2`
  - `skipped_nan_guard`: `4`
- Last recorded finished `h`: `3e-6`

## 4. Smoke / Debug Runs Already Completed

### 4.1 Sparse MeZO smoke tests

- MNLI:
  - [/scratch/jy03364/MeZO_/experiments/sparse_mezo_smoke/medium/roberta_mnli/run_sparse_mezo16/seed16/run_summary.json](/scratch/jy03364/MeZO_/experiments/sparse_mezo_smoke/medium/roberta_mnli/run_sparse_mezo16/seed16/run_summary.json)
- SST-5:
  - [/scratch/jy03364/MeZO_/experiments/sparse_mezo_smoke/medium/roberta_sst5/run_sparse_mezo16/seed16/run_summary.json](/scratch/jy03364/MeZO_/experiments/sparse_mezo_smoke/medium/roberta_sst5/run_sparse_mezo16/seed16/run_summary.json)

### 4.2 OPT-1.3B MNLI quzo16 compatibility / numeric smoke runs

- One-step compatibility check:
  - [/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_mnli_quzo16_one_step_eval/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_mnli_quzo16_one_step_eval/run_summary.json)
- 100-step `h=1e-8` numeric check:
  - [/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_mnli_quzo16_100step_eval/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_mnli_quzo16_100step_eval/run_summary.json)
- 100-step `h=1e-4` stability check:
  - metrics only at [/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_mnli_quzo16_h1e4_param_scan](/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_mnli_quzo16_h1e4_param_scan)

### 4.3 OPT-1.3B Sparse MeZO fp16 targeted smoke runs

- MNLI `h=1e-4` smoke:
  - [/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_sparse_mezo16_mnli_h1e4_smoke/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_sparse_mezo16_mnli_h1e4_smoke/run_summary.json)
  - Final metrics: `accuracy=0.375`, `valid_mismatched_accuracy=0.5`
- SST-5 `h=1e-4` smoke:
  - [/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_sparse_mezo16_sst5_h1e4_smoke/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_sparse_mezo16_sst5_h1e4_smoke/run_summary.json)
  - Final metrics: `accuracy=0.125`, `dev_accuracy=0.2423887587822014`

### 4.4 INT4 pilot smoke matrix

This is the final clean INT4 smoke matrix for the current pilot settings. All 8 combinations completed and wrote `run_summary.json`.

#### Medium path: RoBERTa-large + QuZO int4

Smoke config:

- `max_steps = 50`
- `dataset_mode = full`
- `full_dev_ratio = 0.01`
- `MNLI / SST-5`
- `MeZO / Sparse MeZO`
- `--zo_two_point_precision fp16 --zo_quantization int4`
- Sparse runs use `--sparse_ratio 0.25 --sparse_mask_strategy percentile_per_layer --sparse_scope trainable_only --sparse_log_active_fraction True`

Results:

| Model | Baseline | Dataset | Summary | Tail perf | Sparse stats |
| --- | --- | --- | --- | --- | --- |
| RoBERTa-large | MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/medium/mezo/roberta_mnli/run_smoke/run_summary.json) | `wallclock/step = 10.4256`, `samples/sec = 3.0694`, `max_gpu_memory_gb = 3.8519` | – |
| RoBERTa-large | MeZO | SST-5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/medium/mezo/roberta_sst5/run_smoke/run_summary.json) | `wallclock/step = 0.9236`, `samples/sec = 34.6487`, `max_gpu_memory_gb = 3.8519` | – |
| RoBERTa-large | Sparse MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/medium/sparse_mezo/roberta_mnli/run_smoke/run_summary.json) | `wallclock/step = 12.4283`, `samples/sec = 2.5748`, `max_gpu_memory_gb = 4.2336` | `active_fraction = 0.5230` at the last logged step |
| RoBERTa-large | Sparse MeZO | SST-5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/medium/sparse_mezo/roberta_sst5/run_smoke/run_summary.json) | `wallclock/step = 3.0360`, `samples/sec = 10.5402`, `max_gpu_memory_gb = 4.2342` | `active_fraction = 0.5066` at the last logged step |

#### Large path: OPT-1.3B + QuZO int4

Smoke config:

- `max_steps = 50`
- `dataset_mode = full`
- `num_eval = 8`
- `MNLI / SST5`
- `MeZO / Sparse MeZO`
- `--load_float16 --zo_quantization int4`
- Sparse runs use `--sparse_ratio 0.25 --sparse_mask_strategy percentile_per_layer --sparse_scope trainable_only --sparse_log_active_fraction True`

Results:

| Model | Baseline | Dataset | Summary | Tail perf | Sparse stats |
| --- | --- | --- | --- | --- | --- |
| OPT-1.3B | MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/large/mezo/opt13b_mnli/run_summary.json) | `wallclock/step = 1.1113`, `samples/sec = 14.3980`, `max_gpu_memory_gb = 6.7519` | – |
| OPT-1.3B | MeZO | SST5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/large/mezo/opt13b_sst5/run_summary.json) | `wallclock/step = 1.0896`, `samples/sec = 14.6850`, `max_gpu_memory_gb = 6.5094` | – |
| OPT-1.3B | Sparse MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/large/sparse_mezo/opt13b_mnli/run_summary.json) | `wallclock/step = 4.6403`, `samples/sec = 3.4480`, `max_gpu_memory_gb = 7.9778` | `active_fraction = 0.4801` at the last logged step |
| OPT-1.3B | Sparse MeZO | SST5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke_fix/int4_matrix_final/large/sparse_mezo/opt13b_sst5/run_summary.json) | `wallclock/step = 4.6098`, `samples/sec = 3.4708`, `max_gpu_memory_gb = 7.7357` | `active_fraction = 0.6042` at the last logged step |

## 5. Archived / Relaunched Runs

The previous partial or broken `opt-1.3b / MNLI / quzo16` runs were preserved instead of overwritten.

Archived directories:
- [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli_pre_maybe_log_fix_20260416_013126](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli_pre_maybe_log_fix_20260416_013126)
- [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli_pre_maybe_log_fix_20260416_013126](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli_pre_maybe_log_fix_20260416_013126)
- [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli_relaunch_prep_20260416_073136](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli_relaunch_prep_20260416_073136)
- [/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli_relaunch_prep_20260416_073136](/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli_relaunch_prep_20260416_073136)

## 6. Notes

- `quzo16 opt-1.3b mnli` no longer fails at step 1 because of the `_maybe_log_save_evaluate` compatibility bug.
- Extremely small `h` values on `opt-1.3b + MNLI + fp16/quzo16`, especially `1e-8`, remain numerically risky.
- A direct 100-step diagnostic showed `h=1e-4` is stable, while `h=1e-8` can corrupt parameters with `NaN`s in fp16.
- `Sparse MeZO` on the large-model 16-bit path required one compatibility fix: when QuZO 16-bit uses a single dense direction `z`, the sparse mask now applies to `z` and reuses the masked direction for both perturb/update call sites.
