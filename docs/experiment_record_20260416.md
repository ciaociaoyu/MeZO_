# Experiment Record: 2026-04-16

This file is a lightweight status snapshot of the main experiments currently active in this repository.

## 0.0 Runtime Environments

The current repository setup uses different conda environments for different model paths. This mapping is now treated as part of the experiment contract and should be preserved in future smoke tests, benchmarks, and Slurm launchers.

| Model family | Code path | Conda env | Notes |
|---|---|---|---|
| RoBERTa-large and other medium models | `medium_models/` | `ciao` | This is the environment used by the medium-model trainer path and the current RoBERTa sweeps. |
| OPT-family large models | `large_models/` | `mezo-env` | Used for `opt-1.3b` and other non-Mistral large-model runs. |
| Mistral-7B | `large_models/` | `mezo-mistral` | Mistral uses a separate environment and should not be assumed to share the `mezo-env` stack. |

Environment note:

- `mezo-mistral` is intentionally separate from `mezo-env`.
- The resumable method-speed matrix launcher at [run_zo_method_speed_matrix.py](/scratch/jy03364/MeZO_/experiments/pilot/_shared/speed_bench_h100/run_zo_method_speed_matrix.py) follows the same mapping:
  - `roberta-large -> ciao`
  - `opt-1.3b -> mezo-env`
  - `mistral-7b -> mezo-mistral`

## 0. Pilot Matrix Status

The table below reflects the current status of the pilot matrix as of `2026-04-17`.

| Model | Baseline | Dataset | Precision | Current status | Notes |
|---|---|---|---|---|---|
| OPT-1.3B | MeZO | MNLI | FP16 | Running | Relaunched cleanly as job `44457037`; current fresh result dir already has `5 completed` points through `h=1e-6` |
| OPT-1.3B | MeZO | SST-5 | FP16 | Completed | `14/14 completed` |
| RoBERTa-large | MeZO | MNLI | FP16 | Completed | `9 completed + 5 skipped_nan_guard` |
| RoBERTa-large | MeZO | SST-5 | FP16 | Completed | `9 completed + 5 skipped_nan_guard` |
| OPT-1.3B | MeZO | MNLI | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465908` |
| OPT-1.3B | MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465909` |
| RoBERTa-large | MeZO | MNLI | INT8 | Running | New 8-value / 10k-step pilot running as job `44465906` |
| RoBERTa-large | MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44465907` |
| OPT-1.3B | Sparse MeZO | MNLI | FP16 | Submitted, pending | Old run `44526707` was cancelled after zero-mask collapse at tiny `h`; relaunched cleanly as `44567785` |
| OPT-1.3B | Sparse MeZO | SST-5 | FP16 | Running | Full 14-value search running as job `44526706`; low-`h` completed points `1e-8, 3e-8, 1e-7, 3e-7` are numerically invalid because `active_fraction=0` |
| RoBERTa-large | Sparse MeZO | MNLI | FP16 | Partial, timed out | Job `44285153` hit Slurm time limit after `2 completed + 4 skipped_nan_guard` |
| RoBERTa-large | Sparse MeZO | SST-5 | FP16 | Partial, timed out | Job `44285154` hit Slurm time limit after `3 completed + 4 skipped_nan_guard` |
| OPT-1.3B | Sparse MeZO | MNLI | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44567783` |
| OPT-1.3B | Sparse MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44567784` |
| RoBERTa-large | Sparse MeZO | MNLI | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44567781` |
| RoBERTa-large | Sparse MeZO | SST-5 | INT8 | Submitted, pending | New 8-value / 10k-step pilot queued as job `44567782` |

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

## 0.2 Final H100 Method-Speed Matrix Update

As of `2026-04-18`, the dedicated H100 multi-method speed matrix has finished. This is separate from the still-pending Slurm `h-sweep` jobs listed later in this file.

Matrix location:

- Script: [run_zo_method_speed_matrix.py](/scratch/jy03364/MeZO_/experiments/pilot/_shared/speed_bench_h100/run_zo_method_speed_matrix.py)
- Output tree: `experiments/pilot/<method>/<model>/<task>/<precision>/speed_bench_h100/zo_method_matrix_20260418`
- Summary: [/scratch/jy03364/MeZO_/experiments/pilot/_shared/speed_bench_h100/zo_method_matrix_20260418/summary.jsonl](/scratch/jy03364/MeZO_/experiments/pilot/_shared/speed_bench_h100/zo_method_matrix_20260418/summary.jsonl)
- Full write-up: [docs/sparse_mezo_h100_experiments.md](/scratch/jy03364/MeZO_/docs/sparse_mezo_h100_experiments.md)

Final status:

- `72` total cells
- `66 completed + 6 unsupported`
- The only unsupported cells are `roberta-large + {lozo,hizoo} + int8 + {MNLI,SST-5,BoolQ}`

RoBERTa-large `int8` status:

- Yes, the `roberta-large int8` speed matrix is finished for all currently supported cells.
- Completed:
  - `mezo + MNLI + int8`: `55.100 samples/sec`, `0.581 sec/step`
  - `mezo + SST-5 + int8`: `55.358 samples/sec`, `0.578 sec/step`
  - `mezo + BoolQ + int8`: `22.628 samples/sec`, `1.414 sec/step`
  - `sparse_mezo + MNLI + int8`: `45.342 samples/sec`, `0.706 sec/step`
  - `sparse_mezo + SST-5 + int8`: `45.754 samples/sec`, `0.699 sec/step`
  - `sparse_mezo + BoolQ + int8`: `21.697 samples/sec`, `1.475 sec/step`
- Unsupported rather than unfinished:
  - `lozo + {MNLI,SST-5,BoolQ} + int8`
  - `hizoo + {MNLI,SST-5,BoolQ} + int8`

Precision semantics warning:

- `roberta-large` uses the `medium_models` path, where matrix `int8` means QuZO `--zo_quantization int8`
- `opt-1.3b` and `mistral-7b` use the `large_models` path, where matrix `int8` means `--load_int8 --zo_quantization_bits 32`
- So the matrix-wide `int8` label is not one uniform implementation across all model families

The rest of this file should be read as a historical experiment log for the main sweep jobs. The full speed-matrix tables and the final per-model throughput numbers now live in `docs/sparse_mezo_h100_experiments.md`, Section `8`.

## 1. Running / Pending Full H-Sweeps

As of 2026-04-17, the following experiment jobs are running:

### 1.1 QuZO 16-bit OPT-1.3B MNLI

- Job ID: `44457037`
- Job name: `hsweep14h_quzo16_opt13b_mnli`
- Status: `RUNNING`
- Launcher: [opt13b_mnli_quzo16_14h.sh](/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/jobs/opt13b_mnli_quzo16_14h.sh)
- Result directory: [/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results](/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results)
- Log directory: removed during cleanup because this sweep did not finish.

Notes:
- This is the relaunched run after fixing the `Trainer._maybe_log_save_evaluate` compatibility bug in `large_models/trainer.py`.
- The current run starts clean in a fresh result/log directory.
- The first `h` is `1e-8`, which historically is numerically fragile for `opt-1.3b + MNLI + fp16/quzo16`.
- Current partial summary already records `5 completed` points through `h=1e-6`.

### 1.2 MeZO INT8 RoBERTa-large MNLI

- Job ID: `44465906`
- Job name: `hsweep8h_mezo_int8_roberta_mnli`
- Status: `RUNNING`
- Result directory: [/scratch/jy03364/MeZO_/experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/results](/scratch/jy03364/MeZO_/experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/results)
- Log directory: [/scratch/jy03364/MeZO_/experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/logs](/scratch/jy03364/MeZO_/experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/logs)

Notes:
- This is the new 8-value / 10k-step INT8 pilot on the medium-model path.
- The current pilot uses QuZO low-bit semantics (`--zo_quantization int8`), not `load_int8`.

### 1.3 Sparse MeZO 16-bit OPT-1.3B SST-5

- Job ID: `44526706`
- Job name: `hsweep14h_sparsemezo16_opt13b_sst5`
- Status: `RUNNING`
- Launcher: [opt13b_sst5_sparse_mezo16_14h.sh](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/jobs/opt13b_sst5_sparse_mezo16_14h.sh)
- Result directory: [/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results)
- Log directory: [/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/logs](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/logs)

Notes:
- Current summary records `6 completed` points through `h=3e-6`.
- The low-`h` completed points `1e-8, 3e-8, 1e-7, 3e-7` are numerically invalid because `sparse_mezo_last_stats.active_fraction = 0.0`.
- The valid completed points so far are `1e-6` and `3e-6`, where `active_fraction ≈ 0.25`.

### 1.4 Sparse MeZO 16-bit RoBERTa-large MNLI

- Job ID: `44285153`
- Job name: `hsweep14h_sparsemezo16_roberta_mnli`
- Status: `TIMEOUT`
- Result directory: [/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/results](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/results)
- Log directory: removed during cleanup because this task timed out.

Latest recorded progress before timeout:
- Last recorded finished `h`: `3e-6`
- Current partial summary: `2 completed + 4 skipped_nan_guard`
- The run ended due to Slurm time limit, not a Python crash.

### 1.5 Sparse MeZO 16-bit RoBERTa-large SST-5

- Job ID: `44285154`
- Job name: `hsweep14h_sparsemezo16_roberta_sst5`
- Status: `TIMEOUT`
- Result directory: [/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/results](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/results)
- Log directory: removed during cleanup because this task timed out.

Latest recorded progress before timeout:
- Last recorded finished `h`: `1e-5`
- Current partial summary: `3 completed + 4 skipped_nan_guard`
- The run ended due to Slurm time limit, not a Python crash.

### 1.6 Sparse MeZO 16-bit OPT-1.3B MNLI

- Job ID: `44567785`
- Job name: `hsweep14h_sparsemezo16_opt13b_mnli`
- Status: `PENDING`
- Launcher: [opt13b_mnli_sparse_mezo16_14h.sh](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/jobs/opt13b_mnli_sparse_mezo16_14h.sh)
- Result directory: [/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results)
- Log directory: removed during cleanup because this task never completed.

Notes:
- A targeted smoke test at `h=1e-4` completed successfully before submission.
- The previous job `44526707` was cancelled after `h=1e-8` produced a zero-mask collapse (`active_fraction=0`).
- `nan_guard` has since been tightened so the first non-ignored `nan` now skips the current `h`.
- Current queue reason is `Priority`; the job is not blocked by a dependency.

## 2. Completed Main H-Sweeps

These runs already have finished `summary.jsonl` files and can be treated as completed sweeps.

### 2.1 QuZO 16-bit RoBERTa-large SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/main/mezo/roberta-large/sst5/fp16/h_sweep_14h/results/summary.jsonl)
- Status counts:
  - `completed`: `9`
  - `skipped_nan_guard`: `5`

### 2.2 QuZO 16-bit RoBERTa-large MNLI

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/main/mezo/roberta-large/mnli/fp16/h_sweep_14h/results/summary.jsonl)
- Status counts:
  - `completed`: `9`
  - `skipped_nan_guard`: `5`

### 2.3 QuZO 16-bit OPT-1.3B SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results/summary.jsonl)
- Status counts:
  - `completed`: `14`

## 3. In-Progress Sparse MeZO Summaries

These runs already have partial sweep summaries. The RoBERTa jobs timed out; the OPT SST-5 job is still running.

### 3.1 Sparse MeZO 16-bit RoBERTa-large MNLI

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/results/summary.jsonl)
- Current recorded status counts:
  - `completed`: `2`
  - `skipped_nan_guard`: `4`
- Last recorded finished `h`: `1e-6`
- Terminal job state: `TIMEOUT`

### 3.2 Sparse MeZO 16-bit RoBERTa-large SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/results/summary.jsonl)
- Current recorded status counts:
  - `completed`: `3`
  - `skipped_nan_guard`: `4`
- Last recorded finished `h`: `1e-5`
- Terminal job state: `TIMEOUT`

### 3.3 Sparse MeZO 16-bit OPT-1.3B SST-5

- Summary: [summary.jsonl](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results/summary.jsonl)
- Current recorded status counts:
  - `completed`: `6`
- Last recorded finished `h`: `3e-6`
- Validity note:
  - `1e-8`, `3e-8`, `1e-7`, `3e-7` are completed but numerically invalid because `artifacts.sparse_mezo_last_stats.active_fraction = 0.0`
  - `1e-6` and `3e-6` are the first valid completed points

## 4. Smoke / Debug Runs Already Completed

### 4.1 Sparse MeZO smoke tests

- MNLI:
  - [/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/roberta-large/mnli/fp16/sparse_mezo_smoke/run_sparse_mezo16/seed16/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/roberta-large/mnli/fp16/sparse_mezo_smoke/run_sparse_mezo16/seed16/run_summary.json)
- SST-5:
  - [/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/roberta-large/sst5/fp16/sparse_mezo_smoke/run_sparse_mezo16/seed16/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/roberta-large/sst5/fp16/sparse_mezo_smoke/run_sparse_mezo16/seed16/run_summary.json)

### 4.2 OPT-1.3B MNLI quzo16 compatibility / numeric smoke runs

- One-step compatibility check:
  - [/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_mnli_quzo16_one_step_eval/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_mnli_quzo16_one_step_eval/run_summary.json)
- 100-step `h=1e-8` numeric check:
  - [/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_mnli_quzo16_100step_eval/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_mnli_quzo16_100step_eval/run_summary.json)
- 100-step `h=1e-4` stability check:
  - metrics only at [/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_mnli_quzo16_h1e4_param_scan](/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_mnli_quzo16_h1e4_param_scan)

### 4.3 OPT-1.3B Sparse MeZO fp16 targeted smoke runs

- MNLI `h=1e-4` smoke:
  - [/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_sparse_mezo16_mnli_h1e4_smoke/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/opt-1.3b/mnli/fp16/smoke_fix/opt13b_sparse_mezo16_mnli_h1e4_smoke/run_summary.json)
  - Final metrics: `accuracy=0.375`, `valid_mismatched_accuracy=0.5`
- SST-5 `h=1e-4` smoke:
  - [/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/opt-1.3b/sst5/fp16/smoke_fix/opt13b_sparse_mezo16_sst5_h1e4_smoke/run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/opt-1.3b/sst5/fp16/smoke_fix/opt13b_sparse_mezo16_sst5_h1e4_smoke/run_summary.json)
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
| RoBERTa-large | MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/mezo/roberta-large/mnli/int4/smoke_fix/int4_matrix_final/roberta_mnli/run_smoke/run_summary.json) | `wallclock/step = 10.4256`, `samples/sec = 3.0694`, `max_gpu_memory_gb = 3.8519` | – |
| RoBERTa-large | MeZO | SST-5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/mezo/roberta-large/sst5/int4/smoke_fix/int4_matrix_final/roberta_sst5/run_smoke/run_summary.json) | `wallclock/step = 0.9236`, `samples/sec = 34.6487`, `max_gpu_memory_gb = 3.8519` | – |
| RoBERTa-large | Sparse MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/roberta-large/mnli/int4/smoke_fix/int4_matrix_final/roberta_mnli/run_smoke/run_summary.json) | `wallclock/step = 12.4283`, `samples/sec = 2.5748`, `max_gpu_memory_gb = 4.2336` | `active_fraction = 0.5230` at the last logged step |
| RoBERTa-large | Sparse MeZO | SST-5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/roberta-large/sst5/int4/smoke_fix/int4_matrix_final/roberta_sst5/run_smoke/run_summary.json) | `wallclock/step = 3.0360`, `samples/sec = 10.5402`, `max_gpu_memory_gb = 4.2342` | `active_fraction = 0.5066` at the last logged step |

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
| OPT-1.3B | MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/mnli/int4/smoke_fix/int4_matrix_final/opt13b_mnli/run_summary.json) | `wallclock/step = 1.1113`, `samples/sec = 14.3980`, `max_gpu_memory_gb = 6.7519` | – |
| OPT-1.3B | MeZO | SST5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/mezo/opt-1.3b/sst5/int4/smoke_fix/int4_matrix_final/opt13b_sst5/run_summary.json) | `wallclock/step = 1.0896`, `samples/sec = 14.6850`, `max_gpu_memory_gb = 6.5094` | – |
| OPT-1.3B | Sparse MeZO | MNLI | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/opt-1.3b/mnli/int4/smoke_fix/int4_matrix_final/opt13b_mnli/run_summary.json) | `wallclock/step = 4.6403`, `samples/sec = 3.4480`, `max_gpu_memory_gb = 7.9778` | `active_fraction = 0.4801` at the last logged step |
| OPT-1.3B | Sparse MeZO | SST5 | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/opt-1.3b/sst5/int4/smoke_fix/int4_matrix_final/opt13b_sst5/run_summary.json) | `wallclock/step = 4.6098`, `samples/sec = 3.4708`, `max_gpu_memory_gb = 7.7357` | `active_fraction = 0.6042` at the last logged step |

## 5. Archived / Relaunched Runs

The previous partial or broken `opt-1.3b / MNLI / quzo16` runs were preserved instead of overwritten.

Archived directories:
- [/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_pre_maybe_log_fix_20260416_013126/results](/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_pre_maybe_log_fix_20260416_013126/results)
- [/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_pre_maybe_log_fix_20260416_013126/logs](/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_pre_maybe_log_fix_20260416_013126/logs)
- [/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_relaunch_prep_20260416_073136/results](/scratch/jy03364/MeZO_/experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_relaunch_prep_20260416_073136/results)
- Historical partial logs for `mnli_relaunch_prep_20260416_073136` were deleted during cleanup.
- [/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_pre_zero_mask_relaunch_20260417_122235/results](/scratch/jy03364/MeZO_/experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/archive_runs/mnli_pre_zero_mask_relaunch_20260417_122235/results)
- Historical partial logs for `mnli_pre_zero_mask_relaunch_20260417_122235` were deleted during cleanup.

## 6. Guard Behavior

- `experiments/main/_shared/h_sweep_14h/nan_guard.py` now defaults to `--max-consecutive-nan=1`.
- The 14-value sweep scripts set `NAN_GUARD_LIMIT=1`, so the first non-ignored `nan` in the child logs now skips the current `h`.
- This change was made after `opt-1.3b + Sparse MeZO + MNLI + fp16` showed that repeated `eval_loss=nan` could be masked by interleaved `loss=0.0` lines and never reach the old threshold of 100.

## 6. Notes

- `quzo16 opt-1.3b mnli` no longer fails at step 1 because of the `_maybe_log_save_evaluate` compatibility bug.
- Extremely small `h` values on `opt-1.3b + MNLI + fp16/quzo16`, especially `1e-8`, remain numerically risky.
- A direct 100-step diagnostic showed `h=1e-4` is stable, while `h=1e-8` can corrupt parameters with `NaN`s in fp16.
- `Sparse MeZO` on the large-model 16-bit path required one compatibility fix: when QuZO 16-bit uses a single dense direction `z`, the sparse mask now applies to `z` and reuses the masked direction for both perturb/update call sites.

## 7. Sparse MeZO Runtime Optimization Update

Motivation:

- The previous local Sparse MeZO path was much slower than dense MeZO because it recomputed per-layer percentile thresholds with `torch.kthvalue` at every optimizer step, then generated dense random directions and masked them afterward.
- That implementation detail was substantially heavier than the official SparseMeZO repo, which precomputes threshold statistics and uses a lighter per-step masking path.

Code changes:

- Added `--sparse_mask_refresh_steps` in `medium_models/run.py`.
- Semantics:
  - `0`: freeze the initial sparse mask for the entire run.
  - `1`: refresh the sparse mask every optimizer step.
  - `N > 1`: refresh the sparse mask every `N` optimizer steps.
- Refactored `medium_models/src/sparse_mezo.py` into two phases:
  - `build_sparse_thresholds(...)` computes and caches per-layer thresholds.
  - `build_sparse_masks_from_thresholds(...)` rebuilds masks from cached thresholds.
- Added `sample_masked_normal_like(...)` so sparse runs sample Gaussian noise only on active coordinates instead of materializing a full dense direction first.
- Updated `medium_models/src/quzo.py` so `make_quzo_direction_pair(...)` can consume a sparse mask directly and generate sparse-aware `u1/u2/z`.
- Updated `medium_models/src/trainer.py` so:
  - sparse thresholds and masks are cached across steps according to `sparse_mask_refresh_steps`;
  - QuZO directions are constructed with the sparse mask directly;
  - repeated `direction -> sparse mask -> update -> sparse mask` behavior is removed from the main update path.

Default behavior after this update:

- `sparse_mask_refresh_steps` defaults to `100`.
- This default keeps the mask dynamic but avoids paying the old percentile-selection cost at every step.

Validation:

- `python -m py_compile medium_models/src/sparse_mezo.py medium_models/src/quzo.py medium_models/src/trainer.py medium_models/run.py`
- Lightweight smoke test in conda env `ciao`:
  - verified cached-threshold mask construction;
  - verified masked Gaussian sampling only produces nonzero values on active entries;
  - verified QuZO `u1/u2` remain zero on inactive entries under a sparse mask.

## 8. LOZO / HiZOO / Sparse MeZO Integration Update

Reference check:

- LOZO paper: https://arxiv.org/abs/2410.07698
- LOZO official repo: https://github.com/optsuite/LOZO
- HiZOO paper: https://arxiv.org/abs/2402.15173
- HiZOO official repo: https://github.com/Yanjun-Zhao/HiZOO
- Sparse MeZO paper page: https://openreview.net/forum?id=Tjw0ACu3NL
- Sparse MeZO official repo: https://github.com/NUS-HPC-AI-Lab/SparseMeZO

Code changes:

- Added an explicit `--zo_method` switch in both `medium_models/run.py` and `large_models/run.py`.
- Supported method names:
  - `mezo`
  - `sparse_mezo`
  - `lozo`
  - `lozo_m`
  - `hizoo`
- Added method-specific args:
  - `--lozo_rank`
  - `--lozo_step_interval`
  - `--lozo_beta1`
  - `--hizoo_hessian_smooth_type`
- Implemented LOZO in both trainer paths with the official low-rank perturbation/update structure:
  - matrix parameters use rank-`r` directions `u v^T`;
  - vector parameters fall back to dense Gaussian directions;
  - optional `lozo_m` momentum path follows the official repository structure.
- Implemented HiZOO in both trainer paths with diagonal-Hessian scaling:
  - perturbations use `z / sqrt(H)` as in the paper/repo;
  - Hessian diagonals are updated from the same three function values `f(theta)`, `f(theta + h d)`, `f(theta - h d)`.

Runtime-oriented implementation notes:

- LOZO updates use in-place `addmm_` instead of materializing dense `u @ v^T` tensors first.
- HiZOO intentionally skips the extra post-update logging forward found in the official training code; the update itself still uses the same Hessian estimator, but the local implementation keeps the step at 3 forward passes instead of paying for a 4th bookkeeping pass.
- Sparse MeZO speed measurements below use `--sparse_mask_refresh_steps 0` so the mask is frozen after the initial construction and does not re-run percentile selection during the 2-step smoke run.
- The larger resumable multi-method speed matrix on H100 uses the same model-to-environment mapping listed in Section `0.0`.

Medium-model smoke test:

- Environment: `conda env = ciao`
- Command family: `medium_models/mezo.sh`
- Dataset/model: `SST-2`, `k=16`, `roberta-large`
- Shared knobs:
  - `STEP=2`
  - `EVAL_STEP=1`
  - `BS=1`
  - `LR=1e-6`
  - `EPS=1e-3`
  - `USE_H=False`
  - `ZERO_ORDER_USE_TRAINER_OPTIM=False`
  - `EFFICIENT_ZERO_ORDER=False`
  - `DATALOADER_SHUFFLE=False`

Run summaries:

| Method | Summary | Tail perf | Notes |
| --- | --- | --- | --- |
| MeZO | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/mezo/roberta-large/sst2/fp16/medium_models_result/smoke_mezo_base/seed0/run_summary.json) | `wallclock/step = 0.0753`, `samples/sec = 13.2879`, `max_gpu_memory_gb = 3.6459` | 2 forward passes / step |
| LOZO | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/lozo/roberta-large/sst2/fp16/medium_models_result/smoke_lozo/seed0/run_summary.json) | `wallclock/step = 0.1039`, `samples/sec = 9.6223`, `max_gpu_memory_gb = 1.5717` | slower than MeZO here, but much lower peak memory |
| HiZOO | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/hizoo/roberta-large/sst2/fp16/medium_models_result/smoke_hizoo/seed0/run_summary.json) | `wallclock/step = 0.1223`, `samples/sec = 8.1792`, `max_gpu_memory_gb = 4.0293` | expected slowdown from 3 forward passes / step |
| Sparse MeZO (`ratio=0.2`) | [run_summary.json](/scratch/jy03364/MeZO_/experiments/smoke/sparse_mezo/roberta-large/sst2/fp16/medium_models_result/smoke_sparse/seed0/run_summary.json) | `wallclock/step = 0.1004`, `samples/sec = 9.9573`, `max_gpu_memory_gb = 4.0274` | `active_fraction = 0.200077`, frozen mask |

Relative slowdown vs dense MeZO on this smoke test:

- LOZO: `1.381x` slower (`+28.67 ms/step`)
- HiZOO: `1.625x` slower (`+47.01 ms/step`)
- Sparse MeZO: `1.334x` slower (`+25.17 ms/step`)

Interpretation:

- HiZOO is slowest mainly because the method fundamentally needs one extra forward pass compared with MeZO.
- LOZO is not faster than dense MeZO on this tiny `roberta-large + bs=1 + 2-step` smoke setting because low-rank updates still execute large GEMMs, while dense MeZO here benefits from very cheap elementwise Gaussian directions.
- Sparse MeZO is still slower than dense MeZO in this local implementation even after the earlier mask-cache optimization, because the perturb/update path is still dense over the full parameter tensors and pays boolean-mask indexing overhead; the speedup claim in the paper depends on a more aggressively memory-optimized sparse path and larger-model settings.
