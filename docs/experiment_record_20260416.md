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
