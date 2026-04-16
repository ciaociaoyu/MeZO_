# Experiment Record: 2026-04-16

This file is a lightweight status snapshot of the main experiments currently active in this repository.

## 1. Currently Running

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
  - `completed`: `1`
  - `skipped_nan_guard`: `4`
- Last recorded finished `h`: `1e-6`

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
