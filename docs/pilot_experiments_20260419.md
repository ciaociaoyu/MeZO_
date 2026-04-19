# Pilot Experiment Status: 2026-04-19

This file is the source-of-truth status sheet for the pilot experiments only.

Pilot scope:

- Models: `opt-1.3b`, `roberta-large`
- Baselines: `mezo`, `sparse_mezo`
- Tasks: `sst5`, `mnli`
- Precisions:
  - `fp16` -> `h_sweep_14h`
  - `int8` -> `h_sweep_8h`

This yields `16` pilot combinations in total.

Out of scope for this document:

- `lozo`
- `hizoo`
- `mistral-7b`
- `boolq`
- `int4`
- `speed_bench_h100`

## Directory Layout

After the recent cleanup and reorganization, the current canonical roots are:

- FP16 pilot sweeps:
  - `experiments/main/<baseline>/<model>/<task>/fp16/h_sweep_14h/`
- INT8 pilot sweeps:
  - `experiments/pilot/<baseline>/<model>/<task>/int8/h_sweep_8h/`

Status interpretation:

- If `summary.jsonl` exists, it is treated as the primary status record.
- Residual directories without a valid `summary.jsonl` are not counted as completed.
- `jobs/` without `results/summary.jsonl` means the pilot combination is currently missing runnable output.

## Current Summary

As of `2026-04-19`:

- `5 / 16` pilot combinations are currently usable and completed.
- `4 / 16` are partial or incomplete.
- `7 / 16` are missing or were intentionally deleted during cleanup.
- No pilot Slurm jobs are currently active; `squeue` shows only the interactive session.

## Matrix Status

| Baseline | Model | Task | Precision | Sweep | Status | Evidence | Canonical path |
|---|---|---|---|---|---|---|---|
| MeZO | RoBERTa-large | SST-5 | FP16 | `h_sweep_14h` | Completed | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | MNLI | FP16 | `h_sweep_14h` | Completed | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | SST-5 | FP16 | `h_sweep_14h` | Completed | `summary.jsonl: 14 completed` | `experiments/main/mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | MNLI | FP16 | `h_sweep_14h` | Partial | `summary.jsonl: 12 completed` | `experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | SST-5 | FP16 | `h_sweep_14h` | Partial | `summary.jsonl: 3 completed + 3 skipped_nan_guard` | `experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | MNLI | FP16 | `h_sweep_14h` | Partial | `summary.jsonl: 2 completed + 3 skipped_nan_guard` | `experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | SST-5 | FP16 | `h_sweep_14h` | Completed | `summary.jsonl: 14 completed` | `experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | MNLI | FP16 | `h_sweep_14h` | Incomplete residue | `summary.jsonl` is empty; only a residual `h_1e-8` directory remains | `experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | SST-5 | INT8 | `h_sweep_8h` | Deleted bad result | Old medium-model probe metrics were invalid; `results/` and `logs/` were removed during cleanup | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| MeZO | RoBERTa-large | MNLI | INT8 | `h_sweep_8h` | Deleted bad result | Old medium-model probe metrics were invalid; `results/` and `logs/` were removed during cleanup | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| MeZO | OPT-1.3B | SST-5 | INT8 | `h_sweep_8h` | Completed | `summary.jsonl: 8 completed` | `experiments/pilot/mezo/opt-1.3b/sst5/int8/h_sweep_8h/results` |
| MeZO | OPT-1.3B | MNLI | INT8 | `h_sweep_8h` | Deleted stale partial | Old partial outputs were removed during cleanup; only `jobs/` remains | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | SST-5 | INT8 | `h_sweep_8h` | Deleted stale partial | Stale partial outputs were removed during cleanup; only `jobs/` remains | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | MNLI | INT8 | `h_sweep_8h` | Deleted stale partial | Stale partial outputs were removed during cleanup; only `jobs/` remains | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | SST-5 | INT8 | `h_sweep_8h` | Missing | `jobs/` exists but no valid `results/summary.jsonl` exists | `experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | MNLI | INT8 | `h_sweep_8h` | Missing | `jobs/` exists but no valid `results/summary.jsonl` exists | `experiments/pilot/sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |

## Cleanup Notes

The following pilot outputs were intentionally removed and should not be treated as usable results:

- `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/results`
- `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/logs`
- `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/results`
- `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/logs`
- `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/results`
- `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/results`
- `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/results`

Reason summary:

- The old `roberta-large` INT8 MeZO pilot results were generated before the `medium_models` directional probe consistency fix and therefore had unreliable `zo_directional_probe.csv` metrics such as `mse`, `corr`, and `sign_acc`.
- The removed `opt-1.3b` MNLI INT8 and sparse `roberta-large` INT8 outputs were stale partial artifacts and do not represent complete pilot sweeps.

## Recommended Next Pilot Runs

If the pilot matrix is to be completed from the current repository state, the missing highest-priority reruns are:

1. `MeZO / RoBERTa-large / SST-5 / INT8`
2. `MeZO / RoBERTa-large / MNLI / INT8`
3. `MeZO / OPT-1.3B / MNLI / INT8`
4. `Sparse MeZO / RoBERTa-large / SST-5 / INT8`
5. `Sparse MeZO / RoBERTa-large / MNLI / INT8`
6. `Sparse MeZO / OPT-1.3B / SST-5 / INT8`
7. `Sparse MeZO / OPT-1.3B / MNLI / INT8`
8. `MeZO / OPT-1.3B / MNLI / FP16`
9. `Sparse MeZO / RoBERTa-large / SST-5 / FP16`
10. `Sparse MeZO / RoBERTa-large / MNLI / FP16`
11. `Sparse MeZO / OPT-1.3B / MNLI / FP16`
