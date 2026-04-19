# Pilot Experiment Status: 2026-04-19

This file is the source-of-truth status sheet for the pilot experiments.

## Pilot Scope

- Models:
  - `opt-1.3b`
  - `roberta-large`
- Baselines:
  - `mezo`
  - `sparse_mezo`
- Tasks:
  - `sst5`
  - `mnli`
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

## Canonical Layout

- FP16 pilot sweeps:
  - `experiments/main/<baseline>/<model>/<task>/fp16/h_sweep_14h/`
- INT8 pilot sweeps:
  - `experiments/pilot/<baseline>/<model>/<task>/int8/h_sweep_8h/`

Status interpretation:

- `completed`
  - canonical result already exists and is usable
- `already usable, skipped`
  - completed result exists and was intentionally not resubmitted
- `submitted`
  - the combination has been submitted on `2026-04-19` and is now pending or waiting on dependency
- `blocked`
  - not submitted because of a concrete blocker

## Current Summary

As of `2026-04-19` after QuZO probe validation and resubmission:

- `5 / 16` combinations are already usable and were intentionally skipped.
- `11 / 16` remaining combinations have been submitted.
- `0 / 16` combinations are currently blocked.

QuZO probe validation reference:

- [quzo_probe_smoke_20260419.md](/scratch/jy03364/MeZO_/docs/quzo_probe_smoke_20260419.md)

Submission strategy:

- INT8 chain:
  - `44583619 -> 44583625`
- FP16 chain:
  - `44583626 -> 44583629`

## Matrix Status

| Baseline | Model | Task | Precision | Status | Evidence | Canonical path |
|---|---|---|---|---|---|---|
| MeZO | RoBERTa-large | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | MNLI | FP16 | already usable, skipped | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 14 completed` | `experiments/main/mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | MNLI | FP16 | submitted | historical state: `summary.jsonl: 12 completed`; resumed via job `44583626` | `experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | SST-5 | FP16 | submitted | historical state: `3 completed + 3 skipped_nan_guard`; resumed via job `44583627` | `experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | MNLI | FP16 | submitted | historical state: `2 completed + 3 skipped_nan_guard`; resumed via job `44583628` | `experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 14 completed` | `experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | MNLI | FP16 | submitted | historical state: empty `summary.jsonl` residue; submitted as job `44583629` | `experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | SST-5 | INT8 | submitted | old bad probe outputs were deleted; QuZO smoke validated on A100; submitted as job `44583619` | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| MeZO | RoBERTa-large | MNLI | INT8 | submitted | old bad probe outputs were deleted; submitted as job `44583620` | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| MeZO | OPT-1.3B | SST-5 | INT8 | already usable, skipped | `summary.jsonl: 8 completed` | `experiments/pilot/mezo/opt-1.3b/sst5/int8/h_sweep_8h/results` |
| MeZO | OPT-1.3B | MNLI | INT8 | submitted | stale partial output had been deleted earlier; submitted as job `44583621` | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | SST-5 | INT8 | submitted | stale partial output had been deleted earlier; submitted as job `44583622` | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | MNLI | INT8 | submitted | stale partial output had been deleted earlier; submitted as job `44583623` | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | SST-5 | INT8 | submitted | previously missing; submitted as job `44583624` | `experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | MNLI | INT8 | submitted | previously missing; submitted as job `44583625` | `experiments/pilot/sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |

## Submitted Chains

INT8 chain:

| Order | Label | Job ID | Depends on | Script |
|---|---|---:|---:|---|
| 1 | `int8_roberta_sst5_mezo` | `44583619` | `-` | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_mezo_int8_8h.sh` |
| 2 | `int8_roberta_mnli_mezo` | `44583620` | `44583619` | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_mezo_int8_8h.sh` |
| 3 | `int8_opt_mnli_mezo` | `44583621` | `44583620` | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_mezo_int8_8h.sh` |
| 4 | `int8_roberta_sst5_sparse_mezo` | `44583622` | `44583621` | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_int8_8h.sh` |
| 5 | `int8_roberta_mnli_sparse_mezo` | `44583623` | `44583622` | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int8_8h.sh` |
| 6 | `int8_opt_sst5_sparse_mezo` | `44583624` | `44583623` | `experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/jobs/opt13b_sst5_sparse_mezo_int8_8h.sh` |
| 7 | `int8_opt_mnli_sparse_mezo` | `44583625` | `44583624` | `experiments/pilot/sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_sparse_mezo_int8_8h.sh` |

FP16 chain:

| Order | Label | Job ID | Depends on | Script |
|---|---|---:|---:|---|
| 1 | `fp16_opt_mnli_mezo` | `44583626` | `-` | `experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/jobs/opt13b_mnli_14h.sh` |
| 2 | `fp16_roberta_sst5_sparse_mezo` | `44583627` | `44583626` | `experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/jobs/roberta_sst5_sparse_mezo16_14h.sh` |
| 3 | `fp16_roberta_mnli_sparse_mezo` | `44583628` | `44583627` | `experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/jobs/roberta_mnli_sparse_mezo16_14h.sh` |
| 4 | `fp16_opt_mnli_sparse_mezo` | `44583629` | `44583628` | `experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/jobs/opt13b_mnli_sparse_mezo16_14h.sh` |

## Risk Notes

- `MeZO / INT8 / QuZO`:
  - A100 smoke estimates roughly `40-56h` for a full 8-point sweep
  - the production scripts request `H100`, so these should remain within the current `72h` budget
- `Sparse MeZO / RoBERTa / FP16`:
  - this remains the highest timeout-risk family
  - historical runs only completed a small subset of `h` values before hitting `skipped_nan_guard` or timeout
- `Sparse MeZO / RoBERTa / INT8`:
  - not smoke-validated here
  - still submitted because the 8-point sweep is smaller than the problematic 14-point FP16 sweep, but it should be watched closely

## Immediate Monitor Commands

- Queue status:
  - `squeue -u jy03364 -o '%.18i %.9P %.40j %.8T %.10M %.9l %.20R'`
- RoBERTa INT8 first job:
  - `tail -f experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/logs/slurm_hsweep8h_mezo_int8_roberta_sst5_44583619.out`
- OPT FP16 resume job:
  - `tail -f experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/logs/slurm_hsweep14h_opt13b_mnli_44583626.out`
