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
  - `44584419 -> 44584425`
- FP16 chain:
  - `44585137 -> 44585140`
  - prior chain `44584426 -> 44584429` was cancelled after guard audit:
    `44584426` had no `nan_guard` wrapper in the submitted batch script;
    `44584427 -> 44584429` pointed at a nonexistent local `nan_guard.py`

Path audit hardening on `2026-04-19`:

- added `experiments/audit_script_paths.py` and verified all `experiments/**/*.sh`
  resolve against the repo layout
- added fail-fast path checks to the shared 8h runners and the resubmitted 14h
  job scripts so missing helpers or guard scripts now stop immediately with an
  explicit `[path-check]` error instead of silently empty-running
- fixed `experiments/pilot/_shared/speed_bench_h100/run_zo_method_speed_matrix.py`,
  which had computed `REPO_ROOT` incorrectly and would otherwise generate bad
  launcher paths under `experiments/pilot/...`

## Matrix Status

| Baseline | Model | Task | Precision | Status | Evidence | Canonical path |
|---|---|---|---|---|---|---|
| MeZO | RoBERTa-large | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | MNLI | FP16 | already usable, skipped | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 14 completed` | `experiments/main/mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | MNLI | FP16 | submitted | historical state: `summary.jsonl: 12 completed`; bad run `44584426` was cancelled because the submitted batch script bypassed `nan_guard`; resubmitted as `44585137` | `experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | SST-5 | FP16 | submitted | historical state: `3 completed + 3 skipped_nan_guard`; cancelled pending job `44584427` had a broken local guard path; resubmitted as `44585138` | `experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | MNLI | FP16 | submitted | historical state: `2 completed + 3 skipped_nan_guard`; cancelled pending job `44584428` had a broken local guard path; resubmitted as `44585139` | `experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 14 completed` | `experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | MNLI | FP16 | submitted | historical state: empty `summary.jsonl` residue; cancelled pending job `44584429` had a broken local guard path; resubmitted as `44585140` | `experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | SST-5 | INT8 | submitted | old bad probe outputs were deleted; QuZO smoke validated on A100; submitted as job `44584419` | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| MeZO | RoBERTa-large | MNLI | INT8 | submitted | old bad probe outputs were deleted; submitted as job `44584420` | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| MeZO | OPT-1.3B | SST-5 | INT8 | already usable, skipped | `summary.jsonl: 8 completed` | `experiments/pilot/mezo/opt-1.3b/sst5/int8/h_sweep_8h/results` |
| MeZO | OPT-1.3B | MNLI | INT8 | submitted | stale partial output had been deleted earlier; submitted as job `44584421` | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | SST-5 | INT8 | submitted | stale partial output had been deleted earlier; submitted as job `44584422` | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | MNLI | INT8 | submitted | stale partial output had been deleted earlier; submitted as job `44584423` | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | SST-5 | INT8 | submitted | previously missing; submitted as job `44584424` | `experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | MNLI | INT8 | submitted | previously missing; submitted as job `44584425` | `experiments/pilot/sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |

## Submitted Chains

INT8 chain:

| Order | Label | Job ID | Depends on | Script |
|---|---|---:|---:|---|
| 1 | `int8_roberta_sst5_mezo` | `44584419` | `-` | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_mezo_int8_8h.sh` |
| 2 | `int8_roberta_mnli_mezo` | `44584420` | `44584419` | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_mezo_int8_8h.sh` |
| 3 | `int8_opt_mnli_mezo` | `44584421` | `44584420` | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_mezo_int8_8h.sh` |
| 4 | `int8_roberta_sst5_sparse_mezo` | `44584422` | `44584421` | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_int8_8h.sh` |
| 5 | `int8_roberta_mnli_sparse_mezo` | `44584423` | `44584422` | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int8_8h.sh` |
| 6 | `int8_opt_sst5_sparse_mezo` | `44584424` | `44584423` | `experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/jobs/opt13b_sst5_sparse_mezo_int8_8h.sh` |
| 7 | `int8_opt_mnli_sparse_mezo` | `44584425` | `44584424` | `experiments/pilot/sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_sparse_mezo_int8_8h.sh` |

FP16 chain:

| Order | Label | Job ID | Depends on | Script |
|---|---|---:|---:|---|
| 1 | `fp16_opt_mnli_mezo` | `44585137` | `-` | `experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/jobs/opt13b_mnli_quzo16_14h.sh` |
| 2 | `fp16_roberta_sst5_sparse_mezo` | `44585138` | `44585137` | `experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/jobs/roberta_sst5_sparse_mezo16_14h.sh` |
| 3 | `fp16_roberta_mnli_sparse_mezo` | `44585139` | `44585138` | `experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/jobs/roberta_mnli_sparse_mezo16_14h.sh` |
| 4 | `fp16_opt_mnli_sparse_mezo` | `44585140` | `44585139` | `experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/jobs/opt13b_mnli_sparse_mezo16_14h.sh` |

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
  - `tail -f experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/logs/slurm_hsweep8h_mezo_int8_roberta_sst5_44584419.out`
- OPT FP16 resume job:
  - `tail -f experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/logs/slurm_hsweep14h_quzo16_opt13b_mnli_44585137.out`
