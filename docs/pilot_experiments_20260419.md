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

As of `2026-04-19` after the guard rollout and shard-based relaunch work:

- `5 / 16` combinations are already usable and were intentionally skipped.
- `5 / 16` combinations are now resubmitted as INT8 shard jobs.
- `2 / 16` INT8 combinations have passed short smoke validation but are held for
  wave 2 because the cluster rejected a larger fan-out with
  `QOSMaxSubmitJobPerUserLimit`.
- `4 / 16` FP16 combinations are still stopped and have not yet been relaunched
  under the new shard strategy.

What changed on `2026-04-19`:

- all in-flight pilot jobs touching the new guard/probe functionality were
  cancelled before revalidation
- every remaining INT8 pilot path was smoke-validated before any relaunch
- the 8h launcher was changed from a single serial chain to a shard scheduler:
  `4` shard jobs per experiment, `2` concurrent lanes, lane-local `afterok`
  dependencies
- the launcher now supports partial waves via
  `EXPERIMENT_START_INDEX` / `EXPERIMENT_LIMIT` and per-lane dependency seeds
  via `BASE_DEPENDENCY_JOB_IDS_BY_LANE`
- summary / manifest row deletion is now lock-protected so concurrent shard jobs
  do not race when rewriting shared `summary.jsonl` / `manifest.jsonl`
- the medium few-shot launcher bug for lowercase `mnli` was fixed

Guard / path hardening on `2026-04-19`:

- added `experiments/audit_script_paths.py` and verified all `experiments/**/*.sh`
  resolve against the repo layout
- added fail-fast path checks to the shared 8h runners and the resubmitted 14h
  job scripts so missing helpers or guard scripts now stop immediately with an
  explicit `[path-check]` error instead of silently empty-running
- the old 8h random-prediction guard was relaxed from a single rigid
  “still-random at step 1000” test into a trend-aware check:
  only skip when recent evals remain random-like and show no meaningful
  improvement
- on `2026-04-20`, the same trend-aware random-prediction guard was wired into
  `large_models` so the OPT-family pilot path now supports the same
  classification safety check as `medium_models`; current explicit task
  coverage is `SST-5`, `MNLI`, and `BoolQ`
- added a separate probe-health guard so repeated probe failures can skip an `h`
  without pretending training itself is numerically healthy

QuZO probe validation reference:

- [quzo_probe_smoke_20260419.md](/scratch/jy03364/MeZO_/docs/quzo_probe_smoke_20260419.md)

INT8 wave-1 submission strategy:

- script:
  - `experiments/pilot/_shared/h_sweep_8h/submit_int8_pilot_searches.sh`
- shard layout:
  - shard 1: `1e-6 3e-6`
  - shard 2: `1e-5 3e-5`
  - shard 3: `1e-4 3e-4`
  - shard 4: `1e-3 3e-3`
- lane layout:
  - lane 0:
    `roberta_sst5_mezo_int8 -> opt13b_mnli_mezo_int8 -> roberta_mnli_sparse_mezo_int8`
  - lane 1:
    `roberta_mnli_mezo_int8 -> roberta_sst5_sparse_mezo_int8`
- held for wave 2 because of submit quota:
  - `opt13b_sst5_sparse_mezo_int8`
  - `opt13b_mnli_sparse_mezo_int8`

Wave-2 command template after the current lane tails clear:

- `BASE_DEPENDENCY_JOB_IDS_BY_LANE='44586252:44586253:44586254:44586255,44586243:44586244:44586245:44586246' EXPERIMENT_START_INDEX=5 EXPERIMENT_LIMIT=2 bash experiments/pilot/_shared/h_sweep_8h/submit_int8_pilot_searches.sh`

## INT8 Smoke Validation

These short runs were used only to validate the new guard / probe path before
relaunch. They are not canonical full-pilot results.

Additional guard regression checks completed on `2026-04-20`:

- `conda run -n mezo-env python -m unittest large_models.tests.test_random_prediction_guard`
  passed after wiring the large-model trend guard into the existing evaluation
  callback
- `conda run -n mezo-env python -m unittest large_models.tests.test_quzo_probe_direction`
  still passes, so the QuZO direction probe unit check on the large path
  survived the guard changes
- `conda run -n ciao python -m unittest medium_models.tests.test_directional_probe_consistency`
  still passes in the documented medium-model environment

| Combination | Smoke evidence | Notes |
|---|---|---|
| MeZO / RoBERTa-large / SST-5 / INT8 | `experiments/smoke/mezo/roberta-large/sst5/int8/guard_validation_20260419/.../summary.jsonl` | full-mode smoke completed; no random/probe guard trigger |
| MeZO / RoBERTa-large / MNLI / INT8 | `experiments/smoke/mezo/roberta-large/mnli/int8/guard_validation_fs_20260419/.../summary.jsonl` | few-shot smoke used after fixing lowercase `mnli` template bug |
| MeZO / OPT-1.3B / MNLI / INT8 | `experiments/smoke/mezo/opt-1.3b/mnli/int8/guard_validation_20260419/.../summary.jsonl` | full-mode smoke completed |
| Sparse MeZO / RoBERTa-large / SST-5 / INT8 | `experiments/smoke/sparse_mezo/roberta-large/sst5/int8/guard_validation_20260419/.../summary.jsonl` | full-mode smoke completed |
| Sparse MeZO / RoBERTa-large / MNLI / INT8 | `experiments/smoke/sparse_mezo/roberta-large/mnli/int8/guard_validation_fs_20260419/.../summary.jsonl` | few-shot smoke completed |
| Sparse MeZO / OPT-1.3B / SST-5 / INT8 | `experiments/smoke/sparse_mezo/opt-1.3b/sst5/int8/guard_validation_20260419/.../summary.jsonl` | full-mode smoke completed; slower than the other INT8 paths |
| Sparse MeZO / OPT-1.3B / MNLI / INT8 | `experiments/smoke/sparse_mezo/opt-1.3b/mnli/int8/guard_validation_20260419/.../summary.jsonl` | ultra-short path smoke only; path/summary creation validated, runtime estimate still weak |

## Matrix Status

| Baseline | Model | Task | Precision | Status | Evidence | Canonical path |
|---|---|---|---|---|---|---|
| MeZO | RoBERTa-large | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | MNLI | FP16 | already usable, skipped | `summary.jsonl: 9 completed + 5 skipped_nan_guard` | `experiments/main/mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 14 completed` | `experiments/main/mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| MeZO | OPT-1.3B | MNLI | FP16 | blocked | historical state: `summary.jsonl: 12 completed`; old relaunches were stopped during the guard/probe audit and this 14h path has not yet been re-sharded | `experiments/main/mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | SST-5 | FP16 | blocked | historical state: `3 completed + 3 skipped_nan_guard`; stopped during the guard/probe audit; not yet relaunched under the new shard strategy | `experiments/main/sparse_mezo/roberta-large/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | RoBERTa-large | MNLI | FP16 | blocked | historical state: `2 completed + 3 skipped_nan_guard`; stopped during the guard/probe audit; not yet relaunched under the new shard strategy | `experiments/main/sparse_mezo/roberta-large/mnli/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | SST-5 | FP16 | already usable, skipped | `summary.jsonl: 14 completed` | `experiments/main/sparse_mezo/opt-1.3b/sst5/fp16/h_sweep_14h/results` |
| Sparse MeZO | OPT-1.3B | MNLI | FP16 | blocked | historical state: empty `summary.jsonl` residue; stopped during the guard/probe audit; not yet relaunched under the new shard strategy | `experiments/main/sparse_mezo/opt-1.3b/mnli/fp16/h_sweep_14h/results` |
| MeZO | RoBERTa-large | SST-5 | INT8 | submitted | wave 1 shard jobs `44586231-44586234`; smoke-estimated `~5.66h` per 2-h shard | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| MeZO | RoBERTa-large | MNLI | INT8 | submitted | wave 1 shard jobs `44586235-44586238`; few-shot smoke-estimated `~5.42h` per 2-h shard | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| MeZO | OPT-1.3B | SST-5 | INT8 | already usable, skipped | `summary.jsonl: 8 completed` | `experiments/pilot/mezo/opt-1.3b/sst5/int8/h_sweep_8h/results` |
| MeZO | OPT-1.3B | MNLI | INT8 | submitted | wave 1 shard jobs `44586239-44586242`, dependent on `44586231-44586234`; smoke-estimated `~6.08h` per 2-h shard | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | SST-5 | INT8 | submitted | wave 1 shard jobs `44586243-44586246`, dependent on `44586235-44586238`; smoke-estimated `~7.76h` per 2-h shard | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | MNLI | INT8 | submitted | shard jobs `44586252-44586255`, dependent on `44586239-44586242`; few-shot smoke-estimated `~6.23h` per 2-h shard | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | SST-5 | INT8 | blocked | smoke validated, but held for wave 2 after the launcher hit `QOSMaxSubmitJobPerUserLimit`; rough smoke estimate `~25.73h` per 2-h shard should be treated cautiously | `experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | MNLI | INT8 | blocked | path smoke validated, but held for wave 2 after the launcher hit `QOSMaxSubmitJobPerUserLimit`; runtime estimate is still weak because the smoke was ultra-short | `experiments/pilot/sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |

## Submitted Wave 1

Lane 0:

| Label | Job IDs | Depends on | Script |
|---|---|---|---|
| `roberta_sst5_mezo_int8` | `44586231, 44586232, 44586233, 44586234` | `-` | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_mezo_int8_8h.sh` |
| `opt13b_mnli_mezo_int8` | `44586239, 44586240, 44586241, 44586242` | `afterok:44586231:44586232:44586233:44586234` | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_mezo_int8_8h.sh` |
| `roberta_mnli_sparse_mezo_int8` | `44586252, 44586253, 44586254, 44586255` | `afterok:44586239:44586240:44586241:44586242` | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int8_8h.sh` |

Lane 1:

| Label | Job IDs | Depends on | Script |
|---|---|---|---|
| `roberta_mnli_mezo_int8` | `44586235, 44586236, 44586237, 44586238` | `-` | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_mezo_int8_8h.sh` |
| `roberta_sst5_sparse_mezo_int8` | `44586243, 44586244, 44586245, 44586246` | `afterok:44586235:44586236:44586237:44586238` | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_int8_8h.sh` |

Old 8h relaunch attempt:

- `44586202-44586209` were immediately cancelled after a launcher bug was found:
  logging text from `submit_experiment_group` had polluted the dependency string
  passed to `sbatch`

## Risk Notes

- `QOSMaxSubmitJobPerUserLimit` is now the main scheduling bottleneck for the
  shard plan.
- The 8h launcher bug that corrupted dependency strings is fixed, but only after
  cancelling the first partial attempt.
- `Sparse MeZO / OPT-1.3B / SST-5 / INT8` looks materially slower than the other
  INT8 paths in smoke; it is intentionally held for wave 2 rather than being
  mixed into the first submission wave.
- The RoBERTa MNLI INT8 smokes used few-shot mode to verify the path after the
  lowercase `mnli` bug fix. They are good enough for launcher validation, but
  their runtime estimates are weaker than the full-mode SST-5 numbers.
- The four remaining FP16 pilot combinations are still paused; they need a
  separate 14h sharding/resume pass.

## Immediate Monitor Commands

- Queue status:
  - `squeue -u jy03364 -o '%.18i %.9P %.40j %.8T %.10M %.9l %.20R'`
- Lane-0 head shard:
  - `tail -f experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/logs/slurm_roberta_sst5_mezo_int8_s1_44586231.out`
- Lane-1 head shard:
  - `tail -f experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/logs/slurm_roberta_mnli_mezo_int8_s1_44586235.out`
- Launcher logs:
  - `tail -f experiments/pilot/_shared/h_sweep_8h/logs/submit_int8_pilot_20260419_234943.log`
