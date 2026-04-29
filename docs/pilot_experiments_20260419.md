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

## Current Experiment Contract

Effective `2026-04-20`, the following launch settings are part of the pilot
experiment contract for all new pilot submissions and reruns.

This section is normative for future pilot comparisons. Older historical runs
that do not satisfy these settings may still be useful for debugging, but they
should not be mixed into the main pilot comparison without an explicit note.

Reset note:

- all active exploratory `roberta-large / int8 / 50k` jobs and the unfinished
  `roberta-large / sparse_mezo / int4 / l40s` job were cancelled on
  `2026-04-20` before the next pilot relaunch
- these artifacts remain useful for debugging, but they are off-contract for
  the main pilot comparison

Global pilot requirements:

- fixed run seed:
  - `seed=16`
- fixed data-order seed:
  - `data_seed=16`
- training data order must be randomized:
  - `dataloader_shuffle=True`
- every run must declare its data regime explicitly:
  - `dataset_mode=full`
- current pilot comparisons should use the complete dataset only
- historical `paper_kshot` runs remain useful for debugging, but they are
  off-contract for the current pilot comparison table
- precision-specific default training budget:
  - `fp16`: `max_steps=25000`
  - `int8`: `max_steps=10000`

Immediate reset settings for the next `int8` pilot wave:

- scope:
  - `medium_models / roberta-large / {mezo,sparse_mezo} / {sst5,mnli} / int8`
- fixed training budget:
  - `max_steps=10000`
- fixed micro-batch size:
  - `per_device_train_batch_size=64`
- gradient accumulation:
  - `gradient_accumulation_steps=1`
- canonical `h` sweep:
  - every new pilot sweep must use exactly `8` values
  - the canonical grid is
    `{1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9}`
  - do not use the older mid-range `3e-*` pilot grid in new submissions
- learning-rate policy:
  - do not keep a single fixed `1e-6` as the only searched value
  - the next relaunch should sweep `learning_rate in {1e-6, 1e-7}`
  - `1e-6` remains the primary anchor because it is the current stable value in
    our codebase and is also inside the QuZO RoBERTa grid
  - do not increase above `1e-6` until a current `bs=64` run shows clean
    stability
- weight-decay policy:
  - for paper-aligned comparisons, sweep `weight_decay in {0, 0.1}`
  - if only one short diagnostic run is launched, start with `weight_decay=0`
- evaluation cadence:
  - `eval_steps=1000`
- QuZO probe cadence:
  - `zo_probe_every=200`
  - `zo_probe_num_seeds=16`
- guard policy:
  - keep only the shared `nan_guard`
  - `random_prediction_guard_enabled=False`
  - `zo_probe_health_guard_enabled=False`
  - probe logging may remain enabled for diagnostics, but probe-based early
    exits are off-contract for the current pilot protocol
- sampler requirement:
  - `DATALOADER_SHUFFLE=True` must be exported explicitly from the launcher
- dataset regime:
  - use the complete dataset, i.e. `dataset_mode=full`
  - current pilot submissions and reruns should be launched on full data unless
    an explicit debugging note says otherwise
  - historical low-resource `k-shot` runs should not be mixed into the main
    pilot comparison table

Important enforcement note:

- `medium_models/run.py` documents `dataloader_shuffle=True` as the intended
  training behavior, but legacy launcher layers still override it.
- therefore every new `medium_models` pilot submission must explicitly export
  `DATALOADER_SHUFFLE=True` rather than relying on implicit defaults.
- historical `medium_models` pilot artifacts with `dataloader_shuffle=False`,
  `max_steps=50000`, or unlabeled `dataset_mode=full` should be treated as
  off-contract for the current pilot protocol.

Current RoBERTa low-bit implementation note effective `2026-04-21`:

- `medium_models / roberta-large / {mezo,sparse_mezo} / {sst5,mnli,sst2} /
  int8` now uses one canonical low-bit probe path only:
  `w_base + Q(scale * eps * u_raw)`
- the old post-perturbation whole-parameter resnap path has been removed from
  the program for RoBERTa QuZO `int8/int4`
- current canonical RoBERTa low-bit probe semantics therefore do **not** use:
  `Q_all(w_base + Q(scale * eps * u_raw))`
- historical RoBERTa `int8` artifacts produced before this change should be
  treated as debug-only unless explicitly annotated otherwise
- implementation reference:
  [roberta_int8_implementation.md](/scratch/jy03364/MeZO_/docs/roberta_int8_implementation.md)

## Launch Verification Checklist

Each new pilot run should be checked immediately after startup.

Required metadata / log expectations:

- launch config should show:
  - `seed=16`
  - `data_seed=16`
  - `dataloader_shuffle=True`
  - `dataset_mode=full`
  - `random_prediction_guard_enabled=False`
  - `zo_probe_health_guard_enabled=False`
- for current RoBERTa QuZO `int8/int4` pilot runs, the metadata / config should
  also show:
  - `quzo_lowbit_probe_impl=q_w_plus_hz_resnap`
- for the reset `medium_models / roberta-large / int8` wave, launch config
  should also show:
  - `per_device_train_batch_size=64`
  - `max_steps=10000`
  - `zo_eps` drawn from `{1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9}`
  - `learning_rate=1e-6` or `learning_rate=1e-7`
- `medium_models` training logs should show:
  - `selected sampler=RandomSampler`
  - `training uses RandomSampler (shuffle).`
  - `args.data_seed=16 args.seed=16`
- for paper-aligned RoBERTa runs:
  - treat them as historical/debug-only unless explicitly reintroduced into the
    contract
- any `SequentialSampler` entry on a new `medium_models` pilot run should be
  treated as a contract violation and the run should be isolated from the main
  pilot comparison

Suggested quick checks:

- medium-model stderr / stdout:
  - `rg -n "dataloader_shuffle=|selected sampler=|training uses |args.data_seed=|args.seed=" <train.err or slurm log>`
- summary artifact:
  - `rg -n '"dataloader_shuffle": true|"data_seed": 16|"seed": 16|"max_steps": 10000|"per_device_train_batch_size": 64' <run_summary.json or summary.jsonl>`
- run metadata / stderr:
  - `rg -n 'quzo_lowbit_probe_impl|q_w_plus_hz_resnap' <run_metadata.json or train.err>`

Current RoBERTa pilot note on `2026-04-20`:

- use `per_device_train_batch_size=64` for current `roberta-large` pilot
  relaunches, including the `sparse_mezo / sst5 / fp16` diagnostic sweep
- use `max_steps=25000` for current `fp16` pilot relaunches unless the run is
  explicitly tagged as a short diagnostic

If a rerun intentionally deviates from this contract for debugging, the
deviation should be recorded explicitly in the run directory name and in the
submission note.

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
  - shard 1: `1e-2 1e-3`
  - shard 2: `1e-4 1e-5`
  - shard 3: `1e-6 1e-7`
  - shard 4: `1e-8 1e-9`
- lane layout:
  - lane 0:
    `roberta_sst5_mezo_int8 -> opt13b_mnli_mezo_int8 -> roberta_mnli_sparse_mezo_int8`
  - lane 1:
    `roberta_mnli_mezo_int8 -> roberta_sst5_sparse_mezo_int8`
- held for wave 2 because of submit quota:
  - `opt13b_sst5_sparse_mezo_int8`
  - `opt13b_mnli_sparse_mezo_int8`

Wave-2 command template after the current lane tails clear:

- `BASE_DEPENDENCY_JOB_IDS_BY_LANE='44586361:44586362:44586363:44586364,44586357:44586358:44586359:44586360' EXPERIMENT_START_INDEX=5 EXPERIMENT_LIMIT=2 bash experiments/pilot/_shared/h_sweep_8h/submit_int8_pilot_searches.sh`

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
| MeZO | RoBERTa-large | SST-5 | INT8 | submitted | resubmitted shard jobs `44586345-44586348` after fixing the launcher `--output` override; smoke-estimated `~5.66h` per 2-h shard | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| MeZO | RoBERTa-large | MNLI | INT8 | submitted | resubmitted shard jobs `44586349-44586352` after fixing the launcher `--output` override; few-shot smoke-estimated `~5.42h` per 2-h shard | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| MeZO | OPT-1.3B | SST-5 | INT8 | already usable, skipped | `summary.jsonl: 8 completed` | `experiments/pilot/mezo/opt-1.3b/sst5/int8/h_sweep_8h/results` |
| MeZO | OPT-1.3B | MNLI | INT8 | submitted | resubmitted shard jobs `44586353-44586356`, dependent on `44586345-44586348`; smoke-estimated `~6.08h` per 2-h shard | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | SST-5 | INT8 | submitted | resubmitted shard jobs `44586357-44586360`, dependent on `44586349-44586352`; smoke-estimated `~7.76h` per 2-h shard | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | RoBERTa-large | MNLI | INT8 | submitted | resubmitted shard jobs `44586361-44586364`, dependent on `44586353-44586356`; few-shot smoke-estimated `~6.23h` per 2-h shard | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | SST-5 | INT8 | blocked | smoke validated, but held for wave 2 after the launcher hit `QOSMaxSubmitJobPerUserLimit`; rough smoke estimate `~25.73h` per 2-h shard should be treated cautiously | `experiments/pilot/sparse_mezo/opt-1.3b/sst5/int8/h_sweep_8h/` |
| Sparse MeZO | OPT-1.3B | MNLI | INT8 | blocked | path smoke validated, but held for wave 2 after the launcher hit `QOSMaxSubmitJobPerUserLimit`; runtime estimate is still weak because the smoke was ultra-short | `experiments/pilot/sparse_mezo/opt-1.3b/mnli/int8/h_sweep_8h/` |

## Submitted Wave 1

Lane 0:

| Label | Job IDs | Depends on | Script |
|---|---|---|---|
| `roberta_sst5_mezo_int8` | `44586345, 44586346, 44586347, 44586348` | `-` | `experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_mezo_int8_8h.sh` |
| `opt13b_mnli_mezo_int8` | `44586353, 44586354, 44586355, 44586356` | `afterok:44586345:44586346:44586347:44586348` | `experiments/pilot/mezo/opt-1.3b/mnli/int8/h_sweep_8h/jobs/opt13b_mnli_mezo_int8_8h.sh` |
| `roberta_mnli_sparse_mezo_int8` | `44586361, 44586362, 44586363, 44586364` | `afterok:44586353:44586354:44586355:44586356` | `experiments/pilot/sparse_mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_sparse_mezo_int8_8h.sh` |

Lane 1:

| Label | Job IDs | Depends on | Script |
|---|---|---|---|
| `roberta_mnli_mezo_int8` | `44586349, 44586350, 44586351, 44586352` | `-` | `experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/jobs/roberta_mnli_mezo_int8_8h.sh` |
| `roberta_sst5_sparse_mezo_int8` | `44586357, 44586358, 44586359, 44586360` | `afterok:44586349:44586350:44586351:44586352` | `experiments/pilot/sparse_mezo/roberta-large/sst5/int8/h_sweep_8h/jobs/roberta_sst5_sparse_mezo_int8_8h.sh` |

Old 8h relaunch attempt:

- `44586202-44586209` were immediately cancelled after a launcher bug was found:
  logging text from `submit_experiment_group` had polluted the dependency string
  passed to `sbatch`
- `44586231-44586255` were cancelled before start after a second launcher bug was
  found: the shard launcher had overridden `sbatch --output` with a relative
  path, which Slurm combined with each job's `--chdir` into a duplicated nested
  log directory; the corrected replacement wave is `44586345-44586364`

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
  - `tail -f experiments/pilot/mezo/roberta-large/sst5/int8/h_sweep_8h/logs/slurm_roberta_sst5_mezo_int8_s1_44586345.out`
- Lane-1 head shard:
  - `tail -f experiments/pilot/mezo/roberta-large/mnli/int8/h_sweep_8h/logs/slurm_roberta_mnli_mezo_int8_s1_44586349.out`
- Launcher logs:
  - `tail -f experiments/pilot/_shared/h_sweep_8h/logs/submit_int8_pilot_20260420_013454.log`
