# Low-Bit Sparse/Prefix Multi-Seed Rerun Audit

Date: 2026-06-28

## What Failed

The previous V10 low-bit sparse/prefix multi-seed array
`v10_medium_supplement_results_20260627` failed before low-bit training for the
new seed rows. The Slurm error logs show `FileNotFoundError` while loading
full-data splits:

- `medium_models/data/k-shot-1k-test/sst-2/full-32/train.tsv`
- `medium_models/data/k-shot-1k-test/trec/full-32/train.csv`

The six low-bit lanes were ordered with a prefix row first, so the first missing
prefix split stopped each lane before the following sparse rows could run.

The earlier `failed_submission_46453535` error
`--checkpoint_probe_num_directions must be > 0` belongs to the superseded
high-precision submission, not the low-bit sparse/prefix failure being rerun
here.

## Root Cause

The low-bit runner resolves task data with a lightweight alias resolver in
`tools/smoke_rtnclip_roberta_sst5.py`. It checks alias directories directly and
does not call the newer `medium_models/src/data_utils.py` auto-materialization
path. For GLUE-style tasks, generated full splits initially went to canonical
uppercase task folders, while the low-bit resolver checks lowercase aliases
first.

## Fix Applied Before Rerun

Materialized missing full-data seed splits for seeds 32 and 64:

- `sst-2/full-32`, `sst-2/full-64`
- `trec/full-32`, `trec/full-64`
- `rte/full-32`, `rte/full-64`
- `sst-5/full-32`, `sst-5/full-64`

The splits were generated with the project utility
`medium_models/tools/generate_k_shot_data.py` / `materialize_task_data`, using
`dataset_mode=full` and `full_dev_ratio=0.1`. This preserves the intended
data_seed variation instead of copying seed16.

## Rerun Scope

Only the failed low-bit multi-seed rows are rerun:

- prefix INT4: SST-2, TREC
- sparse p=0.1 INT4: SST-5, RTE
- policies: fixed-small, MeZO default, V10 reference h
- seeds/data_seeds: 32, 64

Total: 24 runs.

The previous failed output directory is left untouched. New outputs go under:

`v10_lowbit_sparse_prefix_multiseed_rerun_20260628/`

