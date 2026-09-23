# OPT-2.7B radius extension

Status: local smoke passed; formal probes and L4 training launched.
Live state is in STATUS.md, run_manifest.csv, summaries/, and logs/.
See METHOD_LOCK.md and configs/ for the predeclared protocol. Original prompts,
token ids, full split sizes and model revision are saved in each run directory.
48 full training configurations run on at most four L4 workers as authorized
on 2026-09-22; the local A100 completed the two expensive multi-precision
true-gradient probe jobs. Keeping the
training matrix on L4 avoids architecture-dependent CUDA RNG stream differences.
Probe batches/directions are not training seeds.

Checkpoints are atomic and replaced every500 steps. Records beyond the latest
checkpoint are rolled back on resume. Both data and direction streams use step
indexed seeds. Each run has an exclusive filesystem lock to avoid duplication.
Summaries and PDF/PNG figures are regenerated from logs by the aggregate command.
No empirical result has been substituted for a theoretical prediction.

## Active allocation and recovery

L4 queue: arrays48421053 (tasks0-1) and48464008 (tasks2-3), at most four
simultaneous training runs,72-hour allocations with checkpoint/requeue at the
walltime warning. CPU aggregation job48421054 waits for both arrays. The local
A100 allocation48419105 completed the expensive probes and smoke tests, not
paired full training.
Initial L4 timing is approximately1.4-1.6 seconds/FP16 step. One20k FP16 run
therefore needs roughly8-9 hours plus full evaluation and checkpoint overhead.
The entire48-run matrix requires multiple days, not a single72-hour allocation.
Low-bit/RTE walltime will be estimated from their own measured runtimes.

Two infrastructure issues were fixed before leaving the queue unattended:
cross-node exclusion now uses atomic directory claims on Lustre; a missing
plot dependency is installed, and reporting failure cannot kill a training
worker. Superseded logs and uncommitted records remain archived for audit.
See EXPERIMENT_LOG.md. No scientific configuration was changed during recovery.

## Commands

Python environment: /home/jy03364/miniconda3/envs/mezo-mistral/bin/python.
Run from the repository root, with ROOT set to this output directory:

```bash
python tools/opt27b_radius_extension.py selftest --root "$ROOT"
python tools/opt27b_radius_extension.py smoke-suite --root "$ROOT"
python tests/test_opt27b_queue_claim.py
python tools/opt27b_radius_extension.py aggregate --root "$ROOT"
```

The already-running monitor writes queue snapshots every five minutes. Do not
start another monitor or resubmit while the current arrays are active. Exact
submission commands are in l4_submission.json, l4_submission_extra.json, and
logs/controller.jsonl.
Each model invocation records its own code hash and configuration.

## Half-hour maintenance

The user-authorized30-minute watchdog is now enabled alongside5-minute status
snapshots. See WATCHDOG.md and WATCHDOG_LATEST.md. A bounded diagnostic agent
is invoked on health alerts; scientific settings remain locked. CPU backup
checks continue if local monitoring is unavailable, without allocating a GPU.
