# Experiment log

## Concurrency increase, 2026-09-22 22:18 EDT

- User authorized up to four concurrent tasks. Existing L4 array48421053
  continues with two one-GPU workers; submitted array48464008, tasks2-3,
  for two more one-L4 queue workers. All workers claim from the same unchanged
  jobs.json with atomic cross-node directory claims. No training config or
  scientific setting changed. New tasks initially PENDING (Priority), so
  concurrent GPU use remained two at submission time.
- The half-hour watchdog now recognizes both declared arrays and enforces a
  four-worker ceiling in its maintenance instructions. Unit test:
  `python -m unittest tests/test_opt27b_watchdog.py` (6 passed).
- Yesterday's A100 allocation48419105 ran six local smoke cases and the
  SST-2/RTE multi-precision true-gradient probes. Probe statuses are complete
  at 2026-09-21 19:36 and 2026-09-22 02:47 EDT, respectively. No full
  training ran on that A100. The Slurm allocation later ended, but the probe
  result statuses remain complete.
- Updated only the pending, experiment-owned CPU aggregate job48421054 to
  `Dependency=afterany:48421053_*,afterany:48464008_*` using
  `scontrol update JobId=48421054 Dependency=afterany:48421053:48464008`.
  Verified with `scontrol show job 48421054 -o`. This prevents a premature
  final aggregate when only the original worker array finishes.
- Post-submit watchdog snapshot recognized both arrays: 8/50 configs
  complete, 2 running, 40 pending, 0 failed, and no health alerts. The two
  additional L4 tasks remained PENDING (Priority), not using GPUs yet.

## 2026-09-21 local preflight

- Local allocation: 48419105, node b7-3, A100-SXM4-80GB. Original allocation
  limit is 2026-09-22 15:21:12 local time. Do not cancel unrelated allocations.
- Environment: mezo-mistral, torch2.5.1, transformers4.48.0, datasets4.3.0.
- SST-2 official training pool: 67,349 examples; RTE: 2,490 examples.
- Reused official MeZO SST2/RTE templates and option-likelihood loss. FP32
  token score accumulation matches the existing stable probe path.
- Reused lr3e-7 and effective BS16 from the prior OPT MeZO option experiments.
  Learning rate, full data, preprocessing and budgets are identical across h.
- Added streaming directions and one master copy for L4 memory limits.
  Optional RTNClip diagnostic-statistics collection can be disabled without
  changing scales or quantized values; synthetic equality tests passed for
  INT4 and INT8, including non-full groups.
- Six local training smoke cases passed. RTE INT4 ran100 steps, median1.647s
  per training step on A100 and 13.12GiB peak allocated GPU memory. SST-2 FP16
  used about11.33GiB. Timing is not a measured L4 runtime estimate.
- Loss equivalence, replayed Gaussian directions, exact center restoration,
  deterministic resumed batch order, vector MSE identities and pooled nMSE
  passed. FP32 clean directional smoke at h1e-4 had nMSE0.0120 over two
  directions; this is a smoke check, not a formal64-direction result.
- An initial launch before the manifest finished failed without running the
  model. Preparation and launch were subsequently sequenced. It is not a
  failed scientific configuration.
- Main training uses L4 only to avoid architecture-dependent CUDA RNG streams
  across paired h runs. The local A100 executes the expensive gradient probes.
- Formal predictions are frozen using calibration batch0 before audit MSE.
  Primary theory comparisons use batch0; three other batches are robustness
  checks with separate summaries.

Live job IDs are recorded in l4_submission.json and logs/controller.jsonl.
The scheduler rejected a48-element array with QOSMaxSubmitJobPerUserLimit.
Submission was changed to two persistent L4 queue workers for the same48
predeclared runs. This obeys the submission/concurrency limits. At walltime,
workers checkpoint and requeue their own exact job IDs to continue the queue.
The first two L4 workers (48420613_0/1) revealed that flock is node-local on
this shared filesystem: both claimed the first run. Both workers and their
dependent aggregation job48420614 were cancelled. Their mixed partial logs
and checkpoint were moved to smoke/invalid_duplicate_l4_launch_sst2_h1e5 and
are excluded from scientific results. The queue now uses atomic mkdir claims,
owner metadata, and explicit stale-owner recovery. Manual cancellation no
longer triggers automatic requeue; only the walltime warning does. The local
A100 probe was independent and continued unaffected.
Every run records config, environment, invocations, attempt stdout/stderr,
training/evaluation curves and an atomic checkpoint. Local monitor snapshots
are written every five minutes while the current allocation remains alive.

## Reporting recovery, 2026-09-21 16:24 EDT

- Array48420798 exited after five minutes because aggregation imported missing
  matplotlib in the shared mezo-mistral environment. No CUDA/OOM/model failure
  caused these exits. Dependent CPU job48420799 failed for the same reason.
- Installed matplotlib3.9.4 plus contourpy1.3.2, cycler0.12.1,
  fonttools4.65.0, kiwisolver1.5.1, pyparsing3.3.3; existing numpy2.0.1,
  torch2.5.1 and transformers4.48.0 were not changed.
- Worker reporting errors now log and continue without terminating model
  children. L4 launch explicitly checks plotting dependencies. Tests cover
  reporting failure isolation, checkpoint rollback, and queue exclusion.
- On restart, uncheckpointed train/eval records are archived under each run's
  interrupted_records directory before restarting from pretrained weights.
  They are not merged into the resumed formal trajectory.
- The formal A100 probe PID1375806 remained healthy and was adopted by the
  recovered controller PID1378685 without restarting or duplicating it.
- Replacement L4 array48421053 has two workers; CPU aggregation48421054 is
  dependency-gated. Source snapshots with reporting_recovery suffix preserve
  this scheduler/reporting amendment. All scientific configs remain unchanged.
- Offline aggregation now generates CSV, PDF and PNG successfully. Early plots
  are partial curves, not completed results or evidence of an accuracy plateau.

## Launch validation, 2026-09-21 16:30 EDT

- Replacement workers both survived the first five-minute report and remained
  RUNNING beyond six minutes. Current nodes are ra8-5 and ra8-7. A100 probe and
  recovered local controller are also live. No new reporting exception occurred.
- Active training rows have unique, contiguous step indices and exactly paired
  sample IDs and direction seeds across h=1e-5 and h=1e-3. The default run had
  passed step222 at the check; no full-run accuracy is available yet.
- The first763 successful formal probe records passed vector decomposition,
  with maximum relative residual4.18e-16. Frozen theory predates audit records.
- Unit test command: mezo-mistral/bin/python tests/test_opt27b_queue_claim.py;
  three tests passed. Python compile checks and shell syntax checks passed.
- Partial PDF/PNG figures were generated and visually inspected. Missing task
  measurements are labeled pending; partial curves are not final paper results.

## Bounded maintenance audit, 2026-09-21 16:39–16:44 EDT

- Outcome: healthy ongoing work; no recovery, cancellation, requeue, model
  launch, scientific/source change or refactor performed by this audit.
  Training: 0/48 complete, 2 running, 46 pending. At 16:42:11 the active
  SST-2 FP16 seed16 h=1e-5 / h=1e-3 runs reached steps701/703 of20000,
  versus582/585 in the supplied16:39:19 snapshot. RTE remains pending.
- Ownership: claim48421053_0 on ra8-5 has worker PID1946273 / recorded model
  PID1946277; claim48421053_1 on ra8-7 has worker PID2995448 / recorded model
  PID2995452. scontrol confirms raw JobId48421061 maps to array task0,
  and raw JobId48421053 maps to task1; this is not an ownership mismatch.
  Both allocations are RUNNING with exactly one L4 each, throttle2, Restarts0.
  Direct remote ps/nvidia-smi attempts using ssh -o BatchMode=yes
  -o ConnectTimeout=10 ra8-5 and ra8-7 failed host-key verification (exit255);
  no bypass attempted. Remote PID liveness is therefore inferred from Slurm,
  ownership metadata and advancing logs, not independently observed via ps.
- Local probe PID1375806 is live on b7-3, sole local nvidia-smi compute process
  (20070MiB); controller PID1378685 remains live, allocation48419105 RUNNING.
  SST-2 probe has2122 unique successful records, batch0 now in INT8 after
  FP32/FP16; RTE probe pending. No completion claim. Original A100 allocation
  deadline remains2026-09-22 15:21:12 EDT; no additional A100 requested.
- CPU validations: all701 shared training steps have identical sample_indices
  and direction_seed, both trajectories contiguous, dataset/scope metadata
  identical. All observed train/eval/calibration/probe numeric values finite;
  all probe directions/d_star/norm_u2 match retained references; no duplicate
  probe keys; maximum vector decomposition residual5.25e-16.
- Both atomic checkpoint.pt files are5303361770 bytes with readable ZIP
  directories and CPU mmap metadata: step500,516 FP16 tensors,2651596800
  parameters, matching config hashes. Checks did not scan full tensor payloads
  or CRCs. Shared disk5.1PiB available,47% used; inode73% used. No stale
  checkpoint indication at the configured500-step interval.
- Recent model stdout advances; stderr contains only offline-cache notices.
  Current reporting stderr contains empty-legend warnings for pending data,
  no new traceback or missing dependency failure. Controller snapshots continue.
  No code fix warranted, so no resume or regression-test suite required.
- Existing watchdog PID1384398 is live with interval1800. During this audit,
  an external operation replaced CPU handoff48421464 (CANCELLED by5943) with
  48421498, PENDING BeginTime2026-09-21 17:10:19. This audit did not perform
  that replacement and did not restart the cancelled job; evidence retained.
- Commands/checks: python logs/watchdog/20260921_163919/audit_checks.py
  (path relative to experiment root; exit0); exact squeue/sacct/scontrol
  queries, ps -p1375806,1378685, nvidia-smi compute query, df -h/-i are
  retained with argv, exit codes and output in audit_evidence.json.
  /home/jy03364/miniconda3/envs/mezo-mistral/bin/python CPU-only heredoc used
  torch.load(path,map_location='cpu',weights_only=True,mmap=True), inspected
  step/config_hash/runtime/tensor shapes and compared config_hash with
  sha256(json.dumps(config,sort_keys=True).encode()); exit0, both matched.
  Follow-up squeue48421498, sacct48421464, ps -p1384398 all exited0.
- Changed files by this audit only: this EXPERIMENT_LOG.md append, and new
  logs/watchdog/20260921_163919/{audit_checks.py,audit_evidence.json,
  checkpoint_metadata.json,followup_evidence.json}. Rationale: preserve bounded
  audit checks, scheduler/progress/log evidence and validation limits.
  Protected inputs/reference/prediction hashes and current/recovery snapshot
  source hashes are retained in audit_evidence.json. Existing A100 process
  predates recovery source; it was not restarted and no claim is made that
  its in-memory code equals today's source. All raw measurements and earlier
  adverse/interrupted evidence retained; unrelated dirty files untouched.
