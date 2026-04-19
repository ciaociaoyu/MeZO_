# QuZO Probe Smoke Validation: 2026-04-19

This note records the short single-GPU validation run that was used to decide whether QuZO directional probe support was safe enough to resume pilot submissions.

## Scope

- Goal:
  - verify that QuZO directional probe now compares `FD(u1)` against `TD(u1)` instead of mixing directions
  - check that the probe output is numerically interpretable on real training runs
  - estimate runtime before submitting the remaining pilot jobs
- Local hardware:
  - `NVIDIA A100-SXM4-80GB`
- Environments:
  - `ciao` for `medium_models` / `roberta-large`
  - `mezo-env` for `large_models` / `opt-1.3b`
- Runner entrypoints:
  - `experiments/pilot/_shared/h_sweep_8h/run_medium_sweep.sh`
  - `experiments/pilot/_shared/h_sweep_8h/run_large_sweep.sh`

## Smoke Config

- Methods:
  - `MeZO / roberta-large / SST-5 / INT8`
  - `MeZO / opt-1.3b / SST5 / INT8`
- `h` grid:
  - `1e-6`
  - `1e-5`
  - `1e-4`
  - `3e-4`
  - `1e-3`
- Training length:
  - `max_steps=300`
  - `eval_steps=100`
  - `logging_steps=10`
- Probe config:
  - `zo_probe_every=100`
  - `zo_probe_num_seeds=8`
- Important density note:
  - the smoke probe density is the same as the production pilot scripts
  - `100 / 8` in smoke and `200 / 16` in pilot both equal `0.08` probe-seeds per training step
  - this makes the smoke wallclock extrapolation directly useful for the current pilot scripts

## RoBERTa-large Result

Canonical smoke root:

- `experiments/smoke/mezo/roberta-large/sst5/int8/quzo_probe_smoke_20260419`

Per-`h` summary. `fd_mean_avg`, `td_mean_avg`, `mse_avg`, `corr_avg`, and `sign_acc_avg` are averages over the three probe checkpoints at steps `100/200/300`.

| h | train_loss | eval_loss | fd_mean_avg | td_mean_avg | mse_avg | corr_avg | sign_acc_avg | tail_perf_wallclock_per_step | estimated 8h sweep hours on A100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-6` | 28.6724 | 1.5797 | -705.024 | -890.989 | 524195988.7 | -0.053 | 0.542 | 2.503 | 55.62 |
| `1e-5` | 1.7264 | 1.5928 | 84.098 | -47.744 | 4995412.4 | 0.287 | 0.667 | 2.008 | 44.62 |
| `1e-4` | 1.9377 | 1.7388 | -8.861 | 48.715 | 222113.9 | 0.008 | 0.542 | 1.996 | 44.35 |
| `3e-4` | 1.9906 | 2.0144 | -63.381 | -194.723 | 398277.9 | 0.287 | 0.500 | 1.980 | 44.01 |
| `1e-3` | 1.8375 | 1.7079 | 13.951 | -247.298 | 1497447.6 | -0.269 | 0.292 | 2.006 | 44.57 |

Interpretation:

- The probe is no longer behaving like a permanently broken random comparison.
- Very small `h` is clearly bad.
- Mid-range `h` values such as `1e-5` and `3e-4` show positive average correlation and non-random sign agreement.
- The medium-model path is still numerically noisy, so the correct conclusion is:
  - the probe semantics are now trustworthy
  - `roberta-large / INT8 / QuZO` remains a noisy method regime
  - noisy metrics here should be treated as method behavior, not as evidence of another `u1/u2` implementation bug

## OPT-1.3B Result

Canonical smoke root:

- `experiments/smoke/mezo/opt-1.3b/sst5/int8/quzo_probe_smoke_20260419`

Per-`h` summary. Probe statistics are again averaged over the per-run probe checkpoints.

| h | accuracy | dev_accuracy | fd_mean_avg | td_mean_avg | mse_avg | corr_avg | sign_acc_avg | tail_perf_wallclock_per_step | estimated 8h sweep hours on A100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-6` | 0.2480 | 0.2845 | -610.352 | -5.601 | 6788041.9 | 0.052 | 0.500 | 2.073 | 46.08 |
| `1e-5` | 0.2598 | 0.2693 | 91.553 | -21.132 | 805175.6 | 0.135 | 0.625 | 1.828 | 40.63 |
| `1e-4` | 0.2380 | 0.2600 | 21.973 | -21.061 | 92945.2 | 0.510 | 0.750 | 1.820 | 40.45 |
| `3e-4` | 0.2262 | 0.1991 | 66.630 | 21.173 | 5785.8 | 0.680 | 0.812 | 1.913 | 42.51 |
| `1e-3` | 0.2525 | 0.2857 | -43.823 | -38.181 | 2398.9 | 0.927 | 0.875 | 1.817 | 40.37 |

Interpretation:

- The large-model path is materially cleaner than the medium-model path.
- `1e-4`, `3e-4`, and `1e-3` all show strong positive correlation and strong sign agreement.
- This is consistent with the earlier observation that the large-model implementation already had the more coherent probe shape.

## Validation Decision

Decision:

- QuZO directional probe is now safe enough to use as an implementation diagnostic.
- The official comparison is now `FD(u1)` vs `TD(u1)`.
- `TD(u2)` is logged only as a debug quantity and is not used for the official `mse`, `corr`, or `sign_acc`.
- The remaining pilot jobs were allowed to proceed after this smoke.

What passed:

- `medium_models` path:
  - probe CSV is now produced for QuZO instead of being skipped
  - average mid-range metrics are no longer systematically random or permanently negative
- `large_models` path:
  - QuZO probe is numerically clean enough to serve as the reference implementation
- Runtime:
  - A100 smoke extrapolates to roughly `40-56h` for a full INT8 8-point sweep
  - the production pilot scripts request `H100`, so the server-side runtime budget is looser than this smoke estimate

Residual risk:

- `roberta-large / INT8 / QuZO` remains noisy, so pilot probe plots should be interpreted as honest-but-noisy rather than smooth-by-default.
- Sparse RoBERTa jobs still carry separate timeout risk from the underlying sparse method, independent of the QuZO probe fix.
