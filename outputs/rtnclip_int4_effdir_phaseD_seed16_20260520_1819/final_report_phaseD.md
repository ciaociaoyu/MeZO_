# Phase D INT4 Effective-Displacement Direction Update Report

Output root: `outputs/rtnclip_int4_effdir_phaseD_seed16_20260520_1819`

## Scope

This is RoBERTa-large / SST-5, seed=16, data_seed=16, bs=64, shuffled RandomSampler, INT4 G128 RTNClip fake-quantized forward, K=1, shared-grid fresh-rounding, FP16 master update. No GPTQ, direct INT update, residual-grid, sparse, or LoRA was run.

Implementation note: this isolated Phase D runner samples dense directions and applies updates only in the INT4-quantized `Linear.weight` tensor space. This keeps `Delta_Q`, active set, and update direction definitions aligned with the INT4 displacement diagnostics.

## Completed Runs

- 16 / 16 short runs completed at 300 steps.
- 4 / 4 selected extension runs completed at 1000 steps.
- No NaN, no non-finite `d_h`, no illegal quantized codes.
- `skip_update_frac = 0` for all runs; active fractions never fell below 1e-3.

## 300-Step Result Highlights

| run | best_acc | last_acc | active_frac | alignment | norm_ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard h=1e-3 | 0.3302 | 0.3302 | 0.1180 | 0.4080 | 2.183 |
| standard h=5e-4 | 0.2799 | 0.2740 | 0.0571 | 0.2824 | 3.043 |
| effdir_global h=5e-4 | 0.3091 | 0.3091 | 0.0129 | 0.1826 | 1.191 |
| effdir_global h=2e-3 | 0.3126 | 0.3126 | 0.2298 | 0.5893 | 1.521 |
| effdir_active h=5e-4 | 0.3021 | 0.3021 | 0.0318 | 0.2511 | 2.044 |
| effdir_secant h=1e-3 | 0.3056 | 0.3056 | 0.0998 | 0.4031 | 1.929 |

## 1000-Step Extensions

| run | best_acc | last_acc | last_loss | active_frac | alignment | norm_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard h=5e-4 | 0.3138 | 0.3138 | 1.5614 | 0.0571 | 0.2824 | 3.042 |
| standard h=1e-3 | 0.2635 | 0.2108 | 1.5944 | 0.1180 | 0.4080 | 2.184 |
| effdir_global h=5e-4 | 0.2951 | 0.2787 | 1.5574 | 0.0110 | 0.1727 | 1.092 |
| effdir_global h=1e-3 | 0.2939 | 0.2892 | 1.5634 | 0.0741 | 0.3751 | 1.582 |

## Answers

1. Which h works best for standard dense INT4?
   - At 300 steps, standard h=1e-3 is best.
   - At 1000 steps, standard h=5e-4 is more stable and better than standard h=1e-3.

2. Does effdir improve small-h INT4?
   - Short-run at h=5e-4: yes. `effdir_global` improves best_acc from 0.2799 to 0.3091 at 300 steps.
   - At 1000 steps: no. `effdir_global h=5e-4` falls behind `standard h=5e-4` (0.2951 vs 0.3138 best_acc).

3. Which scaling is most stable?
   - `effdir_global` gives cleaner norm ratios near 1, but its active fraction shrinks over training at h=5e-4.
   - `effdir_active` is conservative and did not improve accuracy.
   - `effdir_secant` required clipping in about 2-3% of short-run steps and did not beat global or standard.
   - Accuracy-wise, the most stable candidate from these runs is still `standard h=5e-4` for the 1k extension.

4. Should any config be extended to 2k/5k?
   - Do not promote effdir to 5k based on this batch.
   - If one follow-up is useful, run `standard h=5e-4` to 2k as the INT4 small-h control.
   - For effdir, try a narrower follow-up only if testing geometry rather than accuracy: `effdir_global h=1e-3` or lower lr, because it had cleaner geometry but did not win accuracy.

Richardson self-consistency was logged, but the last-step relative errors were noisy across runs; it should be treated as diagnostic context, not as the selection criterion here.
