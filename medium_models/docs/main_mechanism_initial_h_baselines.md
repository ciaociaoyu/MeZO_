# Main Mechanism Initial-h Baselines

This batch compares exactly three h-only initial-radius policies for RoBERTa-large / SST-5 Main mechanism runs:

- `mezo_default`: the existing MeZO default, `h = 1e-3`.
- `fixed_small`: a fixed small perturbation radius, `h = 1e-5`.
- `fd_eps13_raw`: the classical central finite-difference rule, `h = eps_machine^(1/3)`.

The default `h=1e-3` runs are treated as existing references. The add-on launcher only prepares `fixed_small` and `fd_eps13_raw` across FP32, FP16, and INT8 unless a separate explicit default rerun is requested after preflight.

## FD-eps13 Values

- FP32: `np.finfo(np.float32).eps ** (1/3) = 0.004921565763652325`.
- FP16: `np.finfo(np.float16).eps ** (1/3) = 0.0992431640625`.
- INT8: there is no machine-epsilon analogue, so the run is labeled unprincipled and uses the FP16 eps^(1/3) proxy as a stress baseline.

For `fd_eps13_raw`, the safety interval `[1e-5, 1e-2]` is metadata only when `h_schedule_fd_clip_policy=none`. FP16 and INT8 proxy values are intentionally out of window and are not capped.

## Scientific Scope

These are h-only initial-radius baselines. They keep the MeZO estimator, dense random direction, optimizer, learning rate, query count, data settings, data order, and training loop fixed. They do not implement Spall-SPSA, noisy finite-difference interval search, short-run h tuning, or grid snapping.

Short-run h tuning is not the same baseline: it spends training budget to select a radius. The three policies here define h without a tuning sweep.

## INT8 Rule

INT8 runs use G128 groupwise RTNClip fake quantization with an FP16 master update. The two perturbation branches must share the pair grid from the unperturbed FP16 master weights and fresh-round integer codes separately for `+h` and `-h`. GPTQ, residual-grid updates, and direct INT updates are out of scope for this batch.

## Resource Rule

Smoke and long runs should request L4 GPUs only. Long runs are submitted as a six-task Slurm array with exactly one L4 GPU per task and at most three active tasks:

```bash
GPU_GRES=${GPU_GRES:-gpu:L4:1}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-3}
```

If typed GRES is unavailable, use `GPU_GRES=gpu:1 SBATCH_EXTRA="--constraint=L4"`.

## Early-stop Behavior

Existing random-prediction and ZO-probe health guards are enabled when the corresponding flags exist. Early-stop guards may abort pathological runs, but h values are still logged to `metrics_logs/h_schedule.csv` so the selected baseline radius remains visible.
