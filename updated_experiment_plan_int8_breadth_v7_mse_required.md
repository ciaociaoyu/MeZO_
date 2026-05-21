# Updated Experiment Plan INT8 Breadth v7 MSE Required

## 3.6 Main Mechanism Initial-h Baselines

The paper comparison uses exactly three zero-training initial-radius policies:

- `mezo_default`: the existing MeZO default `h_default = 1e-3`.
- `fixed_small`: a fixed small radius `h_small = 1e-5`.
- `fd_eps13_raw`: the classical central finite-difference rule `h_FD = eps_machine^(1/3)`.

The default `h=1e-3` runs already exist and should not be rerun in this batch unless a preflight check proves that an existing default run is missing and `RUN_DEFAULT_IF_MISSING=1` is explicitly set. The add-on batch contains only `fixed_small` and `fd_eps13_raw` across FP32, FP16, and INT8.

Scope: RoBERTa-large / SST-5 / full data / dense two-point MeZO / seed 16 / data seed 16 / 20k steps / eval every 1k steps. These are h-only baselines: estimator, dense direction distribution, optimizer, learning rate, query count, data settings, data order, and training loop stay fixed.

This batch does not implement Spall-SPSA, noisy FD interval search, h grid snapping, GPTQ, residual-grid updates, direct INT updates, LoRA, or additional model families.

## 5.1 L4 Resource Rule

Run the two add-on policies as a six-task Slurm array:

- Policies: `fixed_small`, `fd_eps13_raw`.
- Precisions: `fp32`, `fp16`, `int8`.
- GPU request: exactly one L4 GPU per task.
- Concurrency: at most three active GPU tasks, using array concurrency `%3`.

The launcher defaults to `GPU_GRES=${GPU_GRES:-gpu:L4:1}` and `ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-3}`. If the cluster requires constraints instead of typed GRES, use `GPU_GRES=gpu:1 SBATCH_EXTRA="--constraint=L4"`.

## Appendix A: Three-baseline h Policy Set

### MeZO Default

- Baseline role: existing default reference.
- Rule: `h_default = zero_order_eps = 1e-3`.
- Action in this batch: preflight only; do not rerun unless explicitly requested by `RUN_DEFAULT_IF_MISSING=1`.

### Fixed Small Radius

- Baseline role: test whether simply choosing a very small MeZO-supported radius is enough.
- Rule: `h_small = 1e-5` for every step and every precision.
- No grid snapping and no adaptive-h path.

### Raw FD-eps13

- Baseline role: classical zero-training finite-difference initial-h formula.
- FP32: `np.finfo(np.float32).eps ** (1/3) ~= 0.004921565763652325`.
- FP16: `np.finfo(np.float16).eps ** (1/3) ~= 0.0992431640625`.
- INT8: no principled machine-epsilon analogue; use the FP16 eps^(1/3) proxy as a clearly labeled stress baseline with `fd_principled=false`.
- Safety window `[1e-5, 1e-2]` is metadata for `fd_clip_policy=none`; it must not cap raw FD. FP16 and INT8 proxy are intentionally out of window.

### Shared Scientific Constraints

- Keep `USE_H=False` and `USE_C=False`.
- Keep continuous h; do not snap to a grid.
- Keep the two-point symmetric MeZO/SPSA finite-difference estimator and query count unchanged.
- For INT8, use G128 groupwise RTNClip fake quantization with an FP16 master update, pair-shared grid from the unperturbed FP16 master weights, fresh-rounded `+h` and `-h` integer codes, K=1 refresh, no GPTQ, no direct INT update, and no residual-grid.
- Enable existing early-stop guards for pathological runs when the flags are present. Early stops must record the chosen h rather than hiding it.
