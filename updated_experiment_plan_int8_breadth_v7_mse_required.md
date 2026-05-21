# Updated Experiment Plan INT8 Breadth v7 MSE Required

## Classical h-only baselines for Main mechanism experiments

- The existing MeZO default `h=1e-3` baseline already exists and should not be rerun in this batch unless a preflight check proves it is missing and `RUN_DEFAULT_IF_MISSING=1` is explicitly set.
- New rows in this batch are `FD-eps13` and `Spall-SPSA c_k`.
- Scope: RoBERTa-large / SST-5 / dense two-point MeZO / FP32-FP16-INT8 / 20k steps.
- These rows are h-only baselines: keep the estimator, optimizer, dense direction distribution, learning rate, query count, data settings, and training loop fixed.
- Do not implement full SPSA here. `Spall-SPSA c_k` uses only the perturbation-gain schedule `h_t = h0 / (t + 1)^gamma`.
- Do not implement noise-aware finite-difference interval search in this batch.
- Use continuous h by default; do not snap to a grid unless explicitly requested.
- Enable existing early abort guards for pathological runs when the flags are available.

### FD-eps13

- FP32 raw h: `np.finfo(np.float32).eps ** (1/3) ~= 4.92e-3`.
- FP16 raw h: `np.finfo(np.float16).eps ** (1/3) ~= 9.92e-2`; cap to `1e-2`.
- INT8 has no principled machine-epsilon analogue. Run only as `fd_eps13_capped_stress` with `h=1e-2` and `fd_principled=false`.
- Safety interval: `[1e-5, 1e-2]`.

### Spall-SPSA c_k

- `h_t = h0 / (t + 1)^gamma`.
- `h0 = 1e-3`.
- `gamma = 0.101`.
- Continuous h with optional safety clipping to `[1e-5, 1e-2]`; no grid snapping by default.
