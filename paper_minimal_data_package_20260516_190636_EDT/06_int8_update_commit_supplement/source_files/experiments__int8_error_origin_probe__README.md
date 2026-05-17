# INT8 Error Origin Probe

Purpose: diagnose whether RoBERTa-large + SST-5 + INT8 finite-difference error is mainly
dead-zone collapse, wrong `h`, or extra implementation/noise.

The probe uses the real medium-model ZO perturbation path. For INT8/QuZO this means the
finite difference is evaluated with:

```text
f(Q(w + h z)) and f(Q(w - h z))
```

not the older float-only H-Probe path. The main comparison is against the raw sampled
direction `z` in `d_true = grad^T z`. Quantized `u1/u2` diagnostics are also written for
debugging because the current QuZO update path still updates along `u2`.

Run locally:

```bash
bash experiments/int8_error_origin_probe/run_roberta_sst5_int8_local.sh
```

Quick smoke test:

```bash
NUM_SEEDS=1 H_LIST="1e-3" bash experiments/int8_error_origin_probe/run_roberta_sst5_int8_local.sh
```

Useful overrides:

```bash
NUM_SEEDS=64 \
H_LIST="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,6e-3,1e-2,1.8e-2,3e-2" \
bash experiments/int8_error_origin_probe/run_roberta_sst5_int8_local.sh
```

Outputs:

- Raw run output: `experiments/int8_error_origin_probe/results/roberta_sst5_int8_error_probe/seed16/`
- Probe CSV: `zo_directional_probe.csv`
- Summary table: `experiments/int8_error_origin_probe/analysis/summary.csv`
- Markdown summary: `experiments/int8_error_origin_probe/analysis/summary.md`

Main table columns:

- `h`: finite-difference radius.
- `mse`: mean `(d_fd - d_true)^2`.
- `g2`: empirical `E[d_true^2]`.
- `nmse`: `mse / g2`.
- `corr`: correlation between `d_fd` and `d_true`.
- `sign_acc`: sign agreement.
- `fd_zero_ratio`: fraction of near-zero finite-difference estimates.
- `changed_ratio`: fraction of trainable coordinates that actually changed under `Q(w + h z)`.

Interpretation:

- `nmse << 1`, high `corr`: finite-difference signal is usable.
- `nmse ~= 1`, low `corr`, sign near 0.5, high `fd_zero_ratio` or low `changed_ratio`: likely INT8 dead-zone/signal collapse.
- `nmse >> 1` while `fd_zero_ratio` is not high: likely extra noise or implementation mismatch, such as dynamic scale jitter, RNG/data mismatch, or scalar loss precision amplification.

Probe controls:

- `ZO_PROBE_HLIST`: comma/space-separated list of h values to test in one run.
- `ZO_PROBE_INCLUDE_STEP0=1`: runs the probe before the first training update.
- `ZO_PROBE_ZERO_EPS`: threshold for `fd_zero_ratio`, default `1e-12`.
- `ZO_PROBE_UPDATE_STATS=1`: additionally compare intended update
  `-lr*d_fd*u2` with the actual post-snap update `Q(w-lr*d_fd*u2)-w`.

## Residual Grid Update Backend

`--zo_update_backend residual_grid` replaces the direct low-bit commit

```text
w <- Q(w + delta)
```

with an error-feedback lattice update:

```text
r <- r + delta
k <- commit(r / scale)
q <- clamp(q + k, qmin, qmax)
w <- q * scale
r <- r - k_actual * scale
```

Here `delta = -lr * projected_grad * z`, where `projected_grad` is the MeZO
finite-difference scalar and `z` is the same update direction already reused by
the existing seed path. The committed model state remains the quantized weight;
the residual buffer only stores uncommitted sub-grid update error. This is a
diagnostic backend and currently keeps one full residual tensor per trainable
parameter, so it should not be described as a final low-memory optimizer.

Important flags:

- `--residual_dtype {fp16,bf16,fp32}`
- `--residual_commit_mode {round,floor,stochastic}`
- `--residual_max_code_step N`, where `0` means unlimited and `1` limits each
  coordinate to one integer code per optimizer step
- `--int8_freeze_scale true`, the default for residual semantics
- `--log_update_stats_every N`
- `--save_update_stats_jsonl update_stats.jsonl`

The JSONL update diagnostics include intended/actual update norms, residual
norm, active and saturation fractions, intended-vs-actual cosine, and norm
ratio. `--zo_update_backend fp16_master` exposes the existing FP16-master
diagnostic path for comparison; `direct_int8` keeps the existing QuZO behavior.

Sparse random direction flags for h sweeps:

- `--zo_direction_sparse_mode {none,exact_random,bernoulli}`
- `--zo_direction_sparse_rate P`
- `--zo_sparse_rescale {none,inv_sqrt_p}`
- `--zo_sparse_per_layer_exact true/false`
- `--zo_h H`, an alias for `--zero_order_eps`

The launch script sets `model.eval()` through the existing `zo_forward` path, uses the same
batch for both sides of each finite difference, and disables prediction to keep the local
diagnostic focused on the probe.

Cost note: each `(h, seed)` pair runs two forward passes plus one true-gradient probe, so use
the smoke setting first when checking environment issues.
