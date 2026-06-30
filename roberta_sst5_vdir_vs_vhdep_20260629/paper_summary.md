# V_dir versus V_h_dep Probe Summary

This is a probe-only contribution experiment for the guardrail-window section. No training was launched.

Definitions used:

- `d_star = <g,u>`
- `d_h = [F(w+h u)-F(w-h u)]/(2h)` under the precision-specific forward oracle
- `V_h_dep(h) = E[(d_h-d_star)^2 ||u||^2]`
- `V_dir = E||(<g,u>)u-g||^2`
- `rho(h) = V_h_dep(h) / V_dir`

Scalar true directional nMSE is reported as a diagnostic only; it is not used as rho.

## Probe Setup

- output folder: `/scratch/jy03364/MeZO_/roberta_sst5_vdir_vs_vhdep_20260629`
- model/task: `roberta-large` / SST-5 full data
- seed/data_seed: `16` / `16`
- batch_size: `64`, num_batches: `1`
- directions: `64` with base seed `730000`
- precision modes: fp32, fp16, int8, int4
- low-bit quantizer: existing G128 RTNClip shared-grid fake quantized forward oracle

## Precision Conclusions

### fp32

- h=1e-3: `below random-direction floor`.
- h=1e-5: `below random-direction floor`.
- rho=1 crossing estimates: `6.09996e-09`.
- Minimum rho in grid: `5.12837e-07` at h=`1e-05`.
- Paper sentence: For FP32, the measured radius-dependent term is below the random-direction floor across a broad radius range, supporting a wide high-precision plateau.

### fp16

- h=1e-3: `below random-direction floor`.
- h=1e-5: `near random-direction floor`.
- rho=1 crossing estimates: `7.47798e-06`.
- Minimum rho in grid: `0.00159557` at h=`0.001`.
- Paper sentence: FP16 shows a small-h numerical dead zone, but the default h=1e-3 lies in the probe-reliable region when its rho and reliability diagnostics are favorable.

### int8

- h=1e-3: `below random-direction floor`.
- h=1e-5: `no reliable finite-difference signal`.
- rho=1 crossing estimates: `0.000227472 0.00975348`.
- Minimum rho in grid: `0.0221159` at h=`0.0015`.
- Paper sentence: INT8 remains a default-safe low-bit case when h=1e-3 keeps V_h_dep below or comparable to the random-direction floor.

### int4

- h=1e-3: `above random-direction floor`.
- h=1e-5: `no reliable finite-difference signal`.
- rho=1 crossing estimates: `0.00256551`.
- Minimum rho in grid: `0.388747` at h=`0.005`.
- Paper sentence: INT4 is a boundary case: dense default can be empirically usable, but the finite-difference contribution is not uniformly below the random-direction floor.

## Files

- `raw_direction_metrics.csv`: per precision / h / direction raw quantities.
- `contribution_by_h.csv`: aggregated `V_dir`, `V_h_dep`, `rho`, scalar nMSE, corr, sign, and zero-fraction.
- `representative_points_table.csv`: tiny/default/min-rho/largest-h rows.
- `fig_rho_vs_h_by_precision.pdf/png`
- `fig_vdir_vhdep_components.pdf/png`
- `fig_probe_reliability_vs_h.pdf/png`
