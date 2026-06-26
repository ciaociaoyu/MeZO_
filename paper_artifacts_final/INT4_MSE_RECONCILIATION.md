# INT4 MSE Reconciliation

## Resolution

The old RoBERTa/SST-5/INT4 curve whose minimum appeared near `1e-2` was not the paper directional MSE. It was a geometry/visibility proxy such as `delta_visibility_nmse_mean`, `A_uniform`, or `lowbit_true_nmse`.

The canonical true directional nMSE source is:

`outputs/rtnclip_int4_mse_reprobe_20260521_true_nmse_d16/int4_mse_probe_summary.csv`

The audited script is `tools/probe_int4_dense_fd_nmse.py`. It computes a true gradient by backward, evaluates `d_star=grad^T u`, computes quantized two-point `d_Q`, and pools `(d_Q-d_star)^2 / d_star^2`.

## Required answers

1. Original INT4 curve metric: geometry/visibility proxy, not loss-level `A_true`.
2. Is it A_true? No.
3. Table `fd_true_nmse`: yes, audited as `A_true` / normalized true directional MSE.
4. Why minima differ: geometry improves monotonically as h crosses more quantization intervals, so proxy can look best at `1e-2`; loss directional MSE also includes locality and finite-difference loss behavior, with canonical minimum at `0.002`.
5. Paper true-MSE figure uses the `fd_true_nmse` source above.
6. Geometry/proxy curves are relabeled as visibility diagnostics or removed from MSE figures.

Canonical INT4 true-nMSE minimum: h = `0.002`, nMSE = `0.5421`.

Proxy minima observed for reconciliation: `{'A_uniform': (0.01, 0.0033233741448169), 'delta_visibility_nmse_mean': (0.01, 0.034494639315009), 'lowbit_true_nmse': (0.01, 0.100231443991697)}`.
