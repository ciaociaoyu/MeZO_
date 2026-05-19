# Precision-aware h-window estimator methods

This package is offline analysis only. It reads existing probe outputs and does
not launch training, submit jobs, or change quantizer semantics.

## Estimator 1: quantization-geometry lower bound

For low-bit G128 RTNClip probes, the lower-bound question is whether `h u` is
visible through the shared quantization grid built from the unperturbed FP16
master `w_t`. The analysis uses existing `active_frac` / `code_change_frac`,
effective displacement alignment, and norm-ratio diagnostics.

Default geometry thresholds:

- `tau_align = 0.7`
- `tau_rho_low = 0.7`
- `tau_rho_high = 1.5`
- `tau_code = 0.01`
- `tau_active = 0.01`

The deployable lower-bound estimate is `h_vis_min`, the smallest grid point
passing these geometry checks. This does not use true gradients.

## Estimator 2: effective-displacement diagnostics

The effective displacement is

`Delta_Q(h,u) = Q_t(w_t + h u) - Q_t(w_t - h u)`.

It is compared to `Delta_ideal = 2 h u` using alignment, norm ratio, code
change fraction, clip fraction, and saturation fraction. These diagnostics
are deployable for quantized oracles because they depend on the quantizer and
directions, not on true gradients.

## Estimator 3: Richardson/self-consistency locality

For each matched direction, the analysis compares

`d_h = [L(w + h u) - L(w - h u)] / (2h)`

with a smaller-scale estimate. Exact `h/2` pairs are preferred; when the
existing grid does not contain `h/2`, the script records an `h/3` or nearest
lower-scale smoothness pair in `richardson_pair_type`.

`richardson_relerr(h) = |d_h - d_ref| / max(|d_ref|, eps)`.

Default locality threshold:

- `tau_richardson = 0.3`

This estimator does not require true gradients. Rows without matched
per-direction finite differences are marked `locality_unavailable` and are not
treated as valid by the hybrid estimator.

## Estimator 4: true-direction calibration

When existing probe files contain `d_true = grad(w)^T u`, the analysis reports
`corr_fd_true` and `nMSE_fd_true`. These are retrospective calibration metrics
only and are not used to define the deployable valid window.

## Estimator 5: loss-SNR floor baseline

The requested repeated-base-loss baseline requires repeated evaluations of
`L(w_t)` on the same batch. No complete repeated-base-loss artifact was found
in the inspected files, so `loss_snr_visible` is left unavailable. The summary
lists the missing artifact and a probe-only command template.

## Estimator 6: hybrid precision-aware window

The main deployable rule is:

`valid(h) = geometry_visible(h) AND fd_local(h)`.

The script reports:

- `h_vis_min`: smallest h passing geometry visibility.
- `h_loc_max`: largest h passing Richardson/self-consistency.
- `valid_window`: grid interval satisfying both.
- `smallest_valid`: smallest valid h.
- `log_midpoint_valid`: grid h closest to the geometric midpoint of the valid interval.
- `score_min_valid`: valid h minimizing a weighted defect score.

For sparse directions, raw `h` and active-coordinate `h_active = h / sqrt(p)`
are both reported; selection and interval comparisons use `h_active`.

## Data-source notes

{
  "current_fp32_fp16": 22,
  "current_rtnclip_int4_probe": 11,
  "current_rtnclip_int8_geometry": 7,
  "historical_groupwise256_int8": 23,
  "legacy_dense_sparse": 46
}
