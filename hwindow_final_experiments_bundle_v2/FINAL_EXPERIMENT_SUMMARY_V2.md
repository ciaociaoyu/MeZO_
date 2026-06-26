# Final Experiment Summary V2

## What changed from v1

- Analytical Panel C and `table_analytic_window.csv` now use empirical rho-window centers and empirical MSE optima, not theoretical `h_ref` regressed against its own inputs.
- Precision-window tables separate frozen theoretical windows from empirical accuracy intervals (`best_dev_acc >= best - 0.01`).
- INT4 is explicitly marked as `no tau=1 certificate`; FP32/FP16 are empirical-only; BF16 is recorded as missing from the main table.
- RoBERTa INT4 main table contains only fixed-small, MeZO default, and frozen-reference rows, all full runs.
- OPT is reduced to a cross-architecture sanity table and retains TREC as a failure.
- `radius_provenance.csv` separates `frozen_h_ref`, `legacy_hstar`, and actual `training_h`.

## Supported claims

- The analytical envelope conservatively covers measured MSE in the controlled one-sided surrogate.
- Predicted centers/endpoints and empirical scaling trends are broadly aligned with the frozen theory.
- Default `h=1e-3` is competitive in broad-window settings.
- Analytical reference radii can help in some narrow/extreme low-precision settings, especially prefix INT4, but not universally.

## Unsupported claims

- The reference radius does not universally beat default.
- Empirical accuracy/MSE sweeps do not define the theoretical window.
- OPT results are not a direct original-MeZO reproduction or SOTA comparison.
