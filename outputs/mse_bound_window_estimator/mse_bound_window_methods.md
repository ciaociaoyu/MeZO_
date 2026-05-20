# MSE-bound h-window methods

This package supersedes the earlier Richardson-locality prototype for h-window
selection. It keeps the same precision-aware geometry diagnostics as context,
but the primary selector is now the project MSE-bound shape.

## Empirical MSE-envelope estimator

The fitted curve is:

`mse_hat(h) = alpha / h^2 + beta h^2 + gamma`.

The analysis uses `nMSE_fd_true` first, then `MSE_fd_true`, then a clearly
marked geometry/FD proxy only when true-gradient probe data is unavailable.
The coefficients are constrained nonnegative by enumerated active-set least
squares with Huber-style reweighting in log residual space. The reported
`h_star` is `(alpha / beta)^(1/4)` when both terms are positive.

Windows:

- `W_kappa = {h : mse_hat(h) <= kappa * min_h mse_hat(h)}` for kappa 1.5, 2, 3.
- `W_tau = {h : mse_hat(h) <= tau}` for normalized-MSE thresholds 0.01, 0.03, 0.05, 0.1, 0.2.

Selection policies:

- `h_star_nearest`
- `log_midpoint_W2`
- `smallest_in_W_tau_0.1`
- `score_min`, with no bias toward `1e-3`.

## Theory-proxy estimator

The bound form is:

`B(h) = Delta_eff^2 G^2 / (4 h^2) + 2 Delta_eff L G sqrt(K_u) + 4 h^2 L^2 K_u`.

For this offline package, the theory proxy instantiates constants from
available probes:

- `Delta_eff = 2 sqrt(alpha)` from the small-h fitted term.
- `G_hat` from `d_true_std` or `fd_std` near stable h values.
- `L_hat = sqrt(beta) G_hat / 2` in normalized directional-MSE units.
- `K_u = 1` as a normalized directional proxy.

This proxy is useful as a bound-structured explanation, not an independent
theorem-only estimator. The CSV explicitly marks that scalar calibration is
needed before treating it as standalone.

## Hybrid estimator

The hybrid compares empirical and theory h-stars. Confidence is high if they
agree within a factor of 3. Since the current theory proxy derives key
constants from the empirical envelope, agreement should be interpreted as a
consistency check rather than independent validation.

## Guardrails

No training jobs are launched by this analysis. Training accuracy is read only
for retrospective validation and is not used to fit coefficients or thresholds.
GPTQ, residual-grid, independent Q+/Q- grids, and direct INT updates are not
used.
