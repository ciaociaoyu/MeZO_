# Analytical Window Takeaways

This experiment uses the frozen one-sided quadratic surrogate `f(x)=g^T x + L/2 ||x||^2` with perturbation-space mid-tread quantization.
The analytical curve is an upper envelope, not a fitted curve; no theoretical parameter is refit from measured MSE.

- Mean envelope coverage over measured grid points: 1.000.
- Median center error using empirical rho-window centers: 0.040 log10 units.
- Panel C reports empirical slopes from measured rho-window centers and empirical MSE optima, with theory slopes shown only as reference markers.
- Increasing `Delta` raises `rho_min` and narrows or removes the certified `tau=1` window.
