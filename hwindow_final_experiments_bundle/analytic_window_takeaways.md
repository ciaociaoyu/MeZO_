# Analytical Window Takeaways

This is the single controlled analytical experiment for the frozen paper theory.
It uses the one-sided probe on `f(x)=g^T x + L/2 ||x||^2` at `x=0` with perturbation-space mid-tread quantization.

- Mean analytical-envelope coverage over the simulated grid: 1.000.
- Median center error `abs(log10(h_ref / h_emp_center))`: 0.040.
- Mean slope estimates: log h_ref vs log Delta: 0.500, log h_ref vs log G: 0.500, log h_ref vs log L: -0.500, log h_ref vs log d: -0.500.
- Increasing `Delta` increases `rho_min`; large `Delta` configurations lose the certified `tau=1` window first.
- Finite Monte Carlo error is reported through empirical endpoint mismatch; formulas were not refit or changed.
