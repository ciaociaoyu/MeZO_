# Paper Experiment Takeaways

## Supported Claims
1. Synthetic quantized oracles show the expected U-shaped h-dependent directional MSE, with a left visibility term and a right locality term.
2. Increasing quantization step Delta shifts the left boundary to larger h.
3. The random-direction floor grows with dimension, so convergence can be insensitive across a broader h range even when directional MSE changes.
4. Sparse/effective-subspace perturbations change active fraction and effective dimension, which changes the observed window.
5. Existing real-model results support SafeOverride wording: keep default h=1e-3 when it is inside the safe interval; use conservative override only for default-failure settings.

## Claims Not To Make
- Do not claim selected h always beats default.
- Do not claim interval-aware metrics alone always predict final accuracy.
- Do not present OPT stress-test tasks as exact MeZO OPT benchmark reproduction unless the task set matches the original OPT table.

## Main Paper vs Appendix
- Main paper: synthetic high-dimensional mechanism, real-model precision-window table, and SafeOverride policy.
- Appendix: OPT stress tests, medium/pilot targeted rows, and missing/failed configs.
