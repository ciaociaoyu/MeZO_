# Final Artifact Summary

1. Main figures: analytic window, precision window, SST-5 true directional MSE, INT4 visibility diagnostics, and SST-5 accuracy-vs-h.
2. Appendix figures: OPT cross-architecture sanity plus any paper snippets.
3. The old INT4 MSE figure was wrong because it plotted a geometry/visibility proxy as if it were true directional MSE.
4. Corrected INT4 true-MSE minimum: h = `0.002`, normalized MSE = `0.5421`.
5. Missing reliable true-MSE data: FP32 and FP16 for the final figure.
6. RoBERTa main tables use fixed-small, MeZO default h=1e-3, and precomputed analytical radius policies.
7. OPT supports only a transfer sanity claim: several tasks are non-degenerate/nearer default, but TREC is a substantial failure.
8. Single-seed data: RoBERTa multi-task and OPT comparison rows are seed 16 unless otherwise noted.
9. Do not claim concrete memory overhead where measured peak memory is missing.
10. The generated artifact package passes all automatic checks in `VALIDATION_REPORT.md`.
