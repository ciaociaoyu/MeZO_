# Dense Probe-Window Notes

Source: `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv`

This folder supports the main precision-aware perturbation-window story. It is no-training probe evidence for dense ZO directions on RoBERTa-large / SST-5.

Key observations supported by the copied data:

- FP32 is stable at very small h. In the probe table, FP32 best-by-corr is `1e-05` with corr `0.9999997106604281` and nMSE `6.666072230573788e-07`.
- BF16 best probe h is around `1e-3`: best-by-corr is `0.001` with corr `0.9979581804430964`.
- INT8 best dense probe h is around `3e-3`: best-by-corr is `0.003` with corr `0.9375666196909768`.
- INT8 tiny h is not simply zero movement: at `h=1e-05`, active fraction is `0.9523344237813389`, but corr is `0.12096525191839812`, alignment is `0.07450058398802936`, and norm ratio is `13.424008104733753`.
- `h=1e-2` can have acceptable-looking geometry while derivative quality is poor. For INT8 at `h=0.01`, alignment is `0.9947742097885973` and norm ratio is `1.0052535267221125`, but corr is `0.05671528321337012`. Geometry-only window heuristics are therefore insufficient.

Do not overclaim from this file alone: it establishes probe/finite-difference quality, not final training accuracy.
