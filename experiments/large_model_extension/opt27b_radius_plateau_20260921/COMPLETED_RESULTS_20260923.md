# Completed result snapshot (2026-09-23)

This commit contains completed OPT-2.7B results only. The continuing INT8/INT4
and RTE training jobs are not final results and are not included here. All
training rows below use SST-2, dense perturbations, FP16 forward, full data,
20,000 steps, effective batch size 16, and seed 16. Accuracy is dev accuracy.

| h | Best dev accuracy | Final dev accuracy | Best step |
|---:|---:|---:|---:|
| 1e-5 | 0.922018 | 0.870413 | 9500 |
| 1e-4 | 0.940367 | 0.926606 | 11000 |
| 1e-3 | 0.946101 | 0.936927 | 16500 |
| 3e-3 | 0.886468 | 0.886468 | 20000 |
| 1e-2 | 0.669725 | 0.655963 | 18000 |
| 3e-2 | 0.597477 | 0.597477 | 19000 |

The SST-2 and RTE probe jobs also completed. Each used four batches, 64 audit
directions per batch, and a 16-point radius grid for FP32, FP16, INT8, and
INT4. The target is the clean-gradient directional derivative. The pooled
true directional nMSE and vector-level rho values are in
`summaries/probe_mse_pooled_by_h.csv`; per-direction records are in the two
`probes/*_multprecision_probe/raw_probe_metrics.jsonl` files. The theoretical
windows in `summaries/theory_windows.csv` were frozen before the audit probe.

For SST-2, the measured INT4 nMSE minimum was 0.9816 at h=1.5e-3; for RTE,
it was 1.0429 at h=2e-3. These sampled minima are probe diagnostics, not
training accuracy optima or certified windows. INT4 has no tau=1 window under
the frozen plug-in theory for either task.

The six `runs/sst-2_fp16_*_seed16/` directories contain config, environment,
prompt examples, train/eval JSONL, completion status, summaries, and process
logs. Large checkpoint tensors are intentionally excluded from Git; the raw
records and exact configuration remain on disk. The two complete probe
directories contain batch IDs, frozen predictions, references, calibration
records, per-direction metrics, and logs. `METHOD_LOCK.md` and
`tools/opt27b_radius_extension.py` document the computational method.
