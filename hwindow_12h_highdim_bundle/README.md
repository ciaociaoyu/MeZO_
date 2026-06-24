# H-Window 12h High-Dimensional Bundle

This bundle supports the perturbation-window paper with high-dimensional
mechanism experiments and existing real-model evidence.

## Contents

- `audit_existing_results.md`: audit of existing RoBERTa/OPT probe and training artifacts.
- `synthetic_highdim_raw.csv`: synthetic quantized-oracle measurements across dimension, precision, Delta, active fraction, scale heterogeneity, and h.
- `synthetic_highdim_fit.csv`: fits of `A(h) ~= alpha / h^2 + beta h^2 + gamma`.
- `synthetic_highdim_window.csv`: kappa-window and rho-window endpoints.
- `synthetic_k_averaging_window.csv`: how k-direction averaging changes the rho window.
- `realmodel_interval_metrics.csv`: existing plus newly added interval-aware real-model probe rows.
- `realmodel_loss_mse.csv`: existing loss/fd nMSE rows when available.
- `targeted_training_results.csv`: existing full/medium/pilot training rows indexed for targeted validation.
- `table_highdim_scaling.csv`, `table_precision_window_realmodel.csv`, `table_training_summary.csv`: paper-oriented tables.
- `figures/`: paper and diagnostic figures in PDF/PNG.
- `paper_experiment_takeaways.md`: supported claims and claims to avoid.
- `metadata.json`: environment, command metadata, and provenance.

## New Work In This Run

1. Ran a synthetic high-dimensional quantized-oracle sweep on the local H100.
2. Added mid-p/mid-Delta synthetic add-ons and group-size 64/256 add-ons.
3. Ran a small real-model interval-aware probe for RoBERTa/OPT INT8 on SST-5/TREC with dense and sparse p=0.1 modes.
4. Aggregated existing training logs; no new long training was launched.

## Main Interpretation

The synthetic results are intended to support the high-dimensional mechanism:
quantization visibility controls the left side of the h window, locality controls
the right side, and the random-direction floor grows with dimension/effective
dimension. This explains why a wide h plateau can produce similar convergence
even when directional MSE differs.

For real models, default `h=1e-3` is inside the interval-aware safe region for
the added INT8 RoBERTa/OPT probes. Existing RoBERTa INT4 rows remain the primary
evidence for lower-precision narrow-window behavior.

## Limitations

- Targeted training is aggregated from existing logs only.
- Real-model per-layer scale outlier plotting uses clipping fraction as a proxy;
  per-layer scale histograms were not recomputed in this finalizer.
- OPT stress-test tasks should not be presented as exact original MeZO OPT
  benchmark reproduction unless the task set matches the original OPT table.
