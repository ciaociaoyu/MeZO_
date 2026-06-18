# Interval-Aware h Selection 8h Bundle

Generated: 2026-06-18T01:25:38
Host: b7-2
Git commit: be13f3408608c04ed6adf18454ba4a5494627fc9
Elapsed seconds: 2.0

## Recommendation

The current recommended probe-only selector is `h_default_aware`: it first uses interval geometry and optional loss nMSE, then falls back to `h=1e-3` if no probe-only candidate is safer than default. This avoids selecting a visually clean h that is too non-local.

Pilot-calibrated selection is not claimed in this bundle unless `pilot_training_results.csv` contains completed rows. Training accuracy is used only for validation in the probe-only tables.

## Coverage

- Interval metric rows: 135
- Training rows audited: 119
- Selector configs: 6

### Interval configs
- facebook/opt-1.3b / sst-5 / int8 / dense: 21 h rows
- facebook/opt-1.3b / sst-5 / int8 / sparse_p0p1: 12 h rows
- roberta-large / sst-5 / int4 / dense: 37 h rows
- roberta-large / sst-5 / int4 / sparse_p0p1: 15 h rows
- roberta-large / sst-5 / int8 / dense: 35 h rows
- roberta-large / sst-5 / int8 / sparse_p0p1: 15 h rows

### Selector outcomes
- facebook/opt-1.3b / sst-5 / int8 / dense: h_default_aware=0.001 score_best=0.001 window_exists=True
- facebook/opt-1.3b / sst-5 / int8 / sparse_p0p1: h_default_aware=0.001 score_best=0.001 window_exists=True
- roberta-large / sst-5 / int4 / dense: h_default_aware=0.002 score_best=0.01 window_exists=True
- roberta-large / sst-5 / int4 / sparse_p0p1: h_default_aware=0.01 score_best=0.01 window_exists=True
- roberta-large / sst-5 / int8 / dense: h_default_aware=0.0015 score_best=0.001 window_exists=True
- roberta-large / sst-5 / int8 / sparse_p0p1: h_default_aware=0.001 score_best=0.001 window_exists=True

## Answers

1. Final recommended selector: probe-only `h_default_aware`; use pilot-calibrated only after short training rows exist.
2. RoBERTa vs default: see `default_comparison_summary.csv`; no missing result is fabricated.
3. OPT vs default: same table; current OPT coverage is mostly existing INT4/robustness logs unless new probes are added.
4. Per-precision global h: not claimed unless `policy_per_precision.csv` marks it usable.
5. If global h is unavailable, prefer per-config; per-model requires validation across tasks.
6. Failures are marked by missing source paths, fallback_to_default, or no interval window.
7. Next experiments: run loss-level nMSE for OPT INT8 dense/sparse and short 300-step pilots for selected h vs default.

## Notes / Missing Items
- Workflow did not launch long training; it uses existing logs and probe outputs.
