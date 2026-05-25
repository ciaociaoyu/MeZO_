# L Estimation Summary

Analysis directory: `/scratch/jy03364/MeZO_/analysis/L_estimation_fp32_fp16_20260519_172111`

## Table A: Selectors

| precision | checkpoint | L_mode | selector | selected h2 | L_q50 | L_q90 | L_q95 | SNR2 | stability | flags |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|

## Table B: Primary Comparison

| precision | checkpoint | L_clean32 plateau q90 | L_oracle plateau q90 | L_oracle oldSNR q90 | interpretation |
|---|---|---:|---:|---:|---|

## Interpretation

- The theoretical h-star formula should consume `L_clean32` with `plateau_q90_primary` when that selector is available.
- `L_oracle_precision` is an oracle-consistent diagnostic; it should not replace clean FP32 curvature unless the downstream analysis explicitly wants oracle noise folded into L.
- `L_oracle_oldSNR` is an ablation to expose the previous max-SNR fallback behavior.
- Warnings were emitted; see `L_diagnostics.json` for details.
