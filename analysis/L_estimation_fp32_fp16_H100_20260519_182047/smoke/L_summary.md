# L Estimation Summary

Analysis directory: `analysis/L_estimation_fp32_fp16_H100_20260519_182047/smoke`

## Table A: Selectors

| precision | checkpoint | L_mode | selector | selected h2 | L_q50 | L_q90 | L_q95 | SNR2 | stability | flags |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| fp16 | step_1000 | L_clean32 | old_snr_smallest_pass | 0.003 | 1.86062e-05 | 2.13663e-05 | 2.15363e-05 | 2.29661 | 0.48367 |  |
| fp16 | step_1000 | L_clean32 | old_snr_max_fallback_ablation | 0.003 | 1.86062e-05 | 2.13663e-05 | 2.15363e-05 | 2.29661 | 0.48367 | ablation_only |
| fp16 | step_1000 | L_clean32 | plateau_q90_primary | 0.003 | 1.86062e-05 | 2.13663e-05 | 2.15363e-05 | 2.29661 | 0.48367 |  |
| fp16 | step_1000 | L_clean32 | plateau_q95_conservative | 0.003 | 1.86062e-05 | 2.13663e-05 | 2.15363e-05 | 2.29661 | 0.48367 |  |
| fp16 | step_1000 | L_oracle_precision | old_snr_smallest_pass | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 |  |
| fp16 | step_1000 | L_oracle_precision | old_snr_max_fallback_ablation | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 | ablation_only |
| fp16 | step_1000 | L_oracle_precision | plateau_q90_primary | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 |  |
| fp16 | step_1000 | L_oracle_precision | plateau_q95_conservative | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 |  |
| fp16 | step_1000 | L_oracle_oldSNR | old_snr_smallest_pass | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 |  |
| fp16 | step_1000 | L_oracle_oldSNR | old_snr_max_fallback_ablation | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 | ablation_only |
| fp16 | step_1000 | L_oracle_oldSNR | plateau_q90_primary | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 |  |
| fp16 | step_1000 | L_oracle_oldSNR | plateau_q95_conservative | 0.003 | 1.85943e-05 | 2.1427e-05 | 2.16121e-05 | 2.29002 | 0.461358 |  |

## Table B: Primary Comparison

| precision | checkpoint | L_clean32 plateau q90 | L_oracle plateau q90 | L_oracle oldSNR q90 | interpretation |
|---|---|---:|---:|---:|---|
| fp16 | step_1000 | 2.13663e-05 | 2.1427e-05 | 2.1427e-05 | see selector flags |

## Interpretation

- The theoretical h-star formula should consume `L_clean32` with `plateau_q90_primary` when that selector is available.
- `L_oracle_precision` is an oracle-consistent diagnostic; it should not replace clean FP32 curvature unless the downstream analysis explicitly wants oracle noise folded into L.
- `L_oracle_oldSNR` is an ablation to expose the previous max-SNR fallback behavior.
- Warnings were emitted; see `L_diagnostics.json` for details.
