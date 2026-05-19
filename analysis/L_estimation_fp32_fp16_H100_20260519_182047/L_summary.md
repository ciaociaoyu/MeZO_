# L Estimation Summary

Analysis directory: `analysis/L_estimation_fp32_fp16_H100_20260519_182047`

## Table A: Selectors

| precision | checkpoint | L_mode | selector | selected h2 | L_q50 | L_q90 | L_q95 | SNR2 | stability | flags |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| fp16 | step_1000 | L_clean32 | old_snr_smallest_pass |  |  |  |  |  |  | no_snr_pass |
| fp16 | step_1000 | L_clean32 | old_snr_max_fallback_ablation | 0.005 | 1.19944e-05 | 2.81514e-05 | 2.9914e-05 | 0.976566 | 0.570242 | ablation_only;fallback_max_snr |
| fp16 | step_1000 | L_clean32 | plateau_q90_primary | 0.0015 | 6.34772e-06 | 1.46317e-05 | 1.74658e-05 | 0.592921 | 0.405575 |  |
| fp16 | step_1000 | L_clean32 | plateau_q95_conservative | 0.0015 | 6.34772e-06 | 1.46317e-05 | 1.74658e-05 | 0.592921 | 0.405575 |  |
| fp16 | step_1000 | L_oracle_precision | old_snr_smallest_pass |  |  |  |  |  |  | no_snr_pass |
| fp16 | step_1000 | L_oracle_precision | old_snr_max_fallback_ablation | 1e-05 | 0.00355326 | 0.00972553 | 0.011315 | 1.16503 | 0.898718 | low_h2_noise;ablation_only;fallback_max_snr;selected_low_h2_noise |
| fp16 | step_1000 | L_oracle_precision | plateau_q90_primary | 0.0015 | 6.27603e-06 | 1.47168e-05 | 1.75095e-05 | 0.583991 | 0.396528 |  |
| fp16 | step_1000 | L_oracle_precision | plateau_q95_conservative | 0.0015 | 6.27603e-06 | 1.47168e-05 | 1.75095e-05 | 0.583991 | 0.396528 |  |
| fp16 | step_1000 | L_oracle_oldSNR | old_snr_smallest_pass |  |  |  |  |  |  | no_snr_pass |
| fp16 | step_1000 | L_oracle_oldSNR | old_snr_max_fallback_ablation | 1e-05 | 0.00355326 | 0.00972553 | 0.011315 | 1.16503 | 0.898718 | low_h2_noise;ablation_only;fallback_max_snr;selected_low_h2_noise |
| fp16 | step_1000 | L_oracle_oldSNR | plateau_q90_primary | 0.0015 | 6.27603e-06 | 1.47168e-05 | 1.75095e-05 | 0.583991 | 0.396528 |  |
| fp16 | step_1000 | L_oracle_oldSNR | plateau_q95_conservative | 0.0015 | 6.27603e-06 | 1.47168e-05 | 1.75095e-05 | 0.583991 | 0.396528 |  |
| fp16 | final | L_clean32 | old_snr_smallest_pass | 1e-05 | 2.19169e-05 | 4.09147e-05 | 4.80698e-05 | 2.02347 | 0.167537 |  |
| fp16 | final | L_clean32 | old_snr_max_fallback_ablation | 1e-05 | 2.19169e-05 | 4.09147e-05 | 4.80698e-05 | 2.02347 | 0.167537 | ablation_only |
| fp16 | final | L_clean32 | plateau_q90_primary | 0.001 | 2.02286e-05 | 2.9366e-05 | 3.1035e-05 | 2.69826 | 0.0903291 |  |
| fp16 | final | L_clean32 | plateau_q95_conservative | 0.001 | 2.02286e-05 | 2.9366e-05 | 3.1035e-05 | 2.69826 | 0.0903291 |  |
| fp16 | final | L_oracle_precision | old_snr_smallest_pass | 0.001 | 2.03293e-05 | 2.93404e-05 | 3.05604e-05 | 2.57253 | 0.0864423 |  |
| fp16 | final | L_oracle_precision | old_snr_max_fallback_ablation | 0.001 | 2.03293e-05 | 2.93404e-05 | 3.05604e-05 | 2.57253 | 0.0864423 | ablation_only |
| fp16 | final | L_oracle_precision | plateau_q90_primary | 0.0015 | 2.29623e-05 | 2.93499e-05 | 3.05695e-05 | 3.63213 | 0.283217 |  |
| fp16 | final | L_oracle_precision | plateau_q95_conservative | 0.0015 | 2.29623e-05 | 2.93499e-05 | 3.05695e-05 | 3.63213 | 0.283217 |  |
| fp16 | final | L_oracle_oldSNR | old_snr_smallest_pass | 0.001 | 2.03293e-05 | 2.93404e-05 | 3.05604e-05 | 2.57253 | 0.0864423 |  |
| fp16 | final | L_oracle_oldSNR | old_snr_max_fallback_ablation | 0.001 | 2.03293e-05 | 2.93404e-05 | 3.05604e-05 | 2.57253 | 0.0864423 | ablation_only |
| fp16 | final | L_oracle_oldSNR | plateau_q90_primary | 0.0015 | 2.29623e-05 | 2.93499e-05 | 3.05695e-05 | 3.63213 | 0.283217 |  |
| fp16 | final | L_oracle_oldSNR | plateau_q95_conservative | 0.0015 | 2.29623e-05 | 2.93499e-05 | 3.05695e-05 | 3.63213 | 0.283217 |  |
| fp32 | step_1000 | L_clean32 | old_snr_smallest_pass |  |  |  |  |  |  | no_snr_pass |
| fp32 | step_1000 | L_clean32 | old_snr_max_fallback_ablation | 0.0015 | 1.28846e-05 | 1.97773e-05 | 2.49841e-05 | 1.99154 | 0.166689 | ablation_only;fallback_max_snr |
| fp32 | step_1000 | L_clean32 | plateau_q90_primary | 0.003 | 5.26101e-06 | 1.64806e-05 | 2.13342e-05 | 0.761452 | 0.0437579 |  |
| fp32 | step_1000 | L_clean32 | plateau_q95_conservative | 0.003 | 5.26101e-06 | 1.64806e-05 | 2.13342e-05 | 0.761452 | 0.0437579 |  |
| fp32 | step_1000 | L_oracle_precision | old_snr_smallest_pass |  |  |  |  |  |  | no_snr_pass |
| fp32 | step_1000 | L_oracle_precision | old_snr_max_fallback_ablation | 0.0015 | 1.28846e-05 | 1.97773e-05 | 2.49841e-05 | 1.99154 | 0.166689 | ablation_only;fallback_max_snr |
| fp32 | step_1000 | L_oracle_precision | plateau_q90_primary | 0.003 | 5.26101e-06 | 1.64806e-05 | 2.13342e-05 | 0.761452 | 0.0437579 |  |
| fp32 | step_1000 | L_oracle_precision | plateau_q95_conservative | 0.003 | 5.26101e-06 | 1.64806e-05 | 2.13342e-05 | 0.761452 | 0.0437579 |  |
| fp32 | step_1000 | L_oracle_oldSNR | old_snr_smallest_pass |  |  |  |  |  |  | no_snr_pass |
| fp32 | step_1000 | L_oracle_oldSNR | old_snr_max_fallback_ablation | 0.0015 | 1.28846e-05 | 1.97773e-05 | 2.49841e-05 | 1.99154 | 0.166689 | ablation_only;fallback_max_snr |
| fp32 | step_1000 | L_oracle_oldSNR | plateau_q90_primary | 0.003 | 5.26101e-06 | 1.64806e-05 | 2.13342e-05 | 0.761452 | 0.0437579 |  |
| fp32 | step_1000 | L_oracle_oldSNR | plateau_q95_conservative | 0.003 | 5.26101e-06 | 1.64806e-05 | 2.13342e-05 | 0.761452 | 0.0437579 |  |
| fp32 | final | L_clean32 | old_snr_smallest_pass | 1e-05 | 2.77624e-05 | 6.04881e-05 | 7.45123e-05 | 2.13589 | 0.245754 |  |
| fp32 | final | L_clean32 | old_snr_max_fallback_ablation | 1e-05 | 2.77624e-05 | 6.04881e-05 | 7.45123e-05 | 2.13589 | 0.245754 | ablation_only |
| fp32 | final | L_clean32 | plateau_q90_primary | 0.002 | 4.95757e-05 | 7.01863e-05 | 8.86744e-05 | 4.32461 | 0.386151 |  |
| fp32 | final | L_clean32 | plateau_q95_conservative | 0.002 | 4.95757e-05 | 7.01863e-05 | 8.86744e-05 | 4.32461 | 0.386151 |  |
| fp32 | final | L_oracle_precision | old_snr_smallest_pass | 1e-05 | 2.77624e-05 | 6.04881e-05 | 7.45123e-05 | 2.13589 | 0.245754 |  |
| fp32 | final | L_oracle_precision | old_snr_max_fallback_ablation | 1e-05 | 2.77624e-05 | 6.04881e-05 | 7.45123e-05 | 2.13589 | 0.245754 | ablation_only |
| fp32 | final | L_oracle_precision | plateau_q90_primary | 0.002 | 4.95757e-05 | 7.01863e-05 | 8.86744e-05 | 4.32461 | 0.386151 |  |
| fp32 | final | L_oracle_precision | plateau_q95_conservative | 0.002 | 4.95757e-05 | 7.01863e-05 | 8.86744e-05 | 4.32461 | 0.386151 |  |
| fp32 | final | L_oracle_oldSNR | old_snr_smallest_pass | 1e-05 | 2.77624e-05 | 6.04881e-05 | 7.45123e-05 | 2.13589 | 0.245754 |  |
| fp32 | final | L_oracle_oldSNR | old_snr_max_fallback_ablation | 1e-05 | 2.77624e-05 | 6.04881e-05 | 7.45123e-05 | 2.13589 | 0.245754 | ablation_only |
| fp32 | final | L_oracle_oldSNR | plateau_q90_primary | 0.002 | 4.95757e-05 | 7.01863e-05 | 8.86744e-05 | 4.32461 | 0.386151 |  |
| fp32 | final | L_oracle_oldSNR | plateau_q95_conservative | 0.002 | 4.95757e-05 | 7.01863e-05 | 8.86744e-05 | 4.32461 | 0.386151 |  |

## Table B: Primary Comparison

| precision | checkpoint | L_clean32 plateau q90 | L_oracle plateau q90 | L_oracle oldSNR q90 | interpretation |
|---|---|---:|---:|---:|---|
| fp16 | final | 2.9366e-05 | 2.93499e-05 | 2.93404e-05 | see selector flags |
| fp16 | step_1000 | 1.46317e-05 | 1.47168e-05 | 0.00972553 | oldSNR selected flagged low-h2 noise |
| fp32 | final | 7.01863e-05 | 7.01863e-05 | 6.04881e-05 | see selector flags |
| fp32 | step_1000 | 1.64806e-05 | 1.64806e-05 | 1.97773e-05 | see selector flags |

## Interpretation

- The theoretical h-star formula should consume `L_clean32` with `plateau_q90_primary` when that selector is available.
- `L_oracle_precision` is an oracle-consistent diagnostic; it should not replace clean FP32 curvature unless the downstream analysis explicitly wants oracle noise folded into L.
- `L_oracle_oldSNR` is an ablation to expose the previous max-SNR fallback behavior.
- Warnings were emitted; see `L_diagnostics.json` for details.

## Run Interpretation

- FP16 oldSNR chose a tiny h2 at step_1000: h2=1e-05, L_q90=0.00972553, flags=ablation_only;fallback_max_snr;selected_low_h2_noise;low_h2_noise.
- FP16 final oldSNR did not choose the tiny point: h2=0.001, L_q90=2.93404e-05, flags=ablation_only.
- Plateau selections moved to the 1e-3 to 4e-3 band: fp32/step_1000: clean32 h2=0.003, oracle h2=0.003; fp32/final: clean32 h2=0.002, oracle h2=0.002; fp16/step_1000: clean32 h2=0.0015, oracle h2=0.0015; fp16/final: clean32 h2=0.001, oracle h2=0.0015.
- Clean/oracle plateau L_q90 ratios were small, not >5x: fp32/step_1000: 1x; fp32/final: 1x; fp16/step_1000: 1.01x; fp16/final: 1x.
- FP16 oracle is noisy at tiny h2 for step_1000, but plateau_q90_primary avoided that point; selected oracle rows are not flagged low_h2_noise.
- For theoretical h-star, use L_clean32 + plateau_q90_primary. Use L_oracle_precision + plateau_q90_primary only as oracle-consistent diagnostic, and oldSNR as ablation.
