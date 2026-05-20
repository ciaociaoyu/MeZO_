# INT8 RTNClip MSE Reprobe Summary

This run separates quantizer reconstruction, perturbation visibility, true finite-difference error, and Richardson locality. Reconstruction MSE is not finite-difference MSE.

True-gradient diagnostics: available (computed against unquantized master objective).

## Interpretation

- Small h visibility-bad: yes (`h=1e-5` delta_visibility_nmse=34).
- `h=1e-3` visibility-good: yes; locality-good: yes (delta_visibility_nmse=0.03855, richardson_rmse_rel=0.546, fd_true_nmse=0.06404, corr_fd_true=0.9686).
- `h=1e-2` visibility-good: yes (delta_visibility_nmse=0.003541).
- `h=1e-2` locality-bad: yes by fd_true_nmse (richardson_rmse_rel=0.8737, fd_true_nmse=0.7803).

## Field Semantics

- `weight_recon_mse` and `recon_mse_global` measure `Q_t(w_t)` reconstruction only.
- `plus_recon_mse` and `minus_recon_mse` measure reconstruction of each perturbed state.
- `delta_visibility_nmse` measures whether `Q_t(w_t+h u)-Q_t(w_t-h u)` exposes the intended displacement; it is visibility-only.
- `fd_true_nmse` is the finite-difference true-gradient MSE when true gradients are available.
- `richardson_rmse_rel` is the self-consistency locality proxy from `d_h` versus `d_{h/2}`.

## Per-h Summary

| h | delta_visibility_nmse | alignment | norm_ratio | richardson_rmse_rel | fd_true_nmse | corr_fd_true |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-05 | 34 | 0.1682 | 5.915 | 0.7541 | 18.15 | 0.1893 |
| 3e-05 | 10.59 | 0.2916 | 3.402 | 0.7005 | 16.54 | 0.3478 |
| 0.0001 | 2.493 | 0.5321 | 1.865 | 0.7031 | 2.534 | 0.6221 |
| 0.0003 | 0.3999 | 0.844 | 1.179 | 0.5863 | 0.2569 | 0.9033 |
| 0.001 | 0.03855 | 0.9811 | 1.015 | 0.546 | 0.06404 | 0.9686 |
| 0.0015 | 0.01831 | 0.9909 | 1.005 | 0.3115 | 0.01841 | 0.9911 |
| 0.002 | 0.01126 | 0.9944 | 1.001 | 0.2929 | 0.02642 | 0.9846 |
| 0.003 | 0.006278 | 0.9969 | 0.9984 | 0.2834 | 0.06334 | 0.9745 |
| 0.004 | 0.004606 | 0.9977 | 0.9972 | 0.3176 | 0.1724 | 0.9143 |
| 0.005 | 0.003897 | 0.9981 | 0.9965 | 0.2959 | 0.1543 | 0.93 |
| 0.01 | 0.003541 | 0.9982 | 0.9936 | 0.8737 | 0.7803 | 0.5016 |

## Plots

- `int8_mse_probe_plots/alignment_vs_h.svg`
- `int8_mse_probe_plots/corr_fd_true_vs_h.svg`
- `int8_mse_probe_plots/delta_visibility_nmse_vs_h.svg`
- `int8_mse_probe_plots/fd_true_nmse_vs_h.svg`
- `int8_mse_probe_plots/norm_ratio_vs_h.svg`
- `int8_mse_probe_plots/richardson_rmse_rel_vs_h.svg`
- `int8_mse_probe_plots/training_acc_overlay.svg`

Existing training overlay source: `/scratch/jy03364/MeZO_/outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch/int8_hsearch_summary.csv`.
