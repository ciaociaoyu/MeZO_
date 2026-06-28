# RoBERTa/SST-5 Robust Dual-Scheme Window Results

Generated: 2026-06-28T18:29:56

This folder is probe-only. No new training was run.

Source probe directory: `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627`

Main outputs:
- `empirical_accuracy_windows.csv`: existing h-sweep accuracy good sets.
- `probe_summary_by_h.csv`: scalar nMSE, vector-level rho, corr/sign, d_h statistics.
- `pure_theory_plugin_windows.csv`: direct plug-in windows from Delta_eff and L_loc estimates, with status fields.
- `practical_probe_windows.csv`: threshold-based practical probe-visible windows, including FP16 dead-zone handling.
- `rho_fit_windows.csv`: smooth A/h^2+B h^2 rho-fit windows only where stable.
- `window_comparison_for_paper.csv`: single-row-per-precision comparison for paper use.

Important definitions:
- `scalar_nmse` is not rho.
- `rho_raw = mean((d_h-d_star)^2 ||u||^2) / V_dir_sample`.
- Practical windows use fixed probe thresholds and are not tuned by accuracy.
- Empirical accuracy windows use existing h-sweep results with `best_acc - 0.01` as the primary threshold.
