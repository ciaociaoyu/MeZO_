# Key Experiment Index

This package is organized for paper-writing input, not as a draft.

- `01_probe_window_dense/`: modern dense probe-window evidence for FP32, BF16, and INT8. This is the strongest main-story data.
- `02_fp32_training_historical/`: old FP32 seed13 / 20k sweep, copied as provisional training evidence for P0-1. Historical only.
- `03_fp16_training_historical/`: old FP16 seed13 / 20k and newer FP16 seed16 / 50k summaries. Historical FP16, not BF16.
- `04_bf16_or_fp16_modern_short/`: modern 300-step BF16/low-precision short validation from the probe-window package.
- `05_int8_fp16master_training/`: modern INT8-forward + FP16-master dense training evidence, including 2k anchors and 5k 3-seed selected h values.
- `06_int8_update_commit_supplement/`: direct INT8 and residual-grid update-commit diagnostics. Appendix/supplement only.
- `07_sparse_auxiliary/`: sparse probe and sparse screening evidence. Appendix/auxiliary only.
- `08_ready_tables/`: tables most useful for paper writing.
- `09_ready_figures_or_plot_data/`: plot-ready CSVs and curve manifest.
- `10_missing_for_final/`: P0/P1/P2 gaps before a final clean paper version.

Traceability: every copied source file is listed in `00_manifest/data_sources_inventory.csv` with its original path.
