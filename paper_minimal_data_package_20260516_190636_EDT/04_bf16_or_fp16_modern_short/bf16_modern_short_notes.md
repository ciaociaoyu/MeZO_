# BF16 / FP16 Modern Short Notes

This folder currently contains the modern 300-step BF16 rows from `window_training_summary.csv`.

Use this as short validation only. It is not enough to replace a clean 2k/5k BF16 or FP16 training matrix.

Important distinction:

- BF16 modern probe/training rows are BF16.
- FP16 historical rows in `03_fp16_training_historical/` are FP16.
- Do not merge BF16 and FP16 under one label without explicitly saying so.
