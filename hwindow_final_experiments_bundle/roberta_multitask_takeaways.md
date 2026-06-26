# RoBERTa INT4 Multi-task Takeaways

- Results were copied from raw summary CSVs and retain `run_type`, seed, and exact `h`.
- Dense, sparse p=0.1, and prefix rows are not averaged together.
- Defaults are competitive when they fall inside broad windows.
- Reference-radius policies help in some narrow/extreme low-precision settings but are not claimed to beat default everywhere.
- Prefix/sparse rows with a single seed should be treated as single-seed evidence.
