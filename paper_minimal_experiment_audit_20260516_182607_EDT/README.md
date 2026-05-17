# Minimal Experiment Audit Summary

This directory contains a read-only audit generated from existing result files only. No training was run and no source/training code was modified.

Key outputs:

- `inventory_all_results.csv`: 162 parsed result/probe/diagnostic rows.
- `inventory_minimal_requirements.csv`: mapping of the 10 requested requirements to existing evidence.
- `missing_experiments.csv`: P0/P1/P2 missing items.
- `recommended_next_runs.md`: short prioritized next-run list and direct answers to the 13 audit questions.
- `paper_ready_claims.md`: claim-to-evidence table for the minimal paper framing.

Main conclusion: the precision-aware dense probe-window story is paper-ready; the training-validation story is conditional because FP32/BF16-or-FP16 clean modern training validation is still the main gap.
