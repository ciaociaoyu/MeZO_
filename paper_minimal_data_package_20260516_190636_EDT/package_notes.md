# Minimal Paper Data Package Notes

## What This Package Is

This is an existing-data package for minimal paper writing. It collects, copies, summarizes, and annotates existing experimental data for the precision-aware ZO perturbation-window story on RoBERTa-large / SST-5.

## What This Package Is Not

- It is not a paper draft.
- It does not include `draft_v0.md` or `draft_v0.tex`.
- It does not include new experiments.
- It does not claim all P0 final experiments are complete.

## How To Use The Folders

- Start with `08_ready_tables/` for paper-writing tables.
- Use `01_probe_window_dense/` for the main probe-window story.
- Use `05_int8_fp16master_training/` for INT8-forward + FP16-master training validation.
- Use `02_fp32_training_historical/` and `03_fp16_training_historical/` only with their historical caveats.
- Use `06_int8_update_commit_supplement/` as appendix/supplement evidence.
- Use `07_sparse_auxiliary/` only for auxiliary sparse discussion.
- Use `09_ready_figures_or_plot_data/` for plotting CSVs.

## Key Caveats

- FP32 old training evidence is historical/provisional if using old seed13.
- FP16 evidence may be historical and should not be described as BF16 unless the source is actually BF16.
- INT8-forward + FP16-master is not true INT8 update.
- Direct INT8 / residual-grid are update-commit supplements.
- Sparse is auxiliary.
- Geometry-only window heuristics are insufficient because h=1e-2 can look geometrically visible while derivative correlation is poor.

## Recommended Next Data To Send To ChatGPT

- `08_ready_tables/table_probe_window_by_precision.csv`
- `08_ready_tables/table_training_validation_existing.csv`
- `08_ready_tables/table_int8_update_supplement.csv`
- `03_fp16_training_historical/fp16_old_seed13_notes.md`
- `05_int8_fp16master_training/int8_fp16master_notes.md`
- `package_notes.md`
