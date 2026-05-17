
# Data Inventory Report

## Executive Summary

This audit found dense and sparse probe-window diagnostics, short 300-step training validation, dense INT8 fp16-master 2k training packages, and several residual-grid/update-commit diagnostic packages. The strongest evidence for the main perturbation-visibility story is the May 12 dense/sparse probe-window package, especially `dense_probe_summary.csv` and `sparse_probe_summary.csv`, with raw JSONL and plots inside the package archive. Training evidence is useful but still mostly validation-scale: 300-step runs are short and older `final_acc` summaries are ambiguous, while the May 15 dense A100 table is cleaner because it separates `best_eval_acc` and `last_eval_acc`.

Residual/update-commit evidence is available and should support an appendix or secondary backend discussion. The post-fix residual consistency archive and the later residual H100 package show no-op checks, synthetic residual tests, one-step equation checks, scale drift checks, residual-bound checks, and one 2k promoted residual run. Pre-fix residual-over-scale anomalies should be treated as diagnostic history only.

Before writing the paper, the missing pieces are manuscript resources (`main.tex`, bibliography, style files), final paper-ready figures, completed sparse training summaries, more long-run/multi-seed training evidence, and cleanup of ambiguous `final_acc` fields.

## Data Directories Found

| directory | candidate_file_count |
| --- | --- |
| medium_models/sh_file/sst5 | 862 |
| experiments/pilot/mezo | 464 |
| experiments/smoke/mezo | 323 |
| experiments/pilot/sparse_mezo | 304 |
| experiments/main/mezo | 250 |
| experiments/main/sparse_mezo | 147 |
| experiments/smoke/sparse_mezo | 110 |
| experiments/stage1_fixed_h_fp16_20260428/results | 84 |
| experiments/pilot/hizoo | 52 |
| experiments/pilot/lozo | 52 |
| experiments/future_probe_window_training_20260512_235627_package/dense_runs | 38 |
| experiments/pilot/_shared | 37 |
| experiments/smoke/lozo | 24 |
| experiments/smoke/hizoo | 17 |
| experiments/int8_update_sparse_plan/next_window_sparse_residual_20260515 | 15 |
| medium_models/sh_file/sst-2 | 13 |
| experiments/main/_shared | 12 |
| experiments/int8_update_sparse_plan/probe_window_h100_20260512 | 9 |
| experiments/smoke/validation | 9 |
| experiments/int8_error_origin_probe/results_baseline_20260512 | 6 |

## Main Experiment Lines And Status

| experiment_line | available_data | status |
| --- | --- | --- |
| dense_probe | 33 summary rows | strong diagnostic coverage across FP32/BF16/INT8 |
| sparse_probe | 24 summary rows | diagnostic coverage across p and h_active; training follow-up partial |
| training_validation | 25 parsed training rows | 300-step short validation plus 2k dense runs; limited long-run evidence |
| future_window_training | 6 dense 2k rows, 0 sparse rows | dense package complete for seed 0; sparse incomplete |
| residual_grid | 16 parsed residual run rows | post-fix mechanics clean; 2k residual promotion exists but still secondary |
| paper_draft | 0 manuscript/bib/style files | manuscript/bib/style resources missing by exact names |

## Complete Results

| result | evidence | status |
| --- | --- | --- |
| May 12 dense probe | experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv | complete diagnostic table, raw JSONL/plots in archive |
| May 12 sparse probe | experiments/int8_update_sparse_plan/probe_window_h100_20260512/sparse_probe_summary.csv | complete diagnostic table over p and h_active, raw JSONL/plots in archive |
| May 12 300-step validation | experiments/int8_update_sparse_plan/probe_window_h100_20260512/window_training_summary.csv | 11 short validation runs, no NaNs, final_acc ambiguous |
| Future dense 2k package | experiments/future_probe_window_training_20260512_235627_package/summary_all.csv | 6 seed-0 dense runs, final/best only |
| May 15 dense A100 2k | experiments/int8_update_sparse_plan/next_window_sparse_residual_20260515/cluster_dense_a100/summary_dense_by_h.csv | 8 completed runs with best/last separated |
| Residual post-fix consistency | experiments/int8_update_sparse_plan/results/residual_consistency_20260512_190000_key_results.tar.gz | debug checks and 50-step summaries |
| Residual H100 package | results_packages/residual_local_h100_20260515_172944_essential.tar.gz | sanity checks, 500-step runs, one 2k promoted residual run |

## Partial Or Incomplete Results

| result | evidence | caveat |
| --- | --- | --- |
| future sparse training | experiments/future_probe_window_training_20260512_235627_package/summary_sparse.csv | header only; sparse_partial update_stats exists without completed summary |
| May 15 h=3e-3 dense A100 | experiments/int8_update_sparse_plan/next_window_sparse_residual_20260515/cluster_dense_a100/summary_dense_by_h.csv | 2 seeds, not the same 3-seed count as h=1e-3 and h=2e-3 |
| residual long-run | residual local package | 2k promoted run exists, but no 5k-20k multi-seed summary found |
| paper resources | repo-wide search | main manuscript, bib, and style files missing |

## Results Suitable For Main Paper

- Dense probe h-window diagnostics across FP32/BF16/INT8.
- Sparse INT8 probe diagnostics framed by `h_active = h_raw / sqrt(p)`.
- May 15 dense A100 2k validation can be used as preliminary training validation because best/last metrics are separated.

## Results Suitable For Appendix

- 300-step training validation.
- Checkpoint probe stats from the future training package.
- Residual-grid consistency diagnostics and post-fix update-commit mechanics.
- Residual 500-step and 2k promoted H100 runs, clearly labeled as secondary backend evidence.

## Results Not Safe To Cite Yet

- Any `final_acc` value from summaries that do not distinguish best vs last.
- Sparse training as a completed follow-up; local summary rows are absent.
- Residual-grid as a mature training backend; one 2k run exists but no long multi-seed run was found.
- Manuscript-specific claims, because no local manuscript files were found.

## Missing Files

| missing_or_risky_material | detail |
| --- | --- |
| missing .bib files | No `.bib` files found in scanned file inventory. |
| missing paper scaffold | `main.tex`, `emnlp_sst5_story_with_formulas.tex`, and `emnlp2026_submission_requirements_and_strategy_v2.md` not found. |
| missing style files | No `.sty` or `.cls` files found. |
| missing final paper plots | Result plots exist, but no paper figure directory or manuscript-linked figure names were found. |
| missing raw logs for compact packages | May 15 next-window package says it intentionally excludes full training logs/checkpoints. |
| missing command manifests for older/legacy runs | Several legacy SST-5 h-sweep result dirs have metrics/eval files but no obvious command manifest beside them. |
| missing seeds | Future dense 2k package is seed 0 only; May 15 h=3e-3 has 2 seeds while h=1e-3 and h=2e-3 have 3. |
| missing long dense training | No 5k-20k dense main-line validation summary was found in the named packages. |
| ambiguous final_acc summaries | 300-step window and older future summaries use final_acc/best_acc without separate last_eval_acc; see `summary_inventory.csv`. |
| sparse training incomplete | Future `summary_sparse.csv` has no data rows; sparse_partial has update_stats but no completed run summary. |

## Recommended Next Actions

1. Normalize summary schemas so `best_eval_acc`, `last_eval_acc`, and legacy `final_acc` are explicit.
2. Complete sparse training summaries before citing sparse training outcomes.
3. Add or recover paper resources: manuscript TeX, bibliography, style files, macros, and figure directory.
4. Promote paper figures from existing plots or regenerate from the audited CSVs with reproducible scripts.
5. Run no new experiments as part of this audit; any future long-run work should be scheduled separately.

## Inventory Counts

- Files scanned: 3841.
- Experiment summaries found: 83.
- Archives inspected: 7.
- Archive listings: `archive_contents/*.txt`.
