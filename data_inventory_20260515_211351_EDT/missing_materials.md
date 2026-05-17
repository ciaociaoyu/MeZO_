# Missing Or Risky Materials

| material_or_risk | detail |
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
| residual pre-fix not final evidence | First residual round has near-zero-scale residual-over-scale anomaly; use only as diagnostic history. |
| unsupported draft claims | No manuscript draft files were found locally, so draft claim support could not be audited. |
