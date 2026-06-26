# Missing Items V2

- No `main.tex` or `hwindow_overleaf_draft*.tex` file was found, so the package includes `PAPER_INSERTION_SNIPPETS.tex` instead of a compiled paper PDF.
- BF16 lacks a valid empirical accuracy interval in the located final tables and is omitted from the main precision-window figure/table.
- FP32/FP16 do not have complete `Delta_eff/G/L_loc` frozen-formula provenance tables; they are reported empirical-only.
- Prefix INT4 and several multi-task rows are single-seed; captions and takeaways state this explicitly.
- OPT is a cross-architecture sanity check, with TREC retained as a failure.
