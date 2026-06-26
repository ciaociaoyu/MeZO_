# Artifact Audit

Historical TeX references and previous bundles were scanned. Final main artifacts are regenerated from canonical processed data and listed in `FIGURE_DATA_MANIFEST.csv` and `TABLE_DATA_MANIFEST.csv`.

Key decisions:

- True-MSE plots use only audited `fd_true_nmse` fields.
- Geometry/proxy curves are relabeled as visibility diagnostics.
- The RoBERTa INT4 multi-task main tables use full runs only and do not choose rows by accuracy.
- OPT is retained as a cross-architecture sanity check, with TREC failure included.
- FP32/FP16 true directional MSE curves are omitted because no reliable audited `A_true` data were found.
