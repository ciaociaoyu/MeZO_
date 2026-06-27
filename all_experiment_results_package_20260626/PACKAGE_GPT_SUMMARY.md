# GPT Summary for the MeZO / h-window Results Package

This file is a compact entry point for GPT-style retrieval over the
records-only experiment package. It explains what is in the package, which
files should be read first, how to find results, and which caveats matter.

Package root after extraction:

```text
all_experiment_results_package_20260626/
```

The records-only archive intentionally excludes tokenizer files, model
checkpoints, model weights, and other large binary artifacts. It keeps result
records and intermediate analysis data such as CSV, JSON, JSONL, Markdown,
LaTeX tables, figures, manifests, logs, run configs, run summaries, and
diagnostic outputs.

## Quick Start for GPT

Read these files first, in this order:

1. `PACKAGE_GPT_SUMMARY.md`
   - This high-level map.
2. `README.md`
   - Package generation metadata and inclusion policy.
3. `docs/EXPERIMENT_GUIDE.md`
   - Human-readable overview of the main experiment families.
4. `docs/FORMULAS_AND_ALGORITHMS.md`
   - Definitions of h-window formulas, MeZO, RTNClip, sparse mode, prefix mode,
     residual-grid, and directional-MSE diagnostics.
5. `docs/PARAMETER_DICTIONARY.md`
   - Meaning of common run/config fields.
6. `docs/RESULT_HIGHLIGHTS_AND_CAVEATS.md`
   - Main conclusions and limitations.
7. `indices/run_index.csv`
   - Run-level searchable table.
8. `indices/experiment_family_index.csv`
   - Experiment-root-level searchable table.
9. `indices/artifact_manifest.csv`
   - File provenance and checksum table.
10. `indices/excluded_files.csv`
   - Files deliberately left out of the package.

Recommended shell queries after extraction:

```bash
cd all_experiment_results_package_20260626
rg -n "roberta|sst-5|int4|rtnclip|residual|hstar|fd_true_nmse" .
python - <<'PY'
import pandas as pd
runs = pd.read_csv("indices/run_index.csv")
print(runs.head())
print(runs.columns.tolist())
PY
```

## Top-Level Directory Map

```text
docs/
  Method notes, formula definitions, parameter dictionary, and caveats.

indices/
  Machine-searchable indexes:
  - run_index.csv
  - experiment_family_index.csv
  - artifact_manifest.csv
  - excluded_files.csv

artifacts/
  Copied experiment outputs and analysis bundles from the repo.
  Important subtrees include:
  - outputs/
  - experiments/
  - paper_artifacts_final/
  - hwindow_final_experiments_bundle_v2/
  - hwindow_12h_highdim_bundle/
  - interval_aware_h_probe/
  - interval_h_selection_8h_bundle/
  - synthetic_fit_repair/
  - synthetic_no_leakage_interval/
  - sharp_interval_fit_and_roberta_int4_eval/
  - lowbit_update_experiment/
  - tools/
```

## Package Metadata

From `metadata.json`:

- Generated at: `2026-06-26T21:09:37`
- Source repository root: `/scratch/jy03364/MeZO_`
- Package generation git commit: `f06c3df61be6e9e0af546790ceee288f0eb9982b`
- Included artifacts: `10879`
- Indexed runs: `510`
- Indexed experiment roots: `182`
- Excluded files: `2310`
- Maximum included file size in original package script: `50 MB`

The latest repository commit at the time the records-only archive was prepared
may be newer than the package-generation commit. Use `artifact_manifest.csv`
for file-level provenance.

## What This Package Is For

Use this package to answer questions about:

- RoBERTa-large / SST-5 precision-window sweeps.
- FP32, FP16, BF16, INT8, and INT4 perturbation-radius behavior.
- INT4 RTN/RTNClip dense, sparse, prefix, and residual-grid experiments.
- OPT-1.3B low-bit cross-architecture sanity checks.
- Directional finite-difference MSE, correlation, visibility, active fraction,
  alignment, and norm-ratio diagnostics.
- h-star / h-ref / clean-GL / lowbit-GL radius estimates.
- Interval-aware and frozen-window paper artifact generation.
- Synthetic high-dimensional h-window validation experiments.
- Final paper figure/table provenance.

Do not use this package as a source for:

- Restoring model checkpoints.
- Re-running training from saved weights.
- Loading tokenizer/model files.
- Claiming exact reproducibility of omitted checkpoint states.

## Main Experiment Families

### 1. Final paper artifacts and frozen-window package

Likely locations:

```text
artifacts/paper_artifacts_final/
artifacts/hwindow_final_experiments_bundle/
artifacts/hwindow_final_experiments_bundle_v2/
```

Use these for final paper-facing figures, tables, manifests, and audit reports.
Important files may include:

- `FINAL_ARTIFACT_SUMMARY.md`
- `FINAL_EXPERIMENT_SUMMARY*.md`
- `VALIDATION_REPORT.md`
- `FIGURE_DATA_MANIFEST.csv`
- `TABLE_DATA_MANIFEST.csv`
- `precision_window_theory_vs_empirical.csv`
- `roberta_multitask_main.csv`
- `opt_cross_arch_verified.csv`

### 2. RoBERTa / SST-5 precision-window sweeps

Search keywords:

```text
roberta-large sst5 sst-5 fp32 fp16 bf16 int8 int4 h_sweep rtnclip precision_window
```

Use these to compare measured accuracy intervals, directional-MSE intervals,
and frozen theoretical windows. The package contains both final processed
tables and older intermediate sweep outputs.

Important distinction:

- Theoretical window: defined by frozen formulas using `h_ref`, `rho(h)`, and
  `W_tau`.
- Empirical accuracy good set: usually `Acc(h) >= max_h Acc(h) - 0.01`.
- True directional MSE: only fields verified as
  `E[(d_Q - d_star)^2] / E[d_star^2]`.
- Geometry diagnostics such as `A_cross`, `A_interval`, active fraction,
  alignment, and norm ratio are not true directional MSE.

### 3. RoBERTa INT4 multi-task experiments

Likely locations:

```text
artifacts/outputs/int4_full_data_hstar_dense_sparse_20260522_113710/
artifacts/outputs/int4_cleanGL_hstar_dense_sparsep0p1_20k_20260523_142501/
artifacts/outputs/int4_lowbitL_hstar_dense_sparse_20260522_20260522_223513/
artifacts/outputs/int4_prefix_mezo32_full_data_20k_20260523_062851/
artifacts/outputs/int4_prefix_quantized_cleanGL_20k_20260523_154026/
artifacts/outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/
```

Tasks include SST-2, SST-5, RTE, MNLI, and TREC when available. Modes include:

- dense INT4
- sparse p=0.1
- sparse p=0.01
- prefix
- quantized-prefix / prefix-mezo variants

Common policies:

- `fixed_small`: usually `h=1e-5`
- `mezo_default`: `h=1e-3`
- `hstar_ours`, `cleanGL`, `lowbitL`, or similar analytical-reference policies

Use `indices/run_index.csv` to find exact h values and run summaries. Do not
assume policy names are consistent across historical runs; verify `h`,
`source_path`, and `run_summary.json`.

### 4. Sparse INT4 / effective-dimension experiments

Search keywords:

```text
sparse p0p1 p0p01 highestabs highweight extreme_sparse active_frac mask
```

Important caveat:

- Some older sparse runs used highest-absolute-weight masks and were later
  marked less reliable for method claims.
- Later seed-fixed runs saved masks and fixed seed alignment problems.
- Always read local summary/caveat files near the run directory.

### 5. Prefix experiments

Search keywords:

```text
prefix prefix_int4 prefix_mezo32 int4_prefix quantized_prefix
```

Important caveat:

- There were several prefix variants:
  - full FP32/MeZO-style prefix path;
  - INT4 base with FP32 prefix;
  - prefix parameters also quantized / INT4 finite-difference path.
- Do not merge these variants. Check `run_config.json`, `run_summary.json`,
  and local reports.

### 6. OPT-1.3B experiments

Likely locations:

```text
artifacts/outputs/opt13b_int4_*/
artifacts/outputs/official_*opt13b*/
artifacts/lowbit_update_experiment/
```

Search keywords:

```text
opt13b opt-1.3b facebook/opt-1.3b int4 int8 dense sparse gptq awq qzo
```

The OPT results are cross-architecture sanity checks, not direct reproduction
of every original MeZO benchmark. Some tasks show moderate or substantial gaps
relative to default. Do not hide failed tasks such as TREC when present.

### 7. Residual-grid experiments

Search keywords:

```text
residual residual_grid error_feedback int4_residual_grid int8 residual_local
```

Important conclusion from repository inspection:

- INT8 residual-grid runs include higher SST-5 accuracies around 0.46 in older
  local experiments.
- True INT4 residual-grid RTNClip runs do not show a successful stable training
  record. For example, the INT4 SST-5 20k resume run has best accuracy around
  0.323 and last accuracy around 0.266.
- Do not confuse old INT8 residual-grid success with INT4 residual-grid
  success.

### 8. Synthetic and analytical h-window experiments

Likely locations:

```text
artifacts/synthetic_fit_repair/
artifacts/synthetic_no_leakage_interval/
artifacts/hwindow_12h_highdim_bundle/
artifacts/hwindow_12h_highdim_extra_*/
```

Use these for controlled analysis of:

- `alpha / h^2 + beta h^2 + gamma`
- `alpha / h^2 + beta h^4 + gamma`
- high-dimensional random-direction floors
- visibility and locality terms
- no-target-leakage interval-aware predictor checks

The final paper-facing package freezes the theory and does not introduce a new
selector based on interval-aware replacement theory.

### 9. Low-bit update / GPTQ-style experiments

Likely locations:

```text
artifacts/lowbit_update_experiment/
artifacts/outputs/official_gptq*/
artifacts/outputs/official_autoawq*/
```

Use these for separate low-bit commit-rule and GPTQ/AWQ/QZO investigations.
They are not the main RoBERTa h-window training table.

## Key Data Files

### `indices/run_index.csv`

Run-level table. Useful columns include:

- `run_dir`
- `has_run_config`
- `has_run_summary`
- `status`
- `run_name`
- `model`
- `task`
- `dataset`
- `dataset_mode`
- `precision`
- `quantizer`
- `mode`
- `h`
- `seed`
- `data_seed`
- `steps`
- `steps_completed`
- `best_eval_acc`
- `last_eval_acc`

Typical usage:

```python
import pandas as pd
runs = pd.read_csv("indices/run_index.csv")
sst5_int4 = runs[
    runs.astype(str).apply(lambda col: col.str.contains("sst", case=False, na=False)).any(axis=1)
    & runs.astype(str).apply(lambda col: col.str.contains("int4", case=False, na=False)).any(axis=1)
]
```

### `indices/experiment_family_index.csv`

Experiment-root-level counts and inferred labels. Use it to locate broad
families before drilling down into individual run dirs.

### `indices/artifact_manifest.csv`

File-level provenance:

- original `source_path`
- copied `package_path`
- file size
- SHA256
- artifact kind
- inclusion status

Use this whenever a figure, table, or CSV needs a provenance trail.

### `indices/excluded_files.csv`

Shows omitted files and reasons. Use this before concluding something was lost.
Common omissions: checkpoints, model weights, tokenizer files in the records-only
archive, nested archives, and very large binaries.

## Metric Vocabulary

Important canonical definitions used in the project:

- `h`: perturbation radius for two-point finite differences.
- `h_ref`: frozen theoretical reference radius.
- `hstar`, `h_star`, `hstar_cleanGL`, `hstar_lowbitL`: historical analytical
  radius variants. Verify exact formula/version before comparing.
- `d_Q`: quantized two-point finite difference.
- `d_star`: true directional derivative, usually `<grad, u>`.
- `fd_true_nmse` or true directional nMSE: normalized
  `E[(d_Q - d_star)^2] / E[d_star^2]`, only if code provenance confirms this.
- `corr`: directional correlation; check whether it is loss-level correlation
  or geometry-only.
- `A_cross`, `A_interval`, `A_uniform`: perturbation geometry or interval error,
  not true directional MSE.
- `active_frac`: fraction of coordinates whose quantized plus/minus probes
  differ.
- `alignment`: cosine between effective quantized displacement and intended
  displacement.
- `norm_ratio`: norm of effective displacement divided by norm of intended
  displacement.
- `cleanGL`: h-star estimated from cleaner/high-precision G/L quantities.
- `lowbitL` or low-bit GL: h-star estimated using low-bit-aware quantities.

## Known Caveats

1. Policy names are historical and not always consistent. Always verify the
   actual numeric `h`.
2. Do not merge dense, sparse, prefix, residual-grid, and QZO/GPTQ/AWQ rows.
3. Do not merge different tasks, seeds, dataset modes, or trainable-parameter
   scopes.
4. Some early sparse runs had mask or seed-alignment problems. Prefer later
   seed-fixed outputs when making claims.
5. Some older plots used geometry/proxy diagnostics that should not be called
   true directional MSE.
6. INT4 residual-grid implementation smoke passed, but training did not show
   stable success.
7. OPT results are transfer sanity checks and may not match original MeZO task
   coverage.
8. Single-seed results should be labeled as such.
9. Full, medium, and pilot runs should not be averaged together without clear
   labels.

## Best Practices for Answering Questions from This Package

When asked for a result:

1. Find candidate runs in `indices/run_index.csv`.
2. Open the corresponding `run_config.json` and `run_summary.json`.
3. If the question involves curves over h, find the relevant summary CSV rather
   than only one run summary.
4. Verify metric definitions using local report files or code/scripts in
   `artifacts/tools/`.
5. Use `artifact_manifest.csv` to cite source paths.
6. Report run type (`full`, `medium`, `pilot`, `smoke`) and seed.
7. Mention missing checkpoints or excluded files when relevant.

When asked to compare methods:

1. Ensure same model, task, dataset mode, seed/data seed, precision, quantizer,
   perturbation mode, and trainable scope.
2. Compare `best_eval_acc` with `best_eval_acc`, or `last_eval_acc` with
   `last_eval_acc`; do not mix best and last.
3. Prefer full runs over pilot/medium runs.
4. Keep failed and unfavorable rows visible when summarizing.

## Useful Search Examples

Find RoBERTa SST-5 INT4 residual-grid records:

```bash
rg -n "int4_residual_grid|residual_grid_error_feedback|residual-grid" .
```

Find true directional-MSE related files:

```bash
rg -n "fd_true_nmse|true_directional|d_star|d_Q|nMSE" .
```

Find OPT INT4 dense results:

```bash
rg -n "opt13b|opt-1.3b|facebook/opt-1.3b" artifacts/outputs indices
```

Find final paper manifests:

```bash
find artifacts/paper_artifacts_final -maxdepth 3 -type f | sort
```

Find all run summaries for a family:

```bash
find artifacts/outputs/int4_full_data_hstar_dense_sparse_20260522_113710 -name run_summary.json
```

## Minimal Citation Template

When citing a result from this package, include:

```text
model=<model>, task=<task>, dataset_mode=<mode>, precision=<precision>,
quantizer=<quantizer>, perturbation_mode=<mode>, h=<value>, seed=<seed>,
run_type=<full|medium|pilot|smoke>, metric=<metric_name>,
source=<package_path or source_path from artifact_manifest.csv>
```

## Extraction and Integrity

Records-only archive file:

```text
all_experiment_results_records_only_20260626.tar.zst
```

Verify:

```bash
sha256sum -c all_experiment_results_records_only_20260626.tar.zst.sha256
```

Extract:

```bash
tar --use-compress-program=zstd -xf all_experiment_results_records_only_20260626.tar.zst
```

Split-file reassembly:

```bash
cd all_experiment_results_records_only_20260626_parts
sha256sum -c parts.sha256
cat all_experiment_results_records_only_20260626.tar.zst.part-* > ../all_experiment_results_records_only_20260626.tar.zst
cd ..
sha256sum -c all_experiment_results_records_only_20260626.tar.zst.sha256
```

