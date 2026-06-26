# Reproduction Commands

Run from the repository root. The original v2 package was generated in the `ciao` conda environment.

```bash
python tools/final_frozen_window_package.py --output_dir hwindow_final_experiments_bundle_v2
```

This regenerates all CSVs, figures, LaTeX snippets/tables, metadata, and the zip archive from existing logs and bundles.
No new training jobs or new theory/model fitting are launched by this script.
