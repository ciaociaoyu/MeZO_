# Commands

Environment check:

```bash
conda run -n ciao python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Final package generation:

```bash
conda run -n ciao python tools/final_frozen_window_package.py --output_dir hwindow_final_experiments_bundle
```

Notes:

- `ciao` was selected because it has a CUDA-enabled PyTorch build and working `numpy`, `pandas`, and `matplotlib`.
- `main.tex` was not present in this checkout or inspected top-level zip files, so the paper-source update and LaTeX compile stage is recorded as blocked in `final_missing_items.md` and `experiment_conflicts.md`.
- No training jobs or new h-search jobs were launched by this package command.
