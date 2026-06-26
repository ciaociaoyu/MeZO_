# Final Experiment Summary

## Supported claims

- The frozen analytical envelope produces the expected scaling trends in the controlled one-sided quadratic experiment.
- Existing RoBERTa precision-window data can be separated into theoretical frozen-formula windows and empirical validation intervals.
- Existing RoBERTa INT4 multi-task rows support the conservative claim that default `h=1e-3` is competitive in broad windows and reference radii can help in narrower/extreme low-precision settings.
- Existing OPT rows provide cross-architecture sanity checks only; they are not SOTA or direct original-MeZO reproduction claims.

## Unsupported or intentionally avoided claims

- No claim that an interval-aware selector replaces the frozen theory.
- No claim that empirical MSE/accuracy sweeps define the theoretical window.
- No claim that reference radii beat default on every task.
- No paper source update was made because `main.tex` was missing locally.

## Commands

```bash
python tools/final_frozen_window_package.py --output_dir hwindow_final_experiments_bundle
```
