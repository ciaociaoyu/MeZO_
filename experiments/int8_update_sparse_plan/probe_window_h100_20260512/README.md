# Probe Window H100 Results 2026-05-12

This directory packages the H100 probe-window diagnostics and small 300-step validation runs for the INT8 finite-difference radius experiment.

## Contents

- `probe_window_h100_20260512_results.tar.gz`: compressed result bundle.
- `probe_window_h100_20260512_results.sha256`: checksum for the bundle.
- `dense_probe_summary.csv` / `.md`: dense h-window summary.
- `sparse_probe_summary.csv` / `.md`: sparse h/sqrt(p) probe summary.
- `window_training_summary.csv` / `.md`: 300-step validation summary.

The tarball includes:

- summary files, plots, manifests, and commands used;
- dense/sparse probe JSONL stats and probe metadata;
- training `run_summary.json`, eval files, metrics CSVs, and logs;
- probe/summarization scripts used for this round.

The package intentionally excludes repeated tokenizer/vocab artifacts, model checkpoints, and old `.safetensors` files.

## Source Run Directories

- Dense probe: `runs/probe_window_dense_20260512_193200/`
- Sparse probe: `runs/probe_window_sparse_20260512_194300/`
- Training validation: `runs/window_training_small_20260512_201024/`

## Headline Results

- Dense INT8 probe: best finite-difference/true-gradient correlation at `h=3e-3`, with `corr_fd_true ~= 0.938`.
- Dense BF16 probe: best at `h=1e-3`, with `corr_fd_true ~= 0.998`.
- Dense FP32 probe: stable at very small `h`; `h=1e-5` has `corr_fd_true ~= 1.0`.
- Sparse INT8 probe: curves align better by `h_active = h / sqrt(p)` than raw `h`; best sparse settings are around `h_active=0.006` to `0.012`.
- Small 300-step validation completed 11 runs with no NaNs.

## Checksum

```text
a0f9ab7228dc7285d13d994e20b94d9855fdef5fc3cc6e9b3907f5c3737be08a  probe_window_h100_20260512_results.tar.gz
```
