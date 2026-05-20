# Missing cheap probe commands

The current G128 RTNClip INT8/INT4 artifacts do not contain full true-gradient
`nMSE_fd_true` curves. To make the MSE-envelope estimator paper-ready for the
current low-bit oracle, add or run probe-only commands that compute fixed-batch,
fixed-direction `d_fd` and `d_true` on the main h-grid.

```bash
# Current dense INT8 G128 RTNClip, probe only, no training.
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True \
python tools/rtnclip_roberta_sst5_batch.py \
  --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch \
  --bitwidth 8 --probe_dirs 32 probe-int4

# Current dense INT4 G128 RTNClip near-window/full-grid probe, no training.
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True \
python tools/rtnclip_roberta_sst5_batch.py \
  --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch \
  --bitwidth 4 --probe_dirs 32 probe-int4
```

The existing `probe-int4` path currently writes finite differences and geometry.
For final MSE-bound fitting it still needs true-gradient directional derivatives
or a companion true-gradient probe on the same batch/directions.

Sparse G128 RTNClip p in `{0.01, 0.003}` also needs a probe-only harness that
logs both raw `h` and `h_active = h / sqrt(p)`.
