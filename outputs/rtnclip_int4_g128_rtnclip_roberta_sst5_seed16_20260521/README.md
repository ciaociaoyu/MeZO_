# INT4 G128 RTNClip RoBERTa-large SST-5 h-search

Official INT4 continuation after preflight and mini smoke. L4 lanes, max 6 active scheduler tasks.

Canonical low-bit nMSE field: `lowbit_true_nmse = delta_visibility_nmse`, metric version `dequantized_effective_displacement_nmse_v1`.

Git-tracked artifact set intentionally excludes `checkpoints/` and `state.pt` files; full local checkpoints remain under this output directory on the training host.
