# Next Window Sparse + Residual Results, 2026-05-15

This package contains compact summaries for the 2026-05-15 next-stage INT8 MeZO experiments. It intentionally excludes full training logs and checkpoints.

## Source Run Directories

- Cluster dense A100 runs: `runs/next_window_sparse_20260515_085924_chunk8_a100`
- Local residual H100 runs: `runs/next_residual_local_20260515_082009`

## Cluster Dense INT8 + FP16 Master

- Job ID: `45279501`
- GPU request: `gpu:A100:1`
- Array: `0-7%2`
- Runs completed: 8 / 8
- Training length: 2000 optimizer steps per run
- Update backend: `fp16_master`
- Precision / quantization: `int8`
- Learning rate: `1e-5`

| h | seeds | mean best acc | mean last acc | mean last loss |
| --- | ---: | ---: | ---: | ---: |
| 1e-3 | 3 | 0.3724 | 0.3603 | 1.4945 |
| 2e-3 | 3 | 0.3942 | 0.3927 | 1.4604 |
| 3e-3 | 2 | 0.3829 | 0.3829 | 1.4512 |

Interpretation: among the completed dense runs, `h=2e-3` is the strongest and most stable by both mean best and mean last accuracy. `h=3e-3` remains incomplete for the 3-seed comparison because seed 2 was not part of this 8-job chunk.

## Local Residual Grid Stage A

- GPU: local H100
- Runs completed: 3 / 3
- Training length: 500 optimizer steps per run
- Precision / quantization: `int8`
- Perturbation radius: `h=3e-3`

| run | best acc | last acc | last loss | cos last | norm ratio last | violation frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `round_lr3e-5_step500` | 0.3091 | 0.3091 | 1.5544 | 0.0727 | 13.7647 | 0.0 |
| `round_step1_lr1e-4_clip5_step500` | 0.3689 | 0.3326 | 1.5252 | 0.1309 | 7.6364 | 0.0 |
| `stoch_step1_lr3e-4_clip10_step500` | 0.2728 | 0.2717 | 1.6132 | 0.1653 | 5.8929 | 0.0 |

Interpretation: residual-grid mechanics stayed clean: residual-bound violations are zero, grid error is zero, and scale drift is zero. The best residual Stage-A accuracy was the clipped round setting, but dense INT8 + fp16-master remains the main training line.

## Included Files

- `cluster_dense_a100/summary.csv`
- `cluster_dense_a100/summary.md`
- `cluster_dense_a100/summary_dense.csv`
- `cluster_dense_a100/summary_dense_by_h.csv`
- `cluster_dense_a100/commands.txt`
- `cluster_dense_a100/job_ids.txt`
- `cluster_dense_a100/config_manifest.json`
- `cluster_dense_a100/next_window_sparse_all.sbatch`
- `residual_local_h100/summary.csv`
- `residual_local_h100/summary.md`
- `residual_local_h100/summary_residual.csv`
- `residual_local_h100/commands.txt`
- `residual_local_h100/job_ids.txt`
- `residual_local_h100/config_manifest.json`
- `scripts/summarize_next_experiments.py`
- `scripts/submit_next_window_sparse.sh`
- `scripts/run_next_residual_local.sh`

## Recommendation

Promote dense `h=2e-3` first for longer 5k-20k validation. Complete `h=3e-3` seed 2 before making a final dense three-seed comparison. Keep residual-grid as a secondary update-commit diagnostic rather than the main paper direction.
