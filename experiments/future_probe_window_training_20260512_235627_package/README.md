# Future Probe Window Training Package

Experiment root:

`runs/future_probe_window_training_20260512_235627`

This package keeps the lightweight, reproducible artifacts from the run and
omits duplicated tokenizer files, binary argument dumps, full local tmux logs,
and model checkpoints.

## What Completed

Dense INT8 + FP16 master training completed for seed 0:

| h | best_acc | final_acc | final_eval_loss |
| --- | ---: | ---: | ---: |
| 3e-4 | 0.285714 | 0.285714 | 1.572188 |
| 1e-3 | 0.459016 | 0.459016 | 1.440830 |
| 2e-3 | 0.447307 | 0.447307 | 1.395202 |
| 3e-3 | 0.434426 | 0.434426 | 1.432158 |
| 5e-3 | 0.343091 | 0.343091 | 1.544407 |
| 1e-2 | 0.271663 | 0.271663 | 1.624723 |

The sparse screening did not complete. The only sparse run that started was:

`sparse_int8_p0p003_ha0p003_lr3em6_seed0`

It reached `global_step=166/500` and has no `run_summary.json`, so it is
recorded under `sparse_partial/` and excluded from the completed summary table.
No Python traceback was found in the captured logs.

## Interpretation

The dense 2k-step seed-0 result supports a usable INT8 training window. The
best final accuracy in this run is at `h=1e-3`, with `h=2e-3` close behind and
the lowest eval loss. `h=3e-3`, which was best by earlier dense INT8 probe
correlation, remains competitive but is not the best training point here.
Both too-small (`3e-4`) and too-large (`1e-2`) settings are clearly worse.

For follow-up full runs, prioritize:

- dense INT8 + FP16 master, `h in {1e-3, 2e-3, 3e-3}`, seeds 0/1/2;
- sparse screening needs to be relaunched separately before promotion.

## Included Files

- `summary*.csv` and `summary.md`: generated aggregate summaries.
- `plots/*.png`: generated plots.
- `dense_runs/*/run_summary.json`: per-run final summaries.
- `dense_runs/*/run_metadata.json`: per-run command/config metadata.
- `dense_runs/*/metrics_logs/*.csv`: eval/train curves.
- `dense_runs/*/update_stats.jsonl`: per-step update diagnostics.
- `dense_runs/*/checkpoint_probe_stats.jsonl`: checkpoint probe diagnostics for
  selected dense runs.
- `sparse_partial/*/update_stats.jsonl`: partial sparse diagnostics.
- `commands/launch_local_tmux.sh`: local launch command wrapper used for this
  run.
