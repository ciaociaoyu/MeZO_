# 14-h Sweep Experiments

This suite runs fixed-`h` sweeps for four model/task pairs:

- `roberta-large` + `SST-5`
- `roberta-large` + `MNLI`
- `opt-1.3b` + `SST-5`
- `opt-1.3b` + `MNLI`

Each job script iterates over the same 14 `h` values:

`1e-8`, `3e-8`, `1e-7`, `3e-7`, `1e-6`, `3e-6`, `1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`, `3e-2`

## Layout

- Job scripts: `/Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/jobs`
- Slurm/stdout logs: `/Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/logs`
- Run outputs: `/Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/results`

Run outputs are organized by model, task, `h`, and seed. Example:

`/Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/results/opt-1.3b/mnli/h_1e-5/seed_42/`

## Recorded Metrics

Each run writes machine-readable metrics into its run directory. The key files are:

- `metrics_*.jsonl` and `metrics_*.csv`: train loss over time, eval loss, eval metric, train-probe metric
- `final_metrics.json`: final validation metrics for the run
- `run_summary.json`: run metadata, final metrics, best metric/checkpoint, and artifact paths
- `zo_directional_probe.csv`: directional diagnostics including `fd_mean`, `td_mean`, `mae`, `mse`, `rmse`, `sign_acc`, and `corr`

For the medium-model path, the existing `run_summary.json` and `zo_directional_probe.csv` are reused.

## Running

Submit the four jobs with:

```bash
sbatch /Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/jobs/roberta_sst5_14h.sh
sbatch /Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/jobs/roberta_mnli_14h.sh
sbatch /Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/jobs/opt13b_sst5_14h.sh
sbatch /Users/jichaoyu/Documents/GitHub/MeZO/experiments/h_sweep_14h/jobs/opt13b_mnli_14h.sh
```
