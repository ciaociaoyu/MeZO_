# INT4 Window Preflight

This preflight runs RoBERTa-large / SST-5 K=16 few-shot INT4 G128 RTNClip probes and a single dense 5k training smoke.

## Environment

Preferred environment on this machine:

```bash
conda activate ciao
```

If `ciao` is unavailable, the scripts fall back to the current Python. The actual run environment is logged in `outputs/int4_window_preflight/*/env.json` and `outputs/int4_window_preflight/train_sst5_fisher_dense_int4_h1e-3/slurm_environment.txt`.

## Local H100 Probe

Dense/default-path preflight:

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_ENV_NAME=ciao scripts/probe_int4_local.sh
```

All SST-5 settings:

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_ENV_NAME=ciao scripts/probe_sst5_all_settings.sh
```

Outputs:

- `outputs/int4_window_preflight/probe_sst5_fisher_dense/`
- `outputs/int4_window_preflight/probes_sst5_all_settings/`

Each setting writes `probe_results.csv`, `probe_records.csv/jsonl`, and `hstar_summary.csv/json`.

## One-H100 5k Smoke

Submit the dense INT4 `h=1e-3` 5k training smoke:

```bash
ACCOUNT=${ACCOUNT:-} PARTITION=${PARTITION:-} TIME=${TIME:-04:00:00} GPUS=${GPUS:-1} GPU_TYPE=${GPU_TYPE:-h100} \
  CONDA_ENV_NAME=ciao bash slurm/train_sst5_fisher_dense_int4_h100.sbatch
```

The script does not hardcode account, partition, or absolute paths. It writes to:

`outputs/int4_window_preflight/train_sst5_fisher_dense_int4_h1e-3/`

## Summarize

```bash
python scripts/summarize_int4_window.py --output_root outputs/int4_window_preflight
```

Summary outputs:

- `outputs/int4_window_preflight/summary_probe_table.csv`
- `outputs/int4_window_preflight/summary_int4_window_preflight.md`
- `outputs/int4_window_preflight/summary_int4_window_preflight.json`

## Window Membership

For each setting, `hstar_summary.json` contains:

- `window_low`
- `window_high`
- `h_star`
- `membership_1e-5`
- `membership_1e-3`
- `membership_hstar`

Membership labels:

- `L`: h is left of the MSE window.
- `✓`: h is inside the MSE window.
- `R`: h is right of the MSE window.

## Generated-Only Scripts

The Slurm script submits only the single 5k smoke. It does not launch the later full matrix. The generated local probe scripts are reusable entry points and do not submit scheduler jobs.
