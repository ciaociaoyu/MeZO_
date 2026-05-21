# Main Mechanism Classical h Baselines

This batch adds two h-only baselines for the Main mechanism experiments on RoBERTa-large / SST-5 under FP32, FP16, and INT8.

## Baselines

`fd_eps13` uses the classical central finite-difference rule `h = eps_machine ** (1/3)`.

- FP32: `np.finfo(np.float32).eps ** (1/3) ~= 4.92e-3`.
- FP16: raw `np.finfo(np.float16).eps ** (1/3) ~= 9.92e-2`, capped to `1e-2`.
- INT8: no principled machine-epsilon analogue, so this is reported as a capped stress baseline with `h=1e-2` and `fd_principled=false`.

`spall_ck` uses only Spall's SPSA perturbation-gain schedule:

```text
h_t = h0 / (t + 1)^gamma
h0 = 1e-3
gamma = 0.101
```

This is not full SPSA. The estimator, dense random direction, optimizer, learning rate, query count, data order, and training loop remain the same as the Main mechanism MeZO setup.

## Defaults

- MeZO default `h=1e-3` is not rerun in this batch because it already exists.
- Continuous h is the default; grid snapping is disabled unless `--h_schedule_grid_policy nearest|floor|ceil` is explicitly requested.
- Safety interval: `[1e-5, 1e-2]`.

## INT8 Setting

The long-run scripts request the existing low-bit Main mechanism path with 8-bit quantization, group size 128, and FP16-master update backend. They explicitly avoid GPTQ, residual-grid, and direct INT update.

## Resource Rule

Long runs are submitted as a 6-task Slurm array on L4 GPUs with at most 3 active one-GPU tasks:

```text
GPU_GRES=${GPU_GRES:-gpu:L4:1}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-3}
```

If the cluster does not support typed GRES, use:

```text
GPU_GRES=gpu:1 SBATCH_EXTRA="--constraint=L4"
```
