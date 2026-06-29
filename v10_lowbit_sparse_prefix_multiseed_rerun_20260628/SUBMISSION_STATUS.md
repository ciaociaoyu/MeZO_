# Submission Status

Submitted: 2026-06-28

Slurm job:

- `46474441`
- partition: `gpu_p`
- job name: `v10-lb-rerun`
- array: `0-5%6`
- GPU request: `gpu:H100:1` per lane

Initial queue check:

- status: `PENDING`
- reason: `Resources`

This indicates the rerun was accepted by Slurm but had not yet been scheduled
onto H100 nodes at the first check. The previous data-split failure has been
addressed before submission.

Monitor with:

```bash
squeue -j 46474441 -o '%i %P %j %T %M %D %R'
tail -f v10_lowbit_sparse_prefix_multiseed_rerun_20260628/jobs/v10-lb-rerun_46474441_0.err
```

