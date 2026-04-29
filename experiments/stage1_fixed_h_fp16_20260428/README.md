# Stage 1 Fixed-h FP16 Experiments

## Goal

Run the fixed-h stage of the current pilot plan. This stage only launches
`h = 1e-3` and `h = 1e-5`. The proposed `h*` setting is not launched here; the
`h = 1e-3` runs are configured to emit early-step probe / h-estimation logs that
can be used to derive `h*` later.

## Experiment Contract

- Precision: FP16.
- Models: RoBERTa-large and OPT-1.3B.
- Datasets: full SST-2 and full SST-5.
- Methods: MeZO, Sparse MeZO, MeZO-LoRA.
- Fixed h values for this stage: `1e-3`, `1e-5`.
- Max steps: 5000.
- Dataset mode: full dataset with deterministic split seed.
- Data shuffle: enabled where the training path exposes the option.
- Guard policy: nan guard only. No random-prediction guard, probe guard, or early
  stop guard beyond nan protection.
- Seed policy: `seed=16`, `data_seed=16`, `train_set_seed=16`.
- Save policy: save final model parameters/checkpoint for every completed run.
- GPU target: L4, one GPU per job.

## Matrix

The fixed-h stage contains 24 runs:

```text
2 models x 2 datasets x 3 methods x 2 h values = 24 runs
```

The full launch manifest is in `jobs/stage1_manifest.tsv`.

## Method Settings

- MeZO: `zo_method=mezo`.
- Sparse MeZO: `zo_method=sparse_mezo`, `sparse_ratio=0.25`,
  `sparse_mask_strategy=percentile_per_layer`, `sparse_scope=trainable_only`.
- MeZO-LoRA: MeZO with LoRA enabled, `lora_r=8`, `lora_alpha=16`.

## Shared Training Settings

- Learning rate: `1e-6`, inherited from the current pilot/sweep settings.
- Weight decay: `0`.
- RoBERTa batch size: `64`.
- OPT batch size: `16`.
- Eval/logging cadence: eval every 1000 steps, logging every 10 steps.
- Probe cadence: every 200 steps with 16 probe seeds where supported.

## h* Estimation Notes

RoBERTa/medium runs enable the two-point h-estimation logger while keeping the
active training h fixed. This produces CSV logs for deriving `h*` from early
steps without changing the training perturbation.

Implementation notes after the OPT/LoRA fix:

- `zo_quantization_bits=16` is treated as FP16 model storage plus FP16 ZO
  perturb/probe convention, not as low-bit QuZO.
- RoBERTa/medium FP16 runs convert model parameters and buffers to FP16 storage.
- Fixed-h FP16 runs quantize the actual ZO perturbation to FP16.
- For `h*` estimation, `d_dim` is the effective trainable perturbation dimension.
  Dense MeZO uses all trainable parameters, LoRA uses the `requires_grad`
  parameters after freezing, and Sparse MeZO uses the active sparse mask count.
- `h_estimation.csv` logs `d_dim`, `d_source`, `trainable_params`, and
  `sparse_active_params` for checking the formula inputs.

OPT/large runs do not expose the same medium-model two-point h-estimation
interface in the current code path. They still emit ZO probe CSV logs where
supported, which can be used for post-hoc `h*` estimation.

## Environment

- RoBERTa/medium path uses conda env `ciao`.
- OPT/large path uses conda env `mezo-mistral`.
- The local `mezo-env` environment is not used because it currently fails on
  import with a broken `transformers` installation.

## Output Layout

```text
experiments/stage1_fixed_h_fp16_20260428/
  jobs/
    stage1_manifest.tsv
    run_stage1_case.sh
    stage1_array_l4.sh
    submit_stage1_l4.sh
    status_stage1.sh
  logs/
    <array-index>_<model>_<dataset>_<method>_h<h>/
  results/
    roberta-large/<dataset>/<method>/h_<h>/run/seed16/
    opt-1.3b/<dataset>/<method>/h_<h>/seed_16/
    status/
```

## Launch

Submit the full fixed-h stage to L4:

```bash
bash experiments/stage1_fixed_h_fp16_20260428/jobs/submit_stage1_l4.sh
```

Run one manifest row locally on the current node for command checking:

```bash
DRY_RUN=1 bash experiments/stage1_fixed_h_fp16_20260428/jobs/run_stage1_case.sh 0
```

Check queued/running jobs and completed summaries:

```bash
bash experiments/stage1_fixed_h_fp16_20260428/jobs/status_stage1.sh
```
