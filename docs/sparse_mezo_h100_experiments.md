# Sparse MeZO H100 实验记录

更新时间：`2026-04-14`

## 1. 本轮代码改动对应的方法

- 新方法名：`sparse_mezo`
- 当前实验路径：`medium_models` 的 `roberta-large`，任务为 `MNLI` 和 `SST-5`
- 当前 16-bit 约定保持仓库现有语义：
  - `--zo_two_point_precision fp16`
  - `--zo_quantization_bits 16`
- 本轮固定的 Sparse MeZO 配置：
  - `--sparse_ratio 0.25`
  - `--sparse_mask_strategy percentile_per_layer`
  - `--sparse_scope trainable_only`
  - `--sparse_log_active_fraction True`

## 2. 已完成的 smoke test

### 2.1 `roberta-large + MNLI + sparse_mezo + 16-bit`

- 输出目录：
  - `/scratch/jy03364/MeZO_/experiments/sparse_mezo_smoke/medium/roberta_mnli/run_sparse_mezo16/seed16`
- 关键产物：
  - `run_summary.json`
  - `run_metadata.json`
  - `zo_directional_probe.csv`
- 结果摘要：
  - `best_dev_objective = 0.3596129360835243`
  - dev: `eval_loss = 1.1796875`, `eval_mnli/acc = 0.3596129360835243`
  - test: `eval_loss = 1.1787109375`, `eval_mnli/acc = 0.35537442689760573`
  - 实际 active fraction：`0.25023450732081826`

### 2.2 `roberta-large + SST-5 + sparse_mezo + 16-bit`

- 输出目录：
  - `/scratch/jy03364/MeZO_/experiments/sparse_mezo_smoke/medium/roberta_sst5/run_sparse_mezo16/seed16`
- 关键产物：
  - `run_summary.json`
  - `run_metadata.json`
  - `zo_directional_probe.csv`
- 结果摘要：
  - `best_dev_objective = 0.2775175644028103`
  - dev: `eval_loss = 1.6328125`, `eval_acc = 0.2775175644028103`
  - test: `eval_loss = 1.6640625`, `eval_acc = 0.23619909502262443`
  - 实际 active fraction：`0.25009136066516674`

## 3. 已提交的正式 h-search

### 3.1 `roberta-large + MNLI + sparse_mezo + 16-bit`

- Slurm job id：`44285153`
- 脚本：
  - `experiments/h_sweep_14h/jobs/roberta_mnli_sparse_mezo16_14h.sh`
- 结果目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/mnli`
- 日志目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/roberta-large/mnli`

### 3.2 `roberta-large + SST-5 + sparse_mezo + 16-bit`

- Slurm job id：`44285154`
- 脚本：
  - `experiments/h_sweep_14h/jobs/roberta_sst5_sparse_mezo16_14h.sh`
- 结果目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/sparse_mezo16/roberta-large/sst5`
- 日志目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/sparse_mezo16/roberta-large/sst5`

### 3.3 提交器

- 提交脚本：
  - `experiments/h_sweep_14h/submit_sparse_mezo16_searches.sh`
- 本次重新提交日志：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/submit_sparse_mezo16_20260414_035459.log`

## 4. 其他已完成实验

### 4.1 `opt-1.3b + MNLI + quzo16`

- Slurm job id：`44197021`
- 状态：`COMPLETED`
- 结果目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli`
- 日志目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli`
- `summary.jsonl` 汇总：
  - `14` 个 `h` 全部结束
  - `3 completed`
  - `11 failed`
- `completed` 的 `h`：
  - `1e-8`
  - `3e-8`
  - `1e-7`

## 5. 当前队列快照

以下状态是写本文档时的快照：

- `44285153` `hsweep14h_sparsemezo16_roberta_mnli`
  - 状态：`RUNNING`
- `44285154` `hsweep14h_sparsemezo16_roberta_sst5`
  - 状态：`RUNNING`
- `44197021` `hsweep14h_quzo16_opt13b_mnli`
  - 状态：`COMPLETED`

## 6. 当前使用的 14 个 h 值

来自 `experiments/h_sweep_14h/h_values.sh`：

- `1e-8`
- `3e-8`
- `1e-7`
- `3e-7`
- `1e-6`
- `3e-6`
- `1e-5`
- `3e-5`
- `1e-4`
- `3e-4`
- `1e-3`
- `3e-3`
- `1e-2`
- `3e-2`
