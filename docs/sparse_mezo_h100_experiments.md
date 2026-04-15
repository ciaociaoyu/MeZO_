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

- 旧 job：`44197021`
- 问题：
  - 在后半段 sweep 中触发 `large_models/run.py` 的代码回归，`field` 未从 `dataclasses` 导入，导致 `OurArguments` 在解释阶段直接报 `NameError`
- 修复：
  - 已补回 `large_models/run.py` 中的 `field` 导入
  - 已在 H100 上用 `h=3e-7` 做完整 smoke 验证，确认可成功运行并写出 `run_summary.json` / `run_metadata.json`
- smoke 输出目录：
  - `/scratch/jy03364/MeZO_/experiments/smoke_fix/large/opt13b_mnli_quzo16_h3e-7_numeval32`
- 重新提交后的新 job：
  - `44302039`
- 当前状态：
  - `RUNNING`
- 当前正式结果目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/results/quzo16/opt-1.3b/mnli`
- 当前正式日志目录：
  - `/scratch/jy03364/MeZO_/experiments/h_sweep_14h/logs/quzo16/opt-1.3b/mnli`

## 5. 当前队列快照

以下状态是写本文档时的快照：

- `44285153` `hsweep14h_sparsemezo16_roberta_mnli`
  - 状态：`RUNNING`
- `44285154` `hsweep14h_sparsemezo16_roberta_sst5`
  - 状态：`RUNNING`
- `44302039` `hsweep14h_quzo16_opt13b_mnli`
  - 状态：`RUNNING`

## 6. H100 上 `opt-1.3b` 速度基准

目标：在当前 H100 环境下，测 `opt-1.3b` 在 `MNLI / SST-5 / BoolQ / SQuAD` 上的短程训练速度，对比 `fp16` 和 `int8` 两种当前仓库已有的 large-model 精度路径。

### 6.1 测试参数确认

8 组速度测试统一使用：

- `model_name = facebook/opt-1.3b`
- `trainer = zo`
- `dataset_mode = full`
- `num_k = 16`
- `data_seed = 42`
- `train_set_seed = 42`
- `learning_rate = 1e-6`
- `zo_eps = 1e-4`
- `num_train_epochs = 1`
- `max_steps = 20`
- `per_device_train_batch_size = 16`
- `gradient_accumulation_steps = 1`
- `lr_scheduler_type = constant`
- `save_strategy = no`
- `no_eval = True`
- `logging_steps = 1`
- `zo_probe_every = 0`

结论：

- 梯度累计已关闭：`gradient_accumulation_steps = 1`
- batch size 不小：`per_device_train_batch_size = 16`

本次“精度”定义按当前仓库 large-model 语义执行：

- `fp16`：
  - `--load_float16`
  - `--zo_quantization_bits 16`
- `int8`：
  - `--load_int8`
  - `--zo_quantization_bits 32`

### 6.2 结果目录

- 总目录：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/opt13b_h100_20260414_195901`
- 汇总文件：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/opt13b_h100_20260414_195901/summary.jsonl`

每个任务/精度组合都包含：

- `command.sh`
- `combined.log`
- `time.txt`
- `run/`

### 6.3 吞吐结果

以下以 `train_steps_per_second` 作为主要训练速度指标；`wall_seconds` 包含模型加载和短程启动开销。

| task | mode | train_steps_per_second | train_runtime (20 step) | wall_seconds |
|---|---|---:|---:|---:|
| MNLI | fp16 | 4.584 | 4.3626 | 30.41 |
| SST-5 | fp16 | 6.516 | 3.0694 | 19.10 |
| BoolQ | fp16 | 3.774 | 5.2991 | 27.64 |
| SQuAD | fp16 | 4.643 | 4.3080 | 25.78 |
| MNLI | int8 | 1.315 | 15.2137 | 38.42 |
| SST-5 | int8 | 1.563 | 12.7955 | 25.68 |
| BoolQ | int8 | 0.820 | 24.3757 | 38.32 |
| SQuAD | int8 | 1.196 | 16.7278 | 33.35 |

### 6.4 折算到 10,000 step 的训练时间

下面的估算基于 `train_steps_per_second`，只反映训练 step 时间，不把短程测试里模型加载的固定开销线性放大。

| task | mode | estimated seconds for 10k steps | estimated time for 10k steps |
|---|---|---:|---|
| MNLI | fp16 | 2181.5 | 0h36m21.5s |
| SST-5 | fp16 | 1534.7 | 0h25m34.7s |
| BoolQ | fp16 | 2649.7 | 0h44m09.7s |
| SQuAD | fp16 | 2153.8 | 0h35m53.8s |
| MNLI | int8 | 7604.6 | 2h06m44.6s |
| SST-5 | int8 | 6398.0 | 1h46m38.0s |
| BoolQ | int8 | 12195.1 | 3h23m15.1s |
| SQuAD | int8 | 8361.2 | 2h19m21.2s |

### 6.5 简要结论

- 在当前 `opt-1.3b` large-model 训练路径上，`int8` 明显慢于 `fp16`
- 相对 `fp16`，`int8` 大约慢：
  - `MNLI`: `3.49x`
  - `SST-5`: `4.17x`
  - `BoolQ`: `4.60x`
  - `SQuAD`: `3.88x`
- 因此，如果目标是当前仓库上的训练吞吐，`fp16` 仍然是更优选择

## 7. H100 上 `roberta-large` 速度基准

目标：补齐当前正式 `medium_models` 路径上 `roberta-large` 的短程训练吞吐对比。这里不重跑已经测过的 `opt-1.3b`，只新增 `MNLI / SST-5` 的 `16-bit` 与 `int8`。

### 7.1 路径与范围说明

- 当前正式 `roberta-large` 训练路径是 `medium_models`
- 该路径当前用于正式实验的任务是：
  - `MNLI`
  - `SST-5`
- `BoolQ / SQuAD` 不属于当前 `roberta / medium_models` 这条正式训练路径，因此本轮没有为 `roberta-large` 额外发明一条新训练路径去测它们

### 7.2 测试参数确认

4 组 `roberta-large` 速度测试统一使用：

- `model = roberta-large`
- `trainer = standard + zero_order_optim`
- `dataset_mode = full`
- `num_k = 16`
- `seed = 16`
- `data_seed = 16`
- `learning_rate = 1e-6`
- `zero_order_eps = 1e-4`
- `max_steps = 20`
- `per_device_train_batch_size = 32`
- `gradient_accumulation_steps = 1`
- `dataloader_shuffle = False`
- `use_adaptive_h = False`
- `use_c_scale = False`
- `zo_probe_every = 0`

本次“精度”定义按当前仓库 `medium_models` 语义执行：

- `fp16`：
  - `--zo_two_point_precision fp16`
  - `--zo_quantization_bits 16`
- `int8`：
  - `--zo_two_point_precision fp16`
  - `--zo_quantization int8`

### 7.3 结果目录

- 早期目录：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/roberta_h100_20260414`
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/roberta_h100_20260414_mini`
- 当前以 `BS=32` 重测结果为准：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/roberta_h100_bs32_rerun_20260414_212842`
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/roberta_h100_bs32_rerun_20260414_212842/summary.jsonl`

### 7.4 吞吐结果

`medium_models` 的 `run_summary.json` 不直接写 `train_steps_per_second`，这里使用训练日志中 `global_step=20` 时的 `time=` 近似换算训练吞吐。下面以这次 `BS=32` 重测结果为准。

| task | mode | approx train time for 20 steps | approx train_steps_per_second |
|---|---|---:|---:|
| MNLI | fp16 | 2s | 10.00 |
| MNLI | int8 | 16s | 1.25 |
| SST-5 | fp16 | 2s | 10.00 |
| SST-5 | int8 | 15s | 1.33 |

### 7.5 折算到 10,000 step 的训练时间

| task | mode | estimated seconds for 10k steps | estimated time for 10k steps |
|---|---|---:|---|
| MNLI | fp16 | 1000.0 | 0h16m40s |
| MNLI | int8 | 8000.0 | 2h13m20s |
| SST-5 | fp16 | 1000.0 | 0h16m40s |
| SST-5 | int8 | 7500.0 | 2h05m00s |

### 7.6 简要结论

- 在当前 `roberta-large` medium-model 训练路径上，`int8` 同样明显慢于 `fp16`
- 相对 `fp16`，`int8` 大约慢：
  - `MNLI`: `8.00x`
  - `SST-5`: `7.50x`
- 所以当前仓库下，无论 `opt-1.3b` 还是 `roberta-large`，训练吞吐都仍然是 `fp16` 更占优

## 8. 当前使用的 14 个 h 值

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
