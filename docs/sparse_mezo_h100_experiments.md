# Sparse MeZO H100 实验记录

更新时间：`2026-04-18`

## 0. 环境映射与方法矩阵测速进度

当前仓库的实验环境按模型族固定分配，后续跑 smoke test、speed benchmark、Slurm sweep 都应遵循这套映射：

| 模型 | 代码路径 | conda 环境 | 备注 |
|---|---|---|---|
| `roberta-large` | `medium_models/` | `ciao` | 当前 medium-model 路径和 RoBERTa 相关实验都使用这个环境。 |
| `opt-1.3b` 等非 Mistral 大模型 | `large_models/` | `mezo-env` | 当前 large-model 的 OPT 系实验使用这个环境。 |
| `mistral-7b` | `large_models/` | `mezo-mistral` | Mistral 单独使用这个环境，不与 `mezo-env` 混用。 |

与环境映射对应的可恢复测速脚本：

- [run_zo_method_speed_matrix.py](/scratch/jy03364/MeZO_/experiments/speed_bench_h100/run_zo_method_speed_matrix.py)
- 输出目录：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/zo_method_matrix_20260418`
- 汇总文件：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/zo_method_matrix_20260418/summary.jsonl`

截至 `2026-04-18` 文档更新时，这个 H100 多方法测速矩阵已经跑完，当前最终快照如下：

- 总格子数：`72`
- 已完成：`66`
- 不支持：`6`
  - 唯一的不支持项是 `roberta-large + {lozo,hizoo} + int8 + {MNLI,SST-5,BoolQ}`
- 当前矩阵里没有仍在运行/待完成的格子
- 最终覆盖：
  - `roberta-large`: `18 completed + 6 unsupported`
  - `opt-1.3b`: `24 completed`
  - `mistral-7b`: `24 completed`
- `roberta-large + BoolQ` 已经补齐正式 `medium_models` 路径并纳入矩阵，不再是“不支持”
- 当前 `squeue` 里还能看到的 `sparse` 任务属于正式 `h-sweep`，不是这份测速矩阵

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

注意：本节最早记录的 `int8` 是 `--load_int8 --zo_quantization_bits 32`，也就是 “int8 模型加载 + plain MeZO”，不是 QuZO int8。后文会单独补真正 `--zo_quantization int8` 的 QuZO int8 结果。

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

本次这一组 benchmark 的“精度”定义按当前仓库 large-model 语义执行：

- `fp16`：
  - `--load_float16`
  - `--zo_quantization_bits 16`
- `int8`：
  - `--load_int8`
  - `--zo_quantization_bits 32`

也就是说，这一组里的 `int8` 指的是模型加载路径，不是 QuZO int8 扰动/更新路径。

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

### 6.6 真正的 QuZO int8 (`--zo_quantization int8`) 补测

上面 `6.1` 到 `6.5` 的 `int8` 是 `--load_int8 --zo_quantization_bits 32`，也就是 “int8 模型加载 + plain MeZO”。为了和 `medium_models` 里的 `roberta int8` 公平对齐，我另外补跑了一组真正的 QuZO int8：

- `--load_float16`
- `--zo_quantization int8`

也就是说，这组结果对应的是：

- FP16 模型加载
- QuZO 8-bit 扰动 / 更新路径

结果目录：

- `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/opt13b_quzo_int8_h100_final_20260414_215951`
- 汇总文件：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/opt13b_quzo_int8_h100_final_20260414_215951/summary.jsonl`

其中 `SQuAD` 这次为了避开本地 `datasets` 的损坏缓存，使用了隔离的 `HF_DATASETS_CACHE` 重跑；训练逻辑没有改。

| task | mode | train_steps_per_second | train_runtime (20 step) | wall_seconds |
|---|---|---:|---:|---:|
| MNLI | quzo_int8 | 0.791 | 25.2814 | 58.71 |
| SST-5 | quzo_int8 | 0.881 | 22.6969 | 45.03 |
| BoolQ | quzo_int8 | 0.654 | 30.5612 | 53.29 |
| SQuAD | quzo_int8 | 0.811 | 24.6578 | 54.42 |

按 `10,000 step` 折算：

| task | mode | estimated seconds for 10k steps | estimated time for 10k steps |
|---|---|---:|---|
| MNLI | quzo_int8 | 12642.2 | 3h30m42.2s |
| SST-5 | quzo_int8 | 11350.7 | 3h09m10.7s |
| BoolQ | quzo_int8 | 15290.5 | 4h14m50.5s |
| SQuAD | quzo_int8 | 12330.5 | 3h25m30.5s |

和 `6.3` 里的 `fp16` 相比，真正的 QuZO int8 更慢：

- `MNLI`: `5.80x`
- `SST-5`: `7.40x`
- `BoolQ`: `5.77x`
- `SQuAD`: `5.72x`

这也是为什么前面如果把 `load_int8 + plain MeZO` 和 `QuZO int8` 混在一起看，会得出错误的速度结论。

## 7. H100 上 `roberta-large` 速度基准

目标：补齐当前正式 `medium_models` 路径上 `roberta-large` 的短程训练吞吐对比。这里不重跑已经测过的 `opt-1.3b`，只新增 `MNLI / SST-5` 的 `16-bit` 与 `int8`。

### 7.1 路径与范围说明

- 当前正式 `roberta-large` 训练路径是 `medium_models`
- 这节记录的是最早一轮 `2026-04-14` 的基线测速，当时只先测了：
  - `MNLI`
  - `SST-5`
- 此后已经把 `BoolQ` 补进 `medium_models` 正式路径；最终的 `MeZO / Sparse MeZO / LOZO / HiZOO` 多方法矩阵见第 `8` 节

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

## 8. H100 多方法最终测速矩阵

结果位置：

- 运行脚本：
  - [run_zo_method_speed_matrix.py](/scratch/jy03364/MeZO_/experiments/speed_bench_h100/run_zo_method_speed_matrix.py)
- 输出目录：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/zo_method_matrix_20260418`
- 汇总文件：
  - `/scratch/jy03364/MeZO_/experiments/speed_bench_h100/zo_method_matrix_20260418/summary.jsonl`

矩阵定义：

- 方法：`MeZO`, `Sparse MeZO`, `LOZO`, `HiZOO`
- 模型：`roberta-large`, `opt-1.3b`, `mistral-7b`
- 任务：`MNLI`, `SST-5`, `BoolQ`
- 精度标签：`fp16`, `int8`

重要语义说明：

- `roberta-large` 走 `medium_models` 路径：
  - `fp16` = `--zo_two_point_precision fp16 --zo_quantization_bits 16`
  - `int8` = QuZO `--zo_quantization int8`
- `opt-1.3b` 和 `mistral-7b` 走 `large_models` 路径：
  - `fp16` = `--load_float16`，其中 `mezo/sparse_mezo` 额外带 `--zo_quantization_bits 16`
  - 这份矩阵里的 `int8` = `--load_int8 --zo_quantization_bits 32`
- 也就是说，大模型矩阵里的 `int8` 是“加载为 int8”，不是 QuZO 低比特扰动/更新；它和 `roberta-large` 的 QuZO `int8` 不是同一种语义，横向比较时需要单独说明

最终状态：

- `72` 个格子里，`66 completed + 6 unsupported`
- `6` 个不支持全部来自 `roberta-large + {lozo,hizoo} + int8`
- 原因不是测速中断，而是当前实现明确没有把 `LOZO / HiZOO` 接到 `medium_models` 的 QuZO `int8` 扰动路径上
- `roberta-large` 的 `BoolQ` 正式路径已经补齐，因此当前矩阵里：
  - `roberta-large + BoolQ + {mezo,sparse_mezo,lozo,hizoo} + fp16` 全部完成
  - `roberta-large + BoolQ + {mezo,sparse_mezo} + int8` 也已完成

### 8.0 已完成测速总表

下面这组表只列出已经 `completed` 的格子，不再混入 `unsupported`。如果只想看当前已经跑完的结果，直接看这一节即可。

按每个 `model + task + precision` 取最快方法：

`fp16`

| model | task | fastest method | samples/sec | sec/step |
|---|---|---|---:|---:|
| roberta-large | MNLI | mezo | 508.972 | 0.063 |
| roberta-large | SST-5 | mezo | 588.180 | 0.054 |
| roberta-large | BoolQ | mezo | 171.657 | 0.186 |
| opt-1.3b | MNLI | lozo | 221.612 | 0.072 |
| opt-1.3b | SST-5 | lozo | 316.942 | 0.050 |
| opt-1.3b | BoolQ | lozo | 135.139 | 0.118 |
| mistral-7b | MNLI | lozo | 61.473 | 0.260 |
| mistral-7b | SST-5 | lozo | 89.184 | 0.179 |
| mistral-7b | BoolQ | lozo | 31.739 | 0.504 |

`int8`

| model | task | fastest method | samples/sec | sec/step |
|---|---|---|---:|---:|
| roberta-large | MNLI | mezo | 55.100 | 0.581 |
| roberta-large | SST-5 | mezo | 55.358 | 0.578 |
| roberta-large | BoolQ | mezo | 22.628 | 1.414 |
| opt-1.3b | MNLI | lozo | 46.553 | 0.344 |
| opt-1.3b | SST-5 | lozo | 63.654 | 0.251 |
| opt-1.3b | BoolQ | lozo | 25.916 | 0.617 |
| mistral-7b | MNLI | lozo | 9.233 | 1.733 |
| mistral-7b | SST-5 | lozo | 14.344 | 1.115 |
| mistral-7b | BoolQ | lozo | 4.678 | 3.420 |

完整 `completed` 明细表：

| model | task | method | precision | samples/sec | sec/step |
|---|---|---|---|---:|---:|
| roberta-large | MNLI | mezo | fp16 | 508.972 | 0.063 |
| roberta-large | MNLI | mezo | int8 | 55.100 | 0.581 |
| roberta-large | MNLI | sparse_mezo | fp16 | 171.801 | 0.186 |
| roberta-large | MNLI | sparse_mezo | int8 | 45.342 | 0.706 |
| roberta-large | MNLI | lozo | fp16 | 282.549 | 0.113 |
| roberta-large | MNLI | hizoo | fp16 | 177.368 | 0.180 |
| roberta-large | SST-5 | mezo | fp16 | 588.180 | 0.054 |
| roberta-large | SST-5 | mezo | int8 | 55.358 | 0.578 |
| roberta-large | SST-5 | sparse_mezo | fp16 | 184.020 | 0.174 |
| roberta-large | SST-5 | sparse_mezo | int8 | 45.754 | 0.699 |
| roberta-large | SST-5 | lozo | fp16 | 353.764 | 0.090 |
| roberta-large | SST-5 | hizoo | fp16 | 238.266 | 0.134 |
| roberta-large | BoolQ | mezo | fp16 | 171.657 | 0.186 |
| roberta-large | BoolQ | mezo | int8 | 22.628 | 1.414 |
| roberta-large | BoolQ | sparse_mezo | fp16 | 74.567 | 0.429 |
| roberta-large | BoolQ | sparse_mezo | int8 | 21.697 | 1.475 |
| roberta-large | BoolQ | lozo | fp16 | 69.843 | 0.458 |
| roberta-large | BoolQ | hizoo | fp16 | 45.123 | 0.709 |
| opt-1.3b | MNLI | mezo | fp16 | 180.961 | 0.088 |
| opt-1.3b | MNLI | mezo | int8 | 42.263 | 0.379 |
| opt-1.3b | MNLI | sparse_mezo | fp16 | 7.201 | 2.222 |
| opt-1.3b | MNLI | sparse_mezo | int8 | 3.256 | 4.915 |
| opt-1.3b | MNLI | lozo | fp16 | 221.612 | 0.072 |
| opt-1.3b | MNLI | lozo | int8 | 46.553 | 0.344 |
| opt-1.3b | MNLI | hizoo | fp16 | 142.566 | 0.112 |
| opt-1.3b | MNLI | hizoo | int8 | 29.884 | 0.535 |
| opt-1.3b | SST-5 | mezo | fp16 | 218.605 | 0.073 |
| opt-1.3b | SST-5 | mezo | int8 | 56.297 | 0.284 |
| opt-1.3b | SST-5 | sparse_mezo | fp16 | 7.239 | 2.210 |
| opt-1.3b | SST-5 | sparse_mezo | int8 | 3.310 | 4.834 |
| opt-1.3b | SST-5 | lozo | fp16 | 316.942 | 0.050 |
| opt-1.3b | SST-5 | lozo | int8 | 63.654 | 0.251 |
| opt-1.3b | SST-5 | hizoo | fp16 | 192.834 | 0.083 |
| opt-1.3b | SST-5 | hizoo | int8 | 40.920 | 0.391 |
| opt-1.3b | BoolQ | mezo | fp16 | 113.712 | 0.141 |
| opt-1.3b | BoolQ | mezo | int8 | 24.446 | 0.655 |
| opt-1.3b | BoolQ | sparse_mezo | fp16 | 5.997 | 2.668 |
| opt-1.3b | BoolQ | sparse_mezo | int8 | 3.072 | 5.209 |
| opt-1.3b | BoolQ | lozo | fp16 | 135.139 | 0.118 |
| opt-1.3b | BoolQ | lozo | int8 | 25.916 | 0.617 |
| opt-1.3b | BoolQ | hizoo | fp16 | 88.523 | 0.181 |
| opt-1.3b | BoolQ | hizoo | int8 | 16.893 | 0.947 |
| mistral-7b | MNLI | mezo | fp16 | 44.190 | 0.362 |
| mistral-7b | MNLI | mezo | int8 | 8.394 | 1.906 |
| mistral-7b | MNLI | sparse_mezo | fp16 | 1.178 | 13.577 |
| mistral-7b | MNLI | sparse_mezo | int8 | 0.561 | 28.529 |
| mistral-7b | MNLI | lozo | fp16 | 61.473 | 0.260 |
| mistral-7b | MNLI | lozo | int8 | 9.233 | 1.733 |
| mistral-7b | MNLI | hizoo | fp16 | 35.588 | 0.450 |
| mistral-7b | MNLI | hizoo | int8 | 5.910 | 2.707 |
| mistral-7b | SST-5 | mezo | fp16 | 56.054 | 0.285 |
| mistral-7b | SST-5 | mezo | int8 | 12.373 | 1.293 |
| mistral-7b | SST-5 | sparse_mezo | fp16 | 1.215 | 13.167 |
| mistral-7b | SST-5 | sparse_mezo | int8 | 0.563 | 28.426 |
| mistral-7b | SST-5 | lozo | fp16 | 89.184 | 0.179 |
| mistral-7b | SST-5 | lozo | int8 | 14.344 | 1.115 |
| mistral-7b | SST-5 | hizoo | fp16 | 45.613 | 0.351 |
| mistral-7b | SST-5 | hizoo | int8 | 8.809 | 1.816 |
| mistral-7b | BoolQ | mezo | fp16 | 16.667 | 0.960 |
| mistral-7b | BoolQ | mezo | int8 | 4.433 | 3.610 |
| mistral-7b | BoolQ | sparse_mezo | fp16 | 1.180 | 13.555 |
| mistral-7b | BoolQ | sparse_mezo | int8 | 0.527 | 30.351 |
| mistral-7b | BoolQ | lozo | fp16 | 31.739 | 0.504 |
| mistral-7b | BoolQ | lozo | int8 | 4.678 | 3.420 |
| mistral-7b | BoolQ | hizoo | fp16 | 20.830 | 0.768 |
| mistral-7b | BoolQ | hizoo | int8 | 3.058 | 5.233 |

速度解读摘要：

- `roberta-large` 上，`Sparse MeZO` 比 `MeZO` 慢，但量级还是同一档：
  - `MNLI fp16`: `2.96x` slower
  - `SST-5 fp16`: `3.20x` slower
  - `BoolQ fp16`: `2.30x` slower
- `opt-1.3b` 和 `mistral-7b` 上，当前 large-model `Sparse MeZO` 实现显著更慢：
  - `opt-1.3b`: 大约 `18.96x` 到 `30.20x` 慢于对应 `MeZO fp16`
  - `mistral-7b`: 大约 `14.12x` 到 `46.13x` 慢于对应 `MeZO fp16`
- 这和前面分析一致：当前 large-model `Sparse MeZO` 还是旧的 dense + mask 路径，没有达到论文/官方仓库那种更激进的稀疏执行收益

### 8.1 `roberta-large`

| task | method | fp16 samples/sec | fp16 sec/step | int8 samples/sec | int8 sec/step | note |
|---|---|---:|---:|---:|---:|---|
| MNLI | mezo | 508.972 | 0.063 | 55.100 | 0.581 |  |
| MNLI | sparse_mezo | 171.801 | 0.186 | 45.342 | 0.706 |  |
| MNLI | lozo | 282.549 | 0.113 | unsupported | unsupported | unsupported |
| MNLI | hizoo | 177.368 | 0.180 | unsupported | unsupported | unsupported |
| SST-5 | mezo | 588.180 | 0.054 | 55.358 | 0.578 |  |
| SST-5 | sparse_mezo | 184.020 | 0.174 | 45.754 | 0.699 |  |
| SST-5 | lozo | 353.764 | 0.090 | unsupported | unsupported | unsupported |
| SST-5 | hizoo | 238.266 | 0.134 | unsupported | unsupported | unsupported |
| BoolQ | mezo | 171.657 | 0.186 | 22.628 | 1.414 |  |
| BoolQ | sparse_mezo | 74.567 | 0.429 | 21.697 | 1.475 |  |
| BoolQ | lozo | 69.843 | 0.458 | unsupported | unsupported | unsupported |
| BoolQ | hizoo | 45.123 | 0.709 | unsupported | unsupported | unsupported |

`roberta-large int8` 当前状态结论：

- 已跑完所有当前实现“支持”的格子
- 已完成的是：
  - `mezo + {MNLI,SST-5,BoolQ} + int8`
  - `sparse_mezo + {MNLI,SST-5,BoolQ} + int8`
- 未完成的不是卡住，而是当前实现明确不支持：
  - `lozo + {MNLI,SST-5,BoolQ} + int8`
  - `hizoo + {MNLI,SST-5,BoolQ} + int8`

### 8.2 `opt-1.3b`

| task | method | fp16 samples/sec | fp16 sec/step | int8 samples/sec | int8 sec/step | note |
|---|---|---:|---:|---:|---:|---|
| MNLI | mezo | 180.961 | 0.088 | 42.263 | 0.379 |  |
| MNLI | sparse_mezo | 7.201 | 2.222 | 3.256 | 4.915 |  |
| MNLI | lozo | 221.612 | 0.072 | 46.553 | 0.344 |  |
| MNLI | hizoo | 142.566 | 0.112 | 29.884 | 0.535 |  |
| SST-5 | mezo | 218.605 | 0.073 | 56.297 | 0.284 |  |
| SST-5 | sparse_mezo | 7.239 | 2.210 | 3.310 | 4.834 |  |
| SST-5 | lozo | 316.942 | 0.050 | 63.654 | 0.251 |  |
| SST-5 | hizoo | 192.834 | 0.083 | 40.920 | 0.391 |  |
| BoolQ | mezo | 113.712 | 0.141 | 24.446 | 0.655 |  |
| BoolQ | sparse_mezo | 5.997 | 2.668 | 3.072 | 5.209 |  |
| BoolQ | lozo | 135.139 | 0.118 | 25.916 | 0.617 |  |
| BoolQ | hizoo | 88.523 | 0.181 | 16.893 | 0.947 |  |

### 8.3 `mistral-7b`

| task | method | fp16 samples/sec | fp16 sec/step | int8 samples/sec | int8 sec/step | note |
|---|---|---:|---:|---:|---:|---|
| MNLI | mezo | 44.190 | 0.362 | 8.394 | 1.906 |  |
| MNLI | sparse_mezo | 1.178 | 13.577 | 0.561 | 28.529 |  |
| MNLI | lozo | 61.473 | 0.260 | 9.233 | 1.733 |  |
| MNLI | hizoo | 35.588 | 0.450 | 5.910 | 2.707 |  |
| SST-5 | mezo | 56.054 | 0.285 | 12.373 | 1.293 |  |
| SST-5 | sparse_mezo | 1.215 | 13.167 | 0.563 | 28.426 |  |
| SST-5 | lozo | 89.184 | 0.179 | 14.344 | 1.115 |  |
| SST-5 | hizoo | 45.613 | 0.351 | 8.809 | 1.816 |  |
| BoolQ | mezo | 16.667 | 0.960 | 4.433 | 3.610 |  |
| BoolQ | sparse_mezo | 1.180 | 13.555 | 0.527 | 30.351 |  |
| BoolQ | lozo | 31.739 | 0.504 | 4.678 | 3.420 |  |
| BoolQ | hizoo | 20.830 | 0.768 | 3.058 | 5.233 |  |

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
