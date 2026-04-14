# 代码结构与统一 Metadata 说明

## 1. 总览

这个仓库目前不是单一训练框架，而是两条并行演化的实验主线：

| 路径 | 主要对象 | 主入口 | 主要特点 |
| --- | --- | --- | --- |
| `large_models/` | 大型自回归模型，如 OPT / Mistral / LLaMA | `large_models/run.py` | 基于 HuggingFace `Trainer` 的大模型训练与评估，支持 zero-shot / ICL / regular FT / MeZO / LoRA / Prefix |
| `medium_models/` | 中型 masked LM / 分类模型，主要是 RoBERTa-类模型 | `medium_models/run.py` | 基于较老的 few-shot / prompt 框架演化，支持标准训练、kernel、linearhead、MeZO 及若干 ZO 变体 |

这次改动没有重构两条主线，而是在它们之上增加一个薄的共享层：

- 仓库根目录新增 `run_metadata.py`
- 两个入口脚本都调用这个 helper 生成稳定的 `run_metadata.json`
- 现有的 `run_summary.json` 保留，同时新增对 `run_metadata.json` 的引用和嵌入

目标是让不同方法、不同模型、不同低精度路径都能输出可比较的统一运行 metadata，而不是继续依赖 “QuZO / 非 QuZO” 这种历史标签。

## 2. 当前高层代码结构

### 2.1 `large_models/`

- `large_models/run.py`
  - 真实主入口。
  - 负责 CLI 参数解析、任务构建、模型加载、训练/评估调度、最终 summary 写出。
  - 这次也负责调用共享 helper 写 `run_metadata.json`。
- `large_models/trainer.py`
  - 自定义 `OurTrainer`。
  - 主要承载 MeZO 训练逻辑、ZO 扰动、ZO 更新、directional probe 等。
- `large_models/tasks.py`
  - 任务定义、数据采样、few-shot / full 模式解析。
- `large_models/utils.py`
  - 一些通用工具，包括结果写出辅助函数。
- `large_models/quzo.py`
  - ZO 侧的量化/投影逻辑。
  - 目前 `zo_quantization_bits=8/4` 会走低比特扰动/更新路径。
- `large_models/lora.py` / `large_models/prefix.py`
  - PEFT 路径。
- `large_models/metrics.py`
  - 部分任务指标计算。

`large_models/run.py` 中的重要流程位置：

1. `parse_args()`
   - 解析 `OurArguments`。
2. `Framework.load_model()`
   - 负责模型加载、dtype 选择、LoRA/Prefix、FP8 转换、ZO 侧量化预处理。
3. `Framework.train()`
   - 创建 `OurTrainer`，设置本地 metrics callback，写 `metrics_*.jsonl` / `metrics_*.csv`。
4. `main()`
   - 组织 train/eval/summary/final_metrics，并在入口层写 `run_metadata.json`。

### 2.2 `medium_models/`

- `medium_models/run.py`
  - 真实主入口。
  - 负责参数解析、数据准备、模型创建、trainer 选择、训练/验证/测试、summary 写出。
  - 这次也负责调用共享 helper 写 `run_metadata.json`。
- `medium_models/src/trainer.py`
  - 自定义 `Trainer`，内部承载标准训练和 ZO 路径。
  - 包含 adaptive-h、two-point h estimation、ZO probe、efficient_zero_order 等逻辑。
- `medium_models/src/linearhead_trainer.py`
  - `linearhead` 路径。
- `medium_models/src/kernel_trainer.py` / `kernel_solvers.py`
  - kernel 路径。
- `medium_models/src/dataset.py`
  - `FewShotDataset` 与特征组织。
- `medium_models/src/data_utils.py`
  - few-shot/full 数据目录解析与自动准备。
- `medium_models/src/processors.py`
  - 任务映射、标签数、指标计算映射。
- `medium_models/src/models.py`
  - 模型类型选择与部分模型转换逻辑。
- `medium_models/src/quzo.py`
  - ZO 侧量化/投影逻辑。
- `medium_models/src/prefix.py`
  - Prefix tuning。
- `medium_models/tools/`
  - 数据生成、结果聚合等离线工具。

`medium_models/run.py` 中的重要流程位置：

1. `HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))`
   - CLI 参数解析入口。
2. `resolve_and_prepare_data(...)`
   - few-shot/full 数据解析入口。
3. 模型创建区段
   - `AutoConfig` / `AutoTokenizer` / `model_fn.from_pretrained(...)`
   - Prefix / LoRA / FP8 / ZO 量化预处理都在这里接入。
4. `trainer = trainer_class(...)`
   - 依据 `training_args.trainer` 选择 trainer。
5. 训练/验证/测试结束后
   - 写 `run_summary.json`、`metrics_logs/*.csv`、`zo_directional_probe.csv`、`h_estimation.csv` 等。

## 3. 训练入口、方法选择、模型加载、低精度路径、日志落点

### 3.1 主训练入口

- 大模型主入口：`large_models/run.py:main`
- 中模型主入口：`medium_models/run.py:main`

### 3.2 CLI 参数解析

- 大模型：`large_models/run.py:parse_args`
- 中模型：`medium_models/run.py:main` 中 `HfArgumentParser(...)`

### 3.3 方法选择

- 大模型
  - 主要通过 `args.trainer` 区分：`none` / `regular` / `zo`
  - `PrefixTuning` / `LoRA` / `head_tuning` 是模型适配路径，不直接替代优化方法
- 中模型
  - `training_args.trainer` 决定主 trainer：`standard` / `kernel` / `linearhead`
  - `training_args.zero_order_optim=True` 时进入 MeZO 路径
  - 进一步通过 `efficient_zero_order`、`zo_by_layer`、`zo_variant`、`zero_order_use_trainer_optim` 等参数区分具体 ZO 变体

### 3.4 模型加载与 dtype

- 大模型
  - 位于 `large_models/run.py` 的 `Framework.load_model()`
  - 关键参数：`load_float16`、`load_bfloat16`、`load_int8`
- 中模型
  - 位于 `medium_models/run.py` 的模型创建区段
  - 常见相关参数：`fp16`、`bf16`、`efficient_zero_order_fp16`

### 3.5 低精度 / 量化 / FP8

- 两条主线都在各自 `run.py` 中提供 `maybe_convert_model_to_torchao_float8_training(...)`
  - 负责把兼容的 `nn.Linear` 替换成 torchao Float8 训练算子
  - 现在会顺带记录：
    - `fp8_mode`
    - `converted_linear_layers`
    - `total_linear_layers`
    - `skipped_layer_names`
- 两条主线都使用各自的 `quzo.py`
  - 负责 ZO 扰动/更新侧的量化逻辑
  - 现在统一通过 shared schema 反映为：
    - `int8_snap_enabled`
    - `zo_quantization`

### 3.6 日志与结果目录

- 大模型
  - 训练过程指标：`<output_dir>/metrics_<run_tag>.jsonl` 与 `metrics_<run_tag>.csv`
  - 方向探针：`<output_dir>/zo_directional_probe.csv`
  - 最终 summary：`<output_dir>/run_summary.json` 或 `run_summary_trainset*.json`
  - 最终指标：`<output_dir>/final_metrics*.json`
  - 新增统一 metadata：`<output_dir>/run_metadata.json`
- 中模型
  - 训练过程指标：`<output_dir>/metrics_logs/*.csv`
  - 方向探针：`<output_dir>/zo_directional_probe.csv`
  - h 估计：`<output_dir>/h_estimation.csv`
  - 最终 summary：`<output_dir>/run_summary.json`
  - 新增统一 metadata：`<output_dir>/run_metadata.json`

## 4. 新增的统一 Metadata 机制

### 4.1 入口

共享 helper 位于仓库根目录：

- `run_metadata.py`

核心函数：

- `collect_run_metadata(...)`
  - 从 args、model、运行环境、FP8 转换统计中组装统一 dict
- `write_run_metadata(...)`
  - 写出稳定文件 `run_metadata.json`
- `update_model_run_metadata(...)`
  - 供模型加载/转换阶段向 model 记录统一 metadata 片段

### 4.2 当前接入位置

- `large_models/run.py`
  - 在 `Framework(args, task)` 创建完成后收集并写 metadata
  - 在 `run_summary*.json` 里追加 `run_metadata` 和 `paths.run_metadata_json`
- `medium_models/run.py`
  - 在模型加载、FP8 转换、ZO 量化预处理完成后收集并写 metadata
  - 在 `run_summary.json` 里追加 `run_metadata` 和 `paths.run_metadata_json`

### 4.3 统一 schema

当前实现的 schema 如下。所有字段都会存在；不适用时使用默认值，而不是省略。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `metadata_schema_version` | integer | schema 版本，当前为 `1` |
| `zo_method` | string | 方法/优化器家族的统一标识 |
| `int8_snap_enabled` | boolean | 是否启用了 ZO 路径中的低比特网格 snapping / projection |
| `zo_quantization` | string | ZO 侧量化模式，当前可能为 `none` / `fp16` / `int8` / `int4` |
| `storage_dtype` | string | 模型主要存储 dtype，常见为 `fp32` / `fp16` / `bf16` / `mixed` / `unknown` |
| `compute_dtype` | string | 主要计算 dtype；若存在 FP8 或混合路径，可能为 `mixed` |
| `load_int8` | boolean | 是否走了 HF/int8 风格的模型加载路径 |
| `fp8_mode` | string | `none` / `native` / `emulated` |
| `fp8_native_enabled` | boolean | `fp8_mode == native` 的便捷布尔字段 |
| `converted_linear_layers` | integer | 实际被替换为 FP8 训练算子的 Linear 层数量 |
| `total_linear_layers` | integer | 模型中总 Linear 层数量 |
| `skipped_layer_names` | list[string] | 因尺寸等原因未被 FP8 转换的层名 |
| `device_type` | string | 设备描述，优先返回 CUDA 设备名，否则 `cpu` / `mps` 等 |
| `model_name` | string | 模型名或模型路径 |
| `task_name` | string | 任务名 |
| `seed` | integer or null | 运行 seed |
| `run_output_dir` | string | 本次运行输出目录的绝对路径 |
| `git_commit` | string or null | 当前 git commit |
| `hostname` | string or null | 主机名 |

### 4.4 字段语义约定

- `int8_snap_enabled=true`
  - 表示 ZO 路径里启用了低比特网格投影/离散化。
  - 这不等价于模型参数持久化存成 int8。
- `load_int8=true`
  - 表示模型加载时使用了 int8 加载路径。
  - 这不等价于 ZO snapping。
- `fp8_mode=native/emulated`
  - 反映的是算子级 FP8 训练模式。
  - 这不等价于全模型参数存储为 FP8。
- `zo_method`
  - 虽然字段名叫 `zo_method`，但现在的语义是“运行时采用的方法/优化器家族标识”。
  - 对非 ZO 路径，也要求给出稳定值，例如 `regular`、`standard`、`kernel`、`linearhead`、`inference`。

## 5. 当前两条主线里 `zo_method` 的映射方式

### 5.1 `large_models/`

- `trainer=zo` -> `mezo`
- `linear_probing=True` -> `linear_probing`
- `trainer=regular` -> `regular`
- `trainer=none` -> `inference`

说明：

- 大模型分支目前方法表示比较简单，MeZO 之外没有大量 ZO 子变体枚举。
- `PrefixTuning` / `LoRA` 仍保留在 config 中，不直接塞进 `zo_method`。

### 5.2 `medium_models/`

- `zero_order_optim=True, zo_by_layer=True` -> `mezo_layerwise`
- `zero_order_optim=True, zo_variant=grad_norm` -> `mezo_grad_norm`
- `zero_order_optim=True, zo_variant=param_norm` -> `mezo_param_norm`
- `zero_order_optim=True, efficient_zero_order=True` -> `mezo_efficient`
- `zero_order_optim=True, zero_order_use_trainer_optim=True, optimizer=adam` -> `mezo_adam`
- `zero_order_optim=True, optimizer_variant=signgd` -> `mezo_signgd`
- 其他 `zero_order_optim=True` -> `mezo`
- 非 ZO 路径回退到 `trainer`，例如 `standard` / `kernel` / `linearhead`

说明：

- 这里没有新增 CLI，只是把已有 flag 组合映射为统一字符串。
- 如果后续真的引入新的方法名，应在入口层增加新的稳定映射，而不是让下游解析零散布尔开关。

## 6. 今后所有方法/模型必须满足的公共能力

从现在开始，新增方法或模型时至少要满足下面这些要求：

- 必须能映射到稳定的 `zo_method`
  - 不允许完全依赖私有 tag、目录名、实验脚本名推断方法。
- 必须走统一 metadata 写出
  - 新方法不能绕过 `collect_run_metadata(...)` / `write_run_metadata(...)`。
- 必须填充共享 schema，而不是另起一套私有 JSON 格式
  - 特别是低精度/量化/FP8 相关信息，必须进入共享字段。
- 不适用字段必须给默认值
  - 例如没有 FP8 时，`fp8_mode=none`、`converted_linear_layers=0`、`skipped_layer_names=[]`。
- 模型侧低精度转换逻辑需要把统计信息挂到 model 上
  - 建议继续使用 `update_model_run_metadata(...)`。
- 新方法/模型即使只在单一路径存在，也要保证输出 `run_metadata.json`

## 7. 新增方法 / 新增模型 checklist

### 7.1 新增方法

1. 在对应入口脚本中注册方法选择逻辑。
   - `large_models/run.py` 或 `medium_models/run.py`
2. 给该方法分配稳定的 `zo_method` 字符串。
   - 名称应描述“方法/优化器家族”，不要混入目录标签。
3. 如果方法引入新的低精度行为：
   - 更新 `int8_snap_enabled`
   - 更新 `zo_quantization`
   - 如有 FP8 转换，更新 `fp8_mode`、`converted_linear_layers`、`skipped_layer_names`
4. 确保仍会调用共享 metadata helper。
5. 不要让新方法只写自己的私有 summary 而不写 `run_metadata.json`。

### 7.2 新增模型

1. 在对应模型加载逻辑中注册模型分支。
   - 大模型主要在 `large_models/run.py`
   - 中模型主要在 `medium_models/run.py` 与 `medium_models/src/models.py`
2. 确保 `model_name` 可从 args 或 config 稳定获取。
3. 如果模型支持 FP8 / 特殊低精度层替换：
   - 在转换完成后调用 `update_model_run_metadata(...)`
4. 如果模型没有 Linear 层或统计方式不同：
   - 仍要保证 `converted_linear_layers`、`total_linear_layers`、`skipped_layer_names` 有合理默认值
5. 保持旧脚本兼容
   - 不要为 metadata 改动去重命名既有 CLI

## 8. 当前局限与 TODO

- 仓库仍然是两套历史代码路径。
  - 这次只加了共享 metadata 层，没有把 large / medium 合并成一个训练框架。
- `zo_method` 目前仍是入口层映射出来的字符串。
  - 也就是说，medium 分支的一些方法区分仍然依赖若干 flag 组合。
- FP8 转换 helper 在两条主线里各自保留了一份。
  - 这次只统一了统计输出，没有进一步抽象成单一实现。
- `compute_dtype` 目前是“尽量真实的高层推断值”，不是逐算子精确 tracing。
  - 对混合精度或部分算子降精度场景，会使用 `mixed`。
- 多 train-set 场景下，大模型分支仍以单个 `output_dir` 为主。
  - 当前 `run_metadata.json` 描述的是该输出目录对应的统一运行配置，而不是每个 `trainset` 子结果的独立变体。

## 9. 实际文件落点

- 统一 helper：`run_metadata.py`
- 大模型 metadata 文件：`<large_models 运行 output_dir>/run_metadata.json`
- 中模型 metadata 文件：`<medium_models 运行 output_dir>/run_metadata.json`
- 新文档：`docs/code_structure_and_metadata.md`
