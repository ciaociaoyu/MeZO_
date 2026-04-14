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

### 1.1 顶层目录地图

当前仓库最常用的顶层目录如下：

| 路径 | 作用 |
| --- | --- |
| `large_models/` | 大模型主线。自回归模型、HF `Trainer`、ICL / regular FT / MeZO / QuZO / Sparse MeZO / LoRA / Prefix / head tuning 都在这里接入。 |
| `medium_models/` | 中模型主线。分类 / prompt / few-shot / kernel / linearhead / MeZO / QuZO / Sparse MeZO 的主实验路径。 |
| `experiments/` | 统一放实验脚本、smoke test、h-sweep 结果、日志、提交器。当前 MNLI / SST-5 的 14-value 搜索也在这里。 |
| `docs/` | 项目文档。`code_structure_and_metadata.md` 是当前 canonical 结构文档，`sparse_mezo_h100_experiments.md` 是最近一轮 H100 实验记录。 |
| `run_metadata.py` | 共享 metadata helper，负责生成 `run_metadata.json` 并统一低精度/方法字段。 |
| `README.md` | 仓库总体说明。 |

### 1.2 当前支持范围总表

下面这张表描述的是“代码当前已经接入并能通过现有入口调用”的支持范围，不等同于“每个组合都已经系统验证过”。

| 维度 | `large_models/` | `medium_models/` |
| --- | --- | --- |
| 模型 | 以 `AutoModelForCausalLM` 为主，显式适配 `OPT`、`GPT-2`、`LLaMA`、`Mistral` | 显式模型类型为 `BERT`、`RoBERTa`、`OPT`、`GPT-2`，另有 `AutoModelForSequenceClassification` 路径 |
| 任务 | `SST-2`、`SST-5`、`BoolQ`、`SNLI`、`MNLI`、`RTE`、`SQuAD`、`DROP`、`CB`、`Copa`、`MultiRC`、`ReCoRD`、`WIC`、`WSC` | `CoLA`、`MNLI`、`MNLI-MM`、`MRPC`、`SST-2`/`sst2`、`STS-B`、`QQP`、`QNLI`、`RTE`、`WNLI`、`SNLI`、`MR`、`SST-5`、`SUBJ`、`TREC`、`CR`、`MPQA` |
| 方法 / baseline | `inference`、`regular`、`mezo`、`mezo+quzo`、`sparse_mezo`、`sparse_mezo+quzo`、`linear_probing`，以及和这些组合的 `LoRA` / `Prefix` / `head_tuning` | `standard`、`kernel`、`linearhead`、`mezo`、`mezo` 各种变体、`mezo+quzo`、`sparse_mezo`、`sparse_mezo+quzo`，以及 `prefix_tuning` / `LoRA` / `head_tuning` |
| 精度 / 量化 | `fp32`、`fp16`、`bf16`、`load_int8`、`torchao FP8`、`zo_quantization = none/fp16/int8/int4` | `fp32`、`fp16`、`bf16`、`torchao FP8`、`zo_quantization = none/fp16/int8/int4` |

当前最常跑、最稳定的组合主要是：

- `medium_models` 下的 `roberta-large + MNLI/SST-5`
- `large_models` 下的 `opt-1.3b + MNLI/SST-5`
- 方法上以 `MeZO`、`QuZO`、`Sparse MeZO` 为主
- “16-bit” 在当前正式脚本里通常保持仓库既有语义：`fp16`

### 1.3 常用实验目录与提交约定

当前实验组织方式已经基本固定：

| 路径 | 作用 |
| --- | --- |
| `experiments/h_sweep_14h/` | 当前 MNLI / SST-5 的 14-value `h` 搜索目录。这里的 “14h” 表示 14 个候选 `h` 值，不表示 14 小时。 |
| `experiments/h_sweep_14h/jobs/` | 正式 `sbatch` 作业脚本，包括 `quzo16`、`quzo8`、`sparse_mezo16` 等。 |
| `experiments/h_sweep_14h/results/` | sweep 结果目录，按方法 / 模型 / 任务分层。 |
| `experiments/h_sweep_14h/logs/` | sweep 日志目录，包括 `slurm_*.out` 和每个 `h` 的 `train.log` / `train.err`。 |
| `experiments/sparse_mezo_smoke/` | Sparse MeZO 的 smoke test 结果。 |
| `experiments/fp8_smoke*` | FP8 smoke test 结果。 |

当前仓库的正式批量实验提交风格是：

- 优先沿用 Slurm / `sbatch`
- 每个 sweep task 产出 `summary.jsonl`
- 每个单独 run 产出 `run_summary.json` 与 `run_metadata.json`
- h-sweep 脚本一般通过 `manifest.jsonl` / `summary.jsonl` 记录整体状态

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

- `trainer=zo, sparse_ratio < 1.0` -> `sparse_mezo`
- `trainer=zo, sparse_ratio == 1.0` -> `mezo`
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
- canonical 结构文档：`docs/code_structure_and_metadata.md`
- 最近实验记录：`docs/sparse_mezo_h100_experiments.md`
- 当前 14-value `h` 网格：`experiments/h_sweep_14h/h_values.sh`
- 当前正式 sweep 脚本目录：`experiments/h_sweep_14h/jobs/`
- 当前正式 sweep 结果目录：`experiments/h_sweep_14h/results/`
- 当前正式 sweep 日志目录：`experiments/h_sweep_14h/logs/`

## 10. Sparse MeZO 扩展

这份文档现在把 `docs/code_structure_and_metadata.md` 视为仓库内的 canonical 说明文件；当前没有第二份需要同步维护的 `code_structure_and_metadata*` 副本。

### 10.1 新增 / 修改文件

- 新增：
  - `large_models/sparse_mezo.py`
  - `medium_models/src/sparse_mezo.py`
  - `experiments/h_sweep_14h/jobs/roberta_mnli_sparse_mezo16_14h.sh`
  - `experiments/h_sweep_14h/jobs/roberta_sst5_sparse_mezo16_14h.sh`
  - `experiments/h_sweep_14h/submit_sparse_mezo16_searches.sh`
- 修改：
  - `large_models/run.py`
  - `large_models/trainer.py`
  - `medium_models/run.py`
  - `medium_models/src/trainer.py`
  - `run_metadata.py`
  - `docs/code_structure_and_metadata.md`

### 10.2 稳定方法名与 CLI

- 稳定方法名：`sparse_mezo`
- 新增 CLI 参数：
  - `--sparse_ratio`，默认 `1.0`
  - `--sparse_mask_strategy`，默认 `percentile_per_layer`
  - `--sparse_scope`，默认 `trainable_only`
  - `--sparse_log_active_fraction`，默认 `True`

语义约定：

- `sparse_ratio` 表示“每个 trainable tensor 内期望保留为 active 的坐标比例”。
- 当前 `percentile_per_layer` 实现会在每个 trainable tensor 内保留约 `sparse_ratio` 比例的低 `|param|` 坐标为 active。
- `sparse_ratio=1.0` 会关闭稀疏 masking，并回退为 vanilla MeZO。
- `sparse_scope=trainable_only` 表示只对当前 trainable 参数建立 mask；冻结参数不会被意外稀疏化。
- 训练日志与 `run_summary.json` 会记录实际 `active_fraction`，避免把配置值和实际激活比例混淆。

### 10.3 Sparse MeZO 与 QuZO 的组合顺序

当前组合顺序是显式固定的：

1. 先按现有 MeZO / QuZO 逻辑构造方向向量。
2. 再把 Sparse MeZO mask 施加到 trainable 参数对应的方向张量上。
3. 如果当前是 QuZO 低比特路径（`zo_quantization_bits in {8, 4}`），则在 masked perturbation / update 之后继续沿用原有的参数 snapping / quantization 投影。

因此日志和 metadata 的区分方式为：

- `zo_method=mezo` + `zo_quantization=none/fp16`：vanilla MeZO
- `zo_method=mezo` + `zo_quantization=int8/int4`：MeZO + QuZO
- `zo_method=sparse_mezo` + `zo_quantization=none/fp16`：Sparse MeZO
- `zo_method=sparse_mezo` + `zo_quantization=int8/int4`：Sparse MeZO + QuZO

### 10.4 Metadata 记录

`run_metadata.json` 当前会稳定记录以下 Sparse MeZO 字段：

- `metadata_schema_version=1`
- `zo_method`
- `sparse_mezo_enabled`
- `sparse_ratio`
- `sparse_mask_strategy`
- `sparse_scope`
- `sparse_log_active_fraction`
- 以及原有的 `zo_quantization` / `storage_dtype` / `compute_dtype`

medium 路径的 `run_summary.json` 还会在 `artifacts.sparse_mezo_last_stats` 里保存最近一次 step 的实际统计，包括：

- `configured_ratio`
- `active_params`
- `total_trainable_params`
- `active_fraction`
- `mask_strategy`
- `scope`
- `global_step`

## 11. MNLI / SST-5 的当前实验路径与 16-bit 约定

### 11.1 当前 h-search 路径

仓库内现成的 14-value rough-search / h-sweep 路径位于：

- `experiments/h_sweep_14h/`
- 14-value 网格定义在：`experiments/h_sweep_14h/h_values.sh`
- 当前 RoBERTa MNLI / SST-5 搜索脚本位于：
  - `experiments/h_sweep_14h/jobs/roberta_mnli_quzo16_14h.sh`
  - `experiments/h_sweep_14h/jobs/roberta_sst5_quzo16_14h.sh`

本轮 Sparse MeZO 的正式 14-value 搜索沿用这一条 medium / RoBERTa 路径。

### 11.2 当前“16-bit”含义

MNLI / SST-5 在当前 medium h-search 路径里的 16-bit 约定保持不变：

- `--zo_two_point_precision fp16`
- `--zo_quantization_bits 16`

这里的 `16-bit` 语义仍然是仓库当前的“FP16 MeZO path”，而不是把 `8/4-bit QuZO` 的量化投影逻辑强行应用到 16-bit 上。

### 11.3 当前 14-value h grid

当前仓库里已经存在精确的 14 个候选值，本轮直接复用：

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

## 12. Smoke Test 与 Launcher

### 12.1 H100 smoke test 覆盖

本轮 smoke test 先覆盖当前真正用于 MNLI / SST-5 h-search 的 medium / RoBERTa 路径：

- `roberta-large + MNLI + sparse_mezo + 16-bit`
- `roberta-large + SST-5 + sparse_mezo + 16-bit`

验证点包括：

- 模型加载
- 当前 16-bit 路径
- Sparse MeZO step
- directional probe CSV
- `run_summary.json`
- `run_metadata.json`
- 稀疏 active fraction 日志

### 12.2 正式 launcher 位置

- `experiments/h_sweep_14h/jobs/roberta_mnli_sparse_mezo16_14h.sh`
- `experiments/h_sweep_14h/jobs/roberta_sst5_sparse_mezo16_14h.sh`
- `experiments/h_sweep_14h/submit_sparse_mezo16_searches.sh`

`submit_sparse_mezo16_searches.sh` 采用当前仓库的 Slurm / `sbatch` 风格，并通过 dependency 把两个 full search 串起来，保证单卡环境下同一时刻只会有一个 full GPU 训练作业处于活动状态。
