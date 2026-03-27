# MeZO on Medium-sized Masked Language Models

This part of the code is for MeZO experiments on RoBERTa-large. It is based on [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT) and [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

## Installation

Please install the latest versions of PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)) and Transformers (`transformers`). This code is tested on `torch==2.1.0.dev20230514+cu118` and `transformers==4.28.1` with Python 3.9.7, but should work with older/later versions of these packages too.

## Prepare the data

`run.py` now supports automatic data preparation for both few-shot and full-dataset modes.
When the target split directory is missing, it will:
- check `data/original`
- auto-download datasets via `data/download_dataset.sh` if needed
- auto-materialize the requested split under `data/k-shot-1k-test/$TASK/...`

```bash
TASK=SST-2 K=16 SEED=42 DATASET_MODE=fewshot bash mezo.sh
TASK=SST-2 SEED=42 DATASET_MODE=full bash mezo.sh
```

If you still want to pre-generate data manually, you can do:

```bash
# few-shot (historical behavior)
python tools/generate_k_shot_data.py --mode k-shot-1k-test --dataset_mode fewshot --k 16 --seed 42 --task SST-2

# full-dataset materialization (train/dev/test)
python tools/generate_k_shot_data.py --mode k-shot-1k-test --dataset_mode full --seed 42 --task SST-2
```

Few-shot directories remain `data/k-shot-1k-test/$TASK/$K-$SEED`.
Full-dataset directories use `data/k-shot-1k-test/$TASK/full-$SEED` (legacy `-16-$SEED` is also recognized).

## Usage

Use `run.py` for all functions and refer to `run.py` for the usage of all arguments.
```bash
python run.py {ARGUMENTS}
```

To reproduce our results in the paper, we also provide two example files `finetune.sh` (for all fine-tuning experiments) and `mezo.sh` (for all MeZO experiments). You can run them directly with the following commands (we use the following six datasets in our experiments -- `SST-2`, `sst-5`, `SNLI`, `MNLI`, `RTE`, and `trec`):
```bash
# Adam fine-tuning
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-5 MODEL=roberta-large bash finetune.sh

# Adam fine-tuning + prefix-tuning
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-2 MODEL=roberta-large EXTRA_TAG=prefix bash finetune.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# Adam fine-tuning + LoRA
TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16

# MeZO
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-6 EPS=1e-3 MODEL=roberta-large bash mezo.sh

# MeZO (full dataset via env var)
TASK=SST-2 SEED=42 DATASET_MODE=full BS=64 LR=1e-6 EPS=1e-3 MODEL=roberta-large bash mezo.sh

# MeZO (full dataset via CLI override)
TASK=SST-2 SEED=42 BS=64 LR=1e-6 EPS=1e-3 MODEL=roberta-large bash mezo.sh --dataset_mode full

# MeZO + prefix-tuning
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-2 EPS=1e-1 MODEL=roberta-large EXTRA_TAG=prefix bash mezo.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# MeZO + LoRA
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mezo.sh --apply_lora --lora_r 8 --lora_alpha 16
```
You can designate different hyperparameters by passing different environment variables as shown above. You can also directly add arguments at the end of the command to override the default ones. For all the hyperparameters you can control via environment variables, please refer to `finetune.sh` and `mezo.sh`. For the hyperparameters we used in our experiments, please refer to Appendix D of our paper.


## Gather results

All the results will be stored in `./log`. To analyze the results (for example, examine the grid search), use the following command
```bash
python tools/gather_result.py --condition "{'tag': 'k16-roberta-large-ft', 'task_name': 'sst-2'}"
```

Then the program will find all the trials that satisfy the condition in `./log`, and print the mean/std of the final results. Note that the task names are all lower-cased here.


## Ablations

RoBERTa-large models can be fine-tuned on most single GPUs, so we did not yet implement all of the memory-efficient ZO variants discussed in Appendix B. For now, if you want to run ablations other ZO ablations, you can add the flag `--zero_order_use_trainer_optim`, which will store the ZO gradients in the `param.grad` buffer and then use a PyTorch optimizer as usual. This causes the total memory consumption for ZO to be twice that of inference, which is still substantially less than that of backpropagation. The ablations can then be run with the additional following flags: 
- ZO-Adam: `--optimizer "adam"`
- ZO-Momentum: `--momentum <beta>`
- $n$-SPSA with $n>1$: `--zero_order_sample <n>` and you can add a linear or constant scheduler on it with `--zero_order_sample_scheduler {"linear", "constant"}`
- No prompt: `--few_shot_type finetune`

Appendix B discusses variants of ZO that modify the expectation and the variance. To run those one can use the following flags.
- Modify variance: `--zo_variant {"grad_norm", "param_norm"}`
- Recompute the control variate at the start of each epoch: `--recmopute_norms`
- Modify expectation: `--change_grad_estimate`
