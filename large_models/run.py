import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification, TrainerCallback
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from trainer import OurTrainer
import random

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, MNLI, SQuAD, DROP

    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none"
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                # Head tuning
                from ht_opt import OPTForCausalLM
                model = OPTForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                torch_dtype = None
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                load_kwargs = {}
                if torch_dtype is not None:
                    load_kwargs["torch_dtype"] = torch_dtype
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    **load_kwargs,
                )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                    max_memory={i: f'{free_in_GB-5}GB' for i in range(torch.cuda.device_count())},
                    load_in_8bit=self.args.load_int8,
                )
            model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        # Decoder-only LMs may miss a pad token. Keep tokenizer/model pad ids aligned.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None and model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        if "llama" in self.args.model_name:
            # LLaMA padding token
            tokenizer.pad_token_id = 0 # technically <unk>
            model.config.pad_token_id = tokenizer.pad_token_id
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(),
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}")
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, samples, convert_one_fn):
                self.samples = samples
                self.convert_one_fn = convert_one_fn

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.convert_one_fn(self.samples[idx])


        task_template = self.task.get_template()

        def _convert_one(sample):
            """
            Convert one sample to HF-compatible format.
            We tokenize lazily in __getitem__ to avoid large up-front memory for big datasets (e.g., MNLI full train).
            """
            encoded_candidates, option_lens = encode_prompt(
                self.task, task_template, [], sample, self.tokenizer,
                max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True,
                max_new_tokens=self.args.max_new_tokens
            )
            if self.task.generation:
                correct_candidate_id = 0
            elif isinstance(sample.correct_candidate, list):
                correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
            else:
                correct_candidate_id = sample.candidates.index(sample.correct_candidate)

            if self.args.non_diff:
                # For non-differentiable objective, there is no teacher forcing thus the
                # current answer part is removed
                encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

            if self.args.train_as_classification:
                # For classification, we provide the label as the correct candidate id
                return [{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))]
            if self.args.only_train_option:
                # Otherwise, it is just LM-style teacher forcing
                if self.args.non_diff:
                    # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                    return {"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate}
                return {"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]}
            return {"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]}

        with count_time("Preparing training/evaluation datasets"):
            train_dataset = HFDataset(train_samples, _convert_one)
            eval_dataset = HFDataset(eval_samples, _convert_one)

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        # ---- 训练过程指标日志回调（中文注释）--------------------------------------
        # 目标：在每一次**优化步**（global_step）记录训练 loss，并在评估时记录验证/训练探针集的准确率。
        # 日志输出到 result 目录下，文件名包含 任务名+模型名+eps 等关键信息，方便多组实验对比。
        # 注意：HuggingFace 的 Trainer 在设置了 logging_strategy="steps" 且 logging_steps=1 时，
        # 会在每个优化步触发 on_log 回调（若使用梯度累积，则每累计完成一次为一个优化步）。
        import os
        import json
        import time
        import random

        # 生成当前运行的标签（包含任务名/模型名/样本数/eps 等），用于区分不同实验
        run_tag = result_file_tag(self.args)  # 例如：SST2-opt-125m-eps0.001-...
        logs_dir = os.path.join("result")  # 统一把迭代日志放在 result 目录
        os.makedirs(logs_dir, exist_ok=True)

        class _HistoryWriter:
            """
            简单的历史日志写入器：
            - JSONL：`result/metrics_{run_tag}.jsonl`
            - CSV：  `result/metrics_{run_tag}.csv`
            每一行都会额外带上 task/model/eps 字段，便于后期聚合分析。
            """
            def __init__(self, out_dir: str, run_tag: str, task_name: str, model_name: str, eps: float):
                self.dir = out_dir
                self.run_tag = run_tag
                self.task_name = task_name
                self.model_name = model_name
                self.eps = eps
                self.jsonl_path = os.path.join(self.dir, f"metrics_{self.run_tag}.jsonl")
                self.csv_path = os.path.join(self.dir, f"metrics_{self.run_tag}.csv")
                # 初始化 CSV 表头（包含任务信息）
                if not os.path.exists(self.csv_path):
                    with open(self.csv_path, "w", encoding="utf-8") as f:
                        f.write("time,step,epoch,phase,metric,value,task,model,eps\n")

            def append_jsonl(self, obj: dict):
                # JSONL 中也冗余写入任务信息，方便独立解析
                obj = dict(obj)
                obj.update({
                    "task": self.task_name,
                    "model": self.model_name,
                    "eps": self.eps,
                })
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            def append_csv_row(self, time_s: str, step: int, epoch: float, phase: str, metric: str, value: float):
                with open(self.csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{time_s},{step},{epoch},{phase},{metric},{value},{self.task_name},{self.model_name},{self.eps}\n")

        class MetricsRecorder(TrainerCallback):
            """记录训练 loss（on_log）以及在评估阶段计算并记录指标（on_evaluate）。
            说明：
            - 训练步的 loss 等由 Trainer 传入 logs（需要 logging_strategy=steps 且 logging_steps=1）。
            - on_evaluate 中，调用 framework.evaluate 以得到 eval 的准确率；另外对训练集抽样一小部分做探针评估（避免太慢）。
            """
            def __init__(self, framework, train_samples, eval_samples, out_dir, run_tag: str, train_probe_size: int = 256):
                self.framework = framework
                self.train_samples_full = train_samples
                self.eval_samples = eval_samples
                self.writer = _HistoryWriter(
                    out_dir,
                    run_tag,
                    task_name=self.framework.args.task_name,
                    model_name=self.framework.args.model_name.split("/")[-1],
                    eps=self.framework.args.zo_eps,
                )
                self.train_probe_size = train_probe_size

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                step = int(state.global_step)
                epoch_val = float(state.epoch) if state.epoch is not None else -1
                # 逐项把 logs 内的标量（如 loss、learning_rate）写入
                for k, v in logs.items():
                    if isinstance(v, (int, float)) and k not in ("total_flos",):
                        self.writer.append_jsonl({
                            "time": ts, "step": step, "epoch": epoch_val,
                            "phase": "train", "metric": k, "value": float(v)
                        })
                        self.writer.append_csv_row(ts, step, epoch_val, "train", k, float(v))

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                step = int(state.global_step)
                epoch_val = float(state.epoch) if state.epoch is not None else -1

                # 1) 先记录 Trainer 自身传入的 metrics（通常包含 eval_loss、eval_runtime 等）
                #    统一标记为 phase="eval"，便于和自定义指标一起分析。
                if metrics:
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            self.writer.append_jsonl({
                                "time": ts, "step": step, "epoch": epoch_val,
                                "phase": "eval", "metric": mk, "value": float(mv)
                            })
                            self.writer.append_csv_row(ts, step, epoch_val, "eval", mk, float(mv))

                # 2) 使用自定义的 framework.evaluate 计算任务指标（如 accuracy / F1），也标记为 phase="eval"
                eval_metrics = self.framework.evaluate([], self.eval_samples)
                for mk, mv in eval_metrics.items():
                    self.writer.append_jsonl({
                        "time": ts, "step": step, "epoch": epoch_val,
                        "phase": "eval", "metric": mk, "value": float(mv)
                    })
                    self.writer.append_csv_row(ts, step, epoch_val, "eval", mk, float(mv))

                # 3) 训练集抽样做探针评估（train_probe），减少耗时
                n = min(self.train_probe_size, len(self.train_samples_full) if self.train_samples_full is not None else 0)
                if n > 0:
                    subset = random.sample(self.train_samples_full, n) if len(self.train_samples_full) > n else list(self.train_samples_full)
                    train_metrics = self.framework.evaluate([], subset)
                    for mk, mv in train_metrics.items():
                        self.writer.append_jsonl({
                            "time": ts, "step": step, "epoch": epoch_val,
                            "phase": "train_probe", "metric": mk, "value": float(mv)
                        })
                        self.writer.append_csv_row(ts, step, epoch_val, "train_probe", mk, float(mv))
        # ---- end metrics logging callback -------------------------------------------

        # 确保按“每步”记录日志：
        # 如果使用了梯度累积，"步" 指完成一次累积后的优化步。
        self.args.logging_strategy = "steps"
        self.args.logging_steps = 1
        # 可选：不将日志上报到外部平台（如 wandb），只写本地文件
        if getattr(self.args, "report_to", None) is not None:
            self.args.report_to = []

        trainer = OurTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )
        trainer.add_callback(MetricsRecorder(self, train_samples, eval_samples, logs_dir, run_tag))
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=last_checkpoint) 

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    eps_tag = f"-eps{args.zo_eps:g}"
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + eps_tag + customized_tag


def get_eval_split_samples(task, num_eval=None, seed=0):
    eval_splits = task.get_eval_splits() if hasattr(task, "get_eval_splits") else {"valid": task.valid_samples}
    sampled_splits = {}
    for split_name in eval_splits:
        if num_eval is not None and num_eval > 0:
            sampled_splits[split_name] = task.sample_subset(data_split=split_name, seed=seed, num=num_eval)
        else:
            sampled_splits[split_name] = eval_splits[split_name]
    return sampled_splits


def evaluate_across_splits(framework, train_samples, eval_split_samples, primary_split_name):
    metrics = {}
    for split_name, split_samples in eval_split_samples.items():
        split_metrics = framework.evaluate(train_samples, split_samples)
        if split_name == primary_split_name:
            metrics.update(split_metrics)
        else:
            for metric_name, metric_val in split_metrics.items():
                metrics[f"{split_name}_{metric_name}"] = metric_val
    return metrics


def main():
    args = parse_args()

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,
        seed=args.train_set_seed,
    )
    # Initialize trainer and load model
    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            eval_split_samples = get_eval_split_samples(task, num_eval=args.num_eval, seed=train_set_seed)
            primary_eval_split = "valid" if "valid" in eval_split_samples else list(eval_split_samples.keys())[0]
            eval_samples = eval_split_samples[primary_eval_split]

            if args.trainer != "none":
                if args.num_dev is not None and args.num_dev > 0:
                    # 用户指定了数量
                    dev_samples = train_samples[-args.num_dev:]
                    train_samples = train_samples[:-args.num_dev]
                elif args.num_dev == -1:
                    # 默认切出 1/4 数据作为 dev
                    split_idx = int(0.75 * len(train_samples))
                    dev_samples = train_samples[split_idx:]
                    train_samples = train_samples[:split_idx]
                else:
                    # 不切 dev
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                if not args.no_eval:
                    # No in-context learning if there is training
                    metrics = evaluate_across_splits(
                        framework=framework,
                        train_samples=[],
                        eval_split_samples=eval_split_samples,
                        primary_split_name=primary_eval_split,
                    )
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = evaluate_across_splits(
                    framework=framework,
                    train_samples=train_samples,
                    eval_split_samples=eval_split_samples,
                    primary_split_name=primary_eval_split,
                )

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None and args.num_eval > 0:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)

if __name__ == "__main__": 
    main()
