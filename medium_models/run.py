"""Finetuning the library models for sequence classification on GLUE."""

import csv
import dataclasses
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Union, List

try:
    from accelerate.utils import ParallelismConfig
except Exception:
    class ParallelismConfig:
        pass

import torch
import torch.nn.functional as F

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, PreTrainedTokenizerBase
from src.modeling_roberta import RobertaConfig
from src.modeling_opt import OPTConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.linearhead_trainer import LinearHeadTrainer
from src.dataset import FewShotDataset, OurInputFeatures
from src.data_utils import resolve_and_prepare_data
from src.models import MODEL_TYPES, resize_token_type_embeddings, convert_opt_model
from src.quzo import (
    exact_gptq_available,
    normalize_quantization_algorithm,
    quantization_algorithm_label,
    quantize_model_in_place,
    validate_quzo_bits,
)
from src.sparse_mezo import (
    normalize_sparse_mask_strategy,
    normalize_sparse_scope,
    sparse_mezo_enabled,
    validate_sparse_ratio,
)
from src.h_schedules import H_SCHEDULE_CHOICES, parse_h_grid
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_metadata import collect_run_metadata, update_model_run_metadata, write_run_metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRAINING_ARGS_NAME = "training_args.bin"

MEDIUM_TASK_NAME_ALIASES = {
    "sst2": "sst-2",
}


def normalize_medium_task_name(task_name: str) -> str:
    task_name = str(task_name).strip()
    lowered = task_name.lower()
    if lowered in MEDIUM_TASK_NAME_ALIASES:
        return MEDIUM_TASK_NAME_ALIASES[lowered]
    if lowered in processors_mapping:
        return lowered
    return task_name


def _normalize_for_json(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _normalize_for_json(value.item())
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    return str(value)


def _read_json_if_exists(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _normalize_for_json(json.load(f))
    except Exception as exc:
        logger.warning("Failed to read JSON artifact %s: %s", path, exc)
        return None


def _read_last_csv_row(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        last_row = None
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if any(value not in (None, "") for value in row.values()):
                    last_row = row
        return _normalize_for_json(last_row)
    except Exception as exc:
        logger.warning("Failed to read CSV artifact %s: %s", path, exc)
        return None


def _read_last_jsonl_row(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        last_row = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_row = json.loads(line)
        return _normalize_for_json(last_row)
    except Exception as exc:
        logger.warning("Failed to read JSONL artifact %s: %s", path, exc)
        return None


def _save_model_with_shared_tensor_fallback(trainer, output_dir: str):
    try:
        trainer.save_model(output_dir)
        return
    except RuntimeError as exc:
        message = str(exc)
        if "shared tensors" not in message and "mismatching the transformers base configuration" not in message:
            raise

    logger.warning(
        "Default model save failed due to tied/shared tensors; retrying with safe_serialization=False."
    )
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = trainer.model
    if hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(output_dir, safe_serialization=False)
    else:
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    torch.save(trainer.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def maybe_convert_model_to_torchao_float8_training(model, enabled: bool):
    total_linear_layers = sum(1 for _, module in model.named_modules() if isinstance(module, torch.nn.Linear))
    if not bool(enabled):
        update_model_run_metadata(
            model,
            fp8_mode="none",
            converted_linear_layers=0,
            total_linear_layers=total_linear_layers,
            skipped_layer_names=[],
        )
        return model
    try:
        from torchao.float8 import convert_to_float8_training
        from torchao.float8.config import Float8LinearConfig
    except Exception as exc:
        raise RuntimeError(
            "FP8 training requested via --use_torchao_float8, but torchao.float8 is unavailable."
        ) from exc
    emulate = False
    capability = None
    if torch.cuda.is_available():
        device_index = 0
        if torch.cuda.device_count() > 0:
            capability = torch.cuda.get_device_capability(device_index)
            emulate = capability < (9, 0)
    config = Float8LinearConfig(emulate=emulate)
    incompatible_linear_modules = {
        fqn: module
        for fqn, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ((module.in_features % 16) != 0 or (module.out_features % 16) != 0)
    }
    incompatible_linear_fqns = set(incompatible_linear_modules.keys())

    def module_filter_fn(module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        return fqn not in incompatible_linear_fqns

    convert_to_float8_training(model, module_filter_fn=module_filter_fn, config=config)
    for fqn, module in incompatible_linear_modules.items():
        if "." in fqn:
            parent_fqn, child_name = fqn.rsplit(".", 1)
            parent_module = model.get_submodule(parent_fqn)
        else:
            parent_module = model
            child_name = fqn
        setattr(parent_module, child_name, module)
    mode = "emulated" if emulate else "native"
    if capability is None:
        logger.info("[fp8] converted nn.Linear modules to torchao Float8Linear for training (%s mode)", mode)
    else:
        logger.info(
            "[fp8] converted nn.Linear modules to torchao Float8Linear for training (%s mode, compute_capability=%s.%s)",
            mode,
            capability[0],
            capability[1],
        )
    if incompatible_linear_fqns:
        skipped = ", ".join(sorted(incompatible_linear_fqns)[:8])
        suffix = " ..." if len(incompatible_linear_fqns) > 8 else ""
        logger.info(
            "[fp8] kept %d nn.Linear modules in high precision because their dimensions are not divisible by 16: %s%s",
            len(incompatible_linear_fqns),
            skipped,
            suffix,
        )
    update_model_run_metadata(
        model,
        fp8_mode=mode,
        converted_linear_layers=max(0, total_linear_layers - len(incompatible_linear_fqns)),
        total_linear_layers=total_linear_layers,
        skipped_layer_names=sorted(incompatible_linear_fqns),
    )
    return model


def infer_medium_run_zo_method(training_args) -> str:
    if bool(getattr(training_args, "zero_order_optim", False)):
        zo_method = str(getattr(training_args, "zo_method", "") or "").strip().lower()
        if zo_method:
            return zo_method
        if sparse_mezo_enabled(getattr(training_args, "sparse_ratio", 1.0)):
            return "sparse_mezo"
        if bool(getattr(training_args, "zo_by_layer", False)):
            return "mezo_layerwise"
        zo_variant = str(getattr(training_args, "zo_variant", "") or "").lower()
        if zo_variant == "grad_norm":
            return "mezo_grad_norm"
        if zo_variant == "param_norm":
            return "mezo_param_norm"
        if bool(getattr(training_args, "efficient_zero_order", False)):
            return "mezo_efficient"
        if bool(getattr(training_args, "zero_order_use_trainer_optim", False)):
            optimizer = str(getattr(training_args, "optimizer", "") or "").lower()
            optimizer_variant = str(getattr(training_args, "optimizer_variant", "") or "").lower()
            if optimizer == "adam":
                return "mezo_adam"
            if optimizer_variant == "signgd":
                return "mezo_signgd"
        return "mezo"
    trainer_name = str(getattr(training_args, "trainer", "standard") or "standard").lower()
    return trainer_name


def normalize_zo_method_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "auto"}:
        return None
    aliases = {
        "vanilla": "mezo",
        "s-mezo": "sparse_mezo",
        "sparse": "sparse_mezo",
        "sparse-mezo": "sparse_mezo",
        "sparse_mezo": "sparse_mezo",
        "lozo-m": "lozo_m",
        "lozom": "lozo_m",
        "hessian": "hizoo",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"mezo", "sparse_mezo", "lozo", "lozo_m", "hizoo"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported --zo_method={value!r}. Expected one of {sorted(allowed)}.")
    return normalized


def normalize_zo_update_backend(value: str) -> str:
    normalized = str(value or "direct_int8").strip().lower()
    aliases = {
        "residual": "residual_grid",
        "int8_residual": "residual_grid",
        "int8_residual_accum": "residual_grid",
        "grid_residual": "residual_grid",
        "quzo_fp16_master": "fp16_master",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"direct_int8", "residual_grid", "fp16_master"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported --zo_update_backend={value!r}. Expected one of {sorted(allowed)}.")
    return normalized


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    l2_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use L2 loss (only makes a difference in standard FT)."}
    )
    use_task_word: bool = field(
        default=False,
        metadata={'help': 'uses the task words MLM logit for kernel computation'}
    )

    # LoRA arguments: only for BERT-type model
    apply_lora: bool = field(
        default=False,
        metadata={'help': 'use LoRA for finetuning'}
    )
    lora_alpha: int = field(
        default=None,
        metadata={'help': 'initialization scale for one of the low rank matrices in lora'}
    )
    lora_r: int = field(
        default=None,
        metadata={'help': 'inner rank for lora matrices'}
    )

    # Calibration
    sfc: bool = field(
        default=False,
        metadata={"help": "Whether to use surface form calibration."}
    )

    icl_sfc: bool = field(
        default=False,
        metadata={"help": "Use in-context learning demos in sfc."}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    sfc_prompt: str = field(
        default=None,
        metadata={"help": "SFC prompt"}
    )

    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    dataset_mode: str = field(
        default="auto",
        metadata={"help": "Data mode: auto, fewshot, or full."}
    )

    data_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory for materialized splits (default: infer from --data_dir, else data/k-shot-1k-test)."}
    )

    full_dev_ratio: float = field(
        default=0.1,
        metadata={"help": "For full mode: deterministic stratified validation ratio sampled from original train."}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    gpt3_demo_separator: str = field(
        default="\n\n\n",
        metadata={"help": "Separator between demonstrations"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: List[str] = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."},

    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training or at the."}
    )
    log_file: str = field(
        default='log'
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Compatibility flag for legacy launchers that still pass --overwrite_output_dir."}
    )

    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-parameter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    optimizer: str = field(
        default='adam',
        metadata={'help': 'choose sgd or adam. default is adam'}
    )
    optimizer_variant: str = field(
        default='',
        metadata={'help': 'define variants on optimizer: signgd'}
    )

    trainer: str = field(
        default="standard",
        metadata={"help": "Pick from {standard, kernel, linearhead}"}
    )
    from_linearhead: bool = field(
        default=False,
        metadata={"help": "Whether to initialize head with the linearhead solution. Works for both normal and kernel trainer."}
    )
    lp_early_stopping: bool = field(
        default=False,
        metadata={"help": "When on, increases the tolerance and lowers max_iter in scikit LogisticRegression solver to encourage early stopping."}
    )
    random_model_init: bool = field(
        default=False,
        metadata={'help': 'reinit the model randomly'}
    )
    sweep: bool = field(
        default=False,
        metadata={'help': 'configures the output directories to be informative when running W&B sweep'}
    )
    kernel_formula: str = field(
        default='sgd',
        metadata={"help": "choose kernel formula from {sgd, signgd, asymmetric_signgd}"}
    )
    kernel_solver: str = field(
        default="logistic",
        metadata={"help": "choose kernel solver from {lstsq, logistic, svr, svc, asym (only for asymmetric_signgd)}"}
    )
    load_kernels: str = field(
        default=None,
        metadata={'help': 'when specified, loads the kernels from the folder given here'}
    )
    overwrite_kernels: bool = field(
        default=False,
        metadata={'help': 'when specified, overwrites the kernels in the output_dir and computes them from scratch'}
    )

    exclude_embeddings: bool = field(
        default=False,
        metadata={"help": "Don't use embeddings for kernel computation "}
    )
    exclude_head: bool = field(
        default=False,
        metadata={"help": "Don't use head for kernel computation "}
    )
    only_biases: bool = field(
        default=False,
        metadata={"help": "Only use bias parameters for kernel computation for BitFit-style kernel"}
    )
    exclude_first_layers: int = field(
        default=-1,
        metadata={'help': 'excludes the first N layers from kernel computation'}
    )
    sync_embedding_layers: bool = field(
        default=False,
        metadata={'help': 'sync the input embedding to match output embedding (use with --exclude_first_layers)'}
    )

    kernel_regularization: float = field(
        default=0.0,
        metadata={"help": "Regularization constant for kernel"}
    )
    kernel_gamma: float = field(
        default=1.0,
        metadata={"help": "Gamma for asymmetric kernel solver"}
    )
    binary_classification: bool = field(
        default=False,
        metadata={"help": "If num_classes=2, convert two softmax logits to single sigmoid logit"}
    )
    adjust_for_init: bool = field(
        default=False,
        metadata={'help': 'when on, trains kernel on y-f0 and adds f0 at test time'}
    )
    f0_scaling: float = field(
        default=1.0,
        metadata={'help': 'adjust label scaling, might help with --adjust_for_init perf'}
    )
    zero_order_optim: bool = field(
        default=False,
        metadata={'help': 'when on, trains the model by zero-order optimization'}
    )
    zo_method: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Explicit zero-order method switch. Supported values: mezo, sparse_mezo, lozo, lozo_m, hizoo. When unset, medium_models keeps the existing auto-inference from sparse_ratio / zo_variant / efficient_zero_order.'
        }
    )
    zero_order_eps: float = field(
        default=1e-3,
        metadata={'help': 'eps for zero order optim'}
    )
    h_schedule: str = field(
        default="fixed",
        metadata={
            "help": "Schedule-only finite-difference radius baseline.",
            "choices": [
                "fixed",
                "spall_clip",
                "shamir_clip",
                "ji_sqrtk_clip",
                "ji_theory_clip",
                "pf_vrzo_clip",
            ],
        },
    )
    h_schedule_grid: str = field(
        default="",
        metadata={"help": "Optional comma/space-separated h grid. Empty means continuous clipping only."},
    )
    h_schedule_window_min: float = field(
        default=0.0,
        metadata={"help": "Optional lower clipping bound for schedule-only h. <=0 disables the lower bound."},
    )
    h_schedule_window_max: float = field(
        default=0.0,
        metadata={"help": "Optional upper clipping bound for schedule-only h. <=0 disables the upper bound."},
    )
    h_schedule_h0: float = field(
        default=0.0,
        metadata={"help": "Optional initial/effective radius for schedule-only h. <=0 falls back to window_max, then zero_order_eps."},
    )
    h_schedule_gamma: float = field(
        default=0.101,
        metadata={"help": "Decay exponent for h_schedule=spall_clip."},
    )
    h_schedule_total_steps: int = field(
        default=0,
        metadata={"help": "Horizon T for static horizon-based h schedules. <=0 falls back to max_steps where applicable."},
    )
    h_schedule_d_eff: float = field(
        default=1.0,
        metadata={"help": "Effective dimension used by h_schedule formulas that require d."},
    )
    h_schedule_n_eff: float = field(
        default=1.0,
        metadata={"help": "Effective sample count reserved for schedule baselines that require n."},
    )
    h_schedule_lipschitz_l: float = field(
        default=0.0,
        metadata={"help": "Lipschitz L used by h_schedule=ji_theory_clip. Must be >0 for that schedule."},
    )
    h_schedule_c_delta: float = field(
        default=1.0,
        metadata={"help": "Constant multiplier for h_schedule=shamir_clip."},
    )
    h_schedule_log_csv: bool = field(
        default=True,
        metadata={"help": "Log resolved non-fixed h schedule values to output_dir/metrics_logs/h_schedule.csv."},
    )
    zo_use_true_directional_derivative: bool = field(
        default=False,
        metadata={
            'help': 'For ZO/MeZO: replace finite-difference (loss1-loss2)/(2*eps) with the true directional derivative <grad, z> while keeping the same z direction. Intended for fixed-h ablations; requires one backward (autograd.grad) per z sample.'
        }
    )
    zo_two_point_precision: str = field(
        default="fp32",
        metadata={
            'help': 'Precision used ONLY for the two finite-difference function evaluations (loss1/loss2) in ZO. Choices: fp32, fp16, bf16.'
        }
    )
    precision_mode: str = field(
        default="",
        metadata={"help": "Probe-window convenience alias. Choices: fp32, fp16, bf16, int8. Maps to the existing ZO precision/quantization flags."}
    )
    quant_bits: Optional[int] = field(
        default=None,
        metadata={"help": "Probe-window convenience alias for --zo_quantization_bits."}
    )
    zo_quantization_bits: int = field(
        default=32,
        metadata={
            "help": "ZO-side method switch for medium_models. 32 keeps plain MeZO. 16 keeps the repo's FP16 MeZO convention. 8/4 use the QuZO perturbation/update path. medium_models has no separate load_int8 path."
        }
    )
    zo_quantization: Optional[str] = field(
        default=None,
        metadata={
            "help": "String alias for the same ZO-side method switch. Supported values: fp32/off/none, fp16, int8, int4. Overrides --zo_quantization_bits when set. In medium_models, int8 means QuZO int8, not model-loading int8."
        }
    )
    quantization_algorithm: str = field(
        default="per_tensor_symmetric",
        metadata={
            "help": "Low-bit fake-quant algorithm for QuZO. Supported local values: per_tensor_symmetric, groupwise_int8_block256. Passing gptq records an explicit fallback to groupwise_int8_block256 because exact GPTQ is not implemented here."
        }
    )
    quantization_group_size: int = field(
        default=0,
        metadata={"help": "Group size for groupwise low-bit quantization. For the GPTQ-256 fallback use 256."}
    )
    quantization_block_size: int = field(
        default=0,
        metadata={"help": "Alias for --quantization_group_size used in block/group-size experiment manifests."}
    )
    quantization_calibration_samples: int = field(
        default=0,
        metadata={"help": "Calibration sample count. Exact GPTQ is unavailable in this code path, so this is logged only."}
    )
    zo_update_backend: str = field(
        default="direct_int8",
        metadata={"help": "QuZO low-bit update backend. Choices: direct_int8, residual_grid, fp16_master."}
    )
    residual_dtype: str = field(
        default="fp32",
        metadata={"help": "Residual buffer dtype for --zo_update_backend residual_grid. Choices: fp16, bf16, fp32."}
    )
    residual_commit_mode: str = field(
        default="round",
        metadata={"help": "Residual-grid commit rule. Choices: round, floor, stochastic."}
    )
    residual_max_code_step: int = field(
        default=0,
        metadata={"help": "Max absolute INT code movement per coordinate per optimizer step for residual_grid. 0 means unlimited."}
    )
    residual_commit_threshold: float = field(
        default=0.0,
        metadata={"help": "Only commit residual_grid candidate code moves where abs(acc / scale) is at least this threshold. <=0 disables thresholding."}
    )
    residual_commit_select: str = field(
        default="all",
        metadata={"help": "Residual-grid commit selection. Choices: all, top_abs_acc, norm_budget."}
    )
    residual_target_active_frac: float = field(
        default=0.0,
        metadata={"help": "Optional target active fraction for residual_grid top_abs_acc or norm_budget selection. <=0 disables top-k selection."}
    )
    residual_actual_norm_ratio_cap: float = field(
        default=0.0,
        metadata={"help": "Optional norm-budget cap for residual_grid: ||actual_selected|| <= cap * ||reference||. <=0 disables norm budget."}
    )
    residual_budget_reference: str = field(
        default="acc",
        metadata={"help": "Reference norm for residual_grid norm_budget. Choices: acc, delta."}
    )
    residual_decay: float = field(
        default=1.0,
        metadata={"help": "Residual carry decay for residual_grid. 1.0 keeps exact error-feedback semantics; values <1 are stale-residual ablations."}
    )
    residual_scale_mode: str = field(
        default="tensor",
        metadata={"help": "Residual-grid scale mode. Choices: tensor, channel, block. Block uses an expanded per-block scale tensor for correctness."}
    )
    residual_block_size: int = field(
        default=0,
        metadata={"help": "Block size for residual_scale_mode=block."}
    )
    int8_freeze_scale: bool = field(
        default=True,
        metadata={"help": "Freeze the low-bit parameter scale for residual_grid training. Default true for diagnostic residual semantics."}
    )
    int8_scale_floor: float = field(
        default=0.0,
        metadata={"help": "Optional minimum scale for residual_grid quantization diagnostics. 0 disables flooring."}
    )
    log_update_stats_every: int = field(
        default=0,
        metadata={"help": "Log quantized update diagnostics every N optimizer steps. <=0 logs only when save_update_stats_jsonl is set."}
    )
    save_update_stats_jsonl: str = field(
        default="",
        metadata={"help": "Optional path for JSONL quantized update diagnostics. Relative paths are resolved under output_dir."}
    )
    zo_update_norm_clip: float = field(
        default=0.0,
        metadata={"help": "Optional global intended-update norm clip for direct ZO updates. <=0 disables clipping."}
    )
    zo_scalar_clip: float = field(
        default=0.0,
        metadata={"help": "Optional clip for alpha=learning_rate*projected_grad in direct ZO updates. <=0 disables clipping."}
    )
    debug_residual_grid_consistency: bool = field(
        default=False,
        metadata={"help": "Run residual_grid scale/grid/equation diagnostics on one small batch and exit before training."}
    )
    debug_layer_regex: str = field(
        default="",
        metadata={"help": "Optional regex selecting layers for residual_grid one-step equation diagnostics."}
    )
    debug_num_tensors: int = field(
        default=5,
        metadata={"help": "Number of trainable tensors to include in residual_grid one-step diagnostics."}
    )
    debug_dump_tensor_stats: bool = field(
        default=False,
        metadata={"help": "Write per-tensor weight min/max JSONL during residual_grid consistency diagnostics."}
    )
    debug_save_dir: str = field(
        default="",
        metadata={"help": "Output directory for residual_grid consistency diagnostics. Relative paths resolve under output_dir."}
    )
    probe_window_diagnostics_only: bool = field(
        default=False,
        metadata={"help": "Run signal-only h-window diagnostics and exit before training updates."}
    )
    probe_window_h_list: str = field(
        default="",
        metadata={"help": "Comma/space-separated h values for --probe_window_diagnostics_only. Defaults to --zo_h/--zero_order_eps."}
    )
    num_probe_batches: int = field(
        default=1,
        metadata={"help": "Number of train batches used by probe-window diagnostics."}
    )
    compute_true_grad_directional: bool = field(
        default=True,
        metadata={"help": "Compute one true gradient per probe batch and log grad-dot-direction diagnostics."}
    )
    checkpoint_probe_steps: str = field(
        default="",
        metadata={"help": "Comma/space-separated global steps at which to run signal-only probe-window diagnostics during training, e.g. 0,300,1000,2000. Empty disables it."}
    )
    checkpoint_probe_num_directions: int = field(
        default=16,
        metadata={"help": "Number of directions for training checkpoint probe diagnostics."}
    )
    checkpoint_probe_num_batches: int = field(
        default=1,
        metadata={"help": "Number of train/eval batches for training checkpoint probe diagnostics."}
    )
    checkpoint_probe_compute_true_grad: bool = field(
        default=True,
        metadata={"help": "Whether training checkpoint probe diagnostics compute true grad-dot-direction values."}
    )
    save_checkpoint_probe_stats_jsonl: str = field(
        default="checkpoint_probe_stats.jsonl",
        metadata={"help": "Output JSONL filename/path for training checkpoint probe diagnostics."}
    )
    main_save_checkpoints: bool = field(
        default=False,
        metadata={"help": "If true, the custom MeZO trainer writes restartable checkpoints under output_dir/checkpoints."}
    )
    main_checkpoint_steps: int = field(
        default=0,
        metadata={"help": "Checkpoint interval in optimizer steps for --main_save_checkpoints. <=0 falls back to save_steps."}
    )
    main_save_final_checkpoint: bool = field(
        default=True,
        metadata={"help": "When --main_save_checkpoints is true, save output_dir/checkpoints/final at training end."}
    )
    main_save_best_acc_checkpoint: bool = field(
        default=True,
        metadata={"help": "When --main_save_checkpoints is true, maintain output_dir/checkpoints/best_acc."}
    )
    main_save_best_loss_checkpoint: bool = field(
        default=True,
        metadata={"help": "When --main_save_checkpoints is true, maintain output_dir/checkpoints/best_loss."}
    )
    direction_type: str = field(
        default="dense",
        metadata={"help": "Probe-window direction type. Choices: dense, sparse."}
    )
    sparse_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Alias for --zo_direction_sparse_rate used by probe-window scripts."}
    )
    sparse_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Alias for --zo_direction_sparse_mode used by probe-window scripts."}
    )
    sparse_rescale: Optional[str] = field(
        default=None,
        metadata={"help": "Alias for --zo_sparse_rescale used by probe-window scripts."}
    )
    zo_direction_sparse_rate: float = field(
        default=1.0,
        metadata={"help": "Random ZO direction active probability/fraction. 1.0 disables direction sparsity."}
    )
    zo_direction_sparse_mode: str = field(
        default="none",
        metadata={"help": "Random ZO direction sparsity mode. Choices: none, exact_random, bernoulli."}
    )
    zo_sparse_rescale: str = field(
        default="none",
        metadata={"help": "Sparse direction rescaling. Choices: none, inv_sqrt_p."}
    )
    zo_sparse_per_layer_exact: bool = field(
        default=True,
        metadata={"help": "For exact_random direction sparsity, enforce exact round(p*numel) active coordinates per trainable tensor. False enforces the exact fraction globally."}
    )
    zo_h: Optional[float] = field(
        default=None,
        metadata={"help": "Alias for --zero_order_eps used by sparse h sweeps."}
    )
    probe_diagnostics_only: bool = field(
        default=False,
        metadata={"help": "Signal-only probe mode hint. Use with LR=0/short max_steps; keeps existing training path but records probe diagnostics."}
    )
    num_probe_directions: Optional[int] = field(
        default=None,
        metadata={"help": "Alias for --zo_probe_num_seeds used by probe-only sweeps."}
    )
    save_probe_stats_jsonl: str = field(
        default="",
        metadata={"help": "Optional JSONL path for directional probe diagnostics. Relative paths are resolved under output_dir."}
    )
    sparse_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Sparse MeZO target active fraction per trainable tensor. 1.0 keeps vanilla MeZO behavior with no masking."
        }
    )
    sparse_mask_strategy: str = field(
        default="percentile_per_layer",
        metadata={
            "help": "Sparse MeZO mask rule. Current default percentile_per_layer keeps the lowest-|param| sparse_ratio fraction active in each trainable tensor."
        }
    )
    sparse_scope: str = field(
        default="trainable_only",
        metadata={"help": "Sparse MeZO scope. Current implementation only supports trainable_only."}
    )
    sparse_log_active_fraction: bool = field(
        default=True,
        metadata={"help": "If true, log the realized Sparse MeZO active-parameter fraction during training."}
    )
    sparse_mask_refresh_steps: int = field(
        default=100,
        metadata={
            "help": "Sparse MeZO mask refresh cadence. 0 freezes the initial mask for the full run, 1 refreshes every optimizer step, and N>1 refreshes every N steps."
        }
    )
    lozo_rank: int = field(
        default=2,
        metadata={"help": "LOZO low-rank factor rank r."}
    )
    lozo_step_interval: int = field(
        default=50,
        metadata={"help": "LOZO step interval nu for refreshing the right low-rank factor."}
    )
    lozo_beta1: float = field(
        default=0.9,
        metadata={"help": "LOZO-M momentum coefficient beta1."}
    )
    hizoo_hessian_smooth_type: str = field(
        default="constant0",
        metadata={"help": "HiZOO diagonal-Hessian smoothing schedule. Examples: constant0, constant1e-8, constant1e-6."}
    )
    use_torchao_float8: bool = field(
        default=False,
        metadata={"help": "Swap nn.Linear modules with torchao Float8Linear for training."}
    )
    zo_probe_every: int = field(
        default=0,
        metadata={"help": "Run directional-derivative probe every N training steps. <=0 disables probing."}
    )
    zo_probe_num_seeds: int = field(
        default=16,
        metadata={"help": "Number of random direction seeds used in each directional-derivative probe."}
    )
    zo_probe_log_csv: bool = field(
        default=True,
        metadata={"help": "If true, append directional-derivative probe metrics to output_dir/zo_directional_probe.csv."}
    )
    random_prediction_guard_enabled: bool = field(
        default=False,
        metadata={
            "help": "If true, abort medium-model training early when the first post-threshold eval still looks random or clearly diverged."
        }
    )
    random_prediction_guard_step: int = field(
        default=1000,
        metadata={"help": "First global_step at which the random-prediction early-skip guard is allowed to trigger."}
    )
    random_prediction_guard_acc_tolerance: float = field(
        default=0.05,
        metadata={"help": "Accuracy tolerance above chance used by the random-prediction early-skip guard."}
    )
    random_prediction_guard_loss_tolerance: float = field(
        default=0.03,
        metadata={"help": "Allowed eval-loss slack around the uniform random baseline before the random-prediction early-skip guard fires."}
    )
    random_prediction_guard_bad_loss_excess: float = field(
        default=0.5,
        metadata={"help": "If eval loss exceeds log(num_labels) by at least this margin at the guard step, treat it as clearly diverged and abort early."}
    )
    random_prediction_guard_recent_evals: int = field(
        default=2,
        metadata={"help": "Number of recent evals that must remain random-like before the random-prediction guard can abort."}
    )
    random_prediction_guard_min_loss_drop: float = field(
        default=0.05,
        metadata={"help": "Minimum best-loss improvement over the recent eval window required to treat a random-looking run as still making progress."}
    )
    random_prediction_guard_min_acc_gain: float = field(
        default=0.02,
        metadata={"help": "Minimum best-accuracy improvement over the recent eval window required to treat a random-looking run as still making progress."}
    )
    zo_probe_health_guard_enabled: bool = field(
        default=False,
        metadata={"help": "If true, abort after repeated probe steps produce no finite directional-derivative pairs."}
    )
    zo_probe_health_guard_step: int = field(
        default=2000,
        metadata={"help": "First global_step at which the probe-health guard is allowed to abort training."}
    )
    zo_probe_health_guard_max_bad_probes: int = field(
        default=3,
        metadata={"help": "Maximum consecutive bad probe steps allowed before the probe-health guard aborts."}
    )
    measure_perf_tail: bool = field(
        default=True,
        metadata={"help": "If true, measure wallclock/step, samples/sec, and max GPU memory once over the final tail window of optimizer steps."}
    )
    measure_perf_tail_window_steps: int = field(
        default=10,
        metadata={"help": "Number of final optimizer steps used for the one-shot tail performance snapshot when measure_perf_tail is enabled."}
    )
    prob_as_feature: bool = field(
        default=False,
        metadata={'help': 'in linear head, use log prob as feature'}
    )
    zero_order_use_trainer_optim: bool = field(
        default=False,
        metadata={"help": "Use trainer optimizer for zero order optimization"}
    )
    efficient_zero_order: bool = field(
        default=False,
        metadata={"help": "Efficient zero-order: resample noise vectors instead of saving them. enable different model loading using --hf_inference_model"}
    )
    hf_inference_model: bool = field(
        default=False,
        metadata={"help": "loads the HF model in inference mode across many GPUs. incompatible with --zero_order_use_trainer_optim."}
    )
    efficient_zero_order_fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16 for efficient zero order"}
    )
    zero_order_sample_scheduler: str = field(
        default=None,
        metadata={"help": "Have a sample scheduler. None, 'linear', 'power', or 'constant."}
    )
    scale_lr_with_samples: bool = field(
        default=False,
        metadata={"help": "Scales the LR proportionally to the number of z samples. --learning_rate will be the LR for one z sample."}
    )
    zero_order_sample: int = field(
        default=1,
        metadata={"help": "Sample times for zero-order estimate. If scheduler is 'linear', this number is the max sample number."}
    )
    zero_order_clip_grad: bool = field(
        default=False,
        metadata={"help": "Clip the norm of the gradient for zero order (only when using trainer optimizer)"}
    )

    # MeZO variants
    zo_by_layer: bool = field(
        default=False,
        metadata={"help": "For ZO: estimate the gradients on each layer individually, scales number of forward passes per grad step by a factor of L"}
    )
    zo_variant: str = field(
        default=None,
        metadata={"help": "Choose the MeZO variant: grad_norm or param_norm (see documentation)"}
    )
    use_zo_grad_est: bool = field(
        default=False,
        metadata={"help": "Use zero-order estimate of the gradient for zo variants"}
    )
    recompute_norms: bool = field(
        default=False,
        metadata={'help': 'Recompute the grad or parameter norm (whichever was specified as --zo_variant) at the start of each epoch.'}
    )
    scale_norm_by_num_params: bool = field(
        default=False,
        metadata={'help': 'Scale grad or param norm by 1 / sqrt(num params)'}
    )
    norm_running_update: bool = field(
        default=False,
        metadata={"help": "When performing --zo_by_layer and using --zo_variant 'grad_norm', update the layer grad norms as they are recomputed at each step"}
    )
    change_grad_estimate: bool = field(
        default=False,
        metadata={"help": "Changes the expectation of the ZO gradient estimate according to zo_variant, instead of just modifying the variance"}
    )

    # prefix tuning hyperparameters
    prefix_tuning: bool = field(
        default=False,
        metadata={"help": "Prefix tuning"}
    )
    num_prefix: int = field(
        default=10,
        metadata={"help": "How many prefix tokens to use"}
    )
    no_reparam: bool = field(
        default=False,
        metadata={"help": "No reparameterization trick"}
    )
    prefix_init_by_real_act: bool = field(
        default=False,
        metadata={"help": "For no_reparam case, randomly sample words and take their actual key/value pairs as initialization"}
    )
    layer_wise_optim: bool = field(
        default=False,
        metadata={'help': 'Optimize layer-by-layer (only for prefix + ZO)'}
    )

    max_zo_forward_steps: int = field(
        default=0,
        metadata={'help': 'Stop at this number of ZO forward steps. The trainer will take whichever is reached first, max_steps or max_zo_forward_steps.'}
    )

    untie_emb: bool = field(
        default=False,
        metadata={"help": "Untie embeddings from lm head. Only work for OPT!!"}
    )
    tie_emb: bool = field(
        default=False,
        metadata={"help": "Tie embeddings from lm head. Only work for RoBERTa!!"}
    )

    optimize_acc: bool = field(
        default=False,
        metadata={"help": "Maximize accuracy instead of minimizing loss"}
    )

    ## hessian trainer args
    num_hvp_vecs: int = field(
        default=128,
        metadata={"help": "Number of vectors to use to estimate HVPs"}
    )
    mc_tol: float = field(
        default=0.1,
        metadata={"help": "Tolerance (on std dev) after which MC estimate is deemed converged"}
    )

    head_tuning: bool = field(
        default=False,
        metadata={"help": "Tune the head only"}
    )

    use_adaptive_h: bool = field(
        default=True,
        metadata={"help": "Use adaptive finite difference step size h (based on estimated epsilon_f and nu_3) instead of fixed zero_order_eps"}
    )
    update_noise_every: int = field(
        default=1000,
        metadata={"help": "Number of steps between re-estimating epsilon_f and nu3 during training"}
    )
    enable_additive_h_estimation: bool = field(
        default=False,
        metadata={"help": "Also compute the legacy additive error estimation h even when it is not the active training step size."}
    )
    enable_two_point_h_estimation: bool = field(
        default=False,
        metadata={"help": "Compute the two-point simple estimation h based on Delta/G/L."}
    )
    h_estimation_active_source: str = field(
        default="auto",
        metadata={"help": "Which h drives actual training perturbations: auto, fixed, additive, or two_point."}
    )
    initial_h: float = field(
        default=1e-3,
        metadata={"help": "Initial h value used by both additive and two-point estimators."}
    )
    adaptive_h_ema_beta: float = field(
        default=0.1,
        metadata={"help": "Log-space EMA beta for the legacy additive error estimation h update."}
    )
    adaptive_h_estimate_num_batches: int = field(
        default=4,
        metadata={"help": "Number of batches used by the legacy additive error estimation refresh."}
    )
    adaptive_h_estimate_num_directions: int = field(
        default=3,
        metadata={"help": "Number of random directions used by the legacy additive error estimation refresh."}
    )
    adaptive_h_estimate_reduce: str = field(
        default="mean",
        metadata={"help": "Reduction used for the legacy additive error estimation h aggregation: mean or median."}
    )
    adaptive_h_probe_buffer_size: int = field(
        default=64,
        metadata={"help": "Rolling probe buffer size shared by the additive and two-point h estimators."}
    )
    adaptive_h_min: float = field(
        default=1e-5,
        metadata={"help": "Minimum legacy additive error estimation h."}
    )
    adaptive_h_max: float = field(
        default=0.5,
        metadata={"help": "Maximum legacy additive error estimation h."}
    )
    h_trunc_alpha: float = field(
        default=1.0,
        metadata={"help": "Legacy additive error estimation truncation scaling alpha."}
    )
    dh_h_growth: float = field(
        default=2.0,
        metadata={"help": "Growth factor used by the additive error estimation h search."}
    )
    dh_max_trials: int = field(
        default=10,
        metadata={"help": "Maximum h search trials for the additive error estimation nu3 routine."}
    )
    dh_test_h: float = field(
        default=1e-2,
        metadata={"help": "Diagnostic h used by the additive error estimation test log."}
    )
    nu3_retry: int = field(
        default=3,
        metadata={"help": "Retry count when the additive error estimation nu3 finite difference hits a numerical floor."}
    )
    two_point_h_refresh_every: int = field(
        default=100,
        metadata={"help": "Number of steps between two-point simple estimation h refreshes."}
    )
    two_point_h_window_g: int = field(
        default=5,
        metadata={"help": "Sliding-window size for G in the two-point simple estimation h scheduler."}
    )
    two_point_h_window_l: int = field(
        default=5,
        metadata={"help": "Sliding-window size for L in the two-point simple estimation h scheduler."}
    )
    two_point_h_window_delta: int = field(
        default=3,
        metadata={"help": "Sliding-window size for Delta in the two-point simple estimation h scheduler."}
    )
    two_point_h_num_directions_g: int = field(
        default=4,
        metadata={"help": "Number of directions used for the G probe in the two-point simple estimation h scheduler."}
    )
    two_point_h_num_directions_l: int = field(
        default=4,
        metadata={"help": "Number of directions used for the L probe in the two-point simple estimation h scheduler."}
    )
    two_point_h_beta: float = field(
        default=0.5,
        metadata={"help": "Log-space EMA beta for the two-point simple estimation h update."}
    )
    two_point_h_min: float = field(
        default=1e-5,
        metadata={"help": "Minimum two-point simple estimation h."}
    )
    two_point_h_max: float = field(
        default=0.5,
        metadata={"help": "Maximum two-point simple estimation h."}
    )
    two_point_h_q_l: float = field(
        default=0.5,
        metadata={"help": "Quantile used to aggregate raw L probes in the two-point simple estimation h scheduler."}
    )
    two_point_h_eps_num: float = field(
        default=1e-12,
        metadata={"help": "Small positive constant used when normalizing the two-point simple estimation curvature probe."}
    )
    two_point_h_c2: float = field(
        default=1.0,
        metadata={"help": "c2 constant used to initialize h2 in the two-point simple estimation curvature probe."}
    )
    two_point_h_delta_sample_size: int = field(
        default=4096,
        metadata={"help": "Number of parameter coordinates sampled when estimating Delta on the fp16 path."}
    )
    two_point_h_fixed_probe_batch: bool = field(
        default=True,
        metadata={"help": "Reuse a fixed probe batch for the two-point simple estimation h scheduler when possible."}
    )
    two_point_h_log_csv: bool = field(
        default=True,
        metadata={"help": "If true, append h_additive / h_two_point logs to output_dir/h_estimation.csv."}
    )

    # 是否使用每层的 c 值（cs）对差分步长 h / 扰动进行缩放；
    # 说明：我们的新方法默认不需要分层缩放，因此默认 False。
    # 当设为 True 时，会在 Trainer 内启用按层缩放逻辑（见 use_c_scale 开关）。
    use_c_scale: bool = field(
        default=False,
        metadata={"help": "Whether to use per-layer c (cs) to scale the finite-difference step h / perturbations; default False to match the new method"}
    )
    # Reproducibility / sampling
    data_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Seed for DataLoader shuffling (decoupled from MeZO perturbation RNG). If None, defaults to --seed."}
    )

    # Override HF default (False) to match standard SGD/MeZO assumption of random minibatches.
    dataloader_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataloader (RandomSampler). Recommended True for training."}
    )


@dataclass
class MyDataCollatorWithPadding:
    """
    Implements padding for LM-BFF inputs.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        mask_pos = []
        standard_features = []
        if features[0].sfc_input_ids is not None:
            sfc_batch = self.__call__([OurInputFeatures(input_ids=x.sfc_input_ids, attention_mask=x.sfc_attention_mask, mask_pos=x.sfc_mask_pos) for x in features])

        for item in features:
            standard_item = {}
            for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
                if getattr(item, field) is not None:
                    standard_item[field] = getattr(item, field)
            standard_features.append(standard_item)
            mask_pos.append(item.mask_pos)

        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if any(mask_pos):
            batch["mask_pos"] = torch.tensor(mask_pos)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        if features[0].sfc_input_ids is not None:
            batch["sfc_input_ids"] = sfc_batch["input_ids"]
            batch["sfc_attention_mask"] = sfc_batch["attention_mask"]
            batch["sfc_mask_pos"] = sfc_batch["mask_pos"]
        return batch


def main():
    # --- Ensure ParallelismConfig is visible to HfArgumentParser type-hints ---
    # Some HF versions annotate TrainingArguments with Optional["ParallelismConfig"].
    # get_type_hints() resolves this in the *current module* globals where our
    # DynamicTrainingArguments is defined (i.e., __main__). Even if we imported it
    # at top-level, make it explicit here to avoid any scope/order surprises.
    #import accelerate  # ensure same-env import
    #globals()['ParallelismConfig'] = getattr(
    #    accelerate.utils, 'ParallelismConfig', type('ParallelismConfig', (), {})
    #)
    # -------------------------------------------------------------------------
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.task_name = normalize_medium_task_name(getattr(data_args, "task_name", ""))
    training_args.h_schedule = str(getattr(training_args, "h_schedule", "fixed") or "fixed").strip().lower()
    if training_args.h_schedule not in H_SCHEDULE_CHOICES:
        raise ValueError(f"--h_schedule must be one of {sorted(H_SCHEDULE_CHOICES)}")
    training_args.h_schedule_grid = str(getattr(training_args, "h_schedule_grid", "") or "")
    parse_h_grid(training_args.h_schedule_grid)
    training_args.h_schedule_window_min = float(getattr(training_args, "h_schedule_window_min", 0.0) or 0.0)
    training_args.h_schedule_window_max = float(getattr(training_args, "h_schedule_window_max", 0.0) or 0.0)
    if training_args.h_schedule_window_min < 0.0:
        raise ValueError("--h_schedule_window_min must be >= 0")
    if training_args.h_schedule_window_max < 0.0:
        raise ValueError("--h_schedule_window_max must be >= 0")
    if (
        training_args.h_schedule_window_min > 0.0
        and training_args.h_schedule_window_max > 0.0
        and training_args.h_schedule_window_min > training_args.h_schedule_window_max
    ):
        raise ValueError("--h_schedule_window_min must be <= --h_schedule_window_max")
    training_args.h_schedule_h0 = float(getattr(training_args, "h_schedule_h0", 0.0) or 0.0)
    if training_args.h_schedule_h0 < 0.0:
        raise ValueError("--h_schedule_h0 must be >= 0")
    training_args.h_schedule_gamma = float(getattr(training_args, "h_schedule_gamma", 0.101))
    if training_args.h_schedule_gamma <= 0.0:
        raise ValueError("--h_schedule_gamma must be > 0")
    training_args.h_schedule_total_steps = int(getattr(training_args, "h_schedule_total_steps", 0) or 0)
    if training_args.h_schedule_total_steps < 0:
        raise ValueError("--h_schedule_total_steps must be >= 0")
    training_args.h_schedule_d_eff = float(getattr(training_args, "h_schedule_d_eff", 1.0))
    if training_args.h_schedule_d_eff <= 0.0:
        raise ValueError("--h_schedule_d_eff must be > 0")
    training_args.h_schedule_n_eff = float(getattr(training_args, "h_schedule_n_eff", 1.0))
    if training_args.h_schedule_n_eff <= 0.0:
        raise ValueError("--h_schedule_n_eff must be > 0")
    training_args.h_schedule_lipschitz_l = float(getattr(training_args, "h_schedule_lipschitz_l", 0.0) or 0.0)
    if training_args.h_schedule == "ji_theory_clip" and training_args.h_schedule_lipschitz_l <= 0.0:
        raise ValueError("--h_schedule_lipschitz_l must be > 0 for --h_schedule ji_theory_clip")
    training_args.h_schedule_c_delta = float(getattr(training_args, "h_schedule_c_delta", 1.0))
    if training_args.h_schedule_c_delta <= 0.0:
        raise ValueError("--h_schedule_c_delta must be > 0")
    precision_mode = str(getattr(training_args, "precision_mode", "") or "").strip().lower()
    if precision_mode:
        if precision_mode not in {"fp32", "fp16", "bf16", "int8"}:
            raise ValueError("--precision_mode must be one of fp32, fp16, bf16, int8")
        if precision_mode == "int8":
            training_args.zo_quantization = "int8"
            if str(getattr(training_args, "zo_two_point_precision", "fp32")).lower() == "fp32":
                training_args.zo_two_point_precision = "fp16"
        elif precision_mode == "fp16":
            training_args.zo_quantization = "fp16"
            training_args.zo_two_point_precision = "fp16"
        elif precision_mode == "bf16":
            training_args.zo_quantization = "fp32"
            training_args.zo_two_point_precision = "bf16"
        else:
            training_args.zo_quantization = "fp32"
            training_args.zo_two_point_precision = "fp32"
    if getattr(training_args, "quant_bits", None) is not None:
        training_args.zo_quantization_bits = validate_quzo_bits(getattr(training_args, "quant_bits"))
    training_args.zo_two_point_precision = str(getattr(training_args, "zo_two_point_precision", "fp32")).lower()
    if training_args.zo_two_point_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Invalid --zo_two_point_precision={training_args.zo_two_point_precision}. Allowed: fp32, fp16, bf16")
    zo_quantization_alias = getattr(training_args, "zo_quantization", None)
    if zo_quantization_alias not in (None, ""):
        training_args.zo_quantization_bits = validate_quzo_bits(zo_quantization_alias)
    else:
        training_args.zo_quantization_bits = validate_quzo_bits(getattr(training_args, "zo_quantization_bits", 32))
    requested_quantization_algorithm = str(getattr(training_args, "quantization_algorithm", "per_tensor_symmetric") or "per_tensor_symmetric").strip().lower().replace("-", "_")
    training_args.quantization_requested_algorithm = requested_quantization_algorithm
    quantization_fallback_reason = ""
    algo_for_impl = requested_quantization_algorithm
    quantization_group_size = int(getattr(training_args, "quantization_group_size", 0) or 0)
    quantization_block_size = int(getattr(training_args, "quantization_block_size", 0) or 0)
    if quantization_group_size <= 0 and quantization_block_size > 0:
        quantization_group_size = quantization_block_size
    if quantization_block_size <= 0 and quantization_group_size > 0:
        quantization_block_size = quantization_group_size
    if requested_quantization_algorithm in {"gptq", "gptq256", "gptq_256"}:
        if exact_gptq_available():
            raise ValueError("exact GPTQ was reported available but no GPTQ execution path is wired in medium_models")
        algo_for_impl = "groupwise_int8_block256"
        if quantization_group_size <= 0:
            quantization_group_size = 256
            quantization_block_size = 256
        quantization_fallback_reason = "exact GPTQ/Hessian calibration is not implemented in medium_models; using groupwise symmetric INT8 block-256 fallback"
    if algo_for_impl in {"groupwise_int8_block256", "groupwise_int4_block256"} and quantization_group_size <= 0:
        quantization_group_size = 256
        quantization_block_size = 256
    quantization_impl = normalize_quantization_algorithm(algo_for_impl)
    if quantization_impl == "groupwise_symmetric" and quantization_group_size <= 0:
        raise ValueError("--quantization_algorithm groupwise_* requires --quantization_group_size/--quantization_block_size > 0")
    training_args.quantization_algorithm_impl = quantization_impl
    training_args.quantization_group_size = int(quantization_group_size)
    training_args.quantization_block_size = int(quantization_block_size)
    training_args.quantization_algorithm = quantization_algorithm_label(
        quantization_impl,
        bits=int(training_args.zo_quantization_bits),
        group_size=int(quantization_group_size),
    )
    training_args.quantization_exact_gptq = False
    training_args.quantization_fallback_reason = quantization_fallback_reason
    training_args.quantization_calibration_samples = int(getattr(training_args, "quantization_calibration_samples", 0) or 0)
    if training_args.quantization_calibration_samples < 0:
        raise ValueError("--quantization_calibration_samples must be >= 0")
    if getattr(training_args, "zo_h", None) is not None:
        zo_h = float(getattr(training_args, "zo_h"))
        if (not math.isfinite(zo_h)) or zo_h <= 0.0:
            raise ValueError("--zo_h must be a finite positive float")
        training_args.zero_order_eps = zo_h
        training_args.initial_h = zo_h
        training_args.init_h = zo_h
    training_args.zo_update_backend = normalize_zo_update_backend(getattr(training_args, "zo_update_backend", "direct_int8"))
    training_args.residual_dtype = str(getattr(training_args, "residual_dtype", "fp32")).strip().lower()
    if training_args.residual_dtype not in {"fp16", "float16", "bf16", "bfloat16", "fp32", "float32"}:
        raise ValueError("--residual_dtype must be one of fp16, bf16, fp32")
    training_args.residual_commit_mode = str(getattr(training_args, "residual_commit_mode", "round")).strip().lower()
    if training_args.residual_commit_mode not in {"round", "floor", "stochastic"}:
        raise ValueError("--residual_commit_mode must be one of round, floor, stochastic")
    training_args.residual_max_code_step = int(getattr(training_args, "residual_max_code_step", 0))
    if training_args.residual_max_code_step < 0:
        raise ValueError("--residual_max_code_step must be >= 0")
    training_args.residual_commit_threshold = float(getattr(training_args, "residual_commit_threshold", 0.0) or 0.0)
    if training_args.residual_commit_threshold < 0.0:
        raise ValueError("--residual_commit_threshold must be >= 0")
    training_args.residual_commit_select = str(getattr(training_args, "residual_commit_select", "all")).strip().lower()
    if training_args.residual_commit_select not in {"all", "top_abs_acc", "norm_budget"}:
        raise ValueError("--residual_commit_select must be one of all, top_abs_acc, norm_budget")
    training_args.residual_target_active_frac = float(getattr(training_args, "residual_target_active_frac", 0.0) or 0.0)
    if training_args.residual_target_active_frac < 0.0:
        raise ValueError("--residual_target_active_frac must be >= 0")
    training_args.residual_actual_norm_ratio_cap = float(getattr(training_args, "residual_actual_norm_ratio_cap", 0.0) or 0.0)
    if training_args.residual_actual_norm_ratio_cap < 0.0:
        raise ValueError("--residual_actual_norm_ratio_cap must be >= 0")
    training_args.residual_budget_reference = str(getattr(training_args, "residual_budget_reference", "acc")).strip().lower()
    if training_args.residual_budget_reference not in {"acc", "delta"}:
        raise ValueError("--residual_budget_reference must be one of acc, delta")
    training_args.residual_decay = float(getattr(training_args, "residual_decay", 1.0) if getattr(training_args, "residual_decay", None) is not None else 1.0)
    if (not math.isfinite(training_args.residual_decay)) or training_args.residual_decay < 0.0:
        raise ValueError("--residual_decay must be a finite non-negative float")
    training_args.residual_scale_mode = str(getattr(training_args, "residual_scale_mode", "tensor")).strip().lower()
    if training_args.residual_scale_mode not in {"tensor", "channel", "block"}:
        raise ValueError("--residual_scale_mode must be one of tensor, channel, block")
    training_args.residual_block_size = int(getattr(training_args, "residual_block_size", 0) or 0)
    if training_args.residual_scale_mode == "block" and training_args.residual_block_size <= 0:
        raise ValueError("--residual_block_size must be > 0 when --residual_scale_mode block")
    training_args.int8_scale_floor = float(getattr(training_args, "int8_scale_floor", 0.0) or 0.0)
    if training_args.int8_scale_floor < 0.0:
        raise ValueError("--int8_scale_floor must be >= 0")
    training_args.log_update_stats_every = int(getattr(training_args, "log_update_stats_every", 0))
    if training_args.log_update_stats_every < 0:
        raise ValueError("--log_update_stats_every must be >= 0")
    training_args.debug_num_tensors = int(getattr(training_args, "debug_num_tensors", 5))
    if training_args.debug_num_tensors <= 0:
        raise ValueError("--debug_num_tensors must be > 0")
    training_args.num_probe_batches = int(getattr(training_args, "num_probe_batches", 1))
    if training_args.num_probe_batches <= 0:
        raise ValueError("--num_probe_batches must be > 0")
    training_args.checkpoint_probe_num_directions = int(getattr(training_args, "checkpoint_probe_num_directions", 16))
    if training_args.checkpoint_probe_num_directions <= 0:
        raise ValueError("--checkpoint_probe_num_directions must be > 0")
    training_args.checkpoint_probe_num_batches = int(getattr(training_args, "checkpoint_probe_num_batches", 1))
    if training_args.checkpoint_probe_num_batches <= 0:
        raise ValueError("--checkpoint_probe_num_batches must be > 0")
    training_args.zo_update_norm_clip = float(getattr(training_args, "zo_update_norm_clip", 0.0) or 0.0)
    if training_args.zo_update_norm_clip < 0.0:
        raise ValueError("--zo_update_norm_clip must be >= 0")
    training_args.zo_scalar_clip = float(getattr(training_args, "zo_scalar_clip", 0.0) or 0.0)
    if training_args.zo_scalar_clip < 0.0:
        raise ValueError("--zo_scalar_clip must be >= 0")
    training_args.zo_direction_sparse_rate = validate_sparse_ratio(getattr(training_args, "zo_direction_sparse_rate", 1.0))
    training_args.zo_direction_sparse_mode = str(getattr(training_args, "zo_direction_sparse_mode", "none")).strip().lower()
    if getattr(training_args, "sparse_rate", None) is not None:
        training_args.zo_direction_sparse_rate = getattr(training_args, "sparse_rate")
    if getattr(training_args, "sparse_mode", None) not in (None, ""):
        training_args.zo_direction_sparse_mode = getattr(training_args, "sparse_mode")
    if getattr(training_args, "sparse_rescale", None) not in (None, ""):
        training_args.zo_sparse_rescale = getattr(training_args, "sparse_rescale")
    training_args.direction_type = str(getattr(training_args, "direction_type", "dense") or "dense").strip().lower()
    if training_args.direction_type not in {"dense", "sparse"}:
        raise ValueError("--direction_type must be one of dense, sparse")
    if training_args.direction_type == "dense":
        training_args.zo_direction_sparse_rate = 1.0
        training_args.zo_direction_sparse_mode = "none"
        training_args.zo_sparse_rescale = "none"
    elif str(getattr(training_args, "zo_direction_sparse_mode", "none")).strip().lower() == "none":
        training_args.zo_direction_sparse_mode = "bernoulli"
    training_args.zo_direction_sparse_rate = validate_sparse_ratio(getattr(training_args, "zo_direction_sparse_rate", 1.0))
    training_args.zo_direction_sparse_mode = str(getattr(training_args, "zo_direction_sparse_mode", "none")).strip().lower()
    if training_args.zo_direction_sparse_mode not in {"none", "exact_random", "bernoulli"}:
        raise ValueError("--zo_direction_sparse_mode must be one of none, exact_random, bernoulli")
    training_args.zo_sparse_rescale = str(getattr(training_args, "zo_sparse_rescale", "none")).strip().lower()
    if training_args.zo_sparse_rescale not in {"none", "inv_sqrt_p"}:
        raise ValueError("--zo_sparse_rescale must be one of none, inv_sqrt_p")
    training_args.zo_method = normalize_zo_method_name(getattr(training_args, "zo_method", None))
    training_args.sparse_ratio = validate_sparse_ratio(getattr(training_args, "sparse_ratio", 1.0))
    training_args.sparse_mask_strategy = normalize_sparse_mask_strategy(getattr(training_args, "sparse_mask_strategy", "percentile_per_layer"))
    training_args.sparse_scope = normalize_sparse_scope(getattr(training_args, "sparse_scope", "trainable_only"))
    training_args.sparse_mask_refresh_steps = int(getattr(training_args, "sparse_mask_refresh_steps", 100))
    if training_args.sparse_mask_refresh_steps < 0:
        raise ValueError("--sparse_mask_refresh_steps must be >= 0")
    if int(getattr(training_args, "lozo_rank", 2)) <= 0:
        raise ValueError("--lozo_rank must be > 0")
    if int(getattr(training_args, "lozo_step_interval", 50)) <= 0:
        raise ValueError("--lozo_step_interval must be > 0")
    if not (0.0 <= float(getattr(training_args, "lozo_beta1", 0.9)) < 1.0):
        raise ValueError("--lozo_beta1 must satisfy 0 <= beta1 < 1")
    if getattr(training_args, "zo_method", None) == "sparse_mezo" and (not sparse_mezo_enabled(getattr(training_args, "sparse_ratio", 1.0))):
        raise ValueError("--zo_method=sparse_mezo requires --sparse_ratio < 1.0")
    if getattr(training_args, "zo_method", None) in {"lozo", "lozo_m", "hizoo"}:
        if sparse_mezo_enabled(getattr(training_args, "sparse_ratio", 1.0)):
            raise ValueError(f"--zo_method={training_args.zo_method} is incompatible with --sparse_ratio < 1.0")
        if int(getattr(training_args, "zo_quantization_bits", 32)) in {8, 4}:
            raise ValueError(f"--zo_method={training_args.zo_method} is incompatible with QuZO low-bit perturbations")
    if training_args.zo_update_backend in {"residual_grid", "fp16_master"}:
        if int(getattr(training_args, "zo_quantization_bits", 32)) not in {8, 4}:
            raise ValueError("--zo_update_backend residual_grid/fp16_master requires --zo_quantization int8 or int4")
        if bool(getattr(training_args, "zero_order_use_trainer_optim", False)):
            raise ValueError("--zo_update_backend residual_grid/fp16_master currently requires direct ZO updates; set --zero_order_use_trainer_optim false")
    if int(getattr(training_args, "measure_perf_tail_window_steps", 10)) <= 0:
        raise ValueError("--measure_perf_tail_window_steps must be > 0")
    if int(getattr(training_args, "zo_probe_num_seeds", 16)) <= 0:
        raise ValueError("--zo_probe_num_seeds must be > 0")
    if getattr(training_args, "num_probe_directions", None) is not None:
        training_args.zo_probe_num_seeds = int(getattr(training_args, "num_probe_directions"))
        if training_args.zo_probe_num_seeds <= 0:
            raise ValueError("--num_probe_directions must be > 0")
    if bool(getattr(training_args, "probe_diagnostics_only", False)):
        if int(getattr(training_args, "zo_probe_every", 0)) <= 0:
            training_args.zo_probe_every = 1
        training_args.learning_rate = 0.0
        logger.info("[probe-config] probe_diagnostics_only=true: forcing learning_rate=0 and enabling zo_probe_every=%s", training_args.zo_probe_every)
    if int(getattr(training_args, "random_prediction_guard_recent_evals", 2)) <= 0:
        raise ValueError("--random_prediction_guard_recent_evals must be > 0")
    if float(getattr(training_args, "random_prediction_guard_min_loss_drop", 0.05)) < 0.0:
        raise ValueError("--random_prediction_guard_min_loss_drop must be >= 0")
    if float(getattr(training_args, "random_prediction_guard_min_acc_gain", 0.02)) < 0.0:
        raise ValueError("--random_prediction_guard_min_acc_gain must be >= 0")
    if int(getattr(training_args, "zo_probe_health_guard_step", 2000)) <= 0:
        raise ValueError("--zo_probe_health_guard_step must be > 0")
    if int(getattr(training_args, "zo_probe_health_guard_max_bad_probes", 3)) <= 0:
        raise ValueError("--zo_probe_health_guard_max_bad_probes must be > 0")
    training_args.model_storage_fp16 = bool(
        getattr(training_args, "efficient_zero_order_fp16", False)
        or (
            bool(getattr(training_args, "zero_order_optim", False))
            and int(getattr(training_args, "zo_quantization_bits", 32)) == 16
        )
    )
    training_args.h_estimation_active_source = str(
        getattr(training_args, "h_estimation_active_source", "auto")
    ).lower()
    allowed_h_sources = {"auto", "fixed", "additive", "two_point"}
    if training_args.h_estimation_active_source not in allowed_h_sources:
        raise ValueError(
            f"Invalid --h_estimation_active_source={training_args.h_estimation_active_source}. "
            f"Allowed: {sorted(allowed_h_sources)}"
        )
    if training_args.h_estimation_active_source == "auto":
        training_args.h_estimation_active_source = "additive" if bool(training_args.use_adaptive_h) else "fixed"
    if training_args.h_estimation_active_source == "additive" and not (
        bool(training_args.use_adaptive_h) or bool(getattr(training_args, "enable_additive_h_estimation", False))
    ):
        raise ValueError(
            "--h_estimation_active_source=additive requires --use_adaptive_h or --enable_additive_h_estimation"
        )
    if training_args.h_estimation_active_source == "two_point" and not bool(
        getattr(training_args, "enable_two_point_h_estimation", False)
    ):
        raise ValueError(
            "--h_estimation_active_source=two_point requires --enable_two_point_h_estimation"
        )

    # Append MeZO-related switches into the log filename so that runs are easier to identify.
    # USE_H  -> training_args.use_adaptive_h
    # USE_C  -> training_args.use_c_scale
    # 更改了result文件的命名规则
    if getattr(training_args, "log_file", None):
        base, ext = os.path.splitext(training_args.log_file)
        suffix = (
            f"-USE_H{int(getattr(training_args, 'use_adaptive_h', False))}"
            f"-USE_C{int(getattr(training_args, 'use_c_scale', False))}"
        )
        training_args.log_file = base + suffix + ext

    if training_args.sweep:
        now = datetime.now()
        dt_str = now.strftime('%m_%d_%H_%M_%S')
        training_args.output_dir = os.path.join(training_args.output_dir, dt_str)

    #if model_args.apply_lora:
    #    assert 'roberta' in model_args.model_name_or_path, 'LoRA only implemented for RoBERTa models'

    if training_args.kernel_formula == 'asymmetric_signgd':
        assert training_args.binary_classification, 'asymmetric solver not implemented for multi-class setting, use --binary_classification'

    if training_args.optimizer_variant != '':
        assert training_args.optimizer == 'sgd', 'variants on optimizer are only implemented for SGD'

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True


    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    training_args.local_rank = -1
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    # print("参数")
    # print(training_args.local_rank)

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info(
        "[zo-config] zo_two_point_precision=%s | zero_order_eps=%s | zo_use_true_directional_derivative=%s",
        training_args.zo_two_point_precision,
        training_args.zero_order_eps,
        bool(getattr(training_args, "zo_use_true_directional_derivative", False)),
    )
    logger.info(
        "[h-schedule-config] h_schedule=%s | window=[%s,%s] | h0=%s | gamma=%s | total_steps=%s | d_eff=%s | n_eff=%s | lipschitz_l=%s | c_delta=%s | grid=%s | log_csv=%s",
        str(getattr(training_args, "h_schedule", "fixed")),
        float(getattr(training_args, "h_schedule_window_min", 0.0) or 0.0),
        float(getattr(training_args, "h_schedule_window_max", 0.0) or 0.0),
        float(getattr(training_args, "h_schedule_h0", 0.0) or 0.0),
        float(getattr(training_args, "h_schedule_gamma", 0.101)),
        int(getattr(training_args, "h_schedule_total_steps", 0) or 0),
        float(getattr(training_args, "h_schedule_d_eff", 1.0)),
        float(getattr(training_args, "h_schedule_n_eff", 1.0)),
        float(getattr(training_args, "h_schedule_lipschitz_l", 0.0) or 0.0),
        float(getattr(training_args, "h_schedule_c_delta", 1.0)),
        str(getattr(training_args, "h_schedule_grid", "") or ""),
        bool(getattr(training_args, "h_schedule_log_csv", True)),
    )
    logger.info(
        "[quzo-config] zo_quantization_bits=%s | quzo_lowbit_enabled=%s | quzo_lowbit_probe_impl=%s | quantization_algorithm=%s | quantization_algorithm_impl=%s | requested_quantization_algorithm=%s | group_size=%s | block_size=%s | exact_gptq=%s | calibration_samples=%s | fallback_reason=%s",
        int(getattr(training_args, "zo_quantization_bits", 32)),
        int(getattr(training_args, "zo_quantization_bits", 32)) in {8, 4},
        (
            "q_w_plus_hz_resnap"
            if int(getattr(training_args, "zo_quantization_bits", 32)) in {8, 4}
            else "n/a"
        ),
        str(getattr(training_args, "quantization_algorithm", "per_tensor_symmetric")),
        str(getattr(training_args, "quantization_algorithm_impl", "per_tensor_symmetric")),
        str(getattr(training_args, "quantization_requested_algorithm", getattr(training_args, "quantization_algorithm", "per_tensor_symmetric"))),
        int(getattr(training_args, "quantization_group_size", 0) or 0),
        int(getattr(training_args, "quantization_block_size", 0) or 0),
        bool(getattr(training_args, "quantization_exact_gptq", False)),
        int(getattr(training_args, "quantization_calibration_samples", 0) or 0),
        str(getattr(training_args, "quantization_fallback_reason", "") or ""),
    )
    logger.info(
        "[quzo-update-config] backend=%s | residual_dtype=%s | commit_mode=%s | max_code_step=%s | residual_commit_threshold=%s | residual_commit_select=%s | residual_target_active_frac=%s | residual_actual_norm_ratio_cap=%s | residual_budget_reference=%s | residual_decay=%s | residual_scale_mode=%s | residual_block_size=%s | int8_freeze_scale=%s | int8_scale_floor=%s | update_norm_clip=%s | scalar_clip=%s | log_update_stats_every=%s | update_stats_jsonl=%s",
        str(getattr(training_args, "zo_update_backend", "direct_int8")),
        str(getattr(training_args, "residual_dtype", "fp32")),
        str(getattr(training_args, "residual_commit_mode", "round")),
        int(getattr(training_args, "residual_max_code_step", 0)),
        float(getattr(training_args, "residual_commit_threshold", 0.0) or 0.0),
        str(getattr(training_args, "residual_commit_select", "all")),
        float(getattr(training_args, "residual_target_active_frac", 0.0) or 0.0),
        float(getattr(training_args, "residual_actual_norm_ratio_cap", 0.0) or 0.0),
        str(getattr(training_args, "residual_budget_reference", "acc")),
        float(getattr(training_args, "residual_decay", 1.0) if getattr(training_args, "residual_decay", None) is not None else 1.0),
        str(getattr(training_args, "residual_scale_mode", "tensor")),
        int(getattr(training_args, "residual_block_size", 0) or 0),
        bool(getattr(training_args, "int8_freeze_scale", True)),
        float(getattr(training_args, "int8_scale_floor", 0.0) or 0.0),
        float(getattr(training_args, "zo_update_norm_clip", 0.0) or 0.0),
        float(getattr(training_args, "zo_scalar_clip", 0.0) or 0.0),
        int(getattr(training_args, "log_update_stats_every", 0)),
        str(getattr(training_args, "save_update_stats_jsonl", "") or ""),
    )
    if bool(getattr(training_args, "zero_order_optim", False)):
        logger.info(
            "[sparse-mezo-config] enabled=%s | sparse_ratio=%s | sparse_mask_strategy=%s | sparse_scope=%s | sparse_log_active_fraction=%s | sparse_mask_refresh_steps=%s",
            bool(sparse_mezo_enabled(getattr(training_args, "sparse_ratio", 1.0))),
            float(getattr(training_args, "sparse_ratio", 1.0)),
            str(getattr(training_args, "sparse_mask_strategy", "percentile_per_layer")),
            str(getattr(training_args, "sparse_scope", "trainable_only")),
            bool(getattr(training_args, "sparse_log_active_fraction", True)),
            int(getattr(training_args, "sparse_mask_refresh_steps", 100)),
        )
        logger.info(
            "[sparse-mezo-config] ratio semantics: sparse_ratio targets the active fraction per trainable tensor; ratio=1.0 disables masking. Order: refresh thresholds+mask on the configured cadence -> sample sparse-aware direction -> apply QuZO snapping after the sparse perturb/update path when low-bit QuZO is active."
        )
        logger.info(
            "[zo-direction-sparse-config] mode=%s | rate=%s | rescale=%s | per_layer_exact=%s | zo_h=%s",
            str(getattr(training_args, "zo_direction_sparse_mode", "none")),
            float(getattr(training_args, "zo_direction_sparse_rate", 1.0)),
            str(getattr(training_args, "zo_sparse_rescale", "none")),
            bool(getattr(training_args, "zo_sparse_per_layer_exact", True)),
            getattr(training_args, "zo_h", None),
        )

    # Set seed
    set_seed(training_args.seed)

    # If not provided, use the same seed for data shuffling.
    if getattr(training_args, "data_seed", None) is None:
        training_args.data_seed = training_args.seed

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    data_resolution = resolve_and_prepare_data(
        data_args=data_args,
        training_args=training_args,
        logger=logger,
    )
    data_args.dataset_mode = data_resolution.resolved_dataset_mode
    data_args.data_dir = data_resolution.resolved_data_dir

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            new_sfc_template = data_args.sfc_prompt + ''
            old_template = old_template.replace('*cls*', '')
            old_template = old_template.replace('*bos*', '')
            if data_args.gpt3_in_context_head:
                new_template = new_template.replace('*cls*', '')
                new_template = new_template.replace('*bos*', '')

            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                if "opt" in model_args.model_name_or_path or "gpt" in model_args.model_name_or_path:
                    sub_template = sub_template + "*labelx_{}*".format(instance_id)
                else:
                    sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + data_args.gpt3_demo_separator + sub_template # Put context at the end
                    new_sfc_template = new_sfc_template + data_args.gpt3_demo_separator + sub_template
                else:
                    new_template = sub_template + data_args.gpt3_demo_separator + new_template # Put context at the beginning
                    new_sfc_template = sub_template + data_args.gpt3_demo_separator + new_sfc_template
            if data_args.gpt3_in_context_head:
                new_template = "*bos*" + new_template
                new_sfc_template = "*bos*" + new_sfc_template
            logger.info("| {} => {}".format(data_args.template, new_template))
            logger.info("New SFC template (in-context learning): {}".format(new_sfc_template))
            data_args.template = new_template
            if model_args.icl_sfc:
                data_args.icl_sfc_prompt = new_sfc_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    # Create config
    config_kwargs = {'apply_lora': model_args.apply_lora,
                    'lora_alpha': model_args.lora_alpha,
                    'lora_r': model_args.lora_r}
    if model_args.apply_lora:
        if 'roberta' in model_args.model_name_or_path:
            config = RobertaConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                **config_kwargs)
        else:
            config = OPTConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                **config_kwargs
            )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir
        )

    if training_args.untie_emb:
        logger.warn("Untie embeddings and lm head")
        logger.warn("NOTE that this only works for OPT. By default RoBERTa model embeddings are already untied.")
        config.tie_word_embeddings = False

    if 'prompt' in model_args.few_shot_type:
        model_fn = MODEL_TYPES[config.model_type]
    elif model_args.few_shot_type == 'finetune':
        if training_args.from_linearhead:
            model_fn = MODEL_TYPES[config.model_type]
        else:
            model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )
    if "opt" in model_args.model_name_or_path:
        # Set SEP token
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = 0
    if "gpt2" in model_args.model_name_or_path:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id


    if training_args.hf_inference_model:
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-5}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}

        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            device_map='auto',
            torch_dtype=torch.float16 if training_args.model_storage_fp16 else torch.float32,
            max_memory=max_memory,
        )
    else:
        model_load_kwargs = {}
        if bool(getattr(training_args, "model_storage_fp16", False)):
            model_load_kwargs["torch_dtype"] = torch.float16
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            **model_load_kwargs,
        )

    if bool(getattr(training_args, "model_storage_fp16", False)):
        model.half()
        logger.info("[precision-config] model parameters and buffers converted to FP16 storage")

    if training_args.tie_emb:
        logger.warn("Tie embeddings. Only work for RoBERTa (in our code by default they are not tied)")
        model.tie_emb()

    if training_args.head_tuning:
        if model.config.model_type == "roberta":
            head_name = "lm_head"

        for n, p in model.named_parameters():
            if head_name not in n:
                p.requires_grad = False
            else:
                logger.info(f"Only tuning {n}")

    tokenizer.model_type = model.config.model_type

    if training_args.exclude_first_layers != -1:
        model = convert_opt_model(model, config, training_args.exclude_first_layers)

    if training_args.prefix_tuning:
        from src.prefix import PrefixTuning
        PrefixTuning(model, num_prefix=training_args.num_prefix, reparam=not training_args.no_reparam, float16=training_args.model_storage_fp16, init_by_real_act=training_args.prefix_init_by_real_act)

    # Get our special datasets.
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_train
        else None
    )
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_eval
        else None
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_predict
        else None
    )

    # set_seed(training_args.seed)  # (REMOVED redundant second call)

    if training_args.random_model_init:
        model.init_weights() # reinit weights to random

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    # Pass dataset and argument information to the model
    label_word_source = eval_dataset if eval_dataset is not None else train_dataset
    if label_word_source is None:
        label_word_source = test_dataset
    if label_word_source is not None and label_word_source.label_word_list is not None:
        model.label_word_list = torch.tensor(label_word_source.label_word_list).long().to(training_args.device)
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    if model_args.apply_lora:
        for name, param in model.named_parameters():
            if (name.startswith('roberta') and "lora" not in name) or (name.startswith('opt') and "lora" not in name):
                param.requires_grad_(False)

    maybe_convert_model_to_torchao_float8_training(model, getattr(training_args, "use_torchao_float8", False))

    if (
        bool(getattr(training_args, "zero_order_optim", False))
        and int(getattr(training_args, "zo_quantization_bits", 32)) in {8, 4}
    ):
        quantize_model_in_place(
            model,
            int(training_args.zo_quantization_bits),
            include_frozen=True,
            seed=int(getattr(training_args, "seed", 0)),
            algorithm=str(getattr(training_args, "quantization_algorithm", "per_tensor_symmetric")),
            group_size=int(getattr(training_args, "quantization_group_size", 0) or 0),
            block_size=int(getattr(training_args, "quantization_block_size", 0) or 0),
        )
        logger.info(
            "[quzo-config] quantized model parameters in-place at %d bits before ZO training with algorithm=%s group_size=%s",
            int(training_args.zo_quantization_bits),
            str(getattr(training_args, "quantization_algorithm", "per_tensor_symmetric")),
            int(getattr(training_args, "quantization_group_size", 0) or 0),
        )
    elif bool(getattr(training_args, "zero_order_optim", False)) and int(getattr(training_args, "zo_quantization_bits", 32)) == 16:
        logger.info(
            "[quzo-config] zo_quantization_bits=16 uses FP16 model storage plus FP16 ZO perturb/probe convention; "
            "low-bit QuZO snap remains disabled for FP16"
        )

    run_metadata = collect_run_metadata(
        zo_method=infer_medium_run_zo_method(training_args),
        args=training_args,
        model=model,
        output_dir=training_args.output_dir,
        model_name=model_args.model_name_or_path,
        task_name=data_args.task_name,
        repo_root=str(REPO_ROOT),
        extra_metadata={
            "quzo_lowbit_probe_impl": (
                "q_w_plus_hz_resnap"
                if int(getattr(training_args, "zo_quantization_bits", 32)) in {8, 4}
                else "n/a"
            ),
            "quantization_requested_algorithm": str(getattr(training_args, "quantization_requested_algorithm", getattr(training_args, "quantization_algorithm", "")) or ""),
            "quantization_algorithm": str(getattr(training_args, "quantization_algorithm", "per_tensor_symmetric")),
            "quantization_algorithm_impl": str(getattr(training_args, "quantization_algorithm_impl", "per_tensor_symmetric")),
            "quantization_group_size": int(getattr(training_args, "quantization_group_size", 0) or 0),
            "quantization_block_size": int(getattr(training_args, "quantization_block_size", 0) or 0),
            "quantization_exact_gptq": bool(getattr(training_args, "quantization_exact_gptq", False)),
            "quantization_fallback_reason": str(getattr(training_args, "quantization_fallback_reason", "") or ""),
            "quantization_calibration_samples": int(getattr(training_args, "quantization_calibration_samples", 0) or 0),
            "zo_use_true_directional_derivative": bool(getattr(training_args, "zo_use_true_directional_derivative", False)),
        },
    )
    run_metadata_path = None
    if int(getattr(training_args, "local_rank", -1)) <= 0:
        run_metadata_path = write_run_metadata(run_metadata, training_args.output_dir)

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]

            num_sample = test_dataset.num_sample if eval_dataset is None else eval_dataset.num_sample
            logits = predictions.reshape([num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer_classes = {
        "standard": Trainer,
        "linearhead": LinearHeadTrainer,
    }
    trainer_class = trainer_classes[training_args.trainer]
    trainer_kwargs = {}
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        data_collator=MyDataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        **trainer_kwargs
    )

    if bool(getattr(training_args, "debug_residual_grid_consistency", False)):
        debug_save_dir = str(getattr(training_args, "debug_save_dir", "") or "").strip()
        if not debug_save_dir:
            debug_save_dir = os.path.join(training_args.output_dir, "residual_grid_debug")
        elif not os.path.isabs(debug_save_dir):
            debug_save_dir = os.path.join(training_args.output_dir, debug_save_dir)
        summary = trainer.debug_residual_grid_consistency(
            model,
            save_dir=debug_save_dir,
            layer_regex=(str(getattr(training_args, "debug_layer_regex", "") or "").strip() or None),
            num_tensors=int(getattr(training_args, "debug_num_tensors", 5)),
            dump_tensor_stats=bool(getattr(training_args, "debug_dump_tensor_stats", False)),
        )
        logger.info("[residual-grid-debug] summary: %s", json.dumps(summary, sort_keys=True))
        return

    if bool(getattr(training_args, "probe_window_diagnostics_only", False)):
        summary = trainer.run_probe_window_diagnostics(
            model,
            output_dir=training_args.output_dir,
            num_batches=int(getattr(training_args, "num_probe_batches", 1)),
            num_directions=int(getattr(training_args, "zo_probe_num_seeds", 16)),
            compute_true_grad_directional=bool(getattr(training_args, "compute_true_grad_directional", True)),
        )
        logger.info("[probe-window] summary: %s", json.dumps(summary, sort_keys=True))
        return

    # Calibration
    if model_args.sfc:
        inputs = tokenizer([data_args.sfc_prompt.replace("_", " ")], return_tensors="pt")
        logger.info(f"Calibrating SFC with prompt: {data_args.sfc_prompt}")
        logger.info("Inputs: {}".format(inputs.input_ids))
        inputs = inputs.to(model.device)
        with torch.no_grad():
            model.eval()
            logits = model(**inputs)[0]
        model.sfc_bias = F.log_softmax(logits.squeeze(0).detach())
        logger.info("SFC bias: {}".format(model.sfc_bias))


    # Training
    train_result = None
    train_output = None
    train_objective = None
    if training_args.do_train:
        train_result = trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        if isinstance(train_result, tuple):
            if len(train_result) > 0:
                train_output = train_result[0]
            if len(train_result) > 1:
                train_objective = train_result[1]
        else:
            train_output = train_result

        if training_args.trainer == "hessian":
            # Write the result to log
            with FileLock('log_hessian.lock'):
                with open('log_hessian', 'a') as f:
                    train_result.update(vars(model_args))
                    train_result.update(vars(training_args))
                    train_result.update(vars(data_args))
                    if 'evaluation_strategy' in train_result:
                        train_result.pop('evaluation_strategy')
                    f.write(str(train_result) + '\n')
            exit()

        # Use the early stop, so do not save the model in the end (unless specify save_at_last)

        if training_args.trainer == "standard" or training_args.trainer == "linearhead":
            if training_args.save_at_last:
                _save_model_with_shared_tensor_fallback(trainer, training_args.output_dir)

            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)
                torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
                torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))

            if training_args.evaluate_during_training:
                # Reload the best checkpoint (for eval)
                # model.load_state_dict(trainer.best_model_ckpt)
                # if training_args.prefix_tuning:
                #     # We can load prefix by directly using load_state_dict
                #     model.load_state_dict(torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin")))
                # else:
                #     model = model_fn.from_pretrained(training_args.output_dir)
                # if training_args.exclude_first_layers != -1:
                #     model = convert_opt_model(model, config, training_args.exclude_first_layers)

                # model = model.to(training_args.device)

                # Now we just reload this from memory instead of disk <-- much faster
                best_model_ckpt = getattr(trainer, "best_model_ckpt", None)
                if best_model_ckpt is not None:
                    trainer.model.load_state_dict(best_model_ckpt)
                else:
                    logger.info(
                        "evaluate_during_training=True but no in-memory best checkpoint was recorded; "
                        "keeping the current model weights."
                    )

    # Evaluation
    final_result = {
        'time': str(datetime.today()),
        'output_dir': training_args.output_dir
    }
    train_summary = {}
    if train_output is not None:
        train_global_step = getattr(train_output, "global_step", None)
        train_loss = getattr(train_output, "training_loss", None)
        train_metrics = getattr(train_output, "metrics", None)
        if train_global_step is not None:
            train_summary["global_step"] = int(train_global_step)
            final_result["train_global_step"] = int(train_global_step)
        if train_loss is not None:
            train_summary["training_loss"] = float(train_loss)
            final_result["train_loss"] = float(train_loss)
        if isinstance(train_metrics, dict) and len(train_metrics) > 0:
            train_summary["metrics"] = dict(train_metrics)
        if train_objective is not None:
            train_summary["best_dev_objective"] = train_objective
            final_result["best_dev_objective"] = train_objective

    eval_results = {}
    eval_results_by_task = {}
    eval_output_files = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            eval_output_files[eval_dataset.args.task_name] = output_eval_file
            eval_results_by_task[eval_dataset.args.task_name] = dict(eval_result)
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    test_results_by_task = {}
    test_output_files = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        ### Don't evaluate on mnli-mm for our purposes
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     test_datasets.append(
        #         FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", use_demo=('demo' in model_args.few_shot_type))
        #     )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            test_output_files[test_dataset.args.task_name] = output_test_file
            test_results_by_task[test_dataset.args.task_name] = dict(test_result)
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)


    if trainer.is_world_process_zero():
        metrics_csv_path = os.path.join(
            training_args.output_dir,
            "metrics_logs",
            f"metrics_adaptiveH-{int(training_args.use_adaptive_h)}_cscale-{int(training_args.use_c_scale)}.csv",
        )
        zo_probe_csv_path = os.path.join(training_args.output_dir, "zo_directional_probe.csv")
        h_estimation_csv_path = os.path.join(training_args.output_dir, "h_estimation.csv")
        eval_loss_last5_path = os.path.join(training_args.output_dir, "eval_loss_last5.json")
        probe_stats_path = str(getattr(training_args, "save_probe_stats_jsonl", "") or "").strip()
        if probe_stats_path and not os.path.isabs(probe_stats_path):
            probe_stats_path = os.path.join(training_args.output_dir, probe_stats_path)
        update_stats_path = str(getattr(training_args, "save_update_stats_jsonl", "") or "").strip()
        if update_stats_path and not os.path.isabs(update_stats_path):
            update_stats_path = os.path.join(training_args.output_dir, update_stats_path)
        run_summary_path = os.path.join(training_args.output_dir, "run_summary.json")
        summary_payload = {
            "time": final_result["time"],
            "task_name": data_args.task_name,
            "dataset_mode": getattr(data_args, "dataset_mode", None),
            "output_dir": training_args.output_dir,
            "train": train_summary,
            "eval": eval_results_by_task,
            "test": test_results_by_task,
            "artifacts": {
                "metrics_csv_last_row": _read_last_csv_row(metrics_csv_path),
                "zo_directional_probe_last_row": _read_last_csv_row(zo_probe_csv_path),
                "probe_stats_last_row": _read_last_jsonl_row(probe_stats_path),
                "update_stats_last_row": _read_last_jsonl_row(update_stats_path),
                "direction_sparse_last_stats": getattr(trainer, "latest_zo_direction_sparse_stats", None),
                "sparse_mezo_last_stats": getattr(trainer, "latest_sparse_mezo_stats", None),
                "tail_perf_metrics": getattr(trainer, "latest_perf_tail_metrics", None),
                "h_estimation_last_row": _read_last_csv_row(h_estimation_csv_path),
                "eval_loss_last5": _read_json_if_exists(eval_loss_last5_path),
            },
            "paths": {
                "metrics_csv": metrics_csv_path if os.path.exists(metrics_csv_path) else None,
                "zo_directional_probe_csv": zo_probe_csv_path if os.path.exists(zo_probe_csv_path) else None,
                "probe_stats_jsonl": probe_stats_path if probe_stats_path and os.path.exists(probe_stats_path) else None,
                "update_stats_jsonl": update_stats_path if update_stats_path and os.path.exists(update_stats_path) else None,
                "h_estimation_csv": h_estimation_csv_path if os.path.exists(h_estimation_csv_path) else None,
                "eval_loss_last5_json": eval_loss_last5_path if os.path.exists(eval_loss_last5_path) else None,
                "eval_results": eval_output_files,
                "test_results": test_output_files,
                "run_metadata_json": run_metadata_path,
            },
            "run_metadata": run_metadata,
            "config": {
                "model_args": vars(model_args),
                "training_args": vars(training_args),
                "data_args": vars(data_args),
            },
            "final_result": final_result,
        }
        with open(run_summary_path, "w", encoding="utf-8") as f:
            json.dump(_normalize_for_json(summary_payload), f, ensure_ascii=False, indent=2)
        final_result["run_summary_path"] = run_summary_path

        with FileLock('log.lock'):
            with open(training_args.log_file, 'a') as f:
                log_result = dict(final_result)
                log_result.update(vars(model_args))
                log_result.update(vars(training_args))
                log_result.update(vars(data_args))
                if 'evaluation_strategy' in log_result:
                    log_result.pop('evaluation_strategy')
                f.write(str(log_result) + '\n')

    logger.info('****** Output Dir *******')
    logger.info(training_args.output_dir)

    return eval_results

if __name__ == "__main__":
    main()
