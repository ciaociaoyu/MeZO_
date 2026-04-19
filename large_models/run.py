import logging
import os
import csv
import enum
import sys
from pathlib import Path

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
from dataclasses import dataclass, field, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from quzo import quzo_enabled, quantize_model_in_place, validate_quzo_bits
from sparse_mezo import (
    normalize_sparse_mask_strategy,
    normalize_sparse_scope,
    sparse_mezo_enabled,
    validate_sparse_ratio,
)
from utils import *
from trainer import OurTrainer
import random

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_metadata import collect_run_metadata, update_model_run_metadata, write_run_metadata

MODEL_NAME_ALIASES = {
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-13b": "facebook/opt-13b",
    "opt-30b": "facebook/opt-30b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
}

MODEL_FAMILY_KEYWORDS = [
    ("mistral", "mistral"),
    ("llama", "llama"),
    ("opt", "opt"),
    ("gpt2", "gpt2"),
]


def canonicalize_model_name(model_name: str) -> str:
    model_name = str(model_name).strip()
    return MODEL_NAME_ALIASES.get(model_name.lower(), model_name)


def infer_model_family(model_name: str, config=None) -> Optional[str]:
    config_model_type = getattr(config, "model_type", None)
    if config_model_type in {"opt", "mistral", "llama", "gpt2"}:
        return config_model_type

    normalized_model_name = canonicalize_model_name(model_name).lower()
    for keyword, family in MODEL_FAMILY_KEYWORDS:
        if keyword in normalized_model_name:
            return family
    return config_model_type


def load_hf_auth_token() -> Optional[str]:
    for env_name in ["MEZO_HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"]:
        token = os.environ.get(env_name)
        if token:
            return token.strip()

    token_file = os.path.join(os.path.dirname(__file__), ".hf_token.local")
    if os.path.exists(token_file):
        with open(token_file, "r", encoding="utf-8") as f:
            token = f.read().strip()
        if token:
            return token
    return None


def hf_auth_kwargs() -> dict:
    token = load_hf_auth_token()
    if not token:
        return {}
    return {"use_auth_token": token}


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


def detect_precision_label(args) -> str:
    if bool(getattr(args, "load_int8", False)):
        return "int8"
    if bool(getattr(args, "load_bfloat16", False)):
        return "bf16"
    if bool(getattr(args, "load_float16", False)):
        return "fp16"
    return "fp32"


def infer_large_run_zo_method(args) -> str:
    trainer_name = str(getattr(args, "trainer", "none") or "none").lower()
    if trainer_name == "zo":
        zo_method = str(getattr(args, "zo_method", "") or "").strip().lower()
        if zo_method:
            return zo_method
        if sparse_mezo_enabled(getattr(args, "sparse_ratio", 1.0)):
            return "sparse_mezo"
        return "mezo"
    if bool(getattr(args, "linear_probing", False)):
        return "linear_probing"
    if trainer_name == "regular":
        return "regular"
    if trainer_name == "none":
        return "inference"
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


def get_single_gpu_int8_device_map(model_family: str, device_index: int = 0):
    """
    Force Accelerate to use hook-based dispatch for single-GPU int8 loading.

    With the Transformers/Accelerate/bitsandbytes versions in this repo's
    environment, a one-device int8 dispatch can incorrectly fall back to
    `model.to()`, which bitsandbytes models reject. Mapping one lightweight
    module to an equivalent CUDA device string keeps the full model on the same
    physical GPU while still forcing the hook path.
    """
    split_module_by_family = {
        "opt": "model.decoder.final_layer_norm",
        "llama": "model.norm",
        "mistral": "model.norm",
        "gpt2": "transformer.ln_f",
    }
    split_module = split_module_by_family.get(str(model_family or "").lower())
    if not split_module:
        return None
    return {
        split_module: int(device_index),
        "": f"cuda:{int(device_index)}",
    }


def _normalize_for_json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, enum.Enum):
        return _normalize_for_json(value.value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _normalize_for_json(value.item())
        return [_normalize_for_json(v) for v in value.detach().cpu().tolist()]
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, set):
        return [_normalize_for_json(v) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _normalize_for_json(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        public_attrs = {
            str(k): _normalize_for_json(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
        if public_attrs:
            public_attrs["_class"] = value.__class__.__name__
            return public_attrs
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return value.name
    return value


def _read_json_if_exists(path: str):
    if not path or (not os.path.exists(path)):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return _normalize_for_json(json.load(f))


def _read_last_csv_row(path: str):
    if not path or (not os.path.exists(path)):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
    return _normalize_for_json(last_row)

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST-2" # canonical task names also accept legacy aliases like SST2 and sst-2
    dataset_mode: str = "auto" # auto, fewshot, or full
    num_k: int = 16 # few-shot training examples per class when labels exist; otherwise total examples
    data_seed: int = None # seed controlling dataset-mode sampling / train-dev split

    # Number of examples
    num_train: int = 0 # legacy subset interface; prefer dataset_mode + num_k
    num_dev: int = None # legacy dev-count interface; full mode defaults to full_dev_ratio when unset
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config
    full_dev_ratio: float = 0.1 # full-mode dev ratio, matching medium_models

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # model-loading/storage flag: load base model weights in float16 before any ZO/QuZO logic
    load_bfloat16: bool = False # model-loading/storage flag: load base model weights in bfloat16 before any ZO/QuZO logic
    load_int8: bool = False # model-loading/storage flag: use HF/bitsandbytes-style int8 loading; this is NOT the QuZO int8 perturb/update path
    use_torchao_float8: bool = False # swap nn.Linear modules with torchao Float8Linear for training
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
    zo_method: Optional[str] = None # explicit ZO method switch: mezo / sparse_mezo / lozo / lozo_m / hizoo
    zo_quantization_bits: int = 32 # ZO-side method switch: 32 -> plain MeZO, 16 -> repo's FP16 MeZO convention, 8/4 -> QuZO perturb/update path
    zo_quantization: Optional[str] = None # string alias for the same ZO-side method switch: fp32/off/none -> plain MeZO, fp16 -> FP16 MeZO path, int8/int4 -> QuZO low-bit path
    sparse_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Sparse MeZO target active fraction per trainable tensor. 1.0 disables masking and recovers vanilla MeZO."
        },
    )
    sparse_mask_strategy: str = field(
        default="percentile_per_layer",
        metadata={
            "help": "Sparse MeZO mask rule. percentile_per_layer keeps the lowest-|param| sparse_ratio fraction active in each trainable tensor."
        },
    )
    sparse_scope: str = field(
        default="trainable_only",
        metadata={"help": "Sparse MeZO scope. Current implementation only supports trainable_only."},
    )
    sparse_log_active_fraction: bool = field(
        default=True,
        metadata={"help": "If true, log the realized Sparse MeZO active-parameter fraction during training."},
    )
    zo_probe_every: int = 0 # run G-vs-D directional probe every N optimizer steps; 0 disables it
    zo_probe_num_seeds: int = 16 # number of probe directions per diagnostic step
    zo_probe_log_csv: bool = True # write directional probe rows to output_dir/zo_directional_probe.csv
    measure_perf_tail: bool = field(
        default=True,
        metadata={
            "help": "If true, measure wallclock/step, samples/sec, and max GPU memory once over the final tail window of optimizer steps."
        },
    )
    measure_perf_tail_window_steps: int = field(
        default=10,
        metadata={
            "help": "Number of final optimizer steps used for the one-shot tail performance snapshot when --measure_perf_tail is enabled."
        },
    )
    lozo_rank: int = 2 # rank r in LOZO
    lozo_step_interval: int = 50 # nu in LOZO
    lozo_beta1: float = 0.9 # beta1 for optional LOZO-M momentum path
    hizoo_hessian_smooth_type: str = "constant0" # HiZOO diagonal Hessian smoothing schedule

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
    zo_quantization_alias = getattr(args, "zo_quantization", None)
    if zo_quantization_alias not in (None, ""):
        args.zo_quantization_bits = validate_quzo_bits(zo_quantization_alias)
    else:
        args.zo_quantization_bits = validate_quzo_bits(getattr(args, "zo_quantization_bits", 32))
    args.zo_method = normalize_zo_method_name(getattr(args, "zo_method", None))
    args.sparse_ratio = validate_sparse_ratio(getattr(args, "sparse_ratio", 1.0))
    args.sparse_mask_strategy = normalize_sparse_mask_strategy(getattr(args, "sparse_mask_strategy", "percentile_per_layer"))
    args.sparse_scope = normalize_sparse_scope(getattr(args, "sparse_scope", "trainable_only"))
    if int(getattr(args, "lozo_rank", 2)) <= 0:
        raise ValueError("--lozo_rank must be > 0")
    if int(getattr(args, "lozo_step_interval", 50)) <= 0:
        raise ValueError("--lozo_step_interval must be > 0")
    if not (0.0 <= float(getattr(args, "lozo_beta1", 0.9)) < 1.0):
        raise ValueError("--lozo_beta1 must satisfy 0 <= beta1 < 1")
    if getattr(args, "zo_method", None) == "sparse_mezo" and (not sparse_mezo_enabled(getattr(args, "sparse_ratio", 1.0))):
        raise ValueError("--zo_method=sparse_mezo requires --sparse_ratio < 1.0")
    if getattr(args, "zo_method", None) in {"lozo", "lozo_m", "hizoo"}:
        if sparse_mezo_enabled(getattr(args, "sparse_ratio", 1.0)):
            raise ValueError(f"--zo_method={args.zo_method} is incompatible with --sparse_ratio < 1.0")
        if quzo_enabled(getattr(args, "zo_quantization_bits", 32)):
            raise ValueError(f"--zo_method={args.zo_method} is incompatible with QuZO low-bit perturbations")
    if int(getattr(args, "zo_probe_num_seeds", 16)) <= 0:
        raise ValueError("--zo_probe_num_seeds must be > 0")
    if int(getattr(args, "zo_probe_every", 0)) < 0:
        raise ValueError("--zo_probe_every must be >= 0")
    if int(getattr(args, "measure_perf_tail_window_steps", 10)) <= 0:
        raise ValueError("--measure_perf_tail_window_steps must be > 0")
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_data_args(args):
    args.task_name = tasks.canonicalize_task_name(args.task_name)
    args.dataset_mode = str(getattr(args, "dataset_mode", "auto") or "auto").lower()
    if args.dataset_mode not in {"auto", "fewshot", "full"}:
        raise ValueError(
            f"Unsupported --dataset_mode={args.dataset_mode}. Expected one of ['auto', 'fewshot', 'full']"
        )
    if getattr(args, "data_seed", None) is None:
        args.data_seed = args.train_set_seed if args.train_set_seed is not None else args.seed
    if args.trainer != "none" and args.train_set_seed is None:
        args.train_set_seed = args.data_seed
    args.dataset_mode = tasks.Dataset.resolve_dataset_mode(
        dataset_mode=args.dataset_mode,
        num_train=args.num_train,
        num_k=args.num_k,
    )
    logger.info(
        "[data] task_name=%s dataset_mode=%s num_k=%s data_seed=%s full_dev_ratio=%s",
        args.task_name,
        args.dataset_mode,
        args.num_k,
        args.data_seed,
        args.full_dev_ratio,
    )
    return args


def normalize_model_args(args):
    requested_model_name = args.model_name
    args.model_name = canonicalize_model_name(args.model_name)
    args.model_family = infer_model_family(args.model_name)
    logger.info(
        "[model] requested=%s resolved=%s family_hint=%s",
        requested_model_name,
        args.model_name,
        getattr(args, "model_family", None),
    )
    return args


def split_train_dev_samples(train_samples, num_dev=None, dev_ratio=None, seed=0):
    if not train_samples:
        return train_samples, None

    if num_dev is not None and num_dev > 0:
        dev_count = min(int(num_dev), len(train_samples))
    elif dev_ratio is not None and dev_ratio > 0 and len(train_samples) > 1:
        dev_count = int(round(len(train_samples) * float(dev_ratio)))
        dev_count = max(1, min(dev_count, len(train_samples) - 1))
    else:
        dev_count = 0

    if dev_count <= 0:
        return train_samples, None

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train_samples)).tolist()
    dev_ids = set(order[:dev_count])
    train_part = [sample for idx, sample in enumerate(train_samples) if idx not in dev_ids]
    dev_part = [sample for idx, sample in enumerate(train_samples) if idx in dev_ids]
    return train_part, dev_part


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()
        self.last_train_artifacts = {}


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            auth_kwargs = hf_auth_kwargs()
            config = AutoConfig.from_pretrained(self.args.model_name, **auth_kwargs)
            model_family = infer_model_family(self.args.model_name, config)
            self.args.model_family = model_family
            if auth_kwargs and model_family in {"mistral", "llama"}:
                logger.info("[model] Hugging Face auth token detected for %s", self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            training_manages_devices = self.args.trainer != "none"
            use_auto_device = (
                not self.args.no_auto_device
                and not training_manages_devices
                and torch.cuda.is_available()
            )
            if training_manages_devices and torch.cuda.is_available() and not self.args.no_auto_device:
                logger.info("[model] disabling auto device placement for training; Trainer/Accelerate will place the model")
            if self.args.head_tuning and model_family == "opt":
                # Head tuning
                from ht_opt import OPTForCausalLM
                model = OPTForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    **auth_kwargs,
                )
            elif not use_auto_device:
                # No auto device (use for FSDP)
                torch_dtype = None
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                load_kwargs = {}
                if torch_dtype is not None:
                    load_kwargs["torch_dtype"] = torch_dtype
                if bool(getattr(self.args, "load_int8", False)) and (not training_manages_devices) and torch.cuda.is_available():
                    device_index = torch.cuda.current_device()
                    device_map = get_single_gpu_int8_device_map(model_family, device_index=device_index)
                    if device_map is not None:
                        logger.info(
                            "[model] using single-GPU int8 dispatch workaround on cuda:%d (auto-device disabled)",
                            device_index,
                        )
                        load_kwargs["device_map"] = device_map
                    load_kwargs["load_in_8bit"] = True
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    **auth_kwargs,
                    **load_kwargs,
                )
            else:
                # Auto device loading
                free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                load_kwargs = {
                    "config": config,
                    "torch_dtype": torch_dtype,
                    "load_in_8bit": self.args.load_int8,
                    **auth_kwargs,
                }
                if bool(getattr(self.args, "load_int8", False)) and torch.cuda.device_count() == 1:
                    device_index = torch.cuda.current_device()
                    device_map = get_single_gpu_int8_device_map(model_family, device_index=device_index)
                    if device_map is not None:
                        logger.info(
                            "[model] using single-GPU int8 dispatch workaround on cuda:%d for bitsandbytes compatibility",
                            device_index,
                        )
                        load_kwargs["device_map"] = device_map
                    else:
                        load_kwargs["device_map"] = "auto"
                        load_kwargs["max_memory"] = {device_index: f"{free_in_GB-5}GB"}
                else:
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = {
                        i: f"{free_in_GB-5}GB" for i in range(torch.cuda.device_count())
                    }
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    **load_kwargs,
                )
            model.eval()
            update_model_run_metadata(model, load_int8=bool(getattr(model, "is_loaded_in_8bit", False)))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False, **auth_kwargs)

        # HF tokenizer bug fix
        if model_family == "opt":
            tokenizer.bos_token_id = 0

        # Decoder-only LMs may miss a pad token. Keep tokenizer/model pad ids aligned.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None and model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        if model_family == "llama":
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
            if model_family in {"opt", "llama", "mistral"}:
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        maybe_convert_model_to_torchao_float8_training(model, getattr(self.args, "use_torchao_float8", False))

        if self.args.trainer == "zo" and quzo_enabled(getattr(self.args, "zo_quantization_bits", 32)):
            quantize_model_in_place(
                model,
                int(self.args.zo_quantization_bits),
                include_frozen=True,
                seed=int(getattr(self.args, "seed", 0)),
            )
            logger.info(
                "[quzo-config] quantized model parameters in-place at %d bits before ZO training",
                int(self.args.zo_quantization_bits),
            )
        if self.args.trainer == "zo":
            logger.info(
                "[sparse-mezo-config] enabled=%s | sparse_ratio=%s | sparse_mask_strategy=%s | sparse_scope=%s | sparse_log_active_fraction=%s",
                bool(sparse_mezo_enabled(getattr(self.args, "sparse_ratio", 1.0))),
                float(getattr(self.args, "sparse_ratio", 1.0)),
                str(getattr(self.args, "sparse_mask_strategy", "percentile_per_layer")),
                str(getattr(self.args, "sparse_scope", "trainable_only")),
                bool(getattr(self.args, "sparse_log_active_fraction", True)),
            )
            logger.info(
                "[sparse-mezo-config] ratio semantics: sparse_ratio targets the active fraction per trainable tensor; ratio=1.0 disables masking. Order: construct direction -> apply sparse mask -> apply QuZO snapping after masked perturb/update when low-bit QuZO is active."
            )

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
        # 日志输出到当前 run 的 output_dir 下，方便 sweep 直接按目录收集。
        # 注意：HuggingFace 的 Trainer 在设置了 logging_strategy="steps" 且 logging_steps=1 时，
        # 会在每个优化步触发 on_log 回调（若使用梯度累积，则每累计完成一次为一个优化步）。
        import os
        import json
        import time
        import random

        # 生成当前运行的标签（包含任务名/模型名/样本数/eps 等），用于区分不同实验
        run_tag = result_file_tag(self.args)  # 例如：SST2-opt-125m-eps0.001-...
        logs_dir = self.args.output_dir or os.path.join("result", run_tag)
        os.makedirs(logs_dir, exist_ok=True)
        precision_label = detect_precision_label(self.args)

        class _HistoryWriter:
            """
            简单的历史日志写入器：
            - JSONL：`<output_dir>/metrics_<run_tag>.jsonl`
            - CSV：  `<output_dir>/metrics_<run_tag>.csv`
            每一行都会额外带上 task/model/eps/seed/precision 等字段，便于后期聚合分析。
            """
            def __init__(
                self,
                out_dir: str,
                run_tag: str,
                task_name: str,
                model_name: str,
                eps: float,
                seed: int,
                precision: str,
                zo_quantization_bits: int,
            ):
                self.dir = out_dir
                self.run_tag = run_tag
                self.task_name = task_name
                self.model_name = model_name
                self.eps = eps
                self.seed = int(seed)
                self.precision = precision
                self.zo_quantization_bits = int(zo_quantization_bits)
                self.jsonl_path = os.path.join(self.dir, f"metrics_{self.run_tag}.jsonl")
                self.csv_path = os.path.join(self.dir, f"metrics_{self.run_tag}.csv")
                # 初始化 CSV 表头（包含任务信息）
                if not os.path.exists(self.csv_path):
                    with open(self.csv_path, "w", encoding="utf-8") as f:
                        f.write("time,step,epoch,phase,split,metric,value,task,model,eps,seed,precision,zo_quantization_bits\n")

            def append_jsonl(self, obj: dict):
                # JSONL 中也冗余写入任务信息，方便独立解析
                obj = dict(obj)
                obj.update({
                    "task": self.task_name,
                    "model": self.model_name,
                    "eps": self.eps,
                    "seed": self.seed,
                    "precision": self.precision,
                    "zo_quantization_bits": self.zo_quantization_bits,
                })
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            def append_csv_row(self, time_s: str, step: int, epoch: float, phase: str, metric: str, value: float):
                with open(self.csv_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{time_s},{step},{epoch},{phase},{phase},{metric},{value},"
                        f"{self.task_name},{self.model_name},{self.eps},{self.seed},{self.precision},{self.zo_quantization_bits}\n"
                    )

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
                    seed=self.framework.args.seed,
                    precision=precision_label,
                    zo_quantization_bits=int(getattr(self.framework.args, "zo_quantization_bits", 32)),
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
                            "phase": "train", "split": "train", "metric": k, "value": float(v)
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
                                "phase": "eval", "split": "eval", "metric": mk, "value": float(mv)
                            })
                            self.writer.append_csv_row(ts, step, epoch_val, "eval", mk, float(mv))

                # 2) 使用自定义的 framework.evaluate 计算任务指标（如 accuracy / F1），也标记为 phase="eval"
                eval_metrics = self.framework.evaluate([], self.eval_samples)
                for mk, mv in eval_metrics.items():
                    self.writer.append_jsonl({
                        "time": ts, "step": step, "epoch": epoch_val,
                        "phase": "eval", "split": "eval", "metric": mk, "value": float(mv)
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
                            "phase": "train_probe", "split": "train_probe", "metric": mk, "value": float(mv)
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
        metrics_recorder = MetricsRecorder(self, train_samples, eval_samples, logs_dir, run_tag)
        trainer.add_callback(metrics_recorder)
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

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        self.last_train_artifacts = {
            "run_tag": run_tag,
            "output_dir": self.args.output_dir,
            "metrics_jsonl_path": metrics_recorder.writer.jsonl_path,
            "metrics_csv_path": metrics_recorder.writer.csv_path,
            "zo_directional_probe_csv": getattr(trainer, "_zo_probe_csv_path", None),
            "zo_directional_probe_last_row": getattr(trainer, "latest_zo_probe_row", None),
            "sparse_mezo_last_stats": getattr(trainer, "latest_sparse_mezo_stats", None),
            "tail_perf_metrics": getattr(trainer, "latest_perf_tail_metrics", None),
            "train_metrics": getattr(train_result, "metrics", None),
            "best_metric": getattr(trainer.state, "best_metric", None),
            "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            "trainer_state_path": os.path.join(self.args.output_dir, "trainer_state.json") if self.args.output_dir else None,
        }

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
    quzo_tag = f"-qbits{int(getattr(args, 'zo_quantization_bits', 32))}"
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + eps_tag + quzo_tag + customized_tag


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
    args = normalize_model_args(args)
    args = normalize_data_args(args)

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sample_seed = args.train_set_seed
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,
        seed=train_sample_seed,
        dataset_mode=args.dataset_mode,
        num_k=args.num_k,
    )
    # Initialize trainer and load model
    framework = Framework(args, task)
    metadata_output_dir = args.output_dir or os.path.join("result", result_file_tag(args))
    run_metadata = collect_run_metadata(
        zo_method=infer_large_run_zo_method(args),
        args=args,
        model=framework.model,
        output_dir=metadata_output_dir,
        model_name=args.model_name,
        task_name=args.task_name,
        repo_root=str(REPO_ROOT),
    )
    run_metadata_path = None
    if int(getattr(args, "local_rank", -1)) <= 0:
        run_metadata_path = write_run_metadata(run_metadata, metadata_output_dir)
    framework.run_metadata = run_metadata
    framework.run_metadata_path = run_metadata_path

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            eval_split_samples = get_eval_split_samples(task, num_eval=args.num_eval, seed=train_set_seed)
            primary_eval_split = "valid" if "valid" in eval_split_samples else list(eval_split_samples.keys())[0]
            eval_samples = eval_split_samples[primary_eval_split]

            if args.trainer != "none":
                if args.dataset_mode == "full":
                    if args.num_dev is not None and args.num_dev > 0:
                        train_samples, dev_samples = split_train_dev_samples(
                            train_samples,
                            num_dev=args.num_dev,
                            seed=args.data_seed + train_set_id,
                        )
                    elif args.num_dev == 0:
                        dev_samples = None
                    else:
                        train_samples, dev_samples = split_train_dev_samples(
                            train_samples,
                            dev_ratio=args.full_dev_ratio,
                            seed=args.data_seed + train_set_id,
                        )
                elif args.num_dev is not None and args.num_dev > 0:
                    train_samples, dev_samples = split_train_dev_samples(
                        train_samples,
                        num_dev=args.num_dev,
                        seed=args.data_seed + train_set_id,
                    )
                elif args.num_dev == -1:
                    train_samples, dev_samples = split_train_dev_samples(
                        train_samples,
                        dev_ratio=0.25,
                        seed=args.data_seed + train_set_id,
                    )
                else:
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
                    legacy_result_path = "result/" + result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file
                    write_metrics_to_file(metrics, legacy_result_path)

                    final_metrics_name = "final_metrics.json" if len(train_sets) == 1 else f"final_metrics_trainset{train_set_id}.json"
                    final_metrics_path = os.path.join(args.output_dir, final_metrics_name)
                    write_metrics_to_file(metrics, final_metrics_path)

                    train_artifacts = dict(getattr(framework, "last_train_artifacts", {}) or {})
                    summary_name = "run_summary.json" if len(train_sets) == 1 else f"run_summary_trainset{train_set_id}.json"
                    summary_path = os.path.join(args.output_dir, summary_name)
                    summary_payload = {
                        "task_name": args.task_name,
                        "dataset_mode": getattr(args, "dataset_mode", None),
                        "model_name": args.model_name,
                        "output_dir": args.output_dir,
                        "seed": int(args.seed),
                        "train_set_seed": int(train_set_seed),
                        "h": float(args.zo_eps),
                        "precision": detect_precision_label(args),
                        "zo_quantization_bits": int(getattr(args, "zo_quantization_bits", 32)),
                        "final_metrics": metrics,
                        "paths": {
                            "final_metrics_json": final_metrics_path,
                            "legacy_metrics_json": legacy_result_path,
                            "metrics_jsonl": train_artifacts.get("metrics_jsonl_path"),
                            "metrics_csv": train_artifacts.get("metrics_csv_path"),
                            "zo_directional_probe_csv": train_artifacts.get("zo_directional_probe_csv"),
                            "trainer_state_json": train_artifacts.get("trainer_state_path"),
                            "run_metadata_json": getattr(framework, "run_metadata_path", None),
                        },
                        "artifacts": {
                            "train_metrics": train_artifacts.get("train_metrics"),
                            "best_metric": train_artifacts.get("best_metric"),
                            "best_model_checkpoint": train_artifacts.get("best_model_checkpoint"),
                            "metrics_csv_last_row": _read_last_csv_row(train_artifacts.get("metrics_csv_path")),
                            "zo_directional_probe_last_row": _read_last_csv_row(train_artifacts.get("zo_directional_probe_csv")),
                            "sparse_mezo_last_stats": train_artifacts.get("sparse_mezo_last_stats"),
                            "tail_perf_metrics": train_artifacts.get("tail_perf_metrics"),
                            "trainer_state": _read_json_if_exists(train_artifacts.get("trainer_state_path")),
                        },
                        "run_metadata": getattr(framework, "run_metadata", None),
                        "config": vars(args),
                    }
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(_normalize_for_json(summary_payload), f, ensure_ascii=False, indent=2)

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
            legacy_result_path = "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file
            write_metrics_to_file(metrics, legacy_result_path)
            if args.output_dir:
                write_metrics_to_file(metrics, os.path.join(args.output_dir, "final_metrics.json"))

if __name__ == "__main__": 
    main()
