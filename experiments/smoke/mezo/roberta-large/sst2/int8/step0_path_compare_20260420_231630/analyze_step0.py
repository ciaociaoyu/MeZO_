#!/usr/bin/env python
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


REPO_ROOT = Path("/scratch/jy03364/MeZO_")
sys.path.insert(0, str(REPO_ROOT / "medium_models"))

import run as runmod  # noqa: E402
from src.quzo import _normal_like_with_seed, _seed_from_parts, quantize_tensor  # noqa: E402
from src.trainer import Trainer  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "experiments/smoke/mezo/roberta-large/sst2/int8/step0_path_compare_20260420_231630"
JSON_PATH = OUTPUT_DIR / "step0_path_compare.json"


def seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_sample(t: torch.Tensor, n: int = 8) -> List[float]:
    flat = t.detach().float().reshape(-1)
    return [float(x) for x in flat[: min(n, flat.numel())].cpu().tolist()]


def diff_stats(base: torch.Tensor, other: torch.Tensor) -> Dict[str, float]:
    diff = other.detach().float() - base.detach().float()
    diff_abs = diff.abs()
    return {
        "l2": float(torch.linalg.vector_norm(diff).item()),
        "linf": float(diff_abs.max().item()) if diff.numel() > 0 else 0.0,
        "mean_abs": float(diff_abs.mean().item()) if diff.numel() > 0 else 0.0,
        "nonzero_frac": float((diff_abs > 0).float().mean().item()) if diff.numel() > 0 else 0.0,
    }


def summarize_param(name: str, base: torch.Tensor, path1: torch.Tensor, path2_raw: torch.Tensor) -> Dict[str, object]:
    return {
        "name": name,
        "shape": list(base.shape),
        "base_sample": tensor_sample(base),
        "path1_sample": tensor_sample(path1),
        "path2_fp32_raw_sample": tensor_sample(path2_raw),
        "path1_vs_base": diff_stats(base, path1),
        "path2_fp32_raw_vs_base": diff_stats(base, path2_raw),
        "path1_vs_path2_fp32_raw": diff_stats(path1, path2_raw),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = runmod.HfArgumentParser(
        (runmod.ModelArguments, runmod.DynamicDataTrainingArguments, runmod.DynamicTrainingArguments)
    )
    args_list = [
        "--model_name_or_path",
        "roberta-large",
        "--few_shot_type",
        "prompt",
        "--task_name",
        "SST-2",
        "--template",
        "*cls**sent_0*_It_was*mask*.*sep+*",
        "--mapping",
        "{'0':'terrible','1':'great'}",
        "--data_dir",
        "data/k-shot-1k-test/SST-2/16-16",
        "--dataset_mode",
        "full",
        "--data_root",
        "data/k-shot-1k-test",
        "--full_dev_ratio",
        "0.1",
        "--overwrite_output_dir",
        "--output_dir",
        str(OUTPUT_DIR / "hf_output"),
        "--num_k",
        "16",
        "--seed",
        "16",
        "--data_seed",
        "16",
        "--do_eval",
        "--do_predict",
        "--do_train",
        "--trainer",
        "standard",
        "--optimizer",
        "sgd",
        "--max_steps",
        "10000",
        "--logging_steps",
        "10",
        "--per_device_eval_batch_size",
        "4",
        "--evaluate_during_training",
        "--use_adaptive_h",
        "False",
        "--use_c_scale",
        "False",
        "--per_device_train_batch_size",
        "64",
        "--learning_rate",
        "1e-6",
        "--eval_steps",
        "1000",
        "--weight_decay",
        "0",
        "--zero_order_eps",
        "1e-4",
        "--zero_order_optim",
        "--lr_scheduler_type",
        "constant",
        "--optimizer",
        "sgd",
        "--zero_order_use_trainer_optim",
        "False",
        "--efficient_zero_order",
        "True",
        "--zo_two_point_precision",
        "fp16",
        "--zo_quantization",
        "int8",
        "--quzo_quantize_perturbation_delta",
        "True",
        "--zo_probe_every",
        "200",
        "--zo_probe_num_seeds",
        "16",
        "--zo_probe_log_csv",
        "True",
        "--random_prediction_guard_enabled",
        "False",
        "--zo_probe_health_guard_enabled",
        "False",
    ]
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_list)

    data_args.task_name = runmod.normalize_medium_task_name(getattr(data_args, "task_name", ""))
    training_args.zo_two_point_precision = str(getattr(training_args, "zo_two_point_precision", "fp32")).lower()
    zo_quantization_alias = getattr(training_args, "zo_quantization", None)
    if zo_quantization_alias not in (None, ""):
        training_args.zo_quantization_bits = runmod.validate_quzo_bits(zo_quantization_alias)
    else:
        training_args.zo_quantization_bits = runmod.validate_quzo_bits(
            getattr(training_args, "zo_quantization_bits", 32)
        )
    training_args.zo_method = runmod.normalize_zo_method_name(getattr(training_args, "zo_method", None))
    training_args.sparse_ratio = runmod.validate_sparse_ratio(getattr(training_args, "sparse_ratio", 1.0))
    training_args.sparse_mask_strategy = runmod.normalize_sparse_mask_strategy(
        getattr(training_args, "sparse_mask_strategy", "percentile_per_layer")
    )
    training_args.sparse_scope = runmod.normalize_sparse_scope(getattr(training_args, "sparse_scope", "trainable_only"))
    training_args.sparse_mask_refresh_steps = int(getattr(training_args, "sparse_mask_refresh_steps", 100))
    if "prompt" in model_args.few_shot_type:
        data_args.prompt = True
    training_args.local_rank = -1

    runmod.set_seed(training_args.seed)
    if getattr(training_args, "data_seed", None) is None:
        training_args.data_seed = training_args.seed

    data_resolution = runmod.resolve_and_prepare_data(
        data_args=data_args,
        training_args=training_args,
        logger=logging.getLogger("step0_compare"),
    )
    data_args.dataset_mode = data_resolution.resolved_dataset_mode
    data_args.data_dir = data_resolution.resolved_data_dir

    num_labels = runmod.num_labels_mapping[data_args.task_name]
    config = runmod.AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    model_fn = runmod.MODEL_TYPES[config.model_type]
    tokenizer = runmod.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=[],
        cache_dir=model_args.cache_dir,
    )
    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    tokenizer.model_type = model.config.model_type

    train_dataset = runmod.FewShotDataset(
        data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type)
    )
    eval_dataset = runmod.FewShotDataset(
        data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type)
    )
    if eval_dataset.label_word_list is not None:
        model.label_word_list = torch.tensor(eval_dataset.label_word_list).long().to(training_args.device)
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    runmod.quantize_model_in_place(
        model,
        int(training_args.zo_quantization_bits),
        include_frozen=True,
        seed=int(training_args.seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda _: {},
        data_collator=runmod.MyDataCollatorWithPadding(tokenizer),
    )
    trainer._build_named_parameters_to_optim(model)
    if not hasattr(trainer.state, "zo_forward_step"):
        trainer.state.zo_forward_step = 0

    train_loader = trainer.get_train_dataloader()
    batch = next(iter(train_loader))
    batch = trainer._prepare_inputs(batch)

    base_params = {name: param.detach().clone() for name, param in trainer.named_parameters_to_optim}
    eps = float(trainer._get_training_step_size())
    random_seed = int(np.random.randint(1000000000))
    random_vector = trainer._zo_materialize_random_vector(random_seed)
    loss_seed = 314159
    bits = int(training_args.zo_quantization_bits)

    path1_current_resnap_params: Dict[str, torch.Tensor] = {}
    path1_user_single_quant_params: Dict[str, torch.Tensor] = {}
    path2_params_fp32_raw: Dict[str, torch.Tensor] = {}
    top_diff_rows = []

    for name, param in trainer.named_parameters_to_optim:
        bundle = random_vector[name]
        gaussian_seed = _seed_from_parts(random_seed, name, "gaussian")
        perturb_seed = _seed_from_parts(random_seed, name, "perturb")
        u_raw = _normal_like_with_seed(param.data, gaussian_seed, dtype=torch.float32)

        delta1 = trainer._quzo_build_perturbation_delta(
            param,
            bundle["u1"],
            eps=eps,
            scaling_factor=-1.0,
            bundle=bundle,
        )
        state_seed = int(bundle["state_seed"].item())
        path1_current = quantize_tensor(
            base_params[name].detach().float() + delta1.detach().float(),
            bits,
            seed=state_seed,
            target_dtype=param.data.dtype,
        )
        delta_user = quantize_tensor(
            (-eps) * u_raw.detach().float(),
            bits,
            seed=perturb_seed,
            target_dtype=param.data.dtype,
        )
        path1_user = base_params[name].detach().float() + delta_user.detach().float()
        path2 = base_params[name].detach().float() - eps * u_raw.detach().float()

        path1_current_resnap_params[name] = path1_current
        path1_user_single_quant_params[name] = path1_user.to(dtype=param.data.dtype)
        path2_params_fp32_raw[name] = path2.to(dtype=param.data.dtype)

        diff = diff_stats(path1_user, path2)
        row = summarize_param(name, base_params[name], path1_user, path2)
        row["current_resnap_sample"] = tensor_sample(path1_current)
        row["current_resnap_vs_base"] = diff_stats(base_params[name], path1_current)
        row["current_resnap_vs_user_single_quant"] = diff_stats(path1_current, path1_user)
        top_diff_rows.append((diff["l2"], row))

    top_diff_rows.sort(key=lambda item: item[0], reverse=True)
    top_param_summaries = [row for _, row in top_diff_rows[:8]]

    def compute_loss_with_params(param_map: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            for name, param in trainer.named_parameters_to_optim:
                param.data.copy_(param_map[name].to(dtype=param.data.dtype))
            seed_all(loss_seed)
            model.train()
            loss = trainer._zo_two_point_forward(model, batch)
            return float(loss.detach().float().item())

    try:
        base_loss = compute_loss_with_params(base_params)
        path1_current_loss = compute_loss_with_params(path1_current_resnap_params)
        path1_user_loss = compute_loss_with_params(path1_user_single_quant_params)
        path2_loss_fp32_raw = compute_loss_with_params(path2_params_fp32_raw)
    finally:
        with torch.no_grad():
            for name, param in trainer.named_parameters_to_optim:
                param.data.copy_(base_params[name].to(dtype=param.data.dtype))

    global_path1_current = {"numel": 0, "l2_sq": 0.0, "l1": 0.0, "linf": 0.0, "changed": 0}
    global_path1_user = {"numel": 0, "l2_sq": 0.0, "l1": 0.0, "linf": 0.0, "changed": 0}
    global_path2 = {"numel": 0, "l2_sq": 0.0, "l1": 0.0, "linf": 0.0, "changed": 0}
    global_current_vs_user = {"numel": 0, "l2_sq": 0.0, "l1": 0.0, "linf": 0.0, "changed": 0}
    global_between = {"numel": 0, "l2_sq": 0.0, "l1": 0.0, "linf": 0.0, "changed": 0}

    for name in base_params:
        d1_current = (path1_current_resnap_params[name].detach().float() - base_params[name].detach().float()).abs()
        d1_user = (path1_user_single_quant_params[name].detach().float() - base_params[name].detach().float()).abs()
        d2 = (path2_params_fp32_raw[name].detach().float() - base_params[name].detach().float()).abs()
        dcu = (path1_current_resnap_params[name].detach().float() - path1_user_single_quant_params[name].detach().float()).abs()
        db = (path1_user_single_quant_params[name].detach().float() - path2_params_fp32_raw[name].detach().float()).abs()
        for store, diff in (
            (global_path1_current, d1_current),
            (global_path1_user, d1_user),
            (global_path2, d2),
            (global_current_vs_user, dcu),
            (global_between, db),
        ):
            store["numel"] += diff.numel()
            store["l2_sq"] += float(torch.sum(diff * diff).item())
            store["l1"] += float(torch.sum(diff).item())
            store["linf"] = max(store["linf"], float(diff.max().item()) if diff.numel() > 0 else 0.0)
            store["changed"] += int(torch.sum(diff > 0).item())

    def finalize(store: Dict[str, float]) -> Dict[str, float]:
        numel = max(int(store["numel"]), 1)
        return {
            "numel": int(store["numel"]),
            "l2": math.sqrt(float(store["l2_sq"])),
            "mean_abs": float(store["l1"]) / float(numel),
            "linf": float(store["linf"]),
            "nonzero_frac": float(store["changed"]) / float(numel),
        }

    output = {
        "config": {
            "task": "SST-2",
            "model": "roberta-large",
            "seed": 16,
            "data_seed": 16,
            "batch_size": 64,
            "eps": eps,
            "bits": bits,
            "quzo_quantize_perturbation_delta": True,
        },
        "step0_context": {
            "random_seed": random_seed,
            "loss_seed": loss_seed,
            "batch_size": int(batch["input_ids"].shape[0]),
            "label_sample": [int(x) for x in batch["labels"].detach().cpu().tolist()[:8]],
        },
        "losses": {
            "base_int8_snapped": base_loss,
            "path1_current_int8_delta_then_resnap": path1_current_loss,
            "path1_user_single_quant_no_resnap": path1_user_loss,
            "path2_dequant_then_fp32_raw_u": path2_loss_fp32_raw,
            "path1_current_minus_base": path1_current_loss - base_loss,
            "path1_user_minus_base": path1_user_loss - base_loss,
            "path2_minus_base": path2_loss_fp32_raw - base_loss,
            "path1_current_minus_path1_user": path1_current_loss - path1_user_loss,
            "path1_user_minus_path2": path1_user_loss - path2_loss_fp32_raw,
        },
        "global_param_diffs": {
            "path1_current_vs_base": finalize(global_path1_current),
            "path1_user_vs_base": finalize(global_path1_user),
            "path2_vs_base": finalize(global_path2),
            "path1_current_vs_path1_user": finalize(global_current_vs_user),
            "path1_user_vs_path2": finalize(global_between),
        },
        "top_param_summaries": top_param_summaries,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["losses"], indent=2))
    print(f"[saved] {JSON_PATH}")


if __name__ == "__main__":
    main()
